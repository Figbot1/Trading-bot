#!/usr/bin/env python3
"""
FIGBOT — Railway entrypoint
Incorporates two guiding principles:
1) Shared load: many agents, clear status, graceful degradation.
2) Gentle pacing: explain what's happening in human terms (no mystery failures).

This file:
- Serves dashboard.html at /
- Serves artwork at /assets/*
- Runs democratic_trader_hybrid loop in a background thread
- Adds a "coach" endpoint that turns system state into supportive, actionable guidance
- Adds an optional runtime Alpaca key setter (in-memory) for quick onboarding
"""

import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
import sqlite3

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import democratic_trader_hybrid as hybrid  # type: ignore
import dashboard_api as dash  # type: ignore

app = Flask(__name__)
CORS(app)

# --- Hard fix: ensure DB path is consistent on Railway (CWD can differ under gunicorn) ---
DB_PATH = str(ROOT / "trading_history.db")
hybrid.DB_PATH = DB_PATH
dash.DB_PATH = DB_PATH

# --- Runtime keys (not persisted). Useful for onboarding. ---
_runtime_keys: Dict[str, str] = {}

_worker = None
_stop_evt = threading.Event()
_state: Dict[str, Any] = {
    "running": False,
    "cycles": 0,
    "last_cycle_ts": None,
    "last_timeframe": None,
    "last_error": None,
    "last_trace": None,
    "cycle_seconds": getattr(hybrid, "CYCLE_SECONDS", 60),
    "cycle_timeframes": list(getattr(hybrid, "CYCLE_TIMEFRAMES", ["1m"])),
    "invalid_timeframes": [],
}

_allowed_timeframes = set(getattr(hybrid, "TIMEFRAME_SECONDS", {}).keys())


def _db_rows(query: str, args=(), limit: int | None = None) -> List[sqlite3.Row]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, args)
    rows = cur.fetchall()
    conn.close()
    if limit is not None:
        return rows[:limit]
    return rows


def _recent_activity_snapshot() -> Dict[str, Any]:
    # Lightweight snapshot for UI + coaching.
    snap: Dict[str, Any] = {"calls_10m": 0, "errors_10m": 0, "trades_24h": 0, "preds_10m": 0, "top_asset": None}

    try:
        # calls last 10 minutes
        rows = _db_rows(
            "SELECT ok FROM model_calls WHERE timestamp >= datetime('now','-10 minutes')",
        )
        snap["calls_10m"] = len(rows)
        snap["errors_10m"] = sum(1 for r in rows if int(r["ok"]) == 0)

        # preds last 10 minutes
        prow = _db_rows(
            "SELECT asset FROM predictions WHERE timestamp >= datetime('now','-10 minutes')"
        )
        snap["preds_10m"] = len(prow)

        # trades last 24h
        trow = _db_rows(
            "SELECT asset FROM trades WHERE timestamp >= datetime('now','-24 hours')"
        )
        snap["trades_24h"] = len(trow)

        # top asset in recent predictions
        if prow:
            counts: Dict[str, int] = {}
            for r in prow:
                a = r["asset"]
                counts[a] = counts.get(a, 0) + 1
            snap["top_asset"] = max(counts.items(), key=lambda kv: kv[1])[0]
    except Exception:
        # best-effort; don't fail the app if DB is empty on first boot
        pass

    return snap


def _alpaca_keys_present() -> bool:
    # Accept either canonical Alpaca envs or our runtime keys
    k = os.getenv("APCA_API_KEY_ID") or _runtime_keys.get("APCA_API_KEY_ID")
    s = os.getenv("APCA_API_SECRET_KEY") or _runtime_keys.get("APCA_API_SECRET_KEY")
    return bool(k and s)


def _loop():
    i = 0
    _state["running"] = True
    while not _stop_evt.is_set():
        tfs = list(getattr(hybrid, "CYCLE_TIMEFRAMES", ["1m"])) or ["1m"]
        tf = tfs[i % len(tfs)]
        if tf not in getattr(hybrid, "TIMEFRAME_SECONDS", {}):
            tf = "1m"

        _state["last_timeframe"] = tf
        _state["last_cycle_ts"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        try:
            # propagate runtime keys into env so hybrid.AlpacaClient can see them
            for k, v in _runtime_keys.items():
                os.environ[k] = v

            hybrid.run_cycle(tf)

            _state["last_error"] = None
            _state["last_trace"] = None
        except Exception as exc:
            _state["last_error"] = str(exc)
            _state["last_trace"] = traceback.format_exc(limit=30)

        _state["cycles"] += 1
        i += 1
        _state["cycle_seconds"] = getattr(hybrid, "CYCLE_SECONDS", _state["cycle_seconds"])
        _state["cycle_timeframes"] = list(getattr(hybrid, "CYCLE_TIMEFRAMES", _state["cycle_timeframes"]))

        # Interruptible sleep => Stop works immediately
        _stop_evt.wait(_state["cycle_seconds"])

    _state["running"] = False


def _start_worker() -> bool:
    global _worker
    if _worker and _worker.is_alive():
        return False
    _stop_evt.clear()
    _worker = threading.Thread(target=_loop, daemon=True)
    _worker.start()
    return True


def _stop_worker() -> bool:
    _stop_evt.set()
    return True


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "running": _state["running"], "cycles": _state["cycles"]}


@app.get("/api/version")
def version():
    return {
        "build": os.getenv("RAILWAY_GIT_COMMIT_SHA", os.getenv("APP_BUILD_ID", "unknown")),
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "db_path": DB_PATH,
    }


@app.get("/")
def index():
    return send_file(str(ROOT / "dashboard.html"))


@app.get("/assets/<path:filename>")
def assets(filename: str):
    return send_from_directory(str(ROOT / "assets"), filename, max_age=3600)


# ---- Dashboard read APIs (DB-backed) ----
@app.get("/api/stats")
def stats():
    return dash.stats()

@app.get("/api/experts/leaderboard")
def experts():
    return dash.experts()

@app.get("/api/predictions/recent")
def preds():
    return dash.preds()

@app.get("/api/calls/recent")
def calls():
    return dash.calls()

@app.get("/api/trades/recent")
def trades():
    return dash.trades()

@app.get("/api/system/metrics")
def system_metrics():
    return dash.system_metrics()


# ---- Control endpoints ----
@app.post("/start")
def start():
    started = _start_worker()
    return jsonify({"started": started, **_state})

@app.post("/stop")
def stop():
    _stop_worker()
    return jsonify({"stopped": True, **_state})

@app.post("/config")
def config():
    data = request.json or {}
    if "cycle_seconds" in data:
        try:
            hybrid.CYCLE_SECONDS = max(1, int(data["cycle_seconds"]))
        except Exception:
            pass

    if "timeframes" in data:
        tfs = [t.strip() for t in str(data["timeframes"]).split(",") if t.strip()]
        invalid = [t for t in tfs if t not in _allowed_timeframes]
        valid = [t for t in tfs if t in _allowed_timeframes]
        if valid:
            hybrid.CYCLE_TIMEFRAMES = valid
        _state["invalid_timeframes"] = invalid

    _state["cycle_seconds"] = getattr(hybrid, "CYCLE_SECONDS", _state["cycle_seconds"])
    _state["cycle_timeframes"] = list(getattr(hybrid, "CYCLE_TIMEFRAMES", _state["cycle_timeframes"]))
    return jsonify({"ok": True, **_state})


@app.get("/status")
def status():
    snap = dict(_state)
    snap["alpaca_keys_present"] = _alpaca_keys_present()
    snap["activity"] = _recent_activity_snapshot()
    return jsonify(snap)


# ---- Video-inspired coaching layer (shared load + calm clarity) ----
@app.get("/api/coach")
def coach():
    st = dict(_state)
    activity = _recent_activity_snapshot()
    keys_ok = _alpaca_keys_present()

    messages: List[Dict[str, str]] = []

    # Beat 1: orient
    if st["running"]:
        messages.append({"role": "guide", "title": "We are running.", "text": "You do not have to hold this alone. I will monitor cycles, calls, and trades, and surface what matters."})
    else:
        messages.append({"role": "guide", "title": "We are paused.", "text": "If you want FIGBOT to act, press Start. If you want to observe without action, leave it stopped and review predictions."})

    # Beat 2: key situation
    if not keys_ok:
        messages.append({"role": "guide", "title": "Trading keys not set.", "text": "Cycles can run and write predictions, but live orders will not place until Alpaca keys are provided. You can add them in the Keys panel."})
    else:
        messages.append({"role": "guide", "title": "Keys detected.", "text": "Order placement is enabled. Risk still matters — review recent calls and predictions when volatility rises."})

    # Beat 3: errors / load sharing
    if st.get("last_error"):
        messages.append({"role": "warning", "title": "A cycle failed.", "text": "I caught an error during the last cycle. Scroll to Diagnostics to see the trace. We can fix it step by step."})
    elif activity.get("errors_10m", 0) > 0:
        messages.append({"role": "warning", "title": "Some model calls failed.", "text": "A few calls failed recently. This is normal under rate limits. FIGBOT will fall back and continue; check provider availability if it persists."})
    else:
        messages.append({"role": "guide", "title": "System looks stable.", "text": "Calls are flowing. This is a good time to watch for consensus shifts across agents."})

    # Beat 4: active focus
    if activity.get("top_asset"):
        messages.append({"role": "guide", "title": "Current focus.", "text": f"Most recent predictions are clustering on {activity['top_asset']}. If you feel overwhelmed, focus on one asset at a time."})

    return jsonify({
        "messages": messages[:6],
        "activity": activity,
        "running": bool(st["running"]),
        "last_timeframe": st.get("last_timeframe"),
        "cycles": st.get("cycles", 0),
    })


# ---- Runtime keys (in-memory; good for quick onboarding) ----
@app.post("/api/keys")
def set_keys():
    data = request.json or {}
    key = (data.get("APCA_API_KEY_ID") or "").strip()
    secret = (data.get("APCA_API_SECRET_KEY") or "").strip()
    base = (data.get("APCA_API_BASE_URL") or "").strip()

    if key and secret:
        _runtime_keys["APCA_API_KEY_ID"] = key
        _runtime_keys["APCA_API_SECRET_KEY"] = secret
        if base:
            _runtime_keys["APCA_API_BASE_URL"] = base
        # set env so immediate use works
        os.environ["APCA_API_KEY_ID"] = key
        os.environ["APCA_API_SECRET_KEY"] = secret
        if base:
            os.environ["APCA_API_BASE_URL"] = base
        return jsonify({"ok": True, "stored": True, "note": "Keys stored in memory for this running instance."})

    return jsonify({"ok": False, "stored": False, "error": "Both key and secret are required."}), 400


# Auto-start
if os.getenv("AUTO_START", "1") == "1":
    try:
        hybrid.init_database()
    except Exception:
        pass
    _start_worker()
