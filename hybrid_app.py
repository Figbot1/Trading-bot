#!/usr/bin/env python3
import os
import sys
import threading
import traceback
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import democratic_trader_hybrid as hybrid  # type: ignore
import dashboard_api as dash  # type: ignore

app = Flask(__name__)
CORS(app)

dash.DB_PATH = getattr(hybrid, "DB_PATH", dash.DB_PATH)

_worker = None
_stop_evt = threading.Event()

_state = {
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


def _loop():
    i = 0
    _state["running"] = True
    while not _stop_evt.is_set():
        tfs = list(getattr(hybrid, "CYCLE_TIMEFRAMES", ["1m"])) or ["1m"]
        tf = tfs[i % len(tfs)]
        if tf not in getattr(hybrid, "TIMEFRAME_SECONDS", {}):
            tf = "1m"

        _state["last_timeframe"] = tf
        _state["last_cycle_ts"] = datetime.utcnow().isoformat(timespec="seconds")

        try:
            hybrid.run_cycle(tf)
            _state["last_error"] = None
            _state["last_trace"] = None
        except Exception as exc:
            _state["last_error"] = str(exc)
            _state["last_trace"] = traceback.format_exc(limit=20)

        _state["cycles"] += 1
        i += 1
        _state["cycle_seconds"] = getattr(hybrid, "CYCLE_SECONDS", _state["cycle_seconds"])
        _state["cycle_timeframes"] = list(getattr(hybrid, "CYCLE_TIMEFRAMES", _state["cycle_timeframes"]))

        # Interruptible sleep => stop is immediate
        _stop_evt.wait(_state["cycle_seconds"])

    _state["running"] = False


def _start_worker():
    global _worker
    if _worker and _worker.is_alive():
        return False
    _stop_evt.clear()
    _worker = threading.Thread(target=_loop, daemon=True)
    _worker.start()
    return True


def _stop_worker():
    _stop_evt.set()
    return True


@app.get("/health")
def health():
    return {"ok": True, "running": _state["running"]}


@app.get("/")
def index():
    # Serve dashboard.html from repo root
    p = ROOT / "dashboard.html"
    if p.exists():
        return send_file(str(p))
    return jsonify({"ok": False, "error": "dashboard.html not found"}), 404


@app.get("/assets/<path:filename>")
def assets(filename: str):
    assets_dir = ROOT / "assets"
    return send_from_directory(str(assets_dir), filename, max_age=3600)


# Read-only APIs
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


# Controls
@app.post("/start")
def start():
    started = _start_worker()
    return jsonify({"started": started, **_state})


@app.post("/stop")
def stop():
    _stop_worker()
    return jsonify({"stopped": True, **_state})


@app.get("/status")
def status():
    providers = {
        "ollama": {
            "enabled": bool(getattr(hybrid, "OLLAMA_ENABLED", False)),
            "models": list(getattr(hybrid, "OLLAMA_MODELS", [])),
            "url": getattr(hybrid, "OLLAMA_URL", ""),
        },
        "remote": {
            "enabled": bool(getattr(hybrid, "ENABLE_REMOTE", False) and getattr(hybrid, "REMOTE_MODEL_URL", "") and getattr(hybrid, "REMOTE_MODEL_MODELS", [])),
            "models": list(getattr(hybrid, "REMOTE_MODEL_MODELS", [])),
            "url": getattr(hybrid, "REMOTE_MODEL_URL", ""),
        },
    }
    alpaca_ok = hybrid.AlpacaClient().ok() if hasattr(hybrid, "AlpacaClient") else False
    snapshot = dict(_state)
    snapshot.update({
        "alpaca_ok": alpaca_ok,
        "providers": providers,
        "fallback_btc_trade": getattr(hybrid, "FALLBACK_BTC_TRADE", False),
        "fallback_btc_asset": getattr(hybrid, "FALLBACK_BTC_ASSET", ""),
        "fallback_btc_order_symbol": getattr(hybrid, "FALLBACK_BTC_ORDER_SYMBOL", ""),
        "fallback_btc_notional": getattr(hybrid, "FALLBACK_BTC_NOTIONAL", 0),
    })
    return jsonify(snapshot)


def _last_trade_snapshot() -> dict:
    db_path = getattr(hybrid, "DB_PATH", "trading_history.db")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id, timestamp, asset, action, price, quantity FROM trades ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
    except Exception:
        return {"last_trade": None}

    if not row:
        return {"last_trade": None}

    ts = row["timestamp"]
    age_sec = None
    try:
        age_sec = int((datetime.utcnow() - datetime.fromisoformat(ts)).total_seconds())
    except Exception:
        age_sec = None
    return {
        "last_trade": {
            "id": row["id"],
            "timestamp": ts,
            "asset": row["asset"],
            "action": row["action"],
            "price": row["price"],
            "quantity": row["quantity"],
            "age_sec": age_sec,
        }
    }


@app.get("/api/state")
def state():
    payload = _last_trade_snapshot()
    tf = _state.get("last_timeframe") or "1m"
    try:
        conn = sqlite3.connect(getattr(hybrid, "DB_PATH", "trading_history.db"))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""SELECT timestamp, cycle_ms, assets_count, news_count, picks_stored,
                              evaluated_count, traded, fail_counts, memory_mb, cpu_pct
                       FROM system_metrics
                       ORDER BY id DESC LIMIT 1""")
        sm = cur.fetchone()
        cur.execute("""SELECT asset, direction, confidence, expected_change_percent
                       FROM predictions
                       WHERE timeframe=?
                       ORDER BY timestamp DESC
                       LIMIT 250""", (tf,))
        preds = cur.fetchall()
        conn.close()
    except Exception:
        sm = None
        preds = []

    up_w = 0.0
    down_w = 0.0
    confs = []
    signed_exps = []
    asset_w = {}
    for r in preds:
        conf = float(r["confidence"] or 0)
        exp = float(r["expected_change_percent"] or 0)
        direction = (r["direction"] or "").lower()
        w = max(0.0, conf) * max(0.0, exp)
        if direction == "up":
            up_w += w
            signed_exps.append(exp)
        elif direction == "down":
            down_w += w
            signed_exps.append(-exp)
        confs.append(conf)
        a = (r["asset"] or "").upper()
        asset_w[a] = asset_w.get(a, 0.0) + (w if direction == "up" else -w)

    total_w = up_w + down_w
    if total_w <= 0:
        strength = 0.0
        disagreement = 0.5
        consensus_dir = "neutral"
    else:
        strength = (up_w - down_w) / max(total_w, 1e-9)
        max_share = max(up_w, down_w) / max(total_w, 1e-9)
        disagreement = 1.0 - max_share
        if strength > 0.08:
            consensus_dir = "up"
        elif strength < -0.08:
            consensus_dir = "down"
        else:
            consensus_dir = "neutral"

    top_asset = None
    if asset_w:
        top_asset = sorted(asset_w.items(), key=lambda x: abs(x[1]), reverse=True)[0][0]

    payload.update({
        "consensus": {
            "direction": consensus_dir,
            "strength": round(float(strength), 4),
            "disagreement": round(float(disagreement), 4),
            "avg_conf": round(float(sum(confs) / max(len(confs), 1)), 4),
            "avg_expected": round(float(sum(signed_exps) / max(len(signed_exps), 1)), 4),
            "top_asset": top_asset,
            "timeframe": tf,
        },
        "system": dict(sm) if sm else {},
    })
    return jsonify(payload)


@app.get("/api/version")
def version():
    return jsonify({
        "build": os.getenv("APP_BUILD_ID", "local"),
        "timestamp": os.getenv("APP_BUILD_TS", datetime.utcnow().isoformat(timespec="seconds") + "Z"),
    })


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


# Auto-start (optional)
if os.getenv("AUTO_START", "1") == "1":
    try:
        hybrid.init_database()
    except Exception:
        pass
    _start_worker()
