"""
Dashboard API (importable)
DB-backed read endpoints used by hybrid_app.
"""

from flask import jsonify
import sqlite3

DB_PATH = "trading_history.db"

def _q(query, args=()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, args)
    rows = cur.fetchall()
    conn.close()
    return rows

def _safe_count(table: str, where: str = "") -> int:
    try:
        w = f" WHERE {where}" if where else ""
        return int(_q(f"SELECT COUNT(*) AS n FROM {table}{w}")[0]["n"])
    except Exception:
        return 0

def _safe_avg(sql: str, default=0.0) -> float:
    try:
        v = _q(sql)[0][0]
        return float(v) if v is not None else float(default)
    except Exception:
        return float(default)

def stats():
    total_preds = _safe_count("predictions")
    eval_preds = _safe_count("predictions", "evaluated=1")
    avg_score = _safe_avg("SELECT AVG(correctness_score) FROM predictions WHERE evaluated=1", 0.0)

    total_calls = _safe_count("model_calls")
    ok_calls = _safe_count("model_calls", "ok=1")
    total_trades = _safe_count("trades")

    win_rate = None
    if eval_preds:
        try:
            wins = int(_q("SELECT COUNT(*) AS n FROM predictions WHERE evaluated=1 AND correctness_score >= 0.5")[0]["n"])
            win_rate = wins / eval_preds
        except Exception:
            win_rate = None

    return jsonify({
        "total_predictions": total_preds,
        "evaluated_predictions": eval_preds,
        "avg_correctness_score": round(float(avg_score), 4),
        "model_calls": total_calls,
        "model_calls_ok": ok_calls,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "portfolio_value": None
    })

def experts():
    try:
        rows = _q("""SELECT provider, model, thinking_style, timeframe, weight, ema_score, samples, last_updated
                    FROM expert_weights
                    ORDER BY (ema_score * weight) DESC
                    LIMIT 60""")
    except Exception:
        return jsonify([])
    out=[]
    for r in rows:
        out.append({
            "model": f"{r['provider']}:{r['model']}",
            "style": r["thinking_style"],
            "timeframe": r["timeframe"],
            "score": round(float(r["ema_score"]), 4),
            "weight": round(float(r["weight"]), 6),
            "samples": int(r["samples"]),
            "last_updated": r["last_updated"],
        })
    return jsonify(out)

def preds():
    try:
        rows = _q("""SELECT timestamp, asset, timeframe, direction, confidence, expected_change_percent
                    FROM predictions
                    ORDER BY timestamp DESC
                    LIMIT 120""")
    except Exception:
        return jsonify([])
    return jsonify([dict(r) for r in rows])

def calls():
    try:
        rows = _q("""SELECT timestamp, provider, model, thinking_style, ok, latency_ms, error
                    FROM model_calls
                    ORDER BY timestamp DESC
                    LIMIT 120""")
    except Exception:
        return jsonify([])
    out=[]
    for r in rows:
        out.append({
            "timestamp": r["timestamp"],
            "model": f"{r['provider']}:{r['model']}",
            "status": "ok" if int(r["ok"])==1 else "error",
            "latency_ms": r["latency_ms"],
            "error": r["error"],
            "style": r["thinking_style"],
        })
    return jsonify(out)

def trades():
    try:
        rows = _q("""SELECT timestamp, asset, action, price, quantity
                    FROM trades
                    ORDER BY timestamp DESC
                    LIMIT 120""")
    except Exception:
        return jsonify([])
    return jsonify([dict(r) for r in rows])

def system_metrics():
    try:
        rows = _q("""SELECT timestamp, cpu_pct AS cpu_percent, memory_mb, cycle_ms
                    FROM system_metrics
                    ORDER BY timestamp DESC
                    LIMIT 120""")
    except Exception:
        return jsonify([])
    return jsonify([dict(r) for r in rows])
