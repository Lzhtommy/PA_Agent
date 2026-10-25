"""Performance and AI-quality statistics over closed trades (pure functions)."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

_CONF_BUCKETS = ((0, 40, "<40"), (40, 50, "40-49"), (50, 60, "50-59"),
                 (60, 70, "60-69"), (70, 101, "≥70"))


def _r_multiple(t: dict[str, Any]) -> float | None:
    risk = t.get("risk_amount")
    pnl = t.get("net_pnl")
    if not risk or pnl is None:
        return None
    return float(pnl) / float(risk)


def max_drawdown(values: list[float]) -> tuple[float, float]:
    """Return ``(max_drawdown_abs, max_drawdown_pct)`` of a value series."""
    peak = None
    mdd = mdd_pct = 0.0
    for v in values:
        peak = v if peak is None else max(peak, v)
        dd = peak - v
        if dd > mdd:
            mdd = dd
            mdd_pct = dd / peak * 100 if peak else 0.0
    return mdd, mdd_pct


def summarize(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Stats over closed trades. Drawdown is on realised PnL (not account equity,
    whose floating PnL spikes would overstate it), so it also works per strategy
    version; ``return_pct`` is net PnL over the equity before the first trade."""
    closed = [t for t in trades if t.get("state") == "closed" and t.get("net_pnl") is not None]
    closed.sort(key=lambda t: t.get("closed_ms") or 0)
    pnls = [float(t["net_pnl"]) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    rs = [r for r in (_r_multiple(t) for t in closed) if r is not None]

    start_eq = next((float(t["equity_at_entry"]) for t in closed if t.get("equity_at_entry")), None)
    cumulative, running = [], 0.0
    for p in pnls:
        running += p
        cumulative.append(running)
    base = start_eq or 0.0
    mdd, mdd_pct = max_drawdown([base, *(base + c for c in cumulative)])
    if not start_eq:
        mdd_pct = None
    cum_r, running_r = [], 0.0
    for r in rs:
        running_r += r
        cum_r.append(running_r)
    mdd_r, _ = max_drawdown([0.0, *cum_r])
    return {
        "trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(closed) * 100 if closed else None,
        "net_pnl": sum(pnls),
        "fees": sum(float(t.get("fee") or 0) for t in closed),
        "funding": sum(float(t.get("funding_fee") or 0) for t in closed),
        "avg_win": gross_win / len(wins) if wins else None,
        "avg_loss": -gross_loss / len(losses) if losses else None,
        "payoff_ratio": (gross_win / len(wins)) / (gross_loss / len(losses))
        if wins and losses and gross_loss > 0 else None,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else None,
        "expectancy": sum(pnls) / len(pnls) if pnls else None,
        "avg_r": sum(rs) / len(rs) if rs else None,
        "total_r": sum(rs) if rs else None,
        "max_drawdown": mdd,
        "max_drawdown_pct": mdd_pct,
        "max_drawdown_r": mdd_r,
        "start_equity": start_eq,
        "return_pct": sum(pnls) / start_eq * 100 if start_eq and closed else None,
        "best": max(pnls) if pnls else None,
        "worst": min(pnls) if pnls else None,
    }


def _group_stats(rows: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for key, t in rows:
        groups.setdefault(key, []).append(t)
    out = []
    for key, ts in groups.items():
        pnls = [float(t["net_pnl"]) for t in ts]
        rs = [r for r in (_r_multiple(t) for t in ts) if r is not None]
        out.append({
            "group": key,
            "trades": len(ts),
            "win_rate": sum(1 for p in pnls if p > 0) / len(ts) * 100,
            "net_pnl": sum(pnls),
            "avg_r": sum(rs) / len(rs) if rs else None,
            "total_r": sum(rs) if rs else None,
        })
    return out


def _bucket(value: Any) -> str:
    if value is None:
        return "未知"
    v = int(value)
    for lo, hi, label in _CONF_BUCKETS:
        if lo <= v < hi:
            return label
    return "未知"


def ai_quality(trades: list[dict[str, Any]], signals_by_id: dict[int, dict[str, Any]],
               signal_status: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Break results down by what the AI claimed at signal time."""
    closed = [t for t in trades if t.get("state") == "closed" and t.get("net_pnl") is not None]
    order = [label for _, _, label in _CONF_BUCKETS] + ["未知"]

    def by(key_fn) -> list[dict[str, Any]]:
        rows = []
        for t in closed:
            sig = signals_by_id.get(t.get("signal_id") or -1, {})
            rows.append((key_fn(t, sig), t))
        return _group_stats(rows)

    by_conf = sorted(by(lambda t, s: _bucket(s.get("trade_confidence"))),
                     key=lambda g: order.index(g["group"]) if g["group"] in order else 99)
    by_est = sorted(by(lambda t, s: _bucket(s.get("estimated_win_rate"))),
                    key=lambda g: order.index(g["group"]) if g["group"] in order else 99)
    entry_labels = {"limit": "限价", "stop": "突破", "market": "市价"}
    by_entry = by(lambda t, s: entry_labels.get(t.get("entry_type"), t.get("entry_type") or "?"))
    by_dir = by(lambda t, s: "做多" if t.get("direction") == "long" else "做空")

    def cycle(t, s):
        try:
            return json.loads(s.get("stage1_json") or "{}").get("cycle_position") or "未知"
        except (TypeError, ValueError):
            return "未知"

    by_cycle = by(cycle)
    by_exit = _group_stats([(t.get("close_reason") or "未知", t) for t in closed])
    by_version = by(lambda t, s: version_label(s.get("model"), s.get("prompt_hash"), s.get("timeframe")))
    orders = [s for s in (signal_status or []) if s.get("order_type") not in (None, "none")]
    funnel = Counter((s.get("status") or "?") for s in orders)
    return {
        "by_version": by_version,
        "by_confidence": by_conf,
        "by_estimated_win_rate": by_est,
        "by_entry_type": by_entry,
        "by_direction": by_dir,
        "by_cycle": by_cycle,
        "by_exit_reason": by_exit,
        "signal_funnel": dict(funnel),
        "signals_total": len(signal_status or []),
        "signals_actionable": len(orders),
    }


def version_label(model: str | None, prompt_hash: str | None, timeframe: str | None = None) -> str:
    """Human label for a strategy version (model + prompt hash + timeframe)."""
    if not model:
        label = "未标记"
    elif prompt_hash in (None, "", "legacy"):
        label = f"{model} · 旧版本"
    else:
        label = f"{model} · {prompt_hash}"
    return f"{label} · {timeframe}" if timeframe else label
