"""Convert stage-2 decisions into execution signals and enqueue them in SQLite.

Every completed analysis is emitted (including 不下单) so the executor can
cancel a resting entry when the latest analysis no longer supports it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ORDER_TYPE_MAP: dict[str, str] = {
    "限价单": "limit",
    "突破单": "stop",
    "市价单": "market",
}


def _price(raw: Any) -> float | None:
    from pa_agent.records.trade_logger import _parse_sr_price

    return _parse_sr_price(raw)


def _int_or_none(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(float(str(raw).strip().rstrip("%")))
    except (TypeError, ValueError):
        return None


def direction_from_decision(decision: dict[str, Any]) -> str:
    from pa_agent.ai.decision_continuity import order_direction_sign

    sign = order_direction_sign(str(decision.get("order_direction") or ""))
    return "long" if sign > 0 else "short" if sign < 0 else ""


def build_signal(
    *,
    decision_inner: dict[str, Any],
    stage2_full: dict[str, Any] | None,
    stage1_diagnosis: dict[str, Any] | None,
    symbol: str,
    timeframe: str,
    source: str,
    analysis_key: str,
    record_path: str = "",
    chart_image: str = "",
) -> dict[str, Any]:
    """Pure mapping from a stage-2 decision dict to a ``signals`` row."""
    dec = decision_inner or {}
    order_type = ORDER_TYPE_MAP.get(str(dec.get("order_type") or "").strip(), "none")
    direction = direction_from_decision(dec) if order_type != "none" else ""
    if order_type != "none" and not direction:
        order_type = "none"
    full = dict(stage2_full or {})
    return {
        "signal_uid": f"{source}|{symbol}|{timeframe}|{analysis_key}",
        "source": source,
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "order_type": order_type,
        "entry": _price(dec.get("entry_price")),
        "stop_loss": _price(dec.get("stop_loss_price")),
        "tp1": _price(dec.get("take_profit_price")),
        "tp2": _price(dec.get("take_profit_price_2")),
        "trade_confidence": _int_or_none(dec.get("trade_confidence")),
        "estimated_win_rate": _int_or_none(dec.get("estimated_win_rate")),
        "reasoning": str(dec.get("reasoning") or ""),
        "decision_json": {"decision": dec, **{k: v for k, v in full.items() if k != "decision"}},
        "stage1_json": stage1_diagnosis or {},
        "record_path": record_path,
        "chart_image": chart_image,
    }


def emit_signal(
    settings: Any,
    *,
    decision_inner: dict[str, Any],
    stage2_full: dict[str, Any] | None,
    stage1_diagnosis: dict[str, Any] | None,
    symbol: str,
    timeframe: str,
    source: str,
    analysis_key: str,
    record_path: str = "",
    chart_image: str = "",
    db_path: Path | None = None,
) -> int | None:
    """Enqueue a signal when ``settings.trading.enabled``; returns signal id (or None)."""
    trading = getattr(settings, "trading", None)
    if trading is None or not getattr(trading, "enabled", False):
        return None
    from pa_agent.config.paths import QUANT_DB_PATH
    from pa_agent.trading.store import TradeStore

    signal = build_signal(
        decision_inner=decision_inner,
        stage2_full=stage2_full,
        stage1_diagnosis=stage1_diagnosis,
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        analysis_key=analysis_key,
        record_path=record_path,
        chart_image=chart_image,
    )
    store = TradeStore(db_path or QUANT_DB_PATH)
    try:
        signal_id = store.insert_signal(signal)
    finally:
        store.close()
    if signal_id is not None:
        logger.info(
            "Trading signal #%s queued: %s %s %s %s entry=%s sl=%s",
            signal_id, symbol, timeframe, signal["direction"] or "-",
            signal["order_type"], signal["entry"], signal["stop_loss"],
        )
    return signal_id
