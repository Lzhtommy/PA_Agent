"""Convert stage-2 decisions into execution signals and enqueue them in SQLite.

Every completed analysis is emitted (including 不下单) so the executor can
cancel a resting entry when the latest analysis no longer supports it.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import subprocess
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


def prompt_hash() -> str:
    """Short hash of everything that shapes the AI decision: the prompt files plus
    the ``pa_agent/ai`` and trade-metrics code that assemble, validate and adjust it.

    Signals sharing ``(model, prompt_hash)`` come from the same strategy version, so
    their results can be pooled; any prompt or decision-logic edit starts a new one.
    """
    from pa_agent.config.paths import PROJECT_ROOT, PROMPT_DIR

    pkg = PROJECT_ROOT / "pa_agent"
    files = [p for p in PROMPT_DIR.rglob("*") if p.suffix in (".txt", ".md")]
    files += list((pkg / "ai").rglob("*.py"))
    files.append(pkg / "util" / "trade_metrics.py")
    h = hashlib.sha256()
    for path in sorted(p for p in files if p.is_file()):
        h.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:8]


@functools.cache
def code_sha() -> str:
    """Git commit the process started from (for tracing only; "" outside a checkout)."""
    from pa_agent.config.paths import PROJECT_ROOT

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


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
    signal["model"] = str(getattr(getattr(settings, "provider", None), "model", "") or "")
    try:
        signal["prompt_hash"] = prompt_hash()
    except OSError as exc:
        logger.warning("prompt hash unavailable: %s", exc)
    signal["code_sha"] = code_sha()
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
