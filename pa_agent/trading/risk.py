"""Pure risk / sizing rules — no I/O, fully unit-testable.

AI output is never trusted for safety: every rule here is enforced in code.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any


@dataclass(frozen=True)
class InstrumentSpec:
    inst_id: str
    ct_val: float      # base-currency amount per contract (e.g. 0.01 BTC)
    lot_sz: float      # contract size increment
    min_sz: float      # minimum order size in contracts
    tick_sz: float     # price increment

    @classmethod
    def from_okx(cls, row: dict[str, Any]) -> InstrumentSpec:
        return cls(
            inst_id=str(row["instId"]),
            ct_val=float(row.get("ctVal") or 1),
            lot_sz=float(row.get("lotSz") or 1),
            min_sz=float(row.get("minSz") or row.get("lotSz") or 1),
            tick_sz=float(row.get("tickSz") or 0.01),
        )


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 1.0
    leverage: int = 5
    min_reward_risk: float = 1.0
    min_trade_confidence: int = 0
    min_stop_distance_pct: float = 0.3
    max_stop_distance_pct: float = 8.0
    fee_buffer_pct: float = 0.1
    max_daily_loss_pct: float = 3.0
    max_open_positions: int = 1
    tp1_close_fraction: float = 0.5

    @classmethod
    def from_settings(cls, t: Any) -> RiskConfig:
        return cls(
            risk_per_trade_pct=float(t.risk_per_trade_pct),
            leverage=int(t.leverage),
            min_reward_risk=float(t.min_reward_risk),
            min_trade_confidence=int(t.min_trade_confidence),
            min_stop_distance_pct=float(t.min_stop_distance_pct),
            max_stop_distance_pct=float(t.max_stop_distance_pct),
            fee_buffer_pct=float(t.fee_buffer_pct),
            max_daily_loss_pct=float(t.max_daily_loss_pct),
            max_open_positions=int(t.max_open_positions),
            tp1_close_fraction=float(t.tp1_close_fraction),
        )


@dataclass(frozen=True)
class EntryPlan:
    """Validated, executable plan derived from a signal."""
    direction: str          # long / short
    entry_type: str         # limit / stop / market
    entry: float            # planned entry (market: reference price)
    stop_loss: float
    tp1: float
    tp2: float | None
    reward_risk: float
    note: str = ""


class RiskRejected(Exception):
    """Signal failed a risk rule; message is user-facing (Chinese)."""


# ── numeric helpers ───────────────────────────────────────────────────────────

def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    q = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
    return float(q * Decimal(str(step)))


def round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    q = (Decimal(str(price)) / Decimal(str(tick))).to_integral_value(rounding=ROUND_HALF_UP)
    return float(q * Decimal(str(tick)))


def fmt_num(value: float) -> str:
    """OKX wants plain decimal strings (no exponent)."""
    d = Decimal(str(value)).normalize()
    text = format(d, "f")
    return text if text not in ("-0", "") else "0"


# ── plan validation ───────────────────────────────────────────────────────────

def build_entry_plan(signal: dict[str, Any], ref_price: float, cfg: RiskConfig) -> EntryPlan:
    """Validate *signal* against the current price; raise :class:`RiskRejected`."""
    direction = str(signal.get("direction") or "")
    entry_type = str(signal.get("order_type") or "none")
    if entry_type == "none" or direction not in ("long", "short"):
        raise RiskRejected("非下单信号")
    sl = signal.get("stop_loss")
    tp1 = signal.get("tp1")
    tp2 = signal.get("tp2")
    entry = signal.get("entry")
    if sl is None:
        raise RiskRejected("缺少止损价，拒绝下单")
    if tp1 is None:
        raise RiskRejected("缺少止盈价，拒绝下单")
    if ref_price is None or ref_price <= 0:
        raise RiskRejected("无法获取当前价格")
    sign = 1 if direction == "long" else -1
    note = ""

    conf = signal.get("trade_confidence")
    if cfg.min_trade_confidence > 0 and (conf is None or int(conf) < cfg.min_trade_confidence):
        raise RiskRejected(f"交易把握度 {conf} 低于门槛 {cfg.min_trade_confidence}")

    # Price must not already be beyond the stop.
    if (ref_price - sl) * sign <= 0:
        raise RiskRejected(f"现价 {ref_price} 已越过止损 {sl}，方案失效")

    if entry_type == "market" or entry is None:
        entry_type, entry = "market", float(ref_price)
    elif entry_type == "stop" and (ref_price - entry) * sign >= 0:
        # Breakout level already crossed during analysis latency → take it at market.
        entry_type, entry = "market", float(ref_price)
        note = "突破价已被越过，改为市价入场"
    elif entry_type == "limit" and (entry - ref_price) * sign > 0:
        # Marketable limit (e.g. buy limit above price) would fill instantly.
        entry_type, entry = "market", float(ref_price)
        note = "限价已优于现价，改为市价入场"

    entry = float(entry)
    risk = (entry - sl) * sign
    if risk <= 0:
        raise RiskRejected(f"止损 {sl} 不在入场价 {entry} 的{'下方' if sign > 0 else '上方'}")
    dist_pct = risk / entry * 100
    if dist_pct < cfg.min_stop_distance_pct:
        raise RiskRejected(f"止损距离 {dist_pct:.3f}% 过近（< {cfg.min_stop_distance_pct}%）")
    if dist_pct > cfg.max_stop_distance_pct:
        raise RiskRejected(f"止损距离 {dist_pct:.2f}% 过远（> {cfg.max_stop_distance_pct}%）")

    reward = (tp1 - entry) * sign
    if reward <= 0:
        raise RiskRejected(f"止盈 {tp1} 不在入场价 {entry} 的盈利方向")
    rr = reward / risk
    if rr < cfg.min_reward_risk:
        raise RiskRejected(f"盈亏比 {rr:.2f} 低于 {cfg.min_reward_risk}")
    if tp2 is not None and (tp2 - tp1) * sign <= 0:
        tp2 = None  # TP2 must be beyond TP1; otherwise close everything at TP1
    return EntryPlan(
        direction=direction,
        entry_type=entry_type,
        entry=entry,
        stop_loss=float(sl),
        tp1=float(tp1),
        tp2=float(tp2) if tp2 is not None else None,
        reward_risk=rr,
        note=note,
    )


# ── sizing ────────────────────────────────────────────────────────────────────

def size_position(
    *,
    equity: float,
    avail: float,
    plan: EntryPlan,
    spec: InstrumentSpec,
    cfg: RiskConfig,
) -> tuple[float, float]:
    """Return ``(contracts, risk_amount)``; raise :class:`RiskRejected` if too small.

    The loss budgeted per contract is the stop distance plus ``fee_buffer_pct`` of
    the entry price (round-trip fees + stop slippage), so a stop-out costs about
    ``risk_per_trade_pct`` of equity all-in rather than before costs.
    """
    if equity <= 0:
        raise RiskRejected("账户权益为 0")
    risk_amount = equity * cfg.risk_per_trade_pct / 100
    stop_distance = abs(plan.entry - plan.stop_loss)
    if stop_distance <= 0:
        raise RiskRejected("止损距离为 0")
    loss_per_contract = (stop_distance + plan.entry * cfg.fee_buffer_pct / 100) * spec.ct_val
    contracts = risk_amount / loss_per_contract

    notional_per_contract = plan.entry * spec.ct_val
    max_by_leverage = equity * cfg.leverage / notional_per_contract
    max_by_margin = max(avail, 0) * cfg.leverage * 0.95 / notional_per_contract
    contracts = min(contracts, max_by_leverage, max_by_margin)
    contracts = floor_to_step(contracts, spec.lot_sz)
    if contracts < spec.min_sz:
        raise RiskRejected(
            f"按风险 {cfg.risk_per_trade_pct}% 计算仓位 {contracts} 张，低于最小下单量 {spec.min_sz} 张"
        )
    actual_risk = contracts * loss_per_contract
    return contracts, actual_risk


def split_take_profit(
    contracts: float, plan: EntryPlan, spec: InstrumentSpec, fraction: float
) -> tuple[float, float]:
    """Return ``(tp1_size, tp2_size)``; tp2_size is 0 when everything closes at TP1."""
    if plan.tp2 is None or fraction >= 1:
        return contracts, 0.0
    tp1 = floor_to_step(contracts * fraction, spec.lot_sz)
    tp2 = floor_to_step(contracts - tp1, spec.lot_sz)
    if tp1 < spec.min_sz or tp2 < spec.min_sz:
        return contracts, 0.0
    return tp1, tp2


# ── account-level gates ───────────────────────────────────────────────────────

def daily_loss_blocked(
    realized_today: float, equity_day_start: float, cfg: RiskConfig
) -> tuple[bool, str]:
    if equity_day_start <= 0:
        return False, ""
    limit = equity_day_start * cfg.max_daily_loss_pct / 100
    if realized_today <= -limit:
        return True, (
            f"今日已实现亏损 {realized_today:.2f} 超过上限 {limit:.2f}"
            f"（{cfg.max_daily_loss_pct}%），暂停开新仓"
        )
    return False, ""
