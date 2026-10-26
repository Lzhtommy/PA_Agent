"""Execution simulator that mirrors the live OKX executor.

A signal is decided at the close of bar ``signal_idx``; the order works from the next
bar on. Rules copied from :mod:`pa_agent.trading` (keep them in sync):

* the plan goes through :func:`pa_agent.trading.risk.build_entry_plan` against the
  signal bar's close (crossed breakouts / marketable limits become market orders,
  stop-distance band, minimum reward:risk, TP2 beyond TP1);
* a resting entry expires after ``expiry_bars`` bars and is cancelled if price
  reaches the stop before it fills;
* after the fill: ``tp1_fraction`` closes at TP1, the stop on the rest moves to the
  entry (breakeven), the rest closes at TP2 (everything at TP1 when there is no TP2);
* R is net PnL over the planned risk *including* the fee buffer, like the live
  ``risk_amount`` (so a clean stop-out is about -1R).

Within one price unit (a bar, or a finer sub-bar when given) the order of events is
unknown, so ties resolve against the trade: a unit that touches both the stop and a
target is a stop; the unit an entry fills in can stop it out but cannot take profit;
after TP1, a unit that also trades back to breakeven exits the rest at breakeven.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field, replace
from typing import Sequence

from pa_agent.backtest.data import Bar
from pa_agent.trading.risk import RiskConfig, RiskRejected, build_entry_plan


@dataclass(frozen=True)
class Order:
    signal_idx: int
    direction: str            # long / short
    entry_type: str           # stop / limit / market
    entry: float | None       # None for market
    stop: float
    tp1: float
    tp2: float | None = None
    tag: str = ""


@dataclass(frozen=True)
class SimConfig:
    maker_fee: float = 0.0002          # limit entry / take profit
    taker_fee: float = 0.0005          # market / triggered entry, stop exit
    slippage_pct: float = 0.02         # extra cost on taker fills, % of price
    fee_buffer_pct: float = 0.1        # sizing buffer, as trading.fee_buffer_pct
    tp1_fraction: float = 0.5
    breakeven: bool = True
    expiry_bars: int = 3
    max_hold_bars: int = 180           # safety valve; closed at that bar's close
    risk: RiskConfig = field(default_factory=lambda: RiskConfig(
        min_reward_risk=1.0, min_stop_distance_pct=0.3, max_stop_distance_pct=8.0))


@dataclass
class Result:
    order: Order
    outcome: str              # rejected / expired / invalidated / replaced / stop / tp1_be /
                              # tp1_stop / tp1 / tp2 / timeout / open
    reason: str = ""
    entry_type: str = ""      # after build_entry_plan (may become market)
    planned_entry: float | None = None
    end_idx: int | None = None  # bar where an unfilled entry stopped working
    fill_idx: int | None = None
    fill_px: float | None = None
    exit_idx: int | None = None
    r: float | None = None    # net R, None when never filled
    fee_r: float = 0.0        # fees + slippage, in R

    @property
    def filled(self) -> bool:
        return self.fill_idx is not None


class PriceUnits:
    """Bars of the signal timeframe, optionally split into finer sub-bars for ordering."""

    def __init__(self, bars: Sequence[Bar], sub_bars: Sequence[Bar] | None = None,
                 step_ms: int | None = None) -> None:
        self.bars = list(bars)
        self._sub = list(sub_bars or [])
        self._sub_ts = [b.ts for b in self._sub]
        self._step = step_ms or (self.bars[1].ts - self.bars[0].ts if len(self.bars) > 1 else 0)

    def units(self, idx: int) -> list[Bar]:
        bar = self.bars[idx]
        if not self._sub:
            return [bar]
        lo = bisect_left(self._sub_ts, bar.ts)
        hi = bisect_left(self._sub_ts, bar.ts + self._step)
        parts = self._sub[lo:hi]
        # use the split only when it is complete and consistent with the bar
        if (len(parts) >= 2 and parts[0].ts == bar.ts
                and max(p.high for p in parts) <= bar.high + 1e-9
                and min(p.low for p in parts) >= bar.low - 1e-9):
            return parts
        return [bar]


def simulate(order: Order, prices: PriceUnits, cfg: SimConfig = SimConfig(), *,
             cancel_idx: int | None = None) -> Result:
    """Simulate one order. ``cancel_idx``: cancel a still-unfilled entry at that bar's open."""
    bars = prices.bars
    i = order.signal_idx
    if i + 1 >= len(bars):
        return Result(order, "rejected", "没有后续K线")
    signal = {"direction": order.direction, "order_type": order.entry_type, "entry": order.entry,
              "stop_loss": order.stop, "tp1": order.tp1, "tp2": order.tp2}
    try:
        plan = build_entry_plan(signal, bars[i].close, cfg.risk)
    except RiskRejected as exc:
        return Result(order, "rejected", str(exc))

    sign = 1 if plan.direction == "long" else -1
    res = Result(order, "open", plan.note, entry_type=plan.entry_type, planned_entry=plan.entry)
    risk_unit = abs(plan.entry - plan.stop_loss) + plan.entry * cfg.fee_buffer_pct / 100
    slip = cfg.slippage_pct / 100

    def worse(px: float, side: int) -> float:
        """Price after slippage for a taker fill buying (side=+1) or selling (-1)."""
        return px * (1 + slip * side)

    # ── entry ────────────────────────────────────────────────────────────────
    fill_px = fill_fee = None
    k = i + 1
    last_entry_bar = i + cfg.expiry_bars
    fill_unit = None
    while fill_px is None:
        if k >= len(bars):
            res.outcome, res.reason = "open", "数据结束仍未成交"
            return res
        if cancel_idx is not None and k >= cancel_idx:
            res.outcome, res.reason, res.end_idx = "replaced", "被新信号替换", cancel_idx
            return res
        if plan.entry_type != "market" and k > last_entry_bar:
            res.outcome, res.reason = "expired", f"{cfg.expiry_bars} 根K线内未成交"
            res.end_idx = last_entry_bar
            return res
        for u_i, u in enumerate(prices.units(k)):
            if plan.entry_type == "market":
                fill_px, fill_fee = worse(u.open, sign), cfg.taker_fee
            elif plan.entry_type == "stop":
                trig = (u.high >= plan.entry) if sign > 0 else (u.low <= plan.entry)
                if (u.open - plan.entry) * sign >= 0:
                    fill_px, fill_fee = worse(u.open, sign), cfg.taker_fee
                elif trig:
                    fill_px, fill_fee = worse(plan.entry, sign), cfg.taker_fee
                elif ((u.low if sign > 0 else u.high) - plan.stop_loss) * sign <= 0:
                    res.outcome, res.reason, res.end_idx = "invalidated", "成交前已触及止损", k
                    return res
            else:  # limit
                if (u.open - plan.entry) * sign <= 0:
                    fill_px, fill_fee = u.open, cfg.maker_fee
                elif ((u.low if sign > 0 else u.high) - plan.entry) * sign <= 0:
                    fill_px, fill_fee = plan.entry, cfg.maker_fee
            if fill_px is not None:
                fill_unit = u_i
                break
        if fill_px is None:
            k += 1
    res.fill_idx, res.fill_px = k, fill_px

    # ── exits ────────────────────────────────────────────────────────────────
    stop = plan.stop_loss
    has_tp2 = plan.tp2 is not None and cfg.tp1_fraction < 1
    remaining = 1.0
    legs: list[tuple[float, float, float]] = []   # (fraction, price, fee rate)
    tp1_done = False

    def finish(outcome: str, idx: int) -> Result:
        pnl = sum(f * (px - fill_px) * sign for f, px, _ in legs)
        fees = fill_px * fill_fee + sum(f * px * rate for f, px, rate in legs)
        slip_cost = sum(f * abs(px * slip) for f, px, rate in legs if rate == cfg.taker_fee)
        if fill_fee == cfg.taker_fee:
            slip_cost += abs(fill_px * slip)
        res.outcome, res.exit_idx = outcome, idx
        res.r = (pnl - fees) / risk_unit
        res.fee_r = (fees + slip_cost) / risk_unit
        return res

    def exit_stop(idx: int, px: float) -> Result:
        nonlocal remaining
        legs.append((remaining, worse(px, -sign), cfg.taker_fee))
        remaining = 0.0
        if tp1_done:
            be = cfg.breakeven and abs(stop - fill_px) < 1e-9
            return finish("tp1_be" if be else "tp1_stop", idx)
        return finish("stop", idx)

    def take(px: float, frac: float) -> None:
        nonlocal remaining
        legs.append((frac, px, cfg.maker_fee))
        remaining -= frac

    j = k
    while True:
        if j >= len(bars):
            res.outcome, res.reason = "open", "数据结束仍持仓"
            return res
        units = prices.units(j)
        start = fill_unit if j == k else 0
        for u_i in range(start, len(units)):
            u = units[u_i]
            fill_unit_now = j == k and u_i == fill_unit
            if not fill_unit_now and (u.open - stop) * sign <= 0:        # gap through the stop
                return exit_stop(j, u.open)
            if ((u.low if sign > 0 else u.high) - stop) * sign <= 0:
                return exit_stop(j, stop)
            if fill_unit_now:
                continue                                                  # no profit in the fill unit
            best = u.high if sign > 0 else u.low
            if not tp1_done and (best - plan.tp1) * sign >= 0:
                px = u.open if (u.open - plan.tp1) * sign >= 0 else plan.tp1
                if not has_tp2:
                    take(px, remaining)
                    return finish("tp1", j)
                take(px, cfg.tp1_fraction)
                tp1_done = True
                if cfg.breakeven:
                    stop = fill_px
                    if ((u.low if sign > 0 else u.high) - stop) * sign <= 0:
                        return exit_stop(j, stop)                         # tie → breakeven first
            if tp1_done and has_tp2 and (best - plan.tp2) * sign >= 0:
                px = u.open if (u.open - plan.tp2) * sign >= 0 else plan.tp2
                take(px, remaining)
                return finish("tp2", j)
        if j - k >= cfg.max_hold_bars:
            legs.append((remaining, worse(bars[j].close, -sign), cfg.taker_fee))
            remaining = 0.0
            return finish("timeout", j)
        j += 1


def _pending_at(res: Result, s: int) -> bool:
    """Was this order's entry still resting at the close of bar ``s``?"""
    if res.order.signal_idx >= s:
        return res.order.signal_idx == s
    if res.filled:
        return res.fill_idx > s
    if res.outcome in ("expired", "invalidated", "replaced"):
        return res.end_idx is not None and res.end_idx > s
    return res.outcome == "open"


def run_portfolio(orders: Sequence[Order], prices: PriceUnits, cfg: SimConfig = SimConfig()) -> list[Result]:
    """Live-executor interplay for one instrument: one trade at a time, a new signal
    replaces a still-resting entry, signals while a position is open are skipped."""
    out: list[Result] = []
    active: int | None = None          # index in ``out`` of the last accepted order
    for o in sorted(orders, key=lambda o: o.signal_idx):
        s = o.signal_idx
        if active is not None:
            a = out[active]
            if a.filled and a.fill_idx <= s and (a.exit_idx is None or a.exit_idx > s):
                out.append(Result(o, "skipped", "已有持仓"))
                continue
            if _pending_at(a, s):
                out[active] = simulate(a.order, prices, cfg, cancel_idx=s + 1)
        res = simulate(o, prices, cfg)
        out.append(res)
        if res.outcome != "rejected":
            active = len(out) - 1
    return out


def with_fees(cfg: SimConfig, **kw) -> SimConfig:
    return replace(cfg, **kw)
