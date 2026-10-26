"""Rule-based price-action strategies (definitions fixed in research/okx_backtest/PREREG.md).

Each builder turns oldest-first bars into :class:`~pa_agent.backtest.sim.Order` objects
decided at a bar's close using only that bar and older ones.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from pa_agent.backtest.data import Bar
from pa_agent.backtest.sim import Order
from pa_agent.indicators.atr import atr_full
from pa_agent.indicators.ema import ema_full

PIVOT_RIGHT = 2        # fractal swing: 2 bars on each side
MAX_LEG_BARS = 30      # a pullback leg older than this is over
RANGE_BARS = 30


@dataclass(frozen=True)
class Series:
    bars: list[Bar]
    ema: list[float]
    atr: list[float]
    tick: float

    @property
    def n(self) -> int:
        return len(self.bars)


def prepare(bars: list[Bar], tick: float) -> Series:
    closes = [b.close for b in bars]
    return Series(bars, ema_full(closes, 20),
                  atr_full([b.high for b in bars], [b.low for b in bars], closes, 14), tick)


def _ok(*xs: float) -> bool:
    return all(not math.isnan(x) for x in xs)


def r_targets(direction: str, entry: float, stop: float, tp1_r: float,
              tp2_r: float | None) -> tuple[float, float | None]:
    sign = 1 if direction == "long" else -1
    risk = abs(entry - stop)
    return entry + sign * tp1_r * risk, (entry + sign * tp2_r * risk if tp2_r else None)


EXITS = {"live": (1.0, 2.0), "1r": (1.0, None), "2r": (2.0, None)}


# ── A. trend pullback (H/L triggers) ──────────────────────────────────────────

def _pullback_setups(highs: list[float], lows: list[float]) -> list[tuple[int, int] | None]:
    """Per bar: ``(trigger_number, signal_bar)`` if a buy-stop above this bar would be
    the next H trigger of an active pullback leg, else None. Mirrors
    ``market_features._count_pullback_triggers`` but emits at every bar close."""
    n = len(highs)
    out: list[tuple[int, int] | None] = [None] * n
    state: dict | None = None
    for k in range(n):
        s = k - PIVOT_RIGHT
        if (state is None and s >= 2 and highs[s] > highs[s - 1] and highs[s] > highs[s - 2]
                and highs[s] > highs[s + 1] and highs[s] > highs[k]):
            state = {"s": s, "sh": highs[s], "count": 0, "armed": True, "sig_low": None}
        if state is None:
            continue
        if (k > state["s"] + PIVOT_RIGHT and highs[k] > state["sh"]) or k - state["s"] > MAX_LEG_BARS:
            state = None
            continue
        if not state["armed"] and state["sig_low"] is not None and lows[k] < state["sig_low"]:
            state["armed"] = True
        if state["armed"] and k > state["s"] + PIVOT_RIGHT and highs[k] > highs[k - 1]:
            state["count"] += 1
            state["armed"] = False
            state["sig_low"] = lows[k - 1]
        if state["armed"] and k >= state["s"] + PIVOT_RIGHT:
            out[k] = (state["count"] + 1, k)
    return out


def pullback_orders(s: Series, *, triggers: tuple[int, ...] = (1, 2), exit: str = "live") -> list[Order]:
    tp1_r, tp2_r = EXITS[exit]
    bars = s.bars
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    longs = _pullback_setups(highs, lows)
    shorts = _pullback_setups([-x for x in lows], [-x for x in highs])
    orders = []
    for i in range(10, s.n):
        e, b = s.ema, bars[i]
        if not _ok(e[i], e[i - 10]):
            continue
        if longs[i] and longs[i][0] in triggers and e[i] > e[i - 10] and b.close > e[i]:
            entry, stop = b.high + s.tick, b.low - s.tick
            tp1, tp2 = r_targets("long", entry, stop, tp1_r, tp2_r)
            orders.append(Order(i, "long", "stop", entry, stop, tp1, tp2, f"H{longs[i][0]}"))
        if shorts[i] and shorts[i][0] in triggers and e[i] < e[i - 10] and b.close < e[i]:
            entry, stop = b.low - s.tick, b.high + s.tick
            tp1, tp2 = r_targets("short", entry, stop, tp1_r, tp2_r)
            orders.append(Order(i, "short", "stop", entry, stop, tp1, tp2, f"L{shorts[i][0]}"))
    return orders


# ── trading range (B, C) ──────────────────────────────────────────────────────

def trading_range(s: Series, first: int, last: int, atr_now: float) -> tuple[float, float] | None:
    """``(low, high)`` of bars ``first..last`` if they form a trading range per PREREG."""
    if first < 20 or not _ok(atr_now, s.ema[last], s.ema[last - 20]):
        return None
    window = s.bars[first:last + 1]
    hi = max(b.high for b in window)
    lo = min(b.low for b in window)
    h = hi - lo
    if not 3 * atr_now <= h <= 10 * atr_now:
        return None
    if abs(s.ema[last] - s.ema[last - 20]) >= 0.1 * h:
        return None
    top = sum(1 for b in window if b.high >= hi - 0.2 * h)
    bottom = sum(1 for b in window if b.low <= lo + 0.2 * h)
    return (lo, hi) if top >= 2 and bottom >= 2 else None


def range_fade_orders(s: Series, *, exit: str = "structure") -> list[Order]:
    orders = []
    for i in range(RANGE_BARS, s.n):
        a = s.atr[i]
        rng = trading_range(s, i - RANGE_BARS + 1, i, a)
        if rng is None:
            continue
        lo, hi = rng
        h = hi - lo
        mid = (lo + hi) / 2
        if s.bars[i].close <= mid:
            d, entry, stop = "long", lo + 0.15 * h, lo - 0.5 * a
            tp1, tp2 = mid, hi - 0.15 * h
        else:
            d, entry, stop = "short", hi - 0.15 * h, hi + 0.5 * a
            tp1, tp2 = mid, lo + 0.15 * h
        if exit != "structure":
            tp1, tp2 = r_targets(d, entry, stop, *EXITS[exit])
        orders.append(Order(i, d, "limit", entry, stop, tp1, tp2, "fade"))
    return orders


def close_breakout_orders(s: Series, *, exit: str = "mm") -> list[Order]:
    orders = []
    for i in range(RANGE_BARS + 1, s.n):
        b, a = s.bars[i], s.atr[i - 1]
        rng = trading_range(s, i - RANGE_BARS, i - 1, a)
        if rng is None or b.high <= b.low:
            continue
        lo, hi = rng
        h = hi - lo
        pos = (b.close - b.low) / (b.high - b.low)
        if b.close > hi and pos >= 0.75:
            d, stop, mm = "long", b.low - s.tick, hi + h
        elif b.close < lo and pos <= 0.25:
            d, stop, mm = "short", b.high + s.tick, lo - h
        else:
            continue
        tp1, tp2 = r_targets(d, b.close, stop, 1.0, 2.0)
        if exit == "mm":
            sign = 1 if d == "long" else -1
            if (mm - tp1) * sign > 0:
                tp2 = mm
        elif exit != "live":
            tp1, tp2 = r_targets(d, b.close, stop, *EXITS[exit])
        orders.append(Order(i, d, "market", None, stop, tp1, tp2, "close_bo"))
    return orders


# ── baselines ─────────────────────────────────────────────────────────────────

def random_orders(s: Series, template: list[Order], lo_idx: int, hi_idx: int,
                  rng: random.Random) -> list[Order]:
    """Same count, directions, stop size (in ATRs) and R targets as ``template``, but
    entered at market on random bars in ``[lo_idx, hi_idx)``."""
    out = []
    for o in template:
        ref = s.bars[o.signal_idx].close if o.entry is None else o.entry
        risk = abs(ref - o.stop)
        a0 = s.atr[o.signal_idx]
        if not risk or not _ok(a0):
            continue
        k_atr = risk / a0
        tp1_r = abs(o.tp1 - ref) / risk
        tp2_r = abs(o.tp2 - ref) / risk if o.tp2 is not None else None
        for _ in range(20):
            j = rng.randrange(lo_idx, hi_idx)
            if _ok(s.atr[j]):
                break
        else:
            continue
        sign = 1 if o.direction == "long" else -1
        entry = s.bars[j].close
        stop = entry - sign * k_atr * s.atr[j]
        tp1, tp2 = r_targets(o.direction, entry, stop, tp1_r, tp2_r)
        out.append(Order(j, o.direction, "market", None, stop, tp1, tp2, "random"))
    return out


def shuffled_bars(bars: list[Bar], rng: random.Random) -> list[Bar]:
    """Bars rebuilt from shuffled per-bar price ratios to the previous close: keeps the
    distribution of bar shapes and moves, destroys any sequence structure."""
    ratios = [(b.open / p.close, b.high / p.close, b.low / p.close, b.close / p.close)
              for p, b in zip(bars, bars[1:], strict=False)]
    rng.shuffle(ratios)
    out = [bars[0]]
    for (o, h, lo, c), src in zip(ratios, bars[1:], strict=True):
        pc = out[-1].close
        out.append(Bar(src.ts, pc * o, pc * h, pc * lo, pc * c, src.volume))
    return out
