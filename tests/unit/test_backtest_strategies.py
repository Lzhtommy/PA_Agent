"""Rule strategies for the backtest: causal setups, mirrors, baselines."""
from __future__ import annotations

import random

from pa_agent.backtest.data import Bar
from pa_agent.backtest.strategies import (
    _pullback_setups,
    prepare,
    pullback_orders,
    random_orders,
    shuffled_bars,
)

H4 = 4 * 3_600_000


def _bars(highs, lows):
    return [Bar(i * H4, (h + lo) / 2, h, lo, (h + lo) / 2, 1.0) for i, (h, lo) in enumerate(zip(highs, lows, strict=True))]


def test_pullback_setups_h1_then_h2_after_new_down_leg():
    # swing high at 3, pull back, H1 trigger at 7, deeper low (re-arms), H2 setup
    highs = [10, 11, 12, 15, 13, 12, 11, 12, 11.5, 10.5, 11, 11.8]
    lows = [9, 10, 11, 13, 12, 11, 10, 11, 10.2, 9.5, 10, 11]
    setups = _pullback_setups(highs, lows)
    assert setups[5] == (1, 5)          # leg confirmed at 5: next break of a prior high is H1
    assert setups[7] is None            # triggered at 7 → disarmed
    assert setups[9] == (2, 9)          # 9 broke below the H1 signal bar low → H2 armed


def test_setups_are_causal():
    highs = [10, 11, 12, 15, 13, 12, 11, 12, 11.5, 10.5, 11, 11.8]
    lows = [9, 10, 11, 13, 12, 11, 10, 11, 10.2, 9.5, 10, 11]
    full = _pullback_setups(highs, lows)
    for k in range(5, len(highs)):
        assert _pullback_setups(highs[:k + 1], lows[:k + 1])[k] == full[k]


def test_pullback_orders_respect_trend_filter():
    up = [100 + i * 0.5 + (2 if i % 7 == 3 else 0) for i in range(80)]
    bars = [Bar(i * H4, h - 1, h, h - 2, h - 0.5, 1.0) for i, h in enumerate(up)]
    orders = pullback_orders(prepare(bars, 0.1))
    assert orders and all(o.direction == "long" for o in orders)
    assert all(o.entry > o.stop and o.tp2 > o.tp1 > o.entry for o in orders)


def test_shuffled_bars_keep_shapes_but_not_order():
    rng = random.Random(1)
    bars = [Bar(i * H4, 100 + i, 101 + i, 99 + i, 100.5 + i, 1.0) for i in range(50)]
    sb = shuffled_bars(bars, rng)
    assert len(sb) == len(bars) and [b.ts for b in sb] == [b.ts for b in bars]
    assert all(b.low <= min(b.open, b.close) and b.high >= max(b.open, b.close) for b in sb)


def test_random_orders_copy_shape():
    bars = [Bar(i * H4, 100, 101, 99, 100, 1.0) for i in range(60)]
    s = prepare(bars, 0.1)
    from pa_agent.backtest.sim import Order
    tmpl = [Order(30, "short", "stop", 99.0, 101.0, 97.0, 95.0)]
    (o,) = random_orders(s, tmpl, 20, 59, random.Random(3))
    risk = abs(o.stop - bars[o.signal_idx].close)
    assert o.direction == "short" and o.entry_type == "market"
    assert abs(bars[o.signal_idx].close - o.tp1) == risk and abs(bars[o.signal_idx].close - o.tp2) == 2 * risk
