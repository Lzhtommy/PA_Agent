"""Backtest execution simulator: fills, exits, tie-breaking, sub-bars, portfolio rules."""
from __future__ import annotations

import pytest

from pa_agent.backtest.data import Bar, find_gaps
from pa_agent.backtest.sim import Order, PriceUnits, SimConfig, run_portfolio, simulate

H4 = 4 * 3_600_000
H1 = 3_600_000
FREE = SimConfig(maker_fee=0, taker_fee=0, slippage_pct=0, fee_buffer_pct=0)


def bars(*ohlc):
    """Bars at 4h steps; bar 0 is the signal bar (its close is the reference price)."""
    return [Bar(i * H4, o, h, l, c, 1.0) for i, (o, h, l, c) in enumerate(ohlc)]


def long_stop(**kw):
    base = dict(signal_idx=0, direction="long", entry_type="stop", entry=101.0, stop=99.0,
                tp1=103.0, tp2=105.0)
    base.update(kw)
    return Order(**base)


SIGNAL = (100, 100.5, 99.5, 100)


def test_breakout_long_reaches_tp2():
    p = PriceUnits(bars(SIGNAL, (100, 101.5, 100, 101.2), (101.2, 103.5, 101.1, 103), (103, 105.5, 102.5, 105)))
    r = simulate(long_stop(), p, FREE)
    assert r.outcome == "tp2" and r.fill_idx == 1 and r.exit_idx == 3
    assert r.r == pytest.approx(0.5 * 1 + 0.5 * 2)    # 1R + 2R legs


def test_stop_out_is_minus_one_r_without_costs_and_about_one_r_with_buffer():
    p = PriceUnits(bars(SIGNAL, (100, 101.5, 100.5, 101), (101, 101.2, 98.5, 99)))
    assert simulate(long_stop(), p, FREE).r == pytest.approx(-1.0)
    real = simulate(long_stop(), p, SimConfig())
    assert real.outcome == "stop" and -1.02 < real.r < -0.95  # buffer absorbs fees + slippage
    assert real.fee_r > 0


def test_tp1_then_breakeven():
    p = PriceUnits(bars(SIGNAL, (100, 101.5, 100.5, 101), (101, 103.2, 101, 102.5), (102.5, 102.6, 100.5, 100.8)))
    r = simulate(long_stop(), p, FREE)
    assert r.outcome == "tp1_be" and r.r == pytest.approx(0.5)


def test_entry_expires_after_three_bars():
    p = PriceUnits(bars(SIGNAL, *[(100, 100.8, 99.8, 100.2)] * 5))
    r = simulate(long_stop(), p, FREE)
    assert r.outcome == "expired" and r.end_idx == 3 and r.r is None


def test_entry_invalidated_when_stop_trades_first():
    p = PriceUnits(bars(SIGNAL, (100, 100.2, 98.8, 99), (99, 101.5, 99, 101)))
    assert simulate(long_stop(), p, FREE).outcome == "invalidated"


def test_bar_touching_stop_and_target_counts_as_stop():
    p = PriceUnits(bars(SIGNAL, (100, 101.5, 100.5, 101), (101, 103.5, 98.5, 101)))
    assert simulate(long_stop(), p, FREE).outcome == "stop"


def test_fill_bar_can_stop_out_but_not_take_profit():
    p = PriceUnits(bars(SIGNAL, (100, 103.5, 100, 103), (103, 103.2, 102, 102.5)))
    r = simulate(long_stop(), p, FREE)
    assert r.fill_idx == 1 and r.outcome != "tp1"
    p2 = PriceUnits(bars(SIGNAL, (100, 101.5, 98.5, 99)))
    assert simulate(long_stop(), p2, FREE).outcome == "stop"


def test_sub_bars_resolve_order_inside_a_bar():
    b = bars(SIGNAL, (100, 101.5, 100.5, 101), (101, 103.5, 98.5, 101))
    # bar 2 seen through 1h sub-bars: TP1 first, then a drop back through the entry
    subs = [Bar(2 * H4 + n * H1, *ohlc, 1.0) for n, ohlc in enumerate(
        [(101, 103.5, 101.1, 103), (103, 103.2, 101.5, 102), (102, 102, 98.5, 100.8), (100.8, 101, 100.6, 101)])]
    assert simulate(long_stop(), PriceUnits(b), FREE).outcome == "stop"
    r = simulate(long_stop(), PriceUnits(b, subs, step_ms=H4), FREE)
    assert r.outcome == "tp1_be" and r.r == pytest.approx(0.5)


def test_limit_fills_on_touch_and_at_a_better_gap_open():
    order = Order(0, "long", "limit", 99.0, 97.0, 101.0, None)
    touch = PriceUnits(bars(SIGNAL, (100, 100.2, 98.9, 99.5), (99.5, 101.5, 99.4, 101)))
    r = simulate(order, touch, FREE)
    assert r.fill_px == 99.0 and r.outcome == "tp1" and r.r == pytest.approx(1.0)
    gap = PriceUnits(bars(SIGNAL, (98.5, 99.2, 98.4, 99), (99, 101.5, 98.9, 101)))
    assert simulate(order, gap, FREE).fill_px == 98.5


def test_short_mirror():
    order = Order(0, "short", "stop", 99.0, 101.0, 97.0, 95.0)
    p = PriceUnits(bars(SIGNAL, (100, 100, 98.5, 99), (99, 98.9, 96.8, 97), (97, 97.5, 94.5, 95)))
    r = simulate(order, p, FREE)
    assert r.outcome == "tp2" and r.r == pytest.approx(1.5)


def test_live_risk_rules_apply():
    p = PriceUnits(bars(SIGNAL, (100, 101.5, 100, 101)))
    tight = long_stop(stop=100.9, tp1=101.2, tp2=None)          # 0.1% stop < 0.3% minimum
    assert simulate(tight, p).outcome == "rejected"
    crossed = long_stop(entry=99.8)                              # breakout already crossed
    assert simulate(crossed, p, FREE).entry_type == "market"


def test_portfolio_replaces_pending_and_skips_while_open():
    b = bars(SIGNAL, (100, 100.8, 99.8, 100.2), (100.2, 101.5, 100, 101.2),
             (101.2, 102, 100.8, 101.5), (101.5, 103.5, 101, 103), (103, 103.2, 102, 102.5))
    p = PriceUnits(b)
    first = long_stop()                                           # would fill on bar 2
    second = long_stop(signal_idx=1, entry=101.4, stop=99.4, tp1=103.4, tp2=None)
    third = long_stop(signal_idx=3, entry=101.6, stop=99.6, tp1=103.6, tp2=None)
    out = run_portfolio([first, second, third], p, FREE)
    assert out[0].outcome == "replaced"
    assert out[1].filled and out[1].fill_idx == 2
    assert out[2].outcome == "skipped"


def test_find_gaps():
    b = [Bar(0, 1, 1, 1, 1, 0), Bar(H4, 1, 1, 1, 1, 0), Bar(3 * H4, 1, 1, 1, 1, 0)]
    assert find_gaps(b, H4) == [(H4, 3 * H4)]
