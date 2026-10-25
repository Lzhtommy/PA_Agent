"""Risk / sizing rules for the OKX executor."""
from __future__ import annotations

import pytest

from pa_agent.trading.risk import (
    EntryPlan,
    InstrumentSpec,
    RiskConfig,
    RiskRejected,
    build_entry_plan,
    daily_loss_blocked,
    floor_to_step,
    fmt_num,
    round_to_tick,
    size_position,
    split_take_profit,
)

BTC = InstrumentSpec(inst_id="BTC-USDT-SWAP", ct_val=0.01, lot_sz=0.01, min_sz=0.01, tick_sz=0.1)
CFG = RiskConfig()


def _sig(**kw):
    base = dict(direction="long", order_type="limit", entry=100.0, stop_loss=98.0,
                tp1=104.0, tp2=106.0, trade_confidence=60)
    base.update(kw)
    return base


def test_limit_plan_ok():
    plan = build_entry_plan(_sig(), 101.0, CFG)
    assert plan.entry_type == "limit"
    assert plan.reward_risk == pytest.approx(2.0)
    assert plan.tp2 == 106.0


@pytest.mark.parametrize(
    "kw,ref,msg",
    [
        (dict(stop_loss=None), 101.0, "缺少止损"),
        (dict(tp1=None), 101.0, "缺少止盈"),
        (dict(stop_loss=101.0), 102.0, "不在入场价"),
        (dict(tp1=99.0), 101.0, "盈利方向"),
        (dict(tp1=101.0), 101.5, "盈亏比"),
        (dict(stop_loss=99.8), 101.0, "过近"),
        (dict(stop_loss=80.0, tp1=150.0), 101.0, "过远"),
        (dict(), 97.0, "已越过止损"),
    ],
)
def test_plan_rejections(kw, ref, msg):
    with pytest.raises(RiskRejected, match=msg):
        build_entry_plan(_sig(**kw), ref, CFG)


def test_confidence_gate():
    cfg = RiskConfig(min_trade_confidence=70)
    with pytest.raises(RiskRejected, match="把握度"):
        build_entry_plan(_sig(trade_confidence=60), 101.0, cfg)


def test_short_plan_and_tp2_must_be_beyond_tp1():
    plan = build_entry_plan(
        _sig(direction="short", entry=100.0, stop_loss=102.0, tp1=96.0, tp2=97.0), 99.0, CFG
    )
    assert plan.direction == "short"
    assert plan.tp2 is None  # 97 is not beyond 96 for a short


def test_crossed_breakout_becomes_market():
    plan = build_entry_plan(_sig(order_type="stop", entry=100.0, tp1=106.0), 100.5, CFG)
    assert plan.entry_type == "market"
    assert plan.entry == 100.5
    assert "突破" in plan.note


def test_marketable_limit_becomes_market():
    plan = build_entry_plan(_sig(entry=101.0, tp1=106.0), 100.0, CFG)
    assert plan.entry_type == "market"


def test_size_position_risk_based():
    plan = EntryPlan("long", "limit", 100_000.0, 99_000.0, 102_000.0, None, 2.0)
    cfg = RiskConfig(fee_buffer_pct=0)
    contracts, risk = size_position(equity=10_000, avail=10_000, plan=plan, spec=BTC, cfg=cfg)
    # risk 100U / (1000 * 0.01) = 10 contracts → capped by leverage 5: 50_000 / 1000 = 50 → 10
    assert contracts == pytest.approx(10.0)
    assert risk == pytest.approx(100.0)


def test_size_position_budgets_fees_into_risk():
    plan = EntryPlan("long", "limit", 100_000.0, 99_000.0, 102_000.0, None, 2.0)
    cfg = RiskConfig(fee_buffer_pct=0.25)
    contracts, risk = size_position(equity=10_000, avail=10_000, plan=plan, spec=BTC, cfg=cfg)
    # per contract: (1000 stop + 250 fees) * 0.01 = 12.5U → 100U / 12.5 = 8 contracts
    assert contracts == pytest.approx(8.0)
    assert risk == pytest.approx(100.0)


def test_size_position_capped_by_leverage():
    plan = EntryPlan("long", "limit", 100_000.0, 99_900.0, 100_300.0, None, 3.0)
    contracts, _ = size_position(equity=10_000, avail=10_000, plan=plan, spec=BTC, cfg=CFG)
    # uncapped would be 100 contracts; leverage 5 caps notional at 50k → 50 contracts
    assert contracts == pytest.approx(47.5)  # margin cap: avail * 5 * 0.95 / 1000


def test_size_position_too_small():
    spec = InstrumentSpec("X-USDT-SWAP", ct_val=1, lot_sz=1, min_sz=1, tick_sz=0.1)
    plan = EntryPlan("long", "limit", 100.0, 50.0, 200.0, None, 2.0)
    with pytest.raises(RiskRejected, match="最小下单量"):
        size_position(equity=100, avail=100, plan=plan, spec=spec, cfg=CFG)


def test_split_take_profit():
    plan = EntryPlan("long", "limit", 100.0, 98.0, 104.0, 106.0, 2.0)
    assert split_take_profit(0.05, plan, BTC, 0.5) == (0.02, 0.03)
    assert split_take_profit(0.01, plan, BTC, 0.5) == (0.01, 0.0)  # tp1 below min size
    no_tp2 = EntryPlan("long", "limit", 100.0, 98.0, 104.0, None, 2.0)
    assert split_take_profit(0.05, no_tp2, BTC, 0.5) == (0.05, 0.0)


def test_numeric_helpers():
    assert floor_to_step(0.0567, 0.01) == pytest.approx(0.05)
    assert round_to_tick(100.06, 0.1) == pytest.approx(100.1)
    assert fmt_num(1e-05) == "0.00001"
    assert fmt_num(83000.0) == "83000"


def test_daily_loss_blocked():
    cfg = RiskConfig(max_daily_loss_pct=3.0)
    assert daily_loss_blocked(-299, 10_000, cfg)[0] is False
    blocked, msg = daily_loss_blocked(-300, 10_000, cfg)
    assert blocked and "暂停开新仓" in msg
