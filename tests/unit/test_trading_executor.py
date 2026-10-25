"""Executor lifecycle against an in-memory fake OKX broker."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pa_agent.exchange.okx_rest import OkxError, OkxNetworkError
from pa_agent.trading import executor as executor_mod
from pa_agent.trading.broker import (
    AccountInfo,
    ClosedPosition,
    OrderStatus,
    PositionInfo,
    StopInfo,
)
from pa_agent.trading.executor import KILL_KEY, Executor, ExecutorConfig
from pa_agent.trading.risk import InstrumentSpec, RiskConfig
from pa_agent.trading.store import TradeStore

INST = "BTC-USDT-SWAP"


@dataclass
class _Order:
    cid: str
    side: str
    kind: str          # limit / market / trigger / tp
    size: float
    price: float | None
    state: str = "live"
    filled: float = 0.0
    avg: float | None = None


@dataclass
class _Stop:
    cid: str
    algo_id: str
    size: float
    trigger: float
    state: str = "live"


@dataclass
class FakeBroker:
    demo: bool = True
    price: float = 100.0
    equity: float = 10_000.0
    orders: dict[str, _Order] = field(default_factory=dict)
    stops: dict[str, _Stop] = field(default_factory=dict)
    pos: float = 0.0
    pos_avg: float | None = None
    closed: ClosedPosition | None = None
    fail_stop: bool = False
    fail_cancel_entry: bool = False
    fail_close: bool = False
    fail_tp: bool = False
    entry_network_error: bool = False
    fill_ms: int = 123
    closed_calls: int = 0
    _n: int = 0

    def spec(self, inst_id):
        return InstrumentSpec(inst_id, ct_val=0.01, lot_sz=0.01, min_sz=0.01, tick_sz=0.1)

    def last_price(self, inst_id):
        return self.price

    def account(self):
        return AccountInfo(total_eq=self.equity, avail_eq=self.equity, upl=0.0)

    def prepare_instrument(self, inst_id, leverage, margin_mode):
        pass

    def place_entry(self, *, inst_id, side, entry_type, size, price, stop_loss, margin_mode,
                    cl_ord_id, sl_cl_ord_id):
        kind = "trigger" if entry_type == "stop" else entry_type
        o = _Order(cl_ord_id, side, kind, size, price)
        self.orders[cl_ord_id] = o
        if self.entry_network_error:  # order reaches the exchange but the reply is lost
            self._pending_attached = getattr(self, "_pending_attached", {})
            self._pending_attached[cl_ord_id] = (sl_cl_ord_id, stop_loss)
            raise OkxNetworkError("timeout")
        self._pending_attached = getattr(self, "_pending_attached", {})
        self._pending_attached[cl_ord_id] = (sl_cl_ord_id, stop_loss)
        if kind == "market":
            self._fill(o, self.price)
        self._n += 1
        return f"X{self._n}"

    def _fill(self, o, px, size=None):
        size = o.size if size is None else size
        o.filled = size
        o.avg = px
        o.state = "filled" if size >= o.size else "partially_filled"
        sign = 1 if o.side == "buy" else -1
        if o.kind in ("limit", "market", "trigger"):
            self.pos += sign * size
            self.pos_avg = px
            sl_cid, sl_px = self._pending_attached.get(o.cid, (None, None))
            if sl_cid:
                self._n += 1
                self.stops[sl_cid] = _Stop(sl_cid, f"A{self._n}", size, sl_px)
        else:  # take profit (reduce-only)
            self.pos += sign * size

    def entry_status(self, *, inst_id, entry_type, cl_ord_id, algo_id=""):
        return self.order_status(inst_id, cl_ord_id)

    def cancel_entry(self, *, inst_id, entry_type, cl_ord_id, algo_id=""):
        if self.fail_cancel_entry:
            raise OkxNetworkError("timeout")
        o = self.orders[cl_ord_id]
        if o.state == "filled":
            raise OkxError("51400", "already filled")
        o.state = "canceled"

    def place_take_profit(self, *, inst_id, side, size, price, margin_mode, cl_ord_id):
        if self.fail_tp:
            raise OkxError("51000", "tp rejected")
        self.orders[cl_ord_id] = _Order(cl_ord_id, side, "tp", size, price)

    def order_status(self, inst_id, cl_ord_id):
        o = self.orders.get(cl_ord_id)
        if o is None:
            raise OkxError("51603", "order does not exist")
        return OrderStatus(state=o.state, filled=o.filled, avg_px=o.avg, fill_ms=self.fill_ms if o.filled else None)

    def cancel_order(self, inst_id, cl_ord_id):
        self.orders[cl_ord_id].state = "canceled"

    def stop_status(self, sl_cl_ord_id):
        s = self.stops.get(sl_cl_ord_id)
        if s is None:
            raise OkxError("51603", "not found")
        return OrderStatus(state=s.state, ord_id=s.algo_id)

    def place_stop(self, *, inst_id, side, size, trigger_px, margin_mode, sl_cl_ord_id):
        if self.fail_stop:
            raise OkxError("51000", "bad stop")
        self._n += 1
        self.stops[sl_cl_ord_id] = _Stop(sl_cl_ord_id, f"A{self._n}", size, trigger_px)
        return f"A{self._n}"

    def pending_stops(self, inst_id):
        return [StopInfo(s.algo_id, s.cid, s.size, s.trigger) for s in self.live_stops()]

    def pending_reduce_orders(self, inst_id):
        return [o.cid for o in self.orders.values() if o.kind == "tp" and o.state in ("live", "partially_filled")]

    def cancel_order_id(self, inst_id, ord_id):
        self.orders[ord_id].state = "canceled"

    def cancel_stop(self, inst_id, algo_id):
        for s in self.stops.values():
            if s.algo_id == algo_id:
                s.state = "canceled"

    def position(self, inst_id):
        return PositionInfo(size=round(self.pos, 8), avg_px=self.pos_avg, upl=0.0)

    def open_positions(self):
        return [{"instId": INST, "pos": str(self.pos)}] if self.pos else []

    def last_closed_position(self, inst_id, opened_after_ms):
        return self.closed

    def close_position(self, inst_id, margin_mode):
        if self.fail_close:
            raise OkxError("51000", "close failed")
        self.closed_calls += 1
        self.pos = 0.0

    # helpers for tests
    def live_stops(self):
        return [s for s in self.stops.values() if s.state == "live"]

    def trigger_stop(self):
        for s in self.live_stops():
            s.state = "triggered"
        self.pos = 0.0


@pytest.fixture
def env(tmp_path):
    store = TradeStore(tmp_path / "q.db")
    broker = FakeBroker()
    ex = Executor(store, broker, ExecutorConfig(risk=RiskConfig()))
    return store, broker, ex


def _signal(store, uid="s1", **kw):
    sig = dict(signal_uid=uid, source="test", symbol=INST, timeframe="1h", direction="long",
               order_type="limit", entry=99.0, stop_loss=98.0, tp1=101.0, tp2=103.0,
               trade_confidence=60)
    sig.update(kw)
    return store.insert_signal(sig)


def test_limit_entry_fill_tp_breakeven_close(env):
    store, broker, ex = env
    sid = _signal(store)
    ex.tick()
    trade = store.active_trades(INST)[0]
    assert trade["state"] == "pending"
    assert store.get_signal(sid)["status"] == "processed"
    # risk 100U / (1 * 0.01) = 10_000 contracts → capped by margin (10k*5*0.95/0.99)
    assert trade["contracts"] > 0

    broker._fill(broker.orders[trade["entry_cl_ord_id"]], 99.0)
    ex.tick()
    trade = store.get_trade(trade["id"])
    assert trade["state"] == "open"
    assert trade["tp1_cl_ord_id"] and trade["tp2_cl_ord_id"]
    assert len(broker.live_stops()) == 1

    broker.price = 101.5
    broker._fill(broker.orders[trade["tp1_cl_ord_id"]], 101.0)
    ex.tick()
    trade = store.get_trade(trade["id"])
    assert trade["breakeven_done"] == 1
    stops = broker.live_stops()
    assert len(stops) == 1 and stops[0].trigger == pytest.approx(99.0)

    broker._fill(broker.orders[trade["tp2_cl_ord_id"]], 103.0)
    broker.closed = ClosedPosition(99.0, 102.0, 42.0, 45.0, -3.0, 0.0, 0, 0)
    ex.tick()
    trade = store.get_trade(trade["id"])
    assert trade["state"] == "closed"
    assert trade["close_reason"] == "止盈2"
    assert trade["net_pnl"] == pytest.approx(42.0)
    assert broker.live_stops() == []  # leftover breakeven stop cancelled


def test_stop_loss_exit(env):
    store, broker, ex = env
    _signal(store, order_type="market", entry=None, tp1=104.0, tp2=106.0)
    ex.tick()  # market fills immediately in fake
    ex.tick()  # detect fill → place exits
    trade = store.list_trades()[0]
    assert trade["state"] == "open"
    broker.trigger_stop()
    broker.closed = ClosedPosition(100.0, 98.0, -101.0, -100.0, -1.0, 0.0, 0, 0)
    ex.tick()
    trade = store.get_trade(trade["id"])
    assert trade["state"] == "closed"
    assert trade["close_reason"] == "止损"
    assert broker.orders[trade["tp1_cl_ord_id"]].state == "canceled"


def test_no_order_signal_keeps_pending(env):
    store, broker, ex = env
    _signal(store)
    ex.tick()
    sid = _signal(store, uid="s2", order_type="none", direction="")
    ex.tick()
    trade = store.list_trades()[0]
    assert trade["state"] == "pending"
    assert store.get_signal(sid)["status_reason"] == f"不下单，保留挂单 #{trade['id']}"


def test_opposite_direction_plan_replaces_pending(env):
    store, broker, ex = env
    _signal(store)
    ex.tick()
    _signal(store, uid="s2", direction="short", entry=101.0, stop_loss=102.0, tp1=99.0, tp2=None)
    ex.tick()
    trades = store.list_trades()
    assert [(t["direction"], t["state"]) for t in trades] == [("short", "pending"), ("long", "canceled")]


def test_rejected_plan_cancels_pending_only_if_opposite(env):
    store, broker, ex = env
    _signal(store)
    ex.tick()
    _signal(store, uid="s2", tp1=99.5)  # same direction, fails reward/risk
    ex.tick()
    assert [t["state"] for t in store.list_trades()] == ["pending"]
    _signal(store, uid="s3", direction="short", entry=101.0, stop_loss=102.0, tp1=100.5, tp2=None)
    ex.tick()
    trade = store.list_trades()[0]
    assert trade["state"] == "canceled" and "反方向" in trade["close_reason"]


def test_pending_canceled_when_price_touches_stop_before_fill(env):
    store, broker, ex = env
    _signal(store, order_type="stop", entry=101.0, stop_loss=99.0, tp1=104.0, tp2=None)
    ex.tick()
    trade = store.active_trades()[0]
    broker.price = 99.5
    ex.tick()
    assert store.get_trade(trade["id"])["state"] == "pending"
    broker.price = 99.0
    ex.tick()
    trade = store.get_trade(trade["id"])
    assert trade["state"] == "canceled" and "已触及止损" in trade["close_reason"]


def test_new_plan_replaces_pending_and_same_plan_keeps_it(env):
    store, broker, ex = env
    _signal(store)
    ex.tick()
    same = _signal(store, uid="s2")
    ex.tick()
    assert store.get_signal(same)["status"] == "skipped"
    assert len(store.active_trades()) == 1
    _signal(store, uid="s3", entry=99.5)
    ex.tick()
    trades = store.list_trades()
    assert [t["state"] for t in trades] == ["pending", "canceled"]


def test_open_position_blocks_new_signal(env):
    store, broker, ex = env
    _signal(store, order_type="market", entry=None, tp1=104.0, tp2=106.0)
    ex.tick()
    ex.tick()
    sid = _signal(store, uid="s2", direction="short", stop_loss=102.0, entry=101.0,
                  tp1=97.0, tp2=None)
    ex.tick()
    sig = store.get_signal(sid)
    assert sig["status"] == "skipped" and "已有持仓" in sig["status_reason"]


def test_pending_expiry(env, monkeypatch):
    store, broker, ex = env
    _signal(store)
    ex.tick()
    trade = store.active_trades()[0]
    store.update_trade(trade["id"], expires_ms=1)
    ex.tick()
    assert store.get_trade(trade["id"])["state"] == "canceled"


def test_partial_fill_on_expiry_becomes_open_with_resized_stop(env):
    store, broker, ex = env
    _signal(store)
    ex.tick()
    trade = store.active_trades()[0]
    order = broker.orders[trade["entry_cl_ord_id"]]
    broker._fill(order, 99.0, size=round(order.size / 2, 2))
    store.update_trade(trade["id"], expires_ms=1)
    ex.tick()
    trade = store.get_trade(trade["id"])
    assert trade["state"] == "open"
    assert trade["filled_contracts"] == pytest.approx(round(order.size / 2, 2))
    live = broker.live_stops()
    assert len(live) == 1 and live[0].size == pytest.approx(trade["filled_contracts"])


def test_risk_rejection_recorded(env):
    store, broker, ex = env
    sid = _signal(store, tp1=99.5)  # reward/risk 0.5
    ex.tick()
    sig = store.get_signal(sid)
    assert sig["status"] == "rejected" and "盈亏比" in sig["status_reason"]
    assert store.list_trades() == []


def test_non_crypto_symbol_rejected(env):
    store, broker, ex = env
    sid = _signal(store, symbol="XAUUSD")
    ex.tick()
    assert store.get_signal(sid)["status"] == "rejected"


def test_older_signal_superseded(env):
    store, broker, ex = env
    a = _signal(store, uid="a")
    b = _signal(store, uid="b", entry=99.5)
    ex.tick()
    assert store.get_signal(a)["status"] == "skipped"
    assert store.get_signal(b)["status"] == "processed"


def test_stop_failure_flattens_position(env):
    store, broker, ex = env
    _signal(store, order_type="market", entry=None, tp1=104.0, tp2=106.0)
    ex.tick()
    broker.stops.clear()  # attached SL missing
    broker.fail_stop = True
    ex.tick()
    assert broker.closed_calls == 1


def test_kill_switch_flattens_and_blocks(env):
    store, broker, ex = env
    _signal(store, order_type="market", entry=None, tp1=104.0, tp2=106.0)
    ex.tick()
    ex.tick()
    store.set_kv(KILL_KEY, "1")
    sid = _signal(store, uid="s2")
    ex.tick()
    assert broker.closed_calls == 1
    assert store.get_signal(sid)["status"] == "skipped"


def test_max_open_positions(env):
    store, broker, ex = env
    _signal(store)
    ex.tick()
    sid = _signal(store, uid="eth", symbol="ETH-USDT-SWAP")
    ex.tick()
    sig = store.get_signal(sid)
    assert sig["status"] == "skipped" and "最大同时持仓" in sig["status_reason"]


def test_daily_loss_gate(env):
    store, broker, ex = env
    tid = store.insert_trade(dict(inst_id="ETH-USDT-SWAP", direction="long", entry_type="limit",
                                  state="closed", net_pnl=-400.0,
                                  closed_ms=executor_mod.now_ms()))
    assert tid
    sid = _signal(store)
    ex.tick()
    sig = store.get_signal(sid)
    assert sig["status"] == "skipped" and "暂停开新仓" in sig["status_reason"]


def test_watch_mode_breakout(env):
    store, broker, ex = env
    ex.cfg.breakout_mode = "watch"
    _signal(store, order_type="stop", entry=101.0, stop_loss=99.0, tp1=105.0, tp2=None)
    ex.tick()
    trade = store.active_trades()[0]
    assert trade["state"] == "pending" and not trade["entry_ord_id"]
    broker.price = 101.2
    ex.tick()  # triggers market entry (filled immediately in fake)
    ex.tick()
    trade = store.get_trade(trade["id"])
    assert trade["state"] == "open" and trade["entry_type"] == "market"


# ── regressions from code review ──────────────────────────────────────────────

def test_unconfirmed_cancel_keeps_trade_pending_and_skips_new_entry(env):
    store, broker, ex = env
    _signal(store)
    ex.tick()
    broker.fail_cancel_entry = True
    sid = _signal(store, uid="s2", entry=99.5)
    ex.tick()
    trades = store.list_trades()
    assert [t["state"] for t in trades] == ["pending"]  # old entry still live, no second entry
    assert "撤销尚未确认" in store.get_signal(sid)["status_reason"]
    broker.fail_cancel_entry = False
    ex.tick()  # retried because expires_ms was forced to 0
    assert store.get_trade(trades[0]["id"])["state"] == "canceled"


def test_ambiguous_entry_timeout_is_resolved_by_lookup(env):
    store, broker, ex = env
    broker.entry_network_error = True
    _signal(store)
    ex.tick()
    trade = store.list_trades()[0]
    assert trade["state"] == "pending"  # not "error": the order is live on the exchange
    broker.entry_network_error = False
    broker._fill(broker.orders[trade["entry_cl_ord_id"]], 99.0)
    ex.tick()
    assert store.get_trade(trade["id"])["state"] == "open"


def test_partial_fill_while_pending_is_protected(env):
    store, broker, ex = env
    _signal(store)
    ex.tick()
    trade = store.active_trades()[0]
    order = broker.orders[trade["entry_cl_ord_id"]]
    broker._pending_attached.pop(order.cid)  # OKX arms the attached SL only on full fill
    broker._fill(order, 99.0, size=1.0)
    ex.tick()
    live = broker.live_stops()
    assert store.get_trade(trade["id"])["state"] == "pending"
    assert len(live) == 1 and live[0].size == pytest.approx(1.0)


def test_opened_time_is_first_partial_fill(env):
    # OKX positions-history dates the position from its first fill; settling PnL
    # matches on opened_ms, so a later full fill must not move it.
    store, broker, ex = env
    _signal(store)
    ex.tick()
    trade = store.active_trades()[0]
    order = broker.orders[trade["entry_cl_ord_id"]]
    broker.fill_ms = 1_000
    broker._fill(order, 99.0, size=1.0)
    ex.tick()
    broker.fill_ms = 3_600_000
    broker._fill(order, 99.0)
    ex.tick()
    trade = store.get_trade(trade["id"])
    assert trade["state"] == "open"
    assert trade["opened_ms"] == 1_000


def test_partial_promotion_blocks_new_entry(env):
    store, broker, ex = env
    _signal(store)
    ex.tick()
    trade = store.active_trades()[0]
    broker._fill(broker.orders[trade["entry_cl_ord_id"]], 99.0, size=1.0)
    sid = _signal(store, uid="s2", entry=99.5)
    ex.tick()
    assert [t["state"] for t in store.list_trades()] == ["open"]
    assert store.get_signal(sid)["status"] == "skipped"


def test_stop_resized_after_tp1_even_without_breakeven(env):
    store, broker, ex = env
    ex.cfg.move_stop_to_breakeven = False
    _signal(store, order_type="market", entry=None, tp1=104.0, tp2=106.0)
    ex.tick()
    trade = store.list_trades()[0]
    broker._fill(broker.orders[trade["tp1_cl_ord_id"]], 104.0)
    ex.tick()
    live = broker.live_stops()
    assert len(live) == 1
    assert live[0].size == pytest.approx(abs(broker.pos))
    assert live[0].trigger == pytest.approx(98.0)


def test_duplicate_stops_reduced_to_one(env):
    store, broker, ex = env
    _signal(store, order_type="market", entry=None, tp1=104.0, tp2=106.0)
    ex.tick()
    broker.place_stop(inst_id=INST, side="sell", size=abs(broker.pos), trigger_px=98.0,
                      margin_mode="isolated", sl_cl_ord_id="manualdup")
    assert len(broker.live_stops()) == 2
    ex.tick()
    assert len(broker.live_stops()) == 1


def test_orphans_swept_before_new_entry_and_after_close(env):
    store, broker, ex = env
    broker.place_stop(inst_id=INST, side="sell", size=1, trigger_px=90.0,
                      margin_mode="isolated", sl_cl_ord_id="old")
    _signal(store)
    ex.tick()
    assert [s.cid for s in broker.live_stops()] == []


def test_kill_retries_until_close_succeeds(env):
    store, broker, ex = env
    _signal(store, order_type="market", entry=None, tp1=104.0, tp2=106.0)
    ex.tick()
    broker.fail_close = True
    store.set_kv(KILL_KEY, "1")
    ex.tick()
    assert broker.pos != 0 and len(broker.live_stops()) == 1  # still protected
    broker.fail_close = False
    ex.tick()
    assert broker.closed_calls == 1 and broker.pos == 0


def test_tp_failure_is_retried(env):
    store, broker, ex = env
    broker.fail_tp = True
    _signal(store, order_type="market", entry=None, tp1=104.0, tp2=106.0)
    ex.tick()
    trade = store.list_trades()[0]
    assert trade["state"] == "open" and not trade["tp1_cl_ord_id"]
    broker.fail_tp = False
    ex.tick()
    trade = store.get_trade(trade["id"])
    assert trade["tp1_cl_ord_id"] and trade["tp2_cl_ord_id"]


def test_bad_signal_does_not_block_trade_management(env, monkeypatch):
    store, broker, ex = env
    _signal(store, order_type="market", entry=None, tp1=104.0, tp2=106.0)
    ex.tick()
    trade = store.list_trades()[0]
    bad = _signal(store, uid="bad", symbol="ETH-USDT-SWAP")
    monkeypatch.setattr(executor_mod, "build_entry_plan",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("boom")))
    broker.trigger_stop()
    broker.closed = ClosedPosition(100.0, 98.0, -100.0, -99.0, -1.0, 0.0, 0, 0)
    ex.tick()
    assert store.get_trade(trade["id"])["state"] == "closed"
    assert store.get_signal(bad)["status"] == "error"
