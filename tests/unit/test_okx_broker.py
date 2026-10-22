"""OkxBroker against a scripted OKX REST client."""
from __future__ import annotations

import inspect

from pa_agent.trading.broker import Broker, OkxBroker


class _Rest:
    demo = True

    def __init__(self, **responses):
        self.r = responses
        self.posts = []

    def instrument(self, inst_id):
        return {"instId": inst_id, "ctVal": "0.01", "lotSz": "0.01", "minSz": "0.01", "tickSz": "0.1"}

    def balance(self):
        return self.r["balance"]

    def pending_algo_orders(self, ord_type, inst_id=None):
        return self.r.get("algos", [])

    def pending_orders(self, inst_id=None):
        return self.r.get("orders", [])

    def positions_history(self, inst_id=None, limit=100):
        return self.r.get("history", [])

    def get_algo_order(self, algo_id=None, algo_cl_ord_id=None):
        return self.r["algo"]

    def get_order(self, inst_id, cl_ord_id=None, ord_id=None):
        return self.r["order"]

    def place_order(self, body):
        self.posts.append(("order", body))
        return {"ordId": "1"}

    def place_algo_order(self, body):
        self.posts.append(("algo", body))
        return {"algoId": "9"}


def test_okx_broker_implements_protocol():
    proto = {n for n, _ in inspect.getmembers(Broker, inspect.isfunction) if not n.startswith("_")}
    missing = [n for n in proto if not callable(getattr(OkxBroker, n, None))]
    assert missing == []


def test_account_avail_zero_is_not_replaced():
    b = OkxBroker(_Rest(balance={"totalEq": "1000", "details": [
        {"ccy": "USDT", "availEq": "0", "availBal": "500"}]}))
    assert b.account().avail_eq == 0.0
    b = OkxBroker(_Rest(balance={"totalEq": "1000", "details": [
        {"ccy": "USDT", "availEq": "", "availBal": "500"}]}))
    assert b.account().avail_eq == 500.0


def test_pending_stops_and_reduce_orders():
    b = OkxBroker(_Rest(
        algos=[{"algoId": "7", "algoClOrdId": "pa1s", "sz": "0.5", "slTriggerPx": "98.1"},
               {"algoId": "8", "algoClOrdId": "", "sz": "", "slTriggerPx": "97"}],
        orders=[{"ordId": "11", "reduceOnly": "true"}, {"ordId": "12", "reduceOnly": "false"}],
    ))
    stops = b.pending_stops("BTC-USDT-SWAP")
    assert stops[0].size == 0.5 and stops[0].trigger_px == 98.1
    assert stops[1].size is None
    assert b.pending_reduce_orders("BTC-USDT-SWAP") == ["11"]


def test_last_closed_position_skips_partial_rows():
    b = OkxBroker(_Rest(history=[
        {"type": "1", "cTime": "1000000", "uTime": "2000000", "realizedPnl": "5"},
        {"type": "2", "cTime": "1000000", "uTime": "1900000", "realizedPnl": "42",
         "pnl": "45", "fee": "-3", "fundingFee": "0", "closeAvgPx": "101"},
    ]))
    closed = b.last_closed_position("BTC-USDT-SWAP", 1_000_000)
    assert closed is not None and closed.realized_pnl == 42.0 and closed.close_avg_px == 101.0


def test_trigger_entry_status_follows_child_order():
    b = OkxBroker(_Rest(algo={"state": "effective", "ordId": "55"},
                        order={"state": "filled", "accFillSz": "0.3", "avgPx": "100.5",
                               "ordId": "55", "fillTime": "1700000000000"}))
    st = b.entry_status(inst_id="BTC-USDT-SWAP", entry_type="stop", cl_ord_id="pa1e")
    assert st.state == "filled" and st.filled == 0.3 and st.fill_ms == 1700000000000


def test_entry_payloads():
    rest = _Rest()
    b = OkxBroker(rest)
    b.place_entry(inst_id="BTC-USDT-SWAP", side="buy", entry_type="limit", size=0.05,
                  price=100.04, stop_loss=98.06, margin_mode="isolated",
                  cl_ord_id="pa1e", sl_cl_ord_id="pa1s")
    kind, body = rest.posts[-1]
    assert kind == "order" and body["px"] == "100" and body["sz"] == "0.05"
    assert body["attachAlgoOrds"][0] == {"attachAlgoClOrdId": "pa1s", "slTriggerPx": "98.1",
                                         "slOrdPx": "-1", "slTriggerPxType": "last"}
    b.place_entry(inst_id="BTC-USDT-SWAP", side="sell", entry_type="stop", size=1,
                  price=99.0, stop_loss=101.0, margin_mode="cross",
                  cl_ord_id="pa2e", sl_cl_ord_id="pa2s")
    kind, body = rest.posts[-1]
    assert kind == "algo" and body["ordType"] == "trigger" and body["triggerPx"] == "99"
    assert body["algoClOrdId"] == "pa2e" and body["orderPx"] == "-1"
    b.place_stop(inst_id="BTC-USDT-SWAP", side="sell", size=0.05, trigger_px=99.0,
                 margin_mode="isolated", sl_cl_ord_id="pa1s2")
    kind, body = rest.posts[-1]
    assert body["ordType"] == "conditional" and body["reduceOnly"] is True
    assert body["slTriggerPx"] == "99" and body["slOrdPx"] == "-1"


def test_account_equity_is_usdt_balance():
    b = OkxBroker(_Rest(balance={"totalEq": "87347.6", "upl": "", "details": [
        {"ccy": "USDT", "eq": "87477.1", "availEq": "50263.5", "upl": "12.5"}]}))
    a = b.account()
    assert (a.total_eq, a.avail_eq, a.upl, a.ccy) == (87477.1, 50263.5, 12.5, "USDT")
    a = OkxBroker(_Rest(balance={"totalEq": "1000", "details": []})).account()
    assert (a.total_eq, a.ccy) == (1000.0, "USD")  # no USDT balance: fall back to totalEq
