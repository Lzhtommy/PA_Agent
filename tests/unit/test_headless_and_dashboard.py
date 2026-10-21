"""Headless runner bar-close detection and dashboard API."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from pa_agent.data.base import KlineBar
from pa_agent.headless import HeadlessRunner, SymbolState, is_order_decision, newest_closed_ts
from pa_agent.trading.store import TradeStore

H = 3_600_000


def _bars(newest_open_ms: int, n: int = 5, forming: bool = True) -> list[KlineBar]:
    return [
        KlineBar(seq=i + 1, ts_open=newest_open_ms - i * H, open=1, high=2, low=0.5, close=1.5,
                 volume=1, closed=not (forming and i == 0))
        for i in range(n)
    ]


def test_is_order_decision():
    assert is_order_decision({"order_type": "限价单"})
    assert not is_order_decision({"order_type": "不下单"})
    assert not is_order_decision({"order_type": "市价单", "trade_confidence": 40}, 50)
    assert is_order_decision({"order_type": "市价单", "trade_confidence": "60"}, 50)


def test_newest_closed_ts_skips_forming_bar():
    now = int(time.time() * 1000)
    head = now - now % H  # current hour bar still forming
    assert newest_closed_ts(_bars(head), "1h", "BTC-USDT-SWAP", now) == head - H


class _Src:
    def __init__(self, bars):
        self.bars = bars

    def latest_snapshot(self, n):
        return self.bars


def _runner(bars, run_now=False):
    r = HeadlessRunner.__new__(HeadlessRunner)
    r.ctx = SimpleNamespace(settings=SimpleNamespace(general=SimpleNamespace(analysis_bar_count=50)))
    r.symbols, r.timeframe, r.source_kind = ["BTC-USDT-SWAP"], "1h", "okx"
    r.bar_count, r.run_now, r.poll_s, r._cancel = 50, run_now, 1, None
    r.states = {"BTC-USDT-SWAP": SymbolState("BTC-USDT-SWAP", _Src(bars))}
    r.analyzed = []
    r.analyze = lambda st, b: r.analyzed.append(newest_closed_ts(b, "1h", st.symbol, int(time.time() * 1000)))
    return r


def test_poll_waits_for_next_close_then_analyses_once_per_bar():
    now = int(time.time() * 1000)
    head = now - now % H
    r = _runner(_bars(head))
    assert r.poll_once() == 0          # first sighting: arm, don't analyse
    assert r.poll_once() == 0          # same closed bar
    r.states["BTC-USDT-SWAP"].source.bars = _bars(head + H)  # simulate next hour
    r.states["BTC-USDT-SWAP"].last_closed_ts = head - H
    assert r.poll_once() == 1
    assert r.analyzed == [head]
    assert r.poll_once() == 0


def test_run_now_analyses_immediately():
    now = int(time.time() * 1000)
    r = _runner(_bars(now - now % H), run_now=True)
    assert r.poll_once() == 1


# ── dashboard ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pa_agent.trading.dashboard import create_app

    store = TradeStore(tmp_path / "q.db")
    sid = store.insert_signal(dict(signal_uid="u1", source="gui", symbol="BTC-USDT-SWAP",
                                   timeframe="1h", direction="long", order_type="limit",
                                   entry=100, stop_loss=98, tp1=104, trade_confidence=62,
                                   decision_json={"decision": {"reasoning": "回踩支撑"}},
                                   stage1_json={"cycle_position": "trading_range"}))
    store.mark_signal(sid, "processed", "交易 #1")
    store.insert_trade(dict(signal_id=sid, inst_id="BTC-USDT-SWAP", timeframe="1h",
                            direction="long", entry_type="limit", state="closed",
                            risk_amount=100, net_pnl=150, closed_ms=int(time.time() * 1000)))
    store.record_equity(10_000, 9_000, 0, demo=True)
    app = create_app(store, demo=True, token="secret", chart_root=tmp_path)
    return TestClient(app), store


def test_dashboard_requires_token(client):
    c, _ = client
    assert c.get("/api/overview").status_code == 401
    assert c.get("/api/overview?token=wrong").status_code == 401
    assert c.get("/").status_code == 200  # page itself is public, data is not


def test_dashboard_overview_and_detail(client):
    c, _ = client
    h = {"Authorization": "Bearer secret"}
    o = c.get("/api/overview", headers=h).json()
    assert o["demo"] is True
    assert o["summary"]["trades"] == 1 and o["summary"]["win_rate"] == 100
    trades = c.get("/api/trades", headers=h).json()
    assert trades[0]["trade_confidence"] == 62
    d = c.get(f"/api/trades/{trades[0]['id']}", headers=h).json()
    assert d["signal"]["decision"]["decision"]["reasoning"] == "回踩支撑"
    q = c.get("/api/ai-quality", headers=h).json()
    assert q["by_confidence"][0]["group"] == "60-69"
    assert c.get("/api/chart/1", headers=h).status_code == 404


def test_dashboard_kill_resume(client):
    c, store = client
    h = {"Authorization": "Bearer secret"}
    assert c.post("/api/kill", headers=h).json()["kill_switch"] is True
    assert store.get_kv("kill_switch") == "1"
    c.post("/api/resume", headers=h)
    assert store.get_kv("kill_switch") == "0"
