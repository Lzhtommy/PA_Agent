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


def test_dashboard_candles(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pa_agent.trading.dashboard import create_app

    class Market:
        calls = 0

        def candles(self, inst_id, bar, *, limit=300, after=None):
            Market.calls += 1
            assert (inst_id, bar) == ("BTC-USDT-SWAP", "1H")
            return [["7200000", "2", "3", "1", "2.5", "10"], ["3600000", "1", "2", "0.5", "1.5", "9"]]

        def history_candles(self, inst_id, bar, *, limit=100, after=None):
            return []

    store = TradeStore(tmp_path / "q.db")
    store.insert_trade(dict(inst_id="BTC-USDT-SWAP", timeframe="1h", direction="long",
                            entry_type="limit", state="pending", contracts=2, ct_val=0.01))
    store.insert_trade(dict(inst_id="ETH-USDT-SWAP", timeframe="1h", direction="long",
                            entry_type="limit", state="pending"))
    c = TestClient(create_app(store, demo=True, market=Market()))
    d = c.get("/api/candles?inst_id=BTC-USDT-SWAP&timeframe=1h&limit=50").json()
    assert [b["t"] for b in d["bars"]] == [3600000, 7200000]  # oldest first
    assert d["bars"][1]["c"] == 2.5
    assert [t["inst_id"] for t in d["trades"]] == ["BTC-USDT-SWAP"]
    c.get("/api/candles?inst_id=BTC-USDT-SWAP&timeframe=1h&limit=50")
    assert Market.calls == 1  # cached
    assert c.get("/api/candles?inst_id=BTC-USDT-SWAP&timeframe=7h").status_code == 400


def test_runner_applies_config_saved_from_dashboard(tmp_path, monkeypatch):
    from pa_agent.headless import RUNNER_ACTIVE_KEY, write_runner_config

    opened, closed = [], []

    class Src:
        def __init__(self):
            self.inst_id = ""

        def connect(self):
            pass

        def subscribe(self, sym, tf):
            if sym == "BAD-USDT-SWAP":
                raise ValueError("nope")
            self.inst_id = sym
            opened.append((sym, tf))

        def disconnect(self):
            closed.append(self.inst_id)

    monkeypatch.setattr("pa_agent.data.factory.create_data_source", lambda kind: Src())
    store = TradeStore(tmp_path / "q.db")
    ctx = SimpleNamespace(settings=SimpleNamespace(general=SimpleNamespace(analysis_bar_count=50)))
    r = HeadlessRunner(ctx=ctx, symbols=["BTC-USDT-SWAP"], timeframe="1h", store=store)
    btc = r.states["BTC-USDT-SWAP"]
    r.sync_config()  # nothing saved yet: keeps running as-is
    assert '"1h"' in store.get_kv(RUNNER_ACTIVE_KEY)

    write_runner_config(store, ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "BAD-USDT-SWAP"], "1h")
    r.sync_config()
    assert list(r.states) == ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]   # bad one skipped
    assert r.states["BTC-USDT-SWAP"] is btc                       # unchanged symbol keeps its state
    events_before = len(store.events(limit=100))
    r.sync_config()                                               # no re-apply / event spam
    assert len(store.events(limit=100)) == events_before

    write_runner_config(store, ["ETH-USDT-SWAP"], "15m")
    r.sync_config()
    assert list(r.states) == ["ETH-USDT-SWAP"] and r.timeframe == "15m"
    assert opened[-1] == ("ETH-USDT-SWAP", "15m")                 # timeframe change re-subscribes
    assert set(closed) == {"BTC-USDT-SWAP", "ETH-USDT-SWAP"}


def test_dashboard_runner_config(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pa_agent.exchange.okx_rest import OkxError
    from pa_agent.headless import read_runner_config
    from pa_agent.trading.dashboard import create_app

    class Market:
        def instrument(self, inst_id, inst_type="SWAP"):
            if inst_id == "NOPE-USDT-SWAP":
                raise OkxError("empty", "unknown")
            return {"instId": inst_id}

    store = TradeStore(tmp_path / "q.db")
    c = TestClient(create_app(store, demo=True, market=Market()))
    r = c.get("/api/runner").json()
    assert r["config"] is None and r["running"] is False and "1h" in r["timeframes"]
    r = c.post("/api/runner", json={"symbols": "BTC-USDT-SWAP， eth, BTC-USDT-SWAP", "timeframe": "4h"}).json()
    assert r["config"] == {"symbols": ["BTC-USDT-SWAP", "ETH-USDT-SWAP"], "timeframe": "4h"}
    assert read_runner_config(store) == (["BTC-USDT-SWAP", "ETH-USDT-SWAP"], "4h")
    assert c.post("/api/runner", json={"symbols": "BTC", "timeframe": "7h"}).status_code == 400
    assert c.post("/api/runner", json={"symbols": "NOPE", "timeframe": "1h"}).status_code == 400
    assert c.post("/api/runner", json={"symbols": "", "timeframe": "1h"}).status_code == 400
    assert read_runner_config(store)[1] == "4h"  # rejected saves change nothing


def test_equity_reads_skip_legacy_usd_rows(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)  # equity table as created before the ccy column existed
    con.execute("CREATE TABLE equity (ts_ms INTEGER PRIMARY KEY, total_eq REAL NOT NULL, "
                "avail_eq REAL, upl REAL, demo INTEGER NOT NULL DEFAULT 1)")
    con.execute("INSERT INTO equity VALUES (1, 87300, 50000, 0, 1)")
    con.commit()
    con.close()
    store = TradeStore(db)
    assert store.equity_series() == [] and store.latest_equity() is None
    store.record_equity(87477, 50263, 0, demo=True)
    time.sleep(0.005)  # ts_ms is the primary key
    store.record_equity(1000, 1000, 0, demo=True, ccy="USD")
    rows = store.equity_series(demo=True)
    assert [(r["total_eq"], r["ccy"]) for r in rows] == [(87477, "USDT")]
    assert store.latest_equity()["total_eq"] == 87477
    assert store.first_equity_since(0)["total_eq"] == 87477


def test_dashboard_pagination_and_instrument_filter(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pa_agent.trading.dashboard import create_app

    store = TradeStore(tmp_path / "q.db")
    for i in range(7):
        inst = "ETH-USDT-SWAP" if i % 3 == 0 else "BTC-USDT-SWAP"
        store.insert_signal(dict(signal_uid=f"u{i}", source="headless", symbol=inst, timeframe="1h",
                                 direction="long", order_type="none"))
        store.insert_trade(dict(inst_id=inst, timeframe="1h", direction="long",
                                entry_type="limit", state="canceled"))
        store.log_event("note", f"e{i}")
    c = TestClient(create_app(store, demo=True))

    r = c.get("/api/trades?limit=3&offset=3")
    assert r.headers["x-total-count"] == "7"
    assert [t["id"] for t in r.json()] == [4, 3, 2]  # newest first, second page
    r = c.get("/api/trades?limit=10&inst_id=ETH-USDT-SWAP")
    assert r.headers["x-total-count"] == "3" and {t["inst_id"] for t in r.json()} == {"ETH-USDT-SWAP"}

    r = c.get("/api/signals?limit=2&offset=0&symbol=BTC-USDT-SWAP")
    assert r.headers["x-total-count"] == "4" and len(r.json()) == 2

    r = c.get("/api/events?limit=2&offset=5")
    assert r.headers["x-total-count"] == "7"
    assert [e["message"] for e in r.json()] == ["e1", "e0"]

    assert sorted(c.get("/api/instruments").json()) == ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]


def _versioned_store(tmp_path):
    store = TradeStore(tmp_path / "v.db")
    now = int(time.time() * 1000)
    plan = [("deepseek-flash", "legacy", 150.0), ("claude_cli/sonnet", "abcd1234", -120.0),
            ("claude_cli/sonnet", "abcd1234", -100.0)]
    for i, (model, phash, pnl) in enumerate(plan):
        sid = store.insert_signal(dict(signal_uid=f"o{i}", symbol="BTC-USDT-SWAP", timeframe="1h",
                                       direction="long", order_type="limit", entry=100, stop_loss=98,
                                       tp1=102, model=model, prompt_hash=phash, created_ms=now + i))
        store.mark_signal(sid, "processed", f"交易 #{i + 1}")
        store.insert_trade(dict(signal_id=sid, inst_id="BTC-USDT-SWAP", timeframe="1h",
                                direction="long", entry_type="limit", state="closed",
                                risk_amount=100, net_pnl=pnl, equity_at_entry=10_000,
                                closed_ms=now + i))
    store.insert_signal(dict(signal_uid="n1", symbol="BTC-USDT-SWAP", timeframe="1h",
                             order_type="none", model="claude_cli/sonnet", prompt_hash="abcd1234"))
    return store


def test_dashboard_filters_by_strategy_version(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pa_agent.trading.dashboard import create_app

    c = TestClient(create_app(_versioned_store(tmp_path), demo=True))
    versions = {v["label"]: v for v in c.get("/api/versions").json()}
    sonnet = versions["claude_cli/sonnet · abcd1234 · 1h"]
    assert sonnet["closed"] == 2 and sonnet["signals"] == 3 and sonnet["orders"] == 2
    assert versions["deepseek-flash · 旧版本 · 1h"]["closed"] == 1

    v = "version=" + sonnet["key"]
    s = c.get(f"/api/overview?{v}").json()["summary"]
    assert s["trades"] == 2 and s["total_r"] == pytest.approx(-2.2)
    assert c.get(f"/api/trades?{v}").headers["x-total-count"] == "2"
    assert c.get(f"/api/signals?{v}").headers["x-total-count"] == "3"
    assert c.get(f"/api/signals?{v}&actionable=true").headers["x-total-count"] == "2"
    q = c.get(f"/api/ai-quality?{v}").json()
    assert [g["group"] for g in q["by_version"]] == ["claude_cli/sonnet · abcd1234 · 1h"]
    assert q["signals_total"] == 3 and q["signals_actionable"] == 2
    assert c.get("/api/overview").json()["summary"]["trades"] == 3


def test_summary_drawdown_uses_realised_pnl():
    from pa_agent.trading.stats import summarize

    trades = [dict(state="closed", net_pnl=p, risk_amount=100, equity_at_entry=10_000, closed_ms=i)
              for i, p in enumerate([200.0, -150.0, -100.0, 50.0])]
    s = summarize(trades)
    assert s["max_drawdown"] == pytest.approx(250.0)          # peak +200 → trough -50
    assert s["max_drawdown_pct"] == pytest.approx(250 / 10_200 * 100)
    assert s["max_drawdown_r"] == pytest.approx(2.5)
    assert s["return_pct"] == pytest.approx(0.0)


def test_timeframe_is_part_of_strategy_version(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pa_agent.headless import write_runner_config
    from pa_agent.trading.dashboard import create_app
    from pa_agent.trading.signals import prompt_hash

    store = _versioned_store(tmp_path)
    phash = prompt_hash()
    for tf in ("1h", "4h"):
        store.insert_signal(dict(signal_uid=f"tf{tf}", symbol="BTC-USDT-SWAP", timeframe=tf,
                                 order_type="none", model="claude_cli/sonnet", prompt_hash=phash))
    write_runner_config(store, ["BTC-USDT-SWAP"], "4h")
    c = TestClient(create_app(store, demo=True))
    versions = {v["label"]: v for v in c.get("/api/versions").json()}
    assert versions[f"claude_cli/sonnet · {phash} · 4h"]["current"] is True
    assert versions[f"claude_cli/sonnet · {phash} · 1h"]["current"] is False
    key_4h = versions[f"claude_cli/sonnet · {phash} · 4h"]["key"]
    assert c.get(f"/api/signals?version={key_4h}").headers["x-total-count"] == "1"


def _store_runner(tmp_path, bars, *, trading=True):
    r = _runner(bars)
    r.store = TradeStore(tmp_path / "r.db")
    r.ctx.settings.trading = SimpleNamespace(enabled=trading)
    return r


@pytest.mark.parametrize("minutes_after_close,expected", [(10, 1), (40, 0)])
def test_restart_catches_up_on_recent_unanalysed_bar(tmp_path, monkeypatch, minutes_after_close, expected):
    real = int(time.time() * 1000)
    head = real - real % H - 2 * H
    monkeypatch.setattr(time, "time", lambda: (head + minutes_after_close * 60_000) / 1000)
    r = _store_runner(tmp_path, _bars(head))
    # the bar that closed at `head` has no signal; catch up only within half a bar
    assert r.poll_once() == expected
    assert r.analyzed == [head - H] * expected
    assert r.poll_once() == 0


def test_restart_waits_when_bar_already_analysed(tmp_path):
    now = int(time.time() * 1000)
    head = now - now % H
    r = _store_runner(tmp_path, _bars(head))
    r.store.insert_signal(dict(signal_uid="done", symbol="BTC-USDT-SWAP", timeframe="1h",
                               order_type="none", created_ms=head + 60_000))
    assert r.poll_once() == 0 and r.analyzed == []


def test_no_catch_up_without_signals(tmp_path):
    now = int(time.time() * 1000)
    r = _store_runner(tmp_path, _bars(now - now % H), trading=False)
    assert r.poll_once() == 0


def test_runner_reports_next_analysis_time(tmp_path):
    import json

    from pa_agent.headless import RUNNER_ACTIVE_KEY

    now = int(time.time() * 1000)
    head = now - now % H
    r = _store_runner(tmp_path, _bars(head), trading=False)
    r.poll_once()
    r._report()
    active = json.loads(r.store.get_kv(RUNNER_ACTIVE_KEY))
    assert active["next_ms"] == head + H   # close of the bar forming now
    assert active["analyzing"] == ""


def test_dashboard_cancel_endpoint(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pa_agent.trading.dashboard import create_app

    store = TradeStore(tmp_path / "c.db")
    pending = store.insert_trade(dict(inst_id="BTC-USDT-SWAP", timeframe="4h", direction="short",
                                      entry_type="limit", state="pending"))
    closed = store.insert_trade(dict(inst_id="BTC-USDT-SWAP", timeframe="4h", direction="short",
                                     entry_type="limit", state="closed"))
    c = TestClient(create_app(store, demo=True))
    assert c.post(f"/api/trades/{pending}/cancel").json() == {"requested": True}
    assert store.cancel_requests() == [pending]
    assert c.get("/api/overview").json()["active"][0]["cancel_requested"] is True
    assert c.post(f"/api/trades/{closed}/cancel").status_code == 400
    assert c.post("/api/trades/999/cancel").status_code == 404
