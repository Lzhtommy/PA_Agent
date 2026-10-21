"""OKX REST signing, candle parsing, symbol normalisation and signal mapping."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest

from pa_agent.data.factory import create_data_source, default_symbol_for_kind
from pa_agent.data.okx_source import (
    OkxSource,
    fetch_okx_bars,
    normalize_okx_inst_id,
    parse_okx_candles,
)
from pa_agent.exchange.okx_rest import OkxCredentials, OkxError, OkxRestClient, sign_request
from pa_agent.trading.signals import build_signal


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BTC", "BTC-USDT-SWAP"),
        ("btcusdt", "BTC-USDT-SWAP"),
        ("BTC/USDT", "BTC-USDT-SWAP"),
        ("eth-usdt", "ETH-USDT-SWAP"),
        ("ETH-USDT-SWAP", "ETH-USDT-SWAP"),
        ("BTCUSDT.P", "BTC-USDT-SWAP"),
        ("BTC-USD-SWAP", "BTC-USD-SWAP"),
        ("BTC-USDT-SPOT", "BTC-USDT"),
        ("", ""),
    ],
)
def test_normalize_okx_inst_id(raw, expected):
    assert normalize_okx_inst_id(raw) == expected


def test_factory_creates_okx():
    assert isinstance(create_data_source("okx"), OkxSource)
    assert default_symbol_for_kind("okx") == "BTC-USDT-SWAP"


def _row(ts, close, confirm="1"):
    return [str(ts), "1", str(close + 1), str(close - 1), str(close), "10", "0.1", "1000", confirm]


def test_parse_candles_marks_forming_bar():
    bars = parse_okx_candles([_row(3000, 103, "0"), _row(2000, 102), _row(1000, 101)])
    assert [b.seq for b in bars] == [1, 2, 3]
    assert bars[0].closed is False and bars[1].closed is True
    assert bars[0].amount == 1000.0


class _FakeRest:
    def __init__(self, total):
        self.rows = [_row(1_000_000 - i * 1000, 100 + i) for i in range(total)]
        self.calls = []

    def _page(self, limit, after):
        start = 0
        if after is not None:
            start = next(i for i, r in enumerate(self.rows) if r[0] == after) + 1
        return self.rows[start:start + limit]

    def candles(self, inst, bar, *, limit, after=None):
        self.calls.append(("candles", limit, after))
        # OKX `candles` only serves the most recent 1440 bars
        return self._page(limit, after) if (after is None or int(after) > 1_000_000 - 1440 * 1000) else []

    def history_candles(self, inst, bar, *, limit, after=None):
        self.calls.append(("history", limit, after))
        return self._page(min(limit, 100), after)


def test_fetch_paginates_without_duplicates():
    rest = _FakeRest(800)
    bars = fetch_okx_bars(rest, "BTC-USDT-SWAP", "1h", 650)
    assert len(bars) == 650
    assert len({b.ts_open for b in bars}) == 650
    assert all(a.ts_open > b.ts_open for a, b in zip(bars, bars[1:], strict=False))


def test_sign_request_matches_okx_spec():
    ts, secret = "2020-12-08T09:08:57.715Z", "22582BD0CFF14C41EDBF1AB98506286D"
    body = json.dumps({"instId": "BTC-USDT"})
    expected = base64.b64encode(
        hmac.new(secret.encode(), f"{ts}POST/api/v5/trade/order{body}".encode(), hashlib.sha256).digest()
    ).decode()
    assert sign_request(secret, ts, "post", "/api/v5/trade/order", body) == expected


class _Resp:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._p


class _Http:
    def __init__(self, payload):
        self.payload = payload
        self.last = None

    def request(self, method, url, content=None, headers=None):
        self.last = (method, url, content, headers)
        return _Resp(self.payload)


def test_private_request_headers_and_demo_flag():
    http = _Http({"code": "0", "data": [{"ordId": "1"}]})
    c = OkxRestClient(OkxCredentials("k", "s", "p"), demo=True, http_client=http)
    assert c.place_order({"instId": "BTC-USDT-SWAP"}) == {"ordId": "1"}
    method, url, content, headers = http.last
    assert method == "POST" and url.endswith("/api/v5/trade/order")
    assert headers["x-simulated-trading"] == "1"
    assert headers["OK-ACCESS-KEY"] == "k" and headers["OK-ACCESS-PASSPHRASE"] == "p"
    assert json.loads(content) == {"instId": "BTC-USDT-SWAP"}


def test_error_surfaces_scode_message():
    http = _Http({"code": "1", "msg": "", "data": [{"sCode": "51008", "sMsg": "Insufficient balance"}]})
    c = OkxRestClient(OkxCredentials("k", "s", "p"), http_client=http, max_retries=0)
    with pytest.raises(OkxError, match="51008"):
        c.place_order({})


def test_private_requires_credentials():
    c = OkxRestClient(None, http_client=_Http({"code": "0", "data": []}))
    with pytest.raises(OkxError, match="未配置"):
        c.balance()


def test_build_signal_maps_chinese_fields():
    sig = build_signal(
        decision_inner={
            "order_direction": "做空", "order_type": "突破单", "entry_price": "82000",
            "stop_loss_price": 82500, "take_profit_price": "81000-81200",
            "take_profit_price_2": "", "trade_confidence": "55", "estimated_win_rate": "48%",
        },
        stage2_full={"diagnosis_summary": {"x": 1}},
        stage1_diagnosis={"cycle_position": "trading_range"},
        symbol="BTC-USDT-SWAP", timeframe="1h", source="gui", analysis_key="123",
    )
    assert sig["direction"] == "short" and sig["order_type"] == "stop"
    assert sig["entry"] == 82000 and sig["stop_loss"] == 82500
    assert sig["tp1"] == pytest.approx(81100)  # range midpoint
    assert sig["tp2"] is None
    assert sig["trade_confidence"] == 55 and sig["estimated_win_rate"] == 48
    assert sig["signal_uid"] == "gui|BTC-USDT-SWAP|1h|123"


def test_build_signal_no_order():
    sig = build_signal(decision_inner={"order_type": "不下单", "order_direction": "做多"},
                       stage2_full={}, stage1_diagnosis=None, symbol="BTC-USDT-SWAP",
                       timeframe="1h", source="gui", analysis_key="1")
    assert sig["order_type"] == "none" and sig["direction"] == ""
