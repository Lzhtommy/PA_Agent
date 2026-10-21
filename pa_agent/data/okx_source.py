"""OKX public market-data source (perpetual swaps by default).

Same price feed the trading executor uses, so stop / entry levels the AI derives
from the chart match the instrument that is actually traded.
"""
from __future__ import annotations

import logging
import re

from pa_agent.data.base import (
    DataSource,
    DataSourceTransientError,
    KlineBar,
    normalize_kline_bar,
)

logger = logging.getLogger(__name__)

OKX_DEFAULT_SYMBOL = "BTC-USDT-SWAP"

# Our timeframe strings → OKX ``bar`` values (daily+ use UTC-aligned bars).
_TF_MAP: dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "1d": "1Dutc",
    "1w": "1Wutc",
}

_POPULAR_SWAPS: tuple[str, ...] = (
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "XRP-USDT-SWAP",
    "DOGE-USDT-SWAP",
    "BNB-USDT-SWAP",
    "LTC-USDT-SWAP",
    "ADA-USDT-SWAP",
)

_QUOTES = ("USDT", "USDC", "USD")
_INST_RE = re.compile(r"^[A-Z0-9]+-[A-Z0-9]+(-SWAP|-\d{6})?$")


def normalize_okx_inst_id(symbol: str) -> str:
    """Map user input to an OKX instId.

    ``BTC`` / ``BTCUSDT`` / ``btc-usdt`` / ``BTC/USDT`` → ``BTC-USDT-SWAP``;
    explicit spot (``BTC-USDT``) is kept only when written with a dash *and*
    suffixed ``-SPOT``; full instIds (``ETH-USDT-SWAP``) pass through.
    """
    s = (symbol or "").strip().upper().replace("/", "-").replace("_", "-")
    if not s:
        return ""
    s = s.replace(".P", "").replace("PERP", "")
    if s.endswith("-SPOT"):
        return s[: -len("-SPOT")]
    if _INST_RE.match(s) and (s.endswith("-SWAP") or re.search(r"-\d{6}$", s)):
        return s
    parts = [p for p in s.split("-") if p]
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}-SWAP"
    compact = parts[0] if parts else s
    for q in _QUOTES:
        if compact.endswith(q) and len(compact) > len(q):
            return f"{compact[: -len(q)]}-{q}-SWAP"
    return f"{compact}-USDT-SWAP"


def okx_bar_for_timeframe(timeframe: str) -> str:
    try:
        return _TF_MAP[timeframe]
    except KeyError:
        raise ValueError(
            f"Unsupported timeframe: {timeframe!r}. Use one of {list(_TF_MAP)}"
        ) from None


def parse_okx_candles(rows: list[list[str]]) -> list[KlineBar]:
    """OKX rows (newest first) → KlineBar list (newest first, seq 1..N)."""
    bars: list[KlineBar] = []
    for i, row in enumerate(rows):
        if len(row) < 6:
            continue
        confirm = str(row[8]) if len(row) > 8 else "1"
        quote_vol = float(row[7]) if len(row) > 7 and row[7] not in ("", None) else 0.0
        bars.append(
            normalize_kline_bar(
                KlineBar(
                    seq=i + 1,
                    ts_open=float(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    amount=quote_vol,
                    closed=confirm == "1",
                )
            )
        )
    return bars


class OkxSource(DataSource):
    """K-lines from OKX public REST (no API key needed)."""

    def __init__(self, client=None) -> None:
        self._client = client
        self._symbol = ""
        self._timeframe = ""
        self._connected = False

    def _rest(self):
        if self._client is None:
            from pa_agent.exchange.okx_rest import OkxRestClient

            # Market data is identical on demo/live; public calls need no header.
            self._client = OkxRestClient(demo=False)
        return self._client

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._rest()
        self._connected = True
        logger.info("OkxSource connected")

    def disconnect(self) -> None:
        self._connected = False
        logger.info("OkxSource disconnected")

    # ── Discovery ─────────────────────────────────────────────────────────────

    def list_symbols(self) -> list[str]:
        return list(_POPULAR_SWAPS)

    def supported_timeframes(self) -> list[str]:
        return list(_TF_MAP.keys())

    # ── Subscription ──────────────────────────────────────────────────────────

    def subscribe(self, symbol: str, timeframe: str) -> None:
        okx_bar_for_timeframe(timeframe)
        inst = normalize_okx_inst_id(symbol)
        if not inst:
            raise ValueError("OKX 品种不能为空，例如 BTC-USDT-SWAP")
        self._symbol = inst
        self._timeframe = timeframe
        logger.info("OkxSource subscribed: %s %s", inst, timeframe)

    def unsubscribe(self) -> None:
        self._symbol = ""
        self._timeframe = ""

    @property
    def inst_id(self) -> str:
        return self._symbol

    # ── Data fetch ────────────────────────────────────────────────────────────

    def latest_snapshot(self, n: int) -> list[KlineBar]:
        """Return *n* bars newest-first; bars[0] is the forming bar when unconfirmed."""
        if not self._connected:
            raise DataSourceTransientError("Not connected — call connect() first")
        if not self._symbol or not self._timeframe:
            raise DataSourceTransientError("Not subscribed — call subscribe() first")
        return fetch_okx_bars(self._rest(), self._symbol, self._timeframe, n)


def fetch_okx_bars(client, inst_id: str, timeframe: str, n: int) -> list[KlineBar]:
    """Fetch *n* newest bars, paginating ``candles`` → ``history-candles``."""
    from pa_agent.exchange.okx_rest import OkxError

    bar = okx_bar_for_timeframe(timeframe)
    want = max(1, int(n))
    rows: list[list[str]] = []
    after: str | None = None
    use_history = False
    try:
        while len(rows) < want:
            requested = min(300, want - len(rows))
            page = (
                client.history_candles(inst_id, bar, limit=100, after=after)
                if use_history
                else client.candles(inst_id, bar, limit=requested, after=after)
            )
            if not page:
                if use_history:
                    break
                use_history = True
                continue
            rows.extend(page)
            after = str(page[-1][0])
            if not use_history and len(page) < requested:
                use_history = True
    except OkxError as exc:
        if not rows:
            raise DataSourceTransientError(f"OKX K线获取失败: {exc}") from exc
        logger.warning("OKX pagination stopped early (%d bars): %s", len(rows), exc)

    # De-duplicate by ts (page boundaries) while keeping newest-first order.
    seen: set[str] = set()
    unique: list[list[str]] = []
    for row in rows:
        if row[0] in seen:
            continue
        seen.add(row[0])
        unique.append(row)
    if not unique:
        raise DataSourceTransientError(f"OKX 未返回 {inst_id} {timeframe} 的K线")
    return parse_okx_candles(unique[:want])
