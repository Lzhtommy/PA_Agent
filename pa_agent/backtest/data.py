"""Historical OKX candles for backtests, cached as CSV.

Bars are kept oldest-first (the natural order for a simulation), unlike the live
analysis code which works newest-first.
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FIELDS = ("ts", "open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class Bar:
    ts: int        # bar open time, epoch ms
    open: float
    high: float
    low: float
    close: float
    volume: float


def _parse_rows(rows: list[list[str]]) -> list[Bar]:
    """OKX candle rows → closed bars (rows of a still-forming bar are dropped)."""
    out = []
    for r in rows:
        if len(r) > 8 and str(r[8]) != "1":
            continue
        out.append(Bar(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
    return out


def _with_retry(call: Any, attempts: int = 6) -> Any:
    """Long downloads hit transient network errors; back off and retry them."""
    from pa_agent.exchange.okx_rest import OkxNetworkError

    for n in range(attempts):
        try:
            return call()
        except OkxNetworkError as exc:
            if n == attempts - 1:
                raise
            logger.warning("OKX 网络错误，%ds 后重试: %s", 2 ** n, exc)
            time.sleep(2 ** n)
    return None


def fetch_history(client: Any, inst_id: str, bar: str, start_ms: int, *,
                  before_ms: int | None = None, pause_s: float = 0.12) -> list[Bar]:
    """Closed bars with ``start_ms <= ts < before_ms`` (default: up to now), oldest first.

    Pages backwards through ``/market/history-candles`` (100 rows per call; OKX allows
    20 calls per 2 s, hence ``pause_s``).
    """
    got: dict[int, Bar] = {}
    cursor = str(before_ms) if before_ms else None
    while True:
        rows = _with_retry(lambda c=cursor: client.history_candles(inst_id, bar, limit=100, after=c))
        if not rows:
            break
        page = _parse_rows(rows)
        for b in page:
            if b.ts >= start_ms and (before_ms is None or b.ts < before_ms):
                got[b.ts] = b
        oldest = int(rows[-1][0])
        if oldest <= start_ms or str(oldest) == cursor:
            break
        cursor = str(oldest)
        time.sleep(pause_s)
    return [got[k] for k in sorted(got)]


def read_csv(path: Path) -> list[Bar]:
    with path.open(newline="") as f:
        return [Bar(int(r["ts"]), float(r["open"]), float(r["high"]), float(r["low"]),
                    float(r["close"]), float(r["volume"])) for r in csv.DictReader(f)]


def write_csv(path: Path, bars: list[Bar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_FIELDS)
        for b in bars:
            w.writerow((b.ts, b.open, b.high, b.low, b.close, b.volume))
    tmp.replace(path)


def load_bars(inst_id: str, bar: str, start_ms: int, cache_dir: Path, *,
              client: Any = None, refresh: bool = True) -> list[Bar]:
    """Bars from ``start_ms`` to now, oldest first, using ``cache_dir/<inst>_<bar>.csv``.

    Only the missing head (older than the cache) and tail (newer) are downloaded.
    """
    path = Path(cache_dir) / f"{inst_id}_{bar}.csv"
    cached = read_csv(path) if path.exists() else []
    if client is None and refresh:
        from pa_agent.exchange.okx_rest import OkxRestClient

        client = OkxRestClient(demo=False)
    bars = {b.ts: b for b in cached}
    if refresh:
        if not cached or cached[0].ts > start_ms:
            head_end = cached[0].ts if cached else None
            for b in fetch_history(client, inst_id, bar, start_ms, before_ms=head_end):
                bars[b.ts] = b
        if cached:
            for b in fetch_history(client, inst_id, bar, cached[-1].ts):
                bars[b.ts] = b
        merged = [bars[k] for k in sorted(bars)]
        if len(merged) != len(cached):
            write_csv(path, merged)
            logger.info("%s %s: %d bars cached (%+d)", inst_id, bar, len(merged), len(merged) - len(cached))
    return [b for b in (bars[k] for k in sorted(bars)) if b.ts >= start_ms]


def find_gaps(bars: list[Bar], step_ms: int) -> list[tuple[int, int]]:
    """``(ts_before, ts_after)`` pairs where consecutive bars are more than one step apart."""
    return [(a.ts, b.ts) for a, b in zip(bars, bars[1:], strict=False) if b.ts - a.ts != step_ms]
