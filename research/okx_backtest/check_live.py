"""Cross-check the simulator against closed live (demo) trades.

    uv run python research/okx_backtest/check_live.py <quant.db copy> [--demo]

``--demo`` replays on OKX demo-trading candles (where the demo orders actually filled)
instead of the live market's; the two differ by up to a few hundred dollars on BTC.

Replays each closed trade's original signal (1H bars, 5m sub-bars) with the settings
that were live at the time (no fee buffer, 0.05% minimum stop) and compares the
outcome and R with what OKX actually booked.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from pa_agent.backtest.data import load_bars
from pa_agent.backtest.sim import Order, PriceUnits, SimConfig, simulate
from pa_agent.trading.risk import RiskConfig

HERE = Path(__file__).parent
H1 = 3_600_000
LIVE_THEN = SimConfig(fee_buffer_pct=0.0, slippage_pct=0.0,
                      risk=RiskConfig(min_reward_risk=1.0, min_stop_distance_pct=0.05))
LIVE_REASON = {"止盈2": "tp2", "止损": "stop", "止盈1后保本离场": "tp1_be", "止盈1后止损": "tp1_stop", "止盈": "tp1"}


def main(db_path: str, demo: bool = False) -> None:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT t.id, t.inst_id, t.net_pnl, t.risk_amount, t.close_reason, s.direction, s.order_type, "
        "s.entry, s.stop_loss, s.tp1, s.tp2, s.created_ms FROM trades t JOIN signals s ON s.id = t.signal_id "
        "WHERE t.state = 'closed' AND t.net_pnl IS NOT NULL ORDER BY t.id").fetchall()
    start = min(r["created_ms"] for r in rows) - 10 * H1
    cache: dict[str, PriceUnits] = {}
    print(f"{'#':>3} {'实盘':<8} {'模拟':<8} {'实盘R':>7} {'模拟R':>7}")
    for r in rows:
        inst = r["inst_id"]
        if inst not in cache:
            from pa_agent.exchange.okx_rest import OkxRestClient

            cache_dir = HERE / ("data_demo" if demo else "data_live")
            client = OkxRestClient(demo=demo)
            b1 = load_bars(inst, "1H", start, cache_dir, client=client)
            m5 = load_bars(inst, "5m", start, cache_dir, client=client)
            cache[inst] = PriceUnits(b1, m5, step_ms=H1)
        prices = cache[inst]
        last_closed = r["created_ms"] // H1 * H1 - H1
        idx = next(i for i, b in enumerate(prices.bars) if b.ts == last_closed)
        order = Order(idx, r["direction"], r["order_type"], r["entry"], r["stop_loss"], r["tp1"], r["tp2"])
        res = simulate(order, prices, LIVE_THEN)
        live_r = r["net_pnl"] / r["risk_amount"]
        sim_r = f"{res.r:+.2f}" if res.r is not None else "—"
        match = "✓" if LIVE_REASON.get(r["close_reason"]) == res.outcome else "✗"
        print(f"{r['id']:>3} {r['close_reason']:<8} {res.outcome:<8} {live_r:>+7.2f} {sim_r:>7} {match}")


if __name__ == "__main__":
    main(sys.argv[1], demo="--demo" in sys.argv[2:])
