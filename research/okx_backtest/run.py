"""Run the pre-registered rule strategies (PREREG.md) and write REPORT_DATA.md + results.json.

    uv run python research/okx_backtest/run.py [--refresh] [--holdout]

Without ``--holdout`` only the development period (2020–2023) is evaluated.
"""
from __future__ import annotations

import argparse
import calendar
import json
import random
import statistics
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from pa_agent.backtest.data import load_bars
from pa_agent.backtest.sim import PriceUnits, SimConfig, run_portfolio, simulate
from pa_agent.backtest.strategies import (
    close_breakout_orders,
    prepare,
    pullback_orders,
    random_orders,
    range_fade_orders,
    shuffled_bars,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
INSTS = {"BTC-USDT-SWAP": 0.1, "ETH-USDT-SWAP": 0.01}
H4, H1 = 4 * 3_600_000, 3_600_000


def ms(y: int, m: int = 1, d: int = 1) -> int:
    return calendar.timegm((y, m, d, 0, 0, 0)) * 1000


PERIODS = {"dev": (ms(2020), ms(2024)), "holdout": (ms(2024), ms(2100))}

STRATEGIES = {
    "A 顺势回调": (pullback_orders, {
        "主变体 H1+H2 实盘出场": dict(triggers=(1, 2), exit="live"),
        "只做 H1": dict(triggers=(1,), exit="live"),
        "只做 H2": dict(triggers=(2,), exit="live"),
        "H1+H2 全部 1R": dict(triggers=(1, 2), exit="1r"),
        "H1+H2 全部 2R": dict(triggers=(1, 2), exit="2r"),
    }),
    "B 区间边缘反向": (range_fade_orders, {
        "主变体 结构目标": dict(exit="structure"),
        "1R/2R 目标": dict(exit="live"),
    }),
    "C 收盘确认突破": (close_breakout_orders, {
        "主变体 等距目标": dict(exit="mm"),
        "1R/2R 目标": dict(exit="live"),
    }),
}
PRIMARY = {name: next(iter(variants)) for name, (_, variants) in STRATEGIES.items()}


def summarize(pairs) -> dict:
    """Stats over ``(Result, bars)`` pairs (several instruments may be pooled)."""
    results = [r for r, _ in pairs]
    accepted = [r for r in results if r.outcome not in ("rejected", "skipped")]
    done = [(r, b) for r, b in pairs if r.filled and r.r is not None]
    rs = [r.r for r, _ in done]
    out = {"signals": len(results), "accepted": len(accepted), "trades": len(done),
           "outcomes": dict(Counter(r.outcome for r in results))}
    if not rs:
        return out
    rng = random.Random(7)
    means = sorted(statistics.fmean(rng.choices(rs, k=len(rs))) for _ in range(2000))
    peak = cum = mdd = 0.0
    for r, _ in sorted(done, key=lambda p: p[1][p[0].exit_idx].ts):
        cum += r.r
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    gains = sum(x for x in rs if x > 0)
    losses = -sum(x for x in rs if x <= 0)
    years: dict[str, list[float]] = {}
    sides: dict[str, list[float]] = {}
    for r, b in done:
        years.setdefault(time.strftime("%Y", time.gmtime(b[r.fill_idx].ts / 1000)), []).append(r.r)
        sides.setdefault(r.order.direction, []).append(r.r)
    out.update(
        fill_rate=len(done) / len(accepted) if accepted else None,
        win_rate=sum(x > 0 for x in rs) / len(rs),
        mean_r=statistics.fmean(rs), ci=(means[50], means[1949]), total_r=sum(rs),
        max_dd_r=mdd, profit_factor=gains / losses if losses else None,
        fee_r=statistics.fmean(r.fee_r for r, _ in done),
        ambiguous=sum(r.ambiguous for r, _ in done) / len(done),
        by_year={y: (len(v), statistics.fmean(v)) for y, v in sorted(years.items())},
        by_side={d: (len(v), statistics.fmean(v)) for d, v in sides.items()},
    )
    return out


def in_period(orders, bars, period):
    lo, hi = period
    return [o for o in orders if lo <= bars[o.signal_idx].ts < hi]


def index_range(bars, period):
    lo, hi = period
    idx = [i for i, b in enumerate(bars) if lo <= b.ts < hi]
    return idx[0], idx[-1] + 1


def evaluate(data, builder, kw, period, cfg, *, baselines: bool, reps_random=200, reps_shuffle=20):
    """Pooled BTC+ETH results for one variant in one period (+ baselines)."""
    pooled, pooled_nosub, templates = [], [], []
    for inst, d in data.items():
        orders = in_period(builder(d["series"], **kw), d["bars"], period)
        res = run_portfolio(orders, d["prices"], cfg)
        pooled += [(r, d["bars"]) for r in res]
        if baselines:
            pooled_nosub += run_portfolio(orders, d["prices_nosub"], cfg)
            templates.append((inst, [r.order for r in res if r.filled and r.r is not None]))
    summary = summarize(pooled)
    # sensitivity: same-unit ties resolved in favour of the trade (upper bound, not a verdict input)
    fav_cfg = replace(cfg, favorable_ties=True)
    fav = [r.r for inst, d in data.items()
           for r in run_portfolio(in_period(builder(d["series"], **kw), d["bars"], period), d["prices"], fav_cfg)
           if r.filled and r.r is not None]
    summary["mean_r_favorable"] = statistics.fmean(fav) if fav else None
    if not baselines:
        return summary
    # random entries with the same shape, independent trades, with sub-bars
    rng = random.Random(11)
    rand_means = []
    for _ in range(reps_random):
        rs = []
        for inst, tmpl in templates:
            d = data[inst]
            lo, hi = index_range(d["bars"], period)
            for o in random_orders(d["series"], tmpl, lo, hi - 1, rng):
                r = simulate(o, d["prices"], cfg)
                if r.filled and r.r is not None:
                    rs.append(r.r)
        if rs:
            rand_means.append(statistics.fmean(rs))
    rand_means.sort()
    # shuffled series, no sub-bars (compare with the real series without sub-bars)
    shuf_means = []
    rng = random.Random(13)
    for _ in range(reps_shuffle):
        rs = []
        for d in data.values():
            sb = shuffled_bars(d["bars"], rng)
            s = prepare(sb, d["series"].tick)
            orders = in_period(builder(s, **kw), sb, period)
            rs += [r.r for r in run_portfolio(orders, PriceUnits(sb, step_ms=H4), cfg)
                   if r.filled and r.r is not None]
        if rs:
            shuf_means.append(statistics.fmean(rs))
    real_nosub = [r.r for r in pooled_nosub if r.filled and r.r is not None]
    summary["random"] = {
        "mean": statistics.fmean(rand_means) if rand_means else None,
        "p95": rand_means[int(0.95 * (len(rand_means) - 1))] if rand_means else None,
        "pct_rank": (sum(m < summary.get("mean_r", -9) for m in rand_means) / len(rand_means)
                     if rand_means else None),
    }
    summary["shuffle"] = {
        "real_nosub_mean": statistics.fmean(real_nosub) if real_nosub else None,
        "mean": statistics.fmean(shuf_means) if shuf_means else None,
        "min": min(shuf_means) if shuf_means else None,
        "max": max(shuf_means) if shuf_means else None,
    }
    return summary


def load(refresh: bool):
    data = {}
    start = ms(2019, 12)
    for inst, tick in INSTS.items():
        b4 = load_bars(inst, "4H", start, DATA, refresh=refresh)
        b1 = load_bars(inst, "1H", start, DATA, refresh=refresh)
        data[inst] = {"bars": b4, "series": prepare(b4, tick),
                      "prices": PriceUnits(b4, b1, step_ms=H4),
                      "prices_nosub": PriceUnits(b4, step_ms=H4)}
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    data = load(args.refresh)
    cfg = SimConfig()
    out: dict = {"generated": time.strftime("%Y-%m-%d %H:%M"), "dev": {}, "holdout": {}}
    for name, (builder, variants) in STRATEGIES.items():
        for vname, kw in variants.items():
            t = time.time()
            out["dev"][f"{name} · {vname}"] = evaluate(data, builder, kw, PERIODS["dev"], cfg, baselines=True)
            print(f"dev  {name} · {vname}: {time.time() - t:.0f}s", flush=True)
    if args.holdout:
        for name, (builder, variants) in STRATEGIES.items():
            dev_best = max(variants, key=lambda v: out["dev"][f"{name} · {v}"].get("mean_r", -9))
            for vname in dict.fromkeys([PRIMARY[name], dev_best]):
                out["holdout"][f"{name} · {vname}"] = evaluate(
                    data, builder, variants[vname], PERIODS["holdout"], cfg, baselines=True)
                print(f"hold {name} · {vname}", flush=True)
    (HERE / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote results.json")


if __name__ == "__main__":
    main()
