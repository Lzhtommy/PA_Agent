"""Render results.json (from run.py) as the data tables of REPORT.md.

    uv run python research/okx_backtest/report.py > research/okx_backtest/tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def f(v, nd=2, sign=False):
    if v is None:
        return "—"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def pct(v):
    return "—" if v is None else f"{v:.0%}"


def verdict(s: dict) -> str:
    """PREREG pass criteria (holdout, primary variant)."""
    if not s.get("trades"):
        return "无交易"
    years = s.get("by_year", {})
    checks = [
        s["mean_r"] > 0.05 and s["ci"][0] > 0,
        s["trades"] >= 100,
        sum(m > 0 for _, m in years.values()) > len(years) / 2,
        s.get("random", {}).get("p95") is not None and s["mean_r"] > s["random"]["p95"],
    ]
    return "通过" if all(checks) else "不通过（" + "".join("✓" if c else "✗" for c in checks) + "）"


def table(period: dict, with_verdict: bool) -> list[str]:
    head = ("| 策略 · 变体 | 成交笔数 | 成交率 | 胜率 | 平均 R（95% CI） | 合计 R | 最大回撤 R | 利润因子 "
            "| 手续费 R | 先后不明 | 有利处理时平均 R | 随机入场 平均 / P95 | 优于随机 | 真实(无子K) / 打乱 |")
    if with_verdict:
        head += " 判定 |"
    lines = [head, "|" + "|".join(["---"] * (head.count("|") - 1)) + "|"]
    for name, s in period.items():
        if not s.get("trades"):
            lines.append(f"| {name} | 0 |" + " |" * (head.count("|") - 3))
            continue
        rnd, shf = s.get("random", {}), s.get("shuffle", {})
        row = (f"| {name} | {s['trades']} | {pct(s['fill_rate'])} | {pct(s['win_rate'])} "
               f"| {f(s['mean_r'], sign=True)}（{f(s['ci'][0], sign=True)} ~ {f(s['ci'][1], sign=True)}） "
               f"| {f(s['total_r'], 1, True)} | {f(s['max_dd_r'], 1)} | {f(s['profit_factor'])} "
               f"| {f(s['fee_r'])} | {pct(s.get('ambiguous'))} | {f(s.get('mean_r_favorable'), sign=True)} "
               f"| {f(rnd.get('mean'), sign=True)} / {f(rnd.get('p95'), sign=True)} | {pct(rnd.get('pct_rank'))} "
               f"| {f(shf.get('real_nosub_mean'), sign=True)} / {f(shf.get('mean'), sign=True)} |")
        if with_verdict:
            row += f" {verdict(s)} |"
        lines.append(row)
    return lines


def breakdown(period: dict) -> list[str]:
    lines = ["| 策略 · 变体 | 按年份（笔数 / 平均 R） | 多 / 空 |", "|---|---|---|"]
    for name, s in period.items():
        if not s.get("trades"):
            continue
        years = "，".join(f"{y}: {n} / {m:+.2f}" for y, (n, m) in s["by_year"].items())
        sides = "，".join(f"{'多' if d == 'long' else '空'} {n} / {m:+.2f}" for d, (n, m) in s["by_side"].items())
        lines.append(f"| {name} | {years} | {sides} |")
    return lines


def main() -> None:
    res = json.loads((HERE / "results.json").read_text())
    out = [f"生成于 {res['generated']}", "", "### 开发段 2020–2023", ""]
    out += table(res["dev"], False) + [""] + breakdown(res["dev"])
    if res.get("holdout"):
        out += ["", "### 检验段 2024 至今", ""]
        out += table(res["holdout"], True) + [""] + breakdown(res["holdout"])
    print("\n".join(out))


if __name__ == "__main__":
    main()
