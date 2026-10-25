"""Trading service: executor loop + dashboard, plus operator commands.

    python -m pa_agent.trading run        # executor + dashboard (default)
    python -m pa_agent.trading check      # verify OKX keys / account / instrument
    python -m pa_agent.trading status     # active trades, recent events
    python -m pa_agent.trading report     # performance + AI-quality report
    python -m pa_agent.trading kill       # emergency stop: flatten all, block signals
    python -m pa_agent.trading resume     # clear the emergency stop

OKX credentials: ``settings.trading.okx_*`` or env ``OKX_API_KEY`` /
``OKX_SECRET_KEY`` / ``OKX_PASSPHRASE`` (env wins).
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("pa_agent.trading")


def load_trading_settings() -> Any:
    from pa_agent.config.paths import SETTINGS_JSON_PATH
    from pa_agent.config.settings import load_settings

    return load_settings(SETTINGS_JSON_PATH)


def okx_credentials(trading: Any):
    from pa_agent.exchange.okx_rest import OkxCredentials

    return OkxCredentials(
        api_key=os.environ.get("OKX_API_KEY") or trading.okx_api_key,
        secret_key=os.environ.get("OKX_SECRET_KEY") or trading.okx_secret_key,
        passphrase=os.environ.get("OKX_PASSPHRASE") or trading.okx_passphrase,
    )


def build_executor(settings: Any, db_path: Path | None = None):
    from pa_agent.config.paths import QUANT_DB_PATH
    from pa_agent.exchange.okx_rest import OkxRestClient
    from pa_agent.trading.broker import OkxBroker
    from pa_agent.trading.executor import Executor, ExecutorConfig
    from pa_agent.trading.store import TradeStore

    t = settings.trading
    client = OkxRestClient(okx_credentials(t), demo=t.demo)
    store = TradeStore(db_path or QUANT_DB_PATH)
    return Executor(store, OkxBroker(client), ExecutorConfig.from_settings(t))


def configure_logging(verbose: bool, filename: str = "trading.log") -> None:
    from pa_agent.config.paths import LOGS_DIR

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / filename, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=fmt,
                        handlers=handlers, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _mode_label(demo: bool) -> str:
    return "模拟盘" if demo else "⚠️ 实盘"


# ── commands ──────────────────────────────────────────────────────────────────

def preflight_trading(settings: Any, *, i_understand_live: bool) -> str | None:
    """Return an error message when the trading service must not start."""
    t = settings.trading
    if not okx_credentials(t).is_complete():
        return "OKX API Key / Secret / Passphrase 未配置（settings.json → trading，或环境变量 OKX_*）"
    if not t.demo and not i_understand_live:
        return "trading.demo=false 是实盘模式。确认要用真钱交易，请加参数 --i-understand-live"
    return None


def run_executor_loop(ex: Any, stop: threading.Event, interval_s: float) -> None:
    while not stop.is_set():
        started = time.monotonic()
        try:
            ex.tick()
        except Exception as exc:
            logger.exception("executor tick failed: %s", exc)
            ex.store.log_event("tick_error", f"执行循环异常: {exc}", level="error")
        stop.wait(max(0.2, interval_s - (time.monotonic() - started)))


def start_trading(settings: Any, *, dashboard: bool = True, host: str = "", port: int = 0) -> Any:
    """Build the executor (and dashboard thread); caller runs :func:`run_executor_loop`."""
    t = settings.trading
    ex = build_executor(settings)
    logger.info("交易服务启动：%s，杠杆 %sx，%s，单笔风险 %s%%，数据库 %s",
                _mode_label(t.demo), t.leverage, t.margin_mode, t.risk_per_trade_pct,
                ex.store.path)
    if dashboard:
        from pa_agent.trading.dashboard import serve_in_thread

        serve_in_thread(ex.store, settings, host=host or t.dashboard_host,
                        port=port or t.dashboard_port)
    return ex


def install_stop_signals(stop: threading.Event, message: str) -> None:
    def _sig(_signo, _frame):
        logger.info(message)
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)


def cmd_run(args: argparse.Namespace) -> int:
    settings = load_trading_settings()
    err = preflight_trading(settings, i_understand_live=args.i_understand_live)
    if err:
        print(err)
        return 2
    ex = start_trading(settings, dashboard=not args.no_dashboard, host=args.host, port=args.port)
    stop = threading.Event()
    install_stop_signals(stop, "收到退出信号，停止交易服务（交易所上的挂单和止损保持不变）")
    run_executor_loop(ex, stop, settings.trading.poll_interval_s)
    return 0


def cmd_check(_args: argparse.Namespace) -> int:
    from pa_agent.exchange.okx_rest import OkxError

    settings = load_trading_settings()
    t = settings.trading
    ex = build_executor(settings)
    broker = ex.broker
    print(f"模式: {_mode_label(t.demo)}   信号开关 trading.enabled = {t.enabled}")
    try:
        print(f"行情: BTC-USDT-SWAP 最新价 {broker.last_price('BTC-USDT-SWAP')}")
    except OkxError as exc:
        print(f"✗ 无法访问 OKX 行情（国内网络需要代理 HTTPS_PROXY）: {exc}")
        return 1
    if not okx_credentials(t).is_complete():
        print("✗ OKX API Key / Secret / Passphrase 未配置")
        return 1
    try:
        acct = broker.account()
        print(f"✓ 账户权益 {acct.total_eq:.2f} USD，可用 {acct.avail_eq:.2f} USDT")
        cfg = broker._c.account_config()
        print(f"  持仓模式 {cfg.get('posMode')}（执行器使用 net_mode 单向持仓）"
              f"，账户模式 acctLv={cfg.get('acctLv')}")
        if str(cfg.get("acctLv")) == "1":
            print("✗ 当前是简单交易模式（acctLv=1），不能交易合约。请在 OKX 设置里切换到"
                  "单币种保证金或以上模式")
            return 1
    except OkxError as exc:
        hint = "（模拟盘必须用「模拟交易」里创建的 API Key）" if t.demo else ""
        print(f"✗ 账户接口失败{hint}: {exc}")
        return 1
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    from datetime import datetime

    from pa_agent.config.paths import QUANT_DB_PATH
    from pa_agent.trading.executor import HEARTBEAT_KEY, KILL_KEY
    from pa_agent.trading.store import TradeStore

    store = TradeStore(QUANT_DB_PATH)
    hb = int(store.get_kv(HEARTBEAT_KEY, "0") or 0)
    age = (time.time() * 1000 - hb) / 1000 if hb else None
    print(f"执行器心跳: {'从未运行' if age is None else f'{age:.0f} 秒前'}"
          f"   紧急停止: {'开启' if store.get_kv(KILL_KEY) == '1' else '关闭'}")
    active = store.active_trades()
    print(f"\n活跃交易 {len(active)} 笔")
    for tr in active:
        print(f"  #{tr['id']} {tr['inst_id']} {tr['direction']} {tr['entry_type']} {tr['state']} "
              f"{(tr['filled_contracts'] or tr['contracts']) * tr['ct_val'] * (tr['avg_entry'] or tr['planned_entry'] or 0):,.0f}U "
              f"入场 {tr['avg_entry'] or tr['planned_entry']} "
              f"止损 {tr['stop_loss']} 止盈 {tr['tp1']}/{tr['tp2']}")
    print("\n最近事件")
    for ev in reversed(store.events(limit=15)):
        ts = datetime.fromtimestamp(ev["ts_ms"] / 1000).strftime("%m-%d %H:%M:%S")
        print(f"  {ts} [{ev['level']}] {ev['kind']} {ev['message']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from pa_agent.config.paths import QUANT_DB_PATH
    from pa_agent.trading.stats import ai_quality, summarize
    from pa_agent.trading.store import TradeStore

    store = TradeStore(QUANT_DB_PATH)
    trades = store.list_trades(limit=100_000)
    if args.days:
        since = int((time.time() - args.days * 86400) * 1000)
        trades = [t for t in trades if (t.get("closed_ms") or t.get("created_ms") or 0) >= since]
    s = summarize(trades)

    def f(v, suffix="", nd=2):
        return "—" if v is None else f"{v:,.{nd}f}{suffix}"

    print("=== 收益概览 ===")
    print(f"平仓笔数 {s['trades']}（盈 {s['wins']} / 亏 {s['losses']}）  胜率 {f(s['win_rate'], '%', 1)}")
    print(f"净盈亏 {f(s['net_pnl'])} U   收益率 {f(s['return_pct'], '%')}   最大回撤（已实现） "
          f"{f(s['max_drawdown'])} U ({f(s['max_drawdown_pct'], '%')}, {f(s['max_drawdown_r'])}R)")
    print(f"平均盈利 {f(s['avg_win'])}  平均亏损 {f(s['avg_loss'])}  平均盈/亏 {f(s['payoff_ratio'])}  "
          f"利润因子 {f(s['profit_factor'])}  平均R {f(s['avg_r'])}  合计R {f(s['total_r'])}")
    print(f"手续费 {f(s['fees'])}  资金费 {f(s['funding'])}")

    sigs = {r["id"]: r for r in store._rows("SELECT * FROM signals")}
    q = ai_quality(trades, sigs, list(sigs.values()))
    for title, key in (("按策略版本", "by_version"), ("按 AI 交易把握度", "by_confidence"), ("按 AI 估计胜率", "by_estimated_win_rate"),
                       ("按入场方式", "by_entry_type"), ("按方向", "by_direction"),
                       ("按市场周期", "by_cycle"), ("按离场原因", "by_exit_reason")):
        rows = q[key]
        if not rows:
            continue
        print(f"\n=== {title} ===")
        for g in rows:
            print(f"  {g['group']:<14} {g['trades']:>4} 笔  胜率 {g['win_rate']:5.1f}%  "
                  f"净盈亏 {g['net_pnl']:>10,.2f}  平均R {f(g['avg_r'])}")
    print(f"\n=== 下单信号处理 === 共 {q['signals_total']} 条分析（下单信号 {q['signals_actionable']}）: "
          + ", ".join(f"{k}={v}" for k, v in q["signal_funnel"].items()))
    return 0


def cmd_kill(_args: argparse.Namespace) -> int:
    from pa_agent.config.paths import QUANT_DB_PATH
    from pa_agent.trading.executor import KILL_DONE_KEY, KILL_KEY
    from pa_agent.trading.store import TradeStore

    store = TradeStore(QUANT_DB_PATH)
    store.set_kv(KILL_KEY, "1")
    store.set_kv(KILL_DONE_KEY, "0")
    store.log_event("kill", "命令行触发紧急停止", level="warn")
    print("已设置紧急停止：运行中的交易服务会在下一轮撤销所有挂单并平掉所有仓位。")
    return 0


def cmd_resume(_args: argparse.Namespace) -> int:
    from pa_agent.config.paths import QUANT_DB_PATH
    from pa_agent.trading.executor import KILL_DONE_KEY, KILL_KEY
    from pa_agent.trading.store import TradeStore

    store = TradeStore(QUANT_DB_PATH)
    store.set_kv(KILL_KEY, "0")
    store.set_kv(KILL_DONE_KEY, "0")
    store.log_event("resume", "命令行解除紧急停止")
    print("已解除紧急停止。")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m pa_agent.trading", description="PA Agent OKX 交易服务")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="运行执行器 + 收益面板（默认）")
    run.add_argument("--no-dashboard", action="store_true")
    run.add_argument("--host", default="")
    run.add_argument("--port", type=int, default=0)
    run.add_argument("--i-understand-live", action="store_true", help="实盘模式必须显式确认")
    sub.add_parser("check", help="检查 OKX 连接、密钥和账户模式")
    sub.add_parser("status", help="查看活跃交易和最近事件")
    rep = sub.add_parser("report", help="收益与 AI 质量报告")
    rep.add_argument("--days", type=int, default=0)
    sub.add_parser("kill", help="紧急停止：撤单并平掉所有仓位")
    sub.add_parser("resume", help="解除紧急停止")
    args = p.parse_args(argv)
    configure_logging(args.verbose)
    handlers = {"run": cmd_run, "check": cmd_check, "status": cmd_status, "report": cmd_report,
                "kill": cmd_kill, "resume": cmd_resume}
    if args.cmd is None:
        args = p.parse_args(["run", *(argv or [])])
    return handlers[args.cmd](args)
