"""Headless analysis runner — the GUI's 持续跟踪分析 without a window (VPS-friendly).

    python -m pa_agent.headless --symbols BTC-USDT-SWAP,ETH-USDT-SWAP --timeframe 1h
    python -m pa_agent.headless --symbols BTC-USDT-SWAP --timeframe 1h --trade   # + executor + dashboard

Each time a bar closes, every symbol gets one two-stage analysis (sequentially).
Order decisions are logged to trade_records/*.csv (+ chart), pushed to Feishu /
PushPlus when configured, and queued as trading signals when
``settings.trading.enabled``.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pa_agent.headless")

_ORDER_TYPES = frozenset({"限价单", "突破单", "市价单"})

# Trade-store kv keys shared with the dashboard: the page writes RUNNER_CONFIG_KEY,
# the runner applies it between polls and reports what it runs in RUNNER_ACTIVE_KEY.
RUNNER_CONFIG_KEY = "runner_config"
RUNNER_ACTIVE_KEY = "runner_active"


def read_runner_config(store: Any) -> tuple[list[str], str] | None:
    try:
        cfg = json.loads(store.get_kv(RUNNER_CONFIG_KEY) or "null")
        symbols = [str(s) for s in cfg["symbols"] if str(s).strip()]
        timeframe = str(cfg["timeframe"])
    except (TypeError, ValueError, KeyError):
        return None
    return (symbols, timeframe) if symbols and timeframe else None


def write_runner_config(store: Any, symbols: list[str], timeframe: str) -> None:
    store.set_kv(RUNNER_CONFIG_KEY, json.dumps(
        {"symbols": symbols, "timeframe": timeframe, "ts_ms": int(time.time() * 1000)}))


def is_order_decision(inner: dict[str, Any], confidence_threshold: int = 0) -> bool:
    """Same rule as the GUI's ``has_order_opportunity`` (kept Qt-free)."""
    if str(inner.get("order_type") or "") not in _ORDER_TYPES:
        return False
    if confidence_threshold > 0:
        try:
            conf = int(float(str(inner.get("trade_confidence")).strip()))
        except (TypeError, ValueError):
            return False
        return conf >= confidence_threshold
    return True


def newest_closed_ts(bars: list[Any], timeframe: str, symbol: str, now_ms: int) -> int | None:
    """ts_open of the most recently *closed* bar (bars newest-first)."""
    from pa_agent.data.bar_close_wait import current_forming_ts

    if not bars:
        return None
    forming = current_forming_ts(bars, timeframe, symbol=symbol, now_ms=now_ms)
    for bar in bars:
        if forming is None or int(bar.ts_open) != int(forming):
            return int(bar.ts_open)
    return None


@dataclass
class SymbolState:
    symbol: str
    source: Any
    last_closed_ts: int | None = None
    analyses: int = 0
    failures: int = 0
    last_error: str = ""


@dataclass
class HeadlessRunner:
    ctx: Any                      # AppContext (settings, AI client, assembler, …)
    symbols: list[str]
    timeframe: str
    source_kind: str = "okx"
    bar_count: int = 0
    run_now: bool = False
    poll_s: float = 10.0
    store: Any = None             # TradeStore: page-editable symbols/timeframe (None = fixed)
    states: dict[str, SymbolState] = field(default_factory=dict)
    _cancel: Any = None

    def __post_init__(self) -> None:
        if not self.bar_count:
            self.bar_count = int(self.ctx.settings.general.analysis_bar_count)
        self.apply_config(self.symbols, self.timeframe)

    def _open_source(self, sym: str, timeframe: str) -> SymbolState:
        from pa_agent.data.factory import create_data_source

        src = create_data_source(self.source_kind)
        src.connect()
        src.subscribe(sym, timeframe)
        return SymbolState(symbol=getattr(src, "inst_id", "") or sym, source=src)

    def apply_config(self, symbols: list[str], timeframe: str) -> None:
        """(Re)subscribe to *symbols* on *timeframe*; unchanged symbols keep their state.

        Newly added symbols wait for their next bar close, like at startup.
        """
        keep = self.states if timeframe == self.timeframe else {}
        states: dict[str, SymbolState] = {}
        for sym in symbols:
            old = next((st for name, st in keep.items() if name == sym or st.symbol == sym), None)
            if old is not None:
                states[old.symbol] = old
                continue
            try:
                st = self._open_source(sym, timeframe)
            except Exception as exc:
                logger.error("%s 订阅失败，已跳过: %s", sym, exc)
                continue
            states[st.symbol] = st
        for name, st in self.states.items():
            if states.get(name) is not st:
                try:
                    st.source.disconnect()
                except Exception:
                    pass
        self.states, self.symbols, self.timeframe = states, list(symbols), timeframe

    def sync_config(self) -> None:
        """Pick up symbols/timeframe saved from the dashboard and report what is running."""
        if self.store is None:
            return
        cfg = read_runner_config(self.store)
        if cfg is not None and (cfg[0] != self.symbols or cfg[1] != self.timeframe):
            logger.info("面板修改了分析配置：%s %s → %s %s", ",".join(self.symbols), self.timeframe,
                        ",".join(cfg[0]), cfg[1])
            self.apply_config(*cfg)
            self.store.log_event("runner_config",
                                 f"后台分析已切换为 {','.join(self.states) or '（无可用品种）'} · {self.timeframe}")
        self.store.set_kv(RUNNER_ACTIVE_KEY, json.dumps(
            {"symbols": list(self.states), "timeframe": self.timeframe, "ts_ms": int(time.time() * 1000)}))

    # ── orchestration ─────────────────────────────────────────────────────────

    def _orchestrator(self):
        from pa_agent.orchestrator.two_stage import TwoStageOrchestrator

        c = self.ctx
        return TwoStageOrchestrator(
            client=c.client, assembler=c.assembler, router=c.router, validator=c.validator,
            pending_writer=c.pending_writer, exp_reader=c.exp_reader, settings=c.settings,
        )

    def _fetch(self, st: SymbolState) -> list[Any]:
        from pa_agent.data.snapshot import INDICATOR_WARMUP_BARS

        return list(st.source.latest_snapshot(self.bar_count + INDICATOR_WARMUP_BARS + 5))

    def poll_once(self, stop: threading.Event | None = None) -> int:
        """Check every symbol; analyse those whose bar just closed. Returns analyses run."""
        ran = 0
        for st in self.states.values():
            if stop is not None and stop.is_set():
                break
            try:
                bars = self._fetch(st)
            except Exception as exc:
                if str(exc) != st.last_error:
                    logger.warning("%s K线获取失败: %s", st.symbol, exc)
                st.last_error = str(exc)
                continue
            st.last_error = ""
            now = int(time.time() * 1000)
            closed = newest_closed_ts(bars, self.timeframe, st.symbol, now)
            if closed is None:
                continue
            if st.last_closed_ts is None and not self.run_now:
                st.last_closed_ts = closed
                logger.info("%s %s 已就绪，等待下一根K线收盘后开始分析", st.symbol, self.timeframe)
                continue
            if closed == st.last_closed_ts:
                continue
            st.last_closed_ts = closed
            self.analyze(st, bars)
            ran += 1
        self.run_now = False
        return ran

    def analyze(self, st: SymbolState, bars: list[Any]) -> dict[str, Any] | None:
        """Run one two-stage analysis and dispatch the result. Returns the inner decision."""
        from pa_agent.ai.stage2_payload import prepare_stage2_for_ui
        from pa_agent.orchestrator.analysis_prep import prepare_analysis
        from pa_agent.util.threading import CancelToken

        settings = self.ctx.settings
        g = settings.general
        prep = prepare_analysis(
            bars_raw=bars, symbol=st.symbol, timeframe=self.timeframe,
            bar_count=self.bar_count, now_ms=int(time.time() * 1000),
            force_incremental=False,
            incremental_threshold=int(getattr(g, "incremental_max_new_bars", 10)),
        )
        if prep is None:
            logger.warning("%s 数据不足，跳过本轮", st.symbol)
            return None
        mode = (f"增量（{prep.incremental_detail}）" if prep.incremental_new_bar_count is not None
                else "完整")
        logger.info("▶ %s %s 开始分析（%s）", st.symbol, self.timeframe, mode)
        self._cancel = CancelToken()
        started = time.monotonic()

        def on_event(ev):
            logger.info("  %s %s", st.symbol, getattr(ev, "name", ev))

        try:
            record = self._orchestrator().submit(
                prep.frame, self._cancel, on_event,
                previous_record=prep.previous_record,
                incremental_new_bar_count=prep.incremental_new_bar_count,
            )
        except Exception as exc:
            st.failures += 1
            logger.exception("%s 分析异常: %s", st.symbol, exc)
            return None
        finally:
            self._cancel = None
        st.analyses += 1
        decision = record.stage2_decision or {}
        if not decision:
            st.failures += 1
            exc = record.exception or {}
            logger.warning("%s 分析未完成: %s %s", st.symbol, exc.get("type", ""),
                           exc.get("message", ""))
            return None
        inner = prepare_stage2_for_ui(
            decision,
            stage1_json=record.stage1_diagnosis or None,
            skip_next_bar=not bool(getattr(g, "enable_next_bar_prediction", False)),
        )
        logger.info("◀ %s 结论: %s %s 入场 %s 止损 %s 止盈 %s（%.0fs）", st.symbol,
                    inner.get("order_direction", ""), inner.get("order_type", ""),
                    inner.get("entry_price", "—"), inner.get("stop_loss_price", "—"),
                    inner.get("take_profit_price", "—"), time.monotonic() - started)
        self._dispatch(st, record, decision, inner, prep.frame)
        return inner

    def _dispatch(self, st: SymbolState, record: Any, decision: dict, inner: dict, frame: Any) -> None:
        from pa_agent.trading.signals import emit_signal

        settings = self.ctx.settings
        g = settings.general
        stage1 = record.stage1_diagnosis or None
        key = str(getattr(record.meta, "timestamp_local_ms", int(time.time() * 1000)))
        chart = ""
        if is_order_decision(inner, int(getattr(g, "decision_confidence_threshold", 0) or 0)):
            chart = self._log_and_notify(st, decision, inner, stage1, frame)
        try:
            emit_signal(
                settings, decision_inner=inner, stage2_full=decision, stage1_diagnosis=stage1,
                symbol=st.symbol, timeframe=self.timeframe, source="headless",
                analysis_key=key, chart_image=chart,
            )
        except Exception as exc:
            logger.warning("交易信号写入失败: %s", exc)

    def _log_and_notify(self, st: SymbolState, decision: dict, inner: dict,
                        stage1: dict | None, frame: Any) -> str:
        """CSV + chart (trade_logger) and Feishu / PushPlus pushes; returns chart path."""
        from pa_agent.records.trade_logger import _TRADE_RECORDS_DIR, save_trade_record

        settings = self.ctx.settings
        g = settings.general
        save_trade_record(
            decision_inner=inner, stage2_full=decision, stage1_diagnosis=stage1, frame=frame,
            meta_symbol=st.symbol, meta_timeframe=self.timeframe,
            decision_stance=str(getattr(g, "decision_stance", "") or ""),
            model_name=str(getattr(settings.provider, "model", "") or ""),
            structure_flip_cooldown_bars=int(getattr(g, "structure_flip_cooldown_bars", 3) or 3),
        )
        safe_sym = st.symbol.replace("/", "-").replace("\\", "-")
        imgs = sorted(_TRADE_RECORDS_DIR.glob(f"{safe_sym}_{self.timeframe.replace('/', '-')}_*.png"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        latest = imgs[0] if imgs and time.time() - imgs[0].stat().st_mtime < 120 else None
        try:
            from pa_agent.notify.feishu_notifier import send_order_signal as feishu
            from pa_agent.notify.pushplus_notifier import pushplus_is_active
            from pa_agent.notify.pushplus_notifier import send_order_signal as pushplus

            feishu(decision_inner=inner, stage2_full=decision, symbol=st.symbol,
                   timeframe=self.timeframe, chart_image_path=latest, settings=settings)
            if pushplus_is_active(settings):
                pushplus(decision_inner=inner, stage2_full=decision, symbol=st.symbol,
                         timeframe=self.timeframe, settings=settings)
        except Exception as exc:
            logger.warning("下单信号通知失败: %s", exc)
        return str(latest) if latest else ""

    def cancel(self) -> None:
        if self._cancel is not None:
            self._cancel.set()

    def run_forever(self, stop: threading.Event) -> None:
        logger.info("后台分析启动：%s %s，数据源 %s，每次 %d 根K线", ",".join(self.states),
                    self.timeframe, self.source_kind, self.bar_count)
        while not stop.is_set():
            started = time.monotonic()
            try:
                self.sync_config()
                self.poll_once(stop)
            except Exception as exc:
                logger.exception("轮询异常: %s", exc)
            stop.wait(max(1.0, self.poll_s - (time.monotonic() - started)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m pa_agent.headless",
                                description="PA Agent 后台持续分析（无窗口）")
    p.add_argument("--symbols", default="", help="逗号分隔，默认用 settings 里的 last_symbol")
    p.add_argument("--timeframe", default="", help="如 15m / 1h / 4h，默认用 settings 里的 last_timeframe")
    p.add_argument("--source", default="okx", help="数据源：okx（默认）/ tradingview / yfinance / …")
    p.add_argument("--bars", type=int, default=0, help="每次分析的K线根数，默认 settings.analysis_bar_count")
    p.add_argument("--now", action="store_true", help="启动后立即分析一次，不等下一根K线收盘")
    p.add_argument("--once", action="store_true", help="只分析一次就退出（配合 --now）")
    p.add_argument("--poll", type=float, default=10.0, help="检查K线收盘的间隔秒数")
    p.add_argument("--trade", action="store_true", help="同一进程内运行交易执行器 + 收益面板")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--i-understand-live", action="store_true", help="实盘模式必须显式确认")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    from pa_agent.trading.service import (
        configure_logging,
        install_stop_signals,
        preflight_trading,
        run_executor_loop,
        start_trading,
    )

    configure_logging(args.verbose, filename="headless.log")
    from pa_agent.app_context import AppContext
    from pa_agent.config.settings import provider_api_key_configured

    ctx = AppContext.bootstrap()
    # bootstrap() re-installs the project's handlers (logs/pa_agent.log + console) and
    # quiets the root logger to WARNING; the runner's own progress lines must stay visible.
    logging.getLogger("pa_agent").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    for name in ("pa_agent.ai", "pa_agent.data", "pa_agent.records"):
        logging.getLogger(name).setLevel(logging.WARNING)
    settings = ctx.settings
    if not provider_api_key_configured(settings):
        print("AI 模型 API Key 未配置：先在桌面程序「AI 模型设置」里填写，或编辑 config/settings.json")
        return 2
    store = None
    if args.trade:
        from pa_agent.config.paths import QUANT_DB_PATH
        from pa_agent.trading.store import TradeStore

        store = TradeStore(QUANT_DB_PATH)
    # Precedence: command line > what was last saved from the dashboard > settings.json
    saved = read_runner_config(store) if store is not None else None
    symbols = [s.strip() for s in (args.symbols or (",".join(saved[0]) if saved else "")
                                   or settings.general.last_symbol).split(",") if s.strip()]
    timeframe = args.timeframe or (saved[1] if saved else "") or settings.general.last_timeframe
    if args.trade:
        err = preflight_trading(settings, i_understand_live=args.i_understand_live)
        if err:
            print(err)
            return 2
        if not settings.trading.enabled:
            print("提示：settings.json → trading.enabled 为 false，分析结果不会生成交易信号")

    runner = HeadlessRunner(ctx=ctx, symbols=symbols, timeframe=timeframe, source_kind=args.source,
                            bar_count=args.bars, run_now=args.now or args.once, poll_s=args.poll,
                            store=store)
    if store is not None:
        write_runner_config(store, runner.symbols, runner.timeframe)
    if args.once:
        runner.poll_once()
        return 0

    stop = threading.Event()
    install_stop_signals(stop, "收到退出信号，正在停止（交易所上的挂单和止损保持不变）")
    threads: list[threading.Thread] = []
    if args.trade:
        ex = start_trading(settings, dashboard=not args.no_dashboard)
        threads.append(threading.Thread(target=run_executor_loop, name="executor", daemon=True,
                                        args=(ex, stop, settings.trading.poll_interval_s)))
    for t in threads:
        t.start()
    try:
        runner.run_forever(stop)
    finally:
        runner.cancel()
        stop.set()
        for t in threads:
            t.join(timeout=10)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
