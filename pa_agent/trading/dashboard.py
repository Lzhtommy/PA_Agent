"""Read-mostly web dashboard over the trading SQLite store (FastAPI)."""

import hmac
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from pa_agent.trading.executor import HEARTBEAT_KEY, KILL_DONE_KEY, KILL_KEY
from pa_agent.trading.stats import ai_quality, summarize, version_label
from pa_agent.trading.store import TradeStore, now_ms

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _loads(text: str | None) -> Any:
    try:
        return json.loads(text or "{}")
    except (TypeError, ValueError):
        return {}


def create_app(store: TradeStore, *, demo: bool, token: str = "", chart_root: Path | None = None,
               market: Any = None):
    from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response
    from fastapi.responses import FileResponse, HTMLResponse

    from pa_agent.config.paths import TRADE_RECORDS_DIR

    chart_root = (chart_root or TRADE_RECORDS_DIR).resolve()
    candle_cache: dict[tuple[str, str, int], tuple[float, list[dict[str, Any]]]] = {}
    candle_lock = threading.Lock()

    def market_client():
        nonlocal market
        if market is None:
            from pa_agent.exchange.okx_rest import OkxRestClient

            market = OkxRestClient(demo=False)  # public candles: live prices, no key needed
        return market
    app = FastAPI(title="PA Agent Trading", docs_url=None, redoc_url=None)

    def auth(request: Request) -> None:
        if not token:
            return
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        supplied = supplied or request.query_params.get("token", "")
        if not hmac.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="unauthorized")

    guard = [Depends(auth)]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")

    def since_ms(days: int) -> int:
        return int((time.time() - days * 86400) * 1000) if days else 0

    def stat_trades(days: int, version: str) -> list[dict[str, Any]]:
        """Trades for the statistics cards (period by close time, optional version)."""
        trades = store.list_trades(limit=100_000, version=version or None)
        since = since_ms(days)
        if since:
            trades = [t for t in trades if (t.get("closed_ms") or t.get("created_ms") or 0) >= since]
        return trades

    @app.get("/api/overview", dependencies=guard)
    def overview(days: int = 0, version: str = "") -> dict[str, Any]:
        hb = int(store.get_kv(HEARTBEAT_KEY, "0") or 0)
        return {
            "demo": demo,
            "heartbeat_ms": hb or None,
            "kill_switch": store.get_kv(KILL_KEY) == "1",
            "equity": store.latest_equity(),
            "summary": summarize(stat_trades(days, version)),
            "active": store.active_trades(),
        }

    @app.get("/api/versions", dependencies=guard)
    def versions() -> list[dict[str, Any]]:
        from pa_agent.headless import read_runner_config
        from pa_agent.trading.signals import prompt_hash

        try:
            current = prompt_hash()
        except OSError:
            current = ""
        runner = read_runner_config(store)
        return [{**v, "key": f"{v['model']}|{v['prompt_hash']}|{v['timeframe']}",
                 "label": version_label(v["model"], v["prompt_hash"], v["timeframe"]),
                 "current": bool(current) and v["prompt_hash"] == current
                 and (runner is None or v["timeframe"] == runner[1])}
                for v in store.versions()]

    @app.get("/api/equity", dependencies=guard)
    def equity(days: int = 0) -> list[dict[str, Any]]:
        since = int((time.time() - days * 86400) * 1000) if days else 0
        rows = store.equity_series(since_ms=since, demo=demo)
        # Downsample long series to ≤ 1500 points for the chart.
        step = max(1, len(rows) // 1500)
        return rows[::step] + ([rows[-1]] if rows and (len(rows) - 1) % step else [])

    @app.get("/api/trades", dependencies=guard)
    def trades(response: Response, limit: int = 200, offset: int = 0, state: str = "",
               inst_id: str = "", version: str = "", days: int = 0) -> list[dict[str, Any]]:
        states = tuple(s for s in state.split(",") if s) or None
        if days:  # statistics view (cumulative chart): whole period, no paging
            rows = [t for t in stat_trades(days, version)
                    if not states or t["state"] in states][:min(limit, 2000)]
            response.headers["X-Total-Count"] = str(len(rows))
        else:
            response.headers["X-Total-Count"] = str(store.count_trades(
                states=states, inst_id=inst_id or None, version=version or None))
            rows = store.list_trades(states=states, limit=min(limit, 2000), offset=max(0, offset),
                                     inst_id=inst_id or None, version=version or None)
        sig_ids = {r["signal_id"] for r in rows if r.get("signal_id")}
        conf: dict[int, dict[str, Any]] = {}
        for sid in sig_ids:
            s = store.get_signal(sid)
            if s:
                conf[sid] = {"trade_confidence": s["trade_confidence"],
                             "estimated_win_rate": s["estimated_win_rate"],
                             "timeframe": s["timeframe"],
                             "version": version_label(s["model"], s["prompt_hash"], s["timeframe"])}
        return [{**r, **conf.get(r.get("signal_id") or -1, {})} for r in rows]

    @app.get("/api/trades/{trade_id}", dependencies=guard)
    def trade_detail(trade_id: int) -> dict[str, Any]:
        t = store.get_trade(trade_id)
        if t is None:
            raise HTTPException(status_code=404, detail="trade not found")
        sig = store.get_signal(t["signal_id"]) if t.get("signal_id") else None
        if sig:
            sig = {**sig, "decision": _loads(sig.pop("decision_json")),
                   "stage1": _loads(sig.pop("stage1_json"))}
        return {"trade": t, "signal": sig, "events": store.events(trade_id=trade_id)}

    @app.get("/api/signals", dependencies=guard)
    def signals(response: Response, limit: int = 200, offset: int = 0, symbol: str = "",
                actionable: bool = False, version: str = "") -> list[dict[str, Any]]:
        response.headers["X-Total-Count"] = str(
            store.count_signals(symbol or None, actionable, version or None))
        return store.list_signals(limit=min(limit, 2000), offset=max(0, offset), symbol=symbol or None,
                                  actionable=actionable, version=version or None)

    @app.get("/api/signals/{signal_id}", dependencies=guard)
    def signal_detail(signal_id: int) -> dict[str, Any]:
        sig = store.get_signal(signal_id)
        if sig is None:
            raise HTTPException(status_code=404, detail="signal not found")
        return {**sig, "decision": _loads(sig.pop("decision_json")),
                "stage1": _loads(sig.pop("stage1_json"))}

    @app.get("/api/ai-quality", dependencies=guard)
    def quality(days: int = 0, version: str = "") -> dict[str, Any]:
        sigs = {r["id"]: r for r in store._rows("SELECT * FROM signals")}
        want = tuple((version.split("|") + ["", ""])[:3])
        since = since_ms(days)
        in_scope = [s for s in sigs.values() if s["created_ms"] >= since
                    and (not version or (s["model"], s["prompt_hash"], s["timeframe"]) == want)]
        return ai_quality(stat_trades(days, version), sigs, in_scope)

    @app.get("/api/events", dependencies=guard)
    def events(response: Response, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        response.headers["X-Total-Count"] = str(store.count_events())
        return store.events(limit=min(limit, 1000), offset=max(0, offset))

    @app.get("/api/instruments", dependencies=guard)
    def instruments() -> list[str]:
        return store.instruments()

    @app.get("/api/candles", dependencies=guard)
    def candles(inst_id: str, timeframe: str = "1h", limit: int = 300) -> dict[str, Any]:
        from pa_agent.data.okx_source import okx_bar_for_timeframe
        from pa_agent.exchange.okx_rest import OkxError

        try:
            bar = okx_bar_for_timeframe(timeframe)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        limit = max(10, min(limit, 1000))
        key = (inst_id, bar, limit)
        with candle_lock:
            hit = candle_cache.get(key)
        if hit and time.time() - hit[0] < 10:
            bars = hit[1]
        else:
            raw: list[list[str]] = []
            try:
                raw = market_client().candles(inst_id, bar, limit=limit)
                while len(raw) < limit and raw:
                    older = market_client().history_candles(
                        inst_id, bar, limit=limit - len(raw), after=raw[-1][0])
                    if not older:
                        break
                    raw += older
            except OkxError as exc:
                if not raw:
                    raise HTTPException(status_code=502, detail=f"OKX K线获取失败: {exc}") from None
            bars = [{"t": int(r[0]), "o": float(r[1]), "h": float(r[2]), "l": float(r[3]),
                     "c": float(r[4]), "v": float(r[5])} for r in reversed(raw)]
            with candle_lock:
                candle_cache[key] = (time.time(), bars)
        since = bars[0]["t"] if bars else 0
        trades = [t for t in store.list_trades(limit=2000)
                  if t["inst_id"] == inst_id and (t.get("closed_ms") or now_ms()) >= since]
        return {"inst_id": inst_id, "timeframe": timeframe, "bars": bars, "trades": trades}

    @app.get("/api/runner", dependencies=guard)
    def runner_get() -> dict[str, Any]:
        from pa_agent.data.okx_source import _TF_MAP
        from pa_agent.headless import RUNNER_ACTIVE_KEY, read_runner_config

        cfg = read_runner_config(store)
        active = _loads(store.get_kv(RUNNER_ACTIVE_KEY)) or None
        running = bool(active and now_ms() - int(active.get("ts_ms") or 0) < 120_000)
        return {"config": {"symbols": cfg[0], "timeframe": cfg[1]} if cfg else None,
                "active": active, "running": running, "timeframes": list(_TF_MAP)}

    @app.post("/api/runner", dependencies=guard)
    def runner_set(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        from pa_agent.data.okx_source import _TF_MAP
        from pa_agent.exchange.okx_rest import OkxError
        from pa_agent.headless import write_runner_config
        from pa_agent.trading.executor import resolve_inst_id

        timeframe = str(payload.get("timeframe") or "")
        if timeframe not in _TF_MAP:
            raise HTTPException(status_code=400, detail=f"不支持的周期 {timeframe!r}，可选 {list(_TF_MAP)}")
        raw = payload.get("symbols") or []
        if isinstance(raw, str):
            raw = raw.replace("，", ",").split(",")
        symbols: list[str] = []
        for item in raw:
            text = str(item).strip()
            if not text:
                continue
            inst = resolve_inst_id(text if "-" in text else f"{text.upper()}-USDT-SWAP")
            if not inst:
                raise HTTPException(status_code=400, detail=f"{text} 不是 OKX 永续合约")
            try:
                market_client().instrument(inst)
            except OkxError as exc:
                raise HTTPException(status_code=400, detail=f"OKX 上找不到 {inst}: {exc}") from None
            if inst not in symbols:
                symbols.append(inst)
        if not symbols:
            raise HTTPException(status_code=400, detail="至少填一个品种")
        if len(symbols) > 10:
            raise HTTPException(status_code=400, detail="最多 10 个品种（每根K线收盘都要逐个跑一次 AI 分析）")
        write_runner_config(store, symbols, timeframe)
        store.log_event("runner_config", f"面板保存分析配置：{','.join(symbols)} · {timeframe}")
        return runner_get()

    @app.get("/api/chart/{signal_id}", dependencies=guard)
    def chart(signal_id: int):
        sig = store.get_signal(signal_id)
        if not sig or not sig.get("chart_image"):
            raise HTTPException(status_code=404, detail="no chart")
        path = Path(sig["chart_image"])
        if not path.is_absolute():
            from pa_agent.config.paths import PROJECT_ROOT

            path = PROJECT_ROOT / path
        path = path.resolve()
        if chart_root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="no chart")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/kill", dependencies=guard)
    def kill() -> dict[str, Any]:
        store.set_kv(KILL_KEY, "1")
        store.set_kv(KILL_DONE_KEY, "0")
        store.log_event("kill", "面板触发紧急停止", level="warn")
        return {"kill_switch": True}

    @app.post("/api/resume", dependencies=guard)
    def resume() -> dict[str, Any]:
        store.set_kv(KILL_KEY, "0")
        store.set_kv(KILL_DONE_KEY, "0")
        store.log_event("resume", "面板解除紧急停止")
        return {"kill_switch": False}

    return app


def serve_in_thread(store: TradeStore, settings: Any, *, host: str, port: int) -> threading.Thread:
    import uvicorn

    t = settings.trading
    if host not in _LOOPBACK and not t.dashboard_token:
        raise SystemExit(
            f"面板监听 {host} 对外开放时必须设置 trading.dashboard_token（防止任何人都能看到账户和触发紧急停止）"
        )
    app = create_app(store, demo=t.demo, token=t.dashboard_token)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="dashboard", daemon=True)
    thread.start()
    logger.info("收益面板: http://%s:%s/%s", host, port,
                "?token=…" if t.dashboard_token else "")
    return thread
