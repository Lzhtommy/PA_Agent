"""Read-mostly web dashboard over the trading SQLite store (FastAPI)."""

import hmac
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from pa_agent.trading.executor import HEARTBEAT_KEY, KILL_DONE_KEY, KILL_KEY
from pa_agent.trading.stats import ai_quality, summarize
from pa_agent.trading.store import TradeStore

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _loads(text: str | None) -> Any:
    try:
        return json.loads(text or "{}")
    except (TypeError, ValueError):
        return {}


def create_app(store: TradeStore, *, demo: bool, token: str = "", chart_root: Path | None = None):
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse

    from pa_agent.config.paths import TRADE_RECORDS_DIR

    chart_root = (chart_root or TRADE_RECORDS_DIR).resolve()
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

    @app.get("/api/overview", dependencies=guard)
    def overview(days: int = 0) -> dict[str, Any]:
        since = int((time.time() - days * 86400) * 1000) if days else 0
        trades = store.list_trades(limit=100_000)
        if since:
            trades = [t for t in trades if (t.get("closed_ms") or t.get("created_ms") or 0) >= since]
        equity = store.equity_series(since_ms=since, demo=demo)
        hb = int(store.get_kv(HEARTBEAT_KEY, "0") or 0)
        latest = store.latest_equity()
        return {
            "demo": demo,
            "heartbeat_ms": hb or None,
            "kill_switch": store.get_kv(KILL_KEY) == "1",
            "equity": latest,
            "summary": summarize(trades, equity),
            "active": store.active_trades(),
        }

    @app.get("/api/equity", dependencies=guard)
    def equity(days: int = 0) -> list[dict[str, Any]]:
        since = int((time.time() - days * 86400) * 1000) if days else 0
        rows = store.equity_series(since_ms=since, demo=demo)
        # Downsample long series to ≤ 1500 points for the chart.
        step = max(1, len(rows) // 1500)
        return rows[::step] + ([rows[-1]] if rows and (len(rows) - 1) % step else [])

    @app.get("/api/trades", dependencies=guard)
    def trades(limit: int = 200, offset: int = 0, state: str = "") -> list[dict[str, Any]]:
        states = tuple(s for s in state.split(",") if s) or None
        rows = store.list_trades(states=states, limit=min(limit, 2000), offset=offset)
        sig_ids = {r["signal_id"] for r in rows if r.get("signal_id")}
        conf: dict[int, dict[str, Any]] = {}
        for sid in sig_ids:
            s = store.get_signal(sid)
            if s:
                conf[sid] = {"trade_confidence": s["trade_confidence"],
                             "estimated_win_rate": s["estimated_win_rate"],
                             "timeframe": s["timeframe"]}
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
    def signals(limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        return store.list_signals(limit=min(limit, 2000), offset=offset)

    @app.get("/api/signals/{signal_id}", dependencies=guard)
    def signal_detail(signal_id: int) -> dict[str, Any]:
        sig = store.get_signal(signal_id)
        if sig is None:
            raise HTTPException(status_code=404, detail="signal not found")
        return {**sig, "decision": _loads(sig.pop("decision_json")),
                "stage1": _loads(sig.pop("stage1_json"))}

    @app.get("/api/ai-quality", dependencies=guard)
    def quality() -> dict[str, Any]:
        trades = store.list_trades(limit=100_000)
        sigs = {r["id"]: r for r in store._rows("SELECT * FROM signals")}
        return ai_quality(trades, sigs, list(sigs.values()))

    @app.get("/api/events", dependencies=guard)
    def events(limit: int = 100) -> list[dict[str, Any]]:
        return store.events(limit=min(limit, 1000))

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
