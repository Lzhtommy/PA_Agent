"""High-level OKX broker operations used by the executor.

The executor only talks to :class:`Broker`; tests substitute a fake.
All sizes are in contracts, prices in quote currency (USDT).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from pa_agent.exchange.okx_rest import OkxError, OkxRestClient
from pa_agent.trading.risk import InstrumentSpec, fmt_num, round_to_tick

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderStatus:
    """Normalised order / algo state.

    ``state``: live | partially_filled | filled | canceled | triggered | failed | unknown
    """
    state: str
    filled: float = 0.0
    avg_px: float | None = None
    ord_id: str = ""
    raw: dict[str, Any] | None = None
    fill_ms: int | None = None    # exchange time of the (last) fill


@dataclass(frozen=True)
class StopInfo:
    """A live reduce-only conditional (stop-loss) algo order."""
    algo_id: str
    algo_cl_ord_id: str
    size: float | None      # None = closes the whole position (closeFraction)
    trigger_px: float | None


@dataclass(frozen=True)
class PositionInfo:
    size: float           # signed contracts (net mode): >0 long, <0 short, 0 flat
    avg_px: float | None
    upl: float | None


@dataclass(frozen=True)
class AccountInfo:
    total_eq: float
    avail_eq: float
    upl: float | None


@dataclass(frozen=True)
class ClosedPosition:
    open_avg_px: float | None
    close_avg_px: float | None
    realized_pnl: float      # net of fees and funding (OKX realizedPnl)
    pnl: float | None
    fee: float | None
    funding_fee: float | None
    open_ms: int
    close_ms: int


class Broker(Protocol):
    demo: bool

    def spec(self, inst_id: str) -> InstrumentSpec: ...
    def last_price(self, inst_id: str) -> float: ...
    def account(self) -> AccountInfo: ...
    def prepare_instrument(self, inst_id: str, leverage: int, margin_mode: str) -> None: ...
    def place_entry(self, *, inst_id: str, side: str, entry_type: str, size: float,
                    price: float | None, stop_loss: float, margin_mode: str,
                    cl_ord_id: str, sl_cl_ord_id: str) -> str: ...
    def entry_status(self, *, inst_id: str, entry_type: str, cl_ord_id: str,
                     algo_id: str = "") -> OrderStatus: ...
    def cancel_entry(self, *, inst_id: str, entry_type: str, cl_ord_id: str,
                     algo_id: str = "") -> None: ...
    def place_take_profit(self, *, inst_id: str, side: str, size: float, price: float,
                          margin_mode: str, cl_ord_id: str) -> None: ...
    def order_status(self, inst_id: str, cl_ord_id: str) -> OrderStatus: ...
    def cancel_order(self, inst_id: str, cl_ord_id: str) -> None: ...
    def stop_status(self, sl_cl_ord_id: str) -> OrderStatus: ...
    def pending_stops(self, inst_id: str) -> list[StopInfo]: ...
    def pending_reduce_orders(self, inst_id: str) -> list[str]: ...
    def cancel_order_id(self, inst_id: str, ord_id: str) -> None: ...
    def place_stop(self, *, inst_id: str, side: str, size: float, trigger_px: float,
                   margin_mode: str, sl_cl_ord_id: str) -> str: ...
    def cancel_stop(self, inst_id: str, algo_id: str) -> None: ...
    def position(self, inst_id: str) -> PositionInfo: ...
    def open_positions(self) -> list[dict[str, Any]]: ...
    def last_closed_position(self, inst_id: str, opened_after_ms: int) -> ClosedPosition | None: ...
    def close_position(self, inst_id: str, margin_mode: str) -> None: ...


def _f(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


_ORDER_STATE = {
    "live": "live",
    "partially_filled": "partially_filled",
    "filled": "filled",
    "canceled": "canceled",
    "mmp_canceled": "canceled",
}

_ALGO_STATE = {
    "live": "live",
    "pause": "live",
    "effective": "triggered",
    "partially_effective": "triggered",
    "canceled": "canceled",
    "order_failed": "failed",
    "partially_failed": "failed",
}


class OkxBroker:
    """:class:`Broker` implementation on OKX REST v5 (net position mode)."""

    def __init__(self, client: OkxRestClient) -> None:
        self._c = client
        self.demo = client.demo
        self._specs: dict[str, InstrumentSpec] = {}
        self._prepared: set[tuple[str, int, str]] = set()
        self._pos_mode_checked = False

    # ── market / account ──────────────────────────────────────────────────────

    def spec(self, inst_id: str) -> InstrumentSpec:
        if inst_id not in self._specs:
            self._specs[inst_id] = InstrumentSpec.from_okx(self._c.instrument(inst_id))
        return self._specs[inst_id]

    def last_price(self, inst_id: str) -> float:
        return float(self._c.ticker(inst_id)["last"])

    def account(self) -> AccountInfo:
        bal = self._c.balance()
        total = _f(bal.get("totalEq")) or 0.0
        avail = None
        upl = _f(bal.get("upl"))
        for d in bal.get("details") or []:
            if d.get("ccy") == "USDT":
                avail = _f(d.get("availEq"))
                if avail is None:
                    avail = _f(d.get("availBal"))
                if upl is None:
                    upl = _f(d.get("upl"))
        return AccountInfo(total_eq=total, avail_eq=avail if avail is not None else total, upl=upl)

    def prepare_instrument(self, inst_id: str, leverage: int, margin_mode: str) -> None:
        if not self._pos_mode_checked:
            cfg = self._c.account_config()
            if cfg.get("posMode") != "net_mode":
                try:
                    self._c.set_position_mode("net_mode")
                    logger.info("OKX position mode switched to net_mode")
                except OkxError as exc:
                    raise OkxError(
                        exc.code,
                        "账户为双向持仓模式且无法自动切换为单向持仓（请先平掉所有仓位）: " + exc.msg,
                    ) from exc
            self._pos_mode_checked = True
        key = (inst_id, int(leverage), margin_mode)
        if key not in self._prepared:
            self._c.set_leverage(inst_id, int(leverage), margin_mode)
            self._prepared.add(key)

    # ── entry ─────────────────────────────────────────────────────────────────

    def place_entry(self, *, inst_id: str, side: str, entry_type: str, size: float,
                    price: float | None, stop_loss: float, margin_mode: str,
                    cl_ord_id: str, sl_cl_ord_id: str) -> str:
        tick = self.spec(inst_id).tick_sz
        attach = [{
            "attachAlgoClOrdId": sl_cl_ord_id,
            "slTriggerPx": fmt_num(round_to_tick(stop_loss, tick)),
            "slOrdPx": "-1",
            "slTriggerPxType": "last",
        }]
        if entry_type == "stop":
            assert price is not None
            row = self._c.place_algo_order({
                "instId": inst_id,
                "tdMode": margin_mode,
                "side": side,
                "ordType": "trigger",
                "sz": fmt_num(size),
                "triggerPx": fmt_num(round_to_tick(price, tick)),
                "triggerPxType": "last",
                "orderPx": "-1",
                "algoClOrdId": cl_ord_id,
                "attachAlgoOrds": attach,
            })
            return str(row.get("algoId") or "")
        body: dict[str, Any] = {
            "instId": inst_id,
            "tdMode": margin_mode,
            "side": side,
            "ordType": "limit" if entry_type == "limit" else "market",
            "sz": fmt_num(size),
            "clOrdId": cl_ord_id,
            "attachAlgoOrds": attach,
        }
        if entry_type == "limit":
            assert price is not None
            body["px"] = fmt_num(round_to_tick(price, tick))
        row = self._c.place_order(body)
        return str(row.get("ordId") or "")

    def entry_status(self, *, inst_id: str, entry_type: str, cl_ord_id: str,
                     algo_id: str = "") -> OrderStatus:
        if entry_type != "stop":
            return self.order_status(inst_id, cl_ord_id)
        algo = self._c.get_algo_order(algo_id=algo_id or None,
                                      algo_cl_ord_id=None if algo_id else cl_ord_id)
        state = _ALGO_STATE.get(str(algo.get("state")), "unknown")
        if state != "triggered":
            return OrderStatus(state=state, raw=algo)
        ord_id = str(algo.get("ordId") or "")
        if not ord_id and algo.get("ordIdList"):
            ord_id = str(algo["ordIdList"][0])
        if not ord_id:
            return OrderStatus(state="live", raw=algo)
        order = self._c.get_order(inst_id, ord_id=ord_id)
        st = self._order_status_from_row(order)
        return OrderStatus(state=st.state, filled=st.filled, avg_px=st.avg_px,
                           ord_id=ord_id, raw=order, fill_ms=st.fill_ms)

    def cancel_entry(self, *, inst_id: str, entry_type: str, cl_ord_id: str,
                     algo_id: str = "") -> None:
        if entry_type == "stop":
            if not algo_id:
                algo_id = str(self._c.get_algo_order(algo_cl_ord_id=cl_ord_id).get("algoId") or "")
            self._c.cancel_algo_orders([{"algoId": algo_id, "instId": inst_id}])
        else:
            self._c.cancel_order(inst_id, cl_ord_id=cl_ord_id)

    # ── exits ─────────────────────────────────────────────────────────────────

    def place_take_profit(self, *, inst_id: str, side: str, size: float, price: float,
                          margin_mode: str, cl_ord_id: str) -> None:
        tick = self.spec(inst_id).tick_sz
        self._c.place_order({
            "instId": inst_id,
            "tdMode": margin_mode,
            "side": side,
            "ordType": "limit",
            "sz": fmt_num(size),
            "px": fmt_num(round_to_tick(price, tick)),
            "reduceOnly": True,
            "clOrdId": cl_ord_id,
        })

    @staticmethod
    def _order_status_from_row(row: dict[str, Any]) -> OrderStatus:
        return OrderStatus(
            state=_ORDER_STATE.get(str(row.get("state")), "unknown"),
            filled=_f(row.get("accFillSz")) or 0.0,
            avg_px=_f(row.get("avgPx")),
            ord_id=str(row.get("ordId") or ""),
            raw=row,
            fill_ms=int(_f(row.get("fillTime")) or _f(row.get("uTime")) or 0) or None,
        )

    def order_status(self, inst_id: str, cl_ord_id: str) -> OrderStatus:
        return self._order_status_from_row(self._c.get_order(inst_id, cl_ord_id=cl_ord_id))

    def cancel_order(self, inst_id: str, cl_ord_id: str) -> None:
        self._c.cancel_order(inst_id, cl_ord_id=cl_ord_id)

    def stop_status(self, sl_cl_ord_id: str) -> OrderStatus:
        algo = self._c.get_algo_order(algo_cl_ord_id=sl_cl_ord_id)
        return OrderStatus(
            state=_ALGO_STATE.get(str(algo.get("state")), "unknown"),
            ord_id=str(algo.get("algoId") or ""),
            raw=algo,
        )

    def pending_stops(self, inst_id: str) -> list[StopInfo]:
        out = []
        for row in self._c.pending_algo_orders("conditional", inst_id):
            out.append(StopInfo(
                algo_id=str(row.get("algoId") or ""),
                algo_cl_ord_id=str(row.get("algoClOrdId") or ""),
                size=_f(row.get("sz")),
                trigger_px=_f(row.get("slTriggerPx")),
            ))
        return out

    def pending_reduce_orders(self, inst_id: str) -> list[str]:
        """ordIds of resting reduce-only orders (take-profits) on *inst_id*."""
        return [
            str(r.get("ordId"))
            for r in self._c.pending_orders(inst_id)
            if str(r.get("reduceOnly")).lower() == "true"
        ]

    def cancel_order_id(self, inst_id: str, ord_id: str) -> None:
        self._c.cancel_order(inst_id, ord_id=ord_id)

    def place_stop(self, *, inst_id: str, side: str, size: float, trigger_px: float,
                   margin_mode: str, sl_cl_ord_id: str) -> str:
        tick = self.spec(inst_id).tick_sz
        row = self._c.place_algo_order({
            "instId": inst_id,
            "tdMode": margin_mode,
            "side": side,
            "ordType": "conditional",
            "sz": fmt_num(size),
            "slTriggerPx": fmt_num(round_to_tick(trigger_px, tick)),
            "slOrdPx": "-1",
            "slTriggerPxType": "last",
            "reduceOnly": True,
            "algoClOrdId": sl_cl_ord_id,
        })
        return str(row.get("algoId") or "")

    def cancel_stop(self, inst_id: str, algo_id: str) -> None:
        self._c.cancel_algo_orders([{"algoId": algo_id, "instId": inst_id}])

    # ── positions ─────────────────────────────────────────────────────────────

    def position(self, inst_id: str) -> PositionInfo:
        total = 0.0
        avg = upl = None
        for row in self._c.positions(inst_id):
            size = _f(row.get("pos")) or 0.0
            if row.get("posSide") == "short":
                size = -abs(size)
            total += size
            if size:
                avg = _f(row.get("avgPx"))
                upl = _f(row.get("upl"))
        return PositionInfo(size=total, avg_px=avg, upl=upl)

    def open_positions(self) -> list[dict[str, Any]]:
        return [r for r in self._c.positions() if (_f(r.get("pos")) or 0) != 0]

    def last_closed_position(self, inst_id: str, opened_after_ms: int) -> ClosedPosition | None:
        for row in self._c.positions_history(inst_id, limit=20):
            c_time = int(_f(row.get("cTime")) or 0)
            if c_time < opened_after_ms - 120_000:
                continue
            # type: 1 partial close, 2 close all, 3 liquidation, 4 partial liq, 5 ADL
            if str(row.get("type") or "2") not in ("2", "3", "4", "5"):
                continue
            return ClosedPosition(
                open_avg_px=_f(row.get("openAvgPx")),
                close_avg_px=_f(row.get("closeAvgPx")),
                realized_pnl=_f(row.get("realizedPnl")) or 0.0,
                pnl=_f(row.get("pnl")),
                fee=_f(row.get("fee")),
                funding_fee=_f(row.get("fundingFee")),
                open_ms=c_time,
                close_ms=int(_f(row.get("uTime")) or 0),
            )
        return None

    def close_position(self, inst_id: str, margin_mode: str) -> None:
        self._c.close_position(inst_id, margin_mode)
