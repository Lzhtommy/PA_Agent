"""Order lifecycle engine: signals → risk → OKX orders → fills → PnL.

Single-threaded; :meth:`Executor.tick` is called every ``poll_interval_s``.

Trade states::

    pending  entry order resting on the exchange (or watched locally for breakout)
    open     entry (partly) filled; stop-loss + take-profit orders working
    closed   position flat; PnL settled from OKX positions-history
    canceled entry never filled (expired / superseded / no longer supported)
    rejected failed a risk rule before any order was sent
    error    exchange refused the entry order (nothing live)

Policy (first version, deliberately conservative):
- One trade per instrument. Signals never add to or reverse an open position —
  exits happen only through the stop / take-profit orders.
- The latest analysis supersedes a resting entry: a different plan replaces it,
  a 不下单 decision leaves it to expire.

Safety invariants:
- A trade only leaves ``pending`` once the exchange confirms the entry is
  cancelled or filled — an unconfirmed cancel is retried every tick.
- Ambiguous POST outcomes (timeouts, duplicate client ids) are resolved by
  looking the order up by its client id, never by assuming failure.
- While a position exists, every tick reconciles the exchange's stop orders for
  that instrument to exactly one stop of the right size and price.
"""
from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pa_agent.data.bar_close_wait import timeframe_to_seconds
from pa_agent.exchange.okx_rest import OkxError, is_ambiguous
from pa_agent.trading.broker import Broker, OrderStatus
from pa_agent.trading.risk import (
    EntryPlan,
    RiskConfig,
    RiskRejected,
    build_entry_plan,
    daily_loss_blocked,
    round_to_tick,
    size_position,
    split_take_profit,
)
from pa_agent.trading.store import TradeStore, now_ms

logger = logging.getLogger(__name__)

KILL_KEY = "kill_switch"              # "1" = halt: flatten everything, accept no signals
KILL_DONE_KEY = "kill_switch_done"    # "1" once the kill was announced in the event log
HEARTBEAT_KEY = "executor_heartbeat_ms"
EQUITY_INTERVAL_MS = 60_000
RECONCILE_INTERVAL_MS = 60_000
PNL_LAG_GRACE_MS = 120_000


@dataclass
class ExecutorConfig:
    margin_mode: str = "isolated"
    leverage: int = 5
    pending_expiry_bars: int = 3
    move_stop_to_breakeven: bool = True
    breakout_mode: str = "trigger"   # trigger | watch
    risk: RiskConfig = field(default_factory=RiskConfig)

    @classmethod
    def from_settings(cls, t: Any) -> ExecutorConfig:
        return cls(
            margin_mode=t.margin_mode,
            leverage=int(t.leverage),
            pending_expiry_bars=int(t.pending_expiry_bars),
            move_stop_to_breakeven=bool(t.move_stop_to_breakeven),
            breakout_mode=t.breakout_mode,
            risk=RiskConfig.from_settings(t),
        )


def resolve_inst_id(symbol: str) -> str | None:
    """Map a signal symbol to an OKX swap instId; None when not a crypto pair."""
    from pa_agent.data.market_defaults import is_likely_crypto_symbol
    from pa_agent.data.okx_source import normalize_okx_inst_id

    s = (symbol or "").strip()
    if not s:
        return None
    if s.upper().endswith("-SWAP") or is_likely_crypto_symbol(s):
        return normalize_okx_inst_id(s)
    return None


def _cid(prefix: str) -> str:
    """OKX client order id: alphanumeric, ≤ 32 chars."""
    return f"pa{prefix}{int(time.time())}{secrets.token_hex(3)}"[:32]


def _close_side(direction: str) -> str:
    return "sell" if direction == "long" else "buy"


def _open_side(direction: str) -> str:
    return "buy" if direction == "long" else "sell"


def _sign(direction: str) -> int:
    return 1 if direction == "long" else -1


def _local_day_start_ms() -> int:
    now = datetime.now()
    return int(datetime(now.year, now.month, now.day).timestamp() * 1000)


def _px(price: Any) -> str:
    """Price without float noise (83098.09999999999 → 83098.1)."""
    if price is None:
        return "—"
    return f"{float(price):.8f}".rstrip("0").rstrip(".")


def _usdt(contracts: Any, ct_val: Any, price: Any) -> str:
    """Contract count → approximate USDT notional, e.g. ``18,889U``."""
    try:
        return f"{float(contracts) * float(ct_val) * float(price):,.0f}U"
    except (TypeError, ValueError):
        return f"{contracts} 张"


def _is_watching(trade: dict[str, Any]) -> bool:
    """Breakout entry watched locally — nothing has been sent to the exchange yet."""
    return trade["entry_type"] == "stop" and not trade["entry_algo_id"] and not trade["entry_ord_id"]


class Executor:
    def __init__(self, store: TradeStore, broker: Broker, cfg: ExecutorConfig) -> None:
        self.store = store
        self.broker = broker
        self.cfg = cfg
        self._kill = False
        self._last_equity_ms = 0
        self._last_reconcile_ms = 0

    # ── main loop ─────────────────────────────────────────────────────────────

    def tick(self) -> None:
        self.store.set_kv(HEARTBEAT_KEY, str(now_ms()))
        self._kill = self.store.get_kv(KILL_KEY) == "1"
        if self._kill:
            self._kill_step()
        else:
            self.store.set_kv(KILL_DONE_KEY, "0")
        # Manage existing trades before taking new signals so a bad signal can
        # never starve stop / exit management.
        for trade in self.store.active_trades():
            try:
                self._manage_trade(trade)
            except Exception as exc:
                self._event(trade["id"], "manage_error", f"管理订单失败（下轮重试）: {exc}",
                            level="warn")
        if self._kill:
            for sig in self.store.new_signals():
                self.store.mark_signal(sig["id"], "skipped", "紧急停止中，不接收新信号")
        else:
            self._process_signals()
        t = now_ms()
        if t - self._last_equity_ms >= EQUITY_INTERVAL_MS:
            self._snapshot_equity()
            self._last_equity_ms = t
        if t - self._last_reconcile_ms >= RECONCILE_INTERVAL_MS:
            self._reconcile()
            self._last_reconcile_ms = t

    def _event(self, trade_id: int | None, kind: str, msg: str, *, level: str = "info",
               signal_id: int | None = None) -> None:
        log = logger.warning if level in ("warn", "error") else logger.info
        log("[trade %s] %s: %s", trade_id, kind, msg)
        self.store.log_event(kind, msg, level=level, trade_id=trade_id, signal_id=signal_id)

    # ── signals ───────────────────────────────────────────────────────────────

    def _process_signals(self) -> None:
        signals = self.store.new_signals()
        # Only the newest signal per instrument matters; older ones are superseded.
        latest: dict[str, dict[str, Any]] = {}
        for sig in signals:
            inst = resolve_inst_id(sig["symbol"])
            if inst is None:
                self.store.mark_signal(sig["id"], "rejected",
                                       f"{sig['symbol']} 不是 OKX 可交易的加密货币品种")
                continue
            prev = latest.get(inst)
            if prev is not None:
                self.store.mark_signal(prev["id"], "skipped", f"被更新的信号 #{sig['id']} 取代")
            latest[inst] = {**sig, "inst_id": inst}
        for sig in latest.values():
            try:
                self._handle_signal(sig)
            except Exception as exc:
                self.store.mark_signal(sig["id"], "error", str(exc))
                self._event(None, "signal_error", f"信号 #{sig['id']} 处理失败: {exc}",
                            level="error", signal_id=sig["id"])

    def _handle_signal(self, sig: dict[str, Any]) -> None:
        inst = sig["inst_id"]
        sid = sig["id"]
        active = self.store.active_trades(inst)
        open_trades = [t for t in active if t["state"] == "open"]
        pending = [t for t in active if t["state"] == "pending"]

        if sig["order_type"] == "none":
            # "不下单" means "no new entry now", not "the resting one is wrong": the AI
            # is not shown its pending orders. Keep them until expiry, stop touched
            # before fill, or an opposite-direction plan.
            keep = "，保留挂单 " + "、".join(f"#{t['id']}" for t in pending) if pending else ""
            self.store.mark_signal(sid, "processed", "不下单" + keep)
            return

        if open_trades:
            self.store.mark_signal(
                sid, "skipped", f"{inst} 已有持仓（交易 #{open_trades[0]['id']}），不加仓不反手"
            )
            return

        ref = self.broker.last_price(inst)
        try:
            plan = build_entry_plan(sig, ref, self.cfg.risk)
        except RiskRejected as exc:
            # A rejected opposite-direction plan still means the AI flipped its view.
            for t in pending:
                if t["direction"] != sig["direction"]:
                    self._cancel_pending(t, f"信号 #{sid} 转为反方向（新信号未通过风控）")
            self.store.mark_signal(sid, "rejected", str(exc))
            self._event(None, "risk_reject", f"信号 #{sid}: {exc}", signal_id=sid)
            return

        spec = self.broker.spec(inst)
        for t in pending:
            same = (
                t["direction"] == plan.direction
                and t["entry_type"] == plan.entry_type
                and abs((t["planned_entry"] or 0) - plan.entry) <= spec.tick_sz
                and abs((t["stop_loss"] or 0) - plan.stop_loss) <= spec.tick_sz
            )
            if same:
                self.store.mark_signal(sid, "skipped", f"与挂单中的交易 #{t['id']} 方案相同，继续挂单")
                return
        for t in pending:
            self._cancel_pending(t, f"被新信号 #{sid} 替换")

        # A cancel may have promoted a partial fill to an open position, or not
        # be confirmed yet — never stack a second entry on top of either.
        still = self.store.active_trades(inst)
        if any(t["state"] == "open" for t in still):
            self.store.mark_signal(sid, "skipped", "旧挂单撤销时已部分成交，转为持仓管理，不再开新仓")
            return
        if still:
            self.store.mark_signal(sid, "skipped", "旧挂单撤销尚未确认，本信号跳过（下轮分析会重新给出方案）")
            return

        blocked, why = self._account_gates(exclude_inst=inst)
        if blocked:
            self.store.mark_signal(sid, "skipped", why)
            self._event(None, "risk_block", f"信号 #{sid}: {why}", signal_id=sid)
            return

        self._open_trade(sig, plan, spec)

    def _account_gates(self, *, exclude_inst: str) -> tuple[bool, str]:
        others = [t for t in self.store.active_trades() if t["inst_id"] != exclude_inst]
        if len(others) >= self.cfg.risk.max_open_positions:
            return True, f"已达最大同时持仓/挂单数 {self.cfg.risk.max_open_positions}"
        day_start = _local_day_start_ms()
        realized = sum((t["net_pnl"] or 0.0) for t in self.store.closed_trades_since(day_start))
        first = self.store.first_equity_since(day_start)
        if first is not None:
            equity_start = float(first["total_eq"])
        else:
            equity_start = self.broker.account().total_eq
        return daily_loss_blocked(realized, equity_start, self.cfg.risk)

    def _sweep_orphans(self, inst: str) -> None:
        """Cancel leftover reduce-only stops / TPs on an instrument we hold nothing in."""
        for stop in self.broker.pending_stops(inst):
            self.broker.cancel_stop(inst, stop.algo_id)
            self._event(None, "orphan_cancel", f"撤销 {inst} 遗留的止损单 {stop.algo_cl_ord_id or stop.algo_id}")
        for ord_id in self.broker.pending_reduce_orders(inst):
            self.broker.cancel_order_id(inst, ord_id)
            self._event(None, "orphan_cancel", f"撤销 {inst} 遗留的只减仓挂单 {ord_id}")

    def _open_trade(self, sig: dict[str, Any], plan: EntryPlan, spec) -> None:
        inst = sig["inst_id"]
        sid = sig["id"]
        self._sweep_orphans(inst)
        self.broker.prepare_instrument(inst, self.cfg.leverage, self.cfg.margin_mode)
        acct = self.broker.account()
        try:
            contracts, risk_amount = size_position(
                equity=acct.total_eq, avail=acct.avail_eq, plan=plan, spec=spec, cfg=self.cfg.risk
            )
        except RiskRejected as exc:
            self.store.mark_signal(sid, "rejected", str(exc))
            self._event(None, "risk_reject", f"信号 #{sid}: {exc}", signal_id=sid)
            return

        tf_s = timeframe_to_seconds(sig["timeframe"]) or 3600
        expires = None
        if plan.entry_type != "market":
            expires = now_ms() + self.cfg.pending_expiry_bars * tf_s * 1000
        trade_id = self.store.insert_trade({
            "signal_id": sid,
            "inst_id": inst,
            "timeframe": sig["timeframe"],
            "direction": plan.direction,
            "entry_type": plan.entry_type,
            "state": "pending",
            "demo": int(self.broker.demo),
            "planned_entry": plan.entry,
            "stop_loss": plan.stop_loss,
            "tp1": plan.tp1,
            "tp2": plan.tp2,
            "contracts": contracts,
            "ct_val": spec.ct_val,
            "risk_amount": risk_amount,
            "equity_at_entry": acct.total_eq,
            "expires_ms": expires,
            "entry_cl_ord_id": "",
            "sl_algo_cl_ord_id": "",
        })
        self.store.update_trade(trade_id, entry_cl_ord_id=_cid(f"{trade_id}e"),
                                sl_algo_cl_ord_id=_cid(f"{trade_id}s"))
        trade = self.store.get_trade(trade_id)
        assert trade is not None
        note = f"（{plan.note}）" if plan.note else ""
        kind = {"limit": "限价", "stop": "突破", "market": "市价"}[plan.entry_type]
        desc = (
            f"{'做多' if plan.direction == 'long' else '做空'} {inst}"
            f" {_usdt(contracts, spec.ct_val, plan.entry)} {kind}"
            f" 入场 {_px(plan.entry)} 止损 {_px(plan.stop_loss)} 止盈 {_px(plan.tp1)}"
            f"{' / ' + _px(plan.tp2) if plan.tp2 else ''} 风险 {risk_amount:.2f}U"
            f" 盈亏比 {plan.reward_risk:.2f}{note}"
        )
        if plan.entry_type == "stop" and self.cfg.breakout_mode == "watch":
            self.store.mark_signal(sid, "processed", f"交易 #{trade_id}")
            self._event(trade_id, "entry_watch", "本地盯盘等待突破: " + desc, signal_id=sid)
            return
        try:
            self._send_entry(trade, plan.entry_type,
                             None if plan.entry_type == "market" else plan.entry)
        except OkxError as exc:
            self.store.update_trade(trade_id, state="error", close_reason=f"下单失败: {exc}",
                                    closed_ms=now_ms())
            self.store.mark_signal(sid, "error", f"下单失败: {exc}")
            self._event(trade_id, "entry_error", f"下单失败: {exc}", level="error", signal_id=sid)
            return
        self.store.mark_signal(sid, "processed", f"交易 #{trade_id}")
        self._event(trade_id, "entry_placed", "已下单: " + desc, signal_id=sid)
        if plan.entry_type == "market":
            # Market entries fill at once — place exits now instead of next tick.
            self._manage_pending(self.store.get_trade(trade_id) or trade)

    def _send_entry(self, trade: dict[str, Any], entry_type: str, price: float | None) -> None:
        """Place the entry; resolve ambiguous outcomes by client-id lookup. Raises OkxError."""
        tid = trade["id"]
        try:
            exch_id = self.broker.place_entry(
                inst_id=trade["inst_id"], side=_open_side(trade["direction"]),
                entry_type=entry_type, size=trade["contracts"], price=price,
                stop_loss=trade["stop_loss"], margin_mode=self.cfg.margin_mode,
                cl_ord_id=trade["entry_cl_ord_id"], sl_cl_ord_id=trade["sl_algo_cl_ord_id"],
            )
        except OkxError as exc:
            if not is_ambiguous(exc):
                raise
            status = self._lookup_entry(trade, entry_type)
            if status is None or status.state == "unknown":
                raise
            exch_id = status.ord_id
            self._event(tid, "entry_ambiguous", f"下单返回不明确（{exc}），按订单号查到已在交易所: {status.state}",
                        level="warn")
        fields: dict[str, Any] = {"entry_type": entry_type}
        if entry_type == "stop":
            fields["entry_algo_id"] = exch_id or "pending-lookup"
        else:
            fields["entry_ord_id"] = exch_id or "pending-lookup"
        self.store.update_trade(tid, **fields)

    def _lookup_entry(self, trade: dict[str, Any], entry_type: str | None = None) -> OrderStatus | None:
        entry_type = entry_type or trade["entry_type"]
        algo_id = trade["entry_algo_id"] if trade["entry_algo_id"] != "pending-lookup" else ""
        try:
            return self.broker.entry_status(
                inst_id=trade["inst_id"], entry_type=entry_type,
                cl_ord_id=trade["entry_cl_ord_id"], algo_id=algo_id,
            )
        except OkxError as exc:
            logger.info("entry lookup %s: %s", trade["id"], exc)
            return None

    # ── pending entries ───────────────────────────────────────────────────────

    def _cancel_pending(self, trade: dict[str, Any], reason: str) -> bool:
        """Cancel a resting entry. Returns True once resolved (canceled or promoted).

        If the exchange does not confirm, the trade stays ``pending`` with an
        immediate expiry so the cancel is retried next tick.
        """
        tid = trade["id"]
        if _is_watching(trade):
            self.store.update_trade(tid, state="canceled", close_reason=reason, closed_ms=now_ms())
            self._event(tid, "entry_canceled", reason)
            return True
        try:
            self.broker.cancel_entry(
                inst_id=trade["inst_id"], entry_type=trade["entry_type"],
                cl_ord_id=trade["entry_cl_ord_id"],
                algo_id=trade["entry_algo_id"] if trade["entry_algo_id"] != "pending-lookup" else "",
            )
        except OkxError as exc:
            logger.info("cancel entry %s: %s", tid, exc)
        status = self._lookup_entry(trade)
        if status is not None and status.filled > 0 and status.state in ("filled", "canceled", "failed"):
            self._event(tid, "entry_partial", f"撤单时已成交 {_usdt(status.filled, trade['ct_val'], status.avg_px or trade['planned_entry'])}，转为持仓管理")
            self._on_entry_filled(trade, status)
            return True
        if status is not None and status.state in ("canceled", "failed") and status.filled == 0:
            self.store.update_trade(tid, state="canceled", close_reason=reason, closed_ms=now_ms())
            self._event(tid, "entry_canceled", reason)
            return True
        # live / partially_filled / unknown / lookup failed → retry next tick
        self.store.update_trade(tid, expires_ms=0, close_reason=reason)
        self._event(tid, "cancel_pending", f"撤单未确认（{status.state if status else '查询失败'}），下轮重试",
                    level="warn")
        return False

    def _manage_pending(self, trade: dict[str, Any]) -> None:
        tid = trade["id"]
        expired = trade["expires_ms"] is not None and now_ms() >= int(trade["expires_ms"])
        expiry_reason = trade["close_reason"] or f"{self.cfg.pending_expiry_bars} 根K线内未成交，过期取消"

        if _is_watching(trade):
            if expired or self._kill:
                self._cancel_pending(trade, "紧急停止" if self._kill else expiry_reason)
                return
            last = self.broker.last_price(trade["inst_id"])
            sign = _sign(trade["direction"])
            if (last - trade["planned_entry"]) * sign < 0:
                return
            if (last - trade["stop_loss"]) * sign <= 0:
                self._cancel_pending(trade, f"现价 {last} 已越过止损，突破单取消")
                return
            self._send_entry(trade, "market", None)
            self._event(tid, "entry_triggered", f"价格 {last} 触发突破，市价入场")
            return

        status = self._lookup_entry(trade)
        if status is None:
            return
        if status.state == "filled":
            self._on_entry_filled(trade, status)
            return
        if status.state == "live" and not status.filled and not expired:
            # Breakout entries can see price run to the stop without triggering;
            # the setup is then invalid even though the order is still resting.
            try:
                last = self.broker.last_price(trade["inst_id"])
            except OkxError:
                last = None
            if last is not None and (last - trade["stop_loss"]) * _sign(trade["direction"]) <= 0:
                self._cancel_pending(trade, f"未成交前价格 {_px(last)} 已触及止损 {_px(trade['stop_loss'])}，入场失效")
                return
        if status.state in ("canceled", "failed"):
            if status.filled > 0:
                self._on_entry_filled(trade, status)
            else:
                self.store.update_trade(tid, state="canceled", closed_ms=now_ms(),
                                        close_reason=trade["close_reason"]
                                        or f"入场单在交易所被取消（{status.state}）")
                self._event(tid, "entry_canceled", f"入场单在交易所状态为 {status.state}", level="warn")
            return
        if status.state == "partially_filled" and status.filled > 0:
            # OKX arms the attached stop only after a *complete* fill — protect
            # the filled part ourselves meanwhile.
            if status.filled != (trade["filled_contracts"] or 0):
                self.store.update_trade(tid, filled_contracts=status.filled,
                                        opened_ms=trade["opened_ms"] or status.fill_ms or now_ms())
                self._event(tid, "entry_partial", f"入场部分成交 {_usdt(status.filled, trade['ct_val'], status.avg_px or trade['planned_entry'])}，先为已成交部分挂止损")
            self._sync_stop(self.store.get_trade(tid) or trade, status.filled)
        if expired:
            self._cancel_pending(trade, expiry_reason)

    # ── fills & exits ─────────────────────────────────────────────────────────

    def _on_entry_filled(self, trade: dict[str, Any], status: OrderStatus) -> None:
        tid = trade["id"]
        filled = float(status.filled or trade["contracts"])
        avg = status.avg_px or trade["planned_entry"]
        # A partial fill already opened the position; OKX positions-history dates it
        # from that first fill, and _finalize matches on it.
        opened = trade["opened_ms"] or status.fill_ms or now_ms()
        self.store.update_trade(
            tid, state="open", filled_contracts=filled, remaining_contracts=filled,
            avg_entry=avg, opened_ms=opened, expires_ms=None, close_reason="",
        )
        self._event(tid, "entry_filled", f"入场成交 {_usdt(filled, trade['ct_val'], avg)} @ {_px(avg)}")
        trade = self.store.get_trade(tid) or trade
        self._sync_stop(trade, filled)
        self._place_take_profits(self.store.get_trade(tid) or trade, filled)

    def _place_take_profits(self, trade: dict[str, Any], filled: float) -> None:
        """Place whichever take-profit orders are still missing (idempotent)."""
        tid = trade["id"]
        inst = trade["inst_id"]
        spec = self.broker.spec(inst)
        plan = EntryPlan(direction=trade["direction"], entry_type=trade["entry_type"],
                         entry=trade["avg_entry"] or trade["planned_entry"],
                         stop_loss=trade["stop_loss"], tp1=trade["tp1"], tp2=trade["tp2"],
                         reward_risk=0)
        tp1_sz, tp2_sz = split_take_profit(filled, plan, spec, self.cfg.risk.tp1_close_fraction)
        legs = [("tp1_cl_ord_id", "a", tp1_sz, trade["tp1"], "止盈1")]
        if tp2_sz > 0 and trade["tp2"]:
            legs.append(("tp2_cl_ord_id", "b", tp2_sz, trade["tp2"], "止盈2"))
        placed = []
        for field, tag, size, price, label in legs:
            if trade[field]:
                continue
            cid = _cid(f"{tid}{tag}")
            try:
                self.broker.place_take_profit(inst_id=inst, side=_close_side(trade["direction"]),
                                              size=size, price=price,
                                              margin_mode=self.cfg.margin_mode, cl_ord_id=cid)
            except OkxError as exc:
                if not (is_ambiguous(exc) and self._order_exists(inst, cid)):
                    self._event(tid, "tp_error", f"{label}挂单失败（下轮重试）: {exc}", level="warn")
                    continue
            self.store.update_trade(tid, **{field: cid})
            placed.append(f"{label} {_usdt(size, trade['ct_val'], price)} @ {_px(price)}")
        if placed:
            self._event(tid, "exits_placed", "，".join(placed))

    def _order_exists(self, inst: str, cid: str) -> bool:
        try:
            return self.broker.order_status(inst, cid).state != "unknown"
        except OkxError:
            return False

    def _sync_stop(self, trade: dict[str, Any], size: float) -> bool:
        """Leave exactly one live stop of *size* at ``trade.stop_loss`` on the instrument.

        Adopts a matching stop (e.g. the attached SL armed by OKX), cancels every
        other conditional stop on the instrument, and places one if none match.
        If no stop can be placed the position is flattened. Returns True when protected.
        """
        tid = trade["id"]
        inst = trade["inst_id"]
        spec = self.broker.spec(inst)
        target = round_to_tick(float(trade["stop_loss"]), spec.tick_sz)
        stops = self.broker.pending_stops(inst)

        def matches(s) -> bool:
            size_ok = s.size is None or abs(s.size - size) < spec.lot_sz / 2
            px_ok = s.trigger_px is not None and abs(s.trigger_px - target) < spec.tick_sz / 2
            return size_ok and px_ok

        keeper = next((s for s in stops if s.algo_cl_ord_id == trade["sl_algo_cl_ord_id"] and matches(s)),
                      None) or next((s for s in stops if matches(s)), None)
        for s in stops:
            if keeper is not None and s.algo_id == keeper.algo_id:
                continue
            try:
                self.broker.cancel_stop(inst, s.algo_id)
                self._event(tid, "stop_cancel",
                            f"撤销多余/过期的止损单 {_usdt(s.size, trade['ct_val'], s.trigger_px)} @ {_px(s.trigger_px)}")
            except OkxError as exc:
                logger.info("cancel stale stop %s: %s", s.algo_id, exc)
        if keeper is not None:
            if keeper.algo_cl_ord_id != trade["sl_algo_cl_ord_id"] or keeper.algo_id != trade["sl_algo_id"]:
                self.store.update_trade(tid, sl_algo_cl_ord_id=keeper.algo_cl_ord_id,
                                        sl_algo_id=keeper.algo_id)
            return True

        cid = _cid(f"{tid}s")
        try:
            algo_id = self.broker.place_stop(
                inst_id=inst, side=_close_side(trade["direction"]), size=size,
                trigger_px=target, margin_mode=self.cfg.margin_mode, sl_cl_ord_id=cid,
            )
        except OkxError as exc:
            if is_ambiguous(exc):
                try:
                    st = self.broker.stop_status(cid)
                    if st.state == "live":
                        self.store.update_trade(tid, sl_algo_cl_ord_id=cid, sl_algo_id=st.ord_id)
                        return True
                except OkxError:
                    pass
            # Unprotected position: flatten immediately rather than run naked.
            self._event(tid, "stop_error", f"止损单下单失败，立即市价平仓: {exc}", level="error")
            self.broker.close_position(inst, self.cfg.margin_mode)
            self.store.update_trade(tid, close_reason="止损单无法下达，已强制平仓")
            return False
        self.store.update_trade(tid, sl_algo_cl_ord_id=cid, sl_algo_id=algo_id)
        self._event(tid, "stop_placed", f"止损单 {_usdt(size, trade['ct_val'], target)} @ {_px(target)}")
        return True

    def _manage_open(self, trade: dict[str, Any]) -> None:
        tid = trade["id"]
        inst = trade["inst_id"]
        pos = self.broker.position(inst)
        if pos.size == 0:
            self._finalize(trade)
            return
        size = abs(pos.size)

        if trade["tp1_cl_ord_id"] and not trade["tp1_filled"]:
            try:
                tp1 = self.broker.order_status(inst, trade["tp1_cl_ord_id"])
            except OkxError:
                tp1 = None
            if tp1 is not None and tp1.state == "filled":
                self.store.update_trade(tid, tp1_filled=1, remaining_contracts=size)
                self._event(tid, "tp1_filled", f"止盈1成交 {_usdt(tp1.filled, trade['ct_val'], tp1.avg_px)} @ {_px(tp1.avg_px)}")
                trade = self.store.get_trade(tid) or trade

        if (
            self.cfg.move_stop_to_breakeven
            and trade["tp1_filled"]
            and trade["tp2_cl_ord_id"]
            and not trade["breakeven_done"]
        ):
            self._move_stop_to_breakeven(trade)
            trade = self.store.get_trade(tid) or trade

        if not trade["tp1_filled"]:
            self._place_take_profits(trade, float(trade["filled_contracts"] or size))
            trade = self.store.get_trade(tid) or trade

        # Every tick: exactly one stop sized to the live position (shrinks after TP1).
        self._sync_stop(trade, size)

    def _move_stop_to_breakeven(self, trade: dict[str, Any]) -> None:
        tid = trade["id"]
        inst = trade["inst_id"]
        spec = self.broker.spec(inst)
        be = round_to_tick(float(trade["avg_entry"]), spec.tick_sz)
        last = self.broker.last_price(inst)
        if (last - be) * _sign(trade["direction"]) <= 0:
            self.store.update_trade(tid, breakeven_done=1)
            self._event(tid, "breakeven_skip", f"现价 {last} 已不优于保本价 {be}，保留原止损")
            return
        # _sync_stop (same tick) replaces the old stop with one at the new price.
        self.store.update_trade(tid, stop_loss=be, breakeven_done=1)
        self._event(tid, "breakeven", f"止盈1已成交，剩余仓位止损移至保本价 {be}")

    def _exit_order_states(self, trade: dict[str, Any]) -> dict[str, str]:
        states: dict[str, str] = {}
        for key in ("tp1_cl_ord_id", "tp2_cl_ord_id"):
            if trade[key]:
                try:
                    states[key] = self.broker.order_status(trade["inst_id"], trade[key]).state
                except OkxError:
                    states[key] = "unknown"
        if trade["sl_algo_cl_ord_id"]:
            try:
                states["sl"] = self.broker.stop_status(trade["sl_algo_cl_ord_id"]).state
            except OkxError:
                states["sl"] = "unknown"
        return states

    def _finalize(self, trade: dict[str, Any]) -> None:
        tid = trade["id"]
        inst = trade["inst_id"]
        states = self._exit_order_states(trade)
        # Leave nothing behind: any stop / reduce-only order on this instrument
        # now belongs to no position. Keep the trade open until cleanup succeeds.
        try:
            self._sweep_orphans(inst)
        except OkxError as exc:
            self._event(tid, "cleanup_error", f"平仓后清理挂单失败（下轮重试）: {exc}", level="warn")
            return

        if states.get("tp2_cl_ord_id") == "filled":
            reason = "止盈2"
        elif states.get("tp1_cl_ord_id") == "filled" and not trade["tp2_cl_ord_id"]:
            reason = "止盈"
        elif states.get("tp1_cl_ord_id") == "filled":
            reason = "止盈1后保本离场" if trade["breakeven_done"] else "止盈1后止损"
        elif trade["close_reason"]:
            reason = trade["close_reason"]
        elif states.get("sl") == "triggered":
            reason = "止损"
        else:
            reason = "仓位已平（止损或手动）"

        try:
            closed = self.broker.last_closed_position(inst, int(trade["opened_ms"] or 0))
        except OkxError as exc:
            self._event(tid, "pnl_error", f"读取平仓盈亏失败，稍后重试: {exc}", level="warn")
            return
        if closed is None:
            # positions-history lags the fill — retry for a while before giving up.
            flat_key = f"flat_since_{tid}"
            flat_since = int(self.store.get_kv(flat_key, "0") or 0)
            if not flat_since:
                self.store.set_kv(flat_key, str(now_ms()))
                return
            if now_ms() - flat_since < PNL_LAG_GRACE_MS:
                return
        fields: dict[str, Any] = {"state": "closed", "close_reason": reason,
                                  "closed_ms": now_ms(), "remaining_contracts": 0}
        if closed is not None:
            fields.update(
                avg_exit=closed.close_avg_px,
                realized_pnl=closed.pnl,
                fee=closed.fee,
                funding_fee=closed.funding_fee,
                net_pnl=closed.realized_pnl,
                closed_ms=closed.close_ms or now_ms(),
            )
        self.store.update_trade(tid, **fields)
        pnl_txt = f"{closed.realized_pnl:+.2f}U" if closed else "未取到"
        self._event(tid, "closed", f"平仓（{reason}），净盈亏 {pnl_txt}")

    def _manage_trade(self, trade: dict[str, Any]) -> None:
        if trade["state"] == "pending":
            self._manage_pending(trade)
        elif trade["state"] == "open":
            self._manage_open(trade)

    # ── account-wide ──────────────────────────────────────────────────────────

    def _kill_step(self) -> None:
        """Kill switch: repeated every tick until no managed trade is active."""
        if self.store.get_kv(KILL_DONE_KEY) != "1":
            self._event(None, "kill", "紧急停止：撤销所有挂单并平掉所有仓位", level="warn")
            self.store.set_kv(KILL_DONE_KEY, "1")
        self.flatten_all("紧急停止")

    def flatten_all(self, reason: str) -> bool:
        """Cancel every resting entry and close every managed position.

        Returns True when nothing is left pending and every position close was
        accepted. Open trades settle (finalize) on later ticks once flat.
        """
        ok = True
        for trade in self.store.active_trades():
            try:
                if trade["state"] == "pending":
                    ok = self._cancel_pending(trade, reason) and ok
                    trade = self.store.get_trade(trade["id"]) or trade
                if trade["state"] == "open" and self.broker.position(trade["inst_id"]).size != 0:
                    # close-position with autoCxl first; exits are swept at finalize.
                    self.broker.close_position(trade["inst_id"], self.cfg.margin_mode)
                    self.store.update_trade(trade["id"], close_reason=reason)
            except OkxError as exc:
                ok = False
                self._event(trade["id"], "kill_error", f"{reason}处理失败（下轮重试）: {exc}",
                            level="error")
        return ok

    def _snapshot_equity(self) -> None:
        try:
            acct = self.broker.account()
        except OkxError as exc:
            logger.warning("equity snapshot failed: %s", exc)
            return
        self.store.record_equity(acct.total_eq, acct.avail_eq, acct.upl, demo=self.broker.demo,
                                 ccy=acct.ccy)

    def _reconcile(self) -> None:
        """Warn about exchange positions the executor is not managing."""
        try:
            positions = self.broker.open_positions()
        except OkxError as exc:
            logger.warning("reconcile failed: %s", exc)
            return
        managed = {t["inst_id"] for t in self.store.active_trades()}
        on_exchange = {str(p.get("instId")) for p in positions}
        for inst in sorted(on_exchange - managed):
            key = f"reconcile_warned_{inst}"
            if self.store.get_kv(key) != "1":
                self._event(None, "reconcile", f"交易所上有 {inst} 持仓但不在系统管理中（手动开仓？）",
                            level="warn")
                self.store.set_kv(key, "1")
        for inst in on_exchange & managed:
            self.store.set_kv(f"reconcile_warned_{inst}", "0")
