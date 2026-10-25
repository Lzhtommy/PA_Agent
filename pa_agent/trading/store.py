"""SQLite persistence for signals, trades, events and equity snapshots."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_uid TEXT NOT NULL UNIQUE,
    created_ms INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT '',
    order_type TEXT NOT NULL DEFAULT 'none',
    entry REAL,
    stop_loss REAL,
    tp1 REAL,
    tp2 REAL,
    trade_confidence INTEGER,
    estimated_win_rate INTEGER,
    reasoning TEXT NOT NULL DEFAULT '',
    decision_json TEXT NOT NULL DEFAULT '{}',
    stage1_json TEXT NOT NULL DEFAULT '{}',
    record_path TEXT NOT NULL DEFAULT '',
    chart_image TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'new',
    status_reason TEXT NOT NULL DEFAULT '',
    processed_ms INTEGER,
    model TEXT NOT NULL DEFAULT '',
    prompt_hash TEXT NOT NULL DEFAULT '',
    code_sha TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status, id);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER REFERENCES signals(id),
    inst_id TEXT NOT NULL,
    timeframe TEXT NOT NULL DEFAULT '',
    direction TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    state TEXT NOT NULL,
    demo INTEGER NOT NULL DEFAULT 1,
    planned_entry REAL,
    stop_loss REAL,
    tp1 REAL,
    tp2 REAL,
    contracts REAL NOT NULL DEFAULT 0,
    filled_contracts REAL NOT NULL DEFAULT 0,
    remaining_contracts REAL NOT NULL DEFAULT 0,
    ct_val REAL NOT NULL DEFAULT 0,
    risk_amount REAL,
    equity_at_entry REAL,
    entry_cl_ord_id TEXT NOT NULL DEFAULT '',
    entry_ord_id TEXT NOT NULL DEFAULT '',
    entry_algo_id TEXT NOT NULL DEFAULT '',
    sl_algo_cl_ord_id TEXT NOT NULL DEFAULT '',
    sl_algo_id TEXT NOT NULL DEFAULT '',
    tp1_cl_ord_id TEXT NOT NULL DEFAULT '',
    tp2_cl_ord_id TEXT NOT NULL DEFAULT '',
    tp1_filled INTEGER NOT NULL DEFAULT 0,
    breakeven_done INTEGER NOT NULL DEFAULT 0,
    avg_entry REAL,
    avg_exit REAL,
    realized_pnl REAL,
    fee REAL,
    funding_fee REAL,
    net_pnl REAL,
    close_reason TEXT NOT NULL DEFAULT '',
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL,
    expires_ms INTEGER,
    opened_ms INTEGER,
    closed_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_trades_state ON trades(state, inst_id);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    trade_id INTEGER,
    signal_id INTEGER,
    level TEXT NOT NULL DEFAULT 'info',
    kind TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_trade ON events(trade_id);

CREATE TABLE IF NOT EXISTS equity (
    ts_ms INTEGER PRIMARY KEY,
    total_eq REAL NOT NULL,
    avail_eq REAL,
    upl REAL,
    demo INTEGER NOT NULL DEFAULT 1,
    ccy TEXT NOT NULL DEFAULT 'USDT'
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

ACTIVE_STATES = ("pending", "open")
# kv prefix for "cancel this resting entry" requests from the dashboard; the
# executor (the only code that talks to the exchange) carries them out.
CANCEL_REQUEST_PREFIX = "cancel_request:"
TERMINAL_STATES = ("closed", "canceled", "rejected", "error")


def now_ms() -> int:
    return int(time.time() * 1000)


class TradeStore:
    """Small thread-safe wrapper around one SQLite file (WAL mode)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(_SCHEMA)
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(equity)")}
            if "ccy" not in cols:
                # Rows written before equity switched to USDT hold OKX totalEq (USD).
                self._conn.execute("ALTER TABLE equity ADD COLUMN ccy TEXT NOT NULL DEFAULT 'USD'")
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(signals)")}
            for col in ("model", "prompt_hash", "code_sha"):
                if col not in cols:
                    self._conn.execute(f"ALTER TABLE signals ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── low level ─────────────────────────────────────────────────────────────

    def _exec(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def _rows(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def _row(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self._rows(sql, params)
        return rows[0] if rows else None

    # ── kv ────────────────────────────────────────────────────────────────────

    def get_kv(self, key: str, default: str = "") -> str:
        row = self._row("SELECT value FROM kv WHERE key=?", (key,))
        return row["value"] if row else default

    def set_kv(self, key: str, value: str) -> None:
        self._exec(
            "INSERT INTO kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def request_cancel(self, trade_id: int) -> None:
        self.set_kv(f"{CANCEL_REQUEST_PREFIX}{int(trade_id)}", str(now_ms()))

    def cancel_requests(self) -> list[int]:
        rows = self._rows("SELECT key FROM kv WHERE key LIKE ?", (CANCEL_REQUEST_PREFIX + "%",))
        return sorted(int(r["key"][len(CANCEL_REQUEST_PREFIX):]) for r in rows)

    def clear_cancel_request(self, trade_id: int) -> None:
        self._exec("DELETE FROM kv WHERE key=?", (f"{CANCEL_REQUEST_PREFIX}{int(trade_id)}",))

    # ── signals ───────────────────────────────────────────────────────────────

    def insert_signal(self, signal: dict[str, Any]) -> int | None:
        """Insert a signal; returns its id, or None when ``signal_uid`` already exists."""
        cols = [
            "signal_uid", "created_ms", "source", "symbol", "timeframe", "direction",
            "order_type", "entry", "stop_loss", "tp1", "tp2", "trade_confidence",
            "estimated_win_rate", "reasoning", "decision_json", "stage1_json",
            "record_path", "chart_image", "model", "prompt_hash", "code_sha",
        ]
        values = []
        for c in cols:
            v = signal.get(c)
            if c in ("decision_json", "stage1_json") and not isinstance(v, str):
                v = json.dumps(v or {}, ensure_ascii=False)
            if v is None and c in ("source", "direction", "reasoning", "record_path", "chart_image",
                                   "model", "prompt_hash", "code_sha"):
                v = ""
            values.append(v)
        if values[1] is None:
            values[1] = now_ms()
        try:
            cur = self._exec(
                f"INSERT INTO signals({', '.join(cols)}) VALUES({', '.join('?' * len(cols))})",
                values,
            )
        except sqlite3.IntegrityError:
            return None
        return int(cur.lastrowid)

    def new_signals(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM signals WHERE status='new' ORDER BY id LIMIT ?", (limit,)
        )

    def mark_signal(self, signal_id: int, status: str, reason: str = "") -> None:
        self._exec(
            "UPDATE signals SET status=?, status_reason=?, processed_ms=? WHERE id=?",
            (status, reason, now_ms(), signal_id),
        )

    def get_signal(self, signal_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM signals WHERE id=?", (signal_id,))

    @staticmethod
    def _version_clause(version: str | None) -> tuple[str, list[Any]]:
        """``version`` is ``"model|prompt_hash|timeframe"`` (as returned by :meth:`versions`)."""
        model, phash, timeframe = ((version or "").split("|") + ["", ""])[:3]
        return "model=? AND prompt_hash=? AND timeframe=?", [model, phash, timeframe]

    def _signal_filter(self, symbol: str | None, actionable: bool,
                       version: str | None) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if symbol:
            clauses.append("symbol=?")
            params.append(symbol)
        if actionable:
            clauses.append("order_type != 'none'")
        if version:
            clause, vparams = self._version_clause(version)
            clauses.append(clause)
            params += vparams
        return ("WHERE " + " AND ".join(clauses) + " " if clauses else ""), params

    def list_signals(self, limit: int = 200, offset: int = 0, symbol: str | None = None,
                     actionable: bool = False, version: str | None = None) -> list[dict[str, Any]]:
        where, params = self._signal_filter(symbol, actionable, version)
        return self._rows(
            "SELECT id, signal_uid, created_ms, source, symbol, timeframe, direction, "
            "order_type, entry, stop_loss, tp1, tp2, trade_confidence, estimated_win_rate, "
            "status, status_reason, chart_image, model, prompt_hash "
            f"FROM signals {where}ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )

    def count_signals(self, symbol: str | None = None, actionable: bool = False,
                      version: str | None = None) -> int:
        where, params = self._signal_filter(symbol, actionable, version)
        return self._row(f"SELECT COUNT(*) AS n FROM signals {where}", params)["n"]

    def last_signal_ms(self, symbol: str, timeframe: str) -> int:
        """created_ms of the newest signal for *symbol* on *timeframe* (0 if none)."""
        row = self._row("SELECT MAX(created_ms) AS ms FROM signals WHERE symbol=? AND timeframe=?",
                        (symbol, timeframe))
        return int(row["ms"] or 0)

    def versions(self) -> list[dict[str, Any]]:
        """Strategy versions (model, prompt hash, timeframe) seen in signals, most
        recent first, with sample sizes."""
        return self._rows(
            "SELECT s.model, s.prompt_hash, s.timeframe, MIN(s.created_ms) AS first_ms, "
            "MAX(s.created_ms) AS last_ms, COUNT(*) AS signals, SUM(s.order_type != 'none') AS orders, "
            "(SELECT COUNT(*) FROM trades t JOIN signals s2 ON s2.id = t.signal_id "
            " WHERE t.state = 'closed' AND s2.model = s.model AND s2.prompt_hash = s.prompt_hash "
            " AND s2.timeframe = s.timeframe) AS closed "
            "FROM signals s GROUP BY s.model, s.prompt_hash, s.timeframe ORDER BY last_ms DESC"
        )

    # ── trades ────────────────────────────────────────────────────────────────

    def insert_trade(self, trade: dict[str, Any]) -> int:
        ts = now_ms()
        data = {"created_ms": ts, "updated_ms": ts, **trade}
        cols = list(data)
        cur = self._exec(
            f"INSERT INTO trades({', '.join(cols)}) VALUES({', '.join('?' * len(cols))})",
            [data[c] for c in cols],
        )
        return int(cur.lastrowid)

    def update_trade(self, trade_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_ms"] = now_ms()
        assignments = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE trades SET {assignments} WHERE id=?", [*fields.values(), trade_id])

    def get_trade(self, trade_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM trades WHERE id=?", (trade_id,))

    def active_trades(self, inst_id: str | None = None) -> list[dict[str, Any]]:
        if inst_id:
            return self._rows(
                "SELECT * FROM trades WHERE state IN ('pending','open') AND inst_id=? ORDER BY id",
                (inst_id,),
            )
        return self._rows("SELECT * FROM trades WHERE state IN ('pending','open') ORDER BY id")

    def _trade_filter(self, states: tuple[str, ...] | None, inst_id: str | None,
                      version: str | None = None) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if version:
            clause, vparams = self._version_clause(version)
            clauses.append(f"signal_id IN (SELECT id FROM signals WHERE {clause})")
            params += vparams
        if states:
            clauses.append(f"state IN ({', '.join('?' * len(states))})")
            params += states
        if inst_id:
            clauses.append("inst_id=?")
            params.append(inst_id)
        return ("WHERE " + " AND ".join(clauses) + " " if clauses else ""), params

    def list_trades(
        self, *, states: tuple[str, ...] | None = None, limit: int = 500, offset: int = 0,
        inst_id: str | None = None, version: str | None = None,
    ) -> list[dict[str, Any]]:
        where, params = self._trade_filter(states, inst_id, version)
        return self._rows(
            f"SELECT * FROM trades {where}ORDER BY id DESC LIMIT ? OFFSET ?", (*params, limit, offset)
        )

    def count_trades(self, *, states: tuple[str, ...] | None = None, inst_id: str | None = None,
                     version: str | None = None) -> int:
        where, params = self._trade_filter(states, inst_id, version)
        return self._row(f"SELECT COUNT(*) AS n FROM trades {where}", params)["n"]

    def instruments(self) -> list[str]:
        """Every instrument that has a trade or a signal, most recently used first."""
        rows = self._rows(
            "SELECT name FROM (SELECT inst_id AS name, MAX(created_ms) AS ts FROM trades GROUP BY inst_id "
            "UNION ALL SELECT symbol, MAX(created_ms) FROM signals GROUP BY symbol) "
            "WHERE name != '' GROUP BY name ORDER BY MAX(ts) DESC"
        )
        return [r["name"] for r in rows]

    def closed_trades_since(self, since_ms: int) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM trades WHERE state='closed' AND closed_ms>=? ORDER BY closed_ms",
            (since_ms,),
        )

    # ── events ────────────────────────────────────────────────────────────────

    def log_event(
        self,
        kind: str,
        message: str,
        *,
        level: str = "info",
        trade_id: int | None = None,
        signal_id: int | None = None,
    ) -> None:
        self._exec(
            "INSERT INTO events(ts_ms, trade_id, signal_id, level, kind, message) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (now_ms(), trade_id, signal_id, level, kind, message),
        )

    def events(self, *, trade_id: int | None = None, limit: int = 200,
               offset: int = 0) -> list[dict[str, Any]]:
        if trade_id is not None:
            return self._rows(
                "SELECT * FROM events WHERE trade_id=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (trade_id, limit, offset),
            )
        return self._rows("SELECT * FROM events ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))

    def count_events(self) -> int:
        return self._row("SELECT COUNT(*) AS n FROM events")["n"]

    # ── equity ────────────────────────────────────────────────────────────────

    # Equity reads only return USDT rows so stats never mix USD and USDT values.

    def record_equity(self, total_eq: float, avail_eq: float | None, upl: float | None,
                      *, demo: bool, ccy: str = "USDT") -> None:
        self._exec(
            "INSERT OR REPLACE INTO equity(ts_ms, total_eq, avail_eq, upl, demo, ccy) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (now_ms(), total_eq, avail_eq, upl, int(demo), ccy),
        )

    def equity_series(self, *, since_ms: int = 0, demo: bool | None = None) -> list[dict[str, Any]]:
        if demo is None:
            return self._rows(
                "SELECT * FROM equity WHERE ts_ms>=? AND ccy='USDT' ORDER BY ts_ms", (since_ms,)
            )
        return self._rows(
            "SELECT * FROM equity WHERE ts_ms>=? AND demo=? AND ccy='USDT' ORDER BY ts_ms",
            (since_ms, int(demo)),
        )

    def latest_equity(self) -> dict[str, Any] | None:
        return self._row("SELECT * FROM equity WHERE ccy='USDT' ORDER BY ts_ms DESC LIMIT 1")

    def first_equity_since(self, since_ms: int) -> dict[str, Any] | None:
        return self._row(
            "SELECT * FROM equity WHERE ts_ms>=? AND ccy='USDT' ORDER BY ts_ms LIMIT 1", (since_ms,)
        )
