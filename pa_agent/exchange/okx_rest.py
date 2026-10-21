"""Minimal OKX REST v5 client (public market data + signed private endpoints).

Signing (OKX v5): ``sign = base64(hmac_sha256(secret, ts + METHOD + path_with_query + body))``
with headers ``OK-ACCESS-KEY / OK-ACCESS-SIGN / OK-ACCESS-TIMESTAMP / OK-ACCESS-PASSPHRASE``.
Demo trading (模拟盘) adds ``x-simulated-trading: 1`` and requires a demo API key.

Proxies: httpx honours ``HTTPS_PROXY`` / ``ALL_PROXY`` env vars (``trust_env``).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

OKX_BASE_URL = "https://www.okx.com"

# Error codes worth retrying for idempotent GETs (rate limit / system busy / timeout).
_RETRYABLE_CODES = frozenset({"50001", "50004", "50011", "50013", "50026"})
# Outcomes where a POST may or may not have taken effect on the exchange: callers
# must look the order up by its client id instead of assuming failure.
# 50004 = endpoint timeout (request may still succeed), 51016 = duplicate clOrdId.
AMBIGUOUS_CODES = frozenset({"network", "50001", "50004", "50013", "50026", "51016"})


def is_ambiguous(exc: Exception) -> bool:
    return isinstance(exc, OkxError) and exc.code in AMBIGUOUS_CODES


class OkxError(Exception):
    """OKX API returned a non-zero code (or a sub-item ``sCode`` failure)."""

    def __init__(self, code: str, msg: str, *, data: Any = None) -> None:
        super().__init__(f"OKX {code}: {msg}")
        self.code = str(code)
        self.msg = msg
        self.data = data


class OkxNetworkError(OkxError):
    """Transport-level failure (timeout, DNS, proxy)."""

    def __init__(self, msg: str) -> None:
        super().__init__("network", msg)


@dataclass(frozen=True)
class OkxCredentials:
    api_key: str
    secret_key: str
    passphrase: str

    def is_complete(self) -> bool:
        return bool(self.api_key and self.secret_key and self.passphrase)


def _iso_timestamp() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def sign_request(secret_key: str, timestamp: str, method: str, request_path: str, body: str) -> str:
    """Return the OKX v5 ``OK-ACCESS-SIGN`` value."""
    prehash = f"{timestamp}{method.upper()}{request_path}{body}"
    digest = hmac.new(secret_key.encode(), prehash.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


class OkxRestClient:
    """Thread-safe synchronous OKX REST client."""

    def __init__(
        self,
        credentials: OkxCredentials | None = None,
        *,
        demo: bool = True,
        base_url: str = OKX_BASE_URL,
        timeout_s: float = 10.0,
        max_retries: int = 2,
        http_client: Any = None,
    ) -> None:
        self._creds = credentials
        self.demo = demo
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max(0, int(max_retries))
        self._http = http_client
        self._lock = threading.Lock()

    # ── transport ─────────────────────────────────────────────────────────────

    def _client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.Client(timeout=self._timeout_s, trust_env=True)
        return self._http

    def close(self) -> None:
        if self._http is not None and hasattr(self._http, "close"):
            self._http.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | list[Any] | None = None,
        private: bool = False,
    ) -> list[dict[str, Any]]:
        """Call *path* and return the ``data`` list; raise :class:`OkxError` on failure."""
        method = method.upper()
        clean_params = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        query = f"?{urlencode(clean_params)}" if clean_params else ""
        request_path = f"{path}{query}"
        body_str = json.dumps(body, separators=(",", ":")) if body is not None else ""

        # Never blindly re-send a POST: a timed-out order may already be live.
        attempts = self._max_retries + 1 if method == "GET" else 1
        last_exc: OkxError | None = None
        for attempt in range(attempts):
            headers = {"Content-Type": "application/json"}
            if self.demo:
                headers["x-simulated-trading"] = "1"
            if private:
                if self._creds is None or not self._creds.is_complete():
                    raise OkxError("auth", "OKX API Key / Secret / Passphrase 未配置")
                ts = _iso_timestamp()
                headers.update(
                    {
                        "OK-ACCESS-KEY": self._creds.api_key,
                        "OK-ACCESS-SIGN": sign_request(
                            self._creds.secret_key, ts, method, request_path, body_str
                        ),
                        "OK-ACCESS-TIMESTAMP": ts,
                        "OK-ACCESS-PASSPHRASE": self._creds.passphrase,
                    }
                )
            try:
                with self._lock:
                    resp = self._client().request(
                        method,
                        f"{self._base_url}{request_path}",
                        content=body_str or None,
                        headers=headers,
                    )
            except Exception as exc:
                last_exc = OkxNetworkError(str(exc) or type(exc).__name__)
                logger.warning("OKX %s %s network error (attempt %d): %s",
                               method, path, attempt + 1, exc)
                time.sleep(0.5 * (attempt + 1))
                continue

            try:
                payload = resp.json()
            except ValueError:
                last_exc = OkxError(str(resp.status_code), resp.text[:200])
                if resp.status_code >= 500 or resp.status_code == 429:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise last_exc from None

            code = str(payload.get("code", ""))
            data = payload.get("data") or []
            if code == "0":
                return data if isinstance(data, list) else [data]
            msg = str(payload.get("msg") or "")
            # Batch / order endpoints put the real reason in data[i].sMsg
            if isinstance(data, list) and data and isinstance(data[0], dict):
                s_code = str(data[0].get("sCode") or "")
                if s_code and s_code != "0":
                    msg = f"{msg} [{s_code}] {data[0].get('sMsg', '')}".strip()
            last_exc = OkxError(code, msg, data=data)
            if code in _RETRYABLE_CODES:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise last_exc
        assert last_exc is not None
        raise last_exc

    def get(self, path: str, params: dict[str, Any] | None = None, *, private: bool = False):
        return self.request("GET", path, params=params, private=private)

    def post(self, path: str, body: dict[str, Any] | list[Any], *, private: bool = True):
        return self.request("POST", path, body=body, private=private)

    # ── public market data ────────────────────────────────────────────────────

    def candles(
        self, inst_id: str, bar: str, *, limit: int = 300, after: str | None = None
    ) -> list[list[str]]:
        """Recent candles, newest first: ``[ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]``."""
        return self.get(  # type: ignore[return-value]
            "/api/v5/market/candles",
            {"instId": inst_id, "bar": bar, "limit": str(min(300, limit)), "after": after},
        )

    def history_candles(
        self, inst_id: str, bar: str, *, limit: int = 100, after: str | None = None
    ) -> list[list[str]]:
        return self.get(  # type: ignore[return-value]
            "/api/v5/market/history-candles",
            {"instId": inst_id, "bar": bar, "limit": str(min(100, limit)), "after": after},
        )

    def ticker(self, inst_id: str) -> dict[str, Any]:
        rows = self.get("/api/v5/market/ticker", {"instId": inst_id})
        if not rows:
            raise OkxError("empty", f"no ticker for {inst_id}")
        return rows[0]

    def instrument(self, inst_id: str, inst_type: str = "SWAP") -> dict[str, Any]:
        rows = self.get("/api/v5/public/instruments", {"instType": inst_type, "instId": inst_id})
        if not rows:
            raise OkxError("empty", f"unknown instrument {inst_id}")
        return rows[0]

    def instruments(self, inst_type: str = "SWAP") -> list[dict[str, Any]]:
        return self.get("/api/v5/public/instruments", {"instType": inst_type})

    # ── account ───────────────────────────────────────────────────────────────

    def balance(self, ccy: str | None = None) -> dict[str, Any]:
        rows = self.get("/api/v5/account/balance", {"ccy": ccy}, private=True)
        return rows[0] if rows else {}

    def account_config(self) -> dict[str, Any]:
        rows = self.get("/api/v5/account/config", private=True)
        return rows[0] if rows else {}

    def set_position_mode(self, pos_mode: str = "net_mode") -> None:
        self.post("/api/v5/account/set-position-mode", {"posMode": pos_mode})

    def set_leverage(self, inst_id: str, lever: int, mgn_mode: str) -> None:
        self.post(
            "/api/v5/account/set-leverage",
            {"instId": inst_id, "lever": str(lever), "mgnMode": mgn_mode},
        )

    def positions(self, inst_id: str | None = None) -> list[dict[str, Any]]:
        return self.get(
            "/api/v5/account/positions", {"instType": "SWAP", "instId": inst_id}, private=True
        )

    def positions_history(
        self, inst_id: str | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self.get(
            "/api/v5/account/positions-history",
            {"instType": "SWAP", "instId": inst_id, "limit": str(limit)},
            private=True,
        )

    def bills(self, *, bill_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Account bills (last 7 days). ``type=8`` = funding fee."""
        return self.get(
            "/api/v5/account/bills",
            {"instType": "SWAP", "type": bill_type, "limit": str(limit)},
            private=True,
        )

    # ── trading ───────────────────────────────────────────────────────────────

    def place_order(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.post("/api/v5/trade/order", body)[0]

    def cancel_order(self, inst_id: str, *, cl_ord_id: str | None = None,
                     ord_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"instId": inst_id}
        if cl_ord_id:
            body["clOrdId"] = cl_ord_id
        if ord_id:
            body["ordId"] = ord_id
        return self.post("/api/v5/trade/cancel-order", body)[0]

    def get_order(self, inst_id: str, *, cl_ord_id: str | None = None,
                  ord_id: str | None = None) -> dict[str, Any]:
        rows = self.get(
            "/api/v5/trade/order",
            {"instId": inst_id, "clOrdId": cl_ord_id, "ordId": ord_id},
            private=True,
        )
        if not rows:
            raise OkxError("empty", f"order not found {cl_ord_id or ord_id}")
        return rows[0]

    def place_algo_order(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.post("/api/v5/trade/order-algo", body)[0]

    def get_algo_order(self, *, algo_id: str | None = None,
                       algo_cl_ord_id: str | None = None) -> dict[str, Any]:
        rows = self.get(
            "/api/v5/trade/order-algo",
            {"algoId": algo_id, "algoClOrdId": algo_cl_ord_id},
            private=True,
        )
        if not rows:
            raise OkxError("empty", f"algo order not found {algo_cl_ord_id or algo_id}")
        return rows[0]

    def pending_algo_orders(self, ord_type: str, inst_id: str | None = None) -> list[dict[str, Any]]:
        return self.get(
            "/api/v5/trade/orders-algo-pending",
            {"ordType": ord_type, "instType": "SWAP", "instId": inst_id},
            private=True,
        )

    def cancel_algo_orders(self, items: list[dict[str, str]]) -> list[dict[str, Any]]:
        """*items*: ``[{"algoId": ..., "instId": ...}, ...]``."""
        if not items:
            return []
        return self.post("/api/v5/trade/cancel-algos", items)

    def pending_orders(self, inst_id: str | None = None) -> list[dict[str, Any]]:
        return self.get(
            "/api/v5/trade/orders-pending", {"instType": "SWAP", "instId": inst_id}, private=True
        )

    def close_position(self, inst_id: str, mgn_mode: str) -> None:
        self.post(
            "/api/v5/trade/close-position",
            {"instId": inst_id, "mgnMode": mgn_mode, "autoCxl": True},
        )
