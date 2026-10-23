"""Claude Code CLI client for PA Agent (``claude -p`` headless mode).

Implements the same interface as :class:`pa_agent.ai.deepseek_client.DeepSeekClient`
(``stream_chat`` and ``update_provider``) but runs the local ``claude`` binary,
so calls are billed to the logged-in Claude subscription instead of an API key.

Model ids: ``claude_cli`` (defaults to Sonnet) or ``claude_cli/<alias-or-id>``
such as ``claude_cli/opus`` / ``claude_cli/claude-sonnet-5-5``.

The CLI takes a single user turn per invocation, so multi-turn message lists
are mapped like this:

* Every successful call records ``hash(messages) -> session_id``.
* A later call shaped ``prefix + [assistant, user]`` whose ``prefix`` was sent
  before resumes that session (``--resume <id> --fork-session``) and sends only
  the final user turn. This is how Stage 2 chains after Stage 1 and keeps the
  Anthropic prompt cache warm for the shared prefix.
* Anything else is flattened into one prompt with role labels.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from pa_agent.ai.deepseek_client import AIReply, AIUsage, CancelledError
from pa_agent.config.settings import AIProviderSettings

if TYPE_CHECKING:
    from pa_agent.util.threading import CancelToken

logger = logging.getLogger(__name__)

_CLAUDE_CLI_MODEL = "claude_cli"
_DEFAULT_CLI_MODEL = "sonnet"
_POLL_INTERVAL_S = 0.3
_SESSION_MAP_MAX = 128
#: Session transcripts older than this are deleted from the CLI's project dir.
_SESSION_RETENTION_S = 24 * 3600

#: Dedicated working directory: keeps the CLI's session files separate from
#: interactive Claude Code use and away from this repo's CLAUDE.md.
_WORK_DIR = Path.home() / ".pa_agent" / "claude_cli"

_SESSIONS: "OrderedDict[str, str]" = OrderedDict()
_SESSIONS_LOCK = threading.Lock()


def is_claude_cli_model(model: str | None) -> bool:
    """True for ``claude_cli`` and ``claude_cli/<model>``."""
    m = (model or "").strip().lower()
    return m == _CLAUDE_CLI_MODEL or m.startswith(f"{_CLAUDE_CLI_MODEL}/")


def resolve_claude_cli_model(model: str | None) -> str:
    """``claude_cli/opus`` -> ``opus``; bare ``claude_cli`` -> ``sonnet``."""
    m = (model or "").strip()
    _, _, tail = m.partition("/")
    return tail.strip() or _DEFAULT_CLI_MODEL


def resolve_claude_binary(settings: AIProviderSettings) -> str:
    """Path to the ``claude`` executable.

    ``base_url`` may hold an explicit path (useful under systemd, whose PATH
    usually lacks ``~/.local/bin``); otherwise PATH, then the installer default.
    """
    base = (settings.base_url or "").strip()
    if base and not base.lower().startswith(("http://", "https://")):
        return os.path.expanduser(base)
    found = shutil.which("claude")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "claude"
    if fallback.exists():
        return str(fallback)
    raise RuntimeError(
        "找不到 claude 命令行。请先安装 Claude Code 并登录，"
        "或在「AI 模型」的 base_url 填写 claude 可执行文件的完整路径。"
    )


def _effective_effort(
    settings: AIProviderSettings,
    *,
    thinking: bool | None,
    reasoning_effort: str | None,
) -> str:
    use_thinking = settings.thinking if thinking is None else thinking
    if not use_thinking:
        return "low"
    effort = (reasoning_effort or settings.reasoning_effort or "").strip().lower()
    return effort if effort in ("low", "medium", "high", "xhigh", "max") else "high"


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content or "")


def _messages_key(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [[m.get("role", ""), _content_text(m.get("content"))] for m in messages],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _remember_session(messages: list[dict[str, Any]], session_id: str) -> None:
    with _SESSIONS_LOCK:
        key = _messages_key(messages)
        _SESSIONS[key] = session_id
        _SESSIONS.move_to_end(key)
        while len(_SESSIONS) > _SESSION_MAP_MAX:
            _SESSIONS.popitem(last=False)


def _lookup_session(messages: list[dict[str, Any]]) -> str | None:
    with _SESSIONS_LOCK:
        return _SESSIONS.get(_messages_key(messages))


def _flatten(messages: list[dict[str, Any]]) -> str:
    """Render a non-system message list as one prompt (single user turn: as-is)."""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return _content_text(messages[0].get("content")).strip()
    labels = {"assistant": "[Assistant]", "user": "[User]"}
    parts = []
    for m in messages:
        text = _content_text(m.get("content")).strip()
        if text:
            parts.append(f"{labels.get(str(m.get('role')), '[User]')}\n{text}")
    return "\n\n".join(parts)


def build_cli_turn(
    messages: list[dict[str, Any]],
) -> tuple[str, str, str | None]:
    """Map OpenAI-style messages to ``(system_prompt, prompt, resume_session_id)``."""
    system = "\n\n".join(
        _content_text(m.get("content")) for m in messages if m.get("role") == "system"
    ).strip()
    convo = [m for m in messages if m.get("role") != "system"]
    if (
        len(convo) >= 3
        and convo[-1].get("role") == "user"
        and convo[-2].get("role") == "assistant"
    ):
        session_id = _lookup_session(messages[: len(messages) - 2])
        if session_id:
            return system, _content_text(convo[-1].get("content")).strip(), session_id
    return system, _flatten(convo), None


def _prune_old_sessions(session_id: str) -> None:
    """Delete stale transcripts the CLI wrote for our dedicated work dir."""
    try:
        projects = Path.home() / ".claude" / "projects"
        hits = list(projects.glob(f"*/{session_id}.jsonl"))
        if not hits:
            return
        cutoff = time.time() - _SESSION_RETENTION_S
        for f in hits[0].parent.glob("*.jsonl"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("claude_cli session prune skipped: %s", exc)


def _usage_from_result(usage: dict[str, Any] | None) -> AIUsage:
    u = usage or {}
    fresh = int(u.get("input_tokens") or 0)
    cache_read = int(u.get("cache_read_input_tokens") or 0)
    cache_write = int(u.get("cache_creation_input_tokens") or 0)
    out = int(u.get("output_tokens") or 0)
    prompt = fresh + cache_read + cache_write
    return AIUsage(
        prompt_tokens=prompt,
        cached_prompt_tokens=cache_read,
        completion_tokens=out,
        total_tokens=prompt + out,
    )


def _pump_lines(stream: Any, q: "queue.Queue[str | None]") -> None:
    try:
        for line in iter(stream.readline, ""):
            q.put(line)
    finally:
        q.put(None)


def _drain(q: "queue.Queue[str | None]", wait_s: float = 2.0) -> str:
    """Collect what a pump thread has queued, until its end marker or *wait_s*."""
    parts: list[str] = []
    while True:
        try:
            item = q.get(timeout=wait_s)
        except queue.Empty:
            break
        if item is None:
            break
        parts.append(item)
    return "".join(parts)


class ClaudeCliClient:
    """Runs ``claude -p`` per request and streams its stream-json output."""

    def __init__(
        self,
        settings: AIProviderSettings,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._log = logger_ or logger

    def update_provider(self, settings: AIProviderSettings) -> None:
        """Replace in-memory provider settings."""
        self._settings = settings

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        on_reasoning_token: Callable[[str], None] | None = None,
        on_content_token: Callable[[str], None] | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        cancel_token: "CancelToken | None" = None,
        timeout_s: float = 600.0,
    ) -> AIReply:
        """Run one CLI turn, surfacing text/thinking deltas to the callbacks."""
        if cancel_token is not None and cancel_token.is_set():
            raise CancelledError("Request cancelled before API call")

        binary = resolve_claude_binary(self._settings)
        cli_model = resolve_claude_cli_model(self._settings.model)
        effort = _effective_effort(
            self._settings, thinking=thinking, reasoning_effort=reasoning_effort
        )
        system, prompt, resume_id = build_cli_turn(messages)
        if not prompt:
            raise RuntimeError("Claude CLI 路由：消息内容为空，无法发送。")

        _WORK_DIR.mkdir(parents=True, exist_ok=True)
        sys_file = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", dir=_WORK_DIR, delete=False
        )
        with sys_file:
            sys_file.write(system or "You are a helpful assistant.")

        cmd = [
            binary, "-p",
            "--model", cli_model,
            "--effort", effort,
            "--system-prompt-file", sys_file.name,
            "--tools", "",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",
        ]
        if resume_id:
            cmd += ["--resume", resume_id, "--fork-session"]
        else:
            cmd += ["--session-id", str(uuid.uuid4())]

        self._log.info(
            "ClaudeCliClient.stream_chat: model=%s effort=%s msgs=%d mode=%s prompt_len=%d",
            cli_model,
            effort,
            len(messages),
            "resume" if resume_id else "new",
            len(prompt),
        )

        t0 = time.monotonic()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        result: dict[str, Any] | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=_WORK_DIR,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            os.unlink(sys_file.name)
            raise RuntimeError(f"无法启动 claude 命令行 ({binary}): {exc}") from exc

        lines: "queue.Queue[str | None]" = queue.Queue()
        stderr_lines: "queue.Queue[str | None]" = queue.Queue()
        threading.Thread(target=_pump_lines, args=(proc.stdout, lines), daemon=True).start()
        threading.Thread(target=_pump_lines, args=(proc.stderr, stderr_lines), daemon=True).start()
        try:
            assert proc.stdin is not None
            proc.stdin.write(prompt)
            proc.stdin.close()

            deadline = t0 + timeout_s
            while True:
                if cancel_token is not None and cancel_token.is_set():
                    raise CancelledError("Request cancelled during Claude CLI run")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Claude CLI 超时（{timeout_s:.0f}s）")
                try:
                    line = lines.get(timeout=_POLL_INTERVAL_S)
                except queue.Empty:
                    continue
                if line is None:
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "stream_event":
                    delta = (event.get("event") or {}).get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        content_parts.append(delta["text"])
                        if on_content_token:
                            on_content_token(delta["text"])
                    elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                        reasoning_parts.append(delta["thinking"])
                        if on_reasoning_token:
                            on_reasoning_token(delta["thinking"])
                elif etype == "result":
                    result = event
            proc.wait(timeout=30)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            os.unlink(sys_file.name)

        latency_ms = (time.monotonic() - t0) * 1000
        if result is None or result.get("is_error") or result.get("subtype") != "success":
            stderr_text = _drain(stderr_lines)
            detail = (result or {}).get("result") or stderr_text.strip() or f"exit={proc.returncode}"
            raise RuntimeError(f"Claude CLI 调用失败: {str(detail)[:500]}")

        session_id = str(result.get("session_id") or "")
        content = str(result.get("result") or "".join(content_parts)).strip()
        reasoning = "".join(reasoning_parts).strip()
        usage = _usage_from_result(result.get("usage"))

        if session_id:
            _remember_session(messages, session_id)
            _prune_old_sessions(session_id)

        self._log.info(
            "ClaudeCliClient.stream_chat done: latency=%.0f ms content_chars=%d "
            "prompt=%d cached=%d (%.0f%%) out=%d",
            latency_ms,
            len(content),
            usage.prompt_tokens,
            usage.cached_prompt_tokens,
            usage.cache_hit_rate * 100,
            usage.completion_tokens,
        )

        raw: dict[str, Any] = {
            "id": session_id,
            "model": cli_model,
            "content": content,
            "reasoning_content": reasoning,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cached_prompt_tokens": usage.cached_prompt_tokens,
                "cache_miss_tokens": usage.cache_miss_tokens,
                "cache_hit_rate_pct": round(usage.cache_hit_rate * 100, 1),
            },
            "latency_ms": latency_ms,
            "finish_reason": result.get("stop_reason"),
            "resumed_session": resume_id,
            "total_cost_usd": result.get("total_cost_usd"),
        }
        return AIReply(
            content=content,
            reasoning_content=reasoning,
            raw=raw,
            usage=usage,
            request_id=session_id,
            latency_ms=latency_ms,
        )
