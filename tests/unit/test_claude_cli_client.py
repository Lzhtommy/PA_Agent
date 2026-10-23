"""Tests for the Claude Code CLI client (subprocess mocked)."""
from __future__ import annotations

import io
import json
from typing import Any

import pytest

import pa_agent.ai.claude_cli_client as cli
from pa_agent.ai.claude_cli_client import (
    ClaudeCliClient,
    build_cli_turn,
    is_claude_cli_model,
    resolve_claude_cli_model,
)
from pa_agent.ai.client_factory import create_ai_client
from pa_agent.ai.deepseek_client import supports_kv_prefix_chain
from pa_agent.config.settings import AIProviderSettings, Settings, provider_api_key_configured

SYSTEM = {"role": "system", "content": "SYS"}
S1_USER = {"role": "user", "content": "stage1 task"}


def _events(text: str, session_id: str, *, usage: dict[str, int] | None = None) -> list[dict]:
    return [
        {"type": "system", "subtype": "init", "session_id": session_id},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "thinking_delta", "thinking": "hmm"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 1,
                                           "delta": {"type": "text_delta", "text": text[:3]}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 1,
                                           "delta": {"type": "text_delta", "text": text[3:]}}},
        {"type": "result", "subtype": "success", "is_error": False, "session_id": session_id,
         "result": text, "stop_reason": "end_turn",
         "usage": usage or {"input_tokens": 10, "cache_read_input_tokens": 900,
                            "cache_creation_input_tokens": 90, "output_tokens": 50}},
    ]


class _FakeProc:
    def __init__(self, events: list[dict], returncode: int = 0) -> None:
        self.stdin = io.StringIO()
        self.stdin.close = lambda: None  # type: ignore[method-assign]
        self.stdout = io.StringIO("".join(json.dumps(e) + "\n" for e in events))
        self.stderr = io.StringIO("")
        self.returncode = returncode
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True


@pytest.fixture
def fake_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> list[dict[str, Any]]:
    """Queue fake processes; each launch records its argv and stdin prompt."""
    calls: list[dict[str, Any]] = []
    queued: list[_FakeProc] = []

    def _popen(cmd: list[str], **kwargs: Any) -> _FakeProc:
        proc = queued.pop(0)
        calls.append({"cmd": cmd, "proc": proc, "system": open(cmd[cmd.index("--system-prompt-file") + 1]).read()})
        return proc

    monkeypatch.setattr(cli.subprocess, "Popen", _popen)
    monkeypatch.setattr(cli, "_WORK_DIR", tmp_path)
    monkeypatch.setattr(cli, "_prune_old_sessions", lambda _sid: None)
    cli._SESSIONS.clear()
    calls.append({"queue": queued})  # index 0: handle for tests to enqueue
    return calls


def _settings(model: str = "claude_cli/sonnet") -> AIProviderSettings:
    return AIProviderSettings(model=model, base_url="/fake/claude", api_key="",
                              thinking=True, reasoning_effort="high")


def test_model_routing_helpers() -> None:
    assert is_claude_cli_model("claude_cli")
    assert is_claude_cli_model("Claude_CLI/opus")
    assert not is_claude_cli_model("deepseek-v4-flash")
    assert resolve_claude_cli_model("claude_cli") == "sonnet"
    assert resolve_claude_cli_model("claude_cli/opus") == "opus"


def test_factory_key_check_and_prefix_chain() -> None:
    s = _settings()
    assert isinstance(create_ai_client(s), ClaudeCliClient)
    assert supports_kv_prefix_chain(s)
    full = Settings()
    full.provider = s
    assert provider_api_key_configured(full)


def test_unknown_history_is_flattened() -> None:
    msgs = [SYSTEM, S1_USER, {"role": "assistant", "content": "A"}, {"role": "user", "content": "B"}]
    system, prompt, resume = build_cli_turn(msgs)
    assert system == "SYS"
    assert resume is None
    assert prompt == "[User]\nstage1 task\n\n[Assistant]\nA\n\n[User]\nB"


def test_stream_chat_streams_and_reports_usage(fake_cli: list[dict[str, Any]]) -> None:
    fake_cli[0]["queue"].append(_FakeProc(_events('{"ok":1}', "sess-1")))
    content: list[str] = []
    reasoning: list[str] = []

    reply = ClaudeCliClient(_settings()).stream_chat(
        [SYSTEM, S1_USER],
        on_content_token=content.append,
        on_reasoning_token=reasoning.append,
    )

    call = fake_cli[1]
    cmd = call["cmd"]
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--effort") + 1] == "high"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--session-id" in cmd and "--resume" not in cmd
    assert call["system"] == "SYS"
    assert call["proc"].stdin.getvalue() == "stage1 task"
    assert reply.content == '{"ok":1}'
    assert "".join(content) == '{"ok":1}'
    assert reasoning == ["hmm"]
    assert reply.usage.prompt_tokens == 1000
    assert reply.usage.cached_prompt_tokens == 900
    assert reply.usage.completion_tokens == 50


def test_stage2_chain_resumes_stage1_session(fake_cli: list[dict[str, Any]]) -> None:
    q = fake_cli[0]["queue"]
    q.append(_FakeProc(_events('{"s1":1}', "sess-1")))
    q.append(_FakeProc(_events('{"s2":1}', "sess-2")))
    client = ClaudeCliClient(_settings())

    client.stream_chat([SYSTEM, S1_USER])
    client.stream_chat([
        SYSTEM, S1_USER,
        {"role": "assistant", "content": '{"s1":1, "normalized": true}'},
        {"role": "user", "content": "stage2 task"},
    ])

    cmd = fake_cli[2]["cmd"]
    assert cmd[cmd.index("--resume") + 1] == "sess-1"
    assert "--fork-session" in cmd and "--session-id" not in cmd
    assert fake_cli[2]["proc"].stdin.getvalue() == "stage2 task"


def test_thinking_off_uses_low_effort(fake_cli: list[dict[str, Any]]) -> None:
    fake_cli[0]["queue"].append(_FakeProc(_events("x", "s")))
    ClaudeCliClient(_settings()).stream_chat([SYSTEM, S1_USER], thinking=False)
    cmd = fake_cli[1]["cmd"]
    assert cmd[cmd.index("--effort") + 1] == "low"


def test_error_result_raises_with_message(fake_cli: list[dict[str, Any]]) -> None:
    fake_cli[0]["queue"].append(_FakeProc([
        {"type": "result", "subtype": "success", "is_error": True,
         "result": "Claude usage limit reached", "session_id": "s"},
    ], returncode=1))
    with pytest.raises(RuntimeError, match="usage limit"):
        ClaudeCliClient(_settings()).stream_chat([SYSTEM, S1_USER])
    assert not cli._SESSIONS
