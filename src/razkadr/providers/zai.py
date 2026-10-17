"""zai provider bridge: GLM models of the agent environment via the z-ai
CLI (naryad 085).

The owner has no provider API keys, but the agent's build environment
exposes a real API through the ``z-ai`` CLI (``chat`` / ``vision`` /
``asr``, GLM models). This bridge implements the provider interfaces over
subprocess calls to the CLI, so the pipeline can run with REAL providers
(no fake) and produce honest latency/usage numbers for the cost
methodology of naryad 004 - and the interfaces of naryads 037/039 get a
live check against a real service.

Design mirrors the real adapters (naryads 037-039):

- the CLI runner is injectable (``runner`` callable); the default spawns
  ``subprocess.run`` with a timeout, tests run hermetically against stubs;
- JSON goes to a temp file via ``-o`` and is parsed back (the CLI writes
  progress lines to stdout, the payload to the file);
- no API keys are involved - the CLI authenticates with its own
  environment; nothing secret is logged (there is nothing to mask);
- usage is reported as token units for the cost accounting (naryad 033);
- error mapping: CLI missing -> ProviderConfigError (with a hint),
  nonzero exit / timeout -> TransientError (the caller may retry),
  unusable output (bad JSON, no content) -> ProviderError.

Model: the CLI takes no model flag; the server picks the model
(glm-4-plus at the time of the probe). The ``model`` config value is
accounting metadata only - it lands in run.json, not in the request.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from razkadr.providers.base import ProviderConfigError, ProviderError, TransientError, UsageReporter
from razkadr.providers.rest import wav_duration_s
from razkadr.providers.types import Message, Transcript, TranscriptSegment

CLI_NAME = "z-ai"
DEFAULT_TIMEOUT_S = 120.0
STDERR_LIMIT = 200  # chars of a CLI error tail kept in exception messages
EMPTY_REPLY = "(Empty response)"  # CLI placeholder when the model returns no content


CliRunner = Callable[[list[str]], subprocess.CompletedProcess]


def subprocess_runner(timeout_s: float) -> CliRunner:
    """Default runner: one subprocess call with a hard timeout."""

    def run(argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(argv, capture_output=True, timeout=timeout_s, check=False)

    return run


def default_runner(timeout_s: float) -> CliRunner:
    """Runner factory; module-level so tests can monkeypatch it."""
    return subprocess_runner(timeout_s)


def _run_cli(runner: CliRunner, argv: list[str], out_path: Path) -> dict:
    """One CLI call: runner -> status mapping -> parsed JSON payload."""
    try:
        proc = runner(argv)
    except FileNotFoundError:
        raise ProviderConfigError(
            f"zai: {CLI_NAME} CLI not found in PATH; the zai providers work only "
            "in an environment with the z-ai CLI installed"
        ) from None
    except subprocess.TimeoutExpired:
        raise TransientError("zai: CLI call timed out") from None
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace").strip()[-STDERR_LIMIT:]
        raise TransientError(f"zai: CLI failed with rc={proc.returncode}: {tail}")
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ProviderError("zai: CLI output is missing or not valid JSON") from None
    if not isinstance(data, dict):
        raise ProviderError("zai: CLI output JSON must be an object")
    return data


def _tokens(data: dict) -> list[dict]:
    try:
        usage = data["usage"]
        total = int(usage.get("total_tokens") or 0) or int(usage.get("prompt_tokens", 0)) + int(
            usage.get("completion_tokens", 0)
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        total = 0
    return [{"kind": "tokens", "value": float(total)}]


class ZaiLLM(UsageReporter):
    """GLM chat completions via ``z-ai chat``; bills total tokens."""

    name = "zai"
    task = "llm"

    def __init__(
        self,
        model: str = "glm-4-plus",
        runner: CliRunner | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model = model
        self._runner = runner or default_runner(timeout_s)
        self.last_usage: list[dict] = []

    def complete(self, messages: list[Message]) -> str:
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        prompt = "\n\n".join(m.content for m in messages if m.role != "system")
        if not prompt and not system:
            raise ProviderError("zai: empty messages")
        with tempfile.TemporaryDirectory(prefix="razkadr-zai-") as tmp:
            out = Path(tmp) / "out.json"
            argv = [CLI_NAME, "chat", "-p", prompt, "-o", str(out)]
            if system:
                argv += ["-s", system]
            data = _run_cli(self._runner, argv, out)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("zai: unexpected chat response shape") from None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("zai: empty completion content")
        self.last_usage = _tokens(data)
        return content


class ZaiVLM(UsageReporter):
    """GLM vision via ``z-ai vision``; describes one image per call."""

    name = "zai"
    task = "vlm"

    def __init__(
        self,
        model: str = "glm-4-plus",
        runner: CliRunner | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model = model
        self._runner = runner or default_runner(timeout_s)
        self.last_usage: list[dict] = []

    def describe_image(self, path: Path, prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="razkadr-zai-") as tmp:
            out = Path(tmp) / "out.json"
            argv = [CLI_NAME, "vision", "-p", prompt, "-i", str(path), "-o", str(out)]
            data = _run_cli(self._runner, argv, out)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("zai: unexpected vision response shape") from None
        if not isinstance(content, str):
            raise ProviderError("zai: unexpected vision content type")
        # empty content is a VALID answer for OCR prompts ("no text on frame");
        # the CLI substitutes its own placeholder - map it back to ""
        if content == EMPTY_REPLY:
            content = ""
        self.last_usage = _tokens(data)
        return content


class ZaiASR(UsageReporter):
    """GLM ASR via ``z-ai asr``; the CLI reports flat text (no timecodes),
    so the part becomes a single segment 0..duration (audio parts are
    WAV slices, see stages/transcript.py). Language is not reported by
    the API - the configured/caller language is recorded as-is."""

    name = "zai"
    task = "asr"

    def __init__(
        self,
        model: str = "glm-asr",
        lang: str = "ru",
        runner: CliRunner | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model = model
        self.lang = lang
        self._runner = runner or default_runner(timeout_s)
        self.last_usage: list[dict] = []

    def transcribe(self, audio_path: Path, lang: str | None = None) -> Transcript:
        with tempfile.TemporaryDirectory(prefix="razkadr-zai-") as tmp:
            out = Path(tmp) / "out.json"
            argv = [CLI_NAME, "asr", "-f", str(audio_path), "-o", str(out)]
            data = _run_cli(self._runner, argv, out)
        text = data.get("text")
        if not isinstance(text, str):
            raise ProviderError("zai: unexpected ASR response shape")
        text = text.strip()
        self.last_usage = _tokens(data)
        seg_lang = lang or self.lang
        if not text:
            return Transcript(lang=seg_lang, segments=[])
        duration = wav_duration_s(audio_path)
        return Transcript(
            lang=seg_lang,
            segments=[TranscriptSegment(t_start=0.0, t_end=duration, text=text, lang=seg_lang)],
        )
