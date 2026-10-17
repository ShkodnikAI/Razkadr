"""zai bridge tests (naryad 085): hermetic on a stub runner, no network, no CLI.

The stub runner parses the argv (like the real CLI), writes a canned JSON
payload to the ``-o`` path and returns rc 0. Live tests (marked ``live``)
exercise the real z-ai CLI and are skipped unless the CLI is in PATH -
they never run in CI (no z-ai there), by design.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from razkadr.providers.base import ProviderConfigError, ProviderError, TransientError
from razkadr.providers.layer import ProviderLayer
from razkadr.providers.types import Message
from razkadr.providers.zai import CLI_NAME, ZaiASR, ZaiLLM, ZaiVLM
from razkadr.runctx import RunContext

CHAT_PAYLOAD = {
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Париж"}, "finish_reason": "stop"}],
    "model": "glm-4-plus",
    "usage": {"prompt_tokens": 19, "completion_tokens": 4, "total_tokens": 23},
}


class StubRunner:
    """Parses argv like the z-ai CLI: finds -o, writes the canned payload."""

    def __init__(self, payload: dict | None = None, rc: int = 0, stderr: bytes = b"") -> None:
        self.payload = payload if payload is not None else CHAT_PAYLOAD
        self.rc = rc
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(argv)
        if self.rc != 0:
            return subprocess.CompletedProcess(argv, self.rc, b"", self.stderr)
        out = Path(argv[argv.index("-o") + 1])
        out.write_text(json.dumps(self.payload), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, b"", b"")


def missing_cli_runner(argv: list[str]) -> subprocess.CompletedProcess:
    raise FileNotFoundError(f"[Errno 2] No such file or directory: '{CLI_NAME}'")


def make_wav(path: Path, seconds: float = 0.5, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


# -- LLM ------------------------------------------------------------------------


def test_llm_complete_returns_content_and_usage():
    runner = StubRunner()
    llm = ZaiLLM(runner=runner)
    text = llm.complete([Message(role="user", content="Столица Франции?")])
    assert text == "Париж"
    assert llm.last_usage == [{"kind": "tokens", "value": 23.0}]
    argv = runner.calls[0]
    assert argv[0] == CLI_NAME and argv[1] == "chat"
    assert "Столица Франции?" in argv
    assert "-o" in argv


def test_llm_system_message_becomes_flag():
    runner = StubRunner()
    llm = ZaiLLM(runner=runner)
    llm.complete([Message(role="system", content="Ты переводчик."), Message(role="user", content="Привет")])
    argv = runner.calls[0]
    assert "-s" in argv
    assert "Ты переводчик." in argv


def test_llm_cli_failure_is_transient():
    runner = StubRunner(rc=1, stderr=b"boom")
    llm = ZaiLLM(runner=runner)
    with pytest.raises(TransientError, match="rc=1"):
        llm.complete([Message(role="user", content="x")])


def test_llm_bad_output_is_provider_error(tmp_path):
    class BadWriter(StubRunner):
        def __call__(self, argv):
            Path(argv[argv.index("-o") + 1]).write_text("<html>", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, b"", b"")

    llm = ZaiLLM(runner=BadWriter())
    with pytest.raises(ProviderError, match="not valid JSON"):
        llm.complete([Message(role="user", content="x")])


def test_llm_missing_cli_is_config_error():
    llm = ZaiLLM(runner=missing_cli_runner)
    with pytest.raises(ProviderConfigError, match="z-ai CLI not found"):
        llm.complete([Message(role="user", content="x")])


# -- VLM -------------------------------------------------------------------------


def test_vlm_describes_image():
    runner = StubRunner()
    vlm = ZaiVLM(runner=runner)
    text = vlm.describe_image(Path("frame.jpg"), "Опиши изображение")
    assert text == "Париж"
    argv = runner.calls[0]
    assert argv[1] == "vision"
    assert "frame.jpg" in argv and "Опиши изображение" in argv
    assert vlm.last_usage == [{"kind": "tokens", "value": 23.0}]


def test_vlm_missing_cli_is_config_error():
    vlm = ZaiVLM(runner=missing_cli_runner)
    with pytest.raises(ProviderConfigError):
        vlm.describe_image(Path("frame.jpg"), "p")


def test_vlm_empty_content_is_valid_ocr_answer():
    # OCR prompt says "no text -> empty answer"; the CLI substitutes
    # "(Empty response)" - the adapter must return "" and not raise
    payload = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "(Empty response)"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 125},
    }
    vlm = ZaiVLM(runner=StubRunner(payload=payload))
    assert vlm.describe_image(Path("frame.jpg"), "p") == ""
    assert vlm.last_usage == [{"kind": "tokens", "value": 125.0}]


# -- ASR -------------------------------------------------------------------------


ASR_PAYLOAD = {"text": "Melt butter and add cream", "usage": {"total_tokens": 171}}


def test_asr_single_segment_zero_to_duration(tmp_path):
    runner = StubRunner(payload=ASR_PAYLOAD)
    wav = make_wav(tmp_path / "part_0001.wav", seconds=2.0)
    asr = ZaiASR(lang="en", runner=runner)
    transcript = asr.transcribe(wav, lang="en")
    assert transcript.lang == "en"
    assert len(transcript.segments) == 1
    seg = transcript.segments[0]
    assert seg.t_start == 0.0
    assert abs(seg.t_end - 2.0) < 0.01
    assert seg.text == "Melt butter and add cream"
    assert asr.last_usage == [{"kind": "tokens", "value": 171.0}]
    argv = runner.calls[0]
    assert argv[1] == "asr" and "-f" in argv and str(wav) in argv


def test_asr_empty_text_no_segments(tmp_path):
    runner = StubRunner(payload={"text": "  ", "usage": {"total_tokens": 5}})
    wav = make_wav(tmp_path / "part.wav")
    asr = ZaiASR(lang="en", runner=runner)
    transcript = asr.transcribe(wav)
    assert transcript.segments == []
    assert transcript.lang == "en"  # caller lang wins over constructor lang


def test_asr_bad_shape_is_provider_error(tmp_path):
    runner = StubRunner(payload={"no_text_here": 1})
    asr = ZaiASR(runner=runner)
    with pytest.raises(ProviderError, match="ASR response shape"):
        asr.transcribe(make_wav(tmp_path / "part.wav"))


# -- FACTORIES / layer integration ---------------------------------------------------


def test_factory_asr_zai(tmp_path):
    runner = StubRunner(payload=ASR_PAYLOAD)
    import razkadr.providers.zai as zai_mod

    original = zai_mod.default_runner
    zai_mod.default_runner = lambda timeout_s: runner
    try:
        config = {
            "keys": {"mode": "user"},
            "asr": {"provider": "zai", "lang": "en"},
            "cache": {"dir": str(tmp_path / "cache")},  # isolation from ~/.cache/razkadr/calls
        }
        ctx = RunContext(str(tmp_path / "runs"))
        layer = ProviderLayer(config, ctx)
        wav = make_wav(tmp_path / "p.wav")
        transcript = layer.asr(stage="transcript").transcribe(wav, lang="en")
        assert transcript.segments[0].text == "Melt butter and add cream"
        assert runner.calls, "CLI must be invoked through the layer"
    finally:
        zai_mod.default_runner = original


def test_factory_llm_zai_via_layer(tmp_path):
    runner = StubRunner()
    import razkadr.providers.zai as zai_mod

    original = zai_mod.default_runner
    zai_mod.default_runner = lambda timeout_s: runner
    try:
        config = {
            "keys": {"mode": "user"},
            "llm": {"provider": "zai"},
            "cache": {"dir": str(tmp_path / "cache")},
        }
        ctx = RunContext(str(tmp_path / "runs"))
        layer = ProviderLayer(config, ctx)
        answer = layer.llm(stage="chunks").complete([Message(role="user", content="hi")])
        assert answer == "Париж"
        assert layer.llm(stage="chunks").provider.last_usage == [{"kind": "tokens", "value": 23.0}]
    finally:
        zai_mod.default_runner = original


def test_factory_vlm_zai_no_keys_required(tmp_path):
    runner = StubRunner()
    import razkadr.providers.zai as zai_mod

    original = zai_mod.default_runner
    zai_mod.default_runner = lambda timeout_s: runner
    try:
        config = {
            "keys": {"mode": "user"},
            "vlm": {"provider": "zai"},
            "cache": {"dir": str(tmp_path / "cache")},
        }
        ctx = RunContext(str(tmp_path / "runs"))
        layer = ProviderLayer(config, ctx)
        frame = tmp_path / "f.jpg"
        frame.write_bytes(b"\xff\xd8fakejpeg")
        text = layer.vlm(stage="ocr").describe_image(frame, "describe")
        assert text == "Париж"
    finally:
        zai_mod.default_runner = original


# -- live (skipped without the z-ai CLI; never runs in CI) -----------------------------


pytestmark_live = pytest.mark.live

requires_cli = pytest.mark.skipif(shutil.which(CLI_NAME) is None, reason="z-ai CLI not in PATH")


@requires_cli
@pytest.mark.live
def test_live_chat():
    llm = ZaiLLM()
    text = llm.complete([Message(role="user", content="Ответь одним словом: столица Франции?")])
    assert "ариж" in text or "Paris" in text
    assert llm.last_usage and llm.last_usage[0]["kind"] == "tokens"


@requires_cli
@pytest.mark.live
def test_live_vision(tmp_path):
    vlm = ZaiVLM()
    frame = Path(__file__).parent / "fixtures" / "visual" / "frames" / "f_title.jpg"
    if not frame.is_file():
        pytest.skip("fixture frame missing")
    text = vlm.describe_image(frame, "Что изображено? Одно предложение.")
    assert text.strip()
    assert vlm.last_usage
