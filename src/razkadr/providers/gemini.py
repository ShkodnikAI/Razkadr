"""Google Gemini adapter: ASR via generateContent with a JSON response schema.

Naryad 037, ADR-0006: default ASR ``gemini-3.5-transcribe``. The transcript
stage feeds PCM WAV parts (16 kHz mono, ``part_length_s`` = 120 by default),
which fit the inline base64 request. The model answers with strict JSON
(detected language + segments with second timestamps) enforced by
``response_schema``; temperature 0 for reproducibility. Usage is billed in
seconds of audio (``docs/providers.md`` tariff units); the API key travels
in the ``x-goog-api-key`` header, never in the URL.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from razkadr.providers.base import ProviderError, UsageReporter
from razkadr.providers.rest import (
    DEFAULT_TIMEOUT_S,
    Request,
    Transport,
    call_json,
    urllib_transport,
    wav_duration_s,
)
from razkadr.providers.types import Transcript, TranscriptSegment

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "lang": {"type": "string", "description": "BCP-47 code of the detected speech language"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t_start": {"type": "number", "description": "seconds from audio start"},
                    "t_end": {"type": "number", "description": "seconds from audio start"},
                    "text": {"type": "string"},
                },
                "required": ["t_start", "t_end", "text"],
            },
        },
    },
    "required": ["lang", "segments"],
}

PROMPT = (
    "Transcribe the speech in this audio recording. Split the text into "
    "segments of one to three sentences each. Return only JSON matching the "
    "schema: `lang` is the BCP-47 code of the spoken language, each segment "
    "has `t_start`/`t_end` in seconds from the start of the audio and the "
    "verbatim `text`. Do not translate, do not invent words."
)


def normalize_lang(tag: str | None, fallback: str = "und") -> str:
    """BCP-47 primary subtag, lowercase ("ru-RU" -> "ru", empty -> fallback)."""
    if not tag or not tag.strip():
        return fallback
    primary = tag.strip().split("-")[0].lower()
    return primary if primary.isascii() and primary.isalpha() else fallback


class GeminiASR(UsageReporter):
    """Default ASR adapter (ADR-0006); bills seconds of audio per call."""

    name = "gemini"
    task = "asr"

    def __init__(
        self,
        model: str = "gemini-3.5-transcribe",
        api_key: str = "",
        transport: Transport = urllib_transport,
        base_url: str = GEMINI_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._transport = transport
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self.last_usage: list[dict] = []

    def transcribe(self, audio_path: Path, lang: str | None = None) -> Transcript:
        audio = Path(audio_path).read_bytes()
        duration = wav_duration_s(Path(audio_path))
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                f"{PROMPT} The spoken language is {lang}." if lang else PROMPT
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(audio).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": RESPONSE_SCHEMA,
                "temperature": 0,
            },
        }
        request = self._request(payload)
        data = call_json(self._transport, self.name, request)
        transcript = self._parse(data, lang)
        self._report([{"kind": "seconds", "value": round(duration, 3)}])
        return transcript

    def _request(self, payload: dict) -> Request:
        return Request(
            method="POST",
            url=f"{self._base_url}/{self.model}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            body=json.dumps(payload).encode("utf-8"),
            timeout_s=self._timeout_s,
        )

    def _parse(self, data: dict, lang: str | None) -> Transcript:
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            out_lang = normalize_lang(parsed.get("lang") or lang)
            raw_segments = parsed.get("segments", [])
            if not isinstance(raw_segments, list):
                raise TypeError("segments must be a list")
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            raise ProviderError("gemini: unexpected response shape; no transcript parsed") from None
        segments: list[TranscriptSegment] = []
        for item in raw_segments:
            try:
                t_start = float(item["t_start"])
                t_end = float(item["t_end"])
                seg_text = str(item["text"]).strip()
            except (KeyError, TypeError, ValueError):
                continue  # a malformed segment is dropped, not invented
            if not seg_text or t_end <= t_start:
                continue
            segments.append(TranscriptSegment(t_start=t_start, t_end=t_end, text=seg_text, lang=out_lang))
        segments.sort(key=lambda s: s.t_start)
        return Transcript(lang=out_lang, segments=segments)
