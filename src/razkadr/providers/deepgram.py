"""Deepgram adapter: ASR via /v1/listen (raw audio body).

Naryad 037, ADR-0006: fallback ASR ``nova-3`` (multilingual) — an
independent vendor with per-minute billing. The transcript stage feeds
PCM WAV parts; the request body is the raw WAV bytes with
``Content-Type: audio/wav`` and ``Authorization: Token <key>`` (the key
never appears in the URL). Segments come from paragraph grouping
(``paragraphs=true``); if the provider omits them, words are grouped by
speech pauses (>= 0.8 s). Usage is billed in seconds of audio — the
``metadata.duration`` field of the response, cross-checked against the
WAV header.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from razkadr.providers.base import ProviderError, UsageReporter
from razkadr.providers.gemini import normalize_lang
from razkadr.providers.rest import (
    DEFAULT_TIMEOUT_S,
    Request,
    Transport,
    call_json,
    urllib_transport,
    wav_duration_s,
)
from razkadr.providers.types import Transcript, TranscriptSegment

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

WORD_GAP_S = 0.8  # silence gap that starts a new segment when paragraphs are absent


class DeepgramASR(UsageReporter):
    """Fallback ASR adapter (ADR-0006); bills seconds of audio per call."""

    name = "deepgram"
    task = "asr"

    def __init__(
        self,
        model: str = "nova-3",
        api_key: str = "",
        transport: Transport = urllib_transport,
        base_url: str = DEEPGRAM_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._transport = transport
        self._base_url = base_url
        self._timeout_s = timeout_s
        self.last_usage: list[dict] = []

    def transcribe(self, audio_path: Path, lang: str | None = None) -> Transcript:
        audio = Path(audio_path).read_bytes()
        duration = wav_duration_s(Path(audio_path))
        params = urllib.parse.urlencode(
            {
                "model": self.model,
                "smart_format": "true",
                "punctuate": "true",
                "paragraphs": "true",
                # nova-3 multilingual: "multi" asks the provider to detect the language
                "language": lang if lang else "multi",
            }
        )
        request = Request(
            method="POST",
            url=f"{self._base_url}?{params}",
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"},
            body=audio,
            timeout_s=self._timeout_s,
        )
        data = call_json(self._transport, self.name, request)
        transcript = self._parse(data, lang)
        self._report([{"kind": "seconds", "value": round(self._billed_seconds(data, duration), 3)}])
        return transcript

    def _billed_seconds(self, data: dict, fallback: float) -> float:
        """Provider-side duration (what Deepgram bills) with a WAV-header fallback."""
        try:
            value = float(data["metadata"]["duration"])
            if value > 0:
                return value
        except (KeyError, TypeError, ValueError):
            pass
        return fallback

    def _parse(self, data: dict, lang: str | None) -> Transcript:
        try:
            channel = data["results"]["channels"][0]
            alternative = channel["alternatives"][0]
            detected = channel.get("detected_language") or lang
            out_lang = normalize_lang(detected)
        except (KeyError, IndexError, TypeError):
            raise ProviderError("deepgram: unexpected response shape; no transcript parsed") from None

        segments = self._segments_from_paragraphs(alternative, out_lang)
        if segments is None:
            segments = self._segments_from_words(alternative, out_lang)
        return Transcript(lang=out_lang, segments=segments)

    def _segments_from_paragraphs(self, alternative: dict, out_lang: str) -> list[TranscriptSegment] | None:
        try:
            paragraphs = alternative["paragraphs"]["paragraphs"]
        except (KeyError, TypeError):
            return None
        if not isinstance(paragraphs, list):
            return None
        segments: list[TranscriptSegment] = []
        for para in paragraphs:
            try:
                t_start = float(para["start"])
                t_end = float(para["end"])
                text = " ".join(str(s["text"]).strip() for s in para.get("sentences", [])).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if not text or t_end <= t_start:
                continue
            segments.append(TranscriptSegment(t_start=t_start, t_end=t_end, text=text, lang=out_lang))
        segments.sort(key=lambda s: s.t_start)
        return segments

    def _segments_from_words(self, alternative: dict, out_lang: str) -> list[TranscriptSegment]:
        """Group word timings into segments by speech pauses (>= WORD_GAP_S)."""
        words = alternative.get("words")
        if not isinstance(words, list):
            raise ProviderError("deepgram: response has neither paragraphs nor word timings")
        segments: list[TranscriptSegment] = []
        current: list[str] = []
        t_start = t_end = prev_end = 0.0
        for word in words:
            try:
                w_start = float(word["start"])
                w_end = float(word["end"])
                w_text = str(word.get("punctuated_word") or word.get("word") or "").strip()
            except (KeyError, TypeError, ValueError):
                continue
            if not w_text:
                continue
            if current and w_start - prev_end >= WORD_GAP_S:
                segments.append(
                    TranscriptSegment(t_start=t_start, t_end=t_end, text=" ".join(current), lang=out_lang)
                )
                current = []
            if not current:
                t_start = w_start
            current.append(w_text)
            t_end = w_end
            prev_end = w_end
        if current:
            segments.append(
                TranscriptSegment(t_start=t_start, t_end=t_end, text=" ".join(current), lang=out_lang)
            )
        return segments
