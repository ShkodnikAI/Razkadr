"""Unified result types shared by all providers (provider-independent)."""

from __future__ import annotations

from dataclasses import dataclass, field

Vector = list[float]

VALID_ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class Message:
    """One chat message for the LLM task."""

    role: str  # system | user | assistant
    content: str

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"message role must be one of {VALID_ROLES}, got {self.role!r}")


@dataclass(frozen=True)
class TranscriptSegment:
    t_start: float
    t_end: float
    text: str
    lang: str
    speaker: str | None = None


@dataclass(frozen=True)
class Transcript:
    """ASR result; JSON shape matches schemas/transcript.v1.json."""

    lang: str
    segments: list[TranscriptSegment] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "lang": self.lang,
            "segments": [
                {
                    "t_start": s.t_start,
                    "t_end": s.t_end,
                    "text": s.text,
                    "lang": s.lang,
                    **({"speaker": s.speaker} if s.speaker else {}),
                }
                for s in self.segments
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> Transcript:
        return cls(
            lang=data["lang"],
            segments=[
                TranscriptSegment(
                    t_start=s["t_start"],
                    t_end=s["t_end"],
                    text=s["text"],
                    lang=s["lang"],
                    speaker=s.get("speaker"),
                )
                for s in data.get("segments", [])
            ],
        )
