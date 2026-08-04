"""Shared vocabulary for the speech-to-text providers.

Every provider returns a `TranscriptionResult` so the rest of the backend never
has to know which vendor produced a transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol


class TranscriptionError(RuntimeError):
    """Message is safe to show the user."""


# Sarvam labels speakers "SPEAKER_00"; OpenAI labels them "A", "B". Without
# normalising, the same notes list would render two different styles.
_SPEAKER_PREFIX = re.compile(r"^\s*speaker[\s_\-]*", re.IGNORECASE)


def speaker_label(raw: Optional[str]) -> Optional[str]:
    """Turns a provider speaker id into a single 'Speaker X' display form."""
    if raw is None:
        return None

    cleaned = _SPEAKER_PREFIX.sub("", str(raw)).strip()
    if not cleaned:
        return None

    if cleaned.isdigit():
        # Providers count from zero; humans do not.
        return f"Speaker {int(cleaned) + 1}"

    return f"Speaker {cleaned.upper()}"


@dataclass(frozen=True)
class TranscriptSegment:
    """One contiguous run of speech. [speaker] is None when not diarized."""

    text: str
    speaker: Optional[str] = None


@dataclass(frozen=True)
class TranscriptionResult:
    segments: List[TranscriptSegment]
    provider: str
    language: Optional[str] = None

    def to_text(self) -> str:
        """Renders the transcript the way notes have always stored it."""
        lines = []
        for segment in self.segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            lines.append(f"{segment.speaker}: {text}" if segment.speaker else text)
        return "\n".join(lines).strip()

    @property
    def is_empty(self) -> bool:
        return not self.to_text()


class SttProvider(Protocol):
    """What the pipeline needs from a transcription vendor."""

    name: str

    def is_available(self) -> bool:
        """False when the vendor's API key is not configured."""

    def transcribe(
        self, audio_path: str, language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribes a local audio file, raising TranscriptionError on failure."""
