"""OpenAI transcription, for every language Sarvam does not cover.

Uses `gpt-4o-transcribe-diarize` rather than the cheaper default specifically
because it returns speaker labels. Notes have always rendered as "Speaker 1: …",
and matching that means nothing downstream has to change when a recording is
routed here instead of to Sarvam.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from .languages import to_openai_code
from .types import (
    TranscriptionError,
    TranscriptionResult,
    TranscriptSegment,
    speaker_label,
)

logger = logging.getLogger("STT.OpenAI")

MODEL = "gpt-4o-transcribe-diarize"

# The transcriptions endpoint rejects anything larger. Recordings are made at
# 16 kHz mono, which keeps roughly 100 minutes under this ceiling.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise TranscriptionError("Transcription is not configured on the server.")
        _client = OpenAI(api_key=api_key)
    return _client


def _parse(response, language: Optional[str]) -> TranscriptionResult:
    """Reads a `diarized_json` response into a provider-neutral result."""
    segments: List[TranscriptSegment] = []

    for segment in getattr(response, "segments", None) or []:
        text = (getattr(segment, "text", "") or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                text=text,
                speaker=speaker_label(getattr(segment, "speaker", None)),
            )
        )

    # Diarization can come back empty on very short or single-speaker clips,
    # where the flat transcript is still perfectly usable.
    if not segments:
        plain = (getattr(response, "text", "") or "").strip()
        if plain:
            segments = [TranscriptSegment(text=plain)]

    return TranscriptionResult(segments=segments, provider="openai", language=language)


class OpenAIProvider:
    name = "openai"

    def is_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def transcribe(
        self, audio_path: str, language: Optional[str] = None
    ) -> TranscriptionResult:
        size = os.path.getsize(audio_path)
        if size > MAX_UPLOAD_BYTES:
            raise TranscriptionError(
                "That recording is too long to transcribe in this language. "
                "Please split it into shorter recordings."
            )

        language_code = to_openai_code(language)
        logger.info(
            "OpenAI transcription start (language=%s, bytes=%s)",
            language_code or "auto",
            size,
        )

        try:
            with open(audio_path, "rb") as handle:
                kwargs = {
                    "model": MODEL,
                    "file": handle,
                    "response_format": "diarized_json",
                    # Required for anything over 30 seconds on this model.
                    "chunking_strategy": "auto",
                }
                if language_code:
                    kwargs["language"] = language_code

                response = _get_client().audio.transcriptions.create(**kwargs)

            return _parse(response, language)

        except TranscriptionError:
            raise
        except Exception as exc:
            logger.error("OpenAI transcription error: %s", exc, exc_info=True)
            raise TranscriptionError("Transcription failed. Please try again.") from exc
