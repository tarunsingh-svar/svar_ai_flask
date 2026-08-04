"""Sarvam AI transcription, for Hindi and the other Indian languages.

Kept as the default for Indian speech because saaras handles code-mixed Hinglish
noticeably better than general-purpose models do.

Runs saaras:v3 rather than saarika:v2.5: it covers 23 Indian languages instead of
11, and its `mode` parameter produces romanized output natively, which replaced a
brittle dig through per-language transliteration objects in the response.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
from typing import List, Optional

from .languages import to_sarvam_code, to_sarvam_mode
from .types import (
    TranscriptionError,
    TranscriptionResult,
    TranscriptSegment,
    speaker_label,
)

logger = logging.getLogger("STT.Sarvam")

MODEL = "saaras:v3"

_client = None


def _get_client():
    global _client
    if _client is None:
        from sarvamai import SarvamAI

        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise TranscriptionError("Transcription is not configured on the server.")
        _client = SarvamAI(api_subscription_key=api_key)
    return _client


def _entry_text(entry: dict) -> str:
    """Best available text for one diarized entry.

    On saaras:v3 the requested `mode` has already been applied, so `transcript`
    is authoritative. The nested lookups below are saarika:v2.5's shape, kept as
    a fallback because v3's diarization is still flagged beta by Sarvam and may
    not populate every field consistently.
    """
    transcript = (entry.get("transcript") or "").strip()
    if transcript:
        return transcript

    transcriptions = (entry.get("transcription_output") or {}).get("transcriptions") or {}

    transliterated = (
        ((transcriptions.get("hi-IN") or {}).get("transliteration") or {})
        .get("en", {})
        .get("text", "")
    )
    if transliterated:
        return transliterated

    return (transcriptions.get("en-IN") or {}).get("text", "") or ""


def _parse(payload: dict, language: Optional[str]) -> TranscriptionResult:
    """Turns Sarvam's job output into a provider-neutral result."""
    entries = ((payload.get("diarized_transcript") or {}).get("entries")) or []

    if not entries:
        plain = (payload.get("transcript") or "").strip()
        segments: List[TranscriptSegment] = (
            [TranscriptSegment(text=plain)] if plain else []
        )
    else:
        segments = []
        for entry in entries:
            text = _entry_text(entry).strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    text=text,
                    speaker=speaker_label(entry.get("speaker_id")),
                )
            )

    return TranscriptionResult(segments=segments, provider="sarvam", language=language)


class SarvamProvider:
    name = "sarvam"

    def is_available(self) -> bool:
        return bool(os.getenv("SARVAM_API_KEY"))

    def transcribe(
        self, audio_path: str, language: Optional[str] = None
    ) -> TranscriptionResult:
        out_dir = os.path.join(os.path.dirname(audio_path), "sarvam_out")
        language_code = to_sarvam_code(language)
        mode = to_sarvam_mode(language)

        logger.info(
            "Sarvam transcription start (model=%s, language_code=%s, mode=%s)",
            MODEL,
            language_code or "auto",
            mode,
        )

        try:
            os.makedirs(out_dir, exist_ok=True)

            job_kwargs = {
                "model": MODEL,
                "mode": mode,
                "with_diarization": True,
            }
            if language_code:
                job_kwargs["language_code"] = language_code

            job = _get_client().speech_to_text_job.create_job(**job_kwargs)
            job.upload_files([audio_path])
            job.start()
            job.wait_until_complete()

            if job.is_failed():
                raise TranscriptionError(
                    "The transcription provider rejected this recording."
                )

            job.download_outputs(out_dir)

            json_files = glob.glob(os.path.join(out_dir, "*.json"))
            if not json_files:
                raise TranscriptionError(
                    "No transcript was produced for this recording."
                )

            with open(json_files[0], "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            return _parse(payload, language)

        except TranscriptionError:
            raise
        except Exception as exc:
            logger.error("Sarvam error: %s", exc, exc_info=True)
            raise TranscriptionError("Transcription failed. Please try again.") from exc
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
