"""Runs a recording through the chosen provider, retrying on the other one."""

from __future__ import annotations

import logging
from typing import Optional

from ..observability import capture_exception
from .openai_provider import OpenAIProvider
from .router import OPENAI, SARVAM, provider_chain
from .sarvam_provider import SarvamProvider
from .types import TranscriptionError, TranscriptionResult

logger = logging.getLogger("STT.Pipeline")

_PROVIDERS = {
    SARVAM: SarvamProvider,
    OPENAI: OpenAIProvider,
}


def transcribe(
    audio_path: str,
    language: Optional[str] = None,
    locale: Optional[str] = None,
) -> TranscriptionResult:
    """Transcribes a local file, choosing a provider from [language]/[locale].

    A provider that raises is retried on its fallback, so a vendor outage
    degrades transcription quality instead of failing the user's recording.
    """
    chain = provider_chain(language, locale)
    last_error: Optional[Exception] = None
    empty_result: Optional[TranscriptionResult] = None

    for index, name in enumerate(chain):
        provider = _PROVIDERS[name]()

        if not provider.is_available():
            logger.warning("STT provider %s has no API key, skipping.", name)
            continue

        if index > 0:
            logger.warning("Falling back to STT provider %s.", name)

        try:
            result = provider.transcribe(audio_path, language=language)
        except TranscriptionError as exc:
            logger.error("STT provider %s failed: %s", name, exc)
            capture_exception(exc, stt_provider=name)
            last_error = exc
            continue

        if not result.is_empty:
            return result

        # A silent recording is legitimately empty, but so is a provider that
        # choked quietly. Keep the empty result and let the fallback try.
        logger.warning("STT provider %s returned an empty transcript.", name)
        if empty_result is None:
            empty_result = result

    if empty_result is not None:
        return empty_result

    if last_error is not None:
        raise last_error

    raise TranscriptionError("Transcription is not configured on the server.")
