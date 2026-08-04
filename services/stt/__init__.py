"""Speech-to-text providers and the routing between them.

Sarvam handles the Indian languages it specialises in; OpenAI handles everything
else. `pipeline.transcribe` is the only entry point the rest of the app needs.
"""

from .languages import (
    AUTO,
    LOCALE_DEPENDENT_LANGUAGES,
    SARVAM_LANGUAGES,
    is_locale_dependent,
    is_sarvam_language,
    normalise_language,
    to_sarvam_mode,
)
from .pipeline import transcribe
from .router import OPENAI, SARVAM, provider_chain, select_fallback, select_provider
from .types import (
    SttProvider,
    TranscriptSegment,
    TranscriptionError,
    TranscriptionResult,
)

__all__ = [
    "AUTO",
    "LOCALE_DEPENDENT_LANGUAGES",
    "OPENAI",
    "SARVAM",
    "SARVAM_LANGUAGES",
    "SttProvider",
    "TranscriptSegment",
    "TranscriptionError",
    "TranscriptionResult",
    "is_locale_dependent",
    "is_sarvam_language",
    "normalise_language",
    "provider_chain",
    "select_fallback",
    "select_provider",
    "to_sarvam_mode",
    "transcribe",
]
