"""Chooses which STT provider handles a recording.

Deliberately pure: no I/O, no SDK imports, no API keys. This is the piece most
likely to be subtly wrong, and it is the piece worth testing exhaustively.
"""

from __future__ import annotations

from typing import Optional

from .languages import (
    is_indian_locale,
    is_locale_dependent,
    is_sarvam_language,
    normalise_language,
)

SARVAM = "sarvam"
OPENAI = "openai"


def select_provider(
    language: Optional[str] = None, locale: Optional[str] = None
) -> str:
    """The provider best suited to this recording.

    Rules, in order:

    1. The user pinned a locale-dependent language (Urdu). Sarvam's model is
       trained on Indian Urdu, so it wins only for speakers actually in India.
    2. The user pinned any other language. Route on coverage: Sarvam for the 23
       Indian languages saaras:v3 handles, OpenAI for everything else.
    3. Auto-detect with a locale that is clearly not Indian. OpenAI, which
       covers ~99 languages.
    4. Auto-detect with an Indian locale, or no locale at all. Sarvam.

    Case 4 folds "no locale" in with India on purpose. Clients that predate the
    locale field (older app installs, the web app) send nothing, and they are
    overwhelmingly the existing Indian user base whose Hinglish accuracy depends
    on Sarvam. Defaulting them to OpenAI would silently regress the users we
    already have in order to serve ones we do not have yet.
    """
    normalised = normalise_language(language)

    if normalised:
        if is_locale_dependent(normalised):
            return SARVAM if is_indian_locale(locale) else OPENAI
        return SARVAM if is_sarvam_language(normalised) else OPENAI

    if locale and not is_indian_locale(locale):
        return OPENAI

    return SARVAM


def select_fallback(primary: str, language: Optional[str] = None) -> Optional[str]:
    """The provider to retry with when [primary] fails, or None to give up.

    Asymmetric on purpose, and the asymmetry is the point:

    - Sarvam failing retries on OpenAI, which covers the same languages.
    - OpenAI failing does not retry at all. Reaching OpenAI means the language is
      one Sarvam does not know, and Sarvam answers unknown languages with
      confident nonsense rather than an error — which would be saved as the
      user's transcript. A clear failure beats a plausible-looking wrong answer.
    """
    if primary == SARVAM:
        return OPENAI

    return None


def provider_chain(
    language: Optional[str] = None, locale: Optional[str] = None
) -> list:
    """Ordered providers to attempt for this recording."""
    primary = select_provider(language, locale)
    fallback = select_fallback(primary, language)
    return [primary] if fallback is None else [primary, fallback]
