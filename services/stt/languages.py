"""Language codes and the coverage boundary between the two STT providers.

Kept free of vendor SDK imports so the routing rules can be unit-tested without
network access or API keys.
"""

from __future__ import annotations

from typing import Optional

# Sent by the client when the user has not pinned a language.
AUTO = "auto"

# Languages Sarvam's saaras:v3 model transcribes, as bare ISO-639-1/-3 codes.
#
# Note the deliberate absence of plain "en": Sarvam handles Indian-accented
# English well, but an American or British speaker is better served by OpenAI.
# Indian English is therefore requested explicitly as "en-IN".
SARVAM_LANGUAGES = {
    # Supported by both saarika:v2.5 and saaras:v3.
    "bn",  # Bengali
    "gu",  # Gujarati
    "hi",  # Hindi
    "kn",  # Kannada
    "ml",  # Malayalam
    "mr",  # Marathi
    "or",  # Odia
    "pa",  # Punjabi
    "ta",  # Tamil
    "te",  # Telugu
    # Added by saaras:v3.
    "as",  # Assamese
    "brx",  # Bodo
    "doi",  # Dogri
    "kok",  # Konkani
    "ks",  # Kashmiri
    "mai",  # Maithili
    "mni",  # Manipuri
    "ne",  # Nepali
    "sa",  # Sanskrit
    "sat",  # Santali
    "sd",  # Sindhi
    "ur",  # Urdu
}

# Languages with large populations of speakers outside India, where Sarvam's
# India-trained model is the right choice only for speakers who are actually in
# India. These route on locale rather than on language alone.
LOCALE_DEPENDENT_LANGUAGES = {"ur"}

# Sarvam's own code for each language.
_SARVAM_CODES = {
    "as": "as-IN",
    "bn": "bn-IN",
    "brx": "brx-IN",
    "doi": "doi-IN",
    "gu": "gu-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "kok": "kok-IN",
    "ks": "ks-IN",
    "mai": "mai-IN",
    "ml": "ml-IN",
    "mni": "mni-IN",
    "mr": "mr-IN",
    "ne": "ne-IN",
    "or": "od-IN",  # Sarvam spells Odia "od"
    "pa": "pa-IN",
    "sa": "sa-IN",
    "sat": "sat-IN",
    "sd": "sd-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ur": "ur-IN",
    "en-in": "en-IN",
}

# "od" is Sarvam's spelling of Odia; accept it as an alias for the ISO code.
_LANGUAGE_ALIASES = {"od": "or"}

# saaras:v3 output modes.
MODE_TRANSCRIBE = "transcribe"
MODE_TRANSLIT = "translit"

# Languages rendered in Latin script rather than their own.
#
# Only Hindi. Romanized Hinglish ("mera phone number hai 9840950950") is what
# the app has always produced and what its existing users read fluently, so
# switching them to Devanagari would be a visible regression. Every other Indian
# language gets its native script, because romanized Tamil serves nobody.
_TRANSLITERATED_LANGUAGES = {"hi"}


def normalise_language(language: Optional[str]) -> Optional[str]:
    """Lower-cases and canonicalises a client-supplied language code.

    Returns None for absent or auto-detect, so callers can treat "the user did
    not choose" as a single case.
    """
    if not language:
        return None

    cleaned = language.strip().lower().replace("_", "-")
    if not cleaned or cleaned == AUTO:
        return None

    # Keep the region only where it changes the routing decision.
    if cleaned == "en-in":
        return cleaned

    base = cleaned.split("-")[0]
    return _LANGUAGE_ALIASES.get(base, base)


def is_sarvam_language(language: Optional[str]) -> bool:
    """True when Sarvam can transcribe this language at all.

    Coverage only. Whether Sarvam is the right choice also depends on locale for
    the languages in [LOCALE_DEPENDENT_LANGUAGES]; see `router.select_provider`.
    """
    normalised = normalise_language(language)
    if not normalised:
        return False
    if normalised == "en-in":
        return True
    return normalised in SARVAM_LANGUAGES


def is_locale_dependent(language: Optional[str]) -> bool:
    """True when this language needs the locale to pick a provider."""
    return normalise_language(language) in LOCALE_DEPENDENT_LANGUAGES


def to_sarvam_code(language: Optional[str]) -> Optional[str]:
    """Sarvam's regional code for a language, or None to let it auto-detect."""
    normalised = normalise_language(language)
    if not normalised:
        return None
    return _SARVAM_CODES.get(normalised)


def to_sarvam_mode(language: Optional[str]) -> str:
    """The saaras:v3 output mode for a language.

    Auto-detect resolves to transliteration. Sarvam picks the language itself in
    that case, so a per-language mode is not available — and the users who reach
    this path without pinning anything are overwhelmingly the existing Hindi and
    Hinglish speakers whose transcripts are romanized today.
    """
    normalised = normalise_language(language)
    if normalised is None or normalised in _TRANSLITERATED_LANGUAGES:
        return MODE_TRANSLIT
    return MODE_TRANSCRIBE


def to_openai_code(language: Optional[str]) -> Optional[str]:
    """ISO-639-1 code for OpenAI's `language` hint, or None to auto-detect."""
    normalised = normalise_language(language)
    if not normalised:
        return None
    return normalised.split("-")[0]


def is_indian_locale(locale: Optional[str]) -> bool:
    """True when a device locale suggests the speaker is in India.

    Used when the user has not pinned a language, and to break the tie for the
    locale-dependent languages. Rules, in order:

    1. An explicit "IN" region means India, whatever the language.
    2. A locale-dependent language without an explicit "IN" region is treated as
       outside India, so `ur_PK` and a bare `ur` both go to OpenAI.
    3. Any other Indian language counts regardless of region, so a Hindi speaker
       abroad still reaches the provider that handles Hinglish best.
    """
    if not locale:
        return False

    cleaned = locale.strip().lower().replace("_", "-")
    if not cleaned:
        return False

    parts = cleaned.split("-")
    base = _LANGUAGE_ALIASES.get(parts[0], parts[0])

    if any(part == "in" for part in parts[1:]):
        return True

    if base in LOCALE_DEPENDENT_LANGUAGES:
        return False

    return base in SARVAM_LANGUAGES
