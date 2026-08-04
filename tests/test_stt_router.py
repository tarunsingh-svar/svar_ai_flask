"""The provider routing rules.

This is the highest-value test file in the suite: `select_provider` is pure, it
encodes every decision about which vendor sees a recording, and getting it wrong
is silent — the user just gets a bad transcript.
"""

import pytest

from services.stt.languages import (
    MODE_TRANSCRIBE,
    MODE_TRANSLIT,
    is_indian_locale,
    is_sarvam_language,
    normalise_language,
    to_openai_code,
    to_sarvam_code,
    to_sarvam_mode,
)
from services.stt.router import OPENAI, SARVAM, provider_chain, select_fallback, select_provider


class TestExplicitLanguage:
    @pytest.mark.parametrize(
        "language",
        # The 11 saarika:v2.5 languages plus the 12 added by saaras:v3, minus
        # Urdu which routes on locale and is covered separately.
        [
            "hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "en-IN",
            "as", "brx", "doi", "kok", "ks", "mai", "mni", "ne", "sa", "sat", "sd",
        ],
    )
    def test_indian_languages_go_to_sarvam(self, language):
        assert select_provider(language=language) == SARVAM

    @pytest.mark.parametrize(
        "language", ["de", "es", "fr", "ja", "zh", "ar", "pt", "ru", "ko", "vi"]
    )
    def test_other_languages_go_to_openai(self, language):
        assert select_provider(language=language) == OPENAI

    def test_plain_english_goes_to_openai(self):
        """Sarvam only claims Indian-accented English, requested as en-IN."""
        assert select_provider(language="en") == OPENAI

    def test_explicit_language_overrides_locale(self):
        assert select_provider(language="de", locale="en_IN") == OPENAI
        assert select_provider(language="hi", locale="de_DE") == SARVAM

    def test_case_and_separator_insensitive(self):
        assert select_provider(language="HI") == SARVAM
        assert select_provider(language="en_IN") == SARVAM
        assert select_provider(language="  hi  ") == SARVAM

    def test_region_is_ignored_outside_english(self):
        assert select_provider(language="ta-LK") == SARVAM
        assert select_provider(language="es-MX") == OPENAI


class TestAutoDetect:
    @pytest.mark.parametrize("language", [None, "", "auto", "AUTO"])
    def test_indian_locale_goes_to_sarvam(self, language):
        assert select_provider(language=language, locale="en_IN") == SARVAM

    @pytest.mark.parametrize(
        "locale", ["en_US", "en_GB", "de_DE", "fr_FR", "ja_JP", "pt_BR"]
    )
    def test_foreign_locale_goes_to_openai(self, locale):
        assert select_provider(language="auto", locale=locale) == OPENAI

    def test_indian_language_locale_goes_to_sarvam_regardless_of_region(self):
        """A Hindi speaker abroad still wants the Hinglish-capable provider."""
        assert select_provider(locale="hi_US") == SARVAM
        assert select_provider(locale="ta_LK") == SARVAM

    def test_no_locale_at_all_stays_on_sarvam(self):
        """Clients predating the locale field are the existing Indian users.

        Defaulting them to OpenAI would regress Hinglish accuracy for the users
        we already have in order to serve ones we do not have yet.
        """
        assert select_provider() == SARVAM
        assert select_provider(language="auto", locale=None) == SARVAM
        assert select_provider(language=None, locale="") == SARVAM


class TestFallback:
    def test_sarvam_falls_back_to_openai(self):
        assert select_fallback(SARVAM) == OPENAI

    def test_openai_never_falls_back(self):
        """Reaching OpenAI means Sarvam does not know the language, and Sarvam
        answers unknown languages with confident nonsense rather than an error.
        That nonsense would be saved as the user's transcript, so a clear
        failure is the better outcome."""
        assert select_fallback(OPENAI, language="de") is None
        assert select_fallback(OPENAI, language="ja") is None
        assert select_fallback(OPENAI, language=None) is None
        assert select_fallback(OPENAI, language="auto") is None

    def test_openai_is_never_primary_for_a_sarvam_language(self):
        """Guards the assumption the rule above depends on."""
        for language in ["hi", "ta", "en-IN", "or", "sat"]:
            assert select_provider(language=language) == SARVAM


class TestUrduRoutesOnLocale:
    """Sarvam's Urdu model is trained on Indian Urdu, so it is the right choice
    only for speakers who are actually in India."""

    def test_indian_locale_goes_to_sarvam(self):
        assert select_provider(language="ur", locale="en_IN") == SARVAM
        assert select_provider(language="ur", locale="ur_IN") == SARVAM

    def test_pakistani_locale_goes_to_openai(self):
        assert select_provider(language="ur", locale="ur_PK") == OPENAI

    def test_unknown_locale_goes_to_openai(self):
        assert select_provider(language="ur", locale=None) == OPENAI
        assert select_provider(language="ur", locale="ur") == OPENAI

    def test_auto_detect_follows_the_same_rule(self):
        assert select_provider(locale="ur_PK") == OPENAI
        assert select_provider(locale="ur_IN") == SARVAM

    def test_sarvam_still_reports_coverage(self):
        """Coverage and suitability are different questions."""
        assert is_sarvam_language("ur")


class TestOutputMode:
    def test_hindi_is_romanized(self):
        """Romanized Hinglish is what the app has always produced."""
        assert to_sarvam_mode("hi") == MODE_TRANSLIT

    def test_auto_detect_is_romanized(self):
        """Sarvam picks the language itself, and the users who never pin one are
        overwhelmingly the existing Hindi and Hinglish speakers."""
        assert to_sarvam_mode(None) == MODE_TRANSLIT
        assert to_sarvam_mode("auto") == MODE_TRANSLIT

    @pytest.mark.parametrize("language", ["ta", "bn", "te", "ml", "en-IN", "sat"])
    def test_other_languages_keep_their_own_script(self, language):
        assert to_sarvam_mode(language) == MODE_TRANSCRIBE


class TestProviderChain:
    def test_indian_default_tries_both(self):
        assert provider_chain(locale="en_IN") == [SARVAM, OPENAI]

    def test_foreign_auto_detect_tries_openai_only(self):
        assert provider_chain(language="auto", locale="de_DE") == [OPENAI]

    def test_pinned_foreign_language_tries_openai_only(self):
        assert provider_chain(language="fr") == [OPENAI]

    def test_pinned_indian_language_tries_both(self):
        assert provider_chain(language="ta") == [SARVAM, OPENAI]


class TestLanguageNormalisation:
    @pytest.mark.parametrize("value", [None, "", "  ", "auto", "AUTO", " auto "])
    def test_absent_and_auto_collapse_to_none(self, value):
        assert normalise_language(value) is None

    def test_odia_alias_is_accepted(self):
        """Sarvam spells Odia 'od'; the ISO code is 'or'. Both must work."""
        assert normalise_language("od") == "or"
        assert is_sarvam_language("od")
        assert to_sarvam_code("od") == "od-IN"
        assert to_sarvam_code("or") == "od-IN"

    def test_sarvam_codes_are_regional(self):
        assert to_sarvam_code("hi") == "hi-IN"
        assert to_sarvam_code("en-IN") == "en-IN"
        assert to_sarvam_code("de") is None
        assert to_sarvam_code("auto") is None

    def test_every_sarvam_language_has_a_code(self):
        """A language in the routing set with no code would silently fall
        through to auto-detect instead of being pinned."""
        from services.stt.languages import SARVAM_LANGUAGES

        for language in SARVAM_LANGUAGES:
            assert to_sarvam_code(language) is not None, language

    def test_openai_codes_are_bare_iso_639_1(self):
        assert to_openai_code("hi") == "hi"
        assert to_openai_code("en-IN") == "en"
        assert to_openai_code("pt-BR") == "pt"
        assert to_openai_code("auto") is None


class TestLocaleDetection:
    @pytest.mark.parametrize(
        "locale", ["en_IN", "en-IN", "hi_IN", "hi", "ta", "or", "od", "mr-IN"]
    )
    def test_indian_locales(self, locale):
        assert is_indian_locale(locale)

    @pytest.mark.parametrize(
        "locale", ["en_US", "de_DE", "fr", "pt_BR", "zh_CN", "ur_PK", "ur", None, ""]
    )
    def test_non_indian_locales(self, locale):
        assert not is_indian_locale(locale)

    def test_indonesian_is_not_mistaken_for_india(self):
        """'id' is Indonesian; only a region subtag of 'IN' means India."""
        assert not is_indian_locale("id_ID")
        assert select_provider(locale="id_ID") == OPENAI
