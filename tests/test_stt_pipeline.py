"""Provider selection and cross-provider retry, with fake providers.

A vendor outage should cost transcript quality, not the user's recording.
"""

import pytest

from services.stt import pipeline
from services.stt.router import OPENAI, SARVAM
from services.stt.types import (
    TranscriptionError,
    TranscriptionResult,
    TranscriptSegment,
)


class FakeProvider:
    def __init__(self, name, *, available=True, raises=None, text="transcript"):
        self.name = name
        self._available = available
        self._raises = raises
        self._text = text
        self.calls = []

    def is_available(self):
        return self._available

    def transcribe(self, audio_path, language=None):
        self.calls.append((audio_path, language))
        if self._raises is not None:
            raise self._raises
        segments = (
            [TranscriptSegment(text=self._text)] if self._text else []
        )
        return TranscriptionResult(
            segments=segments, provider=self.name, language=language
        )


@pytest.fixture
def install(monkeypatch):
    """Swaps the provider registry for fakes and hands them back."""

    def _install(sarvam, openai):
        monkeypatch.setitem(pipeline._PROVIDERS, SARVAM, lambda: sarvam)
        monkeypatch.setitem(pipeline._PROVIDERS, OPENAI, lambda: openai)
        return sarvam, openai

    return _install


def test_indian_locale_uses_sarvam_and_never_calls_openai(install):
    sarvam, openai = install(FakeProvider(SARVAM), FakeProvider(OPENAI))

    result = pipeline.transcribe("/tmp/a.m4a", locale="en_IN")

    assert result.provider == SARVAM
    assert len(sarvam.calls) == 1
    assert openai.calls == []


def test_foreign_locale_uses_openai_and_never_calls_sarvam(install):
    sarvam, openai = install(FakeProvider(SARVAM), FakeProvider(OPENAI))

    result = pipeline.transcribe("/tmp/a.m4a", language="auto", locale="de_DE")

    assert result.provider == OPENAI
    assert sarvam.calls == []


def test_sarvam_failure_retries_on_openai(install):
    sarvam, openai = install(
        FakeProvider(SARVAM, raises=TranscriptionError("sarvam is down")),
        FakeProvider(OPENAI),
    )

    result = pipeline.transcribe("/tmp/a.m4a", locale="en_IN")

    assert result.provider == OPENAI
    assert len(sarvam.calls) == 1
    assert len(openai.calls) == 1


def test_openai_failure_does_not_retry_sarvam_for_unsupported_language(install):
    """Sarvam would return plausible-looking nonsense for German."""
    sarvam, openai = install(
        FakeProvider(SARVAM),
        FakeProvider(OPENAI, raises=TranscriptionError("openai is down")),
    )

    with pytest.raises(TranscriptionError, match="openai is down"):
        pipeline.transcribe("/tmp/a.m4a", language="de")

    assert sarvam.calls == []


def test_both_failing_raises_the_last_error(install):
    install(
        FakeProvider(SARVAM, raises=TranscriptionError("sarvam is down")),
        FakeProvider(OPENAI, raises=TranscriptionError("openai is down")),
    )

    with pytest.raises(TranscriptionError, match="openai is down"):
        pipeline.transcribe("/tmp/a.m4a", locale="en_IN")


def test_unconfigured_primary_is_skipped(install):
    sarvam, openai = install(
        FakeProvider(SARVAM, available=False), FakeProvider(OPENAI)
    )

    result = pipeline.transcribe("/tmp/a.m4a", locale="en_IN")

    assert result.provider == OPENAI
    assert sarvam.calls == []


def test_no_configured_provider_raises_a_user_safe_message(install):
    install(
        FakeProvider(SARVAM, available=False),
        FakeProvider(OPENAI, available=False),
    )

    with pytest.raises(TranscriptionError, match="not configured"):
        pipeline.transcribe("/tmp/a.m4a", locale="en_IN")


def test_empty_primary_result_lets_the_fallback_try(install):
    sarvam, openai = install(
        FakeProvider(SARVAM, text=""), FakeProvider(OPENAI, text="recovered")
    )

    result = pipeline.transcribe("/tmp/a.m4a", locale="en_IN")

    assert result.provider == OPENAI
    assert result.to_text() == "recovered"


def test_silent_recording_returns_empty_rather_than_failing(install):
    """A genuinely silent recording should surface as empty, not as an error."""
    install(FakeProvider(SARVAM, text=""), FakeProvider(OPENAI, text=""))

    result = pipeline.transcribe("/tmp/a.m4a", locale="en_IN")

    assert result.is_empty


def test_language_is_passed_through_to_the_provider(install):
    sarvam, _ = install(FakeProvider(SARVAM), FakeProvider(OPENAI))

    pipeline.transcribe("/tmp/a.m4a", language="ta")

    assert sarvam.calls == [("/tmp/a.m4a", "ta")]
