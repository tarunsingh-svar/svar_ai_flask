"""Response parsing for both providers.

Notes have always rendered as "Speaker N: …". These tests pin that shape so a
recording routed to OpenAI reads the same as one routed to Sarvam.
"""

from types import SimpleNamespace

from services.stt.openai_provider import _parse as parse_openai
from services.stt.sarvam_provider import _parse as parse_sarvam
from services.stt.types import speaker_label


def sarvam_entry(speaker_id, *, transliterated=None, english=None, transcript=None):
    transcriptions = {}
    if transliterated is not None:
        transcriptions["hi-IN"] = {"transliteration": {"en": {"text": transliterated}}}
    if english is not None:
        transcriptions["en-IN"] = {"text": english}

    return {
        "speaker_id": speaker_id,
        "transcript": transcript or "",
        "transcription_output": {"transcriptions": transcriptions},
    }


def openai_segment(speaker, text):
    return SimpleNamespace(speaker=speaker, text=text)


class TestSpeakerLabel:
    def test_sarvam_ids_become_one_indexed(self):
        assert speaker_label("SPEAKER_00") == "Speaker 1"
        assert speaker_label("SPEAKER_01") == "Speaker 2"
        assert speaker_label("speaker_2") == "Speaker 3"

    def test_openai_letters_are_preserved(self):
        assert speaker_label("A") == "Speaker A"
        assert speaker_label("B") == "Speaker B"

    def test_no_double_prefix(self):
        """The old formatter produced 'Speaker Speaker 00:' for Sarvam ids."""
        assert speaker_label("SPEAKER_00") == "Speaker 1"
        assert "Speaker Speaker" not in (speaker_label("SPEAKER_00") or "")

    def test_missing_and_blank_ids_yield_no_label(self):
        assert speaker_label(None) is None
        assert speaker_label("") is None
        assert speaker_label("speaker_") is None


class TestSarvamParsing:
    def test_uses_the_transcript_field(self):
        """saaras:v3 applies the requested mode server-side, so `transcript` is
        already in the right script and wins over the legacy nested fields."""
        payload = {
            "diarized_transcript": {
                "entries": [
                    sarvam_entry(
                        "SPEAKER_00",
                        transcript="mera phone number hai 9840950950",
                        transliterated="stale nested value",
                    )
                ]
            }
        }

        result = parse_sarvam(payload, "hi")

        assert result.to_text() == "Speaker 1: mera phone number hai 9840950950"
        assert result.provider == "sarvam"
        assert result.language == "hi"

    def test_falls_back_to_nested_transliteration(self):
        """saaras:v3 diarization is flagged beta, so the saarika:v2.5 response
        shape is still handled rather than trusted to be gone."""
        payload = {
            "diarized_transcript": {
                "entries": [
                    sarvam_entry(
                        "SPEAKER_00", transliterated="main office ja raha hoon"
                    )
                ]
            }
        }

        assert parse_sarvam(payload, "hi").to_text() == (
            "Speaker 1: main office ja raha hoon"
        )

    def test_falls_back_to_english_transcription(self):
        payload = {
            "diarized_transcript": {
                "entries": [
                    sarvam_entry("SPEAKER_00", english="Quarterly numbers look good")
                ]
            }
        }

        assert parse_sarvam(payload, None).to_text() == (
            "Speaker 1: Quarterly numbers look good"
        )

    def test_native_script_passes_through_unchanged(self):
        payload = {
            "diarized_transcript": {
                "entries": [sarvam_entry("SPEAKER_00", transcript="வணக்கம் நண்பர்களே")]
            }
        }

        assert parse_sarvam(payload, "ta").to_text() == (
            "Speaker 1: வணக்கம் நண்பர்களே"
        )

    def test_multiple_speakers_render_one_per_line(self):
        payload = {
            "diarized_transcript": {
                "entries": [
                    sarvam_entry("SPEAKER_00", english="Shall we start?"),
                    sarvam_entry("SPEAKER_01", english="Yes, go ahead."),
                ]
            }
        }

        assert parse_sarvam(payload, None).to_text() == (
            "Speaker 1: Shall we start?\nSpeaker 2: Yes, go ahead."
        )

    def test_undiarized_payload_uses_flat_transcript(self):
        payload = {"transcript": "  a single speaker recording  "}

        result = parse_sarvam(payload, None)

        assert result.to_text() == "a single speaker recording"
        assert result.segments[0].speaker is None

    def test_empty_payload_is_empty(self):
        assert parse_sarvam({}, None).is_empty
        assert parse_sarvam({"transcript": "   "}, None).is_empty

    def test_blank_entries_are_dropped(self):
        payload = {
            "diarized_transcript": {
                "entries": [
                    sarvam_entry("SPEAKER_00", english="   "),
                    sarvam_entry("SPEAKER_01", english="Only this survives"),
                ]
            }
        }

        assert parse_sarvam(payload, None).to_text() == (
            "Speaker 2: Only this survives"
        )


class TestOpenAIParsing:
    def test_diarized_segments_render_like_sarvam(self):
        response = SimpleNamespace(
            text="Guten Tag. Wie geht es dir?",
            segments=[
                openai_segment("A", "Guten Tag."),
                openai_segment("B", "Wie geht es dir?"),
            ],
        )

        result = parse_openai(response, "de")

        assert result.to_text() == (
            "Speaker A: Guten Tag.\nSpeaker B: Wie geht es dir?"
        )
        assert result.provider == "openai"
        assert result.language == "de"

    def test_falls_back_to_flat_text_when_diarization_is_absent(self):
        """Short or single-speaker clips can come back without segments."""
        response = SimpleNamespace(text="  Bonjour tout le monde  ", segments=[])

        result = parse_openai(response, "fr")

        assert result.to_text() == "Bonjour tout le monde"
        assert result.segments[0].speaker is None

    def test_blank_segments_are_dropped(self):
        response = SimpleNamespace(
            text="Hola",
            segments=[openai_segment("A", "  "), openai_segment("B", "Hola")],
        )

        assert parse_openai(response, "es").to_text() == "Speaker B: Hola"

    def test_completely_empty_response_is_empty(self):
        response = SimpleNamespace(text="", segments=[])
        assert parse_openai(response, None).is_empty

    def test_missing_attributes_do_not_raise(self):
        assert parse_openai(SimpleNamespace(), None).is_empty
