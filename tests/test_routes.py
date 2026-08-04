"""Request validation, error mapping, and rate limiting for the HTTP layer.

Every test runs against the anonymous_app fixture, which stubs auth out; the
authentication rules themselves live in test_auth.py.
"""

import io

import pytest


@pytest.fixture
def app_module(anonymous_app):
    """The reloaded app module, for patching its imported collaborators."""
    import app as module

    return module


def _audio_upload(**fields):
    data = {"file": (io.BytesIO(b"fake audio bytes"), "note.m4a")}
    data.update(fields)
    return data


class TestTranscribeUpload:
    def test_missing_file_is_rejected(self, client):
        response = client.post("/transcribe", data={})

        assert response.status_code == 400
        assert "file" in response.get_json()["error"]

    def test_accepted_upload_returns_a_job_id(self, client, monkeypatch, app_module):
        monkeypatch.setattr(
            app_module, "create_transcription_job", lambda *a, **k: "job-1"
        )

        response = client.post(
            "/transcribe", data=_audio_upload(), content_type="multipart/form-data"
        )

        assert response.status_code == 202
        assert response.get_json() == {"job_id": "job-1", "status": "pending"}

    def test_language_and_locale_reach_the_job(
        self, client, monkeypatch, app_module
    ):
        """The whole point of the language picker. If these stop being forwarded
        the app silently falls back to auto-detect and nobody notices."""
        captured = {}

        def fake_create(user_id, file_storage, language=None, locale=None):
            captured.update(
                user_id=user_id, language=language, locale=locale
            )
            return "job-1"

        monkeypatch.setattr(app_module, "create_transcription_job", fake_create)

        client.post(
            "/transcribe",
            data=_audio_upload(language="ta", locale="en_IN"),
            content_type="multipart/form-data",
        )

        assert captured == {
            "user_id": "anonymous",
            "language": "ta",
            "locale": "en_IN",
        }

    def test_a_failure_returns_500_without_leaking_internals(
        self, client, monkeypatch, app_module
    ):
        def explode(*args, **kwargs):
            raise RuntimeError("SARVAM_API_KEY is invalid")

        monkeypatch.setattr(app_module, "create_transcription_job", explode)

        response = client.post(
            "/transcribe", data=_audio_upload(), content_type="multipart/form-data"
        )

        assert response.status_code == 500
        assert "SARVAM_API_KEY" not in response.get_data(as_text=True)


class TestTranscribeUrl:
    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"audio_url": ""},
            {"audio_url": "   "},
            {"audio_url": "http://insecure.example.com/a.m4a"},
            {"audio_url": "file:///etc/passwd"},
            {"audio_url": "ftp://example.com/a.m4a"},
        ],
    )
    def test_non_https_urls_are_rejected(self, client, body):
        response = client.post("/transcribe_url", json=body)

        assert response.status_code == 400

    def test_a_malformed_body_is_a_400_not_a_500(self, client):
        response = client.post(
            "/transcribe_url", data="not json", content_type="application/json"
        )

        assert response.status_code == 400

    def test_https_url_starts_a_job(self, client, monkeypatch, app_module):
        captured = {}

        def fake_create(user_id, audio_url, language=None, locale=None):
            captured.update(audio_url=audio_url, language=language, locale=locale)
            return "job-2"

        monkeypatch.setattr(
            app_module, "create_transcription_job_from_url", fake_create
        )

        response = client.post(
            "/transcribe_url",
            json={
                "audio_url": "https://storage.example.com/a.m4a",
                "language": "de",
                "locale": "de_DE",
            },
        )

        assert response.status_code == 202
        assert response.get_json()["job_id"] == "job-2"
        assert captured["language"] == "de"
        assert captured["locale"] == "de_DE"


class TestTranscribeStatus:
    def test_a_missing_job_is_404(self, client, monkeypatch, app_module):
        monkeypatch.setattr(
            app_module, "get_transcription_job", lambda *a, **k: None
        )

        response = client.get("/transcribe/status/nope")

        assert response.status_code == 404

    def test_the_lookup_is_scoped_to_the_caller(
        self, client, monkeypatch, app_module
    ):
        """Someone else's job id must be indistinguishable from a missing one,
        so the user id has to reach the query."""
        captured = {}

        def fake_get(job_id, user_id):
            captured.update(job_id=job_id, user_id=user_id)
            return None

        monkeypatch.setattr(app_module, "get_transcription_job", fake_get)

        client.get("/transcribe/status/job-9")

        assert captured == {"job_id": "job-9", "user_id": "anonymous"}

    def test_a_complete_job_is_returned(self, client, monkeypatch, app_module):
        job = {
            "status": "complete",
            "transcript": "Speaker 1: hello",
            "provider": "sarvam",
        }
        monkeypatch.setattr(
            app_module, "get_transcription_job", lambda *a, **k: job
        )

        response = client.get("/transcribe/status/job-1")

        assert response.status_code == 200
        assert response.get_json() == job

    def test_a_store_outage_is_503_not_404(self, client, monkeypatch, app_module):
        """A 404 would make the client give up on a job that is still running."""

        def explode(*args, **kwargs):
            raise RuntimeError("supabase unreachable")

        monkeypatch.setattr(app_module, "get_transcription_job", explode)

        response = client.get("/transcribe/status/job-1")

        assert response.status_code == 503


class TestSummarize:
    def test_missing_text_is_rejected(self, client):
        assert client.post("/summarize", json={}).status_code == 400
        assert client.get("/summarize").status_code == 400

    def test_text_is_summarised(self, client, monkeypatch, app_module):
        monkeypatch.setattr(app_module, "generate_text", lambda prompt: "a summary")

        response = client.post("/summarize", json={"text": "long transcript"})

        assert response.status_code == 200
        assert response.get_json()["summary"] == "a summary"


class TestRewriteRoutes:
    def _rewrite_rules(self, app):
        return [
            rule.rule
            for rule in app.url_map.iter_rules()
            if rule.rule.startswith("/generate_")
        ]

    def test_every_rewrite_route_resolves_a_real_config(
        self, anonymous_app, client, monkeypatch, app_module
    ):
        """Each route hard-codes a rewrite id. A typo there is a KeyError that
        only shows up when a user taps that specific action.
        """
        monkeypatch.setattr(
            app_module, "generate_rewrite", lambda config, text: f"ok:{config.rewrite_id}"
        )

        rules = self._rewrite_rules(anonymous_app)
        assert rules, "no rewrite routes found"

        for rule in rules:
            response = client.post(rule, json={"text": "some transcript"})
            assert response.status_code == 200, rule
            assert response.get_json()["result"].startswith("ok:"), rule

    def test_an_empty_transcript_still_returns_cleanly(
        self, client, monkeypatch, app_module
    ):
        monkeypatch.setattr(app_module, "generate_rewrite", lambda config, text: "")

        response = client.post("/generate_journal", json={})

        assert response.status_code == 200


class TestRateLimiting:
    def _enable(self, app_module):
        app_module.limiter.enabled = True

    def test_limits_are_actually_enforced(self, client, monkeypatch, app_module):
        """Positive control for the exemption tests below."""
        self._enable(app_module)
        monkeypatch.setattr(app_module, "generate_text", lambda prompt: "s")

        statuses = {
            client.post("/summarize", json={"text": "hi"}).status_code
            for _ in range(35)
        }

        assert 429 in statuses

    def test_status_polling_is_exempt(self, client, monkeypatch, app_module):
        """Clients poll every 2s for up to 15 minutes. Under the default limit a
        single normal transcription would rate-limit itself.
        """
        self._enable(app_module)
        monkeypatch.setattr(
            app_module, "get_transcription_job", lambda *a, **k: None
        )

        statuses = {
            client.get("/transcribe/status/job-1").status_code for _ in range(60)
        }

        assert statuses == {404}

    def test_the_health_check_is_exempt(self, client, app_module):
        """Cold Render dynos get pinged repeatedly until they wake up."""
        self._enable(app_module)

        statuses = {client.get("/").status_code for _ in range(60)}

        assert statuses == {200}

    def test_transcription_is_limited_more_tightly_than_the_default(
        self, client, monkeypatch, app_module
    ):
        """Transcription is the expensive endpoint, so it gets 5/minute rather
        than the default 30."""
        self._enable(app_module)
        monkeypatch.setattr(
            app_module, "create_transcription_job", lambda *a, **k: "job-1"
        )

        statuses = [
            client.post(
                "/transcribe",
                data=_audio_upload(),
                content_type="multipart/form-data",
            ).status_code
            for _ in range(8)
        ]

        assert statuses[:5] == [202] * 5
        assert 429 in statuses[5:]
