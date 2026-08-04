"""Tests for Supabase token verification and the authenticate-by-default hook.

The structural test in [TestEveryRouteIsPrivate] is the important one: these
endpoints spend real OpenAI and Sarvam credits, and they were all open to anyone
who found the Render URL until recently.
"""

import importlib
import re

import httpx
import pytest
import respx

SUPABASE_URL = "https://project.supabase.co"


def _reload_auth(monkeypatch, *, url=SUPABASE_URL, anon_key="anon-key", anonymous="0"):
    """Reloads services.auth so its import-time env globals are re-read."""
    monkeypatch.setenv("SUPABASE_URL", url)
    monkeypatch.setenv("SUPABASE_ANON_KEY", anon_key)
    monkeypatch.setenv("ALLOW_ANONYMOUS", anonymous)

    import services.auth as auth_module

    importlib.reload(auth_module)
    auth_module._cache.clear()
    return auth_module


def _sample_url(rule) -> str:
    """A concrete path for a route rule, filling any converter with a dummy."""
    return re.sub(r"<[^>]+>", "sample-id", rule.rule)


def _sample_method(rule) -> str:
    for method in ("POST", "GET", "PUT", "DELETE"):
        if method in rule.methods:
            return method
    return "GET"


class TestEveryRouteIsPrivate:
    def test_all_routes_reject_an_unauthenticated_request(self, authenticated_app):
        """Guards the before_request design.

        Authentication is applied globally rather than per-route so that adding
        an endpoint cannot accidentally ship it open. If someone adds a route to
        PUBLIC_ENDPOINTS without thinking, this test is what notices.
        """
        import app as app_module

        client = authenticated_app.test_client()
        unprotected = []

        for rule in authenticated_app.url_map.iter_rules():
            if rule.endpoint in {"static"} | app_module.PUBLIC_ENDPOINTS:
                continue

            response = client.open(
                _sample_url(rule), method=_sample_method(rule)
            )
            if response.status_code != 401:
                unprotected.append((rule.rule, response.status_code))

        assert unprotected == [], f"Routes reachable without auth: {unprotected}"

    def test_the_health_check_stays_public(self, authenticated_app):
        """Clients ping this to wake a cold Render dyno before they have a
        session, so it must not require one."""
        response = authenticated_app.test_client().get("/")

        assert response.status_code == 200


class TestBearerToken:
    @pytest.mark.parametrize(
        "header",
        [
            None,
            "",
            "token-without-scheme",
            "Basic dXNlcjpwYXNz",
            "bearer lowercase-scheme",
            "Bearer ",
            "Bearer    ",
        ],
    )
    def test_malformed_authorization_headers_are_rejected(
        self, authenticated_app, header
    ):
        headers = {} if header is None else {"Authorization": header}

        response = authenticated_app.test_client().post(
            "/generate_journal", headers=headers, json={"text": "hello"}
        )

        assert response.status_code == 401


class TestVerifyToken:
    @respx.mock
    def test_returns_the_user_id_for_a_valid_token(self, monkeypatch):
        auth = _reload_auth(monkeypatch)
        respx.get(f"{SUPABASE_URL}/auth/v1/user").mock(
            return_value=httpx.Response(200, json={"id": "user-123"})
        )

        assert auth.verify_token("good-token") == "user-123"

    @respx.mock
    def test_rejects_a_token_supabase_does_not_accept(self, monkeypatch):
        auth = _reload_auth(monkeypatch)
        respx.get(f"{SUPABASE_URL}/auth/v1/user").mock(
            return_value=httpx.Response(401, json={"msg": "invalid jwt"})
        )

        with pytest.raises(auth.AuthError) as exc:
            auth.verify_token("expired-token")

        assert exc.value.status == 401

    @respx.mock
    def test_rejects_a_response_with_no_user_id(self, monkeypatch):
        auth = _reload_auth(monkeypatch)
        respx.get(f"{SUPABASE_URL}/auth/v1/user").mock(
            return_value=httpx.Response(200, json={})
        )

        with pytest.raises(auth.AuthError):
            auth.verify_token("odd-token")

    @respx.mock
    def test_a_network_failure_is_503_not_401(self, monkeypatch):
        """An outage on Supabase's side is our problem, not a bad session. A 401
        would sign the user out of a working account."""
        auth = _reload_auth(monkeypatch)
        respx.get(f"{SUPABASE_URL}/auth/v1/user").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with pytest.raises(auth.AuthError) as exc:
            auth.verify_token("good-token")

        assert exc.value.status == 503

    def test_missing_server_config_is_500_not_401(self, monkeypatch):
        """A deploy without SUPABASE_URL is a server misconfiguration; telling
        the user their session expired would send them into a login loop."""
        auth = _reload_auth(monkeypatch, url="")

        with pytest.raises(auth.AuthError) as exc:
            auth.verify_token("any-token")

        assert exc.value.status == 500

    @respx.mock
    def test_a_verified_token_is_cached(self, monkeypatch):
        """Clients poll job status every 2s for up to 15 minutes. Without the
        cache that is ~450 auth round trips per transcription."""
        auth = _reload_auth(monkeypatch)
        route = respx.get(f"{SUPABASE_URL}/auth/v1/user").mock(
            return_value=httpx.Response(200, json={"id": "user-123"})
        )

        for _ in range(5):
            assert auth.verify_token("good-token") == "user-123"

        assert route.call_count == 1

    @respx.mock
    def test_an_expired_cache_entry_is_re_verified(self, monkeypatch):
        auth = _reload_auth(monkeypatch)
        route = respx.get(f"{SUPABASE_URL}/auth/v1/user").mock(
            return_value=httpx.Response(200, json={"id": "user-123"})
        )

        auth.verify_token("good-token")
        # Backdate the entry rather than sleeping through the real TTL.
        _, user_id = auth._cache["good-token"]
        auth._cache["good-token"] = (0.0, user_id)

        assert auth.verify_token("good-token") == "user-123"
        assert route.call_count == 2

    @respx.mock
    def test_rejected_tokens_are_never_cached(self, monkeypatch):
        """Otherwise a revoked session could keep working for the TTL."""
        auth = _reload_auth(monkeypatch)
        respx.get(f"{SUPABASE_URL}/auth/v1/user").mock(
            return_value=httpx.Response(401)
        )

        for _ in range(2):
            with pytest.raises(auth.AuthError):
                auth.verify_token("bad-token")

        assert "bad-token" not in auth._cache

    @respx.mock
    def test_the_cache_is_bounded(self, monkeypatch):
        """A hostile client sending distinct tokens must not grow it forever."""
        auth = _reload_auth(monkeypatch)
        respx.get(f"{SUPABASE_URL}/auth/v1/user").mock(
            return_value=httpx.Response(200, json={"id": "user-123"})
        )

        for i in range(1100):
            auth.verify_token(f"token-{i}")

        assert len(auth._cache) <= 1001


class TestAllowAnonymous:
    def test_the_escape_hatch_requires_an_exact_opt_in(self, monkeypatch):
        """Anything other than "1" must leave auth switched on, so a stray
        ALLOW_ANONYMOUS=true or =0 in a deploy cannot open the API."""
        for value in ["0", "true", "yes", "", "TRUE", "on"]:
            auth = _reload_auth(monkeypatch, anonymous=value)
            assert auth.ALLOW_ANONYMOUS is False, value

    def test_one_enables_it(self, monkeypatch):
        auth = _reload_auth(monkeypatch, anonymous="1")

        assert auth.ALLOW_ANONYMOUS is True
