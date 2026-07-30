"""Supabase JWT verification for the AI endpoints.

Until now every route was open: anyone who found the Render URL could spend our
OpenAI and Sarvam credits. Each request must now carry the caller's Supabase
access token as `Authorization: Bearer <jwt>`.

Verification goes through Supabase's `/auth/v1/user` endpoint rather than
checking a signature locally. It costs a round trip, but it works regardless of
whether the project signs tokens with the legacy HS256 secret or a newer
asymmetric key, and it honours revoked sessions. Successful lookups are cached
briefly so polling a transcription job doesn't hammer the auth API.
"""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import Dict, Tuple

import httpx
from flask import g, jsonify, request

logger = logging.getLogger("Auth")

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or ""

# Set to "1" only for local development against a stubbed client.
ALLOW_ANONYMOUS = os.getenv("ALLOW_ANONYMOUS") == "1"

_CACHE_TTL_SECONDS = 60
_cache: Dict[str, Tuple[float, str]] = {}


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.message = message
        self.status = status


def _bearer_token() -> str:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise AuthError("Missing bearer token.")
    token = header[7:].strip()
    if not token:
        raise AuthError("Missing bearer token.")
    return token


def _cached_user_id(token: str) -> str | None:
    entry = _cache.get(token)
    if not entry:
        return None
    expires_at, user_id = entry
    if expires_at < time.time():
        _cache.pop(token, None)
        return None
    return user_id


def verify_token(token: str) -> str:
    """Returns the Supabase user id for a valid access token."""
    cached = _cached_user_id(token)
    if cached:
        return cached

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise AuthError(
            "Auth is not configured on the server.", status=500
        )

    try:
        response = httpx.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_ANON_KEY,
            },
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        logger.error(f"Auth lookup failed: {exc}")
        raise AuthError("Could not verify your session.", status=503) from exc

    if response.status_code != 200:
        raise AuthError("Your session is invalid or has expired.")

    user_id = (response.json() or {}).get("id")
    if not user_id:
        raise AuthError("Your session is invalid or has expired.")

    # Bound the cache so a burst of distinct tokens can't grow it without limit.
    if len(_cache) > 1000:
        _cache.clear()
    _cache[token] = (time.time() + _CACHE_TTL_SECONDS, user_id)

    return user_id


def require_auth(view):
    """Populates `g.user_id` or short-circuits with 401."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        error = authenticate_request()
        if error is not None:
            return error
        return view(*args, **kwargs)

    return wrapper


def authenticate_request():
    """Sets `g.user_id`. Returns a Flask error response, or None on success.

    Used from a `before_request` hook so authentication is the default for
    every route and a newly added endpoint cannot accidentally ship open.
    """
    if ALLOW_ANONYMOUS:
        g.user_id = "anonymous"
        return None

    try:
        g.user_id = verify_token(_bearer_token())
    except AuthError as exc:
        return jsonify({"error": exc.message}), exc.status

    return None
