"""Sentry wiring for the AI backend.

Optional by design: with no SENTRY_DSN set (local development, or a fork without
its own project) this is a no-op and the app behaves exactly as before.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("Observability")

_initialised = False


def init_sentry() -> None:
    """Initialise Sentry once, if a DSN is configured."""
    global _initialised

    if _initialised:
        return

    dsn = (os.getenv("SENTRY_DSN") or "").strip()
    if not dsn:
        logger.info("SENTRY_DSN not set, error reporting disabled.")
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration
    except ImportError:
        logger.warning("sentry-sdk is not installed, error reporting disabled.")
        return

    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT") or "production",
        integrations=[FlaskIntegration()],
        # Transcripts are the user's private notes, and request bodies carry
        # audio and bearer tokens. Never let either reach the error reporter.
        send_default_pii=False,
        max_request_body_size="never",
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE") or 0.0),
    )

    _initialised = True
    logger.info("Sentry initialised.")


def capture_exception(exc: BaseException, **tags) -> None:
    """Report an exception that the app handled and does not re-raise.

    Transcription failures are swallowed into job state so the user sees a
    friendly message; without this they would never surface anywhere.
    """
    if not _initialised:
        return

    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for key, value in tags.items():
                scope.set_tag(key, value)
            sentry_sdk.capture_exception(exc)
    except Exception:  # pragma: no cover - reporting must never break a request
        logger.warning("Failed to report exception to Sentry.", exc_info=True)
