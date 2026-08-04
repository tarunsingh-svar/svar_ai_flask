import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def anonymous_app(monkeypatch):
    """The Flask app with auth stubbed out, for exercising route behaviour."""
    monkeypatch.setenv("ALLOW_ANONYMOUS", "1")

    import importlib

    import services.auth as auth_module

    importlib.reload(auth_module)

    import app as app_module

    importlib.reload(app_module)

    app_module.app.config.update(TESTING=True)
    # Limits are irrelevant to route-behaviour assertions and would make test
    # order significant.
    app_module.limiter.enabled = False
    return app_module.app


@pytest.fixture
def authenticated_app(monkeypatch):
    """The Flask app with real auth wiring, for testing that it rejects."""
    monkeypatch.setenv("ALLOW_ANONYMOUS", "0")

    import importlib

    import services.auth as auth_module

    importlib.reload(auth_module)

    import app as app_module

    importlib.reload(app_module)

    app_module.app.config.update(TESTING=True)
    app_module.limiter.enabled = False
    return app_module.app


@pytest.fixture
def client(anonymous_app):
    return anonymous_app.test_client()
