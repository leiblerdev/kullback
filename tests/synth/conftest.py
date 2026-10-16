"""Every synth test runs with live requests off: a reader the test did not inject is None, never the network."""

from __future__ import annotations

import pytest

from kullback.ai.provider import LIVE_ENV_VAR


@pytest.fixture(autouse=True)
def _live_off(monkeypatch):
    monkeypatch.delenv(LIVE_ENV_VAR, raising=False)
