"""Every synth test runs with live requests off: a reader the test did not inject is None, never the network.

A destination check the test did not script fails the test rather than resolving a host name.
"""

from __future__ import annotations

import pytest

from kullback import domain
from kullback.ai.provider import LIVE_ENV_VAR


@pytest.fixture(autouse=True)
def _live_off(monkeypatch):
    monkeypatch.delenv(LIVE_ENV_VAR, raising=False)


def _no_lookup(host):
    pytest.fail(f"a synth test resolved {host!r}: hand the role the fixture's REFUSE")


@pytest.fixture(autouse=True)
def _no_name_lookup(monkeypatch):
    monkeypatch.setattr(domain, "resolver", _no_lookup)
