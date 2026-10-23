"""The live-call switch the adapter tests open: their transports are fakes, so nothing leaves the machine."""

from __future__ import annotations

import pytest

from kullback.ai import provider as pv


@pytest.fixture
def live():
    """Adapters refuse to run while ALLOW_MODEL_REQUESTS is False; the fake transports keep it offline.

    The flag is turned on through the module's own switch, the one path a person's environment
    takes; conftest's autouse no_live_models restores it to False when the test ends.
    """
    pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"})
