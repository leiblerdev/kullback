"""The model client's read timeout fits the call: default, override, recording, exhaustion."""

from __future__ import annotations

import random

import httpx
import pytest

from kullback.ai import provider as pv


@pytest.fixture
def live():
    """Adapters refuse to run while ALLOW_MODEL_REQUESTS is False; the fake client keeps it offline.

    The flag is turned on through the module's own switch, the one path a person's environment
    takes; conftest's autouse no_live_models restores it to False when the test ends.
    """
    pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"})


class FakeResponse:
    """What the fake client answers with: a status, headers and a JSON body."""

    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.headers = {}
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeClient:
    """A local stand-in for httpx.Client: records the timeout object of every post.

    The script answers each attempt in order; an exception in the script is raised instead of
    answered. A one entry script answers every attempt the same way.
    """

    def __init__(self, script):
        self.script = list(script)
        self.timeouts = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.timeouts.append(timeout)
        action = self.script[min(len(self.timeouts) - 1, len(self.script) - 1)]
        if isinstance(action, Exception):
            raise action
        return action


def ok_answer():
    return FakeResponse({"content": [{"type": "text", "text": "done"}], "usage": {}})


def invented_model(client, env=None, **kwargs):
    return pv.AnthropicModel(
        model_id="harbourline/hl-1",
        api_key="k",
        client=client,
        sleep=lambda seconds: None,
        rng=random.Random(0),
        env={} if env is None else dict(env),
        **kwargs,
    )


def test_default_timeout_splits_short_connect_and_long_read(live):
    client = FakeClient([ok_answer()])
    invented_model(client).query([{"role": "user", "content": "hello"}])
    (timeout,) = client.timeouts
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == pytest.approx(10.0)
    assert timeout.read == pytest.approx(300.0)


def test_env_var_sets_the_read_timeout(live):
    client = FakeClient([ok_answer()])
    invented_model(client, env={"KULLBACK_MODEL_TIMEOUT_S": "45"}).query(
        [{"role": "user", "content": "hello"}]
    )
    (timeout,) = client.timeouts
    assert timeout.connect == pytest.approx(10.0)
    assert timeout.read == pytest.approx(45.0)


def test_unparseable_env_var_is_an_error_naming_the_variable(live):
    with pytest.raises(ValueError, match="KULLBACK_MODEL_TIMEOUT_S"):
        invented_model(FakeClient([]), env={"KULLBACK_MODEL_TIMEOUT_S": "soon"})


@pytest.mark.parametrize("raw", ["-5", "0", "nan"])
def test_non_positive_or_non_finite_env_var_is_an_error_naming_value(live, raw):
    with pytest.raises(ValueError, match=f"KULLBACK_MODEL_TIMEOUT_S.*{raw}"):
        invented_model(FakeClient([]), env={"KULLBACK_MODEL_TIMEOUT_S": raw})


def test_read_timeout_then_answer_records_attempts_and_timeout(live):
    client = FakeClient([httpx.ReadTimeout("the read operation timed out"), ok_answer()])
    reply = invented_model(client).query([{"role": "user", "content": "hello"}])
    assert reply.content == "done"
    assert len(client.timeouts) == 2
    assert reply.exchange.attempts == 2
    assert reply.exchange.read_timeout_s == 300.0


def test_five_read_timeouts_raise_exhausted_naming_the_timeout(live):
    client = FakeClient([httpx.ReadTimeout("the read operation timed out")])
    with pytest.raises(pv.RetryExhausted) as excinfo:
        invented_model(client).query([{"role": "user", "content": "hello"}])
    assert len(client.timeouts) == 5
    assert "read timeout" in str(excinfo.value)
    assert "300" in str(excinfo.value)


def test_five_connect_timeouts_name_the_connect_budget_not_the_read_budget(live):
    client = FakeClient([httpx.ConnectTimeout("connection timed out")])
    with pytest.raises(pv.RetryExhausted) as excinfo:
        invented_model(client).query([{"role": "user", "content": "hello"}])
    assert len(client.timeouts) == 5
    message = str(excinfo.value)
    assert "connect timeout" in message
    assert "10" in message
    assert "read timeout" not in message
    assert "300" not in message
