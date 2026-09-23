import base64
import contextlib

import pytest

from kullback.container_grader import GradeError, GradeReceipt, grade
from kullback.container_world import StepReceipt

CHUNK = 6


def _ok(stdout=b"", stderr=b"", code=0):
    return StepReceipt(stdout, stderr, code, False, False)


class _FakeWorld:
    """Fake at the World boundary: records step commands, answers by script."""

    def __init__(self, answers=()):
        self._answers = list(answers)
        self.commands = []
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self):
        self.reset_calls += 1
        return "fake-container-id"

    def step(self, command):
        self.commands.append(command)
        item = self._answers.pop(0) if self._answers else _ok()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.close_calls += 1
        return True


def _maker(fake):
    def make():
        return fake

    return make


def test_grade_feeds_the_state_then_the_grader_and_passes_on_clean_exit():
    state = b"abcdefABCDEF"
    grader_bytes = b"123456"
    command = "./grader/run ./state"
    token = "ab12cd34"
    grader_dir = "/workspace/grader-" + token
    final = "STATE_DIR=/workspace/state GRADER_DIR=" + grader_dir + "; export STATE_DIR GRADER_DIR; " + command
    answers = [_ok()] * 7 + [_ok(b"graded\n", b"", 0)]
    fake = _FakeWorld(answers)
    receipt = grade(
        _maker(fake),
        state_tar=state,
        grader_tar=grader_bytes,
        command=command,
        chunk_bytes=CHUNK,
        token=token,
    )
    assert isinstance(receipt, GradeReceipt)
    assert receipt.exit_code == 0
    assert receipt.stdout == b"graded\n"
    assert receipt.passed is True
    first = base64.b64encode(b"abcdef").decode("ascii")
    second = base64.b64encode(b"ABCDEF").decode("ascii")
    only = base64.b64encode(b"123456").decode("ascii")
    assert fake.commands == [
        "mkdir /workspace/state",
        "printf %s '" + first + "' >> /workspace/.state.b64",
        "printf %s '" + second + "' >> /workspace/.state.b64",
        "base64 -d /workspace/.state.b64 | tar -xf - -C /workspace/state && rm /workspace/.state.b64",
        "mkdir " + grader_dir,
        "printf %s '" + only + "' >> /workspace/.grader-" + token + ".b64",
        "base64 -d /workspace/.grader-"
        + token
        + ".b64 | tar -xf - -C "
        + grader_dir
        + " && rm /workspace/.grader-"
        + token
        + ".b64",
        final,
    ]
    assert base64.b64decode(first + second) == state
    assert base64.b64decode(only) == grader_bytes
    assert fake.reset_calls == 1
    assert fake.close_calls == 1


def test_grade_hides_the_grader_in_a_fresh_token_dir_made_after_the_state_unpacks():
    token = "ab12cd34"
    fake = _FakeWorld()
    grade(_maker(fake), state_tar=b"abcdef", grader_tar=b"123456", command="true", chunk_bytes=CHUNK, token=token)
    grader_dir = "/workspace/grader-" + token
    staging = "/workspace/.grader-" + token + ".b64"
    mentions = [c for c in fake.commands if "grader" in c]
    assert len(mentions) == 4
    assert all(grader_dir in c or staging in c for c in mentions)
    assert sum(1 for c in mentions if grader_dir in c) == 3
    assert all("/workspace/grader " not in c and "/workspace/grader/" not in c for c in fake.commands)
    unpack = "base64 -d /workspace/.state.b64 | tar -xf - -C /workspace/state && rm /workspace/.state.b64"
    assert fake.commands.index(unpack) < fake.commands.index("mkdir " + grader_dir)

    def grader_dirs(fake):
        return [c for c in fake.commands if c.startswith("mkdir /workspace/grader-")]

    first_fake = _FakeWorld()
    grade(_maker(first_fake), state_tar=b"", grader_tar=b"", command="true")
    second_fake = _FakeWorld()
    grade(_maker(second_fake), state_tar=b"", grader_tar=b"", command="true")
    assert len(grader_dirs(first_fake)) == 1
    assert len(grader_dirs(second_fake)) == 1
    assert grader_dirs(first_fake) != grader_dirs(second_fake)


@pytest.mark.parametrize(
    ("answer", "passed"),
    [
        (StepReceipt(b"out", b"err", 1, False, False), False),
        (StepReceipt(b"part", b"", 0, True, False), False),
        (StepReceipt(b"part", b"", 0, False, True), False),
        (StepReceipt(b"ok\n", b"", 0, False, False), True),
    ],
    ids=["exit-1-has-no-pass", "timed-out-has-no-pass", "truncated-has-no-pass", "clean-exit-0-passes"],
)
def test_grade_passed_only_for_clean_exit_zero(answer, passed):
    fake = _FakeWorld([_ok(), _ok(), answer])
    receipt = grade(_maker(fake), state_tar=b"", grader_tar=b"", command="true")
    assert receipt.passed is passed
    assert receipt.exit_code == answer.exit_code
    assert receipt.stdout == answer.stdout
    assert receipt.stderr == answer.stderr
    assert receipt.timed_out == answer.timed_out
    assert receipt.truncated == answer.truncated
    assert fake.close_calls == 1


@pytest.mark.parametrize(
    ("answers", "stage"),
    [
        ([StepReceipt(b"", b"nope\n", 1, False, False)], "state mkdir"),
        ([_ok(), StepReceipt(b"", b"nope\n", 1, False, False)], "state chunk 0"),
        ([_ok(), _ok(), StepReceipt(b"", b"nope\n", 1, False, False)], "state unpack"),
        ([_ok(), _ok(), StepReceipt(b"part", b"", 0, True, False)], "state unpack"),
        ([_ok(), _ok(), StepReceipt(b"part", b"", 0, False, True)], "state unpack"),
        ([_ok(), _ok(), _ok(), StepReceipt(b"", b"exists\n", 1, False, False)], "grader mkdir"),
    ],
    ids=[
        "mkdir-failure-names-stage",
        "chunk-failure-names-stage",
        "unpack-failure-names-stage",
        "unpack-timeout-names-stage",
        "unpack-truncation-names-stage",
        "grader-mkdir-failure-names-stage",
    ],
)
def test_grade_feeding_failure_names_stage_skips_grade_and_closes(answers, stage):
    fake = _FakeWorld(answers)
    with pytest.raises(GradeError, match=stage):
        grade(
            _maker(fake),
            state_tar=b"123456",
            grader_tar=b"",
            command="./grader/run",
            chunk_bytes=CHUNK,
            token="ab12cd34",
        )
    assert "./grader/run" not in fake.commands
    assert all("export STATE_DIR GRADER_DIR" not in c for c in fake.commands)
    assert fake.close_calls == 1


def test_grade_close_runs_and_interrupt_propagates_on_step_interrupt():
    fake = _FakeWorld([_ok(), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        grade(_maker(fake), state_tar=b"123456", grader_tar=b"", command="true", chunk_bytes=CHUNK)
    assert fake.close_calls == 1


_NO_INPUTS = {"state_tar": b"", "grader_tar": b"", "command": "true"}


@pytest.mark.parametrize(
    ("kwargs", "calls"),
    [
        pytest.param({**_NO_INPUTS, "state_tar": b"123", "chunk_bytes": 7}, 0, id="chunk-not-multiple-of-3"),
        pytest.param({**_NO_INPUTS, "state_tar": b"123", "chunk_bytes": 0}, 0, id="chunk-zero"),
        pytest.param({**_NO_INPUTS, "state_tar": b"123", "chunk_bytes": "48"}, 0, id="chunk-not-int"),
        pytest.param({**_NO_INPUTS, "state_tar": b"12345", "max_input_bytes": 4}, 0, id="oversize-state"),
        pytest.param({**_NO_INPUTS, "grader_tar": b"12345", "max_input_bytes": 4}, 0, id="oversize-grader"),
        pytest.param({**_NO_INPUTS, "max_input_bytes": None}, 0, id="max-none"),
        pytest.param({**_NO_INPUTS, "max_input_bytes": "16777216"}, 0, id="max-str"),
        pytest.param({**_NO_INPUTS, "max_input_bytes": True}, 0, id="max-bool"),
        pytest.param({**_NO_INPUTS, "max_input_bytes": 0}, 0, id="max-zero"),
        pytest.param({**_NO_INPUTS, "max_input_bytes": -1}, 0, id="max-negative"),
        pytest.param({**_NO_INPUTS, "max_input_bytes": 16_777_217}, 0, id="max-above-ceiling"),
        pytest.param({**_NO_INPUTS, "state_tar": "123"}, 0, id="non-bytes-state"),
        pytest.param({**_NO_INPUTS, "grader_tar": 123}, 0, id="non-bytes-grader"),
        pytest.param({**_NO_INPUTS, "command": ""}, 0, id="empty-command"),
        pytest.param({**_NO_INPUTS, "token": "AB12CD34"}, 0, id="token-uppercase"),
        pytest.param({**_NO_INPUTS, "token": "ab12"}, 0, id="token-too-short"),
        pytest.param({**_NO_INPUTS, "token": "ab12/cd34"}, 0, id="token-slash"),
        pytest.param({**_NO_INPUTS, "token": "ab12'cd34"}, 0, id="token-quote"),
        pytest.param({**_NO_INPUTS, "token": 12345}, 0, id="token-non-str-int"),
        pytest.param({**_NO_INPUTS, "token": b"ab12cd34"}, 0, id="token-non-str-bytes"),
        pytest.param({**_NO_INPUTS, "max_input_bytes": 16_777_216}, 1, id="max-at-ceiling-is-accepted"),
    ],
)
def test_grade_refuses_bad_inputs_before_any_world_call_and_accepts_the_ceiling(kwargs, calls):
    fake = _FakeWorld()
    expectation = pytest.raises(GradeError) if calls == 0 else contextlib.nullcontext()
    with expectation:
        receipt = grade(_maker(fake), **kwargs)
        assert receipt.passed is True
    assert fake.reset_calls == calls
    assert fake.close_calls == calls
    assert bool(fake.commands) is bool(calls)
