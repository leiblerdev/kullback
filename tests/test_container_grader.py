import base64

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


def test_grade_feeds_two_chunk_state_and_one_chunk_grader_in_order():
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


def test_grade_uses_token_grader_dir_everywhere_and_no_bare_grader_path():
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


def test_grade_state_unpacks_before_grader_dir_exists():
    token = "ab12cd34"
    fake = _FakeWorld()
    grade(_maker(fake), state_tar=b"abcdef", grader_tar=b"123456", command="true", chunk_bytes=CHUNK, token=token)
    unpack = "base64 -d /workspace/.state.b64 | tar -xf - -C /workspace/state && rm /workspace/.state.b64"
    assert fake.commands.index(unpack) < fake.commands.index("mkdir /workspace/grader-" + token)


def test_grade_draws_different_grader_dirs_without_token():
    def grader_dirs(fake):
        return [c for c in fake.commands if c.startswith("mkdir /workspace/grader-")]

    first_fake = _FakeWorld()
    grade(_maker(first_fake), state_tar=b"", grader_tar=b"", command="true")
    second_fake = _FakeWorld()
    grade(_maker(second_fake), state_tar=b"", grader_tar=b"", command="true")
    assert len(grader_dirs(first_fake)) == 1
    assert len(grader_dirs(second_fake)) == 1
    assert grader_dirs(first_fake) != grader_dirs(second_fake)


def test_grade_final_step_exports_both_dirs_before_command():
    token = "ab12cd34"
    fake = _FakeWorld()
    grade(_maker(fake), state_tar=b"", grader_tar=b"", command="./grader/run", token=token)
    assert fake.commands[-1] == (
        "STATE_DIR=/workspace/state GRADER_DIR=/workspace/grader-"
        + token
        + "; export STATE_DIR GRADER_DIR; ./grader/run"
    )


@pytest.mark.parametrize(
    ("payload", "want_chunks"),
    [(b"123456", 1), (b"1234567", 2)],
    ids=["exact-chunk-bytes-is-one-chunk", "one-byte-over-is-two-chunks"],
)
def test_grade_splits_payload_at_chunk_boundaries(payload, want_chunks):
    fake = _FakeWorld()
    grade(_maker(fake), state_tar=payload, grader_tar=b"", command="true", chunk_bytes=CHUNK)
    chunks = [c for c in fake.commands if c.startswith("printf %s")]
    assert len(chunks) == want_chunks


def test_grade_empty_inputs_create_dirs_and_skip_chunks_and_unpack():
    token = "ab12cd34"
    fake = _FakeWorld([_ok(), _ok(), _ok(b"", b"", 0)])
    receipt = grade(_maker(fake), state_tar=b"", grader_tar=b"", command="true", token=token)
    assert fake.commands == [
        "mkdir /workspace/state",
        "mkdir /workspace/grader-" + token,
        "STATE_DIR=/workspace/state GRADER_DIR=/workspace/grader-"
        + token
        + "; export STATE_DIR GRADER_DIR; true",
    ]
    assert receipt.passed is True


@pytest.mark.parametrize(
    ("answers", "stage"),
    [
        ([StepReceipt(b"", b"exists\n", 1, False, False)], "state mkdir"),
        ([_ok(), StepReceipt(b"", b"exists\n", 1, False, False)], "grader mkdir"),
    ],
    ids=["state-mkdir-failure-names-stage", "grader-mkdir-failure-names-stage"],
)
def test_grade_mkdir_failure_names_stage_skips_grade_and_closes(answers, stage):
    token = "ab12cd34"
    fake = _FakeWorld(answers)
    with pytest.raises(GradeError, match=stage):
        grade(_maker(fake), state_tar=b"", grader_tar=b"", command="./grader/run", token=token)
    assert all("export STATE_DIR GRADER_DIR" not in c for c in fake.commands)
    assert fake.close_calls == 1


@pytest.mark.parametrize(
    "token",
    ["AB12CD34", "ab12", "ab12/cd34", "ab12'cd34", 12345, b"ab12cd34"],
    ids=["uppercase", "too-short", "slash", "quote", "non-str-int", "non-str-bytes"],
)
def test_grade_refuses_bad_token_before_any_world_call(token):
    fake = _FakeWorld()
    with pytest.raises(GradeError):
        grade(_maker(fake), state_tar=b"", grader_tar=b"", command="true", token=token)
    assert fake.reset_calls == 0
    assert fake.commands == []
    assert fake.close_calls == 0


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
    ],
    ids=[
        "mkdir-failure-names-stage",
        "chunk-failure-names-stage",
        "unpack-failure-names-stage",
        "unpack-timeout-names-stage",
        "unpack-truncation-names-stage",
    ],
)
def test_grade_feeding_failure_names_stage_skips_grade_and_closes(answers, stage):
    fake = _FakeWorld(answers)
    with pytest.raises(GradeError, match=stage):
        grade(_maker(fake), state_tar=b"123456", grader_tar=b"", command="./grader/run", chunk_bytes=CHUNK)
    assert "./grader/run" not in fake.commands
    assert fake.close_calls == 1


def test_grade_close_runs_and_interrupt_propagates_on_step_interrupt():
    fake = _FakeWorld([_ok(), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        grade(_maker(fake), state_tar=b"123456", grader_tar=b"", command="true", chunk_bytes=CHUNK)
    assert fake.close_calls == 1


def _bad_case_kwargs(case):
    if case == "chunk-not-multiple-of-3":
        return {"state_tar": b"123", "grader_tar": b"", "command": "true", "chunk_bytes": 7}
    if case == "chunk-zero":
        return {"state_tar": b"123", "grader_tar": b"", "command": "true", "chunk_bytes": 0}
    if case == "chunk-not-int":
        return {"state_tar": b"123", "grader_tar": b"", "command": "true", "chunk_bytes": "48"}
    if case == "oversize-state":
        return {"state_tar": b"12345", "grader_tar": b"", "command": "true", "max_input_bytes": 4}
    if case == "oversize-grader":
        return {"state_tar": b"", "grader_tar": b"12345", "command": "true", "max_input_bytes": 4}
    if case == "max-none":
        return {"state_tar": b"", "grader_tar": b"", "command": "true", "max_input_bytes": None}
    if case == "max-str":
        return {"state_tar": b"", "grader_tar": b"", "command": "true", "max_input_bytes": "16777216"}
    if case == "max-bool":
        return {"state_tar": b"", "grader_tar": b"", "command": "true", "max_input_bytes": True}
    if case == "max-zero":
        return {"state_tar": b"", "grader_tar": b"", "command": "true", "max_input_bytes": 0}
    if case == "max-negative":
        return {"state_tar": b"", "grader_tar": b"", "command": "true", "max_input_bytes": -1}
    if case == "max-above-ceiling":
        return {"state_tar": b"", "grader_tar": b"", "command": "true", "max_input_bytes": 16_777_217}
    if case == "non-bytes-state":
        return {"state_tar": "123", "grader_tar": b"", "command": "true"}
    if case == "non-bytes-grader":
        return {"state_tar": b"", "grader_tar": 123, "command": "true"}
    return {"state_tar": b"", "grader_tar": b"", "command": ""}


@pytest.mark.parametrize(
    "case",
    [
        "chunk-not-multiple-of-3",
        "chunk-zero",
        "chunk-not-int",
        "oversize-state",
        "oversize-grader",
        "max-none",
        "max-str",
        "max-bool",
        "max-zero",
        "max-negative",
        "max-above-ceiling",
        "non-bytes-state",
        "non-bytes-grader",
        "empty-command",
    ],
    ids=[
        "chunk-not-multiple-of-3",
        "chunk-zero",
        "chunk-not-int",
        "oversize-state",
        "oversize-grader",
        "max-none",
        "max-str",
        "max-bool",
        "max-zero",
        "max-negative",
        "max-above-ceiling",
        "non-bytes-state",
        "non-bytes-grader",
        "empty-command",
    ],
)
def test_grade_refuses_bad_inputs_before_any_world_call(case):
    fake = _FakeWorld()
    with pytest.raises(GradeError):
        grade(_maker(fake), **_bad_case_kwargs(case))
    assert fake.reset_calls == 0
    assert fake.commands == []
    assert fake.close_calls == 0


def test_grade_accepts_ceiling_max_input_bytes():
    fake = _FakeWorld()
    receipt = grade(_maker(fake), state_tar=b"", grader_tar=b"", command="true", max_input_bytes=16_777_216)
    assert receipt.passed is True
    assert fake.reset_calls == 1
    assert fake.close_calls == 1
