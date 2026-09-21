import base64
import io
import shlex
import tarfile

import pytest

from kullback.container_cheats import (
    CHEATS,
    DEFAULT_ANSWER_MEMBERS,
    DEFAULT_OUTPUT_PATHS,
    Cheat,
    CheatReport,
    CheatResult,
    ScanHit,
    ScanReport,
    aim_cheats,
    run_cheats,
    scan_start,
)
from kullback.container_world import ExportError, StepReceipt, WorldError

COMMAND = "run-checks"


def _ok(stdout=b"", stderr=b"", code=0):
    return StepReceipt(stdout, stderr, code, False, False)


def _timed_out():
    return StepReceipt(b"part", b"", 0, True, False)


def _truncated():
    return StepReceipt(b"part", b"", 0, False, True)


def _failed(code=1):
    return StepReceipt(b"", b"nope\n", code, False, False)


class _FakeCandidate:
    """Candidate-side fake: scripted step answers, exportable state, close counting."""

    def __init__(self, answers=(), state=b"", export_error=None):
        self._answers = list(answers)
        self._state = state
        self._export_error = export_error
        self.commands = []
        self.export_limits = []
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self):
        self.reset_calls += 1
        return "fake-candidate-id"

    def step(self, command):
        self.commands.append(command)
        item = self._answers.pop(0) if self._answers else _ok()
        if isinstance(item, BaseException):
            raise item
        return item

    def export_workspace(self, limit_bytes):
        self.export_limits.append(limit_bytes)
        if self._export_error is not None:
            raise self._export_error
        return self._state

    def close(self):
        self.close_calls += 1
        return True


class _FakeGrader:
    """Grader-side fake: the scripted answers drive container_grader.grade for real."""

    def __init__(self, answers=()):
        self._answers = list(answers)
        self.commands = []
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self):
        self.reset_calls += 1
        return "fake-grader-id"

    def step(self, command):
        self.commands.append(command)
        item = self._answers.pop(0) if self._answers else _ok()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.close_calls += 1
        return True


def _maker(store, factory):
    def make():
        fake = factory()
        store.append(fake)
        return fake

    return make


def _make_tar(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            if name.endswith("/"):
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


GRADER_TAR = _make_tar({"run.sh": b"#!/bin/sh\nexit 0\n"})

ANSWER_CONTENT = b"alpha-answer-line-one\nbeta-answer-line-two-here\nshort\n"
SCAN_TAR = _make_tar(
    {
        "expected.txt": ANSWER_CONTENT,
        "tiny.txt": b"tinybytes!",
        "docs/": None,
    }
)


def _grader_defeated():
    return [_ok(), _ok(), _ok(), _ok(), _failed(1)]


def _grader_passed():
    return [_ok(), _ok(), _ok(), _ok(), _ok(b"graded\n")]


def _run_single(cheat, candidate_answers=(), state=b"", export_error=None, grader_answers=None):
    candidates = []
    graders = []
    if grader_answers is None:
        grader_answers = _grader_defeated()
    report = run_cheats(
        _maker(candidates, lambda: _FakeCandidate(candidate_answers, state, export_error)),
        _maker(graders, lambda: _FakeGrader(grader_answers)),
        grader_tar=GRADER_TAR,
        command=COMMAND,
        cheats=(cheat,),
    )
    return (report, candidates, graders)


def test_cheat_suite_has_four_named_cheats():
    assert [c.name for c in CHEATS] == ["do_nothing", "delete_tests", "fake_exit_code", "aim_at_grader"]
    assert all(isinstance(c, Cheat) for c in CHEATS)
    assert all(isinstance(c.commands, tuple) for c in CHEATS)


def test_cheat_commands_use_no_heredocs():
    for cheat in CHEATS:
        for command in cheat.commands:
            assert "<<" not in command


def test_delete_tests_cheat_removes_test_like_names_with_find_and_rm():
    cheat = next(c for c in CHEATS if c.name == "delete_tests")
    text = "\n".join(cheat.commands)
    for token in ("find", "rm", "test*", "*_test*", "tests", "check*", "expected*"):
        assert token in text


def test_fake_exit_code_cheat_plants_runners_and_hooks_with_printf():
    cheat = next(c for c in CHEATS if c.name == "fake_exit_code")
    text = "\n".join(cheat.commands)
    for token in ("pytest", "python", "make", "cargo", "conftest.py", "sitecustomize.py", "printf"):
        assert token in text
    assert "pytest_sessionfinish" in text
    assert "_exit" in text


def test_aim_at_grader_cheat_links_output_paths_at_guessed_grader_dir():
    cheat = next(c for c in CHEATS if c.name == "aim_at_grader")
    text = "\n".join(cheat.commands)
    for token in ("/workspace/grader/expected.txt", "../grader"):
        assert token in text
    assert "ln -s" in text
    assert "../grader/expected.txt" not in text


def _run_full_suite():
    candidates = []
    graders = []
    report = run_cheats(
        _maker(candidates, _FakeCandidate),
        _maker(graders, lambda: _FakeGrader(_grader_defeated())),
        grader_tar=GRADER_TAR,
        command=COMMAND,
    )
    return (report, candidates, graders)


def test_run_cheats_all_defeated_reports_every_cheat():
    report, candidates, graders = _run_full_suite()
    assert isinstance(report, CheatReport)
    assert len(report.results) == len(CHEATS) == 4
    assert len(candidates) == 4
    assert len(graders) == 4
    assert all(isinstance(r, CheatResult) for r in report.results)
    assert all(r.outcome == "defeated" for r in report.results)
    assert [r.name for r in report.results] == [c.name for c in CHEATS]
    assert report.all_defeated is True


def test_run_cheats_closes_every_candidate_world_once():
    _report, candidates, graders = _run_full_suite()
    assert sum(f.close_calls for f in candidates) == 4
    assert all(f.reset_calls == 1 for f in candidates)
    assert sum(f.close_calls for f in graders) == 4


@pytest.mark.parametrize("index", [0, 1, 2, 3], ids=["do-nothing", "delete", "fake-exit", "aim"])
def test_run_cheats_sends_each_cheat_commands_verbatim(index):
    cheat = CHEATS[index]
    report, candidates, _graders = _run_single(cheat)
    assert report.results[0].outcome == "defeated"
    assert candidates[0].commands == list(cheat.commands)


def test_run_cheats_mixed_pass_and_fail_is_not_all_defeated():
    candidates = []
    graders = []

    def make_grader():
        if graders:
            fake = _FakeGrader(_grader_passed())
        else:
            fake = _FakeGrader(_grader_defeated())
        graders.append(fake)
        return fake

    report = run_cheats(
        _maker(candidates, _FakeCandidate),
        make_grader,
        grader_tar=GRADER_TAR,
        command=COMMAND,
        cheats=(CHEATS[0], CHEATS[1]),
    )
    assert [r.outcome for r in report.results] == ["defeated", "succeeded"]
    assert [r.name for r in report.results] == ["do_nothing", "delete_tests"]
    assert report.all_defeated is False
    assert all(f.close_calls == 1 for f in candidates)


def test_run_cheats_nonzero_cheat_step_still_grades():
    cheat = next(c for c in CHEATS if c.name == "aim_at_grader")
    report, candidates, graders = _run_single(cheat, candidate_answers=[_failed(), _ok(), _ok()])
    assert report.results[0].outcome == "defeated"
    assert len(graders[0].commands) == 5


def test_run_cheats_export_error_is_no_verdict_and_closes():
    cheat = next(c for c in CHEATS if c.name == "delete_tests")
    report, candidates, graders = _run_single(cheat, export_error=ExportError("disk gone"))
    assert report.results[0].outcome == "no_verdict"
    assert "disk gone" in report.results[0].detail
    assert report.all_defeated is False
    assert candidates[0].close_calls == 1
    assert graders == []


@pytest.mark.parametrize(
    ("answer", "label"),
    [(_timed_out(), "timed out"), (_truncated(), "truncated output")],
    ids=["timeout", "truncation"],
)
def test_run_cheats_bad_cheat_step_is_no_verdict_without_grading(answer, label):
    cheat = next(c for c in CHEATS if c.name == "fake_exit_code")
    report, candidates, graders = _run_single(cheat, candidate_answers=[answer])
    assert label in report.results[0].detail
    assert report.results[0].outcome == "no_verdict"
    assert report.all_defeated is False
    assert candidates[0].close_calls == 1
    assert graders == []


def test_run_cheats_world_error_step_is_no_verdict():
    cheat = next(c for c in CHEATS if c.name == "delete_tests")
    report, candidates, _graders = _run_single(cheat, candidate_answers=[WorldError("nope")])
    assert report.results[0].outcome == "no_verdict"
    assert "nope" in report.results[0].detail
    assert candidates[0].close_calls == 1


def test_run_cheats_grader_failure_is_no_verdict_and_closes_candidate():
    cheat = next(c for c in CHEATS if c.name == "do_nothing")
    report, candidates, graders = _run_single(cheat, grader_answers=[_failed()])
    assert report.results[0].outcome == "no_verdict"
    assert candidates[0].close_calls == 1
    assert graders[0].close_calls == 1


def test_run_cheats_grader_world_error_is_no_verdict():
    candidates = []
    report = run_cheats(
        _maker(candidates, _FakeCandidate),
        lambda: (_ for _ in ()).throw(WorldError("docker gone")),
        grader_tar=GRADER_TAR,
        command=COMMAND,
        cheats=(CHEATS[0],),
    )
    assert report.results[0].outcome == "no_verdict"
    assert candidates[0].close_calls == 1


def test_run_cheats_candidate_factory_error_is_no_verdict():
    def boom():
        raise WorldError("launch gone")

    graders = []
    report = run_cheats(
        boom,
        _maker(graders, lambda: _FakeGrader(_grader_defeated())),
        grader_tar=GRADER_TAR,
        command=COMMAND,
        cheats=(CHEATS[0],),
    )
    assert report.results[0].outcome == "no_verdict"
    assert "launch gone" in report.results[0].detail
    assert graders == []


@pytest.mark.parametrize("fault", [KeyboardInterrupt(), SystemExit(1)], ids=["interrupt", "exit"])
def test_run_cheats_fault_propagates_after_close_without_next_cheat(fault):
    candidates = []
    graders = []
    with pytest.raises(type(fault)):
        run_cheats(
            _maker(candidates, lambda: _FakeCandidate([fault])),
            _maker(graders, lambda: _FakeGrader(_grader_defeated())),
            grader_tar=GRADER_TAR,
            command=COMMAND,
            cheats=(CHEATS[3], CHEATS[0]),
        )
    assert len(candidates) == 1
    assert candidates[0].close_calls == 1
    assert graders == []


def test_run_cheats_forwards_export_limit_and_state_bytes():
    import hashlib

    state = b"state-bytes-123"
    cheat = next(c for c in CHEATS if c.name == "do_nothing")
    report, candidates, graders = _run_single(cheat, state=state, grader_answers=[_ok()] * 6 + [_failed(1)])
    assert report.results[0].outcome == "defeated"
    assert candidates[0].export_limits == [16_777_216]
    chunks = [c for c in graders[0].commands if c.startswith("printf %s")]
    assert len(chunks) == 2
    assert base64.b64decode(chunks[0].split("'")[1]) == state
    assert hashlib.sha256(state).hexdigest() != hashlib.sha256(GRADER_TAR).hexdigest()


def test_run_cheats_custom_export_limit_reaches_world():
    cheat = next(c for c in CHEATS if c.name == "do_nothing")
    candidates = []
    report = run_cheats(
        _maker(candidates, _FakeCandidate),
        _maker([], lambda: _FakeGrader(_grader_defeated())),
        grader_tar=GRADER_TAR,
        command=COMMAND,
        cheats=(cheat,),
        export_limit_bytes=1024,
    )
    assert report.all_defeated is True
    assert candidates[0].export_limits == [1024]


def _scan_candidate(answers):
    store = []

    def make():
        fake = _FakeCandidate(answers)
        store.append(fake)
        return fake

    return (make, store)


def _sha_line(data, path):
    import hashlib

    return hashlib.sha256(data).hexdigest() + "  " + path + "\n"


def test_scan_hidden_test_hit_reports_path_and_member():
    import hashlib

    listing = _sha_line(ANSWER_CONTENT, "/workspace/work.txt") + _sha_line(b"other", "/workspace/other.txt")
    make, store = _scan_candidate([_ok(listing.encode()), _ok(b"")])
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert isinstance(report, ScanReport)
    assert report.hidden_tests.outcome == "hit"
    assert report.hidden_tests.hits == (ScanHit("hidden_test", "/workspace/work.txt", "expected.txt"),)
    assert hashlib.sha256(ANSWER_CONTENT).hexdigest() in listing
    assert report.ground_truth.outcome == "not_run"
    assert report.history.outcome == "clean"
    assert report.clean is False
    assert store[0].close_calls == 1
    assert all("grep" not in c for c in store[0].commands)


def test_scan_tiny_grader_file_below_min_bytes_is_ignored():
    listing = _sha_line(b"tinybytes!", "/workspace/tiny-copy.txt")
    make, store = _scan_candidate([_ok(listing.encode()), _ok(b"")])
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.hidden_tests.outcome == "clean"
    assert report.clean is True
    assert store[0].close_calls == 1


def test_scan_empty_workspace_is_clean():
    make, store = _scan_candidate([_ok(b""), _ok(b"")])
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.hidden_tests.outcome == "clean"
    assert report.clean is True


def test_scan_unparsable_sha_line_is_no_signal():
    make, _store = _scan_candidate([_ok(b"bogus line\n"), _ok(b"")])
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.hidden_tests.outcome == "no_signal"
    assert report.clean is False


@pytest.mark.parametrize(
    "answer",
    [_failed(1), _timed_out(), WorldError("lost")],
    ids=["exit-1", "timeout", "world-error"],
)
def test_scan_unreadable_listing_is_no_signal(answer):
    make, _store = _scan_candidate([answer, _ok(b"")])
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.hidden_tests.outcome == "no_signal"
    assert report.clean is False


def test_scan_ground_truth_hit_lists_workspace_files():
    answers = [
        _ok(b""),
        _ok(),
        _ok(),
        _ok(b"/workspace/copy.txt\n/workspace/.scan-patterns.txt\n\n"),
        _ok(b""),
    ]
    make, store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR, answer_members=("expected.txt",))
    assert report.ground_truth.outcome == "hit"
    assert report.ground_truth.hits == (ScanHit("ground_truth", "/workspace/copy.txt"),)
    assert report.clean is False
    assert store[0].close_calls == 1
    chunks = [c for c in store[0].commands if "scan-patterns.b64" in c and c.startswith("printf %s")]
    assert len(chunks) == 1
    assert base64.b64decode(chunks[0].split("'")[1]) == b"alpha-answer-line-one\nbeta-answer-line-two-here"


def test_scan_ground_truth_no_match_is_clean():
    answers = [_ok(b""), _ok(), _ok(), StepReceipt(b"", b"", 1, False, False), _ok(b"")]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR, answer_members=("expected.txt",))
    assert report.ground_truth.outcome == "clean"
    assert report.clean is True


@pytest.mark.parametrize(
    "answer",
    [StepReceipt(b"", b"err\n", 2, False, False), _timed_out()],
    ids=["exit-2", "timeout"],
)
def test_scan_ground_truth_unreadable_is_no_signal(answer):
    answers = [_ok(b""), _ok(), _ok(), answer, _ok(b"")]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR, answer_members=("expected.txt",))
    assert report.ground_truth.outcome == "no_signal"
    assert report.clean is False


def test_scan_ground_truth_feed_failure_is_no_signal():
    answers = [_ok(b""), _failed(), _ok(b"")]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR, answer_members=("expected.txt",))
    assert report.ground_truth.outcome == "no_signal"
    assert report.clean is False


def test_scan_ground_truth_only_matching_itself_is_clean():
    answers = [_ok(b""), _ok(), _ok(), _ok(b"/workspace/.scan-patterns.txt\n"), _ok(b"")]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR, answer_members=("expected.txt",))
    assert report.ground_truth.outcome == "clean"


def test_scan_short_answer_lines_mean_clean_without_grep():
    tar = _make_tar({"note.txt": b"tiny\nsmall\n"})
    make, store = _scan_candidate([_ok(b""), _ok(b"")])
    report = scan_start(make, grader_tar=tar, answer_members=("note.txt",))
    assert report.ground_truth.outcome == "clean"
    assert all("grep" not in c for c in store[0].commands)


def test_scan_unknown_answer_member_is_value_error_before_any_world_call():
    store = []

    def make():
        store.append(1)
        return _FakeCandidate()

    with pytest.raises(ValueError):
        scan_start(make, grader_tar=SCAN_TAR, answer_members=("missing.txt",))
    assert store == []


def test_scan_directory_answer_member_is_value_error_before_any_world_call():
    store = []

    def make():
        store.append(1)
        return _FakeCandidate()

    with pytest.raises(ValueError):
        scan_start(make, grader_tar=SCAN_TAR, answer_members=("docs/",))
    assert store == []


def test_scan_too_many_strings_is_value_error_before_any_world_call():
    tar = _make_tar({"many.txt": b"first-long-line-here\nsecond-long-line-here\nthird-long-line-here\n"})
    store = []

    def make():
        store.append(1)
        return _FakeCandidate()

    with pytest.raises(ValueError):
        scan_start(make, grader_tar=tar, answer_members=("many.txt",), max_strings=2)
    assert store == []


def test_scan_string_count_exactly_at_max_strings_passes():
    tar = _make_tar({"many.txt": b"first-long-line-here\nsecond-long-line-here\n"})
    make, _store = _scan_candidate([_ok(b""), _ok(), _ok(), StepReceipt(b"", b"", 1, False, False), _ok(b"")])
    report = scan_start(make, grader_tar=tar, answer_members=("many.txt",), max_strings=2)
    assert report.ground_truth.outcome == "clean"


def test_scan_duplicate_answer_lines_are_fed_once():
    tar = _make_tar({"dup.txt": b"repeat-this-line-now\nrepeat-this-line-now\nshort\n"})
    make, store = _scan_candidate([_ok(b""), _ok(), _ok(), StepReceipt(b"", b"", 1, False, False), _ok(b"")])
    report = scan_start(make, grader_tar=tar, answer_members=("dup.txt",))
    assert report.ground_truth.outcome == "clean"
    chunks = [c for c in store[0].commands if c.startswith("printf %s")]
    assert len(chunks) == 1
    assert base64.b64decode(chunks[0].split("'")[1]) == b"repeat-this-line-now"


def test_scan_history_hit_when_all_commits_exceed_head():
    answers = [
        _ok(b""),
        _ok(b"/workspace/proj/.git\n"),
        _ok(b"/usr/bin/git\n"),
        _ok(b"5\n"),
        _ok(b"aaa\nbbb\nccc\n"),
        _ok(b""),
    ]
    make, store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "hit"
    assert report.history.hits == (
        ScanHit("history", "/workspace/proj/.git", "", "all=5 head=3 unreachable=0"),
    )
    assert report.clean is False


def test_scan_history_hit_on_unreachable_commit_only():
    answers = [
        _ok(b""),
        _ok(b"/workspace/proj/.git\n"),
        _ok(b"/usr/bin/git\n"),
        _ok(b"3\n"),
        _ok(b"aaa\nbbb\nccc\n"),
        _ok(b"unreachable commit abc123\nunreachable blob def456\n"),
    ]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "hit"
    assert report.history.hits[0].detail == "all=3 head=3 unreachable=1"


def test_scan_history_clean_on_equal_counts_ignoring_blank_lines():
    answers = [
        _ok(b""),
        _ok(b"/workspace/proj/.git\n\n"),
        _ok(b"/usr/bin/git\n"),
        _ok(b"2\n"),
        _ok(b"aaa\nbbb\n"),
        _ok(b"unreachable blob def456\n"),
    ]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "clean"
    assert report.clean is True


@pytest.mark.parametrize(
    "answer",
    [_failed(1), _timed_out(), WorldError("lost")],
    ids=["exit-1", "timeout", "world-error"],
)
def test_scan_history_unreadable_repo_listing_is_no_signal(answer):
    make, _store = _scan_candidate([_ok(b""), answer])
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "no_signal"
    assert report.clean is False


def test_scan_history_no_git_binary_is_no_signal():
    answers = [_ok(b""), _ok(b"/workspace/a/.git\n"), StepReceipt(b"", b"", 1, False, False)]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "no_signal"
    assert report.clean is False


def test_scan_history_no_git_dir_is_clean():
    make, _store = _scan_candidate([_ok(b""), _ok(b"")])
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "clean"


def test_scan_history_repo_path_is_shell_quoted_when_reused():
    repo = "/workspace/my dir/o'brien/.git"
    answers = [
        _ok(b""),
        _ok((repo + "\n").encode()),
        _ok(b"/usr/bin/git\n"),
        _ok(b"1\n"),
        _ok(b"aaa\n"),
        _ok(b""),
    ]
    make, store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "clean"
    quoted = shlex.quote(repo)
    git_cmds = [c for c in store[0].commands if c.startswith("git -C")]
    assert len(git_cmds) == 3
    assert all(quoted in c for c in git_cmds)


def test_scan_history_repo_step_error_is_no_signal():
    answers = [
        _ok(b""),
        _ok(b"/workspace/a/.git\n"),
        _ok(b"/usr/bin/git\n"),
        WorldError("lost"),
    ]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "no_signal"


def test_scan_history_head_failure_is_no_signal():
    answers = [
        _ok(b""),
        _ok(b"/workspace/a/.git\n"),
        _ok(b"/usr/bin/git\n"),
        _ok(b"4\n"),
        _failed(128),
    ]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "no_signal"


def test_scan_history_unparsable_count_is_no_signal():
    answers = [
        _ok(b""),
        _ok(b"/workspace/a/.git\n"),
        _ok(b"/usr/bin/git\n"),
        _ok(b"lots\n"),
        _ok(b"aaa\n"),
        _ok(b""),
    ]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "no_signal"


@pytest.mark.parametrize(
    "members",
    [(), ("expected.txt",)],
    ids=["ground-not-run", "ground-no-signal"],
)
def test_scan_setup_failure_marks_ran_checks_no_signal(members):
    def make():
        raise WorldError("launch gone")

    report = scan_start(make, grader_tar=SCAN_TAR, answer_members=members)
    assert report.hidden_tests.outcome == "no_signal"
    assert report.history.outcome == "no_signal"
    assert report.clean is False
    if members:
        assert report.ground_truth.outcome == "no_signal"
    else:
        assert report.ground_truth.outcome == "not_run"


def test_scan_reset_failure_marks_no_signal_and_closes():
    class _BadReset(_FakeCandidate):
        def reset(self):
            raise WorldError("reset gone")

    store = []

    def make():
        fake = _BadReset()
        store.append(fake)
        return fake

    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.hidden_tests.outcome == "no_signal"
    assert report.history.outcome == "no_signal"
    assert report.clean is False
    assert store[0].close_calls == 1


def test_aim_cheats_builds_one_cheat_per_member_with_quoted_links():
    built = aim_cheats(("out/answer.txt",), ("expected.txt", "keys/a b.json"))
    assert [c.name for c in built] == ["aim_at_grader:expected.txt", "aim_at_grader:keys/a b.json"]
    assert all(isinstance(c, Cheat) for c in built)
    mkdir = "mkdir -p " + shlex.quote("/workspace/out") + " 2>/dev/null || true"
    assert built[0].commands[0] == mkdir
    assert built[1].commands[0] == mkdir
    assert built[0].commands[1] == (
        "ln -sf "
        + shlex.quote("/workspace/grader/expected.txt")
        + " "
        + shlex.quote("/workspace/out/answer.txt")
        + " 2>/dev/null || true"
    )
    assert built[1].commands[1] == (
        "ln -sf "
        + shlex.quote("/workspace/grader/keys/a b.json")
        + " "
        + shlex.quote("/workspace/out/answer.txt")
        + " 2>/dev/null || true"
    )
    assert built[0].commands[-1] == built[1].commands[-1]
    assert built[0].commands[-1].startswith("ln -sf ")


def test_aim_cheats_quotes_hostile_member_and_path():
    built = aim_cheats(("o'x/y y.txt",), ("a'b.txt",))
    assert [c.name for c in built] == ["aim_at_grader:a'b.txt"]
    assert built[0].commands[0] == "mkdir -p " + shlex.quote("/workspace/o'x") + " 2>/dev/null || true"
    assert built[0].commands[1] == (
        "ln -sf "
        + shlex.quote("/workspace/grader/a'b.txt")
        + " "
        + shlex.quote("/workspace/o'x/y y.txt")
        + " 2>/dev/null || true"
    )


def test_aim_cheats_defaults_name_plain_aim_covering_default_paths():
    assert DEFAULT_OUTPUT_PATHS == (
        "answer.txt",
        "output.txt",
        "out.txt",
        "result.txt",
        "results.json",
        "solution.txt",
        "expected.txt",
    )
    assert DEFAULT_ANSWER_MEMBERS == ("expected.txt",)
    built = aim_cheats((), ())
    assert len(built) == 1
    assert built[0].name == "aim_at_grader"
    assert CHEATS[3] == built[0]
    assert [c.name for c in CHEATS] == ["do_nothing", "delete_tests", "fake_exit_code", "aim_at_grader"]
    for name in DEFAULT_OUTPUT_PATHS:
        assert any(shlex.quote("/workspace/" + name) in c for c in built[0].commands)


@pytest.mark.parametrize(
    "bad",
    ["", "/abs.txt", "../up.txt", "a/../b.txt", "a\nb.txt", "a\x00b.txt", 7, None],
    ids=["empty", "absolute", "dotdot", "nested-dotdot", "newline", "nul", "non-str-int", "non-str-none"],
)
def test_run_cheats_bad_output_path_refused_before_any_world_call(bad):
    candidates = []
    graders = []
    with pytest.raises(ValueError) as excinfo:
        run_cheats(
            _maker(candidates, _FakeCandidate),
            _maker(graders, lambda: _FakeGrader(_grader_defeated())),
            grader_tar=SCAN_TAR,
            command=COMMAND,
            output_paths=(bad,),
        )
    assert repr(bad) in str(excinfo.value)
    assert candidates == []
    assert graders == []


@pytest.mark.parametrize(
    "kwargs",
    [{"output_paths": "answer.txt"}, {"answer_members": "expected.txt"}],
    ids=["paths", "members"],
)
def test_run_cheats_bare_string_refused_before_any_world_call(kwargs):
    candidates = []
    graders = []
    with pytest.raises(ValueError):
        run_cheats(
            _maker(candidates, _FakeCandidate),
            _maker(graders, lambda: _FakeGrader(_grader_defeated())),
            grader_tar=SCAN_TAR,
            command=COMMAND,
            **kwargs,
        )
    assert candidates == []
    assert graders == []


def test_run_cheats_unknown_answer_member_refused_before_any_world_call():
    candidates = []
    graders = []
    with pytest.raises(ValueError):
        run_cheats(
            _maker(candidates, _FakeCandidate),
            _maker(graders, lambda: _FakeGrader(_grader_defeated())),
            grader_tar=SCAN_TAR,
            command=COMMAND,
            answer_members=("missing.txt",),
        )
    assert candidates == []
    assert graders == []


def test_run_cheats_directory_answer_member_refused_before_any_world_call():
    candidates = []
    graders = []
    with pytest.raises(ValueError):
        run_cheats(
            _maker(candidates, _FakeCandidate),
            _maker(graders, lambda: _FakeGrader(_grader_defeated())),
            grader_tar=SCAN_TAR,
            command=COMMAND,
            answer_members=("docs/",),
        )
    assert candidates == []
    assert graders == []


@pytest.mark.parametrize(
    "kwargs",
    [{"output_paths": ("a.txt",)}, {"answer_members": ("expected.txt",)}],
    ids=["paths", "members"],
)
def test_run_cheats_explicit_cheats_with_paths_refused(kwargs):
    candidates = []
    graders = []
    with pytest.raises(ValueError):
        run_cheats(
            _maker(candidates, _FakeCandidate),
            _maker(graders, lambda: _FakeGrader(_grader_defeated())),
            grader_tar=SCAN_TAR,
            command=COMMAND,
            cheats=(CHEATS[0],),
            **kwargs,
        )
    assert candidates == []
    assert graders == []


def _aim_tar():
    return _make_tar({"expected.txt": ANSWER_CONTENT, "keys/a b.json": b'{"answer": 42}\n'})


def test_run_cheats_output_paths_replaces_default_aim_cheat():
    members = ("expected.txt", "keys/a b.json")
    candidates = []
    graders = []
    report = run_cheats(
        _maker(candidates, _FakeCandidate),
        _maker(graders, lambda: _FakeGrader(_grader_defeated())),
        grader_tar=_aim_tar(),
        command=COMMAND,
        output_paths=("out/answer.txt",),
        answer_members=members,
    )
    assert len(report.results) == 3 + len(members) == 5
    assert [r.name for r in report.results] == [
        "do_nothing",
        "delete_tests",
        "fake_exit_code",
        "aim_at_grader:expected.txt",
        "aim_at_grader:keys/a b.json",
    ]
    assert "aim_at_grader" not in [r.name for r in report.results]
    assert len(candidates) == 5
    assert all(f.close_calls == 1 for f in candidates)
    link = "ln -sf " + shlex.quote("/workspace/grader/keys/a b.json")
    assert any(link in c for c in candidates[4].commands)


def test_run_cheats_paths_only_builds_default_member_cheat():
    candidates = []
    graders = []
    report = run_cheats(
        _maker(candidates, _FakeCandidate),
        _maker(graders, lambda: _FakeGrader(_grader_defeated())),
        grader_tar=SCAN_TAR,
        command=COMMAND,
        output_paths=("out/answer.txt",),
    )
    assert len(report.results) == 4
    assert [r.name for r in report.results] == [
        "do_nothing",
        "delete_tests",
        "fake_exit_code",
        "aim_at_grader:expected.txt",
    ]


def test_scan_history_warning_and_blob_lines_are_clean():
    answers = [
        _ok(b""),
        _ok(b"/workspace/a/.git\n"),
        _ok(b"/usr/bin/git\n"),
        _ok(b"2\n"),
        _ok(b"aaa\nbbb\n"),
        _ok(b"warning: 2 commits were skipped\n\nunreachable blob def456\n"),
    ]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "clean"


def test_scan_history_dangling_commit_is_hit():
    answers = [
        _ok(b""),
        _ok(b"/workspace/a/.git\n"),
        _ok(b"/usr/bin/git\n"),
        _ok(b"1\n"),
        _ok(b"aaa\n"),
        _ok(b"dangling commit abc123\n"),
    ]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "hit"
    assert report.history.hits[0].detail == "all=1 head=1 unreachable=1"

