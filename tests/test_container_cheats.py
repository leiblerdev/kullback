import base64
import hashlib
import io
import shlex
import tarfile

import pytest

from kullback.container_cheats import (
    CHEATS,
    CheatReport,
    CheatResult,
    ScanHit,
    ScanReport,
    _all_defeated,
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


def test_all_defeated_needs_a_run_and_every_cheat_defeated():
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
    folds = [
        ((), False),
        ((CheatResult("a", "defeated", ""),), True),
        ((CheatResult("a", "defeated", ""), CheatResult("b", "succeeded", "")), False),
        ((CheatResult("a", "no_verdict", "x"),), False),
    ]
    assert [_all_defeated(results) for results, _defeated in folds] == [defeated for _results, defeated in folds]


def test_run_cheats_nonzero_cheat_step_still_grades():
    cheat = next(c for c in CHEATS if c.name == "aim_at_grader")
    report, candidates, graders = _run_single(cheat, candidate_answers=[_failed(), _ok(), _ok()])
    assert report.results[0].outcome == "defeated"
    assert len(graders[0].commands) == 5


class _BadClose(_FakeCandidate):
    def close(self):
        self.close_calls += 1
        raise WorldError("stuck lid")


class _InterruptedClose(_FakeCandidate):
    def close(self):
        self.close_calls += 1
        raise KeyboardInterrupt()


def _raising(error):
    def make(*_args):
        raise error

    return make


def _cheat(name):
    return next(c for c in CHEATS if c.name == name)


EXPORT_LIMIT = 16_777_216


@pytest.mark.parametrize(
    ("cheats", "candidate", "grader", "outcomes", "detail", "closes", "exports", "graders_made"),
    [
        pytest.param(
            (_cheat("delete_tests"),),
            lambda _i: _FakeCandidate([WorldError("nope")]),
            lambda: _FakeGrader(_grader_defeated()),
            ["no_verdict"],
            "nope",
            [1],
            [[]],
            0,
            id="world-error-step",
        ),
        pytest.param(
            (_cheat("delete_tests"),),
            lambda _i: _FakeCandidate(export_error=ExportError("disk gone")),
            lambda: _FakeGrader(_grader_defeated()),
            ["no_verdict"],
            "disk gone",
            [1],
            [[EXPORT_LIMIT]],
            0,
            id="export-error",
        ),
        pytest.param(
            (_cheat("fake_exit_code"),),
            lambda _i: _FakeCandidate([_timed_out()]),
            lambda: _FakeGrader(_grader_defeated()),
            ["no_verdict"],
            "timed out",
            [1],
            [[]],
            0,
            id="cheat-step-timeout",
        ),
        pytest.param(
            (_cheat("fake_exit_code"),),
            lambda _i: _FakeCandidate([_truncated()]),
            lambda: _FakeGrader(_grader_defeated()),
            ["no_verdict"],
            "truncated output",
            [1],
            [[]],
            0,
            id="cheat-step-truncation",
        ),
        pytest.param(
            (_cheat("do_nothing"),),
            lambda _i: _FakeCandidate(),
            lambda: _FakeGrader([_failed()]),
            ["no_verdict"],
            "",
            [1],
            [[EXPORT_LIMIT]],
            1,
            id="grader-failure",
        ),
        pytest.param(
            (CHEATS[0],),
            lambda _i: _FakeCandidate(),
            _raising(WorldError("docker gone")),
            ["no_verdict"],
            "",
            [1],
            [[EXPORT_LIMIT]],
            0,
            id="grader-world-error",
        ),
        pytest.param(
            (CHEATS[0],),
            _raising(WorldError("launch gone")),
            lambda: _FakeGrader(_grader_defeated()),
            ["no_verdict"],
            "launch gone",
            [],
            [],
            0,
            id="candidate-factory-error",
        ),
        pytest.param(
            (CHEATS[0], CHEATS[1]),
            lambda i: _BadClose() if i == 0 else _FakeCandidate(),
            lambda: _FakeGrader(_grader_defeated()),
            ["no_verdict", "defeated"],
            "stuck lid",
            [1, 1],
            [[EXPORT_LIMIT], [EXPORT_LIMIT]],
            1,
            id="close-error-then-next-cheat-runs",
        ),
    ],
)
def test_a_cheat_run_that_breaks_is_no_verdict_never_a_defeat(
    cheats, candidate, grader, outcomes, detail, closes, exports, graders_made
):
    candidates = []
    graders = []

    def make_candidate():
        fake = candidate(len(candidates))
        candidates.append(fake)
        return fake

    report = run_cheats(
        make_candidate,
        _maker(graders, grader),
        grader_tar=GRADER_TAR,
        command=COMMAND,
        cheats=cheats,
    )
    assert [r.outcome for r in report.results] == outcomes
    assert detail in report.results[0].detail
    assert report.all_defeated is False
    assert [f.close_calls for f in candidates] == closes
    assert [f.export_limits for f in candidates] == exports
    assert len(graders) == graders_made
    assert all(f.close_calls == 1 for f in graders)


@pytest.mark.parametrize(
    ("fault", "candidate", "cheats"),
    [
        (KeyboardInterrupt, lambda: _FakeCandidate([KeyboardInterrupt()]), (CHEATS[3], CHEATS[0])),
        (SystemExit, lambda: _FakeCandidate([SystemExit(1)]), (CHEATS[3], CHEATS[0])),
        (KeyboardInterrupt, _InterruptedClose, (CHEATS[0], CHEATS[1])),
    ],
    ids=["interrupt", "exit", "interrupt-in-close"],
)
def test_run_cheats_fault_propagates_after_close_without_next_cheat(fault, candidate, cheats):
    candidates = []
    graders = []
    with pytest.raises(fault):
        run_cheats(
            _maker(candidates, candidate),
            _maker(graders, lambda: _FakeGrader(_grader_defeated())),
            grader_tar=GRADER_TAR,
            command=COMMAND,
            cheats=cheats,
        )
    assert len(candidates) == 1
    assert candidates[0].close_calls == 1
    assert graders == []


def _scan_candidate(answers):
    store = []

    def make():
        fake = _FakeCandidate(answers)
        store.append(fake)
        return fake

    return (make, store)


ID_A = "a" * 40
ID_B = "b" * 40
ID_C = "c" * 40
ID_D = "d" * 40
ID_E = "e" * 40


def _history_answers(all_out, head_out, fsck_out, repo=b"/workspace/a/.git\n"):
    return [_ok(b""), _ok(repo), _ok(b"/usr/bin/git\n"), all_out, head_out, fsck_out]


def _ids(*ids):
    return _ok(("\n".join(ids) + "\n").encode())


def _sha_line(data, path):
    return hashlib.sha256(data).hexdigest() + "  " + path + "\n"


def test_scan_hidden_test_hit_reports_path_and_member():
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


@pytest.mark.parametrize(
    ("answers", "members", "ground"),
    [
        ([_ok(b""), _ok(b"")], (), "not_run"),
        ([_ok(b""), _ok(), _ok(), StepReceipt(b"", b"", 1, False, False), _ok(b"")], ("expected.txt",), "clean"),
    ],
    ids=["empty-workspace", "ground-truth-no-match"],
)
def test_scan_with_nothing_found_is_clean(answers, members, ground):
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR, answer_members=members)
    assert report.hidden_tests.outcome == "clean"
    assert report.ground_truth.outcome == ground
    assert report.clean is True


def _bad_reset():
    class _BadReset(_FakeCandidate):
        def reset(self):
            raise WorldError("reset gone")

    return _BadReset()


_HIDDEN_NO_SIGNAL = {"hidden_tests": "no_signal"}
_GROUND_NO_SIGNAL = {"ground_truth": "no_signal"}
_HISTORY_NO_SIGNAL = {"history": "no_signal"}
_GROUND = ("expected.txt",)


@pytest.mark.parametrize(
    ("candidate", "members", "want", "close_error", "closes"),
    [
        pytest.param(
            lambda: _FakeCandidate([_failed(1), _ok(b"")]), (), _HIDDEN_NO_SIGNAL, "", [1], id="listing-exit-1"
        ),
        pytest.param(
            lambda: _FakeCandidate([_timed_out(), _ok(b"")]), (), _HIDDEN_NO_SIGNAL, "", [1], id="listing-timeout"
        ),
        pytest.param(
            lambda: _FakeCandidate([WorldError("lost"), _ok(b"")]),
            (),
            _HIDDEN_NO_SIGNAL,
            "",
            [1],
            id="listing-world-error",
        ),
        pytest.param(
            lambda: _FakeCandidate([_ok(b"bogus line\n"), _ok(b"")]),
            (),
            _HIDDEN_NO_SIGNAL,
            "",
            [1],
            id="unparsable-sha-line",
        ),
        pytest.param(
            lambda: _FakeCandidate([_ok(b""), _ok(), _ok(), StepReceipt(b"", b"err\n", 2, False, False), _ok(b"")]),
            _GROUND,
            _GROUND_NO_SIGNAL,
            "",
            [1],
            id="ground-truth-exit-2",
        ),
        pytest.param(
            lambda: _FakeCandidate([_ok(b""), _ok(), _ok(), _timed_out(), _ok(b"")]),
            _GROUND,
            _GROUND_NO_SIGNAL,
            "",
            [1],
            id="ground-truth-timeout",
        ),
        pytest.param(
            lambda: _FakeCandidate([_ok(b""), _failed(), _ok(b"")]),
            _GROUND,
            _GROUND_NO_SIGNAL,
            "",
            [1],
            id="ground-truth-feed-failure",
        ),
        pytest.param(
            lambda: _FakeCandidate(_history_answers(_failed(128), _ids(ID_A), _ok(b""))),
            (),
            _HISTORY_NO_SIGNAL,
            "",
            [1],
            id="history-all-listing-exit-128",
        ),
        pytest.param(
            lambda: _FakeCandidate(_history_answers(_truncated(), _ids(ID_A), _ok(b""))),
            (),
            _HISTORY_NO_SIGNAL,
            "",
            [1],
            id="history-all-listing-truncated",
        ),
        pytest.param(
            lambda: _FakeCandidate(_history_answers(_ok(b"lots\n"), _ids(ID_A), _ok(b""))),
            (),
            _HISTORY_NO_SIGNAL,
            "",
            [1],
            id="history-junk-in-all",
        ),
        pytest.param(
            lambda: _FakeCandidate(_history_answers(_ids(ID_A), _ok(b"lots\n"), _ok(b""))),
            (),
            _HISTORY_NO_SIGNAL,
            "",
            [1],
            id="history-junk-in-head",
        ),
        pytest.param(
            lambda: _FakeCandidate([_ok(b""), _failed(1)]), (), _HISTORY_NO_SIGNAL, "", [1], id="repo-listing-exit-1"
        ),
        pytest.param(
            lambda: _FakeCandidate([_ok(b""), _timed_out()]), (), _HISTORY_NO_SIGNAL, "", [1], id="repo-listing-timeout"
        ),
        pytest.param(
            lambda: _FakeCandidate([_ok(b""), WorldError("lost")]),
            (),
            _HISTORY_NO_SIGNAL,
            "",
            [1],
            id="repo-listing-world-error",
        ),
        pytest.param(
            lambda: _FakeCandidate([_ok(b""), _ok(b"/workspace/a/.git\n"), StepReceipt(b"", b"", 1, False, False)]),
            (),
            _HISTORY_NO_SIGNAL,
            "",
            [1],
            id="no-git-binary",
        ),
        pytest.param(
            lambda: _FakeCandidate([_ok(b""), _ok(b"/workspace/a/.git\n"), _ok(b"/usr/bin/git\n"), WorldError("lost")]),
            (),
            _HISTORY_NO_SIGNAL,
            "",
            [1],
            id="repo-step-world-error",
        ),
        pytest.param(
            lambda: _FakeCandidate(_history_answers(_ids(ID_A, ID_B, ID_C, ID_D), _failed(128), _ok(b""))),
            (),
            _HISTORY_NO_SIGNAL,
            "",
            [1],
            id="history-head-failure",
        ),
        pytest.param(
            lambda: _BadClose([_ok(b""), _ok(b"")]),
            (),
            {"hidden_tests": "clean", "history": "clean"},
            "stuck lid",
            [1],
            id="close-error-keeps-outcomes",
        ),
        pytest.param(
            _raising(WorldError("launch gone")),
            (),
            {"hidden_tests": "no_signal", "history": "no_signal", "ground_truth": "not_run"},
            "",
            [],
            id="setup-failure-ground-not-run",
        ),
        pytest.param(
            _raising(WorldError("launch gone")),
            _GROUND,
            {"hidden_tests": "no_signal", "history": "no_signal", "ground_truth": "no_signal"},
            "",
            [],
            id="setup-failure-ground-no-signal",
        ),
        pytest.param(
            _bad_reset,
            (),
            {"hidden_tests": "no_signal", "history": "no_signal"},
            "",
            [1],
            id="reset-failure-closes",
        ),
    ],
)
def test_an_unreadable_scan_is_no_signal_never_clean(candidate, members, want, close_error, closes):
    store = []

    def make():
        fake = candidate()
        store.append(fake)
        return fake

    report = scan_start(make, grader_tar=SCAN_TAR, answer_members=members)
    assert {name: getattr(report, name).outcome for name in want} == want
    assert close_error in report.close_error
    assert report.clean is False
    assert [f.close_calls for f in store] == closes


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


def test_scan_ground_truth_only_matching_itself_is_clean():
    answers = [_ok(b""), _ok(), _ok(), _ok(b"/workspace/.scan-patterns.txt\n"), _ok(b"")]
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR, answer_members=("expected.txt",))
    assert report.ground_truth.outcome == "clean"


@pytest.mark.parametrize(
    ("answers", "outcome", "hits"),
    [
        pytest.param(
            _history_answers(
                _ids(ID_A, ID_B, ID_C, ID_D, ID_E), _ids(ID_A, ID_B, ID_C), _ok(b""), repo=b"/workspace/proj/.git\n"
            ),
            "hit",
            (ScanHit("history", "/workspace/proj/.git", "", "all=5 head=3 unreachable=0"),),
            id="all-commits-exceed-head",
        ),
        pytest.param(
            _history_answers(
                _ids(ID_A, ID_B, ID_C),
                _ids(ID_A, ID_B, ID_C),
                _ok(b"unreachable commit abc123\nunreachable blob def456\n"),
                repo=b"/workspace/proj/.git\n",
            ),
            "hit",
            (ScanHit("history", "/workspace/proj/.git", "", "all=3 head=3 unreachable=1"),),
            id="unreachable-commit-only",
        ),
        pytest.param(
            _history_answers(_ids(ID_A), _ids(ID_A), _ok(b"dangling commit abc123\n")),
            "hit",
            (ScanHit("history", "/workspace/a/.git", "", "all=1 head=1 unreachable=1"),),
            id="dangling-commit",
        ),
        pytest.param(
            _history_answers(
                _ids(ID_A, ID_B),
                _ok((ID_A + "\n\n" + ID_B + "\n").encode()),
                _ok(b"unreachable blob def456\n"),
                repo=b"/workspace/proj/.git\n\n",
            ),
            "clean",
            (),
            id="equal-counts-ignoring-blank-lines",
        ),
        pytest.param(
            _history_answers(
                _ids(ID_A, ID_B), _ids(ID_A, ID_B), _ok(b"warning: 2 commits were skipped\n\nunreachable blob def456\n")
            ),
            "clean",
            (),
            id="warning-and-blob-lines",
        ),
        pytest.param([_ok(b""), _ok(b"")], "clean", (), id="no-git-dir"),
    ],
)
def test_scan_history_hits_only_on_commits_hidden_beyond_head(answers, outcome, hits):
    make, _store = _scan_candidate(answers)
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == outcome
    assert report.history.hits == hits
    assert report.clean is (outcome == "clean")


def _run_with(**kwargs):
    def call(make_candidate, make_grader):
        return run_cheats(make_candidate, make_grader, **{"grader_tar": SCAN_TAR, "command": COMMAND, **kwargs})

    return call


def _scan_with(**kwargs):
    def call(make_candidate, _make_grader):
        return scan_start(make_candidate, **{"grader_tar": SCAN_TAR, **kwargs})

    return call


_BAD_OUTPUT_PATHS = [
    ("", "empty"),
    ("/abs.txt", "absolute"),
    ("../up.txt", "dotdot"),
    ("a/../b.txt", "nested-dotdot"),
    ("a\nb.txt", "newline"),
    ("a\x00b.txt", "nul"),
    (7, "non-str-int"),
    (None, "non-str-none"),
]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        pytest.param(_run_with(grader_tar=GRADER_TAR, cheats=()), "", id="run-empty-suite-tuple"),
        pytest.param(_run_with(grader_tar=GRADER_TAR, cheats=None), "", id="run-empty-suite-none"),
        *[
            pytest.param(_run_with(output_paths=(bad,)), repr(bad), id="run-output-path-" + name)
            for bad, name in _BAD_OUTPUT_PATHS
        ],
        pytest.param(_run_with(output_paths="answer.txt"), "", id="run-bare-string-paths"),
        pytest.param(_run_with(answer_members="expected.txt"), "", id="run-bare-string-members"),
        pytest.param(_run_with(answer_members=("missing.txt",)), "", id="run-unknown-answer-member"),
        pytest.param(_run_with(answer_members=("docs/",)), "", id="run-directory-answer-member"),
        pytest.param(_run_with(cheats=(CHEATS[0],), output_paths=("a.txt",)), "", id="run-explicit-cheats-with-paths"),
        pytest.param(
            _run_with(cheats=(CHEATS[0],), answer_members=("expected.txt",)), "", id="run-explicit-cheats-with-members"
        ),
        pytest.param(_scan_with(answer_members=("missing.txt",)), "", id="scan-unknown-answer-member"),
        pytest.param(_scan_with(answer_members=("docs/",)), "", id="scan-directory-answer-member"),
        pytest.param(
            _scan_with(
                grader_tar=_make_tar(
                    {"many.txt": b"first-long-line-here\nsecond-long-line-here\nthird-long-line-here\n"}
                ),
                answer_members=("many.txt",),
                max_strings=2,
            ),
            "",
            id="scan-too-many-strings",
        ),
    ],
)
def test_bad_cheat_arguments_are_refused_before_any_container_starts(call, message):
    candidates = []
    graders = []
    with pytest.raises(ValueError) as excinfo:
        call(
            _maker(candidates, _FakeCandidate),
            _maker(graders, lambda: _FakeGrader(_grader_defeated())),
        )
    assert message in str(excinfo.value)
    assert candidates == []
    assert graders == []


def test_a_hostile_repo_path_is_quoted_in_every_command():
    repo = "/workspace/my dir/o'brien/.git"
    make, store = _scan_candidate(_history_answers(_ids(ID_A), _ids(ID_A), _ok(b""), repo=(repo + "\n").encode()))
    report = scan_start(make, grader_tar=SCAN_TAR)
    assert report.history.outcome == "clean"
    quoted = shlex.quote(repo)
    git_cmds = [c for c in store[0].commands if c.startswith("git -C")]
    assert git_cmds == [
        "git -C " + quoted + " rev-list --all --reflog",
        "git -C " + quoted + " rev-list HEAD",
        "git -C " + quoted + " fsck --unreachable --no-reflogs",
    ]
    assert all("|" not in c for c in store[0].commands)
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
