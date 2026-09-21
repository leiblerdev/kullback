"""Cheat suite and birth scan for container Tasks.

At birth every Task must defeat a code-written cheat suite (delete the tests,
fake the exit code, hard-code the expected output), the do-nothing Run must
fail, and the starting container is scanned for ground-truth strings, a fix
commit in the history and hidden tests.

On the quoted hard-code cheat: generic code cannot know a Task's expected
output unless the starting container leaks it, and the leak is what the
ground-truth string check finds. So this module ships no Task-specific
hard-code cheat; the founder rules on whether that reading holds.

Two live facts shape the suite. The live world mounts /workspace noexec: a
planted executable is refused by the kernel (exit 126) when a grader calls it
by name, so in that world the hook files (conftest.py, sitecustomize.py) are
the part of fake_exit_code that can bite. The runners stay, because the suite
must not assume the mount options of every world.
"""

from __future__ import annotations

import base64
import hashlib
import io
import posixpath
import re
import shlex
import tarfile
from collections.abc import Callable
from dataclasses import dataclass

from kullback.container_grader import GradeError, grade
from kullback.container_world import WorldError

_DO_NOTHING_COMMANDS: tuple[str, ...] = ()

_DELETE_TESTS_COMMANDS: tuple[str, ...] = (
    "find /workspace \\( -name 'test*' -o -name '*_test*' -o -name 'tests'"
    " -o -name 'check*' -o -name 'expected*' \\) -exec rm -rf {} + 2>/dev/null || true",
    "rm -rf /workspace/tests /workspace/test* /workspace/check* /workspace/expected* 2>/dev/null || true",
)

_FAKE_RUNNERS_TOP = (
    "for n in pytest python python3 sh make node npm go cargo cmp diff test; do "
    "printf '#!/bin/sh\\nexit 0\\n' > /workspace/$n && chmod +x /workspace/$n; done 2>/dev/null || true"
)

_FAKE_RUNNERS_BIN = (
    "for n in pytest python python3 sh make node npm go cargo cmp diff test; do "
    "printf '#!/bin/sh\\nexit 0\\n' > /workspace/bin/$n && chmod +x /workspace/bin/$n; done 2>/dev/null || true"
)

_FAKE_CONFTEST = (
    "printf 'def pytest_sessionfinish(session, exitstatus):\\n    session.exitstatus = 0\\n'"
    " > /workspace/conftest.py 2>/dev/null || true"
)

_FAKE_SITECUSTOMIZE = (
    "printf 'import atexit as _a\\nimport os as _o\\n_a.register(_o._exit, 0)\\n'"
    " > /workspace/sitecustomize.py 2>/dev/null || true"
)

_FAKE_EXIT_COMMANDS: tuple[str, ...] = (
    "mkdir -p /workspace/bin 2>/dev/null || true",
    _FAKE_RUNNERS_TOP,
    _FAKE_RUNNERS_BIN,
    _FAKE_CONFTEST,
    _FAKE_SITECUSTOMIZE,
)

_AIM_DIR_LINK = "ln -sf ../grader /workspace/grader 2>/dev/null || true"

DEFAULT_OUTPUT_PATHS: tuple[str, ...] = (
    "answer.txt",
    "output.txt",
    "out.txt",
    "result.txt",
    "results.json",
    "solution.txt",
    "expected.txt",
)

DEFAULT_ANSWER_MEMBERS: tuple[str, ...] = ("expected.txt",)

_SHA_LIST_COMMAND = "find /workspace -type f -size +0 -exec sha256sum {} +"
_GIT_FIND_COMMAND = "find /workspace -maxdepth 4 -name .git"
_HAVE_GIT_COMMAND = "command -v git"
_PATTERNS_B64_PATH = "/workspace/.scan-patterns.b64"
_PATTERNS_PATH = "/workspace/.scan-patterns.txt"
_GREP_PATTERNS_COMMAND = "grep -rlFf /workspace/.scan-patterns.txt /workspace --exclude=.scan-patterns.txt"
_PATTERN_CHUNK_BYTES = 48000
_SHA_RE = re.compile(r"([0-9a-f]{64})  (.+)")


@dataclass(frozen=True)
class Cheat:
    name: str
    commands: tuple[str, ...]


@dataclass(frozen=True)
class CheatResult:
    name: str
    outcome: str
    detail: str = ""


@dataclass(frozen=True)
class CheatReport:
    results: tuple[CheatResult, ...]
    all_defeated: bool


@dataclass(frozen=True)
class ScanHit:
    kind: str
    path: str
    member: str = ""
    detail: str = ""


@dataclass(frozen=True)
class ScanCheck:
    name: str
    outcome: str
    hits: tuple[ScanHit, ...] = ()


@dataclass(frozen=True)
class ScanReport:
    hidden_tests: ScanCheck
    ground_truth: ScanCheck
    history: ScanCheck
    clean: bool


def _aim_commands(paths: tuple[str, ...], member: str) -> tuple[str, ...]:
    commands = []
    for path in paths:
        parent = "/workspace/" + posixpath.dirname(path)
        commands.append("mkdir -p " + shlex.quote(parent) + " 2>/dev/null || true")
        commands.append(
            "ln -sf "
            + shlex.quote("/workspace/grader/" + member)
            + " "
            + shlex.quote("/workspace/" + path)
            + " 2>/dev/null || true"
        )
    commands.append(_AIM_DIR_LINK)
    return tuple(commands)


def aim_cheats(output_paths=(), answer_members=()) -> tuple[Cheat, ...]:
    """Build one aim-at-grader cheat per answer member over the output paths."""
    paths = output_paths or DEFAULT_OUTPUT_PATHS
    members = answer_members or DEFAULT_ANSWER_MEMBERS
    plain = not output_paths and not answer_members
    built = []
    for member in members:
        if plain:
            name = "aim_at_grader"
        else:
            name = "aim_at_grader:" + member
        built.append(Cheat(name, _aim_commands(paths, member)))
    return tuple(built)


CHEATS: tuple[Cheat, ...] = (
    Cheat("do_nothing", _DO_NOTHING_COMMANDS),
    Cheat("delete_tests", _DELETE_TESTS_COMMANDS),
    Cheat("fake_exit_code", _FAKE_EXIT_COMMANDS),
) + aim_cheats((), ())


def _check_output_paths(paths: tuple) -> None:
    for path in paths:
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in path.split("/")
            or "\n" in path
            or "\x00" in path
        ):
            raise ValueError("bad output path: " + repr(path))


def _resolve_cheats(grader_tar: bytes, output_paths, answer_members, cheats: tuple[Cheat, ...]) -> tuple[Cheat, ...]:
    if isinstance(output_paths, str) or isinstance(answer_members, str):
        raise ValueError(
            "output_paths and answer_members must be tuples of str, got: " + repr((output_paths, answer_members))
        )
    paths = tuple(output_paths)
    names = tuple(answer_members)
    if cheats is not CHEATS and (paths or names):
        raise ValueError("cheats= with output_paths/answer_members: two sources of truth")
    _check_output_paths(paths)
    with tarfile.open(fileobj=io.BytesIO(grader_tar)) as tar:
        _require_regular_members(tar, names)
    if cheats is CHEATS and (paths or names):
        return CHEATS[:3] + aim_cheats(paths, names)
    return cheats


def _play_cheat(world, cheat: Cheat) -> str | None:
    for command in cheat.commands:
        receipt = world.step(command)
        if receipt.timed_out:
            return "cheat command timed out: " + command
        if receipt.truncated:
            return "cheat command truncated output: " + command
    return None


def _attempt_cheat(make_candidate, make_grader, cheat: Cheat, grader_tar: bytes, command: str, export_limit: int):
    world = None
    reason = None
    state_tar = b""
    try:
        world = make_candidate()
        world.reset()
        reason = _play_cheat(world, cheat)
        if reason is None:
            state_tar = world.export_workspace(export_limit)
    except WorldError as exc:
        reason = "candidate world raised " + type(exc).__name__ + ": " + str(exc)
    finally:
        if world is not None:
            world.close()
    if reason is not None:
        return CheatResult(cheat.name, "no_verdict", reason)
    try:
        receipt = grade(make_grader, state_tar=state_tar, grader_tar=grader_tar, command=command)
    except (GradeError, WorldError) as exc:
        return CheatResult(cheat.name, "no_verdict", "grade raised " + type(exc).__name__ + ": " + str(exc))
    if receipt.passed:
        return CheatResult(cheat.name, "succeeded", "")
    return CheatResult(cheat.name, "defeated", "")


def run_cheats(
    make_candidate: Callable,
    make_grader: Callable,
    *,
    grader_tar: bytes,
    command: str,
    export_limit_bytes: int = 16_777_216,
    cheats: tuple[Cheat, ...] = CHEATS,
    output_paths=(),
    answer_members=(),
) -> CheatReport:
    resolved = _resolve_cheats(grader_tar, output_paths, answer_members, cheats)
    results = []
    for cheat in resolved:
        results.append(_attempt_cheat(make_candidate, make_grader, cheat, grader_tar, command, export_limit_bytes))
    done = tuple(results)
    defeated = True
    for result in done:
        if result.outcome != "defeated":
            defeated = False
    return CheatReport(done, defeated)


def _grader_hashes(tar: tarfile.TarFile, min_file_bytes: int) -> dict[str, str]:
    table = {}
    for member in tar.getmembers():
        if member.isreg() and member.size >= min_file_bytes:
            digest = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            table[digest] = member.name
    return table


def _require_regular_members(tar: tarfile.TarFile, names: tuple) -> None:
    for name in names:
        try:
            member = tar.getmember(name)
        except KeyError:
            raise ValueError("unknown answer member: " + name) from None
        if not member.isreg():
            raise ValueError("answer member is not a file: " + name)


def _answer_patterns(tar: tarfile.TarFile, names: tuple, min_string_bytes: int, max_strings: int) -> list[str]:
    _require_regular_members(tar, names)
    found = []
    seen = set()
    for name in names:
        member = tar.getmember(name)
        text = tar.extractfile(member).read().decode("utf-8", errors="replace")
        for line in text.splitlines():
            stripped = line.strip()
            if len(stripped) >= min_string_bytes and stripped not in seen:
                seen.add(stripped)
                found.append(stripped)
    if len(found) > max_strings:
        raise ValueError("too many answer strings: " + str(len(found)) + " over limit " + str(max_strings))
    return found


def _check_hidden_tests(world, table: dict[str, str]) -> ScanCheck:
    try:
        receipt = world.step(_SHA_LIST_COMMAND)
    except WorldError:
        return ScanCheck("hidden_tests", "no_signal", ())
    if receipt.timed_out or receipt.truncated or receipt.exit_code != 0:
        return ScanCheck("hidden_tests", "no_signal", ())
    hits = []
    for line in receipt.stdout.decode("utf-8", errors="replace").splitlines():
        match = _SHA_RE.fullmatch(line)
        if match is None:
            return ScanCheck("hidden_tests", "no_signal", ())
        if match.group(1) in table:
            hits.append(ScanHit("hidden_test", match.group(2), table[match.group(1)]))
    if hits:
        return ScanCheck("hidden_tests", "hit", tuple(hits))
    return ScanCheck("hidden_tests", "clean", ())


def _require_feed_ok(receipt, stage: str) -> None:
    if receipt.timed_out or receipt.truncated or receipt.exit_code != 0:
        raise WorldError(stage + " failed: no signal")


def _feed_patterns(world, data: bytes) -> None:
    offset = 0
    index = 0
    while offset < len(data):
        chunk = base64.b64encode(data[offset : offset + _PATTERN_CHUNK_BYTES]).decode("ascii")
        _require_feed_ok(world.step("printf %s '" + chunk + "' >> " + _PATTERNS_B64_PATH), "patterns chunk " + str(index))
        offset += _PATTERN_CHUNK_BYTES
        index += 1
    _require_feed_ok(
        world.step("base64 -d " + _PATTERNS_B64_PATH + " > " + _PATTERNS_PATH + " && rm " + _PATTERNS_B64_PATH),
        "patterns unpack",
    )


def _check_ground_truth(world, patterns: list[str]) -> ScanCheck:
    if not patterns:
        return ScanCheck("ground_truth", "clean", ())
    try:
        _feed_patterns(world, "\n".join(patterns).encode("utf-8"))
        receipt = world.step(_GREP_PATTERNS_COMMAND)
    except WorldError:
        return ScanCheck("ground_truth", "no_signal", ())
    if receipt.timed_out or receipt.truncated:
        return ScanCheck("ground_truth", "no_signal", ())
    if receipt.exit_code == 1:
        return ScanCheck("ground_truth", "clean", ())
    if receipt.exit_code != 0:
        return ScanCheck("ground_truth", "no_signal", ())
    hits = []
    for line in receipt.stdout.decode("utf-8", errors="replace").splitlines():
        if not line or line == _PATTERNS_PATH:
            continue
        hits.append(ScanHit("ground_truth", line))
    if hits:
        return ScanCheck("ground_truth", "hit", tuple(hits))
    return ScanCheck("ground_truth", "clean", ())


def _count_unreachable(data: bytes) -> int:
    count = 0
    for line in data.decode("utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in ("unreachable", "dangling") and fields[1] == "commit":
            count += 1
    return count


def _check_one_repo(world, repo: str) -> tuple[bool, ScanHit | None]:
    quoted = shlex.quote(repo)
    try:
        all_receipt = world.step("git -C " + quoted + " rev-list --all --reflog | sort -u | wc -l")
        head_receipt = world.step("git -C " + quoted + " rev-list HEAD")
        fsck_receipt = world.step("git -C " + quoted + " fsck --unreachable --no-reflogs")
    except WorldError:
        return (False, None)
    for receipt in (all_receipt, head_receipt, fsck_receipt):
        if receipt.timed_out or receipt.truncated or receipt.exit_code != 0:
            return (False, None)
    try:
        total = int(all_receipt.stdout.decode("ascii").strip())
    except ValueError:
        return (False, None)
    head = len(head_receipt.stdout.splitlines())
    unreachable = _count_unreachable(fsck_receipt.stdout)
    if total > head or unreachable > 0:
        detail = "all=" + str(total) + " head=" + str(head) + " unreachable=" + str(unreachable)
        return (True, ScanHit("history", repo, "", detail))
    return (True, None)


def _check_history(world) -> ScanCheck:
    try:
        found = world.step(_GIT_FIND_COMMAND)
        if found.timed_out or found.truncated or found.exit_code != 0:
            return ScanCheck("history", "no_signal", ())
        repos = [line for line in found.stdout.decode("utf-8", errors="replace").splitlines() if line]
        if not repos:
            return ScanCheck("history", "clean", ())
        have = world.step(_HAVE_GIT_COMMAND)
        if have.timed_out or have.truncated or have.exit_code != 0:
            return ScanCheck("history", "no_signal", ())
        hits = []
        for repo in repos:
            ok, hit = _check_one_repo(world, repo)
            if not ok:
                return ScanCheck("history", "no_signal", ())
            if hit is not None:
                hits.append(hit)
        if hits:
            return ScanCheck("history", "hit", tuple(hits))
        return ScanCheck("history", "clean", ())
    except WorldError:
        return ScanCheck("history", "no_signal", ())


def scan_start(
    make_candidate: Callable,
    *,
    grader_tar: bytes,
    answer_members=(),
    min_file_bytes: int = 16,
    min_string_bytes: int = 12,
    max_strings: int = 500,
) -> ScanReport:
    """Scan the starting container for leaked grader material.

    The ground-truth check writes the answer lines into the scan world. That
    world is used for nothing else and is always closed; a caller must never
    pass the world a Candidate will use.
    """
    names = tuple(answer_members)
    with tarfile.open(fileobj=io.BytesIO(grader_tar)) as tar:
        table = _grader_hashes(tar, min_file_bytes)
        strings = _answer_patterns(tar, names, min_string_bytes, max_strings)
    patterns = list(strings) if names else None
    if patterns is None:
        idle_ground = ScanCheck("ground_truth", "not_run", ())
    else:
        idle_ground = ScanCheck("ground_truth", "no_signal", ())
    world = None
    try:
        world = make_candidate()
        world.reset()
        hidden = _check_hidden_tests(world, table)
        if patterns is None:
            ground = idle_ground
        else:
            ground = _check_ground_truth(world, patterns)
        history = _check_history(world)
    except WorldError:
        hidden = ScanCheck("hidden_tests", "no_signal", ())
        ground = idle_ground
        history = ScanCheck("history", "no_signal", ())
    finally:
        if world is not None:
            world.close()
    clean = True
    for check in (hidden, ground, history):
        if check.outcome == "not_run":
            continue
        if check.outcome != "clean":
            clean = False
    return ScanReport(hidden, ground, history, clean)
