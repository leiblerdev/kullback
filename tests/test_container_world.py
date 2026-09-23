import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from unittest import mock

import pytest

from kullback.container_world import (
    CaptureIncomplete,
    ContainerWorld,
    DockerUnavailable,
    ExecutorResult,
    ExecutorTimeout,
    ImagePinError,
    LaunchFailure,
    LimitError,
    OutputLimitExceeded,
    StateError,
    UnresolvedError,
    WorldError,
)

IMG = "registry.example.com:5000/tasks/shell@sha256:" + "ab" * 32
DOCK = "/usr/bin/docker"
GOOD = {"Config": {"Volumes": {}}}
GOOD_EMPTY = {"Config": {}}
IDA = "aa" * 32
IDB = "bb" * 32


def make_handler(calls, behavior):
    def handler(argv, timeout):
        assert isinstance(argv, list)
        assert all(isinstance(a, str) for a in argv)
        calls.append((list(argv), timeout))
        if len(argv) > 2 and argv[1] == "exec" and argv[-1] == ":":
            return ExecutorResult(0, b"", b"")
        return behavior(list(argv), timeout)

    return handler


def std_behavior(cid, ref, extra=None):
    def behavior(argv, timeout):
        if argv[1] == "create":
            return ExecutorResult(0, (cid + "\n").encode(), b"")
        if argv[1] == "inspect":
            w = ref[0]
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            return ExecutorResult(0, (cid + "\n").encode(), b"")
        if argv[1] == "rm":
            return ExecutorResult(0, (argv[-1] + "\n").encode(), b"")
        if extra is not None:
            return extra(argv, timeout)
        raise AssertionError("unexpected " + argv[1])

    return behavior


def image_ok_payload(image=IMG):
    digest = image.split("@sha256:")[1]
    ident = "sha256:" + digest
    return (ident + "|" + json.dumps([image]) + "|" + json.dumps({})).encode()


def r4_image_payload(ident=None, repo=None, vols=None):
    digest = IMG.split("@sha256:")[1]
    ident = ident if ident is not None else "sha256:" + digest
    repo = repo if repo is not None else [IMG]
    vols = vols if vols is not None else {}
    return (ident + "|" + json.dumps(repo) + "|" + json.dumps(vols)).encode()


def make_world(calls, behavior, **kw):
    kw.setdefault("image_metadata", dict(GOOD))
    kw.setdefault("docker", DOCK)
    orig = behavior
    def wrapped(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, image_ok_payload(), b"")
        return orig(argv, timeout)
    return ContainerWorld(IMG, executor=make_handler(calls, wrapped), **kw)


def make_raw_world(calls, behavior, **kw):
    kw.setdefault("image_metadata", dict(GOOD))
    kw.setdefault("docker", DOCK)
    return ContainerWorld(IMG, executor=make_handler(calls, behavior), **kw)


def _fresh_std(calls, extra=None, cid=IDA, **kw):
    ref = [None]
    world = make_world(calls, std_behavior(cid, ref, extra), **kw)
    ref[0] = world
    world.reset()
    return world


def _scripted(items):
    pending = list(items)

    def behavior(argv, timeout):
        item = pending.pop(0)
        if callable(item):
            return item(argv)
        if isinstance(item, BaseException):
            raise item
        return item

    return behavior


def _killpg_recorder(calls, seen):
    real = os.killpg

    def _recording(pid, sig):
        calls.append((pid, sig))
        if pid in {proc.pid for proc in seen}:
            return real(pid, sig)
        return None

    return _recording


@contextmanager
def _owned_children():
    seen = []
    real = subprocess.Popen

    def _recording(*args, **kwargs):
        proc = real(*args, **kwargs)
        seen.append(proc)
        return proc

    with mock.patch.object(subprocess, "Popen", _recording):
        try:
            yield seen
        finally:
            for proc in seen:
                try:
                    if proc.poll() is None:
                        proc.kill()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                for stream in (proc.stdout, proc.stderr):
                    try:
                        if stream is not None:
                            stream.close()
                    except (OSError, ValueError):
                        pass


def _owner_behavior(cid, ref, rms, inspect_answer=None):
    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] in ("create", "start"):
            return ExecutorResult(0, (cid + "\n").encode(), b"")
        if argv[1] == "inspect":
            if inspect_answer is not None:
                return ExecutorResult(0, inspect_answer(argv, w).encode(), b"")
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "rm":
            rms.append(argv)
            assert argv[-1] == cid
            return ExecutorResult(0, (cid + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])

    return behavior


def _r4_recovery_behavior(calls, ref, n, rms, owned):
    def behavior(argv, timeout):
        w = ref[0]
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, r4_image_payload(), b"")
        if argv[1] == "create":
            n[0] += 1
            if n[0] == 1:
                raise ExecutorTimeout(b"", b"")
            return ExecutorResult(0, (IDB + "\n").encode(), b"")
        if argv[1] == "inspect":
            if argv[-1] == w._name:
                return ExecutorResult(0, (owned[0] + "|" + w._nonce + "|container_world").encode(), b"")
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            return ExecutorResult(0, (IDB + "\n").encode(), b"")
        if argv[1] == "rm":
            rms.append(list(argv))
            assert argv[-1] == owned[0]
            return ExecutorResult(0, (owned[0] + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])

    return behavior


def _probe_world_behavior(ref, calls, rms, probe_answers):
    n = [0]

    def fake(argv, timeout):
        assert isinstance(argv, list)
        calls.append((list(argv), timeout))
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, image_ok_payload(), b"")
        w = ref[0]
        if argv[1] == "create":
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "exec":
            n[0] += 1
            item = probe_answers[min(n[0] - 1, len(probe_answers) - 1)]
            if isinstance(item, BaseException):
                raise item
            return item
        if argv[1] == "rm":
            rms.append(list(argv))
            return ExecutorResult(0, (argv[-1] + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])

    return fake


@pytest.mark.parametrize(
    ("image", "pinned"),
    [
        ("ubuntu:latest", False),
        ("ubuntu", False),
        ("python:3.12-slim", False),
        ("", False),
        ("-e evil@sha256:" + "ab" * 32, False),
        ("http://evil.com/img@sha256:" + "ab" * 32, False),
        ("user@host/img@sha256:" + "ab" * 32, False),
        ("repo/img@sha256:xyz", False),
        ("repo/img@sha256:" + "ab" * 31, False),
        ("repo:latest@sha256:" + "ab" * 32, False),
        ("repo/img@sha256:" + "zz" * 32, False),
        ("repo/img @sha256:" + "ab" * 32, False),
        ("host:5000/repo:latest@sha256:" + "ab" * 32, False),
        ("repo:latest-build@sha256:" + "ab" * 32, True),
        ("host:5000/repo@sha256:" + "ab" * 32, True),
        ("host:5000/repo:v1@sha256:" + "ab" * 32, True),
    ],
    ids=[
        "latest-tag",
        "bare-name",
        "minor-tag",
        "empty",
        "option-prefix",
        "url-image",
        "userinfo-image",
        "nonhex-digest",
        "short-digest",
        "latest-with-pin",
        "nonhex-bytes",
        "whitespace-name",
        "port-latest-tag",
        "accepts-tag-with-latest-prefix",
        "accepts-registry-port-no-tag",
        "accepts-registry-port-with-tag",
    ],
)
def test_only_digest_pinned_non_latest_images_are_accepted(image, pinned):
    no_call = lambda a, t: (_ for _ in ()).throw(AssertionError("no call"))  # noqa: E731
    if pinned:
        ContainerWorld(image, docker=DOCK, image_metadata=dict(GOOD), executor=no_call)
    else:
        with pytest.raises(ImagePinError):
            ContainerWorld(image, docker=DOCK, image_metadata=dict(GOOD))
    ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), executor=no_call)


@pytest.mark.parametrize(
    ("where", "value"),
    [
        ("init", {"timeout_s": True}),
        ("init", {"timeout_s": 0}),
        ("init", {"timeout_s": float("inf")}),
        ("init", {"output_limit_bytes": False}),
        ("init", {"output_limit_bytes": 0}),
        ("init", {"memory_mb": True}),
        ("init", {"cpus": 0}),
        ("init", {"cpus": True}),
        ("init", {"pids_limit": -1}),
        ("init", {"pids_limit": 1.5}),
        ("export", 0),
        ("export", -5),
        ("export", True),
        ("export", "1024"),
        ("export", None),
        ("export", 16777217),
    ],
    ids=[
        "timeout-bool",
        "timeout-zero",
        "timeout-infinite",
        "limit-bool",
        "limit-zero",
        "memory-bool",
        "cpus-zero",
        "cpus-bool",
        "pids-negative",
        "pids-float",
        "export-zero",
        "export-negative",
        "export-bool",
        "export-str",
        "export-none",
        "export-above-max",
    ],
)
def test_rejects_bad_limits(where, value):
    if where == "init":
        with pytest.raises(LimitError):
            ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), **value)
        return
    calls = []
    world = _fresh_std(calls)
    n_exec_before = len([c for c in calls if c[0][1] == "exec"])
    with pytest.raises(LimitError):
        world.export_workspace(value)
    assert world._active is True
    assert len([c for c in calls if c[0][1] == "exec"]) == n_exec_before


def test_create_argv_is_exactly_the_isolation_flags_with_no_mounts_env_network_privilege():
    calls = []
    world = _fresh_std(calls)
    create = next(argv for argv, _ in calls if len(argv) > 1 and argv[1] == "create")
    assert create == [
        DOCK, "create",
        "--network", "none",
        "--pull", "never",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--memory", "512m",
        "--cpus", "1.0",
        "--pids-limit", "128",
        "--tmpfs", "/workspace:rw,size=64m,mode=1777",
        "--entrypoint", "",
        "--workdir", "/workspace",
        "--no-healthcheck",
        "--label", "kullback.world=" + world._nonce,
        "--label", "kullback.owner=container_world",
        "--name", world._name,
        IMG, "sleep", "infinity",
    ]
    for forbidden in ["-v", "--volume", "--mount", "-e", "--env", "--env-file", "--publish", "-p", "--privileged", "--device", "--gpus", "docker.sock"]:
        assert forbidden not in create
    assert "--network" in create
    assert create[create.index("--network") + 1] == "none"
    assert "host" not in create
    assert world._container_id == IDA


def test_command_passed_as_single_argv_element():
    calls = []
    evil = 'x"; touch /pwn; echo pwned # $(rm -rf /)'
    seen = {}

    def extra(argv, timeout):
        if argv[1] == "exec":
            seen["argv"] = argv
            return ExecutorResult(0, b"ok\n", b"")
        raise AssertionError("unexpected " + argv[1])

    world = _fresh_std(calls, extra)
    receipt = world.step(evil)
    assert receipt.stdout == b"ok\n"
    argv = seen["argv"]
    assert len(argv) == 6
    assert argv[:5] == [DOCK, "exec", IDA, "sh", "-c"]
    assert argv[5] == evil


def test_reset_starts_a_fresh_container_and_removes_the_prior_one():
    calls = []
    ref = [None]
    fresh = [IDA, IDB]

    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] == "create":
            return ExecutorResult(0, (fresh[0] + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            return ExecutorResult(0, (fresh[0] + "\n").encode(), b"")
        if argv[1] == "rm":
            assert argv[-1] == IDA
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world(calls, behavior)
    ref[0] = world
    cid = world.reset()
    assert cid == IDA
    assert [argv[1] for argv, _ in calls] == ["image", "create", "inspect", "start", "exec"]
    assert world._active is True
    fresh[0] = IDB
    assert world.reset() == IDB
    verbs = [argv[1] for argv, _ in calls]
    assert verbs == ["image", "create", "inspect", "start", "exec", "inspect", "rm", "image", "create", "inspect", "start", "exec"]
    assert world._container_id == IDB
    assert world._active is True


@pytest.mark.parametrize(
    ("raw_out", "raw_err"),
    [
        (b"  spaced\tout\nline2\n", "err-ünicode-é\n".encode()),
        (b"\xff\xfe\x00raw\x01\x00end", b"\x00\xff"),
        (b"\xff\xfe\x00raw\x01", b""),
    ],
    ids=["whitespace-and-unicode", "invalid-utf8-and-nul", "invalid-utf8-empty-stderr"],
)
def test_step_preserves_raw_bytes(raw_out, raw_err):
    def extra(argv, timeout, _out=raw_out, _err=raw_err):
        if argv[1] == "exec":
            return ExecutorResult(0, _out, _err)
        raise AssertionError("unexpected " + argv[1])

    world = _fresh_std([], extra)
    receipt = world.step("printf x")
    assert receipt.stdout == raw_out
    assert receipt.stderr == raw_err
    assert receipt.exit_code == 0
    assert receipt.timed_out is False
    assert receipt.truncated is False
    assert (b"\x00" not in raw_out) or (b"\x00" in receipt.stdout)


def test_nonzero_exit_keeps_the_world_but_timeout_kills_and_requires_reset():
    calls = []
    answers = [ExecutorResult(3, b"", b"boom\n"), ExecutorTimeout(b"partial-out", b"partial-err")]

    def extra(argv, timeout):
        if argv[1] == "exec":
            item = answers.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        raise AssertionError("unexpected " + argv[1])

    world = _fresh_std(calls, extra)
    receipt = world.step("exit 3")
    assert receipt.exit_code == 3
    assert receipt.timed_out is False
    assert receipt.truncated is False
    assert receipt.stderr == b"boom\n"
    assert world._active is True
    assert "rm" not in [argv[1] for argv, _ in calls]

    receipt = world.step("sleep 999")
    assert receipt.timed_out is True
    assert receipt.truncated is False
    assert receipt.exit_code == -1
    assert receipt.stdout == b"partial-out"
    verbs = [argv[1] for argv, _ in calls]
    assert "rm" in verbs
    with pytest.raises(StateError):
        world.step("echo hi")
    world.reset()
    assert world._active is True


@pytest.mark.parametrize(
    ("answer", "kw", "cap"),
    [
        (ExecutorResult(0, b"x" * 50, b""), {"output_limit_bytes": 10}, 10),
        (OutputLimitExceeded(b"0123456789abcdef", b""), {}, 65536),
    ],
    ids=["returned-past-cap", "raised-by-executor"],
)
def test_output_overflow_kills_and_marks_truncated(answer, kw, cap):
    calls = []

    def extra(argv, timeout):
        if argv[1] == "exec":
            if isinstance(answer, BaseException):
                raise answer
            return answer
        raise AssertionError("unexpected " + argv[1])

    world = _fresh_std(calls, extra, **kw)
    receipt = world.step("yes")
    assert receipt.truncated is True
    assert receipt.timed_out is False
    assert receipt.exit_code == -1
    assert len(receipt.stdout) + len(receipt.stderr) <= cap
    assert [argv[1] for argv, _ in calls].count("rm") == 1
    with pytest.raises(StateError):
        world.step("echo hi")


def test_close_idempotent():
    calls = []
    world = _fresh_std(calls)
    assert world.close() is True
    assert world.close() is False
    assert [argv[1] for argv, _ in calls].count("rm") == 1


def test_close_removes_only_a_container_this_world_owns():
    foreign_answers = {
        "wrong-nonce": lambda argv, w: argv[-1] + "|wrong-nonce|container_world",
        "wrong-id": lambda argv, w: IDB + "|" + w._nonce + "|container_world",
        "wrong-owner": lambda argv, w: argv[-1] + "|" + w._nonce + "|other-owner",
    }
    for name, answer in foreign_answers.items():
        rms = []
        ref = [None]
        world = make_world([], _owner_behavior(IDA, ref, rms, answer))
        ref[0] = world
        with pytest.raises(UnresolvedError):
            world.reset()
        assert rms == [], name
        assert world._unresolved is True, name

    rms = []
    ref_a = [None]
    ref_b = [None]
    world_a = make_world([], _owner_behavior(IDA, ref_a, rms))
    ref_a[0] = world_a
    world_b = make_world([], _owner_behavior(IDB, ref_b, rms))
    ref_b[0] = world_b
    world_a.reset()
    world_b.reset()
    assert world_a._nonce != world_b._nonce
    world_a.close()
    assert rms == [[DOCK, "rm", "-f", IDA]]
    assert world_b._container_id == IDB

    ref = [None]
    owned = ["ee" * 32]
    n = [0]
    rms = []
    world = make_raw_world([], _r4_recovery_behavior([], ref, n, rms, owned))
    ref[0] = world
    with pytest.raises(UnresolvedError):
        world.reset()
    assert world.close() is True
    assert rms == [[DOCK, "rm", "-f", owned[0]]]
    assert world._unresolved is False
    assert world.reset() == IDB

    rms = []

    def collision(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, r4_image_payload(), b"")
        if argv[1] == "create":
            raise ExecutorTimeout(b"", b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, ("ff" * 32 + "|foreign-nonce|container_world").encode(), b"")
        if argv[1] == "rm":
            rms.append(argv)
            return ExecutorResult(0, b"ff\n", b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_raw_world([], collision)
    with pytest.raises(UnresolvedError):
        world.reset()
    with pytest.raises(UnresolvedError):
        world.close()
    assert rms == []
    assert world._unresolved is True


def test_missing_image_daemon_or_create_id_fails_closed():
    def no_such_image(argv, timeout):
        if argv[1] == "create":
            return ExecutorResult(1, b"", b"Error: No such image\n")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], no_such_image)
    with pytest.raises(DockerUnavailable):
        world.reset()
    assert world._container_id is None
    assert world._active is False
    assert world._unresolved is False

    world = make_world([], _scripted([LaunchFailure("docker binary gone")]))
    with pytest.raises(DockerUnavailable):
        world.reset()
    assert world._container_id is None

    world = make_world([], _scripted([ExecutorResult(0, b"\n", b"")]))
    with pytest.raises(UnresolvedError):
        world.reset()
    assert world._container_id is None
    with pytest.raises(UnresolvedError):
        world.reset()

    for mode in ["nonzero-missing", "nonzero-ambiguous", "timeout"]:
        calls = []
        def unreadable(argv, timeout, _m=mode):
            if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
                if _m == "nonzero-missing":
                    return ExecutorResult(1, b"", b"Error: No such image")
                if _m == "nonzero-ambiguous":
                    return ExecutorResult(1, b"", b"permission denied")
                raise ExecutorTimeout(b"", b"")
            if argv[1] == "create":
                raise AssertionError("create must not run when unreadable")
            raise AssertionError("unexpected " + argv[1])
        world = make_raw_world(calls, unreadable)
        with pytest.raises(WorldError):
            world.reset()
        assert [c for c in calls if c[0][1] == "create"] == []


def test_step_and_export_before_reset_raise():
    world = make_world([], lambda a, t: ExecutorResult(0, b"", b""))
    with pytest.raises(StateError):
        world.step("echo hi")
    with pytest.raises(StateError):
        world.export_workspace(1024)


def test_the_real_runner_caps_combined_output():
    from kullback.container_world import _bounded_run

    over = "import sys; sys.stdout.buffer.write(b'o' * 600); sys.stdout.flush(); sys.stderr.buffer.write(b'e' * 600); sys.stderr.flush()"
    exact = "import sys; sys.stdout.buffer.write(b'o' * 512); sys.stderr.buffer.write(b'e' * 512)"
    plus_one = "import sys; sys.stdout.buffer.write(b'o' * 512); sys.stderr.buffer.write(b'e' * 513)"
    for script in (over, plus_one):
        with pytest.raises(OutputLimitExceeded) as caught:
            _bounded_run([sys.executable, "-c", script], 20.0, 1024)
        assert len(caught.value.stdout) + len(caught.value.stderr) == 1024
    res = _bounded_run([sys.executable, "-c", exact], 20.0, 1024)
    assert res.returncode == 0
    assert len(res.stdout) + len(res.stderr) == 1024

    def extra(argv, timeout):
        if argv[1] == "exec":
            raise ExecutorTimeout(b"o" * 2000, b"e" * 2000)
        raise AssertionError("unexpected " + argv[1])

    world = _fresh_std([], extra, output_limit_bytes=1024)
    receipt = world.step("sleep 1")
    assert receipt.timed_out is True
    assert len(receipt.stdout) + len(receipt.stderr) == 1024


def test_the_real_runner_bounds_a_hung_command():
    from kullback.container_world import _bounded_run

    calls = []
    script = "import sys, time; sys.stdout.buffer.write(b'x' * 100); sys.stdout.flush(); time.sleep(30)"
    begin = time.monotonic()
    with _owned_children() as seen:
        with mock.patch.object(os, "killpg", _killpg_recorder(calls, seen)):
            with pytest.raises(ExecutorTimeout) as caught:
                _bounded_run([sys.executable, "-c", script], 1.0, 65536)
    elapsed = time.monotonic() - begin
    assert elapsed < 12
    assert caught.value.stdout == b"x" * 100
    assert len(caught.value.stdout) + len(caught.value.stderr) <= 65536
    assert len(calls) == 1
    assert len(seen) == 1
    assert seen[0].poll() is not None

    parent = "import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], stdout=sys.stdout, stderr=sys.stderr); sys.stdout.flush()"
    begin = time.monotonic()
    with pytest.raises(CaptureIncomplete) as caught:
        _bounded_run([sys.executable, "-c", parent], 30.0, 65536)
    elapsed = time.monotonic() - begin
    assert elapsed < 12
    assert len(caught.value.stdout) + len(caught.value.stderr) <= 65536


def test_an_image_declaring_volumes_is_refused():
    with pytest.raises(ImagePinError):
        ContainerWorld(IMG, docker=DOCK, image_metadata={"Config": {"Volumes": {"/data": {}}}})
    with pytest.raises(ImagePinError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=None)

    calls = []
    def declares_volumes(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, r4_image_payload(vols={"/data": {}}), b"")
        if argv[1] == "create":
            raise AssertionError("create must not run when volumes declared")
        raise AssertionError("unexpected " + argv[1])
    world = make_raw_world(calls, declares_volumes)
    with pytest.raises(ImagePinError):
        world.reset()
    assert [c for c in calls if c[0][1] == "create"] == []

    ref = [None]
    def no_volumes_key(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, image_ok_payload(), b"")
        return std_behavior(IDA, ref)(argv, timeout)
    world = ContainerWorld(IMG, docker=DOCK, executor=make_handler([], no_volumes_key), image_metadata=dict(GOOD_EMPTY))
    ref[0] = world
    assert world.reset() == IDA


def test_r4_step_incomplete_maps_to_truncated_not_success():
    ref = [None]
    def behavior(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, r4_image_payload(), b"")
        w = ref[0]
        if argv[1] == "create":
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "exec":
            raise CaptureIncomplete(b"part", b"")
        if argv[1] == "rm":
            return ExecutorResult(0, (argv[-1] + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])
    world = make_raw_world([], behavior)
    ref[0] = world
    world.reset()
    receipt = world.step("echo hi")
    assert receipt.truncated is True
    assert receipt.timed_out is False
    assert receipt.exit_code == -1
    assert receipt.stdout == b"part"


def test_r4_image_identity_mismatch_blocks_creation():
    for payload in [
        r4_image_payload(repo=["other@sha256:" + "bb" * 32]),
        r4_image_payload(ident="not-an-id"),
        b"badshape",
        b"\x00bad",
    ]:
        calls = []
        def behavior(argv, timeout, _p=payload):
            if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
                return ExecutorResult(0, _p, b"")
            if argv[1] == "create":
                raise AssertionError("create must not run on mismatch")
            raise AssertionError("unexpected " + argv[1])
        world = make_raw_world(calls, behavior)
        with pytest.raises(ImagePinError):
            world.reset()
        assert [c for c in calls if c[0][1] == "create"] == []


def test_an_interrupt_reaps_the_child_the_runner_started(tmp_path):
    import fcntl
    import os as _os
    import signal as _signal
    import threading as _threading

    from kullback import container_world as _cw
    from kullback.container_world import _bounded_run
    lock = tmp_path / "hold.lock"
    lock.write_bytes(b"")
    child = "import fcntl, time, sys; f = open(sys.argv[1], 'rb'); fcntl.flock(f.fileno(), fcntl.LOCK_EX); time.sleep(30)"
    timer = _threading.Timer(0.5, lambda: _os.kill(_os.getpid(), _signal.SIGINT))
    timer.start()
    begin = time.monotonic()
    try:
        _bounded_run([sys.executable, "-c", child, str(lock)], 30.0, 65536)
        raise AssertionError("must not return success after interruption")
    except KeyboardInterrupt:
        pass
    finally:
        timer.cancel()
    elapsed = time.monotonic() - begin
    assert elapsed < 12
    with open(str(lock), "rb") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    real_wait_reap = _cw._wait_reap
    state = {"n": 0}

    def _flaky_wait_reap(proc, wait_s):
        state["n"] += 1
        if state["n"] == 1:
            raise KeyboardInterrupt()
        return real_wait_reap(proc, wait_s)

    with _owned_children() as seen:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
        with mock.patch.object(_cw, "_wait_reap", _flaky_wait_reap):
            with pytest.raises(KeyboardInterrupt):
                _cw._reap_owned(proc)
        assert state["n"] >= 2
        assert proc.returncode is not None
        assert proc.poll() is not None
        assert seen == [proc]


def test_r9_shell_probe_refuses_image_without_sh():
    calls = []
    ref = [None]
    rms = []
    world = ContainerWorld(
        IMG,
        docker=DOCK,
        image_metadata=dict(GOOD),
        executor=_probe_world_behavior(
            ref, calls, rms, [ExecutorResult(127, b"", b"sh: not found\n"), ExecutorResult(0, b"", b"")]
        ),
    )
    ref[0] = world
    with pytest.raises(ImagePinError, match="must provide sh"):
        world.reset()
    assert rms == [[DOCK, "rm", "-f", IDA]]
    assert world._container_id is None
    assert world._active is False
    assert world.reset() == IDA


def test_export_sends_direct_tar_argv_and_returns_raw_bytes_under_its_own_limit():
    # Direct argv, no sh -c: the tar arguments are fixed strings with no
    # shell metacharacters, so a shell would only add an injection surface.
    raw = b"\xff\xfe\x00tar-bytes\x01\x00end"
    seen = {}

    def extra(argv, timeout):
        if argv[1] == "exec":
            if argv[-1] == ":":
                return ExecutorResult(0, b"", b"")
            seen.setdefault("argv", argv)
            return ExecutorResult(0, raw, b"")
        raise AssertionError("unexpected " + argv[1])

    world = _fresh_std([], extra, output_limit_bytes=10)
    assert len(raw) > 10
    assert world.export_workspace(1024) == raw
    assert seen["argv"] == [DOCK, "exec", IDA, "tar", "-cf", "-", "-C", "/workspace", "."]
    receipt = world.step("yes")
    assert receipt.truncated is True
    assert len(receipt.stdout) + len(receipt.stderr) <= 10
