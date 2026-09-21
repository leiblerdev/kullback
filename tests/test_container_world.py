import json
import os
import sys
import time

import pytest

from kullback.container_world import (
    CaptureIncomplete,
    CleanupError,
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
IDC = "cc" * 32
IDD = "dd" * 32
R2_A = "aa" * 32
R2_META = {"Config": {"Volumes": {}}}


def make_handler(calls, behavior):
    def handler(argv, timeout):
        assert isinstance(argv, list)
        assert all(isinstance(a, str) for a in argv)
        calls.append((list(argv), timeout))
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


def test_rejects_tag_only_image():
    for bad in ["ubuntu:latest", "ubuntu", "python:3.12-slim", ""]:
        with pytest.raises(ImagePinError):
            ContainerWorld(bad, docker=DOCK, image_metadata=dict(GOOD))


def test_rejects_latest_option_url_userinfo():
    bads = [
        "-e evil@sha256:" + "ab" * 32,
        "http://evil.com/img@sha256:" + "ab" * 32,
        "user@host/img@sha256:" + "ab" * 32,
        "repo/img@sha256:xyz",
        "repo/img@sha256:" + "ab" * 31,
        "repo:latest@sha256:" + "ab" * 32,
        "repo/img@sha256:" + "zz" * 32,
        "repo/img @sha256:" + "ab" * 32,
    ]
    for bad in bads:
        with pytest.raises(ImagePinError):
            ContainerWorld(bad, docker=DOCK, image_metadata=dict(GOOD))
    ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), executor=lambda a, t: (_ for _ in ()).throw(AssertionError("no call")))


def test_rejects_bad_limits():
    with pytest.raises(LimitError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), timeout_s=True)
    with pytest.raises(LimitError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), timeout_s=0)
    with pytest.raises(LimitError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), timeout_s=float("inf"))
    with pytest.raises(LimitError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), output_limit_bytes=False)
    with pytest.raises(LimitError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), output_limit_bytes=0)
    with pytest.raises(LimitError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), memory_mb=True)
    with pytest.raises(LimitError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), cpus=0)
    with pytest.raises(LimitError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), cpus=True)
    with pytest.raises(LimitError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), pids_limit=-1)
    with pytest.raises(LimitError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=dict(GOOD), pids_limit=1.5)


def test_create_uses_exact_isolation_flags():
    calls = []
    ref = [None]
    world = make_world(calls, std_behavior(IDA, ref))
    ref[0] = world
    world.reset()
    create = [argv for argv, _ in calls if len(argv) > 1 and argv[1] == "create"][0]
    assert (DOCK, "create") == (create[0], create[1])
    for flag, value in [
        ("--network", "none"),
        ("--pull", "never"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--user", "65534:65534"),
        ("--pids-limit", "128"),
    ]:
        assert flag in create
        assert create[create.index(flag) + 1] == value
    assert "--read-only" in create
    assert "--tmpfs" in create
    assert create[create.index("--tmpfs") + 1].startswith("/workspace:")
    assert "kullback.world=" + world._nonce in create
    assert "kullback.owner=container_world" in create
    assert world._name in create
    assert create[-3] == IMG
    assert "--entrypoint" in create
    assert create[create.index("--entrypoint") + 1] == ""
    assert "--workdir" in create
    assert create[create.index("--workdir") + 1] == "/workspace"
    assert "--no-healthcheck" in create


def test_create_has_no_mounts_env_network_privilege():
    calls = []
    ref = [None]
    world = make_world(calls, std_behavior(IDA, ref))
    ref[0] = world
    world.reset()
    create = [argv for argv, _ in calls if len(argv) > 1 and argv[1] == "create"][0]
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
    ref = [None]

    def extra(argv, timeout):
        if argv[1] == "exec":
            seen["argv"] = argv
            return ExecutorResult(0, b"ok\n", b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world(calls, std_behavior(IDA, ref, extra))
    ref[0] = world
    world.reset()
    receipt = world.step(evil)
    assert receipt.stdout == b"ok\n"
    argv = seen["argv"]
    assert len(argv) == 6
    assert argv[:5] == [DOCK, "exec", IDA, "sh", "-c"]
    assert argv[5] == evil


def test_reset_returns_fresh_id_and_starts():
    calls = []
    ref = [None]
    world = make_world(calls, std_behavior(IDA, ref))
    ref[0] = world
    cid = world.reset()
    assert cid == IDA
    verbs = [argv[1] for argv, _ in calls]
    assert verbs == ["image", "create", "inspect", "start"]
    assert world._active is True


def test_reset_removes_prior_container():
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
    world.reset()
    fresh[0] = IDB
    world.reset()
    verbs = [argv[1] for argv, _ in calls]
    assert verbs == ["image", "create", "inspect", "start", "inspect", "rm", "image", "create", "inspect", "start"]
    assert world._container_id == IDB


def test_step_preserves_raw_output():
    raw_out = b"  spaced\tout\nline2\n"
    raw_err = "err-ünicode-é\n".encode()
    ref = [None]

    def extra(argv, timeout):
        if argv[1] == "exec":
            return ExecutorResult(0, raw_out, raw_err)
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], std_behavior(IDA, ref, extra))
    ref[0] = world
    world.reset()
    receipt = world.step("printf x")
    assert receipt.stdout == raw_out
    assert receipt.stderr == raw_err
    assert receipt.exit_code == 0
    assert receipt.timed_out is False
    assert receipt.truncated is False


def test_step_preserves_invalid_utf8_and_nul():
    raw = b"\xff\xfe\x00raw\x01\x00end"
    ref = [None]

    def extra(argv, timeout):
        if argv[1] == "exec":
            return ExecutorResult(0, raw, b"\x00\xff")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], std_behavior(IDA, ref, extra))
    ref[0] = world
    world.reset()
    receipt = world.step("printf x")
    assert receipt.stdout == raw
    assert receipt.stderr == b"\x00\xff"
    assert b"\x00" in receipt.stdout


def test_nonzero_exit_is_not_timeout():
    ref = [None]

    def extra(argv, timeout):
        if argv[1] == "exec":
            return ExecutorResult(3, b"", b"boom\n")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], std_behavior(IDA, ref, extra))
    ref[0] = world
    world.reset()
    receipt = world.step("exit 3")
    assert receipt.exit_code == 3
    assert receipt.timed_out is False
    assert receipt.truncated is False
    assert receipt.stderr == b"boom\n"
    assert world._active is True


def test_step_timeout_kills_and_requires_reset():
    calls = []
    ref = [None]

    def extra(argv, timeout):
        if argv[1] == "exec":
            raise ExecutorTimeout(b"partial-out", b"partial-err")
        raise AssertionError("unexpected " + argv[1])

    world = make_world(calls, std_behavior(IDA, ref, extra))
    ref[0] = world
    world.reset()
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


def test_output_overflow_kills_and_marks_truncated():
    calls = []
    ref = [None]

    def extra(argv, timeout):
        if argv[1] == "exec":
            return ExecutorResult(0, b"x" * 50, b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world(calls, std_behavior(IDA, ref, extra), output_limit_bytes=10)
    ref[0] = world
    world.reset()
    receipt = world.step("yes")
    assert receipt.truncated is True
    assert receipt.timed_out is False
    assert receipt.exit_code == -1
    assert len(receipt.stdout) + len(receipt.stderr) <= 10
    assert [argv[1] for argv, _ in calls].count("rm") == 1
    with pytest.raises(StateError):
        world.step("echo hi")


def test_executor_overflow_exception_path():
    ref = [None]

    def extra(argv, timeout):
        if argv[1] == "exec":
            raise OutputLimitExceeded(b"0123456789abcdef", b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], std_behavior(IDA, ref, extra))
    ref[0] = world
    world.reset()
    receipt = world.step("yes")
    assert receipt.truncated is True
    assert receipt.exit_code == -1
    assert len(receipt.stdout) + len(receipt.stderr) <= 65536


def test_close_idempotent():
    calls = []
    ref = [None]
    world = make_world(calls, std_behavior(IDA, ref))
    ref[0] = world
    world.reset()
    assert world.close() is True
    assert world.close() is False
    assert [argv[1] for argv, _ in calls].count("rm") == 1


def test_close_refuses_mismatched_label():
    rms = []
    ref = [None]

    def behavior(argv, timeout):
        if argv[1] == "create":
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (argv[-1] + "|wrong-nonce|container_world").encode(), b"")
        if argv[1] == "start":
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "rm":
            rms.append(argv)
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    ref[0] = world
    with pytest.raises(UnresolvedError):
        world.reset()
    assert rms == []
    assert world._unresolved is True


def test_close_refuses_mismatched_id():
    rms = []
    ref = [None]

    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] == "create":
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (IDB + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "rm":
            rms.append(argv)
            return ExecutorResult(0, (IDB + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    ref[0] = world
    with pytest.raises(UnresolvedError):
        world.reset()
    assert rms == []


def test_close_refuses_wrong_owner_label():
    ref = [None]

    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] in ("create", "start"):
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|other-owner").encode(), b"")
        if argv[1] == "rm":
            raise AssertionError("rm must not run on owner mismatch")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    ref[0] = world
    with pytest.raises(UnresolvedError):
        world.reset()
    assert world._container_id == IDA or world._unresolved is True


def test_two_instances_cannot_clean_each_other():
    rms = []
    ref_a = [None]
    ref_b = [None]

    def behavior_for(cid, world_ref):
        def behavior(argv, timeout):
            w = world_ref[0]
            if argv[1] == "create":
                return ExecutorResult(0, (cid + "\n").encode(), b"")
            if argv[1] == "inspect":
                return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
            if argv[1] == "start":
                return ExecutorResult(0, (cid + "\n").encode(), b"")
            if argv[1] == "rm":
                rms.append(argv)
                assert argv[-1] == cid
                return ExecutorResult(0, (cid + "\n").encode(), b"")
            raise AssertionError("unexpected " + argv[1])

        return behavior

    world_a = make_world([], behavior_for(IDA, ref_a))
    ref_a[0] = world_a
    world_b = make_world([], behavior_for(IDB, ref_b))
    ref_b[0] = world_b
    world_a.reset()
    world_b.reset()
    assert world_a._nonce != world_b._nonce
    world_a.close()
    assert rms == [[DOCK, "rm", "-f", IDA]]
    assert world_b._container_id == IDB


def test_missing_image_fails_closed():
    def behavior(argv, timeout):
        if argv[1] == "create":
            return ExecutorResult(1, b"", b"Error: No such image\n")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    with pytest.raises(DockerUnavailable):
        world.reset()
    assert world._container_id is None
    assert world._active is False
    assert world._unresolved is False


def test_daemon_missing_fails_closed():
    def behavior(argv, timeout):
        raise LaunchFailure("docker binary gone")

    world = make_world([], behavior)
    with pytest.raises(DockerUnavailable):
        world.reset()
    assert world._container_id is None


def test_malformed_create_output_fails_closed():
    def behavior(argv, timeout):
        if argv[1] == "create":
            return ExecutorResult(0, b"\n", b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    with pytest.raises(UnresolvedError):
        world.reset()
    assert world._container_id is None
    with pytest.raises(UnresolvedError):
        world.reset()


def test_step_without_reset_raises():
    world = make_world([], lambda a, t: ExecutorResult(0, b"", b""))
    with pytest.raises(StateError):
        world.step("echo hi")


def test_step_rejects_non_string_command():
    calls = []
    ref = [None]
    world = make_world(calls, std_behavior(IDA, ref))
    ref[0] = world
    world.reset()
    with pytest.raises(WorldError):
        world.step(b"echo hi")


def test_r2_timeout_rm_failure_keeps_handle():
    ref = [None]

    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] in ("create", "start"):
            return ExecutorResult(0, (R2_A + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (R2_A + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "exec":
            raise ExecutorTimeout(b"partial-out", b"partial-err")
        if argv[1] == "rm":
            return ExecutorResult(1, b"", b"rm failed\n")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    ref[0] = world
    world.reset()
    with pytest.raises(WorldError) as caught:
        world.step("sleep 999")
    assert world._container_id == R2_A
    assert world._active is False
    assert caught.value.__cause__ is not None or isinstance(caught.value, CleanupError)
    if isinstance(caught.value, CleanupError):
        assert caught.value.receipt.stdout == b"partial-out"


def test_r2_inspect_nonzero_keeps_handle_and_blocks_next_start():
    ref = [None]
    n = [0]

    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] in ("create", "start"):
            return ExecutorResult(0, (R2_A + "\n").encode(), b"")
        if argv[1] == "inspect":
            n[0] += 1
            if n[0] == 1:
                return ExecutorResult(0, (R2_A + "|" + w._nonce + "|container_world").encode(), b"")
            return ExecutorResult(1, b"", b"no such container\n")
        if argv[1] == "exec":
            raise ExecutorTimeout(b"out", b"err")
        if argv[1] == "rm":
            raise AssertionError("rm must not run after inspect failure")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    ref[0] = world
    world.reset()
    with pytest.raises(WorldError):
        world.step("sleep 1")
    assert world._container_id == R2_A
    assert world._active is False
    with pytest.raises(WorldError):
        world.reset()


def test_r2_inspect_timeout_keeps_handle():
    ref = [None]
    n = [0]

    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] in ("create", "start"):
            return ExecutorResult(0, (R2_A + "\n").encode(), b"")
        if argv[1] == "inspect":
            n[0] += 1
            if n[0] == 1:
                return ExecutorResult(0, (R2_A + "|" + w._nonce + "|container_world").encode(), b"")
            raise ExecutorTimeout(b"", b"")
        if argv[1] == "exec":
            raise ExecutorTimeout(b"out", b"err")
        if argv[1] == "rm":
            raise AssertionError("rm must not run after inspect timeout")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    ref[0] = world
    world.reset()
    with pytest.raises(WorldError):
        world.step("sleep 1")
    assert world._container_id == R2_A


def test_r2_rm_oserror_keeps_handle_and_retry_succeeds():
    ref = [None]
    calls = []
    attempt = [0]

    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] in ("create", "start"):
            cid = R2_A if len([c for c in calls if c[0][1] == "create"]) <= 1 else IDB
            return ExecutorResult(0, (cid + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "exec":
            raise ExecutorTimeout(b"out", b"err")
        if argv[1] == "rm":
            attempt[0] += 1
            if attempt[0] == 1:
                raise OSError("daemon hiccup")
            return ExecutorResult(0, (argv[-1] + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world(calls, behavior)
    ref[0] = world
    world.reset()
    with pytest.raises(WorldError):
        world.step("sleep 1")
    assert world._container_id == R2_A
    cid = world.reset()
    assert cid == IDB
    assert world._active is True


def test_r2_combined_cap_real_runner():
    from kullback.container_world import _bounded_run

    script = "import sys; sys.stdout.buffer.write(b'o' * 600); sys.stdout.flush(); sys.stderr.buffer.write(b'e' * 600); sys.stderr.flush()"
    with pytest.raises(OutputLimitExceeded) as caught:
        _bounded_run([sys.executable, "-c", script], 20.0, 1024)
    assert len(caught.value.stdout) + len(caught.value.stderr) == 1024


def test_r2_exact_cap_real_runner_passes():
    from kullback.container_world import _bounded_run

    script = "import sys; sys.stdout.buffer.write(b'o' * 512); sys.stderr.buffer.write(b'e' * 512)"
    res = _bounded_run([sys.executable, "-c", script], 20.0, 1024)
    assert res.returncode == 0
    assert len(res.stdout) + len(res.stderr) == 1024


def test_r2_cap_plus_one_real_runner_fails_exact():
    from kullback.container_world import _bounded_run

    script = "import sys; sys.stdout.buffer.write(b'o' * 512); sys.stderr.buffer.write(b'e' * 513)"
    with pytest.raises(OutputLimitExceeded) as caught:
        _bounded_run([sys.executable, "-c", script], 20.0, 1024)
    assert len(caught.value.stdout) + len(caught.value.stderr) == 1024


def test_r2_partial_timeout_bounded_real_runner():
    from kullback.container_world import _bounded_run

    script = "import sys, time; sys.stdout.buffer.write(b'x' * 100); sys.stdout.flush(); time.sleep(30)"
    begin = time.monotonic()
    with pytest.raises(ExecutorTimeout) as caught:
        _bounded_run([sys.executable, "-c", script], 1.0, 65536)
    elapsed = time.monotonic() - begin
    assert elapsed < 12
    assert caught.value.stdout == b"x" * 100
    assert len(caught.value.stdout) + len(caught.value.stderr) <= 65536


def test_r2_inherited_pipe_bounded():
    import time

    from kullback.container_world import _bounded_run

    parent = "import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], stdout=sys.stdout, stderr=sys.stderr); sys.stdout.flush()"
    begin = time.monotonic()
    with pytest.raises(CaptureIncomplete) as caught:
        _bounded_run([sys.executable, "-c", parent], 30.0, 65536)
    elapsed = time.monotonic() - begin
    assert elapsed < 12
    assert len(caught.value.stdout) + len(caught.value.stderr) <= 65536


def test_r2_timeout_receipt_is_combined_bounded():
    ref = [None]

    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] in ("create", "start"):
            return ExecutorResult(0, (R2_A + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (R2_A + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "exec":
            raise ExecutorTimeout(b"o" * 2000, b"e" * 2000)
        if argv[1] == "rm":
            return ExecutorResult(0, (R2_A + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior, output_limit_bytes=1024)
    ref[0] = world
    world.reset()
    receipt = world.step("sleep 1")
    assert receipt.timed_out is True
    assert len(receipt.stdout) + len(receipt.stderr) == 1024


def test_r2_bytes_receipt_preserves_non_utf8():
    raw = b"\xff\xfe\x00raw\x01"
    ref = [None]

    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] == "create":
            return ExecutorResult(0, (R2_A + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (R2_A + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            return ExecutorResult(0, (R2_A + "\n").encode(), b"")
        if argv[1] == "exec":
            return ExecutorResult(0, raw, b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    ref[0] = world
    world.reset()
    receipt = world.step("printf x")
    assert receipt.stdout == raw


def test_r2_control_limit_one_still_cleans_up():
    calls = []
    ref = [None]
    world = make_world(calls, std_behavior(IDA, ref), output_limit_bytes=1)
    ref[0] = world
    world.reset()
    assert world._container_id == IDA

    def extra(argv, timeout):
        if argv[1] == "exec":
            raise ExecutorTimeout(b"a", b"b")
        raise AssertionError("unexpected " + argv[1])

    world._executor = make_handler(calls, std_behavior(IDA, ref, extra))
    receipt = world.step("sleep 1")
    assert receipt.timed_out is True
    assert world._container_id is None


def test_r2_short_id_is_unresolved():
    from kullback.container_world import UnresolvedError

    def behavior(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, image_ok_payload(), b"")
        if argv[1] == "create":
            return ExecutorResult(0, b"cid001\n", b"")
        raise AssertionError("unexpected " + argv[1])

    world = ContainerWorld(IMG, docker=DOCK, image_metadata=R2_META, executor=make_handler([], behavior))
    with pytest.raises(UnresolvedError):
        world.reset()
    with pytest.raises(UnresolvedError):
        world.reset()


def test_r2_flag_and_uppercase_ids_rejected():
    for bad in [b"-flag\n", b"ABCDEF" * 10 + b"\n", b"ab\n", b"" + b"a" * 63 + b"\n"]:
        def behavior(argv, timeout, _bad=bad):
            if argv[1] == "create":
                return ExecutorResult(0, _bad, b"")
            raise AssertionError("unexpected " + argv[1])

        world = make_world([], behavior)
        with pytest.raises(UnresolvedError):
            world.reset()


def test_r2_executor_result_types_fail_closed():
    ref = [None]
    bads = [
        ExecutorResult(True, b"", b""),
        ExecutorResult(0, "str", b""),
        ExecutorResult(0, b"", "str"),
        ExecutorResult(0, None, b""),
    ]
    for bad in bads:
        def behavior(argv, timeout, _bad=bad):
            if argv[1] == "create":
                return _bad
            raise AssertionError("unexpected " + argv[1])

        world = make_world([], behavior)
        ref[0] = world
        with pytest.raises(WorldError):
            world.reset()


def test_r2_none_result_fails_closed():
    ref = [None]

    def behavior(argv, timeout):
        if argv[1] == "create":
            return None
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    ref[0] = world
    with pytest.raises(WorldError):
        world.reset()


def test_r2_create_timeout_is_unresolved():
    ref = [None]

    def behavior(argv, timeout):
        if argv[1] == "create":
            raise ExecutorTimeout(b"", b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    ref[0] = world
    with pytest.raises(UnresolvedError):
        world.reset()
    assert world._unresolved is True
    with pytest.raises(UnresolvedError):
        world.reset()


def test_r2_create_overflow_is_unresolved():
    ref = [None]

    def behavior(argv, timeout):
        if argv[1] == "create":
            raise OutputLimitExceeded(b"x" * 10, b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world([], behavior)
    ref[0] = world
    with pytest.raises(UnresolvedError):
        world.reset()
    assert world._unresolved is True


def test_r2_start_nonzero_cleans_up_and_allows_retry():
    ref = [None]
    calls = []
    n = [0]

    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] == "create":
            cid = IDA if n[0] == 0 else IDB
            return ExecutorResult(0, (cid + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            n[0] += 1
            if n[0] == 1:
                return ExecutorResult(1, b"", b"start failed\n")
            return ExecutorResult(0, (IDB + "\n").encode(), b"")
        if argv[1] == "rm":
            return ExecutorResult(0, (argv[-1] + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world(calls, behavior)
    ref[0] = world
    with pytest.raises(DockerUnavailable):
        world.reset()
    assert world._container_id is None
    cid = world.reset()
    assert cid == IDB


def test_r2_start_timeout_retains_and_retry_removes():
    ref = [None]
    calls = []
    n = [0]

    def behavior(argv, timeout):
        w = ref[0]
        if argv[1] == "create":
            cid = IDA if len([c for c in calls if c[0][1] == "create"]) == 1 else IDB
            return ExecutorResult(0, (cid + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            n[0] += 1
            if n[0] == 1:
                raise ExecutorTimeout(b"", b"")
            return ExecutorResult(0, (IDB + "\n").encode(), b"")
        if argv[1] == "rm":
            return ExecutorResult(0, (argv[-1] + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])

    world = make_world(calls, behavior)
    ref[0] = world
    with pytest.raises(CleanupError):
        world.reset()
    assert world._container_id == IDA
    cid = world.reset()
    assert cid == IDB


def test_r2_volumes_metadata_rejected():
    with pytest.raises(ImagePinError):
        ContainerWorld(IMG, docker=DOCK, image_metadata={"Config": {"Volumes": {"/data": {}}}})
    with pytest.raises(ImagePinError):
        ContainerWorld(IMG, docker=DOCK, image_metadata=None)


def test_r2_missing_volumes_key_is_valid():
    ref = [None]
    def behavior(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, image_ok_payload(), b"")
        return std_behavior(IDA, ref)(argv, timeout)
    world = ContainerWorld(IMG, docker=DOCK, executor=make_handler([], behavior), image_metadata=dict(GOOD_EMPTY))
    ref[0] = world
    assert world.reset() == IDA


def test_r2_docker_missing_path_real_fails_but_fake_allows():
    with pytest.raises(DockerUnavailable):
        ContainerWorld(IMG, docker="/nonexistent/docker-xyz", image_metadata=dict(GOOD))
    ref = [None]
    def behavior2(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, image_ok_payload(), b"")
        return std_behavior(IDA, ref)(argv, timeout)
    world = ContainerWorld(IMG, docker="/nonexistent/docker-xyz", image_metadata=dict(GOOD), executor=make_handler([], behavior2))
    ref[0] = world
    assert world.reset() == IDA


def test_r2_unexecutable_docker_path_rejected_for_real(tmp_path):
    path = tmp_path / "docker-fake"
    path.write_bytes(b"not-executable")
    path.chmod(0o644)
    assert os.path.isfile(str(path))
    with pytest.raises(DockerUnavailable):
        ContainerWorld(IMG, docker=str(path), image_metadata=dict(GOOD))


def r4_image_payload(ident=None, repo=None, vols=None):
    digest = IMG.split("@sha256:")[1]
    ident = ident if ident is not None else "sha256:" + digest
    repo = repo if repo is not None else [IMG]
    vols = vols if vols is not None else {}
    return (ident + "|" + json.dumps(repo) + "|" + json.dumps(vols)).encode()


def r4_make_unresolved(timeout_create=True):
    ref = [None]
    calls = []
    def behavior(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, r4_image_payload(), b"")
        if argv[1] == "create":
            raise ExecutorTimeout(b"", b"")
        raise AssertionError("unexpected " + argv[1])
    world = make_raw_world(calls, behavior)
    ref.append(world)
    return (world, calls)


def test_r4_unresolved_inspect_variants_keep_reservation():
    variants = [
        ExecutorResult(1, b"", b"Error: No such container"),
        ExecutorResult(1, b"", b"permission denied"),
        ExecutorResult(1, b"", b"Cannot connect to the Docker daemon"),
    ]
    for bad in variants:
        calls = []
        n = [0]
        def behavior(argv, timeout, _bad=bad, _n=n):
            if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
                return ExecutorResult(0, r4_image_payload(), b"")
            if argv[1] == "create":
                _n[0] += 1
                if _n[0] == 1:
                    raise ExecutorTimeout(b"", b"")
                raise AssertionError("new create must not run while unresolved")
            if argv[1] == "inspect":
                return _bad
            raise AssertionError("unexpected " + argv[1])
        world = make_raw_world(calls, behavior)
        with pytest.raises(UnresolvedError):
            world.reset()
        assert world._unresolved is True
        creates = [c for c in calls if c[0][1] == "create"]
        assert len(creates) == 1
        with pytest.raises(UnresolvedError):
            world.close()
        assert world._unresolved is True
        with pytest.raises(UnresolvedError):
            world.reset()
        creates = [c for c in calls if c[0][1] == "create"]
        assert len(creates) == 1


def test_r4_unresolved_timeout_and_oserror_keep_reservation():
    for mode in ["timeout", "oserror"]:
        calls = []
        n = [0]
        def behavior(argv, timeout, _mode=mode, _n=n):
            if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
                return ExecutorResult(0, r4_image_payload(), b"")
            if argv[1] == "create":
                _n[0] += 1
                if _n[0] == 1:
                    raise ExecutorTimeout(b"", b"")
                raise AssertionError("new create must not run")
            if argv[1] == "inspect":
                if _mode == "timeout":
                    raise ExecutorTimeout(b"", b"")
                raise OSError("daemon hiccup")
            raise AssertionError("unexpected " + argv[1])
        world = make_raw_world(calls, behavior)
        with pytest.raises(UnresolvedError):
            world.reset()
        with pytest.raises(WorldError):
            world.close()
        assert world._unresolved is True
        with pytest.raises(UnresolvedError):
            world.reset()
        assert len([c for c in calls if c[0][1] == "create"]) == 1


def test_r4_successful_recovery_removes_only_owned():
    ref = [None]
    calls = []
    owned = ["ee" * 32]
    n = [0]
    rms = []
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
    world = make_raw_world(calls, behavior)
    ref[0] = world
    with pytest.raises(UnresolvedError):
        world.reset()
    assert world.close() is True
    assert rms == [[DOCK, "rm", "-f", owned[0]]]
    assert world._unresolved is False
    cid = world.reset()
    assert cid == IDB


def test_r4_name_collision_never_removes_foreign():
    ref = [None]
    calls = []
    rms = []
    def behavior(argv, timeout):
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
    world = make_raw_world(calls, behavior)
    ref[0] = world
    with pytest.raises(UnresolvedError):
        world.reset()
    with pytest.raises(UnresolvedError):
        world.close()
    assert rms == []
    assert world._unresolved is True


def test_r4_new_session_descendant_is_incomplete_bounded():
    import time

    from kullback.container_world import _bounded_run
    parent = "import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], stdout=sys.stdout, stderr=sys.stderr, start_new_session=True); sys.stdout.buffer.write(b'hi'); sys.stdout.flush()"
    begin = time.monotonic()
    with pytest.raises(CaptureIncomplete) as caught:
        _bounded_run([sys.executable, "-c", parent], 10.0, 65536)
    elapsed = time.monotonic() - begin
    assert elapsed < 12
    assert len(caught.value.stdout) + len(caught.value.stderr) <= 65536


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


def test_r4_image_volumes_block_creation_without_create():
    calls = []
    def behavior(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, r4_image_payload(vols={"/data": {}}), b"")
        if argv[1] == "create":
            raise AssertionError("create must not run when volumes declared")
        raise AssertionError("unexpected " + argv[1])
    world = make_raw_world(calls, behavior)
    with pytest.raises(ImagePinError):
        world.reset()
    assert [c for c in calls if c[0][1] == "create"] == []


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


def test_r4_image_unreadable_blocks_creation():
    for mode in ["nonzero-missing", "nonzero-ambiguous", "timeout"]:
        calls = []
        def behavior(argv, timeout, _m=mode):
            if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
                if _m == "nonzero-missing":
                    return ExecutorResult(1, b"", b"Error: No such image")
                if _m == "nonzero-ambiguous":
                    return ExecutorResult(1, b"", b"permission denied")
                raise ExecutorTimeout(b"", b"")
            if argv[1] == "create":
                raise AssertionError("create must not run when unreadable")
            raise AssertionError("unexpected " + argv[1])
        world = make_raw_world(calls, behavior)
        with pytest.raises(WorldError):
            world.reset()
        assert [c for c in calls if c[0][1] == "create"] == []


def test_r4_supplied_hint_does_not_substitute_runtime():
    calls = []
    def behavior(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, r4_image_payload(vols={"/x": {}}), b"")
        if argv[1] == "create":
            raise AssertionError("create must not run")
        raise AssertionError("unexpected " + argv[1])
    world = make_raw_world(calls, behavior, image_metadata=dict(GOOD))
    with pytest.raises(ImagePinError):
        world.reset()


def test_r4_ambiguous_create_keeps_reservation_proven_allows_retry():
    calls = []
    n = [0]
    def behavior(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, r4_image_payload(), b"")
        if argv[1] == "create":
            n[0] += 1
            if n[0] == 1:
                return ExecutorResult(1, b"", b"Error: daemonperiodic failure")
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (IDA + "|" + ref[0]._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])
    ref = [None]
    world = make_raw_world(calls, behavior)
    ref.append(world)
    ref[0] = world
    with pytest.raises(UnresolvedError):
        world.reset()
    assert world._unresolved is True
    with pytest.raises(UnresolvedError):
        world.reset()
    assert n[0] == 1


def test_r4_proven_image_missing_allows_retry():
    calls = []
    n = [0]
    ref = [None]
    def behavior(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, r4_image_payload(), b"")
        if argv[1] == "create":
            n[0] += 1
            if n[0] == 1:
                return ExecutorResult(1, b"", b"Error: No such image")
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (IDA + "|" + ref[0]._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        if argv[1] == "rm":
            return ExecutorResult(0, (argv[-1] + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])
    world = make_raw_world(calls, behavior)
    ref.append(world)
    ref[0] = world
    with pytest.raises(DockerUnavailable):
        world.reset()
    assert world._unresolved is False
    cid = world.reset()
    assert cid == IDA
    assert n[0] == 2


def test_r5_create_keyboard_interrupt_keeps_reservation():
    calls = []
    n = [0]
    def behavior(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, image_ok_payload(), b"")
        if argv[1] == "create":
            n[0] += 1
            raise KeyboardInterrupt()
        if argv[1] == "inspect":
            return ExecutorResult(1, b"", b"Error: No such container")
        raise AssertionError("unexpected " + argv[1])
    world = make_raw_world(calls, behavior)
    with pytest.raises(KeyboardInterrupt):
        world.reset()
    assert world._unresolved is True
    assert world._container_id is None
    with pytest.raises(UnresolvedError):
        world.reset()
    assert n[0] == 1
    with pytest.raises(UnresolvedError):
        world.close()


def test_r5_exec_keyboard_interrupt_deactivates_keeps_id():
    calls = []
    ref = [None]
    n = [0]
    def behavior(argv, timeout):
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
            raise KeyboardInterrupt()
        if argv[1] == "rm":
            return ExecutorResult(0, (argv[-1] + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])
    world = make_raw_world(calls, behavior)
    ref[0] = world
    world.reset()
    with pytest.raises(KeyboardInterrupt):
        world.step("echo hi")
    assert world._active is False
    assert world._container_id is None
    assert n[0] == 1
    with pytest.raises(StateError):
        world.step("echo hi")
    assert n[0] == 1


def test_r5_exec_interrupt_cleanup_failure_keeps_id_chains():
    calls = []
    ref = [None]
    def behavior(argv, timeout):
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
            raise KeyboardInterrupt()
        if argv[1] == "rm":
            raise KeyboardInterrupt()
        raise AssertionError("unexpected " + argv[1])
    world = make_raw_world(calls, behavior)
    ref[0] = world
    world.reset()
    with pytest.raises(KeyboardInterrupt) as caught:
        world.step("echo hi")
    assert world._active is False
    assert world._container_id == IDA
    assert caught.value.__cause__ is not None


def test_r5_start_keyboard_interrupt_retains_safe_retry():
    calls = []
    ref = [None]
    n = [0]
    rms = []
    def behavior(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, image_ok_payload(), b"")
        w = ref[0]
        if argv[1] == "create":
            cid = IDA if len([c for c in calls if c[0][1] == "create"]) == 1 else IDB
            return ExecutorResult(0, (cid + "\n").encode(), b"")
        if argv[1] == "inspect":
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            n[0] += 1
            if n[0] == 1:
                raise KeyboardInterrupt()
            return ExecutorResult(0, (IDB + "\n").encode(), b"")
        if argv[1] == "rm":
            rms.append(list(argv))
            return ExecutorResult(0, (argv[-1] + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])
    world = make_raw_world(calls, behavior)
    ref[0] = world
    with pytest.raises(KeyboardInterrupt):
        world.reset()
    assert world._container_id == IDA
    assert world._active is False
    cid = world.reset()
    assert cid == IDB
    assert rms == [[DOCK, "rm", "-f", IDA]]


def _r5_cleanup_rest(argv, ref, n, verb):
    w = ref[0]
    if argv[1] == "create":
        return ExecutorResult(0, (IDA + "\n").encode(), b"")
    if argv[1] == "inspect":
        n[0] += 1
        if verb == "inspect" and n[0] > 1:
            raise KeyboardInterrupt()
        return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
    if argv[1] == "start":
        return ExecutorResult(0, (IDA + "\n").encode(), b"")
    if argv[1] == "exec":
        raise ExecutorTimeout(b"out", b"err")
    if argv[1] == "rm":
        if verb == "rm":
            raise KeyboardInterrupt()
        return ExecutorResult(0, (argv[-1] + "\n").encode(), b"")
    raise AssertionError("unexpected " + argv[1])


def _r5_cleanup_behavior(ref, n, verb):
    def behavior(argv, timeout, _v=verb, _n=n, _r=ref):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, image_ok_payload(), b"")
        return _r5_cleanup_rest(argv, _r, _n, _v)
    return behavior


def test_r5_inspect_keyboard_interrupt_retains():
    calls = []
    ref = [None]
    n = [0]
    world = make_raw_world(calls, _r5_cleanup_behavior(ref, n, "inspect"))
    ref[0] = world
    world.reset()
    with pytest.raises(KeyboardInterrupt):
        world.step("sleep 1")
    assert world._active is False
    assert world._container_id == IDA


def test_r5_rm_keyboard_interrupt_retains():
    calls = []
    ref = [None]
    n = [0]
    world = make_raw_world(calls, _r5_cleanup_behavior(ref, n, "rm"))
    ref[0] = world
    world.reset()
    with pytest.raises(KeyboardInterrupt):
        world.step("sleep 1")
    assert world._active is False
    assert world._container_id == IDA


def test_r5_proven_predispatch_clears_allows_retry():
    calls = []
    n = [0]
    ref = [None]
    def behavior(argv, timeout):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, image_ok_payload(), b"")
        if argv[1] == "create":
            n[0] += 1
            if n[0] == 1:
                raise LaunchFailure("docker binary gone")
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        w = ref[0]
        if argv[1] == "inspect":
            return ExecutorResult(0, (argv[-1] + "|" + w._nonce + "|container_world").encode(), b"")
        if argv[1] == "start":
            return ExecutorResult(0, (IDA + "\n").encode(), b"")
        raise AssertionError("unexpected " + argv[1])
    world = make_raw_world(calls, behavior)
    ref[0] = world
    with pytest.raises(DockerUnavailable):
        world.reset()
    assert world._unresolved is False
    assert world._container_id is None
    cid = world.reset()
    assert cid == IDA
    assert n[0] == 2


def test_r5_bounded_run_real_sigint_reaps_owned_child(tmp_path):
    import fcntl
    import os as _os
    import signal as _signal
    import threading as _threading

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


def _r6_generic_behavior(n, mode):
    def behavior(argv, timeout, _m=mode, _n=n):
        if len(argv) > 2 and argv[1] == "image" and argv[2] == "inspect":
            return ExecutorResult(0, image_ok_payload(), b"")
        if argv[1] == "create":
            _n[0] += 1
            if _n[0] == 1:
                if _m == "oserror":
                    raise OSError("side effect then ambiguous")
                raise DockerUnavailable("ambiguous")
            raise AssertionError("second create must not run")
        if argv[1] == "inspect":
            return ExecutorResult(1, b"", b"Error: No such container")
        raise AssertionError("unexpected " + argv[1])
    return behavior


def test_r6_generic_create_failure_retains_blocks_second_create():
    for mode in ["oserror", "docker-unavailable"]:
        calls = []
        n = [0]
        world = make_raw_world(calls, _r6_generic_behavior(n, mode))
        with pytest.raises(UnresolvedError):
            world.reset()
        assert world._unresolved is True
        assert n[0] == 1
        with pytest.raises(UnresolvedError):
            world.reset()
        assert n[0] == 1
        with pytest.raises(UnresolvedError):
            world.close()


def test_r6_proven_launch_failure_subprocess_and_fixture(tmp_path):
    import time

    from kullback.container_world import _bounded_run
    begin = time.monotonic()
    with pytest.raises(LaunchFailure):
        _bounded_run(["/nonexistent/kullback-docker-xyz-123"], 10.0, 65536)
    assert time.monotonic() - begin < 12
    script = tmp_path / "hello.py"
    script.write_text("import sys; sys.stdout.buffer.write(b'hi')")
    res = _bounded_run([sys.executable, str(script)], 10.0, 65536)
    assert res.returncode == 0
    assert res.stdout == b"hi"


def test_r7_setup_capture_failure_reaps_owned_child():
    import selectors as _selectors
    import subprocess as _subprocess
    from unittest import mock as _mock

    from kullback.container_world import _bounded_run

    real_popen = _subprocess.Popen
    real_selector = _selectors.DefaultSelector
    seen = []

    def _recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        seen.append(proc)
        return proc

    class _OnePipeSelector(real_selector):
        def register(self, fileobj, events, data=None):
            if data == "err":
                return None
            return super().register(fileobj, events, data=data)

    begin = time.monotonic()
    with _mock.patch.object(_subprocess, "Popen", _recording_popen):
        with _mock.patch.object(_selectors, "DefaultSelector", _OnePipeSelector):
            with pytest.raises(CaptureIncomplete) as caught:
                _bounded_run([sys.executable, "-c", "import time; time.sleep(30)"], 10.0, 65536)
    elapsed = time.monotonic() - begin
    try:
        assert elapsed < 12
        assert len(seen) == 1
        assert seen[0].poll() is not None
        assert len(caught.value.stdout) + len(caught.value.stderr) <= 65536
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


def test_r7_no_group_signal_after_leader_reaped():
    import subprocess as _subprocess
    from unittest import mock as _mock

    from kullback.container_world import _bounded_run

    real_killpg = os.killpg
    real_popen = _subprocess.Popen
    calls = []
    seen = []

    def _recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        seen.append(proc)
        return proc

    def _counting_killpg(pid, sig):
        calls.append((pid, sig))
        return real_killpg(pid, sig)

    parent = "import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(15)'], stdout=sys.stdout, stderr=sys.stderr); sys.stdout.flush()"
    begin = time.monotonic()
    with _mock.patch.object(_subprocess, "Popen", _recording_popen):
        with _mock.patch.object(os, "killpg", _counting_killpg):
            with pytest.raises(CaptureIncomplete) as caught:
                _bounded_run([sys.executable, "-c", parent], 30.0, 65536)
    elapsed = time.monotonic() - begin
    assert elapsed < 12
    assert calls == []
    assert len(seen) == 1
    assert seen[0].poll() is not None
    assert len(caught.value.stdout) + len(caught.value.stderr) <= 65536


def test_r7_timeout_kills_owned_child_once_and_reaps():
    import subprocess as _subprocess
    from unittest import mock as _mock

    from kullback.container_world import _bounded_run

    real_killpg = os.killpg
    real_popen = _subprocess.Popen
    calls = []
    seen = []

    def _recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        seen.append(proc)
        return proc

    def _counting_killpg(pid, sig):
        calls.append((pid, sig))
        return real_killpg(pid, sig)

    script = "import sys, time; sys.stdout.buffer.write(b'x' * 100); sys.stdout.flush(); time.sleep(30)"
    begin = time.monotonic()
    with _mock.patch.object(_subprocess, "Popen", _recording_popen):
        with _mock.patch.object(os, "killpg", _counting_killpg):
            with pytest.raises(ExecutorTimeout) as caught:
                _bounded_run([sys.executable, "-c", script], 1.0, 65536)
    elapsed = time.monotonic() - begin
    try:
        assert elapsed < 12
        assert caught.value.stdout == b"x" * 100
        assert len(calls) == 1
        assert len(seen) == 1
        assert seen[0].poll() is not None
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


def test_r8_post_reap_drain_after_exit_sends_no_signal():
    import selectors as _selectors
    from unittest import mock as _mock

    from kullback import container_world as _cw

    real_killpg = os.killpg
    calls = []

    def _counting_killpg(pid, sig):
        calls.append((pid, sig))
        return real_killpg(pid, sig)

    stale_pid = (1 << 30) | 12345
    rf = None
    sel = None
    try:
        r, w = os.pipe()
        try:
            os.write(w, b"o" * 2000)
        finally:
            os.close(w)
        rf = os.fdopen(r, "rb")
        sel = _selectors.DefaultSelector()
        sel.register(rf, _selectors.EVENT_READ, data="out")

        class _ReapedProc:
            pid = stale_pid
            returncode = 5

            def poll(self):
                return 5

            def kill(self):
                return None

            def wait(self, timeout=None):
                return 5

        proc = _ReapedProc()
        out = bytearray()
        err = bytearray()
        total = [0]
        with _mock.patch.object(os, "killpg", _counting_killpg):
            with pytest.raises(OutputLimitExceeded) as caught:
                _cw._finish_run(proc, sel, out, err, total, 1024, 5, False, time.monotonic() + 30)
        assert calls == []
        assert len(caught.value.stdout) + len(caught.value.stderr) == 1024
    finally:
        try:
            if sel is not None:
                sel.close()
        except Exception:
            pass
        try:
            if rf is not None:
                rf.close()
        except Exception:
            pass


def test_r8_timeout_drain_overflow_kills_once_only():
    import selectors as _selectors
    from unittest import mock as _mock

    from kullback import container_world as _cw

    real_killpg = os.killpg
    calls = []

    def _counting_killpg(pid, sig):
        calls.append((pid, sig))
        return real_killpg(pid, sig)

    stale_pid = (1 << 30) | 54321
    rf = None
    sel = None
    try:
        r, w = os.pipe()
        try:
            os.write(w, b"z" * 2000)
        finally:
            os.close(w)
        rf = os.fdopen(r, "rb")
        sel = _selectors.DefaultSelector()
        sel.register(rf, _selectors.EVENT_READ, data="out")

        class _LiveThenReaped:
            pid = stale_pid
            returncode = None

            def poll(self):
                return self.returncode

            def kill(self):
                return None

            def wait(self, timeout=None):
                if self.returncode is None:
                    self.returncode = -9
                return self.returncode

        proc = _LiveThenReaped()
        out = bytearray()
        err = bytearray()
        total = [0]
        with _mock.patch.object(os, "killpg", _counting_killpg):
            with pytest.raises(ExecutorTimeout) as caught:
                _cw._finish_run(proc, sel, out, err, total, 1024, 0, False, time.monotonic() - 1)
        assert len(calls) == 1
        assert len(caught.value.stdout) + len(caught.value.stderr) <= 1024
    finally:
        try:
            if sel is not None:
                sel.close()
        except Exception:
            pass
        try:
            if rf is not None:
                rf.close()
        except Exception:
            pass


def test_r8_reap_owned_keyboard_interrupt_propagates_and_reaps():
    import subprocess as _subprocess
    from unittest import mock as _mock

    from kullback import container_world as _cw

    real_wait_reap = _cw._wait_reap
    state = {"n": 0}

    def _flaky_wait_reap(proc, wait_s):
        state["n"] += 1
        if state["n"] == 1:
            raise KeyboardInterrupt()
        return real_wait_reap(proc, wait_s)

    proc = _subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=_subprocess.DEVNULL,
        stdout=_subprocess.PIPE,
        stderr=_subprocess.PIPE,
        shell=False,
        start_new_session=True,
    )
    try:
        with _mock.patch.object(_cw, "_wait_reap", _flaky_wait_reap):
            with pytest.raises(KeyboardInterrupt):
                _cw._reap_owned(proc)
        assert state["n"] >= 2
        assert proc.returncode is not None
        assert proc.poll() is not None
    finally:
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
