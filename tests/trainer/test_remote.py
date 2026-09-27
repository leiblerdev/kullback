"""Remote backends over SSH or this machine: shell, GPU status, runs, sync."""

from __future__ import annotations

import json
import sys
import time

from kullback.trainer.remote import (
    RemoteConfig,
    gpu_status,
    parse_nvidia_smi,
    run,
    run_alive,
    shell,
    start_run,
    stop_run,
    sync,
)


def make_cfg(tmp_path, name="remote"):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    return RemoteConfig(host="local", root=str(root), python=sys.executable)

def test_run_returns_output_and_nonzero_without_raising(tmp_path):
    cfg = make_cfg(tmp_path)
    result = run(cfg, "echo hello; echo oops >&2; exit 3")
    assert result["returncode"] == 3
    assert "hello" in result["stdout"]
    assert "oops" in result["stderr"]
    assert result["timed_out"] is False


def test_run_timeout_sets_timed_out(tmp_path):
    cfg = make_cfg(tmp_path)
    result = run(cfg, "sleep 5", timeout_s=1)
    assert result["timed_out"] is True


def test_shell_appends_one_log_line_per_call(tmp_path):
    cfg = make_cfg(tmp_path)
    workdir = tmp_path / "work"
    shell(cfg, "echo one", workdir=workdir)
    shell(cfg, "echo two", workdir=workdir)
    lines = (workdir / "training" / "shell.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["host"] == "local"
    assert first["command"] == "echo one"
    assert first["returncode"] == 0
    assert "seconds" in first and "time" in first


def test_parse_nvidia_smi_reads_two_fixed_gpu_lines():
    text = (
        "0, NVIDIA A100-SXM4-40GB, 1234, 40960, 56, 42\n"
        "1, NVIDIA A100-SXM4-40GB, 0, 40960, 0, 31\n"
    )
    gpus = parse_nvidia_smi(text)
    assert len(gpus) == 2
    assert gpus[0] == {
        "index": 0,
        "name": "NVIDIA A100-SXM4-40GB",
        "memory_used_mib": 1234,
        "memory_total_mib": 40960,
        "utilization_pct": 56,
        "temperature_c": 42,
    }
    assert gpus[1]["index"] == 1
    assert gpus[1]["temperature_c"] == 31


def test_gpu_status_reports_error_and_no_gpus_when_the_query_fails(tmp_path):
    import shutil

    import pytest

    if shutil.which("nvidia-smi") is not None:
        pytest.skip("nvidia-smi is installed")
    cfg = RemoteConfig(host="local", root=str(tmp_path), python=sys.executable)
    result = gpu_status(cfg)
    assert result["gpus"] == []
    assert result["error"].strip() != ""


def write_sleeper(root, run_id):
    script = (
        "import time\n"
        f"open('runs/{run_id}/marker.txt', 'w').write('up')\n"
        "time.sleep(60)\n"
    )
    (root / "hello.py").write_text(script, encoding="utf-8")


def wait_for(path, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.1)
    return False


def test_start_run_launches_script_and_stop_marks_stopped(tmp_path):
    cfg = make_cfg(tmp_path)
    workdir = tmp_path / "work"
    root = tmp_path / "remote"
    write_sleeper(root, "abc123")
    try:
        started = start_run(cfg, "abc123", "hello.py", workdir=workdir)
        assert started["started"] is True
        assert isinstance(started["pid"], int)
        assert run_alive(cfg, "abc123") is True
        assert wait_for(root / "runs" / "abc123" / "marker.txt")
        (root / "runs" / "abc123" / "state.json").write_text(
            json.dumps({"status": "running", "step": 3}), encoding="utf-8")
        stopped = stop_run(cfg, "abc123", workdir=workdir)
        assert stopped["stopped"] is True
        assert run_alive(cfg, "abc123") is False
        state = json.loads((root / "runs" / "abc123" / "state.json").read_text(encoding="utf-8"))
        assert state["status"] == "stopped"
        assert state["step"] == 3
    finally:
        stop_run(cfg, "abc123")


def test_second_start_while_alive_is_refused(tmp_path):
    cfg = make_cfg(tmp_path)
    root = tmp_path / "remote"
    write_sleeper(root, "dup1")
    try:
        first = start_run(cfg, "dup1", "hello.py")
        assert first["started"] is True
        second = start_run(cfg, "dup1", "hello.py")
        assert second["started"] is False
        assert "reason" in second
    finally:
        stop_run(cfg, "dup1")


def test_bad_run_id_is_refused(tmp_path):
    cfg = make_cfg(tmp_path)
    result = start_run(cfg, "Bad ID!", "hello.py")
    assert result["started"] is False
    assert "reason" in result


def test_sync_copies_files_into_training_run_dir(tmp_path):
    cfg = make_cfg(tmp_path)
    root = tmp_path / "remote"
    workdir = tmp_path / "work"
    rundir = root / "runs" / "r7"
    rundir.mkdir(parents=True)
    (rundir / "state.json").write_text('{"status": "running"}', encoding="utf-8")
    (rundir / "run.log").write_text("log lines", encoding="utf-8")
    result = sync(cfg, "r7", workdir)
    assert result["copied"] >= 2
    assert "seconds" in result
    dest = workdir / "training" / "r7"
    assert (dest / "state.json").exists()
    assert (dest / "run.log").exists()


def test_stop_and_sync_refuse_a_run_id_with_shell_characters(tmp_path):
    import pytest

    cfg = make_cfg(tmp_path)
    root = tmp_path / "remote"
    (root / "y").write_text("canary", encoding="utf-8")
    with pytest.raises(ValueError, match="bad run_id"):
        stop_run(cfg, "x; rm -rf y")
    with pytest.raises(ValueError, match="bad run_id"):
        sync(cfg, "x; rm -rf y", tmp_path / "work")
    assert (root / "y").read_text(encoding="utf-8") == "canary"


def test_sync_of_a_run_never_started_returns_an_error(tmp_path):
    cfg = make_cfg(tmp_path)
    result = sync(cfg, "ghost1", tmp_path / "work")
    assert result["copied"] == 0
    assert result["error"].strip() != ""


def test_load_config_missing_file_names_path_and_fields(tmp_path):
    import pytest

    from kullback.trainer.remote import load_config

    with pytest.raises(FileNotFoundError, match="remote.json"):
        load_config(tmp_path / "work")
    try:
        load_config(tmp_path / "work")
    except FileNotFoundError as exc:
        assert "host" in str(exc)
        assert "root" in str(exc)


def test_start_run_marks_failed_when_the_child_dies_at_once(tmp_path):
    cfg = make_cfg(tmp_path)
    root = tmp_path / "remote"
    (root / "boom.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    got = start_run(cfg, "dead1", "boom.py", workdir=tmp_path / "work")
    assert got["started"] is False
    assert got["reason"] == "exited at once"
    assert any("RuntimeError" in line for line in got["log_tail"])
    assert len(got["log_tail"]) <= 20
    state = json.loads((root / "runs" / "dead1" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"


def test_start_run_keeps_a_final_state_written_before_exit(tmp_path):
    cfg = make_cfg(tmp_path)
    root = tmp_path / "remote"
    (root / "fin.py").write_text(
        "import json\n"
        "json.dump({'status': 'finished', 'step': 5},\n"
        "          open('runs/fin1/state.json', 'w'))\n",
        encoding="utf-8")
    got = start_run(cfg, "fin1", "fin.py", workdir=tmp_path / "work")
    assert got["started"] is True
    assert got["exited"] is True and got["status"] == "finished"
    state = json.loads((root / "runs" / "fin1" / "state.json").read_text(encoding="utf-8"))
    assert state == {"status": "finished", "step": 5}


def test_second_start_of_a_run_id_is_refused_without_touching_its_state(tmp_path):
    cfg = make_cfg(tmp_path)
    root = tmp_path / "remote"
    write_sleeper(root, "once1")
    try:
        first = start_run(cfg, "once1", "hello.py", workdir=tmp_path / "work")
        assert first["started"] is True
        before = (root / "runs" / "once1" / "state.json").read_text(encoding="utf-8")
        second = start_run(cfg, "once1", "hello.py", workdir=tmp_path / "work")
        assert second["started"] is False
        assert second["reason"] == "run once1 already exists; start a new run id"
        assert (root / "runs" / "once1" / "state.json").read_text(encoding="utf-8") == before
    finally:
        stop_run(cfg, "once1")


def test_start_with_a_missing_python_leaves_state_failed(tmp_path):
    root = tmp_path / "remote"
    root.mkdir(parents=True, exist_ok=True)
    cfg = RemoteConfig(host="local", root=str(root), python="/nonexistent/python-for-test")
    got = start_run(cfg, "badpy1", "hello.py", workdir=tmp_path / "work")
    assert got["started"] is False
    state = json.loads((root / "runs" / "badpy1" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"


def test_start_marks_failed_when_the_launch_command_fails(tmp_path):
    cfg = make_cfg(tmp_path)
    root = tmp_path / "remote"
    rundir = root / "runs" / "nolaunch1"
    (rundir / "pid").mkdir(parents=True)
    got = start_run(cfg, "nolaunch1", "missing.py", workdir=tmp_path / "work")
    assert got["started"] is False
    state = json.loads((rundir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"


def test_start_reports_a_script_failed_state_as_not_started(tmp_path):
    cfg = make_cfg(tmp_path)
    root = tmp_path / "remote"
    (root / "fail.py").write_text(
        "import json\n"
        "json.dump({'status': 'failed', 'step': 2},\n"
        "          open('runs/fail1/state.json', 'w'))\n",
        encoding="utf-8")
    got = start_run(cfg, "fail1", "fail.py", workdir=tmp_path / "work")
    assert got["started"] is False
    assert got["reason"] == "exited at once" and got["status"] == "failed"
    state = json.loads((root / "runs" / "fail1" / "state.json").read_text(encoding="utf-8"))
    assert state == {"status": "failed", "step": 2}
