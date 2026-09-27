"""Remote backends for the training agent tools: shell, GPU status, run control, sync."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time

from pydantic import BaseModel, ConfigDict, Field, field_validator

RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")

def _check_run_id(run_id: str) -> None:
    """Refuse a run id that could escape its shell command."""
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"bad run_id {run_id!r}")

NVIDIA_QUERY = (
    "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
    "utilization.gpu,temperature.gpu --format=csv,noheader,nounits"
)

TAIL_CHARS = 20_000


class RemoteConfig(BaseModel):
    """How to reach the GPU machine; host local runs on this machine."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(min_length=1)
    root: str = Field(min_length=1)
    python: str = "python3"
    timeout_s: int = 120

    @field_validator("root")
    @classmethod
    def _root_absolute(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError(f"root must be absolute, got {value!r}")
        return value


def load_config(workdir) -> RemoteConfig:
    """Read <workdir>/training/remote.json into a RemoteConfig."""
    path = os.path.join(str(workdir), "training", "remote.json")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"missing {path}: create it with host, root, python, timeout_s"
        ) from None
    return RemoteConfig.model_validate(data)


def _argv(cfg: RemoteConfig, command: str) -> list[str]:
    quoted = shlex.quote(cfg.root)
    inner = f"cd {quoted} && {command}"
    if cfg.host == "local":
        return ["bash", "-lc", inner]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        cfg.host,
        inner,
    ]


def _tail(text: str) -> str:
    if len(text) > TAIL_CHARS:
        return text[-TAIL_CHARS:]
    return text


def _log_shell(workdir, cfg: RemoteConfig, command: str, returncode: int, seconds: float) -> None:
    logdir = os.path.join(str(workdir), "training")
    os.makedirs(logdir, exist_ok=True)
    line = {
        "time": time.time(),
        "host": cfg.host,
        "command": command,
        "returncode": returncode,
        "seconds": seconds,
    }
    with open(os.path.join(logdir, "shell.log"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")


def run(cfg: RemoteConfig, command: str, *, timeout_s=None, workdir=None) -> dict:
    """Run one shell command on the GPU machine; never raise on non-zero exit."""
    limit = cfg.timeout_s if timeout_s is None else timeout_s
    start = time.monotonic()
    try:
        proc = subprocess.run(
            _argv(cfg, command),
            capture_output=True,
            text=True,
            timeout=limit,
        )
        seconds = time.monotonic() - start
        result = {
            "returncode": proc.returncode,
            "stdout": _tail(proc.stdout or ""),
            "stderr": _tail(proc.stderr or ""),
            "seconds": seconds,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        seconds = time.monotonic() - start
        out = exc.stdout
        err = exc.stderr
        result = {
            "returncode": -1,
            "stdout": _tail(out.decode() if isinstance(out, bytes) else (out or "")),
            "stderr": _tail(err.decode() if isinstance(err, bytes) else (err or "")),
            "seconds": seconds,
            "timed_out": True,
        }
    if workdir is not None:
        _log_shell(workdir, cfg, command, result["returncode"], result["seconds"])
    return result


def shell(cfg: RemoteConfig, command: str, *, workdir) -> dict:
    """Run with logging always on."""
    return run(cfg, command, workdir=workdir)


def parse_nvidia_smi(text: str) -> list[dict]:
    """Parse nvidia-smi csv output into one dict per GPU."""
    gpus: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mib": int(float(parts[2])),
                    "memory_total_mib": int(float(parts[3])),
                    "utilization_pct": int(float(parts[4])),
                    "temperature_c": int(float(parts[5])),
                }
            )
        except ValueError:
            continue
    return gpus


def gpu_status(cfg: RemoteConfig, workdir=None) -> dict:
    """Report GPUs on the GPU machine; empty list plus error when nvidia-smi fails."""
    result = run(cfg, NVIDIA_QUERY, workdir=workdir)
    if result["returncode"] != 0:
        detail = (result.get("stderr") or result.get("stdout") or "nvidia-smi failed").strip()
        return {"gpus": [], "error": detail}
    return {"gpus": parse_nvidia_smi(result.get("stdout", ""))}


def _read_pid(cfg: RemoteConfig, run_id: str, workdir=None) -> int | None:
    result = run(cfg, f"cat runs/{run_id}/pid", workdir=workdir)
    if result["returncode"] != 0:
        return None
    try:
        return int(result["stdout"].strip().split()[0])
    except (ValueError, IndexError):
        return None


def _pid_alive(cfg: RemoteConfig, pid: int, workdir=None) -> bool:
    result = run(cfg, f"kill -0 {pid}", workdir=workdir)
    return result["returncode"] == 0


def _write_state(cfg: RemoteConfig, run_id: str, state: dict, workdir=None) -> None:
    payload = json.dumps(state)
    run(
        cfg,
        f"printf '%s' {shlex.quote(payload)} > runs/{run_id}/state.json",
        workdir=workdir,
    )


def _mark_failed(cfg: RemoteConfig, run_id: str, workdir=None) -> None:
    """Mark state.json failed where the launch never left a final state."""
    _write_state(cfg, run_id, {"status": "failed", "step": None, "checkpoint": None,
                               "updated_at": time.time()}, workdir)


def _read_state(cfg: RemoteConfig, run_id: str, workdir=None) -> dict | None:
    """The state the script left in state.json, or None where it is missing or broken."""
    result = run(cfg, f"cat runs/{run_id}/state.json", workdir=workdir)
    if result["returncode"] != 0:
        return None
    try:
        body = json.loads(result["stdout"])
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def start_run(cfg: RemoteConfig, run_id: str, script: str, *, args: str = "", workdir=None) -> dict:
    """Launch script on the GPU machine under runs/<run_id>/; return the pid."""
    try:
        _check_run_id(run_id)
    except ValueError as exc:
        return {"started": False, "reason": str(exc)}
    exists = run(cfg, f"test -f runs/{run_id}/state.json", workdir=workdir)
    if exists["returncode"] == 0:
        return {"started": False, "reason": f"run {run_id} already exists; start a new run id"}
    setup = run(cfg, f"mkdir -p runs/{run_id}", workdir=workdir)
    if setup["returncode"] != 0:
        return {"started": False, "reason": setup["stderr"].strip() or setup["stdout"].strip()}
    _write_state(cfg, run_id, {"status": "running", "step": None, "checkpoint": None,
                               "updated_at": time.time()}, workdir)
    launch = (
        "{ nohup " + cfg.python + " " + script + " " + args
        + " > runs/" + run_id + "/run.log 2>&1 < /dev/null & echo $! > runs/" + run_id + "/pid; }"
    )
    launched = run(cfg, launch, workdir=workdir)
    if launched["returncode"] != 0:
        _mark_failed(cfg, run_id, workdir)
        return {"started": False, "reason": launched["stderr"].strip() or launched["stdout"].strip()}
    pid = _read_pid(cfg, run_id, workdir)
    if pid is None:
        _mark_failed(cfg, run_id, workdir)
        return {"started": False, "reason": f"run {run_id} launched but pid file is missing"}
    time.sleep(1)
    if _pid_alive(cfg, pid, workdir):
        return {"started": True, "run_id": run_id, "pid": pid}
    left = _read_state(cfg, run_id, workdir)
    status = left.get("status") if left is not None else "running"
    if status != "running" and status != "failed":
        return {"started": True, "run_id": run_id, "pid": pid,
                "exited": True, "status": status}
    if status == "running":
        _mark_failed(cfg, run_id, workdir)
    logged = run(cfg, f"cat runs/{run_id}/run.log", workdir=workdir)
    tail = (logged.get("stdout") or "").splitlines()[-20:]
    return {"started": False, "reason": "exited at once", "status": "failed", "log_tail": tail}


def stop_run(cfg: RemoteConfig, run_id: str, workdir=None) -> dict:
    """Stop a running run with TERM then KILL; mark state.json stopped."""
    _check_run_id(run_id)
    pid = _read_pid(cfg, run_id, workdir)
    if pid is None:
        return {"stopped": False, "reason": f"no pid file for run {run_id}"}
    run(cfg, f"kill -TERM {pid}", workdir=workdir)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not _pid_alive(cfg, pid, workdir):
            break
        time.sleep(0.2)
    if _pid_alive(cfg, pid, workdir):
        run(cfg, f"kill -KILL {pid}", workdir=workdir)
    state: dict = {"status": "stopped", "step": None, "checkpoint": None, "updated_at": time.time()}
    current = run(cfg, f"cat runs/{run_id}/state.json", workdir=workdir)
    if current["returncode"] == 0:
        try:
            prior = json.loads(current["stdout"])
            if isinstance(prior, dict):
                prior["status"] = "stopped"
                prior["updated_at"] = time.time()
                state = prior
        except (ValueError, TypeError):
            pass
    _write_state(cfg, run_id, state, workdir)
    return {"stopped": True, "run_id": run_id, "pid": pid}


def run_alive(cfg: RemoteConfig, run_id: str, workdir=None) -> bool:
    """True when the run pid file exists and that pid is alive."""
    _check_run_id(run_id)
    pid = _read_pid(cfg, run_id, workdir)
    if pid is None:
        return False
    return _pid_alive(cfg, pid, workdir)


def _count_files(path: str) -> int:
    total = 0
    for _root, _dirs, files in os.walk(path):
        total += len(files)
    return total


def sync(cfg: RemoteConfig, run_id: str, workdir) -> dict:
    """Copy root/runs/<run_id>/ into <workdir>/training/<run_id>/."""
    _check_run_id(run_id)
    dest = os.path.join(str(workdir), "training", run_id)
    os.makedirs(dest, exist_ok=True)
    start = time.monotonic()
    if cfg.host == "local":
        src = os.path.join(cfg.root, "runs", run_id)
        try:
            shutil.copytree(src, dest, dirs_exist_ok=True)
        except FileNotFoundError as exc:
            return {"run_id": run_id, "copied": 0, "seconds": time.monotonic() - start,
                    "error": str(exc)}
        return {"run_id": run_id, "copied": _count_files(dest), "seconds": time.monotonic() - start}
    source = f"{cfg.host}:{cfg.root}/runs/{run_id}/"
    try:
        proc = subprocess.run(
            ["rsync", "-az", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10", source, dest + "/"],
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {"run_id": run_id, "copied": 0, "seconds": time.monotonic() - start,
                "error": f"rsync timed out: {exc}"}
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "rsync failed").strip()
        return {"run_id": run_id, "copied": 0, "seconds": time.monotonic() - start, "error": detail}
    return {"run_id": run_id, "copied": _count_files(dest), "seconds": time.monotonic() - start}


def push(cfg: RemoteConfig, workdir, subdir: str = "scripts") -> dict:
    """Copy <workdir>/training/<subdir>/ to root/<subdir>/."""
    if not subdir or subdir.startswith("/") or ".." in subdir.split("/"):
        return {"copied": 0, "seconds": 0.0, "error": f"bad subdir {subdir!r}"}
    src = os.path.join(str(workdir), "training", subdir)
    start = time.monotonic()
    if cfg.host == "local":
        dest = os.path.join(cfg.root, subdir)
        try:
            shutil.copytree(src, dest, dirs_exist_ok=True)
        except FileNotFoundError as exc:
            return {"copied": 0, "seconds": time.monotonic() - start,
                    "error": str(exc)}
        return {"copied": _count_files(dest), "seconds": time.monotonic() - start}
    dest = f"{cfg.host}:{cfg.root}/{subdir}/"
    try:
        proc = subprocess.run(
            ["rsync", "-az", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10", src + "/", dest],
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {"copied": 0, "seconds": time.monotonic() - start,
                "error": f"rsync timed out: {exc}"}
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "rsync failed").strip()
        return {"copied": 0, "seconds": time.monotonic() - start, "error": detail}
    return {"copied": _count_files(src), "seconds": time.monotonic() - start}
