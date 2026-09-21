from __future__ import annotations

import json
import math
import os
import re
import secrets
import selectors
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Callable


class WorldError(Exception):
    pass


class ImagePinError(WorldError):
    pass


class LimitError(WorldError):
    pass


class DockerUnavailable(WorldError):
    pass


class LaunchFailure(DockerUnavailable):
    pass


class OwnershipError(WorldError):
    pass


class StateError(WorldError):
    pass


class UnresolvedError(WorldError):
    pass


class CleanupError(WorldError):
    def __init__(self, message: str, receipt=None):
        super().__init__(message)
        self.receipt = receipt


class ExecutorTimeout(Exception):
    def __init__(self, stdout: bytes = b"", stderr: bytes = b""):
        super().__init__("executor timeout")
        self.stdout = stdout
        self.stderr = stderr


class OutputLimitExceeded(Exception):
    def __init__(self, stdout: bytes = b"", stderr: bytes = b""):
        super().__init__("output limit exceeded")
        self.stdout = stdout
        self.stderr = stderr


class CaptureIncomplete(Exception):
    def __init__(self, stdout: bytes = b"", stderr: bytes = b""):
        super().__init__("capture incomplete")
        self.stdout = stdout
        self.stderr = stderr


@dataclass
class ExecutorResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass
class StepReceipt:
    stdout: bytes
    stderr: bytes
    exit_code: int
    timed_out: bool
    truncated: bool


OWNER_LABEL = "kullback.owner=container_world"
OWNER_KEY = "kullback.owner"
OWNER_VALUE = "container_world"
WORLD_KEY = "kullback.world"
USER_SPEC = "65534:65534"
WORKSPACE_TMPFS = "/workspace:rw,size=64m,mode=1777"
CONTROL_LIMIT = 8192
GRACE_S = 2.0
_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_IMGID_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


def _check_image_name(name: str) -> None:
    if not name or "@" in name or name.startswith("-"):
        raise ImagePinError("image name is invalid")
    if ":latest" in name:
        raise ImagePinError("image must not use the latest tag")


def _check_image_digest(digest: str) -> None:
    if len(digest) != 64:
        raise ImagePinError("digest must be 64 hex characters")
    for ch in digest:
        if ch not in "0123456789abcdefABCDEF":
            raise ImagePinError("digest must be hex")


def _validate_image(image: object) -> None:
    if not isinstance(image, str) or not image:
        raise ImagePinError("image must be a non-empty string")
    for bad in ("\x00", "\n", "\r", " ", "\t"):
        if bad in image:
            raise ImagePinError("image contains forbidden whitespace")
    if image.startswith("-"):
        raise ImagePinError("image must not look like a CLI option")
    if "://" in image:
        raise ImagePinError("image must not be a URL")
    parts = image.split("@sha256:")
    if len(parts) != 2:
        raise ImagePinError("image must be pinned as name@sha256:<digest>")
    _check_image_name(parts[0])
    _check_image_digest(parts[1])


def _split_image(image: str) -> tuple[str, str]:
    parts = image.split("@sha256:")
    return (parts[0], parts[1].lower())


def _validate_image_metadata(meta: object) -> None:
    if not isinstance(meta, dict):
        raise ImagePinError("image metadata must be a dict")
    cfg = meta.get("Config")
    if not isinstance(cfg, dict):
        raise ImagePinError("image metadata Config must be a dict")
    vols = cfg.get("Volumes")
    if vols is None:
        return
    if not isinstance(vols, dict):
        raise ImagePinError("image Volumes must be a dict")
    if len(vols) != 0:
        raise ImagePinError("image declares volumes")


def _check_repo_digests(raw: str, image: str) -> None:
    try:
        items = json.loads(raw)
    except Exception as exc:
        raise ImagePinError("image RepoDigests unreadable") from exc
    if not isinstance(items, list):
        raise ImagePinError("image RepoDigests has bad shape")
    if image not in items:
        raise ImagePinError("image identity mismatch")


def _check_volumes_json(raw: str) -> None:
    try:
        vols = json.loads(raw)
    except Exception as exc:
        raise ImagePinError("image Volumes unreadable") from exc
    if vols is None:
        return
    if isinstance(vols, dict) and len(vols) == 0:
        return
    raise ImagePinError("image declares volumes")


def _check_float(name: str, value: object, lo: float, hi: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LimitError(name + " must be a number")
    if not math.isfinite(value) or not (lo < value <= hi):
        raise LimitError(name + " out of range")


def _check_int(name: str, value: object, lo: int, hi: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LimitError(name + " must be an integer")
    if not (lo <= value <= hi):
        raise LimitError(name + " out of range")


def _validate_limits(timeout_s, output_limit_bytes, memory_mb, cpus, pids_limit) -> None:
    _check_float("timeout_s", timeout_s, 0, 3600)
    _check_int("output_limit_bytes", output_limit_bytes, 1, 16777216)
    _check_int("memory_mb", memory_mb, 16, 8192)
    _check_float("cpus", cpus, 0, 16)
    _check_int("pids_limit", pids_limit, 1, 4096)


def _check_docker_string(docker: object) -> str:
    if not isinstance(docker, str) or not docker:
        raise DockerUnavailable("docker executable must be a non-empty string")
    for bad in ("\x00", "\n", "\r", " ", "\t"):
        if bad in docker:
            raise DockerUnavailable("docker executable is invalid")
    if docker.startswith("-"):
        raise DockerUnavailable("docker executable must not look like an option")
    return docker


def _strict_docker_path(path: str) -> str:
    if not os.path.isfile(path):
        raise DockerUnavailable("docker executable not found")
    if not os.access(path, os.X_OK):
        raise DockerUnavailable("docker executable not executable")
    return path


def _resolve_docker(docker: object, allow_missing: bool) -> str:
    if docker is None:
        found = shutil.which("docker")
        if not found:
            raise DockerUnavailable("docker executable not found on PATH")
        return found
    clean = _check_docker_string(docker)
    if "/" not in clean:
        found = shutil.which(clean)
        if not found:
            raise DockerUnavailable("docker executable not found on PATH")
        return found
    if allow_missing:
        return clean
    return _strict_docker_path(clean)


def _validate_result(res: object) -> ExecutorResult:
    if not isinstance(res, ExecutorResult):
        raise WorldError("executor returned unexpected result")
    if isinstance(res.returncode, bool) or not isinstance(res.returncode, int):
        raise WorldError("executor returncode has bad type")
    if not isinstance(res.stdout, (bytes, bytearray)) or not isinstance(res.stderr, (bytes, bytearray)):
        raise WorldError("executor output has bad type")
    return ExecutorResult(res.returncode, bytes(res.stdout), bytes(res.stderr))


def _is_valid_id(cid: object) -> bool:
    return isinstance(cid, str) and _ID_RE.match(cid) is not None


def _trim_combined(out: bytes, err: bytes, limit: int) -> tuple[bytes, bytes]:
    total = len(out) + len(err)
    if total <= limit:
        return (out, err)
    excess = total - limit
    if len(err) >= excess:
        return (out, err[: len(err) - excess])
    excess = excess - len(err)
    return (out[: len(out) - excess], b"")


def _kill_tree(proc) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _set_nonblocking(proc) -> None:
    for stream in (proc.stdout, proc.stderr):
        try:
            os.set_blocking(stream.fileno(), False)
        except OSError:
            pass
        except ValueError:
            pass


def _register_pipes(sel, proc) -> None:
    for stream, tag in ((proc.stdout, "out"), (proc.stderr, "err")):
        try:
            sel.register(stream, selectors.EVENT_READ, data=tag)
        except OSError:
            pass
        except ValueError:
            pass
        except KeyError:
            pass


def _store_chunk(buf: bytearray, data: bytes, total, limit: int, proc) -> bool:
    remaining = limit - total[0]
    if remaining <= 0:
        _kill_tree(proc)
        return True
    if len(data) > remaining:
        buf.extend(data[:remaining])
        total[0] = limit
        _kill_tree(proc)
        return True
    buf.extend(data)
    total[0] = total[0] + len(data)
    return False


def _forget(sel, fobj) -> None:
    try:
        sel.unregister(fobj)
    except Exception:
        pass


def _read_fd(sel, fobj, tag: str, out: bytearray, err: bytearray, total, limit: int, proc) -> tuple[bool, bool]:
    try:
        fd = fobj.fileno()
    except ValueError:
        _forget(sel, fobj)
        return (False, True)
    try:
        data = os.read(fd, 8192)
    except BlockingIOError:
        return (False, False)
    except OSError:
        _forget(sel, fobj)
        return (False, True)
    if data == b"":
        _forget(sel, fobj)
        return (False, False)
    buf = out if tag == "out" else err
    if _store_chunk(buf, data, total, limit, proc):
        return (True, False)
    return (False, False)


def _pump_once(sel, out: bytearray, err: bytearray, total, limit: int, proc, wait_s: float) -> tuple[bool, bool]:
    try:
        events = sel.select(timeout=wait_s)
    except OSError:
        return (False, True)
    overflow = False
    incomplete = False
    for key, _mask in events:
        ov, inc = _read_fd(sel, key.fileobj, key.data, out, err, total, limit, proc)
        overflow = overflow or ov
        incomplete = incomplete or inc
    return (overflow, incomplete)


def _drain_available(sel, out: bytearray, err: bytearray, total, limit: int, proc) -> tuple[bool, bool]:
    overflow = False
    incomplete = False
    for _i in range(20):
        if not sel.get_map():
            break
        try:
            events = sel.select(timeout=0)
        except OSError:
            incomplete = True
            break
        if not events:
            break
        ov, inc = _pump_once(sel, out, err, total, limit, proc, 0)
        overflow = overflow or ov
        incomplete = incomplete or inc
        if overflow:
            break
    return (overflow, incomplete)


def _wait_reap(proc, wait_s: float) -> None:
    try:
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        pass
    except OSError:
        pass


def _close_same_thread(sel, proc) -> None:
    try:
        if sel is not None:
            sel.close()
    except Exception:
        pass
    for stream in (proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass
        except ValueError:
            pass


def _kill_if_live(proc) -> bool:
    try:
        if proc.poll() is not None:
            return False
    except OSError:
        return False
    _kill_tree(proc)
    return True


def _reap_owned(proc) -> None:
    try:
        if proc.poll() is None:
            _kill_tree(proc)
        _wait_reap(proc, GRACE_S)
    except Exception:
        pass
    except BaseException:
        try:
            if proc.returncode is None:
                _kill_tree(proc)
        except Exception:
            pass
        try:
            _wait_reap(proc, GRACE_S)
        except Exception:
            pass
        raise


def _run_until_exit(proc, sel, out: bytearray, err: bytearray, total, limit: int, deadline: float) -> tuple[int, bool]:
    incomplete = False
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            return (0, incomplete)
        poll = proc.poll()
        if poll is not None:
            return (poll, incomplete)
        wait_s = remaining if remaining < 0.2 else 0.2
        ov, inc = _pump_once(sel, out, err, total, limit, proc, wait_s)
        incomplete = incomplete or inc
        if ov:
            return (-99, incomplete)


def _drain_after_exit(proc, sel, out: bytearray, err: bytearray, total, limit: int, exit_at: float, incomplete: bool) -> tuple[bool, bool]:
    overflow = False
    while True:
        if not sel.get_map():
            return (overflow, incomplete)
        now = time.monotonic()
        if now - exit_at > GRACE_S:
            return (overflow, True)
        ov, inc = _pump_once(sel, out, err, total, limit, proc, 0.2)
        overflow = overflow or ov
        incomplete = incomplete or inc
        if overflow:
            return (overflow, incomplete)


def _finish_run(proc, sel, out: bytearray, err: bytearray, total, limit: int, rc: int, incomplete: bool, deadline: float) -> ExecutorResult:
    if rc == 0 and time.monotonic() >= deadline:
        _kill_if_live(proc)
        _wait_reap(proc, GRACE_S)
        _drain_available(sel, out, err, total, limit, proc)
        bout, berr = _trim_combined(bytes(out), bytes(err), limit)
        raise ExecutorTimeout(bout, berr)
    if rc == -99:
        _kill_if_live(proc)
        _wait_reap(proc, GRACE_S)
        _drain_available(sel, out, err, total, limit, proc)
        bout, berr = _trim_combined(bytes(out), bytes(err), limit)
        raise OutputLimitExceeded(bout, berr)
    if proc.poll() is None:
        _kill_tree(proc)
        _wait_reap(proc, GRACE_S)
        _drain_available(sel, out, err, total, limit, proc)
        bout, berr = _trim_combined(bytes(out), bytes(err), limit)
        raise ExecutorTimeout(bout, berr)
    exit_at = time.monotonic()
    ov, inc = _drain_after_exit(proc, sel, out, err, total, limit, exit_at, incomplete)
    if ov:
        _wait_reap(proc, 0)
        bout, berr = _trim_combined(bytes(out), bytes(err), limit)
        raise OutputLimitExceeded(bout, berr)
    if inc:
        _wait_reap(proc, 0)
        bout, berr = _trim_combined(bytes(out), bytes(err), limit)
        raise CaptureIncomplete(bout, berr)
    bout = bytes(out)
    berr = bytes(err)
    if len(bout) + len(berr) > limit:
        bout, berr = _trim_combined(bout, berr, limit)
        raise OutputLimitExceeded(bout, berr)
    return ExecutorResult(proc.returncode, bout, berr)


def _bounded_run(argv: list, timeout: float, limit: int) -> ExecutorResult:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise LimitError("limit must be a positive integer")
    try:
        proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise LaunchFailure(str(exc)) from exc
    except OSError as exc:
        raise LaunchFailure(str(exc)) from exc
    sel = None
    saw_finish = False
    try:
        _set_nonblocking(proc)
        sel = selectors.DefaultSelector()
        _register_pipes(sel, proc)
        if proc.stdout is None or proc.stderr is None or len(sel.get_map()) != 2:
            raise CaptureIncomplete(b"", b"")
        out = bytearray()
        err = bytearray()
        total = [0]
        start = time.monotonic()
        deadline = start + timeout
        rc, incomplete = _run_until_exit(proc, sel, out, err, total, limit, deadline)
        saw_finish = True
        return _finish_run(proc, sel, out, err, total, limit, rc, incomplete, deadline)
    except (ExecutorTimeout, OutputLimitExceeded, CaptureIncomplete):
        if not saw_finish:
            _reap_owned(proc)
        raise
    except BaseException:
        _reap_owned(proc)
        raise
    finally:
        _close_same_thread(sel, proc)


class ContainerWorld:
    def __init__(
        self,
        image: str,
        timeout_s: float = 30.0,
        output_limit_bytes: int = 65536,
        memory_mb: int = 512,
        cpus: float = 1.0,
        pids_limit: int = 128,
        docker=None,
        executor: Callable | None = None,
        image_metadata=None,
    ):
        _validate_image(image)
        _validate_limits(timeout_s, output_limit_bytes, memory_mb, cpus, pids_limit)
        _validate_image_metadata(image_metadata)
        self.image = image
        self.timeout_s = timeout_s
        self.output_limit_bytes = output_limit_bytes
        self.memory_mb = memory_mb
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.docker = _resolve_docker(docker, executor is not None)
        self._executor = executor
        self._image_metadata = image_metadata
        self._nonce = secrets.token_hex(8)
        self._name = "kullback-cw-" + self._nonce
        self._container_id: str | None = None
        self._active = False
        self._unresolved = False

    def _call(self, argv: list, timeout: float, limit: int) -> ExecutorResult:
        if self._executor is not None:
            return _validate_result(self._executor(list(argv), timeout))
        return _bounded_run(list(argv), timeout, limit)

    def _control(self, argv: list, timeout: float) -> ExecutorResult:
        try:
            res = self._call(argv, timeout, CONTROL_LIMIT)
        except FileNotFoundError as exc:
            raise DockerUnavailable(str(exc)) from exc
        except OSError as exc:
            raise DockerUnavailable(str(exc)) from exc
        if len(res.stdout) + len(res.stderr) > CONTROL_LIMIT:
            raise WorldError("control output exceeds bound")
        return res

    def _exec_call(self, argv: list, timeout: float) -> ExecutorResult:
        try:
            return self._call(argv, timeout, self.output_limit_bytes)
        except FileNotFoundError as exc:
            raise DockerUnavailable(str(exc)) from exc
        except OSError as exc:
            raise DockerUnavailable(str(exc)) from exc

    def _create_argv(self) -> list:
        return [
            self.docker, "create",
            "--network", "none",
            "--pull", "never",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", USER_SPEC,
            "--memory", str(self.memory_mb) + "m",
            "--cpus", str(self.cpus),
            "--pids-limit", str(self.pids_limit),
            "--tmpfs", WORKSPACE_TMPFS,
            "--entrypoint", "",
            "--workdir", "/workspace",
            "--no-healthcheck",
            "--label", WORLD_KEY + "=" + self._nonce,
            "--label", OWNER_LABEL,
            "--name", self._name,
            self.image, "sleep", "infinity",
        ]

    def _image_argv(self) -> list:
        return [
            self.docker, "image", "inspect",
            "--format", "{{.Id}}|{{json .RepoDigests}}|{{json .Config.Volumes}}",
            self.image,
        ]

    def _inspect_image_raw(self):
        try:
            return self._control(self._image_argv(), self.timeout_s)
        except DockerUnavailable:
            raise
        except (ExecutorTimeout, OutputLimitExceeded, CaptureIncomplete, WorldError) as exc:
            raise UnresolvedError("image inspection is ambiguous") from exc

    def _check_image_fields(self, text: str) -> None:
        first, sep, rest = text.partition("|")
        if not sep:
            raise ImagePinError("image inspect has bad shape")
        second, sep2, third = rest.partition("|")
        if not sep2:
            raise ImagePinError("image inspect has bad shape")
        first = first.strip()
        second = second.strip()
        third = third.strip()
        if not _IMGID_RE.match(first):
            raise ImagePinError("image Id has bad shape")
        _check_repo_digests(second, self.image)
        _check_volumes_json(third)

    def _verify_image(self) -> None:
        res = self._inspect_image_raw()
        if res.returncode != 0:
            try:
                detail = res.stderr.decode("utf-8", errors="replace")[-500:]
            except Exception:
                detail = ""
            if "no such image" in detail.lower():
                raise DockerUnavailable(detail)
            raise UnresolvedError("image inspection failed")
        try:
            text = res.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ImagePinError("image inspect is not text") from exc
        if "\x00" in text:
            raise ImagePinError("image inspect has bad shape")
        self._check_image_fields(text.strip())

    def _parse_inspect(self, text: str, want: str) -> tuple[str, str, str]:
        got_id, sep, rest = text.partition("|")
        if not sep:
            raise OwnershipError("inspect output has bad shape")
        got_world, sep2, got_owner = rest.partition("|")
        if not sep2:
            raise OwnershipError("inspect output has bad shape")
        got_id = got_id.strip()
        got_world = got_world.strip()
        got_owner = got_owner.strip()
        if not _is_valid_id(got_id):
            raise OwnershipError("inspect id has bad shape")
        if got_id != want and want != self._name:
            raise OwnershipError("container ownership mismatch")
        return (got_id, got_world, got_owner)

    def _check_owner_labels(self, got_world: str, got_owner: str) -> None:
        if got_world != self._nonce or got_owner != OWNER_VALUE:
            raise OwnershipError("container ownership mismatch")

    def _inspect_owner(self, cid: str) -> None:
        argv = [self.docker, "inspect", "--format", '{{.Id}}|{{ index .Config.Labels "kullback.world" }}|{{ index .Config.Labels "kullback.owner" }}', cid]
        res = self._control(argv, self.timeout_s)
        if res.returncode != 0:
            raise OwnershipError("inspect failed for owned container")
        try:
            text = res.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OwnershipError("inspect output is not text") from exc
        if "\x00" in text:
            raise OwnershipError("inspect output has bad shape")
        got_id, got_world, got_owner = self._parse_inspect(text.strip(), cid)
        if got_id != cid:
            raise OwnershipError("container ownership mismatch")
        self._check_owner_labels(got_world, got_owner)

    def _remove_owned(self) -> None:
        cid = self._container_id
        if cid is None:
            return
        self._active = False
        self._inspect_owner(cid)
        res = self._control([self.docker, "rm", "-f", cid], self.timeout_s)
        if res.returncode != 0:
            raise WorldError("removal failed for owned container")
        self._container_id = None
        self._active = False

    def _fetch_unresolved_text(self) -> str:
        argv = [self.docker, "inspect", "--format", '{{.Id}}|{{ index .Config.Labels "kullback.world" }}|{{ index .Config.Labels "kullback.owner" }}', self._name]
        try:
            res = self._control(argv, self.timeout_s)
        except DockerUnavailable:
            raise
        except (ExecutorTimeout, OutputLimitExceeded, CaptureIncomplete, WorldError) as exc:
            raise UnresolvedError("unresolved inspect is ambiguous") from exc
        if res.returncode != 0:
            raise UnresolvedError("unresolved inspect did not prove absence")
        try:
            text = res.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnresolvedError("unresolved inspect is not text") from exc
        if "\x00" in text:
            raise UnresolvedError("unresolved inspect has bad shape")
        return text.strip()

    def _remove_unresolved_id(self, got_id: str) -> None:
        try:
            rm = self._control([self.docker, "rm", "-f", got_id], self.timeout_s)
        except DockerUnavailable:
            raise
        except (ExecutorTimeout, OutputLimitExceeded, CaptureIncomplete, WorldError) as exc:
            raise UnresolvedError("unresolved cleanup is ambiguous") from exc
        if rm.returncode != 0:
            raise UnresolvedError("unresolved cleanup failed")

    def _recover_unresolved(self) -> bool:
        text = self._fetch_unresolved_text()
        try:
            got_id, got_world, got_owner = self._parse_inspect(text, self._name)
        except OwnershipError as exc:
            raise UnresolvedError("unresolved ownership cannot be established") from exc
        try:
            self._check_owner_labels(got_world, got_owner)
        except OwnershipError as exc:
            raise UnresolvedError("unresolved labels do not match") from exc
        self._remove_unresolved_id(got_id)
        self._unresolved = False
        self._container_id = None
        self._active = False
        return True

    def _fail_unresolved(self, message: str) -> UnresolvedError:
        self._active = False
        self._unresolved = True
        return UnresolvedError(message)

    def _cleanup_after_output(self, out: bytes, err: bytes, timed_out: bool, truncated: bool) -> StepReceipt:
        receipt = StepReceipt(out, err, -1, timed_out, truncated)
        try:
            self._remove_owned()
        except (WorldError, ExecutorTimeout, OutputLimitExceeded, CaptureIncomplete) as exc:
            raise CleanupError("cleanup failed after bounded output", receipt) from exc
        return receipt

    def _create_once(self) -> str:
        self._active = False
        self._unresolved = True
        try:
            res = self._control(self._create_argv(), self.timeout_s)
        except LaunchFailure:
            self._unresolved = False
            raise
        except DockerUnavailable as exc:
            raise self._fail_unresolved("create dispatch ambiguous") from exc
        except (ExecutorTimeout, OutputLimitExceeded, CaptureIncomplete, WorldError) as exc:
            raise self._fail_unresolved("create did not confirm ownership") from exc
        except BaseException:
            self._unresolved = True
            raise
        return self._accept_create_result(res)

    def _accept_create_result(self, res: ExecutorResult) -> str:
        if res.returncode != 0:
            try:
                raw = bytes(res.stderr)
                detail = raw.decode("utf-8", errors="replace")[-500:]
            except Exception:
                raw = b""
                detail = ""
            if b"no such image" in raw.lower():
                self._unresolved = False
                raise DockerUnavailable(detail)
            raise self._fail_unresolved("create failed ambiguously")
        try:
            cid = res.stdout.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise self._fail_unresolved("create output is not text") from exc
        if "\x00" in cid or not _is_valid_id(cid):
            raise self._fail_unresolved("create output is malformed")
        self._container_id = cid
        self._active = False
        return cid

    def _verify_and_start(self, cid: str) -> str:
        try:
            self._inspect_owner(cid)
        except DockerUnavailable:
            raise
        except (OwnershipError, ExecutorTimeout, OutputLimitExceeded, CaptureIncomplete, WorldError) as exc:
            raise self._fail_unresolved("create ownership cannot be established") from exc
        try:
            started = self._control([self.docker, "start", cid], self.timeout_s)
        except (ExecutorTimeout, OutputLimitExceeded, CaptureIncomplete, WorldError) as exc:
            raise CleanupError("start did not confirm", None) from exc
        if started.returncode != 0:
            try:
                self._remove_owned()
            except (WorldError, ExecutorTimeout, OutputLimitExceeded, CaptureIncomplete) as exc:
                raise CleanupError("start failed and cleanup failed", None) from exc
            try:
                detail = started.stderr.decode("utf-8", errors="replace")[-500:]
            except Exception:
                detail = ""
            self._unresolved = False
            raise DockerUnavailable(detail)
        self._active = True
        self._unresolved = False
        return cid

    def reset(self) -> str:
        if self._unresolved and self._container_id is None:
            raise UnresolvedError("creation is unresolved")
        if self._container_id is not None:
            self._remove_owned()
        self._verify_image()
        cid = self._create_once()
        return self._verify_and_start(cid)

    def step(self, command: str) -> StepReceipt:
        if self._unresolved:
            raise UnresolvedError("world creation is unresolved")
        if not self._active or self._container_id is None:
            raise StateError("world needs reset before step")
        if not isinstance(command, str):
            raise WorldError("command must be a string")
        try:
            res = self._exec_call([self.docker, "exec", self._container_id, "sh", "-c", command], self.timeout_s)
        except ExecutorTimeout as exc:
            out, err = _trim_combined(bytes(exc.stdout), bytes(exc.stderr), self.output_limit_bytes)
            return self._cleanup_after_output(out, err, True, False)
        except (OutputLimitExceeded, CaptureIncomplete) as exc:
            out, err = _trim_combined(bytes(exc.stdout), bytes(exc.stderr), self.output_limit_bytes)
            return self._cleanup_after_output(out, err, False, True)
        except BaseException as intr:
            self._active = False
            try:
                self._remove_owned()
            except BaseException as clean_exc:
                raise intr from clean_exc
            raise
        if len(res.stdout) + len(res.stderr) > self.output_limit_bytes:
            out, err = _trim_combined(bytes(res.stdout), bytes(res.stderr), self.output_limit_bytes)
            return self._cleanup_after_output(out, err, False, True)
        return StepReceipt(bytes(res.stdout), bytes(res.stderr), res.returncode, False, False)

    def close(self) -> bool:
        if self._unresolved:
            self._recover_unresolved()
            return True
        if self._container_id is None:
            return False
        self._remove_owned()
        return True
