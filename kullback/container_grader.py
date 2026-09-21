"""Grade an exported end state inside a fresh, locked-down container.

This module never takes the Candidate's World at all, only the exported bytes:
the Candidate's end state travels as an opaque tar archive that is unpacked only
inside the fresh grader container. The tar bytes are untrusted and are never
parsed or unpacked on the host. The World forbids mounts and the executor has
no stdin, so bytes go in through step() in base64 chunks.

A state symlink or member path aimed at a known grader path could read the
answers, so the grader directory is unpredictable: /workspace/grader-<token>
with a token drawn after the Candidate's world is gone. The state is fed and
unpacked completely before the grader directory is created, so nothing in the
state can be written through to a directory that does not exist yet.

The grade command runs with STATE_DIR and GRADER_DIR set in its environment,
so it refers to the two inputs through those variables instead of fixed paths.

Empty inputs still create their directory and skip the chunk and unpack steps.

max_input_bytes defaults to 16777216, matching the export ceiling, since a
state can never be larger than what export allows.

Known cost: one exec per chunk, to be replaced by executor stdin in the speed pass.
"""

from __future__ import annotations

import base64
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass


class GradeError(Exception):
    pass


@dataclass
class GradeReceipt:
    exit_code: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    truncated: bool
    passed: bool


_TOKEN_RE = re.compile(r"[0-9a-f]{8,64}")


def _check_chunk_bytes(chunk_bytes: object) -> int:
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int):
        raise GradeError("chunk_bytes must be an integer")
    if chunk_bytes <= 0 or chunk_bytes % 3 != 0:
        raise GradeError("chunk_bytes must be a positive multiple of 3")
    return chunk_bytes


def _check_token(token: object) -> str:
    if token is None:
        return secrets.token_hex(16)
    if not isinstance(token, str) or _TOKEN_RE.fullmatch(token) is None:
        raise GradeError("token must match [0-9a-f]{8,64}")
    return token


def _validate_inputs(
    state_tar: object,
    grader_tar: object,
    command: object,
    chunk_bytes: object,
    max_input_bytes: int,
    token: object,
) -> tuple[int, str]:
    if not isinstance(state_tar, bytes):
        raise GradeError("state_tar must be bytes")
    if not isinstance(grader_tar, bytes):
        raise GradeError("grader_tar must be bytes")
    if not isinstance(command, str) or command == "":
        raise GradeError("command must be a non-empty string")
    size = _check_chunk_bytes(chunk_bytes)
    resolved = _check_token(token)
    if len(state_tar) > max_input_bytes:
        raise GradeError("state_tar exceeds max_input_bytes")
    if len(grader_tar) > max_input_bytes:
        raise GradeError("grader_tar exceeds max_input_bytes")
    return (size, resolved)


def _require_ok(receipt, stage: str) -> None:
    if receipt.timed_out:
        raise GradeError(stage + " timed out: no verdict")
    if receipt.truncated:
        raise GradeError(stage + " truncated output: no verdict")
    if receipt.exit_code != 0:
        raise GradeError(stage + " exited with code " + str(receipt.exit_code) + ": no verdict")


def _feed_input(world, label: str, dirname: str, data: bytes, chunk_bytes: int) -> None:
    _require_ok(world.step("mkdir /workspace/" + dirname), label + " mkdir")
    if len(data) == 0:
        return
    for offset in range(0, len(data), chunk_bytes):
        text = base64.b64encode(data[offset : offset + chunk_bytes]).decode("ascii")
        _require_ok(
            world.step("printf %s '" + text + "' >> /workspace/." + dirname + ".b64"),
            label + " chunk " + str(offset // chunk_bytes),
        )
    _require_ok(
        world.step(
            "base64 -d /workspace/." + dirname + ".b64 | tar -xf - -C /workspace/" + dirname + " && rm /workspace/." + dirname + ".b64"
        ),
        label + " unpack",
    )


def grade(
    make_world: Callable,
    *,
    state_tar: bytes,
    grader_tar: bytes,
    command: str,
    chunk_bytes: int = 48000,
    max_input_bytes: int = 16_777_216,
    token: str | None = None,
) -> GradeReceipt:
    size, resolved = _validate_inputs(state_tar, grader_tar, command, chunk_bytes, max_input_bytes, token)
    grader_dir = "grader-" + resolved
    world = make_world()
    try:
        world.reset()
        _feed_input(world, "state", "state", state_tar, size)
        _feed_input(world, "grader", grader_dir, grader_tar, size)
        receipt = world.step(
            "STATE_DIR=/workspace/state GRADER_DIR=/workspace/" + grader_dir + "; export STATE_DIR GRADER_DIR; " + command
        )
    finally:
        world.close()
    passed = receipt.exit_code == 0 and not receipt.timed_out and not receipt.truncated
    return GradeReceipt(
        receipt.exit_code,
        bytes(receipt.stdout),
        bytes(receipt.stderr),
        receipt.timed_out,
        receipt.truncated,
        passed,
    )
