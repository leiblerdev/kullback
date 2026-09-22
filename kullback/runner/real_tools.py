"""A tool run for real in a sandbox, beside the routes that imitate one (D262).

The order of preference is real, then compiled code, then the recording, then the
model stand-in. A Real tool owns a container world through an injected factory, so
this module never imports the container backend: any world with reset, step,
export and close conforms. Every world failure is returned as data, never raised.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Protocol


class Receipt(Protocol):
    """What one world step answers: an exit code, both streams, and two fault flags."""

    exit_code: int
    stdout: Any
    stderr: Any
    timed_out: bool
    truncated: bool


class RealWorld(Protocol):
    """The container surface a Real tool needs: reset once, step often, export, close."""

    def reset(self) -> str: ...
    def step(self, command: str) -> Receipt: ...
    def export_workspace(self, limit_bytes: int) -> bytes: ...
    def close(self) -> bool: ...


class RealOutcome(NamedTuple):
    """What one real call did: the result, or the routed error class and message.

    `world_answered` tells whether the world ran the call: True when the command ran
    (any exit code, any output) or when no world was needed at all (a bad batch, which
    is the caller's own business error). False only when the world could not run the
    call: a start or exec failure, a timeout, an unavailable runtime, an export or
    output limit refused. The Router reads this field, never the message text.
    """

    result: Any
    error_class: Optional[str] = None
    error_message: Optional[str] = None
    receipts: Optional[list] = None
    world_answered: bool = True


class ExportLimitError(Exception):
    """A workspace export refused: more bytes than the caller's limit allows (D262)."""


def limit_kept_export(kept: bytes, limit_bytes: int) -> bytes:
    """Kept bytes under the caller's limit, refused like the live export when over (D262).

    Over the limit is the same refusal the open path gives, never a silent truncation
    and never a larger payload.
    """
    if len(kept) > limit_bytes:
        raise ExportLimitError("workspace export exceeded its byte limit")
    return kept


# The container World's own failure names, matched by name so this module never
# imports the backend (D262). A timeout is transient, every other world failure is
# unknown: neither is the customer's answer, and each marks the outcome unanswered
# so the Router ends the Run against the Environment, never the Candidate.
_WORLD_ERROR_NAMES = frozenset({
    "WorldError", "ImagePinError", "LimitError", "DockerUnavailable", "LaunchFailure",
    "OwnershipError", "StateError", "UnresolvedError", "ExportError", "CleanupError",
    "ExecutorTimeout", "OutputLimitExceeded", "CaptureIncomplete", "TimeoutError",
})
_TIMEOUT_ERROR_NAMES = frozenset({"TimeoutError", "ExecutorTimeout"})


def _is_world_error(exc: Exception) -> bool:
    """Whether the failure came from the world: a family name anywhere in its lineage."""
    return any(cls.__name__ in _WORLD_ERROR_NAMES for cls in type(exc).__mro__)


def _raise_if_not_world(exc: Exception) -> None:
    """A failure no world explains is a bug elsewhere, so it propagates after cleanup."""
    if not _is_world_error(exc):
        raise exc


def _close_ignoring_world_error(world: RealWorld) -> None:
    """Release a world whose reset failed; a close from the same family is ignored."""
    try:
        world.close()
    except Exception as exc:
        _raise_if_not_world(exc)


def _error_message(exc: Exception) -> str:
    """The failure in words: its name, with what it said when it said anything."""
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _world_error_class(exc: Exception) -> str:
    """Timeouts may pass on retry, so they are transient; the rest of the family is unknown."""
    return "transient" if type(exc).__name__ in _TIMEOUT_ERROR_NAMES else "unknown"


def _text_of(stream: Any) -> str:
    """One stream as text: bytes decode, anything else arrives as it is."""
    return stream.decode("utf-8", errors="replace") if isinstance(stream, (bytes, bytearray)) else stream


def default_render(args: dict) -> list[str]:
    """The shell batch as the turn carried it: one command per entry's keystrokes (D262).

    Duration stays out: it measures a wait the agent asked for, not part of the command.
    A batch without a commands list, or an entry without its keystrokes string, is a bad
    call, so it raises ValueError, which the caller answers as a business error.
    """
    entries = (args or {}).get("commands")
    if not isinstance(entries, list):
        raise ValueError("a shell batch needs a commands list")
    commands = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("keystrokes"), str):
            raise ValueError("a shell entry needs its keystrokes string")
        commands.append(entry["keystrokes"])
    return commands


def default_result(receipts: list) -> str:
    """The shape the recordings use: the terminal output text, stdout then stderr per step."""
    return "".join(_text_of(receipt.stdout) + _text_of(receipt.stderr) for receipt in receipts)


class RealTool:
    """One tool run for real: render the args, step the world, shape the receipts (D262)."""

    def __init__(self, name: str, make_world: Callable[[], RealWorld],
                 render: Optional[Callable[[dict], list[str]]] = None,
                 result_of: Optional[Callable[[list], Any]] = None):
        self.name = name
        self.make_world = make_world
        self.render = render or default_render
        self.result_of = result_of or default_result
        self._world: Optional[RealWorld] = None

    def open(self) -> RealWorld:
        """Reset the world once per Run, lazily on the first call, and keep it.

        A reset that raises after the container was created still releases it: the world
        is kept before resetting so the failure path can close it, and the keep is cleared
        so the next call starts fresh. A close that fails with a world error is ignored
        beside the reset failure it follows; any other close failure propagates.
        """
        if self._world is None:
            world = self.make_world()
            self._world = world
            try:
                world.reset()
            except Exception:
                self._world = None
                _close_ignoring_world_error(world)
                raise
        return self._world

    @property
    def opened(self) -> bool:
        """Whether the world is open now: reset ran and close has not released it."""
        return self._world is not None

    def call(self, args: Optional[dict] = None) -> RealOutcome:
        """Run the rendered commands in order; the first world failure decides the outcome.

        A bad batch is a business error and opens nothing. A world failure (the WorldError
        family by name, a timed out or truncated step) is returned with `world_answered`
        False, keeping its own error class and message for the record, so the Router can
        end the Run. A non-zero exit changes nothing: the scaffold types every entry of a
        batch whatever the earlier exit codes were, so the batch runs to its end and the
        result is always the shaped receipts, with the exit codes kept on the receipts.
        """
        try:
            commands = self.render(dict(args or {}))
        except ValueError as exc:
            return RealOutcome(None, "business_error", _error_message(exc), [])
        try:
            world = self.open()
        except Exception as exc:
            _raise_if_not_world(exc)
            return RealOutcome(None, _world_error_class(exc), _error_message(exc), [],
                               world_answered=False)
        return self._run(world, commands)

    def _run(self, world: RealWorld, commands: list[str]) -> RealOutcome:
        """Step every command in order; only a world failure ends the batch early.

        Exit codes never stop the batch and never change the result shape: the recordings
        this route imitates carry the terminal text of the whole batch, typed entry by entry
        whatever each entry exited with, so the result is always the shaped receipts.
        """
        receipts: list = []
        for command in commands:
            try:
                receipt = world.step(command)
            except Exception as exc:
                _raise_if_not_world(exc)
                return RealOutcome(None, _world_error_class(exc), _error_message(exc), receipts,
                                   world_answered=False)
            receipts.append(receipt)
            if receipt.timed_out:
                return RealOutcome(None, "transient", "the command timed out", receipts,
                                   world_answered=False)
            if receipt.truncated:
                return RealOutcome(None, "unknown", "the command output exceeded its limit",
                                   receipts, world_answered=False)
        return RealOutcome(self.result_of(receipts), None, None, receipts)

    def end_state(self, limit_bytes: int) -> bytes:
        """Export the workspace bytes; a tool never called has nothing to export."""
        if self._world is None:
            return b""
        return bytes(self._world.export_workspace(limit_bytes))

    def close(self) -> bool:
        """Release the world; a tool never called opens nothing and reports False."""
        world, self._world = self._world, None
        if world is None:
            return False
        return bool(world.close())


def real_tools_from(spec: Optional[dict],
                    make_world_for: Callable[[dict], Callable[[], RealWorld]]) -> dict[str, RealTool]:
    """Build the RealTool map from an Environment's real_tools declaration (D262).

    Each entry names its image and limits, and `make_world_for` turns that declaration
    into the factory the tool opens lazily. Tools not declared here are imitated.
    """
    tools = {}
    for name, declaration in (spec or {}).items():
        tools[name] = RealTool(name, make_world_for(declaration or {}))
    return tools
