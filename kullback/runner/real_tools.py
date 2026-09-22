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
    """What one real call did: the result, or the routed error class and message."""

    result: Any
    error_class: Optional[str] = None
    error_message: Optional[str] = None
    receipts: Optional[list] = None


# The container World's own failure names, matched by name so this module never
# imports the backend (D262). A timeout is transient, every other world failure is
# unknown: neither is the customer's answer and neither ends the Run.
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
        """Reset the world once per Run, lazily on the first call, and keep it."""
        if self._world is None:
            world = self.make_world()
            world.reset()
            self._world = world
        return self._world

    def call(self, args: Optional[dict] = None) -> RealOutcome:
        """Run the rendered commands in order; the first world failure decides the outcome.

        A bad batch is a business error and opens nothing. A world failure (the WorldError
        family by name, a timed out or truncated step) is returned as a routed error and the
        Run continues. A non-zero exit changes nothing: the scaffold types every entry of a
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
            return RealOutcome(None, _world_error_class(exc), _error_message(exc), [])
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
                return RealOutcome(None, _world_error_class(exc), _error_message(exc), receipts)
            receipts.append(receipt)
            if receipt.timed_out:
                return RealOutcome(None, "transient", "the command timed out", receipts)
            if receipt.truncated:
                return RealOutcome(None, "unknown", "the command output exceeded its limit", receipts)
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
