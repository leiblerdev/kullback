"""The second adapter behind the intake seam: terminal agent recordings in terminus-2 shape.

It votes with positive evidence (a rows envelope whose rows carry a turn list, or one recording
with a turn list and a trial marker), yields one recording per row, and maps each recording to a
Trace with one tool, the shell, whose calls are read out of the assistant text. Grader and
grouping columns leave the trace for the sidecar, the same way the first adapter treats its own.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterator, Optional

from kullback.runner.records import RawPtr, ToolCall, Trace

# The wire vocabulary of this format, kept here so no other module names it (G12, wave rule 5).
COLUMN_TURNS = "conversations"
COLUMN_TRIAL = "trial_name"
FIELD_COMMANDS = "commands"
FIELD_KEYSTROKES = "keystrokes"
FIELD_DURATION = "duration"

# The one tool these recordings show: the shell the agent typed into, named by what it is.
TOOL_SHELL = "shell"

# A user turn shows terminal output when the scaffold stamped one of these headers on it. The
# opening turn carries the state header with the instruction; scaffold complaints carry neither.
OUTPUT_HEADERS = ("New Terminal Output:", "Current terminal state:")


class Terminus2Adapter:
    """Terminal recordings: a turn list of user and assistant messages, shell calls as JSON text."""

    name = "terminus_2"
    maps = True
    display = "terminus-2"

    def detect(self, document: Any, jsonl: bool = False) -> tuple[float, list[str]]:
        if isinstance(document, list):
            return _vote_on_records(document, "list records carry a turn list",
                                    "no list record carries a turn list", 0.8)
        if not isinstance(document, dict):
            return (0.0, ["not a parsed object, so no shape to read"])
        rows = document.get("rows")
        if isinstance(rows, list):
            return _vote_on_envelope(rows)
        if _is_recording(document):
            return (0.8, ["carries a turn list"] + _positive_extras(document))
        return (0.0, ["neither a rows list nor a turn list with a trial marker"])

    def recordings(self, document: Any) -> Iterator[Any]:
        if isinstance(document, list):
            return iter(_unwrap(item) for item in document)
        if isinstance(document, dict) and isinstance(document.get("rows"), list):
            return iter(_unwrap(item) for item in document["rows"])
        return iter([document])

    def environment(self, document: Any) -> dict:
        return {}

    def sidecar(self, recording: Any, document: Any) -> dict:
        """Every column beside the turn list: outcomes, grader fields and grouping hints alike."""
        if not isinstance(recording, dict):
            return {}
        return {key: value for key, value in recording.items() if key != COLUMN_TURNS}

    def to_trace(self, recording: dict, ctx: Any) -> Trace:
        from kullback.builder.ingest import detect_truncation
        from kullback.runner.records import Turn

        turns_raw = recording.get(COLUMN_TURNS) or []
        ptr = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index)
        trace_id = str(recording.get(COLUMN_TRIAL)
                       or f"{ctx.raw_hash[:12]}-{ctx.index}")
        turns, calls = [], []
        for msg_index, message in enumerate(turns_raw):
            turn, fresh = _map_turn(message, msg_index, trace_id, ctx)
            turns.append(Turn(**turn))
            calls.extend(fresh)
        _attach_results(turns_raw, calls, ctx, detect_truncation)
        return Trace(
            trace_id=trace_id,
            raw_hash=ctx.raw_hash,
            ingest_version=ctx.ingest_version,
            source=self.name,
            turns=turns,
            tool_calls=calls,
            tools_declared=None,
            system_prompt=None,
            tools_declared_ptr=None,
            system_prompt_ptr=None,
            info_ptr=None,
            raw_ptr=ptr,
        )


def _vote_on_envelope(rows: list) -> tuple[float, list[str]]:
    """The vote for a rows envelope: size plus the turn-list evidence in a sample."""
    heads = [item for item in rows[:20] if isinstance(item, dict)]
    hits = sum(1 for item in heads if _is_recording(_unwrap(item)))
    if hits:
        return (0.9, [f"has a rows list with {len(rows)} entries",
                      f"{hits} of {len(heads)} sampled rows carry a turn list"])
    return (0.0, ["has a rows list but no sampled row carries a turn list"])


def _vote_on_records(records: list, found: str, missing: str,
                     confidence: float) -> tuple[float, list[str]]:
    """The vote for a bare list of records, sampled the same way as an envelope."""
    heads = [item for item in records[:20] if isinstance(item, dict)]
    hits = sum(1 for item in heads if _is_recording(_unwrap(item)))
    if hits:
        return (confidence, [f"{hits} of {len(heads)} {found}"])
    return (0.0, [missing])


def _map_turn(message: dict, msg_index: int, trace_id: str, ctx: Any) -> tuple[dict, list]:
    """One message to its turn plus the shell calls it requests, if it requests any."""
    here = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index, msg_index=msg_index)
    role = message.get("role") or "assistant"
    if role not in ("user", "assistant"):
        role = "assistant"
    text = message.get("content")
    content = text if text is None or isinstance(text, str) else json.dumps(
        text, ensure_ascii=False, default=str)
    turn = {"idx": msg_index, "role": role, "content": content, "raw_ptr": here}
    entries = _command_entries(text) if role == "assistant" else None
    calls = [_shell_call(entry, trace_id, here) for entry in entries] if entries else []
    return (turn, calls)


def _unwrap(item: Any) -> Any:
    """A rows-envelope entry holds its recording under row; a bare item is its own recording."""
    if isinstance(item, dict) and isinstance(item.get("row"), dict):
        return item["row"]
    return item


def _is_recording(candidate: Any) -> bool:
    """Positive shape: a turn list of user and assistant messages only, with a trial marker."""
    if not isinstance(candidate, dict):
        return False
    turns = candidate.get(COLUMN_TURNS)
    if not isinstance(turns, list) or not turns:
        return False
    if not all(isinstance(turn, dict) and turn.get("role") in ("user", "assistant")
               for turn in turns):
        return False
    return isinstance(candidate.get(COLUMN_TRIAL), str)


def _positive_extras(document: dict) -> list[str]:
    """The evidence beyond the turn list that this is one recording of this format."""
    found = [key for key in ("run_id", "episode", "trial_name", "original_source", "agent",
                             "model", "task", "result") if key in document]
    return [f"carries {', '.join(found)}"] if found else []


def turn_kind(text: Any) -> str:
    """Why one assistant turn yields calls or none: commands, empty_commands,
    json_without_commands, broken_json, or no_json. The survey script reads this so the
    pipeline and the report count unparsed turns the same way; nothing here repairs text."""
    kind, _ = _parsed_commands(text)
    return kind


def _parsed_commands(text: Any) -> tuple[str, Optional[list[dict]]]:
    """One strict parse shared by the survey counter and the mapper below."""
    if not isinstance(text, str) or not text.strip().startswith("{"):
        return ("no_json", None)
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return ("broken_json", None)
    if not isinstance(parsed, dict):
        return ("broken_json", None)
    commands = parsed.get(FIELD_COMMANDS)
    if not isinstance(commands, list):
        return ("json_without_commands", None)
    if not commands:
        return ("empty_commands", None)
    if _bad_entry(commands):
        return ("broken_json", None)
    return ("commands", commands)


def _bad_entry(commands: list) -> bool:
    """True when any command entry is not an object carrying the keystrokes string."""
    return any(not isinstance(entry, dict)
               or not isinstance(entry.get(FIELD_KEYSTROKES), str) for entry in commands)


def _command_entries(text: Any) -> Optional[list[dict]]:
    """The shell commands one assistant turn requests, or None when the turn carries none.

    Strict on purpose: the turn must be a JSON object with a commands list whose every entry is
    an object carrying the keystrokes string. Anything else (prose, broken JSON, a completion
    marker, an unexpected shape) yields no calls; the turn text itself is never repaired.
    """
    kind, entries = _parsed_commands(text)
    return entries if kind == "commands" else None


def _shell_call(entry: dict, trace_id: str, here: RawPtr) -> ToolCall:
    """One command entry as one call on the shell; the entry's measured time rides as latency."""
    return ToolCall(
        id=None,
        name=TOOL_SHELL,
        args={FIELD_KEYSTROKES: entry[FIELD_KEYSTROKES]},
        requestor="assistant",
        raw_ptr=here,
        trace_id=trace_id,
        has_result=False,
        resolved=False,
        latency_ms=_latency_ms(entry.get(FIELD_DURATION)),
    )


def _latency_ms(duration: Any) -> Optional[float]:
    """The entry's measured seconds as milliseconds, or None when it says nothing usable.

    Booleans, strings, negatives and non-finite values are not times, so they stay
    unrecorded instead of becoming a plausible-looking latency."""
    if isinstance(duration, bool):
        return None
    if not isinstance(duration, (int, float)) or not math.isfinite(duration):
        return None
    if duration < 0:
        return None
    return float(duration) * 1000.0


def _attach_results(turns_raw: list, calls: list, ctx: Any,
                    detect_truncation: Any) -> None:
    """Answer each command batch with the terminal output the next user turn shows for it.

    The batch shares one output turn, and the recording never marks where one command's output
    ends and the next begins, so splitting it would claim boundaries the recording does not
    show. The whole output lands on the batch's last call and the earlier calls stay unobserved
    (no result, unresolved), which is what the admission rules read. A batch whose next turn is
    missing, not the user's, or carries no terminal header (a scaffold complaint, the opening
    instruction) leaves every call unobserved the same way.
    """
    by_turn: dict[int, list] = {}
    for call in calls:
        index = call.raw_ptr.msg_index if call.raw_ptr else None
        by_turn.setdefault(index, []).append(call)
    for msg_index, group in by_turn.items():
        if not isinstance(msg_index, int):
            continue
        answer = turns_raw[msg_index + 1] if msg_index + 1 < len(turns_raw) else None
        if (answer is None or answer.get("role") != "user"
                or not _is_terminal_output(answer.get("content"))):
            continue
        last = group[-1]
        content = answer.get("content")
        last.truncated, last.visible_len, last.cut_marker = detect_truncation(content)
        last.result = content
        last.has_result = True
        last.resolved = True
        last.result_ptr = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index,
                                 msg_index=msg_index + 1)


def _is_terminal_output(content: Any) -> bool:
    """A scaffold-stamped terminal transcript, not an instruction and not a complaint."""
    return isinstance(content, str) and any(header in content for header in OUTPUT_HEADERS)


def trace_hash(trace: Trace) -> str:
    """Content hash of a Trace with its own hash field blanked, stable across runs."""
    from kullback.runner.records import as_dict, content_hash

    body = as_dict(trace)
    body["hash"] = ""
    return content_hash(body)
