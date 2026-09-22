"""The second adapter behind the intake seam: terminal agent recordings in terminus-2 shape.

It votes with positive evidence (a rows envelope whose rows carry a turn list, or one recording
with a turn list and a trial marker), yields one recording per row, and maps each recording to a
Trace with one tool, the shell. One shell call per assistant command turn carries the batch as
the turn carries it; the first command object in the text is the call and the prose around it
stays the assistant message. Grader and grouping columns leave the trace for the sidecar, the
same way the first adapter treats its own."""

from __future__ import annotations

import json
import math
from typing import Any, Iterator, Optional

from kullback.runner.records import RawPtr, ToolCall, Trace

# The wire vocabulary of this format, kept here so no other module names it (G12, wave rule 5).
COLUMN_TURNS = "conversations"
COLUMN_TRIAL = "trial_name"
COLUMN_SOURCE = "original_source"
FIELD_COMMANDS = "commands"
FIELD_KEYSTROKES = "keystrokes"
FIELD_DURATION = "duration"

# The two boundary headers of the opening turn: the instruction sits between them.
HEADER_TASK = "Task Description:"
HEADER_STATE = "Current terminal state:"

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
        """Every column beside the turn list: outcomes, grader fields and grouping hints alike.

        Two derived keys join them: `task_ref` names the task the recording claims
        (the trial marker up to its first `__`, with the source column beside it)
        and `instruction` carries the opening turn's instruction, or None where
        the turn carries no such headers. The recording's own text is the truth;
        a fetched copy can only cross-check it, never replace it.
        """
        if not isinstance(recording, dict):
            return {}
        fields = {key: value for key, value in recording.items() if key != COLUMN_TURNS}
        fields["task_ref"] = _task_ref(recording)
        fields["instruction"] = _opening_instruction(recording.get(COLUMN_TURNS))
        return fields

    def to_trace(self, recording: dict, ctx: Any) -> Trace:
        from kullback.builder.ingest import detect_truncation
        from kullback.runner.records import Turn

        turns_raw = recording.get(COLUMN_TURNS) or []
        ptr = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index)
        trace_id = str(recording.get(COLUMN_TRIAL)
                       or f"{ctx.raw_hash[:12]}-{ctx.index}")
        turns, calls = [], []
        for msg_index, message in enumerate(turns_raw):
            answer = turns_raw[msg_index + 1] if msg_index + 1 < len(turns_raw) else None
            turn, fresh = _map_turn(message, msg_index, trace_id, ctx, answer)
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


def _map_turn(message: dict, msg_index: int, trace_id: str, ctx: Any,
              answer: Any = None) -> tuple[dict, list]:
    """One message to its turn plus the shell call it requests, if it requests one.

    One call per command turn: the scaffold takes the turn's whole command list and returns
    one screen, so the call carries the list as the turn carries it and the batch is never
    split. A non-empty command list is always a call, unresolved when no output answers it.
    An empty command list is a call only when the recording shows it was run, which is when
    the next turn is the user's and carries terminal output; an empty list the recording
    ends on, or one answered by a scaffold note, is the assistant saying it is done and no
    call at all. The Trace claims nothing the recording does not show, in either direction."""
    here = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index, msg_index=msg_index)
    role = message.get("role") or "assistant"
    if role not in ("user", "assistant"):
        role = "assistant"
    text = message.get("content")
    content = text if text is None or isinstance(text, str) else json.dumps(
        text, ensure_ascii=False, default=str)
    turn = {"idx": msg_index, "role": role, "content": content, "raw_ptr": here}
    kind, entries = _parsed_commands(text) if role == "assistant" else ("no_json", None)
    submits = kind in COMMAND_KINDS or (kind == "empty_commands" and _answered_by_output(answer))
    calls = [_batch_call(entries if entries is not None else [], trace_id, here)] \
        if submits else []
    return (turn, calls)


def _answered_by_output(answer: Any) -> bool:
    """True when the turn after an assistant turn is the user's and shows terminal output."""
    return (isinstance(answer, dict) and answer.get("role") == "user"
            and _is_terminal_output(answer.get("content")))


def _unwrap(item: Any) -> Any:
    """A rows-envelope entry holds its recording under row; a bare item is its own recording."""
    if isinstance(item, dict) and isinstance(item.get("row"), dict):
        return item["row"]
    return item


def _task_ref(recording: dict) -> dict:
    """The task the recording claims: the trial marker before its first `__`, and the source."""
    trial = recording.get(COLUMN_TRIAL)
    task_id = trial.split("__", 1)[0] if isinstance(trial, str) else None
    source = recording.get(COLUMN_SOURCE)
    return {"id": task_id, "source": source if isinstance(source, str) else None}


def _opening_instruction(turns: Any) -> Optional[str]:
    """The instruction of the first user turn, between the two boundary headers.

    None when there is no user turn, when its content is not text, or when
    either header is absent. The text between the headers is the instruction;
    the headers themselves are scaffold, not content.
    """
    first_user = next((turn for turn in turns or []
                       if isinstance(turn, dict) and turn.get("role") == "user"), None)
    content = first_user.get("content") if first_user is not None else None
    if not isinstance(content, str) or HEADER_TASK not in content:
        return None
    after = content.split(HEADER_TASK, 1)[1]
    if HEADER_STATE not in after:
        return None
    return after.split(HEADER_STATE, 1)[0].strip()


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


# An assistant turn parsed far enough to submit a batch: the whole text is the object, or the
# object sits inside prose. Empty turns submit an empty batch; the rest submit nothing.
COMMAND_KINDS = ("commands_whole", "commands_embedded")


def turn_kind(text: Any) -> str:
    """How one assistant turn parsed: commands_whole, commands_embedded, empty_commands,
    json_without_commands, broken_json, or no_json. The survey script reads this so the
    pipeline and the report count turns the same way; nothing here repairs text."""
    kind, _ = _parsed_commands(text)
    return kind


def _parsed_commands(text: Any) -> tuple[str, Optional[list[dict]]]:
    """One strict parse shared by the survey counter and the mapper below.

    The first JSON object in the text that carries a command list is the call; the prose
    around it stays the assistant message. Strict inside each object: every entry must
    carry the keystrokes string, and a broken object stays broken (the scan moves on, the
    text is never repaired)."""
    if not isinstance(text, str):
        return ("no_json", None)
    stripped = text.strip()
    spans = _object_spans(stripped)
    if not spans:
        return ("no_json", None)
    fallback = None
    for start, end in spans:
        try:
            parsed = json.loads(stripped[start:end])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        commands = parsed.get(FIELD_COMMANDS)
        if not isinstance(commands, list):
            fallback = fallback or "json_without_commands"
            continue
        if not commands:
            fallback = fallback or "empty_commands"
            continue
        if _bad_entry(commands):
            continue
        whole = "_whole" if (start, end) == (0, len(stripped)) else "_embedded"
        return ("commands" + whole, commands)
    if fallback is not None:
        return (fallback, [] if fallback == "empty_commands" else None)
    return ("broken_json", None)


def _object_spans(text: str) -> list[tuple[int, int]]:
    """Every balanced brace span in order, strings honoured so braces in prose do not count."""
    spans = []
    pos, end = 0, len(text)
    while pos < end:
        if text[pos] != "{":
            pos += 1
            continue
        depth, instr, esc, cur = 0, False, False, pos
        closed = None
        while cur < end:
            char = text[cur]
            if instr:
                if esc:
                    esc = False
                elif char == "\\":
                    esc = True
                elif char == '"':
                    instr = False
            elif char == '"':
                instr = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    closed = cur + 1
                    break
            cur += 1
        if closed is None:
            pos += 1
        else:
            spans.append((pos, closed))
            pos = closed
    return spans


def _bad_entry(commands: list) -> bool:
    """True when any command entry is not an object carrying the keystrokes string."""
    return any(not isinstance(entry, dict)
               or not isinstance(entry.get(FIELD_KEYSTROKES), str) for entry in commands)


def command_entries(text: Any) -> Optional[list[dict]]:
    """The shell batch one assistant turn requests, or None when the turn carries none.

    Empty batches yield an empty list (they still submit, and the scaffold still answers);
    anything without a command list yields None. The survey counts through this so both
    read the same batches."""
    kind, entries = _parsed_commands(text)
    if kind in COMMAND_KINDS:
        return entries
    if kind == "empty_commands":
        return []
    return None


def _batch_call(entries: list, trace_id: str, here: RawPtr) -> ToolCall:
    """One command turn as one call on the shell, carrying the batch as the turn carries it."""
    return ToolCall(
        id=None,
        name=TOOL_SHELL,
        args=_batch_args(entries),
        requestor="assistant",
        raw_ptr=here,
        trace_id=trace_id,
        has_result=False,
        resolved=False,
        latency_ms=None,
    )


def _batch_args(entries: list) -> dict:
    """The call's arguments: each entry's keystrokes and duration, in order.

    Duration stays an argument of the entry, not a latency of the call: it measures a wait
    the agent asked for, not a time anything took. A missing or unusable duration is left
    out rather than recorded as a plausible-looking number."""
    batch = []
    for entry in entries:
        item = {FIELD_KEYSTROKES: entry[FIELD_KEYSTROKES]}
        duration = _usable_duration(entry.get(FIELD_DURATION))
        if duration is not None:
            item[FIELD_DURATION] = duration
        batch.append(item)
    return {FIELD_COMMANDS: batch}


def _usable_duration(duration: Any) -> Optional[float]:
    """A duration worth carrying, or None when the entry says nothing usable.

    Booleans, strings, negatives and non-finite values are not waits, so they stay out
    instead of becoming a plausible-looking argument."""
    if isinstance(duration, bool):
        return None
    if not isinstance(duration, (int, float)) or not math.isfinite(duration):
        return None
    if duration < 0:
        return None
    return float(duration)


def _attach_results(turns_raw: list, calls: list, ctx: Any,
                    detect_truncation: Any) -> None:
    """Answer each batch with the terminal output the next user turn shows for it.

    The whole output lands on the turn's one call, verbatim and unsplit: a warnings block
    riding with the output stays in the result as shown, and only the opening turn counts
    as the customer's request. A non-empty batch whose next turn is missing, not the user's,
    or carries no terminal header (a scaffold complaint) leaves its call unobserved, which
    is what the admission rules read; an empty command list in the same position never
    became a call, as _map_turn says. Output after a turn where no command object was found
    means the adapter is stricter than the scaffold was; the output stays unclaimed and an
    explicitly unobserved marker call records the contradiction, so the recording cannot
    come out clean."""
    by_turn = {}
    for call in calls:
        index = call.raw_ptr.msg_index if call.raw_ptr else None
        by_turn[index] = call
    for msg_index, message in enumerate(turns_raw):
        if (message.get("role") or "assistant") != "assistant":
            continue
        answer = turns_raw[msg_index + 1] if msg_index + 1 < len(turns_raw) else None
        if not _answered_by_output(answer):
            continue
        call = by_turn.get(msg_index)
        if call is None:
            calls.append(_batch_call([], _trace_of(calls, ctx), _ptr_at(ctx, msg_index)))
            continue
        content = answer.get("content")
        call.truncated, call.visible_len, call.cut_marker = detect_truncation(content)
        call.result = content
        call.has_result = True
        call.resolved = True
        call.result_ptr = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index,
                                 msg_index=msg_index + 1)


def _trace_of(calls: list, ctx: Any) -> str:
    """The trace id for a marker call: the id the turn's own calls carry, if any."""
    for call in calls:
        if call.trace_id:
            return call.trace_id
    return f"{ctx.raw_hash[:12]}-{ctx.index}"


def _ptr_at(ctx: Any, msg_index: int) -> RawPtr:
    """The pointer a marker call cites: the assistant turn the output followed."""
    return RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index, msg_index=msg_index)


def _is_terminal_output(content: Any) -> bool:
    """A scaffold-stamped terminal transcript, not an instruction and not a complaint."""
    return isinstance(content, str) and any(header in content for header in OUTPUT_HEADERS)


# A scaffold note refusing a turn names the refusal; a warning about extra text means the
# scaffold found the object too. Only the first disagrees with an adapter that parsed.
_ERROR_MARKS = ("parsing errors", "Invalid JSON", "Missing required fields")
_WARNING_MARKS = ("WARNINGS", "warnings:")
_COMPLETION_MARKS = ("mark the task as complete",)


def note_kind(text: Any) -> str:
    """What a non-output user turn is: error, warning, completion, or other scaffold note."""
    if not isinstance(text, str):
        return "other"
    if any(mark in text for mark in _COMPLETION_MARKS):
        return "completion"
    if any(mark in text for mark in _ERROR_MARKS):
        return "error"
    if any(mark in text for mark in _WARNING_MARKS):
        return "warning"
    return "other"


def contradictions(turns: list) -> dict:
    """Both consistency directions over one recording's turns, as counts.

    output_without_commands: terminal output after a turn where no command object was
    found (the adapter is stricter than the scaffold was). complaint_after_parsed: a
    scaffold error note after a turn the adapter parsed (the adapter is more lenient).
    warning_after_parsed: a scaffold warning after a parsed turn (both found the object,
    so no disagreement, counted apart). The mapper enforces the same two rules: the
    first leaves an unobserved marker call, the second an unobserved batch call, so a
    recording with either contradiction cannot come out as a clean complete record."""
    counts = {"output_without_commands": 0, "complaint_after_parsed": 0,
              "warning_after_parsed": 0}
    for first, second in zip(turns, turns[1:], strict=False):
        if (first.get("role") or "assistant") != "assistant":
            continue
        if (second.get("role") or "user") != "user":
            continue
        kind = turn_kind(first.get("content"))
        parsed = kind in COMMAND_KINDS
        if _is_terminal_output(second.get("content")):
            if not parsed and kind != "empty_commands":
                counts["output_without_commands"] += 1
        elif parsed:
            note = note_kind(second.get("content"))
            if note == "error":
                counts["complaint_after_parsed"] += 1
            elif note == "warning":
                counts["warning_after_parsed"] += 1
    return counts


def trace_hash(trace: Trace) -> str:
    """Content hash of a Trace with its own hash field blanked, stable across runs."""
    from kullback.runner.records import as_dict, content_hash

    body = as_dict(trace)
    body["hash"] = ""
    return content_hash(body)
