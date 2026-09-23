from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from kullback.builder import ingest, prefixes
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, as_dict, content_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def run_cli(argv):
    return cohort_module().run(argv)


def cohort_module():
    return importlib.import_module("agenttrove_prefix_cohort")


RAW = "f" * 64
SIM = 0


def ptr(msg):
    return RawPtr(file_hash=RAW, sim_index=SIM, msg_index=msg)


def turn(idx, role="assistant", content="invented-text"):
    return Turn(idx=idx, role=role, content=content, raw_ptr=ptr(idx))


def placed(idx, raw, role="assistant", content="invented-text"):
    return Turn(idx=idx, role=role, content=content, raw_ptr=ptr(raw))


def call(msg, answer=None, resolved=True, truncated=False, empty=False, batch=1, null=False, owner="invented-trial-1"):
    if empty:
        args = {"commands": []}
    else:
        args = {"commands": [{"keystrokes": "invented-cmd", "duration": 0.2} for _ in range(batch)]}
    result = None
    result_ptr = None
    has_result = False
    if answer is not None:
        result = None if null else "New Terminal Output:\ninvented-output"
        result_ptr = ptr(answer)
        has_result = True
    return ToolCall(
        name="shell",
        args=args,
        result=result,
        raw_ptr=ptr(msg),
        result_ptr=result_ptr,
        trace_id=owner,
        has_result=has_result,
        resolved=resolved,
        truncated=truncated,
    )


def trace_of(turns, calls, trace_id="invented-trial-1"):
    return Trace(
        trace_id=trace_id,
        raw_hash=RAW,
        ingest_version="test",
        source="terminus_2",
        turns=turns,
        tool_calls=calls,
        raw_ptr=RawPtr(file_hash=RAW, sim_index=SIM),
        hash="",
    )


def resolved_pair(msg, answer):
    return call(msg, answer=answer, resolved=True)


def test_the_tail_after_the_last_result_is_trimmed_with_any_empty_done():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na"), turn(3), turn(4)]
    calls = [resolved_pair(1, 2), call(3, answer=None, resolved=False)]
    out = prefixes.select_prefix(trace_of(turns, calls))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.turn_end == 2
    assert out.retained_calls == (0,)
    assert out.standing == "evidence_only"
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na"), turn(3)]
    calls = [resolved_pair(1, 2)]
    out = prefixes.select_prefix(trace_of(turns, calls))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.turn_end == 2
    assert 3 not in out.retained_turns


def invented_interior_unresolved_case(kind):
    if kind == "open_range":
        turns = [
            turn(0, "user", "invented-open"),
            turn(1),
            turn(2, "user", "New Terminal Output:\na"),
            turn(3),
            turn(4),
            turn(5, "user", "New Terminal Output:\nb"),
        ]
        calls = [resolved_pair(1, 2), call(3, answer=None, resolved=False), resolved_pair(4, 5)]
    else:
        turns = [turn(0), turn(1), turn(2, "user", "New Terminal Output:\na")]
        calls = [call(0, answer=2, resolved=True), call(1, answer=None, resolved=False)]
    return turns, calls


@pytest.mark.parametrize("kind", ["open_range", "closed_range"])
def test_interior_unresolved_withheld(kind):
    turns, calls = invented_interior_unresolved_case(kind)
    out = prefixes.select_prefix(trace_of(turns, calls))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "interior_unresolved"


def invented_unanswered_trace_case(kind):
    if kind == "unanswered":
        turns = [turn(0, "user", "invented-open"), turn(1), turn(2)]
        return trace_of(turns, [call(1, answer=None, resolved=False)]), "no_resolved_call"
    return trace_of([], []), "empty_trace"


@pytest.mark.parametrize("kind", ["unanswered", "empty"])
def test_no_answered_calls_withheld(kind):
    trace, reason = invented_unanswered_trace_case(kind)
    out = prefixes.select_prefix(trace)
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == reason


def test_batch_kept_whole():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    made = call(1, answer=2, resolved=True, batch=3)
    assert len(made.args["commands"]) == 3
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.retained_calls == (0,)
    assert len(made.args["commands"]) == 3


def invented_benign_selection_case(kind):
    if kind == "warning":
        turns = [
            turn(0, "user", "invented-open"),
            turn(1),
            turn(2, "user", "New Terminal Output:\nWARNINGS invented-note\nout"),
        ]
        made = resolved_pair(1, 2)
    else:
        turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
        made = call(1, answer=2, resolved=True, null=True)
    return turns, made


@pytest.mark.parametrize("kind", ["warning", "null_result"])
def test_selection_tolerates_benign_content(kind):
    turns, made = invented_benign_selection_case(kind)
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.turn_end == 2


def test_orphan_marker_excludes():
    turns = [
        turn(0, "user", "invented-open"),
        turn(1),
        turn(2, "user", "New Terminal Output:\na"),
        turn(3),
        turn(4, "user", "New Terminal Output:\nb"),
    ]
    marker = call(1, answer=None, resolved=False, empty=True)
    third = call(3, answer=4, resolved=True)
    out = prefixes.select_prefix(trace_of(turns, [marker, third]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "orphan_marker"


def invented_foreign_pointer_trace(kind):
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    if kind == "result_ptr":
        made = resolved_pair(1, 2)
        made.result_ptr = RawPtr(file_hash="0" * 64, sim_index=SIM, msg_index=2)
        return trace_of(turns, [made])
    if kind == "root_ptr":
        parent = trace_of(turns, [resolved_pair(1, 2)])
        return parent.model_copy(update={"raw_ptr": RawPtr(file_hash="0" * 64, sim_index=SIM)})
    made = call(1, answer=2, owner="invented-other")
    return trace_of(turns, [made])


def invented_faulty_trace(kind):
    """One invented recording per provenance fault the selector must withhold, plus one it allows."""
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    if kind in ("result_ptr", "root_ptr", "call_owner"):
        return invented_foreign_pointer_trace(kind)
    if kind in ("duplicate_call", "duplicate_answer"):
        return trace_of(*invented_ambiguous_provenance_case(kind))
    if kind in ("result_to_assistant", "call_on_user_turn"):
        misplaced_turns, made = invented_misplaced_pointer_case(kind)
        return trace_of(misplaced_turns, [made])
    if kind == "unordered":
        made = resolved_pair(1, 2)
        made.result_ptr = ptr(0)
        return trace_of(turns, [made])
    if kind == "missing":
        made = resolved_pair(1, 2)
        made.result_ptr = None
        return trace_of(turns, [made])
    if kind == "truncated":
        return trace_of(turns, [call(1, answer=2, resolved=True, truncated=True)])
    if kind == "stale":
        parent = trace_of(turns, [resolved_pair(1, 2)])
        body = as_dict(parent)
        body["hash"] = ""
        claimed = parent.model_copy(update={"hash": content_hash(body)})
        changed = claimed.model_copy(deep=True)
        changed.turns[1] = changed.turns[1].model_copy(update={"content": "invented-changed"})
        return changed
    assert kind == "owner_absent"
    return trace_of(turns, [call(1, answer=2, owner=None)])


@pytest.mark.parametrize(("kind", "reason"), [
    ("result_ptr", "foreign_pointer"),
    ("root_ptr", "foreign_pointer"),
    ("call_owner", "foreign_pointer"),
    ("unordered", "unordered_pointer"),
    ("missing", "malformed_pointer"),
    ("result_to_assistant", "malformed_pointer"),
    ("call_on_user_turn", "malformed_pointer"),
    ("duplicate_call", "ambiguous_provenance"),
    ("duplicate_answer", "ambiguous_provenance"),
    ("truncated", "truncated_result"),
    ("stale", "stale_hash"),
    ("owner_absent", None),
])
def test_a_provenance_fault_withholds_the_recording_with_its_reason(kind, reason):
    out = prefixes.select_prefix(invented_faulty_trace(kind))
    if reason is None:
        assert isinstance(out, prefixes.PrefixSelection), "a call with no owner is not a foreign one"
    else:
        assert isinstance(out, prefixes.Withheld)
        assert out.reason == reason


def invented_ambiguous_provenance_case(kind):
    if kind == "duplicate_call":
        turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
        calls = [resolved_pair(1, 2), resolved_pair(1, 2)]
    else:
        turns = [turn(0), turn(1), turn(2, "user", "New Terminal Output:\na")]
        calls = [call(0, answer=2, resolved=True), call(1, answer=2, resolved=True)]
    return turns, calls


def test_cohort_id_deterministic():
    def build():
        turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
        return trace_of(turns, [resolved_pair(1, 2)])

    first = prefixes.select_prefix(build())
    second = prefixes.select_prefix(build())
    assert isinstance(first, prefixes.PrefixSelection)
    assert isinstance(second, prefixes.PrefixSelection)
    assert first.cohort_id == second.cohort_id
    assert first.cohort_id != first.parent_trace_id


def test_heldout_rejected_first():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na"), turn(3)]
    calls = [resolved_pair(1, 2), call(3, answer=None, resolved=False)]
    parent = trace_of(turns, calls, trace_id="invented-heldout")
    out = prefixes.select_prefix(parent, heldout_ids=frozenset({"invented-heldout"}))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "heldout_parent"


def test_floor_and_standing_untouched():
    assert ingest.MIN_TASK_ELIGIBLE_SHARE == 0.75
    assert prefixes.EVIDENCE_ONLY == ingest.EVIDENCE_ONLY
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    out = prefixes.select_prefix(trace_of(turns, [resolved_pair(1, 2)]))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.standing == ingest.EVIDENCE_ONLY
    assert ingest.MIN_TASK_ELIGIBLE_SHARE == 0.75


def test_sparse_idx_uses_raw_mapping():
    turns = [
        placed(10, 0, "user", "invented-open"),
        placed(20, 1),
        placed(30, 2, "user", "New Terminal Output:\na"),
        placed(40, 3),
    ]
    out = prefixes.select_prefix(trace_of(turns, [resolved_pair(1, 2)]))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.turn_end == 2
    assert out.retained_turns == (0, 1, 2)


def invented_misplaced_pointer_case(kind):
    if kind == "result_to_assistant":
        turns = [turn(0), turn(1), turn(2, "user", "New Terminal Output:\na")]
        made = call(0, answer=1, resolved=True)
    else:
        turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
        made = call(0, answer=2, resolved=True)
    return turns, made


def invented_cli_recording(complete):
    batch = json.dumps({"analysis": "invented", "commands": [{"keystrokes": "invented-cli", "duration": 0.2}]})
    empty = json.dumps({"analysis": "invented", "commands": []})
    turns = [
        {"role": "user", "content": "Invented open"},
        {"role": "assistant", "content": batch},
        {"role": "user", "content": "New Terminal Output:\ninvented-cli-output"},
    ]
    if complete:
        turns.append({"role": "assistant", "content": empty})
    else:
        turns.append({"role": "assistant", "content": batch})
    return {
        "conversations": turns,
        "trial_name": "invented-cli-complete" if complete else "invented-cli-open",
        "run_id": "invented-run",
        "episode": 1,
        "original_source": "invented",
    }


def invented_cli_input(path, rows):
    envelope = {"rows": [{"row": recording, "row_idx": index} for index, recording in enumerate(rows)]}
    target = Path(path) / "invented.json"
    target.write_text(json.dumps(envelope), encoding="utf-8")
    return path


def invented_cli_src(tmp_path, rows):
    src = tmp_path / "src"
    src.mkdir()
    invented_cli_input(src, rows)
    return src


def test_the_cli_and_publish_refuse_to_overwrite_an_existing_output(tmp_path):
    src = invented_cli_src(tmp_path, [invented_cli_recording(True)])
    output = tmp_path / "m.json"
    output.write_text("invented-prior", encoding="utf-8")
    assert run_cli(["--input-dir", str(src), "--output", str(output)]) == 2
    assert output.read_text(encoding="utf-8") == "invented-prior"
    direct = tmp_path / "direct.json"
    direct.write_bytes(b"invented-prior")
    assert cohort_module().publish_staged(direct, '{"a": 1}') is False
    assert direct.read_bytes() == b"invented-prior"


def test_cli_writes_the_manifest_and_refuses_a_missing_empty_or_file_input(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    not_a_dir = tmp_path / "file.json"
    not_a_dir.write_text("{}", encoding="utf-8")
    for bad in (tmp_path / "nope", empty, not_a_dir):
        assert run_cli(["--input-dir", str(bad), "--output", str(tmp_path / "m.json")]) == 2
    src = invented_cli_src(tmp_path, [invented_cli_recording(True), invented_cli_recording(False)])
    output = tmp_path / "m.json"
    assert run_cli(["--input-dir", str(src), "--output", str(output)]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["totals"]["recordings"] == 2
    assert manifest["totals"]["original_eligible"] == 1
    assert manifest["totals"]["selected_prefixes"] == 1
    assert manifest["writer_eligible"] is False
    assert manifest["inputs_unchanged"] is True


def invented_heldout_file(path, text):
    target = Path(path) / "heldout.json"
    target.write_text(text, encoding="utf-8")
    return str(target)


def test_cli_heldout_excludes(tmp_path):
    src = invented_cli_src(tmp_path, [invented_cli_recording(False)])
    output = tmp_path / "m.json"
    heldout = invented_heldout_file(tmp_path, '["invented-cli-open"]')
    assert run_cli(["--input-dir", str(src), "--output", str(output), "--heldout-ids", heldout]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["totals"]["selected_prefixes"] == 0
    assert manifest["files"][0]["candidates"] == 1
    assert manifest["heldout_anchor"] == "caller-supplied"
    assert manifest["heldout_ids"] == 1


@pytest.mark.parametrize(
    "after,code,written",
    [({"a.json": "invented-changed"}, 2, False), ({"a.json": "invented-digest"}, 0, True)],
    ids=["drift", "stable"],
)
def test_finish_checks_inputs_before_writing(tmp_path, after, code, written):
    output = tmp_path / "sub" / "m.json"
    before = [("a.json", 3, "invented-digest")]
    totals = {"recordings": 1, "original_eligible": 0, "selected_prefixes": 0}
    assert cohort_module().finish(str(output), before, after, [], totals, "none", 0) == code
    assert output.exists() is written
    if written:
        assert json.loads(output.read_text(encoding="utf-8"))["inputs_unchanged"] is True
    assert cohort_module().inputs_error(before, {"a.json": "invented-digest"}) is None
    for changed in ({"a.json": "invented-changed"}, {}, {"a.json": "invented-digest", "b.json": "invented-extra"},
                    {"b.json": "invented-digest"}):
        assert cohort_module().inputs_error(before, changed) is not None, changed
