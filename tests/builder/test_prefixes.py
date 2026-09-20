from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

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


def res_ptr(msg):
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
        result_ptr = res_ptr(answer)
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


def test_tail_trimmed_after_last_result():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na"), turn(3), turn(4)]
    calls = [resolved_pair(1, 2), call(3, answer=None, resolved=False)]
    out = prefixes.select_prefix(trace_of(turns, calls))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.turn_end == 2
    assert out.retained_calls == (0,)
    assert out.standing == "evidence_only"


def test_interior_unresolved_withholds_despite_later_result():
    turns = [
        turn(0, "user", "invented-open"),
        turn(1),
        turn(2, "user", "New Terminal Output:\na"),
        turn(3),
        turn(4),
        turn(5, "user", "New Terminal Output:\nb"),
    ]
    calls = [resolved_pair(1, 2), call(3, answer=None, resolved=False), resolved_pair(4, 5)]
    out = prefixes.select_prefix(trace_of(turns, calls))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "interior_unresolved"


def test_no_answered_calls_withheld():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2)]
    calls = [call(1, answer=None, resolved=False)]
    out = prefixes.select_prefix(trace_of(turns, calls))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "no_resolved_call"


def test_empty_trace_withheld():
    out = prefixes.select_prefix(trace_of([], []))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "empty_trace"


def test_batch_kept_whole():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    made = call(1, answer=2, resolved=True, batch=3)
    assert len(made.args["commands"]) == 3
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.retained_calls == (0,)
    assert len(made.args["commands"]) == 3


def test_benign_warning_with_output_retained():
    turns = [
        turn(0, "user", "invented-open"),
        turn(1),
        turn(2, "user", "New Terminal Output:\nWARNINGS invented-note\nout"),
    ]
    calls = [resolved_pair(1, 2)]
    out = prefixes.select_prefix(trace_of(turns, calls))
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


def test_trailing_empty_done_dropped():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na"), turn(3)]
    calls = [resolved_pair(1, 2)]
    out = prefixes.select_prefix(trace_of(turns, calls))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.turn_end == 2
    assert 3 not in out.retained_turns


def test_foreign_pointer_withheld():
    made = resolved_pair(1, 2)
    made.result_ptr = RawPtr(file_hash="0" * 64, sim_index=SIM, msg_index=2)
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "foreign_pointer"


def test_unordered_pointer_withheld():
    made = resolved_pair(1, 2)
    made.result_ptr = res_ptr(0)
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "unordered_pointer"


def test_missing_pointer_withheld():
    made = resolved_pair(1, 2)
    made.result_ptr = None
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "malformed_pointer"


def test_duplicate_call_provenance_withheld():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    first = resolved_pair(1, 2)
    second = resolved_pair(1, 2)
    out = prefixes.select_prefix(trace_of(turns, [first, second]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "ambiguous_provenance"


def test_truncated_result_withheld():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    made = call(1, answer=2, resolved=True, truncated=True)
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "truncated_result"


def test_mutation_isolation():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    made = resolved_pair(1, 2)
    parent = trace_of(turns, [made])
    before = copy.deepcopy(parent.model_dump(mode="json"))
    out = prefixes.select_prefix(parent)
    assert isinstance(out, prefixes.PrefixSelection)
    out.last_result_ptr.msg_index = 99
    assert parent.tool_calls[0].result_ptr.msg_index == 2
    assert parent.model_dump(mode="json") == before


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


def test_closed_range_withholds_hidden_interior():
    turns = [turn(0), turn(1), turn(2, "user", "New Terminal Output:\na")]
    first = call(0, answer=2, resolved=True)
    second = call(1, answer=None, resolved=False)
    out = prefixes.select_prefix(trace_of(turns, [first, second]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "interior_unresolved"


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


def test_result_to_assistant_turn_rejected():
    turns = [turn(0), turn(1), turn(2, "user", "New Terminal Output:\na")]
    made = call(0, answer=1, resolved=True)
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "malformed_pointer"


def test_call_on_user_turn_rejected():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    made = call(0, answer=2, resolved=True)
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "malformed_pointer"


def test_observed_null_resolves():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    made = call(1, answer=2, resolved=True, null=True)
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.turn_end == 2


def test_stale_hash_withheld():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    parent = trace_of(turns, [resolved_pair(1, 2)])
    body = as_dict(parent)
    body["hash"] = ""
    honest = content_hash(body)
    claimed = parent.model_copy(update={"hash": honest})
    changed = claimed.model_copy(deep=True)
    changed.turns[1] = changed.turns[1].model_copy(update={"content": "invented-changed"})
    out = prefixes.select_prefix(changed)
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "stale_hash"


def test_duplicate_answer_withheld():
    turns = [turn(0), turn(1), turn(2, "user", "New Terminal Output:\na")]
    first = call(0, answer=2, resolved=True)
    second = call(1, answer=2, resolved=True)
    out = prefixes.select_prefix(trace_of(turns, [first, second]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "ambiguous_provenance"


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


def test_cli_missing_input(tmp_path):
    assert run_cli(["--input-dir", str(tmp_path / "nope"), "--output", str(tmp_path / "m.json")]) == 2


def test_cli_empty_input(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_cli(["--input-dir", str(empty), "--output", str(tmp_path / "m.json")]) == 2


def test_cli_nondirectory_input(tmp_path):
    target = tmp_path / "file.json"
    target.write_text("{}", encoding="utf-8")
    assert run_cli(["--input-dir", str(target), "--output", str(tmp_path / "m.json")]) == 2


def test_cli_refuses_existing_output(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    invented_cli_input(src, [invented_cli_recording(True)])
    output = tmp_path / "m.json"
    output.write_text("invented-prior", encoding="utf-8")
    assert run_cli(["--input-dir", str(src), "--output", str(output)]) == 2
    assert output.read_text(encoding="utf-8") == "invented-prior"


def test_cli_rejects_output_inside_input(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    invented_cli_input(src, [invented_cli_recording(True)])
    assert run_cli(["--input-dir", str(src), "--output", str(src / "m.json")]) == 2


def test_cli_happy_path(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    invented_cli_input(src, [invented_cli_recording(True), invented_cli_recording(False)])
    output = tmp_path / "m.json"
    assert run_cli(["--input-dir", str(src), "--output", str(output)]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["totals"]["recordings"] == 2
    assert manifest["totals"]["original_eligible"] == 1
    assert manifest["totals"]["selected_prefixes"] == 1
    assert manifest["writer_eligible"] is False
    assert manifest["inputs_unchanged"] is True


def test_root_pointer_foreign():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    parent = trace_of(turns, [resolved_pair(1, 2)])
    other = parent.model_copy(update={"raw_ptr": RawPtr(file_hash="0" * 64, sim_index=SIM)})
    out = prefixes.select_prefix(other)
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "foreign_pointer"


def test_call_owner_foreign():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    made = call(1, answer=2, owner="invented-other")
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "foreign_pointer"


def test_call_owner_absent_allowed():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    made = call(1, answer=2, owner=None)
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.PrefixSelection)


def invented_heldout_file(path, text):
    target = Path(path) / "heldout.json"
    target.write_text(text, encoding="utf-8")
    return str(target)


def test_cli_heldout_string(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    invented_cli_input(src, [invented_cli_recording(False)])
    output = tmp_path / "m.json"
    heldout = invented_heldout_file(tmp_path, '"invented-cli-open"')
    assert run_cli(["--input-dir", str(src), "--output", str(output), "--heldout-ids", heldout]) == 2
    assert not output.exists()


def test_cli_heldout_object(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    invented_cli_input(src, [invented_cli_recording(False)])
    output = tmp_path / "m.json"
    heldout = invented_heldout_file(tmp_path, '{"invented-cli-open": true}')
    assert run_cli(["--input-dir", str(src), "--output", str(output), "--heldout-ids", heldout]) == 2
    assert not output.exists()


def test_cli_heldout_null(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    invented_cli_input(src, [invented_cli_recording(False)])
    output = tmp_path / "m.json"
    heldout = invented_heldout_file(tmp_path, "null")
    assert run_cli(["--input-dir", str(src), "--output", str(output), "--heldout-ids", heldout]) == 2
    assert not output.exists()


def test_cli_heldout_numeric_items(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    invented_cli_input(src, [invented_cli_recording(False)])
    output = tmp_path / "m.json"
    heldout = invented_heldout_file(tmp_path, '["invented-cli-open", 7]')
    assert run_cli(["--input-dir", str(src), "--output", str(output), "--heldout-ids", heldout]) == 2
    assert not output.exists()


def test_cli_heldout_missing(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    invented_cli_input(src, [invented_cli_recording(False)])
    output = tmp_path / "m.json"
    assert (
        run_cli(["--input-dir", str(src), "--output", str(output), "--heldout-ids", str(tmp_path / "gone.json")]) == 2
    )
    assert not output.exists()


def test_cli_heldout_excludes(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    invented_cli_input(src, [invented_cli_recording(False)])
    output = tmp_path / "m.json"
    heldout = invented_heldout_file(tmp_path, '["invented-cli-open"]')
    assert run_cli(["--input-dir", str(src), "--output", str(output), "--heldout-ids", heldout]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["totals"]["selected_prefixes"] == 0
    assert manifest["files"][0]["candidates"] == 1
    assert manifest["heldout_anchor"] == "caller-supplied"
    assert manifest["heldout_ids"] == 1


def test_inputs_error_matches():
    before = [("a.json", 3, "invented-digest")]
    assert cohort_module().inputs_error(before, {"a.json": "invented-digest"}) is None


def test_inputs_error_drift():
    before = [("a.json", 3, "invented-digest")]
    assert cohort_module().inputs_error(before, {"a.json": "invented-changed"}) is not None


def test_inputs_error_missing():
    before = [("a.json", 3, "invented-digest")]
    assert cohort_module().inputs_error(before, {}) is not None


def test_finish_refuses_drift(tmp_path):
    output = tmp_path / "sub" / "m.json"
    before = [("a.json", 3, "invented-digest")]
    totals = {"recordings": 1, "original_eligible": 0, "selected_prefixes": 0}
    code = cohort_module().finish(str(output), before, {"a.json": "invented-changed"}, [], totals, "none", 0)
    assert code == 2
    assert not output.exists()


def test_finish_writes_when_stable(tmp_path):
    output = tmp_path / "sub" / "m.json"
    before = [("a.json", 3, "invented-digest")]
    totals = {"recordings": 1, "original_eligible": 0, "selected_prefixes": 0}
    code = cohort_module().finish(str(output), before, {"a.json": "invented-digest"}, [], totals, "none", 0)
    assert code == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["inputs_unchanged"] is True
