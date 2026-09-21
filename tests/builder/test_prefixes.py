from __future__ import annotations

import copy
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
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


def test_tail_trimmed_after_last_result():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na"), turn(3), turn(4)]
    calls = [resolved_pair(1, 2), call(3, answer=None, resolved=False)]
    out = prefixes.select_prefix(trace_of(turns, calls))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.turn_end == 2
    assert out.retained_calls == (0,)
    assert out.standing == "evidence_only"


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


def test_trailing_empty_done_dropped():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na"), turn(3)]
    calls = [resolved_pair(1, 2)]
    out = prefixes.select_prefix(trace_of(turns, calls))
    assert isinstance(out, prefixes.PrefixSelection)
    assert out.turn_end == 2
    assert 3 not in out.retained_turns


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


@pytest.mark.parametrize("kind", ["result_ptr", "root_ptr", "call_owner"])
def test_foreign_pointer_withheld(kind):
    out = prefixes.select_prefix(invented_foreign_pointer_trace(kind))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "foreign_pointer"


def test_unordered_pointer_withheld():
    made = resolved_pair(1, 2)
    made.result_ptr = ptr(0)
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


def invented_ambiguous_provenance_case(kind):
    if kind == "duplicate_call":
        turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
        calls = [resolved_pair(1, 2), resolved_pair(1, 2)]
    else:
        turns = [turn(0), turn(1), turn(2, "user", "New Terminal Output:\na")]
        calls = [call(0, answer=2, resolved=True), call(1, answer=2, resolved=True)]
    return turns, calls


@pytest.mark.parametrize("kind", ["duplicate_call", "duplicate_answer"])
def test_ambiguous_provenance_withheld(kind):
    turns, calls = invented_ambiguous_provenance_case(kind)
    out = prefixes.select_prefix(trace_of(turns, calls))
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


@pytest.mark.parametrize("kind", ["result_to_assistant", "call_on_user_turn"])
def test_misplaced_pointer_withheld(kind):
    turns, made = invented_misplaced_pointer_case(kind)
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.Withheld)
    assert out.reason == "malformed_pointer"


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


@pytest.mark.parametrize("kind", ["missing", "empty", "nondirectory"])
def test_cli_invalid_input_rejected(tmp_path, kind):
    if kind == "missing":
        src = tmp_path / "nope"
    elif kind == "empty":
        src = tmp_path / "empty"
        src.mkdir()
    else:
        src = tmp_path / "file.json"
        src.write_text("{}", encoding="utf-8")
    assert run_cli(["--input-dir", str(src), "--output", str(tmp_path / "m.json")]) == 2


def test_cli_refuses_existing_output(tmp_path):
    src = invented_cli_src(tmp_path, [invented_cli_recording(True)])
    output = tmp_path / "m.json"
    output.write_text("invented-prior", encoding="utf-8")
    assert run_cli(["--input-dir", str(src), "--output", str(output)]) == 2
    assert output.read_text(encoding="utf-8") == "invented-prior"


def test_cli_rejects_output_inside_input(tmp_path):
    src = invented_cli_src(tmp_path, [invented_cli_recording(True)])
    assert run_cli(["--input-dir", str(src), "--output", str(src / "m.json")]) == 2


def test_cli_happy_path(tmp_path):
    src = invented_cli_src(tmp_path, [invented_cli_recording(True), invented_cli_recording(False)])
    output = tmp_path / "m.json"
    assert run_cli(["--input-dir", str(src), "--output", str(output)]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["totals"]["recordings"] == 2
    assert manifest["totals"]["original_eligible"] == 1
    assert manifest["totals"]["selected_prefixes"] == 1
    assert manifest["writer_eligible"] is False
    assert manifest["inputs_unchanged"] is True


def test_call_owner_absent_allowed():
    turns = [turn(0, "user", "invented-open"), turn(1), turn(2, "user", "New Terminal Output:\na")]
    made = call(1, answer=2, owner=None)
    out = prefixes.select_prefix(trace_of(turns, [made]))
    assert isinstance(out, prefixes.PrefixSelection)


def invented_heldout_file(path, text):
    target = Path(path) / "heldout.json"
    target.write_text(text, encoding="utf-8")
    return str(target)


@pytest.mark.parametrize(
    "heldout_text",
    ['"invented-cli-open"', '{"invented-cli-open": true}', "null", '["invented-cli-open", 7]', None],
    ids=["string", "object", "null", "numeric_items", "missing"],
)
def test_cli_heldout_shape_rejected(tmp_path, heldout_text):
    src = invented_cli_src(tmp_path, [invented_cli_recording(False)])
    output = tmp_path / "m.json"
    if heldout_text is None:
        heldout = str(tmp_path / "gone.json")
    else:
        heldout = invented_heldout_file(tmp_path, heldout_text)
    assert run_cli(["--input-dir", str(src), "--output", str(output), "--heldout-ids", heldout]) == 2
    assert not output.exists()


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


def test_inputs_error_stable_matches():
    before = [("a.json", 3, "invented-digest")]
    assert cohort_module().inputs_error(before, {"a.json": "invented-digest"}) is None


@pytest.mark.parametrize(
    "after",
    [
        {"a.json": "invented-changed"},
        {},
        {"a.json": "invented-digest", "b.json": "invented-extra"},
        {"b.json": "invented-digest"},
    ],
    ids=["drift", "missing", "added", "renamed"],
)
def test_inputs_error_change_detected(after):
    before = [("a.json", 3, "invented-digest")]
    assert cohort_module().inputs_error(before, after) is not None


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


@pytest.mark.parametrize("kind", ["directory", "missing", "regular"])
def test_entry_error_classifies_path(tmp_path, kind):
    if kind == "missing":
        target = tmp_path / "gone.json"
    else:
        target = tmp_path / "x.json"
        if kind == "directory":
            target.mkdir()
        else:
            target.write_text("{}", encoding="utf-8")
    assert (cohort_module().entry_error(target) is not None) is (kind != "regular")


@pytest.mark.parametrize("kind", ["directory", "broken_symlink"])
def test_cli_rejects_nonregular_input(tmp_path, kind):
    src = invented_cli_src(tmp_path, [invented_cli_recording(True)])
    if kind == "directory":
        (src / "extra.json").mkdir()
    else:
        (src / "broken.json").symlink_to(src / "gone.json")
    output = tmp_path / "m.json"
    assert run_cli(["--input-dir", str(src), "--output", str(output)]) == 2
    assert not output.exists()


@pytest.mark.parametrize("func_name", ["evaluate_file", "take_snapshot"])
def test_missing_inputs_report_errors(tmp_path, func_name):
    if func_name == "evaluate_file":
        first, err = cohort_module().evaluate_file(tmp_path / "gone.json", frozenset())
    else:
        first, err = cohort_module().take_snapshot([tmp_path / "gone.json"])
    assert first is None
    assert err is not None


def test_evaluate_file_unreadable(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses file permissions, so chmod-based unreadability cannot be observed")
    target = tmp_path / "x.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o000)
    try:
        entry, err = cohort_module().evaluate_file(target, frozenset())
    finally:
        target.chmod(0o644)
    assert entry is None
    assert err is not None


@pytest.mark.parametrize("func_name", ["take_digest_map", "take_snapshot"])
def test_snapshot_rejects_fifo(tmp_path, func_name):
    if not hasattr(os, "mkfifo"):
        pytest.skip("platform has no FIFO support for the blocking-read regression")
    target = tmp_path / "late.json"
    os.mkfifo(target)
    rows, err = getattr(cohort_module(), func_name)([target])
    assert rows is None
    assert err is not None


@pytest.mark.parametrize("kind", ["directory", "broken_link"])
def test_closing_snapshot_rejects_nonregular(tmp_path, kind):
    target = tmp_path / "late.json"
    if kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(tmp_path / "gone.json")
    rows, err = cohort_module().take_digest_map([target])
    assert rows is None
    assert err is not None


def test_snapshot_roundtrip_regular(tmp_path):
    invented_cli_input(tmp_path, [invented_cli_recording(True)])
    target = tmp_path / "invented.json"
    rows, err = cohort_module().take_snapshot([target])
    assert err is None
    assert rows[0][0] == "invented.json"
    assert rows[0][1] == target.stat().st_size
    digests, err = cohort_module().take_digest_map([target])
    assert err is None
    assert digests["invented.json"] == rows[0][2]


def test_publish_writes_valid_json_without_leftovers(tmp_path):
    output = tmp_path / "m.json"
    assert cohort_module().publish_staged(output, '{"a": 1}') is True
    assert json.loads(output.read_text(encoding="utf-8")) == {"a": 1}
    assert [path.name for path in tmp_path.iterdir()] == ["m.json"]


def test_publish_refuses_existing(tmp_path):
    output = tmp_path / "m.json"
    output.write_bytes(b"invented-prior")
    assert cohort_module().publish_staged(output, '{"a": 1}') is False
    assert output.read_bytes() == b"invented-prior"


def test_finish_unserializable_then_retry(tmp_path):
    output = tmp_path / "sub" / "m.json"
    before = [("a.json", 3, "invented-digest")]
    totals = {"recordings": 1, "original_eligible": 0, "selected_prefixes": 0}
    entries = [{"file": "a.json", "bad": object()}]
    code = cohort_module().finish(str(output), before, {"a.json": "invented-digest"}, entries, totals, "none", 0)
    assert code == 2
    assert not output.exists()
    code = cohort_module().finish(str(output), before, {"a.json": "invented-digest"}, [], totals, "none", 0)
    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["inputs_unchanged"] is True


def test_cli_blocked_parent(tmp_path):
    src = invented_cli_src(tmp_path, [invented_cli_recording(True)])
    blocker = tmp_path / "blocker"
    blocker.write_text("invented-file", encoding="utf-8")
    output = blocker / "m.json"
    assert run_cli(["--input-dir", str(src), "--output", str(output)]) == 2
    assert blocker.read_text(encoding="utf-8") == "invented-file"


def test_publish_concurrent_winner_kept(tmp_path):
    output = tmp_path / "m.json"
    barrier = threading.Barrier(2)
    results = []

    def bidder(text):
        barrier.wait(timeout=60)
        results.append((text, cohort_module().publish_staged(str(output), text)))

    first = threading.Thread(target=bidder, args=('{"bidder": 1}',))
    second = threading.Thread(target=bidder, args=('{"bidder": 2}',))
    first.start()
    second.start()
    first.join(timeout=60)
    second.join(timeout=60)
    assert not first.is_alive() and not second.is_alive()
    wins = [text for text, ok in results if ok]
    losses = [text for text, ok in results if not ok]
    assert len(wins) == 1 and len(losses) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(wins[0])
    assert [path.name for path in tmp_path.iterdir()] == ["m.json"]


def test_publish_encoding_failure_cleans_up(tmp_path):
    output = tmp_path / "m.json"
    with pytest.raises(ValueError):
        cohort_module().publish_staged(str(output), "invented \ud800 text")
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
    assert cohort_module().publish_staged(str(output), '{"retry": true}') is True
    assert json.loads(output.read_text(encoding="utf-8")) == {"retry": True}


def test_publish_rlimit_failure_cleans_up(tmp_path):
    resource = pytest.importorskip("resource")
    if not hasattr(signal, "SIGXFSZ"):
        pytest.skip("platform cannot ignore SIGXFSZ for the bounded write-failure regression")
    del resource
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    output = tmp_path / "m.json"
    code = "\n".join(
        [
            "import os, resource, signal, sys",
            "sys.path.insert(0, sys.argv[1])",
            "resource.setrlimit(resource.RLIMIT_FSIZE, (8192, 8192))",
            "signal.signal(signal.SIGXFSZ, signal.SIG_IGN)",
            "from agenttrove_prefix_cohort import publish_staged",
            "try:",
            "    ok = publish_staged(sys.argv[2], 'x' * 100000)",
            "except Exception as exc:",
            "    print('raised:' + type(exc).__name__)",
            "else:",
            "    print('returned:' + str(ok))",
        ]
    )
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run(
        [sys.executable, "-c", code, str(scripts_dir), str(output)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert "raised:PublishError" in done.stdout
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
    assert cohort_module().publish_staged(str(output), '{"retry": true}') is True
    assert json.loads(output.read_text(encoding="utf-8")) == {"retry": True}
