"""The derivation as the Examiner runs it: derive_all over the Builder's inputs writes what the pipeline's
derive_verifier stage wrote, one Task at a time when asked, never over a store that names a body."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import PTR
from examiner.worlds import anchor_of, make_world, probe_runner_over
from gates import verifier_fixtures as VF
from kullback import sampling
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.builder.pipeline import Anchor
from kullback.examiner import reference, stage
from kullback.gates import verifier_suite
from kullback.gates.artifacts import D79_CHECKS, D79_STAGES
from kullback.gates.ledger import GateLedger
from kullback.runner import budget
from kullback.runner.records import Task, ToolCall, Trace, Verifier

# D218 rule 2 put `round` and `updated_at` on every row of the file, so a reader can see the live file
# move after a round closed its own table. They are on the file and not on the rows the stage answers
# with, which is what `_read` here strips before comparing.
STATUS_KEYS = {"reference_confirmed", "verifier_passed", "reason", "recordings", "rerolls", "judged",
               "assisted_tools", "blocking_tools", "tool_calls_replayed", "tool_calls_differing"}
REFERENCE_KEYS = {"references", "recordings", "failed", "groups", "reason", "judged", "judge_reason",
                  "judge_abstained", "judge_uncited", "abstain_reason", "judge_calls", "judge_fallback",
                  "judge_passes", "judge_second", "judge_residue_resolved", "judge_residue_abstained",
                  "no_correct_recording", "no_correct_key", "judge_citations", "survivor_labels",
                  "residue_derived", "survivor_reason", "survivor_chosen", "survivor_scores",
                  "survivors_capped"}
# The key the world's two End states differ on, and the value the re-roll `wrong` holds at it (D186).
# The tool is the one the shared fixture Runs write with, read from there rather than spelled again.
WRONG_ROW = f"{next(iter(VF.WRITE_TOOLS))}.#w999.order_id"


def _fails_the_wrong_row(reason: str, *, evidence: tuple = ("end_states",)) -> str:
    """A judge reply that fails the state which cancelled the wrong order, cited as D186 asks."""
    return json.dumps({"failed": [{"group": "B", "key": WRONG_ROW, "value": "#W999", "expected": "A",
                                   "reason": reason}],
                       "evidence": list(evidence), "reason": reason})


def _read(path: Path):
    """One record off disk, with D218's round and time stamps taken off each status row.

    The stamps say when the live file last moved and are the file's own (rule 2); what the tests
    below compare is the rows the derivation wrote, which is what the stage answers with.
    """
    body = json.loads(path.read_text(encoding="utf-8"))
    if path.name != "task_status.json":
        return body
    return {task_id: {key: value for key, value in row.items() if key not in stage.STAMPS}
            for task_id, row in body.items()}


def _derive(workdir: Path, inputs: dict, **kwargs) -> dict:
    ctx = stage.ExamContext(workdir, GateLedger(workdir), anchor=kwargs.pop("anchor", None))
    return stage.derive_all(ctx, inputs, **kwargs)


def test_derive_all_over_the_builder_store_writes_the_task_status_and_references_the_pipeline_stage_wrote(
        fixture_build, tmp_path):
    workdir = fixture_build.copy(tmp_path)
    inputs = fixture_build.inputs_for(workdir)
    out = _derive(workdir, inputs, anchor=anchor_of(workdir))
    status = _read(workdir / "task_status.json")
    references = _read(workdir / "references.json")
    task_ids = {task.id for task in inputs["tasks"]}
    assert set(status) == task_ids == set(references) and out["task_status"] == status
    # No Task of the small fixture has a Reference: the rows are the stage's no-Reference rows, each
    # with the reason the setup review reads, and the same rows land in the returned status.
    assert all(set(row) == STATUS_KEYS and row["reference_confirmed"] is False for row in status.values())
    assert all(set(row) == REFERENCE_KEYS and row["references"] == [] for row in references.values())
    assert not (workdir / "verifiers").exists() and out["verifiers"] == []
    stages = [row["stage"] for row in _read(workdir / "gates.json")]
    assert stages[-2:] == ["compile_policy", "derive_verifier"]
    assert (workdir / "constraints_check.json").is_file()


def test_derive_for_one_task_leaves_the_other_tasks_rows_untouched(fixture_build, tmp_path):
    workdir = fixture_build.copy(tmp_path)
    inputs = fixture_build.inputs_for(workdir)
    _derive(workdir, inputs)
    first, *others = [task.id for task in inputs["tasks"]]
    status = _read(workdir / "task_status.json")
    for task_id in others:
        status[task_id]["marker"] = "left alone"
    (workdir / "task_status.json").write_text(json.dumps(status), encoding="utf-8")
    out = _derive(workdir, inputs, only=first)
    after = _read(workdir / "task_status.json")
    assert out["task_status"] == after and set(after) == {first, *others}
    assert "marker" not in after[first] and set(after[first]) == STATUS_KEYS
    assert all(after[task_id]["marker"] == "left alone" for task_id in others)
    assert set(_read(workdir / "references.json")) == set(after)
    with pytest.raises(ValueError, match="no Task is named"):
        _derive(workdir, inputs, only="no-such-task")


def test_a_task_without_a_reference_gets_no_verifier_and_names_only_the_tools_its_own_calls_differ_on(
        fixture_build, tmp_path):
    workdir = fixture_build.copy(tmp_path)
    inputs = fixture_build.inputs_for(workdir)
    assisted = set(inputs["assisted_tools"])
    assert assisted, "the fixture build has at least one assisted tool"
    status = _derive(workdir, inputs)["task_status"]
    named = {task_id: row for task_id, row in status.items() if row["assisted_tools"]}
    assert named, "one Task's seed Trace calls an assisted tool"
    for row in named.values():
        assert row["reference_confirmed"] is False and row["verifier_passed"] is False
        assert set(row["assisted_tools"]) <= assisted
        # D171: the corpus fact stays on the row, and only a tool one of this Task's own recorded
        # calls parts from blocks it; the reason names those tools and no others.
        assert set(row["blocking_tools"]) <= assisted
        assert set(row["blocking_tools"]) == set(row["tool_calls_differing"])
        assert all(tool in row["reason"] for tool in row["blocking_tools"])
        if not row["blocking_tools"]:
            assert "do not replay" not in row["reason"]
    assert not (workdir / "verifiers").exists()


def _library_trace(trace_id: str, tools: list) -> Trace:
    """One Trace of a library agent, for the rows that read which tools a Task's Runs called."""
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="1", source="test", raw_ptr=PTR,
                 tool_calls=[ToolCall(name=name, args={}, raw_ptr=PTR) for name in tools])


def _no_reference_row(tmp_path, fidelity_row: dict, reason: str = "recordings disagree on the End state") -> dict:
    ctx = stage.ExamContext(tmp_path, GateLedger(tmp_path))
    return stage.no_reference_status(
        ctx, Task(id="t1", run_ids=["r1"]), SimpleNamespace(reason=reason, judged=False),
        seed_replays=[{"path": "p"}], replays={}, rerolls={},
        traces={"r1": _library_trace("r1", ["get_loan_details"])},
        assisted_tools={"get_loan_details"}, fidelity_row=fidelity_row)


def test_a_task_whose_own_calls_all_replay_is_not_blocked_by_a_miss_on_another_tasks_call(tmp_path):
    """D171: assisted is a corpus ruling over every recorded call of a tool. A Task the body answers
    the way the recording did is not blocked by it, and its row keeps the real reason it failed."""
    row = _no_reference_row(tmp_path, {"get_loan_details": {"replayed": 4, "differing": 0, "reasons": []}})
    assert row["assisted_tools"] == ["get_loan_details"] and row["blocking_tools"] == []
    assert row["reason"] == "recordings disagree on the End state"
    assert row["tool_calls_replayed"] == {"get_loan_details": 4} and row["tool_calls_differing"] == {}


def test_a_task_with_a_differing_own_call_stays_blocked_and_its_reason_names_the_call(tmp_path):
    differs = "get_loan_details({'loan_id': 'L9'}): hard columns differ: due_on: ours 2099-12-31, recorded 2026-01-05"
    row = _no_reference_row(tmp_path, {"get_loan_details": {"replayed": 3, "differing": 1, "reasons": [differs]}})
    assert row["blocking_tools"] == ["get_loan_details"]
    assert "get_loan_details" in row["reason"] and "due_on" in row["reason"]
    assert row["reason"].endswith("recordings disagree on the End state")
    assert row["tool_calls_differing"] == {"get_loan_details": 1}


def test_a_task_gets_its_reference_on_the_body_of_a_tool_that_is_assisted_over_the_corpus(tmp_path):
    """The Reference is built on the body, not a stand-in, when the Task's own calls all replay; the
    row records which calls the body was exercised on, and the D79 suite still judges the Verifier."""
    world = make_world(tmp_path)
    fidelity = {"tools": {"get_loan_details": {"calls": 4, "replayed": 3, "differing": 1, "assisted": True}},
                "tasks": {"t1": {"get_loan_details": {"replayed": 3, "differing": 0, "reasons": []}}}}
    out = _derive(world.workdir, dict(world.inputs, tool_fidelity=fidelity, assisted_tools=["get_loan_details"]),
                  probe_model=object(), run_probe=probe_runner_over())
    row = out["task_status"]["t1"]
    assert row["reference_confirmed"] is True and row["verifier_passed"] is True
    assert row["tool_calls_replayed"] == {"get_loan_details": 3} and row["tool_calls_differing"] == {}


def test_the_suite_for_a_repaired_verifier_runs_every_d79_check_the_derivation_ran(derived):
    verifier = Verifier.model_validate(_read(derived.workdir / "verifiers" / "t1.json"))
    status = _read(derived.workdir / "task_status.json")["t1"]
    task = derived.inputs["tasks"][0]
    gates = stage.suite_for(task, verifier, [derived.paths["ref"], derived.paths["alt"]], canon_rules=None,
                            write_tools=VF.WRITE_TOOLS, user_rules={}, rules_trace="ref", probe_model=object(),
                            run_probe=probe_runner_over(), may_probe=True)
    assert [g.stage for g in gates] == list(D79_STAGES)
    assert set(status["checks"]) == set(D79_CHECKS) and all(status["checks"].values())
    assert all(g.passed for g in gates)


def test_the_context_seeds_from_the_anchor_when_given_and_from_every_run_without_one(fixture_build, tmp_path):
    # The fixture is too small to hold a Run out of any Task, so its anchor seeds from every Run;
    # a build large enough holds a share out, and the context seeds from the rest (D81).
    task = fixture_build.inputs["tasks"][0]
    assert set(task.run_ids) and not anchor_of(fixture_build.workdir).held_out.get(task.id)
    plain = stage.ExamContext(tmp_path, GateLedger(tmp_path))
    assert plain.seed_runs(task.id, task.run_ids) == list(task.run_ids)
    empty = stage.ExamContext(tmp_path, GateLedger(tmp_path), anchor=anchor_of(fixture_build.workdir))
    assert empty.seed_runs(task.id, task.run_ids) == list(task.run_ids)
    held = Task(id="t", run_ids=["a", "b", "c"])
    anchored = stage.ExamContext(tmp_path, GateLedger(tmp_path),
                                 anchor=Anchor(held_out={"t": ["b"]}, unguarded=[]))
    assert anchored.seed_runs(held.id, held.run_ids) == ["a", "c"]
    assert stage.seed_ids(anchored, held) == {"a", "c"} and stage.seed_ids(plain, held) == {"a", "b", "c"}


def test_the_loophole_probe_runs_through_the_runner_callable_given_and_the_inputs_never_hold_bodies(tmp_path):
    world = make_world(tmp_path)
    seen = []

    def run_probe(model, verifier):
        seen.append((model, verifier))
        return VF.wrong_run()

    marker = object()
    out = _derive(world.workdir, world.inputs, probe_model=marker, run_probe=run_probe)
    assert len(seen) == 1 and seen[0][0] is marker and seen[0][1].task_id == "t1"
    assert out["task_status"]["t1"]["checks"]["loophole_probe_fails"] is True
    assert not set(stage.FORBIDDEN_INPUTS) & set(world.inputs)
    # Without a model the callable is never run and the check is listed as not run, not as passed.
    quiet = make_world(tmp_path / "quiet")
    out = _derive(quiet.workdir, quiet.inputs, run_probe=run_probe)
    assert len(seen) == 1 and "verifier_loophole" in out["task_status"]["t1"]["not_run"]
    assert out["task_status"]["t1"]["verifier_passed"] is False


def _counted_probe(calls: list):
    """A `run_probe` that records which Task it was asked for, so a Run made twice is visible."""
    def run_probe(model, verifier):
        calls.append(verifier.task_id)
        return VF.wrong_run()

    return run_probe


def _derived_bytes(workdir: Path, tasks: int) -> dict:
    """What the derivation wrote, without D218's round and time stamps.

    `updated_at` is a wall clock, so two runs of the same derivation write it differently and always
    will; the claim these bytes carry is that the rows themselves are the same, which is what the
    stamps are stripped for here and nowhere else.
    """
    names = ["task_status.json", "references.json"] + [f"verifiers/t{n}.json" for n in range(1, tasks + 1)]
    return {name: without_stamps(workdir / name) for name in names}


def without_stamps(path: Path) -> bytes:
    """One record's bytes, with each status row's `round` and `updated_at` taken off (D218 rule 2)."""
    body = path.read_bytes()
    if path.name != "task_status.json":
        return body
    rows = {task_id: {key: value for key, value in row.items() if key not in stage.STAMPS}
            for task_id, row in json.loads(body).items()}
    return json.dumps(rows, indent=2, sort_keys=True, default=str).encode("utf-8")


def test_derive_all_on_four_workers_writes_the_task_status_references_and_verifiers_one_worker_writes(tmp_path):
    serial = make_world(tmp_path / "serial", tasks=4)
    threaded = make_world(tmp_path / "threaded", tasks=4)
    for world, workers in ((serial, 1), (threaded, 4)):
        out = _derive(world.workdir, world.inputs, probe_model=object(), run_probe=probe_runner_over(),
                      workers=workers)
        assert out["ran"] == 4 and out["cached"] == 0 and len(out["verifiers"]) == 4
    assert _derived_bytes(serial.workdir, 4) == _derived_bytes(threaded.workdir, 4)


def test_a_second_derive_over_unchanged_inputs_runs_no_task_probes_none_again_and_reports_every_task_cached(tmp_path):
    world = make_world(tmp_path, tasks=3)
    calls: list[str] = []
    first = _derive(world.workdir, world.inputs, probe_model=object(), run_probe=_counted_probe(calls), workers=2)
    assert first["ran"] == 3 and first["cached"] == 0 and sorted(calls) == ["t1", "t2", "t3"]
    before = _derived_bytes(world.workdir, 3)
    second = _derive(world.workdir, world.inputs, probe_model=object(), run_probe=_counted_probe(calls), workers=2)
    assert second["cached"] == 3 and second["ran"] == 0
    assert sorted(calls) == ["t1", "t2", "t3"], "the loophole probe's Run is not made again for a cached Task"
    assert second["task_status"] == first["task_status"]
    assert [v.task_id for v in second["verifiers"]] == [v.task_id for v in first["verifiers"]]
    assert _derived_bytes(world.workdir, 3) == before


def test_only_the_task_whose_intent_changed_is_derived_again(tmp_path):
    world = make_world(tmp_path, tasks=3)
    calls: list[str] = []
    _derive(world.workdir, world.inputs, probe_model=object(), run_probe=_counted_probe(calls), workers=2)
    moved = dict(world.inputs, intents={"t2": {"task_id": "t2", "grounded": True,
                                               "text": "cancel the pending order and say it is cancelled"}})
    out = _derive(world.workdir, moved, probe_model=object(), run_probe=_counted_probe(calls), workers=2)
    assert out["ran"] == 1 and out["cached"] == 2
    assert calls[3:] == ["t2"], "only the Task whose Intent moved is derived and probed again"
    assert len(out["verifiers"]) == 3, "the other two Tasks' Verifiers come back off the cache"


def test_a_task_whose_rerolls_changed_is_derived_again(tmp_path):
    world = make_world(tmp_path, tasks=3)
    calls: list[str] = []
    first = _derive(world.workdir, world.inputs, probe_model=object(), run_probe=_counted_probe(calls), workers=2)
    assert first["task_status"]["t3"]["rerolls"] == 1
    rerolls = {task_id: [dict(row) for row in rows] for task_id, rows in world.inputs["rerolls"].items()}
    rerolls["t3"].append({"run_id": "ref2", "path": world.paths["ref"], "termination_reason": "success"})
    out = _derive(world.workdir, dict(world.inputs, rerolls=rerolls), probe_model=object(),
                  run_probe=_counted_probe(calls), workers=2)
    assert out["ran"] == 1 and out["cached"] == 2 and calls[3:] == ["t3"]
    assert out["task_status"]["t3"]["rerolls"] == 2 and out["task_status"]["t1"]["rerolls"] == 1


def test_the_code_hash_is_part_of_the_key_so_an_edited_module_derives_every_task_again(tmp_path):
    world = make_world(tmp_path, tasks=2)

    def derive(**kwargs):
        return _derive(world.workdir, world.inputs, probe_model=object(), run_probe=probe_runner_over(), **kwargs)

    assert derive(code_hash="the modules as they stand")["ran"] == 2
    assert derive(code_hash="the modules as they stand")["cached"] == 2
    assert derive(code_hash="one module edited since")["ran"] == 2, "a changed module is a changed derivation"
    # The default is the bytes of this module, derive.py, reference.py and verifier_suite.py, and it
    # is the same input to the key: a key written under a pinned hash is not read back under it.
    assert derive()["ran"] == 2 and derive()["cached"] == 2
    assert stage.module_code_hash() not in ("the modules as they stand", "one module edited since")


def test_a_half_written_cache_entry_is_a_miss_and_not_a_failed_derivation(tmp_path):
    world = make_world(tmp_path, tasks=2)
    _derive(world.workdir, world.inputs, probe_model=object(), run_probe=probe_runner_over())
    entries = sorted((world.workdir / "examiner" / "cache").rglob("*.json"))
    assert [path.parent.name for path in entries] == ["t1", "t2"], "one entry per Task, under its id"
    entries[0].write_text('{"format": 1, "task_id": "t1", "key"', encoding="utf-8")  # killed mid-write
    out = _derive(world.workdir, world.inputs, probe_model=object(), run_probe=probe_runner_over())
    assert out["ran"] == 1 and out["cached"] == 1 and len(out["verifiers"]) == 2


@pytest.mark.parametrize("name", stage.FORBIDDEN_INPUTS[:4])
def test_inputs_from_refuses_a_store_that_names_bodies_db_schema_or_the_environment(name):
    store = {"tasks": [Task(id="t")], "sigs": [], "constraints": [], name: {"anything": 1}}
    with pytest.raises(ValueError, match=name):
        stage.inputs_from(store)
    del store[name]
    store["policy_text"] = "the policy"
    with pytest.raises(ValueError, match="policy_text"):
        stage.inputs_from(store)
    del store["policy_text"]
    assert set(stage.inputs_from(dict(store, extra="ignored"))) == {"tasks", "sigs", "constraints"}


def test_derive_all_rewrites_the_scorecard_after_the_task_status(tmp_path):
    world = make_world(tmp_path)
    (world.workdir / "tasks.json").write_text(json.dumps({"tasks": [{"id": "t1", "run_ids": ["ref"]}]}),
                                             encoding="utf-8")
    (world.workdir / "runs.json").write_text(json.dumps({"runs": [{"run_id": "ref", "task_id": "t1"}]}),
                                            encoding="utf-8")
    (world.workdir / "scorecard.json").write_text("{}", encoding="utf-8")
    _derive(world.workdir, world.inputs, probe_model=object(), run_probe=probe_runner_over())
    status_at = os.stat(world.workdir / "task_status.json").st_mtime_ns
    card_at = os.stat(world.workdir / "scorecard.json").st_mtime_ns
    assert card_at >= status_at
    coverage = _read(world.workdir / "scorecard.json")["task_coverage"]
    assert coverage["tasks_covered"] == 1 and coverage["uncovered"] == [], \
        "the card counts the Task the status just verdicted"


# --- the judge the derivation builds (D12, D185) ---------------------------------------

def test_the_derivation_hands_the_task_to_the_agent_judge_and_the_record_says_what_it_looked_at(tmp_path):
    """A Task whose Runs ended in two states goes to the judge, and derive_all is what builds it: the
    reference record carries the tool it called before it failed a state."""
    world = make_world(tmp_path, rerolls=("wrong",))
    model = TestModel([
        ModelReply(content=None, tool_calls=[ToolCallRequest(id="c1", name="rows", arguments={"group": "B"})]),
        ModelReply(content=_fails_the_wrong_row("it wrote to another row", evidence=("rows",)))])
    out = _derive(world.workdir, world.inputs, judge_model=model, judge_agent=True)
    row = _read(world.workdir / "references.json")["t1"]
    assert row["judged"] and row["judge_fallback"] is None and not row["judge_abstained"]
    assert [call["tool"] for call in row["judge_calls"]] == ["rows"]
    assert [r["run_id"] for r in row["references"]] == ["ref"] and list(row["failed"]) == ["wrong"]
    assert out["task_status"]["t1"]["reference_confirmed"] is True


def test_the_agent_judge_is_built_only_when_the_derivation_is_asked_for_it(tmp_path, monkeypatch):
    """D185: the one-shot judge is the default and the agent is an opt-in, so a build that names a
    judge model and nothing else never constructs one."""
    built = []
    real = stage.judge_mod.AgentJudge
    monkeypatch.setattr(stage.judge_mod, "AgentJudge",
                        lambda model, **kw: built.append(model) or real(model, **kw))

    world = make_world(tmp_path, rerolls=("wrong",))
    _derive(world.workdir, world.inputs,
            judge_model=TestModel([_fails_the_wrong_row("it wrote elsewhere")]))
    assert built == [], "the default judge is one call over the prompt, not an agent"
    one_shot = _read(world.workdir / "references.json")["t1"]
    assert one_shot["judged"] and one_shot["judge_calls"] == [] and one_shot["judge_fallback"] is None

    asked = make_world(tmp_path / "asked", rerolls=("wrong",))
    _derive(asked.workdir, asked.inputs, judge_agent=True,
            judge_model=TestModel([_fails_the_wrong_row("it wrote elsewhere", evidence=("rows",))]))
    assert len(built) == 1, "the flag is what builds the judge with a bounded look"


def test_the_round_counts_say_how_many_judgements_cited_nothing_the_states_differ_on(tmp_path):
    """D186's count sits beside `judged`, so a round can be read for how much of the judge's work
    landed somewhere it was never shown; the reason is on the reference row."""
    shared = json.dumps({"failed": [{"group": "B", "key": "stated:#W123", "value": "told to the user",
                                     "expected": "A",
                                     "reason": "the buyer never confirmed the cancellation"}],
                         "evidence": ["end_states"], "reason": "no confirmation"})
    world = make_world(tmp_path, rerolls=("wrong",))
    _derive(world.workdir, world.inputs, judge_model=TestModel([shared]))
    metrics = _ruling(world.workdir)["metrics"]
    assert metrics["judged"] == 1 and metrics["judge_uncited"] == 1
    row = _read(world.workdir / "references.json")["t1"]
    assert row["judge_uncited"] and not any(why.startswith("judge:") for why in row["failed"].values())
    assert row["abstain_reason"] == "a failure rested on nothing the states differ on: stated:#W123"
    # The judgement failed nothing, so both states are still in and the derivation is what settles
    # them (D198); the D186 accounting above is unchanged by that.
    assert row["residue_derived"] and row["survivor_labels"] == ["A", "B"]

    kept = make_world(tmp_path / "kept", rerolls=("wrong",))
    _derive(kept.workdir, kept.inputs, judge_model=TestModel([_fails_the_wrong_row("it wrote elsewhere")]))
    cited = _ruling(kept.workdir)["metrics"]
    assert cited["judged"] == 1 and cited["judge_uncited"] == 0 and cited["references"] == 1


def test_which_judge_ruled_is_in_the_cache_key_so_turning_the_agent_on_derives_again(tmp_path):
    """The judge's name is in the key and so is which judge it is: the one-shot answer must not be
    read back off disk for a run that asked for the agent."""
    world = make_world(tmp_path, rerolls=("wrong",))
    common = {"format": stage.CACHE_FORMAT, "judge": "vendor/small"}
    task, recordings = world.inputs["tasks"][0], []
    keys = {agent: stage.cache_key(task, recordings, {**common, "judge_agent": agent},
                                   intents={}, user_rules={}, traces={})
            for agent in (False, True)}
    assert keys[False] != keys[True]


# --- the false-rejection pool: only the Runs that did the Task (D133) -----------------

def _answering_run(run_id: str = "held"):
    """A finished Run that read nothing and wrote nothing: it did not do a Task that asks for a write."""
    return VF.make_run(run_id, [VF.user("Please cancel it."), VF.assistant("I am not able to do that.")])


def _world_with_a_held_out_run(tmp_path, run):
    """The one-Task world with a second recording of it, held out as the Task's anchor (D81)."""
    world = make_world(tmp_path, rerolls=("alt",))
    path = VF.write_events_jsonl(run, world.workdir / "runs" / "t1" / f"{run.run_id}.jsonl")
    world.inputs["replays"]["t1"][run.run_id] = {"trace_id": run.run_id, "run_id": run.run_id,
                                                 "confirmed": True, "path": path, "reasons": []}
    world.inputs["tasks"] = [Task(id="t1", intent=VF.TASK.intent, run_ids=["ref", run.run_id])]
    return world, Anchor(held_out={"t1": [run.run_id]}, unguarded=[])


def _ruling(workdir: Path, stage_name: str = "derive_verifier") -> dict:
    return [row for row in _read(workdir / "gates.json") if row["stage"] == stage_name][-1]


def test_the_pool_leaves_out_a_held_out_run_that_wrote_nothing_on_a_task_that_asks_for_a_write(tmp_path):
    """D133 counted a held-out Run legitimate on its success termination alone, so a Run that answered
    and stopped made the Verifier that caught it read as over-strict."""
    world, anchor = _world_with_a_held_out_run(tmp_path, _answering_run())
    out = _derive(world.workdir, world.inputs, anchor=anchor)
    row = out["task_status"]["t1"]
    assert row["did_not_reach_reference"] == ["held"]
    assert row["failed_recordings"]["held"] == stage.WROTE_NOTHING
    assert _ruling(world.workdir)["metrics"]["did_not_reach_reference"] == 1


def test_the_pool_keeps_a_held_out_run_whose_settled_end_state_is_the_references(tmp_path):
    """The same writes by another route (D46) are the same End state, and that Run is what the number
    is measured over."""
    world, anchor = _world_with_a_held_out_run(tmp_path, VF.alt_path_run().model_copy(
        update={"run_id": "held"}))
    out = _derive(world.workdir, world.inputs, anchor=anchor)
    row = out["task_status"]["t1"]
    assert row["did_not_reach_reference"] == [] and "held" not in row["failed_recordings"]
    assert _ruling(world.workdir)["metrics"]["did_not_reach_reference"] == 0


def test_a_held_out_run_that_wrote_to_another_entity_is_left_out_with_its_own_reason(tmp_path):
    world, anchor = _world_with_a_held_out_run(tmp_path, VF.wrong_run())
    row = _derive(world.workdir, world.inputs, anchor=anchor)["task_status"]["t1"]
    assert row["failed_recordings"]["wrong"] == stage.WROTE_OTHERWISE


# --- the second path a lone Reference is re-rolled for (D189) --------------------------

def _reroll_runner_over(world, make_run, calls: list):
    """A `run_rerolls(task_id, count, prefix)` that writes one batch of hand-built Runs to the Task's
    folder and answers the rows the Builder's callable answers, so no model is asked."""
    def run_rerolls(task_id: str, count: int, prefix: str) -> list:
        calls.append((task_id, count, prefix))
        rows = []
        for number in range(count):
            run = make_run().model_copy(deep=True, update={"run_id": f"{prefix}-{task_id}-{number}"})
            path = VF.write_events_jsonl(run, world.workdir / "runs" / task_id / f"{run.run_id}.jsonl")
            rows.append({"run_id": run.run_id, "path": path,
                         "termination_reason": run.termination_reason})
        return rows

    return run_rerolls


def _lone_reference_world(root: Path):
    """The one-Task world with nothing beside its recording: check 5 has no second path to score."""
    return make_world(root, rerolls=())


def test_a_task_whose_reference_stands_alone_buys_one_batch_and_stops_when_a_second_path_appears(tmp_path):
    world = _lone_reference_world(tmp_path)
    calls: list = []
    out = _derive(world.workdir, world.inputs, run_rerolls=_reroll_runner_over(world, VF.alt_path_run, calls),
                  round_number=2)
    row = out["task_status"]["t1"]
    assert len(calls) == 1 and calls[0][:2] == ("t1", stage.SECOND_PATH_RUNS)
    assert calls[0][2] == "second-path-r2-b1", "the batch names the round and the batch it is"
    assert row["second_path"] == stage.second_path_row(1, stage.SECOND_PATH_RUNS, found=True)
    assert row["checks"]["second_path_passes"] is True and "verifier_alt_path" not in row["not_run"]
    assert out["second_path_runs"] == stage.SECOND_PATH_RUNS and out["ceiling_reached"] is False
    metrics = _ruling(world.workdir)["metrics"]
    assert metrics["second_path_found"] == 1 and metrics["second_path_exhausted"] == 0
    # The batch's Runs join the Task's Runs like any other re-roll, under the reason that bought them.
    rows = _read(world.workdir / "examiner" / "rerolls.json")["t1"]
    assert [row["reason"] for row in rows] == ["second_path"] * stage.SECOND_PATH_RUNS
    assert {row["batch"] for row in rows} == {1}


def test_the_cap_holds_and_the_task_keeps_the_check_not_run_with_the_reason_it_spent(tmp_path):
    """A Run that reached another End state is not a second path however many batches buy one, so the
    Task keeps its Reference, keeps check 5 not run, and its row says what the search cost."""
    world = _lone_reference_world(tmp_path)
    calls: list = []
    out = _derive(world.workdir, world.inputs, run_rerolls=_reroll_runner_over(world, VF.wrong_run, calls))
    row = out["task_status"]["t1"]
    assert len(calls) == stage.SECOND_PATH_BATCHES == 3
    assert row["second_path"] == stage.second_path_row(3, 3 * stage.SECOND_PATH_RUNS, found=False)
    assert row["second_path"]["reason"] == "no second path in 3 batches"
    assert row["reference_confirmed"] is True, "a batch that landed elsewhere never unseats the Reference"
    assert row["checks"]["second_path_passes"] is False and "verifier_alt_path" in row["not_run"]
    assert len(row["did_not_reach_reference"]) == 3 * stage.SECOND_PATH_RUNS
    assert _ruling(world.workdir)["metrics"]["second_path_exhausted"] == 1


def test_the_batches_are_read_back_on_a_re_run_and_no_further_batch_is_bought(tmp_path):
    world = _lone_reference_world(tmp_path)
    calls: list = []
    runner = _reroll_runner_over(world, VF.alt_path_run, calls)
    first = _derive(world.workdir, world.inputs, run_rerolls=runner)
    assert len(calls) == 1 and first["task_status"]["t1"]["second_path"]["found"] is True
    cached = _derive(world.workdir, world.inputs, run_rerolls=runner)
    assert cached["cached"] == 1 and len(calls) == 1, "the Task's key holds and it is served from disk"
    # With the cache gone the Task derives again, and it still buys nothing: the rows the reason
    # names are read back off the Examiner's own re-roll file and give check 5 its second path.
    shutil.rmtree(world.workdir / "examiner" / "cache")
    again = _derive(world.workdir, world.inputs, run_rerolls=runner)
    row = again["task_status"]["t1"]
    assert again["ran"] == 1 and len(calls) == 1
    assert row["second_path"] == stage.second_path_row(0, 0, found=True)
    assert row["checks"]["second_path_passes"] is True


def test_a_task_that_already_has_a_second_path_buys_no_batch(tmp_path):
    world = make_world(tmp_path, rerolls=("alt",))
    calls: list = []
    out = _derive(world.workdir, world.inputs, run_rerolls=_reroll_runner_over(world, VF.alt_path_run, calls))
    row = out["task_status"]["t1"]
    assert calls == [] and row["second_path"]["batches"] == 0 and row["second_path"]["found"] is True
    assert row["checks"]["second_path_passes"] is True
    assert not (world.workdir / "examiner" / "rerolls.json").exists()


def test_without_a_reroll_runner_the_check_stays_not_run_and_the_row_says_why(tmp_path):
    world = _lone_reference_world(tmp_path)
    row = _derive(world.workdir, world.inputs)["task_status"]["t1"]
    assert row["second_path"] == stage.second_path_row(0, 0, found=False)
    assert row["second_path"]["reason"] == stage.NO_REROLL_RUNNER
    assert "verifier_alt_path" in row["not_run"]


def test_a_batch_that_hits_the_run_ceiling_stops_the_search_and_the_call_says_so(tmp_path):
    world = _lone_reference_world(tmp_path)

    def run_rerolls(task_id, count, prefix):
        raise budget.BudgetExceeded("reroll", task_id, 2.0, 1.0, 0.5)

    out = _derive(world.workdir, world.inputs, run_rerolls=run_rerolls)
    row = out["task_status"]["t1"]
    assert out["ceiling_reached"] is True and row["reference_confirmed"] is True
    assert row["second_path"] == stage.second_path_row(0, 0, found=False, reason=stage.CEILING_REACHED)


# --- the second path synthesised from the Reference Run (D199) --------------------------

def _variant_runner_over(world, make_run, seen: list):
    """A `run_variant(task_id, calls, run_id, answer)` that writes one hand-built Run to the Task's
    folder, so the stage's rule is exercised without an Environment to replay through."""
    def run_variant(task_id: str, calls: list, run_id: str, transcript: list = ()) -> dict:
        seen.append({"task_id": task_id, "run_id": run_id, "transcript": list(transcript),
                     "calls": [call["name"] for call in calls]})
        run = make_run().model_copy(deep=True, update={"run_id": run_id})
        path = VF.write_events_jsonl(run, world.workdir / "runs" / task_id / f"{run.run_id}.jsonl")
        return {"run_id": run.run_id, "path": path, "termination_reason": run.termination_reason}

    return run_variant


def _one_call_run(run_id: str = "ref"):
    """The Reference's events with its only read taken out: one call, so no rewrite of it exists."""
    events = [e for e in VF.reference_events()
              if e["payload"].get("name") != "get_order_details" and e["payload"].get("id") != "c0"]
    return VF.make_run(run_id, events)


def test_a_lone_reference_whose_call_path_can_be_rewritten_gets_a_second_path_written_from_it(tmp_path):
    """The buying found nothing, so the Reference's own call path is rewritten, replayed, and kept
    because it reaches the same End state; check 5 is scored on it like any second path."""
    world = _lone_reference_world(tmp_path)
    bought: list = []
    seen: list = []
    out = _derive(world.workdir, world.inputs,
                  run_rerolls=_reroll_runner_over(world, VF.wrong_run, bought),
                  run_variant=_variant_runner_over(world, VF.alt_path_run, seen), round_number=2)
    row = out["task_status"]["t1"]
    assert len(bought) == stage.SECOND_PATH_BATCHES, "the buying is spent before anything is written"
    assert [call["calls"] for call in seen] == [["get_order_details", "get_order_details",
                                                 next(iter(VF.WRITE_TOOLS))]]
    assert [turn["role"] for turn in seen[0]["transcript"]] == ["user", "assistant", "user", "assistant"], \
        "a variant varies the call path inside the conversation the Run it came from had"
    assert seen[0]["transcript"][-1]["content"] == "Your order #W123 is cancelled and 150.0 is refunded."
    assert row["second_path"]["synthesised"] == 1 and row["second_path"]["synth_kinds"] == ["insert_read"]
    assert row["second_path"]["synth_tried"] == 1 and row["second_path"]["found"] is True
    assert row["second_path"]["structural"] is False
    assert row["checks"]["second_path_passes"] is True and "verifier_alt_path" not in row["not_run"]
    metrics = _ruling(world.workdir)["metrics"]
    assert metrics["second_path_synthesised"] == 1 and metrics["single_path_by_structure"] == 0
    assert metrics["synth_variants_tried"] == 1 and metrics["synth_variants_kept"] == 1


def test_a_rewrite_that_lands_on_another_end_state_is_not_kept_and_the_check_stays_not_run(tmp_path):
    world = _lone_reference_world(tmp_path)
    seen: list = []
    out = _derive(world.workdir, world.inputs,
                  run_variant=_variant_runner_over(world, VF.wrong_run, seen))
    row = out["task_status"]["t1"]
    assert len(seen) == 1 and row["second_path"]["synth_tried"] == 1
    assert row["second_path"]["synth_kept"] == 0 and row["second_path"]["found"] is False
    assert row["second_path"]["reason"] == stage.SYNTH_KEPT_NONE
    assert row["checks"]["second_path_passes"] is False and row["verifier_passed"] is False
    assert _ruling(world.workdir)["metrics"]["synth_variants_kept"] == 0


def test_a_path_single_by_structure_is_named_and_stops_blocking_when_the_pool_holds_a_run_at_the_reference(
        tmp_path):
    """A Reference of one call offers no rewrite: no pair to reorder, no read to insert, none to drop.
    The Task says so, and the held-out Run that reached the same End state is the second path."""
    world = make_world(tmp_path, rerolls=())
    VF.write_events_jsonl(_one_call_run(), Path(world.paths["ref"]))
    held = _one_call_run("held")
    path = VF.write_events_jsonl(held, world.workdir / "runs" / "t1" / "held.jsonl")
    world.inputs["replays"]["t1"]["held"] = {"trace_id": "held", "run_id": "held", "confirmed": True,
                                             "path": path, "reasons": []}
    anchor = Anchor(held_out={"t1": ["held"]}, unguarded=[])
    world.inputs["tasks"] = [Task(id="t1", intent=VF.TASK.intent, run_ids=["ref", "held"])]
    seen: list = []
    out = _derive(world.workdir, world.inputs, anchor=anchor,
                  probe_model=object(), run_probe=probe_runner_over(),
                  run_variant=_variant_runner_over(world, VF.alt_path_run, seen))
    row = out["task_status"]["t1"]
    assert seen == [], "there is nothing to replay, so nothing is replayed"
    assert row["second_path"]["structural"] is True
    assert row["second_path"]["reason"] == stage.SINGLE_PATH_BY_STRUCTURE
    assert row["checks"]["second_path_passes"] is False and "verifier_alt_path" in row["not_run"]
    assert row["pool_at_reference"] == ["held"] and row["second_path_waived"] is True
    assert row["verifier_passed"] is True, "the pool holds the second path this Task cannot have"
    metrics = _ruling(world.workdir)["metrics"]
    assert metrics["single_path_by_structure"] == 1 and metrics["second_path_waived"] == 1


def test_a_path_single_by_structure_keeps_blocking_when_no_pool_run_reached_the_reference(tmp_path):
    world = make_world(tmp_path, rerolls=())
    VF.write_events_jsonl(_one_call_run(), Path(world.paths["ref"]))
    row = _derive(world.workdir, world.inputs, probe_model=object(), run_probe=probe_runner_over(),
                  run_variant=_variant_runner_over(world, VF.alt_path_run, []))["task_status"]["t1"]
    assert row["second_path"]["structural"] is True and row["pool_at_reference"] == []
    assert row["second_path_waived"] is False and row["verifier_passed"] is False


# --- a residue settled by deriving a Verifier per survivor (D198) ------------------

def _survivor(label: str, *, checks_passed: int, failed: tuple = (), fraction=None, held_out: int = 0) -> dict:
    """One survivor's row as `settle_residue` writes it, for the choice on its own."""
    return {"label": label, "runs": 1, "checks_passed": checks_passed, "checks_failed": list(failed),
            "checks_not_run": [], "passed": not failed, "false_rejection": fraction,
            "held_out": held_out}


def test_the_survivor_whose_verifier_fails_a_check_loses_to_the_one_whose_verifier_stands_up():
    """The judge is out of the question here: the choice is the suite's, so a survivor whose
    Verifier a mutation of the Reference slips past is not the state the Task is confirmed on."""
    rows = [_survivor("A", checks_passed=8, failed=("mutation_flips",)), _survivor("B", checks_passed=9)]
    best, why, said = stage.choose_survivor(rows, ["A", "B"])
    assert best["label"] == "B" and why == "survivor_chosen"
    assert said.startswith("survivor_chosen: B of 2 surviving End states, its Verifier passing 9 checks")


def test_two_survivors_whose_verifiers_are_equal_on_every_check_take_the_first_by_recording_order():
    """Nothing the harness can ask tells the two End states apart, which is a fact about them and
    not a reason to prefer either, so the Task takes the one its recordings reached first."""
    rows = [_survivor("A", checks_passed=9, fraction=0.5, held_out=2),
            _survivor("B", checks_passed=9, fraction=0.5, held_out=2)]
    best, why, said = stage.choose_survivor(rows, ["A", "B"])
    assert best["label"] == "A" and why == "survivors_equivalent"
    assert "the Task takes A, the first by recording order" in said
    # The order the rows arrive in is the recording order, not the order the choice sorted them into.
    assert stage.choose_survivor(list(reversed(rows)), ["B", "A"])[0]["label"] == "B"


def test_the_lower_false_rejection_settles_survivors_that_pass_the_same_checks():
    rows = [_survivor("A", checks_passed=9, fraction=1.0, held_out=3),
            _survivor("B", checks_passed=9, fraction=0.0, held_out=3)]
    best, why, _ = stage.choose_survivor(rows, ["A", "B"])
    assert best["label"] == "B" and why == "survivor_chosen"


def test_when_no_survivors_verifier_stands_up_the_task_keeps_no_reference_and_the_reason_names_the_best():
    rows = [_survivor("A", checks_passed=6, failed=("mutation_flips", "empty_fails")),
            _survivor("B", checks_passed=8, failed=("oracle_passes",))]
    best, why, said = stage.choose_survivor(rows, ["A", "B"])
    assert best is None and why == "survivors_all_fail"
    assert said == ("survivors_all_fail: a Verifier was derived from each of 2 surviving End states "
                    "and none of them passed the suite; the best is B, which passed 8 checks and "
                    "failed oracle_passes")


def test_a_residue_is_settled_by_the_derivation_and_the_round_counts_say_what_it_settled(tmp_path):
    """End to end: the judge abstains over two End states, the derivation derives a Verifier from
    each, and the Task keeps the Reference that choice made with the reason on its row."""
    abstains = json.dumps({"failed": [], "evidence": ["end_states"], "reason": "cannot tell"})
    world = make_world(tmp_path, rerolls=("wrong",))
    out = _derive(world.workdir, world.inputs, judge_model=TestModel([abstains, abstains]))
    row = _read(world.workdir / "references.json")["t1"]
    assert row["residue_derived"] and row["survivor_labels"] == ["A", "B"]
    assert row["survivor_reason"] == "survivors_equivalent" and row["survivor_chosen"] == "A"
    assert [score["label"] for score in row["survivor_scores"]] == ["A", "B"]
    assert [r["run_id"] for r in row["references"]] == ["ref"]
    assert out["task_status"]["t1"]["reference_confirmed"] is True
    metrics = _ruling(world.workdir)["metrics"]
    assert metrics["residue_derived"] == 1 and metrics["survivors_equivalent"] == 1
    assert metrics["survivor_chosen"] == 0 and metrics["survivors_all_fail"] == 0
    assert metrics["survivors_capped"] == 0


def test_a_task_derives_at_most_the_capped_number_of_survivors_and_says_it_chose_over_a_slice(tmp_path):
    """The cost of settling a residue is one derivation per surviving End state, so it is capped; a
    Task over the cap chose among the first states by recording order and its row says so."""
    world = make_world(tmp_path, rerolls=("wrong", "extra", "rr2"))
    fn = verifier_suite.canon_fn({})
    records = [reference.load(world.paths[run_id], reference.RECORDING, run_id=run_id,
                              write_tools=VF.WRITE_TOOLS, fn=fn)
               for run_id in ("ref", "wrong", "extra", "rr2")]
    groups = reference.group(records)
    assert len(groups) == 4, "the world's four Runs reach four End states"
    confirmation = reference.Confirmation(recordings=records, survivors=groups)
    stage.settle_residue(Task(id="t1", intent=VF.TASK.intent, run_ids=["ref"]), confirmation,
                         canon_rules={}, write_tools=set(VF.WRITE_TOOLS), constraints=[],
                         intents={}, user_rules={}, fn=fn, cap=2)
    assert confirmation.survivors_capped is True
    assert {row["label"] for row in confirmation.survivor_scores} == {"A", "B"}, \
        "the two states the recordings reached first, and not the two the cap left out"
    assert confirmation.survivor_chosen in {"A", "B"}


# --- keyed draws in the derivation (D212) -----------------------------------

def _probed(workdir: Path) -> set[str]:
    """The Tasks that spent a probe slot, off the cache entry each derivation wrote."""
    out = set()
    for entry in (workdir / "examiner" / "cache").glob("*/*.json"):
        body = _read(entry)
        if body.get("probed"):
            out.add(body["task_id"])
    return out


def test_which_tasks_spend_the_probe_budget_does_not_depend_on_the_order_the_tasks_are_handed_over(tmp_path):
    """D212: a bounded budget goes out by the Tasks' own keys, so re-clustering the list moves nobody."""
    forward = make_world(tmp_path / "forward", tasks=5)
    backward = make_world(tmp_path / "backward", tasks=5)
    backward.inputs["tasks"] = list(reversed(backward.inputs["tasks"]))
    kwargs = dict(probe_model=object(), run_probe=probe_runner_over(), probe_limit=2)
    _derive(forward.workdir, forward.inputs, **kwargs)
    first = _probed(forward.workdir)
    _derive(backward.workdir, backward.inputs, **kwargs)
    second = _probed(backward.workdir)
    assert len(first) == 2 and first == second


def test_a_task_added_to_the_build_does_not_take_the_probe_slot_of_a_task_whose_key_outranks_it(tmp_path):
    """The budget is still bounded, so a slot can change hands; what it must not do is change hands
    because a Task appeared earlier in the list than the Task already holding it."""
    four = make_world(tmp_path / "four", tasks=4)
    five = make_world(tmp_path / "five", tasks=5)
    kwargs = dict(probe_model=object(), run_probe=probe_runner_over(), probe_limit=2)
    _derive(four.workdir, four.inputs, **kwargs)
    before = _probed(four.workdir)
    order = sampling.keyed_order(stage.PROBE_KIND, [t.id for t in five.inputs["tasks"]],
                                 sampling.build_salt(five.workdir))
    _derive(five.workdir, five.inputs, **kwargs)
    after = _probed(five.workdir)
    assert after == set(order[:2])
    assert before <= set(sampling.keyed_order(stage.PROBE_KIND, [t.id for t in four.inputs["tasks"]],
                                              sampling.build_salt(four.workdir))[:2])


def test_a_second_search_for_a_second_path_takes_batch_numbers_above_the_ones_already_bought(tmp_path):
    """D212: a count that must grow adds higher attempt indexes; it never writes over an earlier one."""
    world = _lone_reference_world(tmp_path)
    calls: list = []
    runner = _reroll_runner_over(world, VF.wrong_run, calls)
    stage.second_path_search("t1", _confirmation_of(world), workdir=world.workdir, run_rerolls=runner,
                             round_number=1, write_tools=set(VF.WRITE_TOOLS),
                             fn=verifier_suite.canon_fn({}), atoms=[], cap=2)
    stage.second_path_search("t1", _confirmation_of(world), workdir=world.workdir, run_rerolls=runner,
                             round_number=1, write_tools=set(VF.WRITE_TOOLS),
                             fn=verifier_suite.canon_fn({}), atoms=[], cap=2)
    assert [prefix for _, _, prefix in calls] == [
        "second-path-r1-b1", "second-path-r1-b2", "second-path-r1-b3", "second-path-r1-b4"]
    assert {row["batch"] for row in stage.second_path_rows(world.workdir, "t1")} == {1, 2, 3, 4}
    assert stage.next_batch(world.workdir, "t1") == 5


def _confirmation_of(world) -> reference.Confirmation:
    """The world's Reference as a settled Confirmation, so the search has something to look past."""
    fn = verifier_suite.canon_fn({})
    record = reference.load(world.paths["ref"], reference.RECORDING, run_id="ref",
                            write_tools=VF.WRITE_TOOLS, fn=fn)
    return reference.Confirmation(recordings=[record], references=[record])


# --- concurrent batches in the derive window (speed-1) ---------------------

def _reroll_rows(workdir: Path) -> dict:
    """The Examiner's own re-roll rows with the workdir taken out, so two runs compare."""
    body = json.loads((workdir / "examiner" / "rerolls.json").read_text(encoding="utf-8"))
    return {task_id: [{key: row[key] for key in ("run_id", "batch", "reason", "termination_reason")}
                       for row in rows]
            for task_id, rows in body.items()}


def _rich_reference_events():
    """The Reference's events with two more reads: four calls, so four rewrites exist."""
    events = VF.reference_events()
    first = [VF.call("get_loyalty_balance", {"user_id": "u1"}, kind="read", cid="c8"),
             VF.result({"user_id": "u1", "points": 40}, cid="c8")]
    second = [VF.call("get_shipping_status", {"order_id": "#W123"}, kind="read", cid="c9"),
              VF.result({"order_id": "#W123", "status": "in_transit"}, cid="c9")]
    return events[:5] + first + second + events[5:]


def _rich_lone_world(root: Path):
    """The one-Task world whose recording makes four calls: the synthesis tries four rewrites."""
    world = make_world(root, rerolls=())
    rich = VF.make_run("ref", _rich_reference_events())
    VF.write_events_jsonl(rich, Path(world.inputs["replays"]["t1"]["ref"]["path"]))
    return world


def _counted_overlap(track: dict, lock: threading.Lock):
    """A context counting how many wrapped calls are in flight, and the peak."""
    class _Overlap:
        def __enter__(self):
            with lock:
                track["in_flight"] += 1
                track["max"] = max(track["max"], track["in_flight"])

        def __exit__(self, *exc):
            with lock:
                track["in_flight"] -= 1
            return False

    return _Overlap()


def _overlapped(fn, track: dict, lock: threading.Lock, delay: float = 0.02):
    """One shared-pool job wrapped so the test sees how many of a Task's jobs overlap."""
    def counted(*args, **kwargs):
        with _counted_overlap(track, lock):
            time.sleep(delay)
            return fn(*args, **kwargs)

    return counted


def _sequenced_rerolls(world, makers: list, task_calls: list):
    """One batch per maker in order, so the search misses twice and finds on the third batch."""
    remaining = list(makers)

    def sequenced(task_id: str, count: int, prefix: str) -> list:
        make = remaining[0]
        rows = _reroll_runner_over(world, make, task_calls)(task_id, count, prefix)
        if len(remaining) > 1:
            del remaining[0]
        return rows

    return sequenced


def test_one_tasks_surviving_end_states_derive_on_the_shared_pool_at_once(tmp_path, monkeypatch):
    """The residue's survivors are independent derivations: on eight workers both are in flight
    together, and the rows read exactly as the one-worker run's rows read."""
    abstain = json.dumps({"failed": [], "evidence": ["end_states"], "reason": "cannot tell"})
    peaks, bodies = {}, {}
    for name, workers in (("serial", 1), ("threaded", 8)):
        world = make_world(tmp_path / name, rerolls=("wrong",))
        track = {"in_flight": 0, "max": 0}
        counted = _overlapped(stage._derive_survivor, track, threading.Lock())
        monkeypatch.setattr(stage, "_derive_survivor", counted)
        out = _derive(world.workdir, world.inputs, judge_model=TestModel([abstain, abstain]),
                      workers=workers)
        assert out["ran"] == 1
        peaks[name] = track["max"]
        bodies[name] = _derived_bytes(world.workdir, 1)
        monkeypatch.undo()
    assert peaks["serial"] == 1
    assert peaks["threaded"] == 2, "both survivors derive without waiting for each other"
    assert bodies["threaded"] == bodies["serial"]


def test_one_tasks_rewrite_runs_replay_on_the_shared_pool_at_once(tmp_path):
    """The synth variant runs stay serial while the variant gate holds (docs/todo.md "Next
    re-freeze: flip the speed-1 variant gate"): one at a time on any worker count, and the rows
    and the bought batches read exactly as the one-worker run's read. Restore the overlap half
    when the gate flips."""
    peaks, bodies, tried = {}, {}, {}
    for name, workers in (("serial", 1), ("threaded", 8)):
        world = _rich_lone_world(tmp_path / name)
        track = {"in_flight": 0, "max": 0}
        counted_runner = _overlapped(_variant_runner_over(world, VF.alt_path_run, []),
                                     track, threading.Lock())
        out = _derive(world.workdir, world.inputs,
                      run_rerolls=_reroll_runner_over(world, VF.wrong_run, []),
                      run_variant=counted_runner, round_number=2, workers=workers)
        assert out["ran"] == 1
        row = out["task_status"]["t1"]["second_path"]
        tried[name] = row["synth_tried"]
        peaks[name] = track["max"]
        bodies[name] = (_derived_bytes(world.workdir, 1), _reroll_rows(world.workdir))
    assert tried["serial"] == tried["threaded"] >= 2, "the world offers several rewrites"
    assert peaks["serial"] == 1
    assert peaks["threaded"] == 1, "gated serial until the tally lock re-freezes"
    assert bodies["threaded"] == bodies["serial"]


def test_the_probe_budget_hands_the_same_slots_out_on_one_worker_and_on_eight(tmp_path):
    """D212 still holds under the shared pool: the slots are assigned by keyed order before any
    Task's probe dispatches, so the worker count moves no slot."""
    bodies, probed = {}, {}
    for name, workers in (("serial", 1), ("threaded", 8)):
        world = make_world(tmp_path / name, tasks=5)
        out = _derive(world.workdir, world.inputs, probe_model=object(),
                      run_probe=probe_runner_over(), probe_limit=2, workers=workers)
        assert out["ran"] == 5
        probed[name] = _probed(world.workdir)
        bodies[name] = _derived_bytes(world.workdir, 5)
    order = sampling.keyed_order(stage.PROBE_KIND, [f"t{n}" for n in range(1, 6)],
                                 sampling.build_salt(make_world(tmp_path / "salt", tasks=5).workdir))
    assert probed["serial"] == probed["threaded"] == set(order[:2])
    assert bodies["threaded"] == bodies["serial"]


def test_three_bought_batches_leave_the_same_rows_and_files_on_one_worker_and_on_eight(tmp_path):
    """The batches stay serial per Task under the shared pool: the same three prefixes are bought,
    and the rows and the re-roll file read the same on either worker count."""
    bodies, calls = {}, {}
    makers = [VF.wrong_run, VF.wrong_run, VF.alt_path_run]
    for name, workers in (("serial", 1), ("threaded", 8)):
        world = make_world(tmp_path / name, rerolls=())
        task_calls: list = []
        sequenced = _sequenced_rerolls(world, makers, task_calls)
        out = _derive(world.workdir, world.inputs, run_rerolls=sequenced,
                      probe_model=object(), run_probe=probe_runner_over(),
                      round_number=2, workers=workers)
        assert out["ran"] == 1
        calls[name] = list(task_calls)
        bodies[name] = (_derived_bytes(world.workdir, 1), _reroll_rows(world.workdir))
    assert [prefix for _, _, prefix in calls["serial"]] == [
        "second-path-r2-b1", "second-path-r2-b2", "second-path-r2-b3"]
    assert calls["threaded"] == calls["serial"]
    assert bodies["threaded"] == bodies["serial"]


def test_a_failed_first_gather_item_cancels_the_jobs_that_never_start():
    """A gather whose first item fails raises that failure in item order and never starts the
    pending jobs: on two threads six items run at most the failed one and the in-flight ones."""
    started = {"count": 0}
    lock = threading.Lock()
    release = threading.Event()

    def job(index: int) -> int:
        with lock:
            started["count"] += 1
        if index == 0:
            raise ValueError("the first rewrite fell over")
        assert release.wait(timeout=10)
        return index

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        with pytest.raises(ValueError, match="fell over"):
            stage._ordered_gather(pool, list(range(6)), job)
        assert started["count"] <= 3, "the failed item plus at most the two in flight ran"
    finally:
        release.set()
        pool.shutdown(cancel_futures=True)


def test_the_two_pools_together_never_run_more_jobs_than_they_were_given():
    """The outer per-Task pool and the inner per-survivor pool share one ceiling: with both busy
    the peak in flight never passes the worker count."""
    sem = threading.Semaphore(2)
    track = {"in_flight": 0, "max": 0}
    lock = threading.Lock()

    def job(index: int) -> int:
        with lock:
            track["in_flight"] += 1
            track["max"] = max(track["max"], track["in_flight"])
        try:
            time.sleep(0.15)
            return index
        finally:
            with lock:
                track["in_flight"] -= 1

    outer = ThreadPoolExecutor(max_workers=2)
    inner = ThreadPoolExecutor(max_workers=2)
    try:
        outers = [outer.submit(stage._guarded, sem, lambda index=index: job(index))
                  for index in range(4)]
        inners = stage._ordered_gather(inner, list(range(4, 8)), job, sem=sem)
        outs = [future.result() for future in outers]
    finally:
        outer.shutdown(cancel_futures=True)
        inner.shutdown(cancel_futures=True)
    assert sorted(outs) == [0, 1, 2, 3]
    assert inners == [4, 5, 6, 7]
    assert track["max"] <= 2, "the two pools share one ceiling of two"
