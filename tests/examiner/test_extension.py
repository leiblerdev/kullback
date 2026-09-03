"""The Examiner as an extension on the agent core: what it registers, what its hooks refuse, the context
guards over the session's entries, and the two ways a session is driven (D120, D122, D123, D124)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from examiner.worlds import WORLD_TASK as T
from examiner.worlds import World, drive, events_of, make_world, probe_runner_over
from gates import verifier_fixtures as VF
from kullback.agent.events import StageEnd, StageStart, ToolExecutionEnd, ToolExecutionStart
from kullback.agent.extensions import load_extensions
from kullback.agent.harness import AgentHarness
from kullback.agent.session import SessionStore
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.examiner import agent as examiner_agent
from kullback.examiner import extension as ext
from kullback.examiner import skills
from kullback.examiner import tools as tools_mod
from kullback.examiner.extension import examiner_extension
from kullback.gates import gates_over

SEVEN = ["read", "derive", "probe", "repair", "refuse", "reroll", "finding"]
COMPARED = ("task_status.json", "references.json", "constraints_check.json", "gates.json", "scorecard.json")


def _reply(content, *calls):
    return ModelReply(content=content, tool_calls=[ToolCallRequest(id=cid, name=name, arguments=args)
                                                   for cid, name, args in calls])


def _run(harness: AgentHarness, message: str = "go") -> list:
    async def go():
        return [event async for event in harness.prompt(message)]
    return asyncio.run(go())


def _harness(world: World, agent_model=None, session: Path | None = None, subscribers=(), **kwargs):
    plan = world.plan(probe_model=object(), run_probe=probe_runner_over(), **kwargs)
    store = SessionStore.load(session) if session is not None else None
    return plan, examiner_agent.examiner_harness(plan, agent_model, subscribers, session=store)


def _entry(harness: AgentHarness, call_id: str) -> str:
    entry = harness.context.entry_id_for(call_id)
    assert entry is not None, f"the session knows no entry for {call_id}"
    return entry


def _probe_call(cid: str, run, bug_class: str = "other") -> tuple:
    return (cid, "probe", {"task_id": T, "bug_class": bug_class, "events": events_of(run),
                           "termination_reason": run.termination_reason})


def _tree(workdir: Path) -> dict:
    def read(path: Path) -> bytes:
        return path.read_bytes().replace(str(workdir.resolve()).encode(), b"<workdir>").replace(
            str(workdir).encode(), b"<workdir>")
    out = {name: read(workdir / name) for name in COMPARED if (workdir / name).is_file()}
    for folder in ("verifiers", "examiner/history", "probes"):
        for path in sorted((workdir / folder).rglob("*.json")):
            out[str(path.relative_to(workdir))] = read(path)
    return out


def test_the_extension_registers_the_seven_tools_the_three_sections_the_probe_skill_and_the_two_hooks(world):
    plan, harness = _harness(world)
    assert isinstance(harness, AgentHarness), "the Examiner is an extension on the core, not a harness of its own"
    assert harness.registry.names() == SEVEN
    assert [s.name for s in harness.sections] == ["examiner", "examiner_rules", "examiner_tasks", "skill:probe"]
    assert "never edit the Environment" in harness.system and "kullback/gates" in harness.system
    assert f"{T}: 1 Runs" in harness.system and "`derive` takes `all`" in harness.system
    assert harness.context.catalog_skills == {"probe": skills.PROBE_SKILL} and harness.context.loaded_skills == {"probe"}
    assert "Probe skill" in harness.system
    tool_call = [h.hook_name for h in harness.hooks.tool_call]
    assert tool_call[:2] == ["examiner_reads_only_its_surface", "repair_guard"]
    assert "gate_rulings" in [h.hook_name for h in harness.hooks.tool_result]
    assert plan.unprotect is not ext.ExaminerPlan.unprotect and plan.entry_id_for("nothing") is None
    schema = harness.registry.get("reroll").schema()
    assert schema["input_schema"]["properties"]["count"]["minimum"] == 1
    assert set(harness.registry.get("probe").schema()["input_schema"]["properties"]["bug_class"]["enum"]) == \
        set(skills.BUG_CLASSES) | {"other"}


@pytest.mark.parametrize("name", ext.FORBIDDEN_READS)
@pytest.mark.parametrize("shape", ["{name}", "kullback/{name}", "build/{name}/x.json", "{name}/tables.json"])
def test_a_read_naming_bodies_env_sandbox_or_an_environment_artifact_is_blocked_before_it_runs(derived, name, shape):
    plan, harness = _harness(derived)
    value = shape.format(name=name)
    result = drive(harness, "read", {"kind": "run", "id": value})
    assert result.is_error and "blocked by examiner_reads_only_its_surface" in result.content
    assert value in result.content and "D123" in result.content
    assert ext.names_forbidden_path({"nested": [{"deep": value}]}) == value
    assert ext.names_forbidden_path("runs/t1/ref.jsonl") is None


def test_a_tool_call_naming_a_path_under_gates_or_runner_is_blocked_and_becomes_an_error_result(derived):
    plan, harness = _harness(derived)
    for path in ("kullback/gates/trust.py", "gates/", "kullback/runner", "src/kullback/runner/loop.py"):
        result = drive(harness, "finding", {"kind": "other", "text": f"see {path} for the rule"})
        assert result.is_error and "blocked by examiner_reads_only_its_surface" in result.content, path
        assert path in result.content and "D122" in result.content
    assert not (derived.workdir / "examiner" / "findings.json").is_file() or \
        json.loads((derived.workdir / "examiner" / "findings.json").read_text(encoding="utf-8")) == []
    fine = drive(harness, "finding", {"kind": "other", "text": "the gates ruled and the runner ran"})
    assert fine.is_error is False


def test_the_examiner_has_no_tool_that_writes_a_body_a_table_or_the_environment(fixture_build, tmp_path):
    workdir = fixture_build.copy(tmp_path)
    inputs = fixture_build.inputs_for(workdir)
    compiled = ("bodies.json", "db.json", "schema.json", "environment.json", "tool_sigs.json")
    before = {name: hashlib.sha256((workdir / name).read_bytes()).hexdigest() for name in compiled}
    env_before = sorted(str(p.relative_to(workdir)) for p in (workdir / "env").rglob("*") if p.is_file())
    result = examiner_agent.run_examiner(workdir, inputs=inputs)
    assert result["status"] == "complete" and set(result["tasks"]) == {t.id for t in inputs["tasks"]}
    after = {name: hashlib.sha256((workdir / name).read_bytes()).hexdigest() for name in compiled}
    assert after == before
    assert sorted(str(p.relative_to(workdir)) for p in (workdir / "env").rglob("*") if p.is_file()) == env_before
    names = [tool.name for tool in tools_mod.examiner_tools(ext.ExaminerPlan(workdir=workdir, inputs=inputs))]
    assert names == SEVEN and not {"build", "compile_tool", "grow", "recluster", "replay"} & set(names)
    produced = {name for tool in tools_mod.examiner_tools(ext.ExaminerPlan(workdir=workdir, inputs=inputs))
                for name in (tool.result_model.model_fields["produced"].default_factory() if "produced"
                             in tool.result_model.model_fields else [])}
    assert not produced & {"bodies", "db", "schema", "environment", "assisted_tools", "tasks"}
    assert produced == {"verifiers", "task_status", "history", "probes", "refusals", "rerolls", "task_runs", "findings"}


def test_the_tool_result_hook_runs_the_gates_bound_to_what_the_tool_produced(derived):
    plan, harness = _harness(derived)
    result = drive(harness, "probe", {"task_id": T, "bug_class": "other", "events": events_of(VF.wrong_run())})
    assert result.details["produced"] == ["probes"]
    bound = [spec.name for spec in gates_over("probes")]
    assert [row["stage"] for row in result.details["gate_rulings"]] == bound
    assert "gate rulings:" in result.content and all(row["pass"] for row in result.details["gate_rulings"])
    derived_again = drive(harness, "derive", {"target": T})
    stages = [row["stage"] for row in derived_again.details["gate_rulings"]]
    assert stages == list(dict.fromkeys(spec.name for artifact in ("verifiers", "task_status", "history")
                                        for spec in gates_over(artifact)))
    plain = drive(harness, "read", {"kind": "task_status"})
    assert "gate_rulings" not in plain.details and "gate rulings" not in plain.content


def test_a_failed_ruling_protects_the_result_until_the_next_call_on_the_same_task_unprotects_it(derived, tmp_path):
    model = TestModel([_reply(None, _probe_call("c1", VF.alt_path_run())),
                       _reply(None, _probe_call("c2", VF.wrong_run())),
                       _reply("read")])
    plan, harness = _harness(derived, model, session=tmp_path / "session.jsonl")
    _run(harness)
    first, second = _entry(harness, "c1"), _entry(harness, "c2")
    protected = harness.context.protected
    assert first not in protected, "the second probe on the Task is the act on the first ruling"
    assert protected[second].startswith("unacted ruling: ") and "probe_pool" in protected[second]
    assert protected[second].endswith(f" on {T}")


def test_a_finding_protects_the_entry_it_refers_to_until_it_is_closed(derived, tmp_path):
    model = TestModel([_reply(None, ("c1", "read", {"kind": "verifier", "id": T})),
                       _reply(None, ("c2", "finding", {"task_id": T, "kind": "fidelity", "text": "the reads differ",
                                                       "about_call_id": "c1"})),
                       _reply("filed")])
    plan, harness = _harness(derived, model, session=tmp_path / "session.jsonl")
    events = _run(harness)
    filed = next(e for e in events if isinstance(e, ToolExecutionEnd) and e.tool_name == "finding")
    read_entry = _entry(harness, "c1")
    assert filed.result.details["finding"]["about_entry_id"] == read_entry
    assert harness.context.protected[read_entry] == "open finding finding-1"
    assert plan.close_findings(["finding-1"]) == ["finding-1"]
    assert read_entry not in harness.context.protected
    assert json.loads((derived.workdir / "examiner" / "findings.json").read_text(encoding="utf-8"))[0]["status"] == "closed"


def test_a_repair_protects_the_read_it_rests_on_while_it_runs(derived, tmp_path):
    seen: dict[str, dict] = {}

    def watcher(api):
        def note(call, result):
            seen[call.name] = dict(api.harness.context.protected)
            return None
        note.hook_name = "watcher"
        api.tool_result(note)

    plan = derived.plan(probe_model=object(), run_probe=probe_runner_over())
    row = next(a for a in plan.current(T).atoms if a.id == "w0.reason")
    add = [{"id": "w0.reason", "kind": "required", "payload": dict(row.target or {})}]
    model = TestModel([_reply(None, ("c1", "read", {"kind": "verifier", "id": T})),
                       _reply(None, ("c2", "repair", {"task_id": T, "reason": "require it", "drop": ["w0.reason"],
                                                      "add": add})),
                       _reply("done")])
    harness = AgentHarness(model=model, session=SessionStore.load(tmp_path / "session.jsonl"))
    load_extensions(harness, [watcher, examiner_extension(plan)])
    events = _run(harness)
    repair = next(e for e in events if isinstance(e, ToolExecutionEnd) and e.tool_name == "repair")
    assert repair.is_error is False and repair.result.details["accepted"] is True
    read_entry = _entry(harness, "c1")
    assert seen["repair"].get(read_entry) == f"repair in progress on task {T}", "protected while the repair ran"
    assert read_entry not in harness.context.protected, "released when the repair result landed"
    assert not [r for r in harness.context.protected.values() if r.startswith("unacted")]


def test_the_driver_derives_through_the_hooks_and_the_model_driven_session_leaves_the_same_files(tmp_path):
    driven = make_world(tmp_path / "driver")
    events: list = []
    result = examiner_agent.run_examiner(driven.workdir, inputs=driven.inputs, probe_model=object(),
                                         run_probe=probe_runner_over(), subscribers=[events.append])
    assert result["status"] == "complete" and result["trusted"] == [T] and result["rulings"] == [
        "compile_policy", "derive_verifier"]
    assert result["tool_result"]["is_error"] is False and result["tool_result"]["content"].startswith("derive all")
    assert [e.name for e in events if isinstance(e, StageStart)] == ["derive_verifier"]
    ended = [e for e in events if isinstance(e, StageEnd)]
    assert ended[-1].counts["status"] == "ran" and ended[-1].counts["verifiers"] == 1
    assert isinstance(events[-1], ToolExecutionEnd) and events[-1].tool_name == "derive"
    started = next(e for e in events if isinstance(e, ToolExecutionStart))
    assert started.tool_call_id == examiner_agent.DRIVER_CALL_ID

    modelled = make_world(tmp_path / "model")
    model = TestModel([_reply(None, ("c1", "derive", {"target": "all"})), _reply("every ruling read.")])
    other = examiner_agent.run_examiner(modelled.workdir, inputs=modelled.inputs, probe_model=object(),
                                        run_probe=probe_runner_over(), agent_model=model)
    assert len(model.calls) == 2 and other["trusted"] == [T]
    assert _tree(modelled.workdir) == _tree(driven.workdir)
    assert other["tasks"] == result["tasks"]


def test_the_examiner_session_is_recorded_under_its_own_file(world, tmp_path):
    builder_session = tmp_path / "session.jsonl"
    builder_session.write_text('{"kind": "not touched"}\n', encoding="utf-8")
    path = world.workdir / "examiner" / "session.jsonl"
    model = TestModel([_reply(None, ("c1", "derive", {"target": "all"})), _reply("done")])
    examiner_agent.run_examiner(world.workdir, inputs=world.inputs, probe_model=object(), run_probe=probe_runner_over(),
                                agent_model=model, session_path=path)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "the Examiner's transcript is on disk"
    text = path.read_text(encoding="utf-8")
    assert '"derive"' in text and "skill" in text
    assert builder_session.read_text(encoding="utf-8") == '{"kind": "not touched"}\n'
    again = SessionStore.load(path)
    assert len(again.entries) == len(lines) - sum(1 for row in lines if row.get("kind") == "leaf")


def test_the_examiner_harness_refuses_a_model_turn_under_the_code_driver(world):
    plan, harness = _harness(world)
    assert isinstance(harness.model, examiner_agent.DriverModel)
    with pytest.raises(RuntimeError, match="no model turn"):
        harness.model.query([], None, None)
    events = _run(harness, "derive everything")
    turn = next(e for e in events if e.type == "turn_end")
    assert turn.message.stop_reason == "error" and "no model turn was asked for" in turn.message.error_message
    assert not [e for e in events if isinstance(e, ToolExecutionStart)], "the driver's harness ran nothing"
    assert not (world.workdir / "task_status.json").is_file()
    assert drive(harness, "derive", {"target": "all"}).is_error is False


def test_a_model_that_never_calls_derive_is_an_examiner_error(world):
    model = TestModel([_reply(None, ("c1", "read", {"kind": "task_status"})), _reply("nothing to do")])
    with pytest.raises(examiner_agent.ExaminerError, match="never called derive"):
        examiner_agent.run_examiner(world.workdir, inputs=world.inputs, agent_model=model)
    assert not (world.workdir / "task_status.json").is_file()
    broken = TestModel([_reply(None, ("c1", "derive", {"target": "no-such-task"})), _reply("gave up")])
    with pytest.raises(examiner_agent.ExaminerError, match="no Task is named no-such-task"):
        examiner_agent.run_examiner(world.workdir, inputs=world.inputs, agent_model=broken)


def test_the_probe_skill_names_the_eight_bug_classes():
    assert len(skills.BUG_CLASSES) == 8 and len(set(skills.BUG_CLASSES)) == 8
    for bug_class in skills.BUG_CLASSES:
        assert bug_class.capitalize()[:12].lower() in skills.PROBE_SKILL.lower()
        assert bug_class[0].upper() + bug_class[1:] + "." in skills.PROBE_SKILL
    assert "stays in the pool forever" in skills.PROBE_SKILL and "three" in skills.PROBE_SKILL
    assert skills.PROBE_SKILL_NAME == "probe"
