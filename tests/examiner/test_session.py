"""The Examiner called by the Builder: derive by code, then one model session (D79, D128, D230)."""

from __future__ import annotations

from examiner.worlds import make_world
from gates.examiner_fixtures import SIGS
from kullback.agent.events import ToolExecutionEnd
from kullback.agent.extensions import load_extensions
from kullback.agent.harness import AgentHarness, drive_tool
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.examiner import session as S
from kullback.examiner.exam_files import ExamRoot
from kullback.gates.probes import version_hash
from kullback.runner import tool
from kullback.runner.records import Verifier, as_dict, read_json, write_json
from tests.runner.test_tool import _env_with_trace


def reply(content, *calls) -> ModelReply:
    """One scripted assistant turn, the shape tests/builder/session_fixtures.py:reply gives."""
    return ModelReply(content=content, tool_calls=[ToolCallRequest(id=f"c{i}", name=n, arguments=a)
                                                  for i, (n, a) in enumerate(calls)])


def materialize(world) -> None:
    """The derivation's store as workdir files: tasks, sigs, replays, re-rolls, rules."""
    workdir = world.workdir
    (workdir / "tasks").mkdir(exist_ok=True)
    for task in world.inputs["tasks"]:
        write_json(workdir / "tasks" / f"{task.id}.json", as_dict(task))
    write_json(workdir / "tool_sigs.json", [as_dict(sig) for sig in SIGS])
    write_json(workdir / "constraints.json", [])
    write_json(workdir / "canon-rules.json", {})
    write_json(workdir / "replays.json", world.inputs["replays"])
    write_json(workdir / "rerolls.json", world.inputs["rerolls"])


def test_examine_without_a_model_returns_derive_filed_findings_with_rows(tmp_path):
    world = make_world(tmp_path, confirmed=False)
    materialize(world)
    findings = S.examine(world.workdir, model=None)
    assert findings, "an unconfirmed world owes a fidelity finding"
    assert all(f.rows for f in findings)
    assert all(f.source == "derive" for f in findings)
    assert {f.kind for f in findings} <= {"assisted_tool", "fidelity", "reference_disagreement",
                                          "suite", "false_rejection", "environment", "other"}
    written = read_json(world.workdir / "findings.json", [])
    assert len(written) == len(findings) and all(w["rows"] for w in written)


def _scripted_events(world, replies):
    events: list = []
    S.examine(world.workdir, model=TestModel(replies), subscribers=[events.append])
    return events


def _ends(events, tool):
    return [e for e in events if isinstance(e, ToolExecutionEnd) and e.tool_name == tool]


def test_fresh_session_seeds_version_1_accepted_for_each_derived_verifier(tmp_path):
    world = make_world(tmp_path)
    materialize(world)
    _scripted_events(world, [reply("done")])
    derived = Verifier.model_validate(read_json(world.workdir / "verifiers" / "t1.json"))
    written = read_json(world.workdir / "exam" / "history.json")
    [row] = written["t1"]["versions"]
    assert (row["verifier_version"], row["accepted"], row["by"]) == ("1", True, "derive")
    assert row["reason"] == "the Verifier on disk before any repair"
    assert row["content_hash"] == version_hash(derived)


def test_model_propose_verifier_is_refused_while_the_loophole_probe_cannot_run(tmp_path):
    world = make_world(tmp_path)
    materialize(world)
    events = _scripted_events(world, [
        reply("proposing", ("propose_verifier", {"task_id": "t1", "reason": "tighten",
                                                 "drop": [], "add": []})),
        reply("done"),
    ])
    [proposed] = _ends(events, "propose_verifier")
    assert proposed.is_error is True
    assert "verifier_loophole fail (not run: no model" in proposed.result.content
    assert "rows" in proposed.result.content
    assert not (world.workdir / "exam" / "verifiers" / "t1.json").exists()
    written = read_json(world.workdir / "exam" / "history.json")
    assert [v["accepted"] for v in written["t1"]["versions"]] == [True, False]
    assert written["t1"]["versions"][-1]["rejected_by"] == ["verifier_loophole"]
    assert not (world.workdir / "exam" / "task_status.json").exists(), "a refusal writes no status"


def test_model_write_of_verifiers_carries_rulings_runs_write_refused(tmp_path):
    world = make_world(tmp_path)
    materialize(world)
    events = _scripted_events(world, [
        reply("proposing", ("write", {"path": "verifiers/t1.json", "content": "{}"})),
        reply("bad idea", ("write", {"path": "runs/x.jsonl", "content": "{}"})),
        reply("done"),
    ])
    writes = _ends(events, "write")
    assert writes[0].is_error is False
    assert writes[0].result.details["rulings"], "the verifiers binding rules on the write"
    assert writes[1].is_error is True


def test_scripted_reroll_is_refused_when_the_allowance_is_spent(tmp_path):
    world = make_world(tmp_path)
    materialize(world)
    events: list = []
    S.examine(world.workdir, model=TestModel([
        reply("buying", ("reroll", {"task_id": "t1", "count": 1})),
        reply("done"),
    ]), reroll_model=object(), allowance_usd=0.0, subscribers=[events.append])
    [bought] = _ends(events, "reroll")
    assert bought.is_error is True
    assert "allowance is spent" in bought.result.content


def test_model_finding_publishes_and_duplicates_are_refused_with_the_id(tmp_path):
    world = make_world(tmp_path)
    materialize(world)
    args = {"task_id": "t1", "kind": "fidelity", "text": "replays differ",
            "rows": [{"run_id": "ref"}], "path": "env/tools/x.py", "change": "answer the call"}
    events = _scripted_events(world, [
        reply("filing", ("finding", args)),
        reply("filing again", ("finding", args)),
        reply("done"),
    ])
    filed = _ends(events, "finding")
    assert filed[0].is_error is False
    finding_id = filed[0].result.details["finding"]["finding_id"]
    assert filed[1].is_error is True
    assert finding_id in filed[1].result.content
    bus_lines = (world.workdir / "bus.jsonl").read_text().splitlines()
    assert any(finding_id in line for line in bus_lines)
    written = read_json(world.workdir / "findings.json", [])
    assert any(w.get("finding_id") == finding_id for w in written)


def test_bash_is_not_registered_and_base_tools_are_scoped_to_exam(tmp_path):
    world = make_world(tmp_path)
    root = ExamRoot(workdir=world.workdir)
    harness = AgentHarness(model=TestModel([]))
    load_extensions(harness, [S.examiner_extension(root)])
    names = harness.registry.names()
    assert "bash" not in names
    assert {"read", "write", "edit", "grep", "find", "ls", "web_search",
            "propose_verifier", "probe", "finding", "reroll"} <= set(names)
    S.expose(world.workdir)
    refused = drive_tool(harness, "write", {"path": "runs/x.jsonl", "content": "{}"})
    assert refused.is_error is True


def _rename_loop() -> TestModel:
    """A model that reads the widget and writes the striped label, then stops, on every Run."""
    return TestModel([
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m1", name="describe_widget", arguments={"widget_id": "w1"})]),
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m2", name="rename_widget",
                            arguments={"widget_id": "w1", "label": "striped"})]),
        ModelReply(content="Done.", model="test"),
    ], loop=True)


def test_examine_with_a_reroll_model_buys_second_path_rerolls(tmp_path):
    """A lone Reference buys re-rolls through the runner callable: Runs on disk, rows kept."""
    root = _env_with_trace(tmp_path / "env")
    seed = tool.run(root, "widget_task", _rename_loop(), workdir=root)
    write_json(root / "replays.json", {"widget_task": {
        "rec1": {"trace_id": "rec1", "run_id": seed.run_id,
                 "confirmed": True, "path": seed.path}}})
    write_json(root / "rerolls.json", {"widget_task": []})
    write_json(root / "constraints.json", [])
    S.examine(root, model=None, reroll_model=_rename_loop())
    bought = sorted((root / "runs").glob("second-path-*.jsonl"))
    assert bought, "a Reference standing alone buys re-rolls through the callable"
    rows = (read_json(root / "examiner" / "rerolls.json", {}) or {}).get("widget_task") or []
    assert rows and all(row.get("reason") == "second_path" for row in rows)
