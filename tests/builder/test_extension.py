"""The Builder as an extension on the agent core (phase 4): the tools, the two hooks, the driver and the
model-driven path, and that both leave the same artifacts as build.build()."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from test_build import Bodies

from kullback.agent.events import StageEnd, StageStart, ToolExecutionEnd
from kullback.agent.harness import AgentHarness
from kullback.agent.messages import ToolCall
from kullback.agent.tools import AgentTool, ToolResult
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.builder import agent as builder_agent
from kullback.builder import build as build_module
from kullback.builder import extension as ext
from kullback.builder import tools as builder_tools
from kullback.builder.build import BuildPlan

COMPARED = ("bodies.json", "constraints.json", "gates.json", "environment.json", "replays.json", "task_status.json",
            "tasks.json", "schema.json", "tool_sigs.json", "user_facts.json", "vocabulary.json", "references.json")


def _reply(content, *calls):
    return ModelReply(content=content, tool_calls=[ToolCallRequest(id=f"c{i}", name=n, arguments=a)
                                                   for i, (n, a) in enumerate(calls)])


def _collect(aiter):
    async def go():
        return [event async for event in aiter]
    return asyncio.run(go())


def _fixture(request) -> Path:
    return Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"


@pytest.fixture(scope="module")
def driven(tmp_path_factory, request):
    """The CLI's path: code issues build(target) through the harness's hooks, no model turn."""
    workdir = tmp_path_factory.mktemp("driver")
    events = []
    result = builder_agent.run_builder(workdir, model=Bodies(), files=[_fixture(request)], max_attempts=0,
                                       subscribers=[events.append])
    return {"workdir": workdir, "result": result, "events": events, "tree": _tree(workdir)}


@pytest.fixture(scope="module")
def model_driven(tmp_path_factory, request):
    """The same build with a scripted model driving: it is asked for build(target) and calls it."""
    workdir = tmp_path_factory.mktemp("agent")
    agent_model = TestModel([_reply(None, ("build", {"target": "environment"})), _reply("every ruling read.")])
    events = []
    result = builder_agent.run_builder(workdir, model=Bodies(), files=[_fixture(request)], max_attempts=0,
                                       agent_model=agent_model, subscribers=[events.append])
    return {"workdir": workdir, "result": result, "events": events, "agent_model": agent_model, "tree": _tree(workdir)}


def _tree(workdir: Path) -> dict:
    """The build's artifacts by relative name, with the workdir's own absolute path (which replays.json
    records for every Run) replaced, so two workdirs compare on what was built."""
    def read(path: Path) -> bytes:
        return path.read_bytes().replace(str(workdir.resolve()).encode(), b"<workdir>").replace(
            str(workdir).encode(), b"<workdir>")
    out = {}
    for name in COMPARED:
        path = workdir / name
        if path.is_file():
            out[name] = read(path)
    for folder in ("intents", "tasks", "verifiers", "user_rules"):
        for path in sorted((workdir / folder).glob("*.json")):
            out[f"{folder}/{path.name}"] = read(path)
    for path in sorted((workdir / "runs").rglob("*.jsonl")):
        out[str(path.relative_to(workdir))] = read(path)
    return out


# --- the driver path ---------------------------------------------------------

def test_the_driver_builds_the_environment_and_reports_the_rulings(driven):
    result = driven["result"]
    assert result["status"] == "complete" and result["env_id"] and result["target"] == "environment"
    assert result["tool_result"]["is_error"] is False
    assert {"ingest", "mine", "cluster", "compile_tools", "compile_policy", "intent", "vocabulary",
            "build_user_rules", "tau2_export", "replay_reference", "rerolls", "derive_verifier"} <= set(result["rulings"])
    state = json.loads((driven["workdir"] / "pipeline" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete" and state["statuses"]["build_environment"] == "ran"
    assert set(state["statuses"]) == {"ingest", "mine", "cluster", "canon_rules", "starting_state", "compile_tools",
                                      "compile_policy", "judge_lessons", "intent", "vocabulary", "user_rules",
                                      "build_environment", "replay_reference", "rerolls", "derive_verifier"}


def test_stage_events_reach_the_subscribers_in_order_and_the_tool_end_comes_last(driven):
    events = driven["events"]
    starts = [e.name for e in events if isinstance(e, StageStart)]
    ends = [e.name for e in events if isinstance(e, StageEnd)]
    assert starts == ends and starts[:3] == ["ingest", "mine", "cluster"] and starts[-1] == "derive_verifier"
    assert isinstance(events[-1], ToolExecutionEnd) and events[-1].tool_name == "build"
    compile_end = next(e for e in events if isinstance(e, StageEnd) and e.name == "compile_tools")
    assert compile_end.counts["status"] == "ran" and "parses" in compile_end.counts["rulings"]
    assert compile_end.counts["produced"] == ["bodies", "assisted_tools"]


def test_the_tool_result_carries_a_short_text_and_the_payload_in_details(driven):
    content = driven["result"]["tool_result"]["content"]
    assert content.startswith("build environment: complete")
    assert "stages:" in content and "rulings:" in content
    assert len(content) < 2000, "the payload does not enter the context"
    assert "task_" not in content.split("gate rulings:")[0].split("rulings:")[0]


# --- the tool_result hook -----------------------------------------------------

def test_the_tool_result_hook_appends_the_registered_gates_over_what_the_tool_produced(driven):
    events = driven["events"]
    end = next(e for e in events if isinstance(e, ToolExecutionEnd))
    result = end.result
    assert "gate rulings:" in result.content
    rulings = result.details["gate_rulings"]
    names = [r["stage"] for r in rulings]
    assert names[:3] == ["ingest", "cluster", "compile_tools"]
    assert {"compile_policy", "intent", "vocabulary", "build_user_rules", "replay_reference"} <= set(names)
    # The same rulings the stages recorded, decided again by the same functions over the store the
    # tool left; the hook writes nothing, so gates.json is the stages' and only theirs. ingest's
    # ruling is a stage gate and lives in state.json; cluster's was in gates.json until the
    # compile_tools stage overwrote the file with the sandbox rulings, as it always has.
    recorded = {g["stage"]: g for g in json.loads((driven["workdir"] / "gates.json").read_text(encoding="utf-8"))}
    state = json.loads((driven["workdir"] / "pipeline" / "state.json").read_text(encoding="utf-8"))
    recorded.update({g["stage"]: g for g in state["gates"]})
    assert "cluster" not in recorded
    for ruling in rulings:
        if ruling["stage"] == "cluster":
            assert ruling["pass"] is True
            continue
        assert ruling["stage"] in recorded, ruling["stage"]
        assert ruling["pass"] == recorded[ruling["stage"]]["pass"], ruling["stage"]
        assert ruling["failures"] == recorded[ruling["stage"]]["failures"], ruling["stage"]


def test_every_ruling_the_build_wrote_to_disk_is_one_the_registry_names(driven):
    """The empirical half of D122's pin. `tests/gates/test_package.py` reads the Builder's source for
    rulings made outside the registry; this reads what a whole build actually left on disk, so a
    ruling written in a shape that scan does not know still has to be a registered gate's."""
    import kullback.gates as gates_pkg

    registered = {stage for spec in gates_pkg.GATES for stage in spec.rulings}
    recorded = {g["stage"] for g in json.loads((driven["workdir"] / "gates.json").read_text(encoding="utf-8"))}
    state = json.loads((driven["workdir"] / "pipeline" / "state.json").read_text(encoding="utf-8"))
    recorded |= {g["stage"] for g in state["gates"] if isinstance(g, dict)}
    assert recorded, "the build recorded no ruling at all, so this pin would pass over nothing"
    assert recorded - registered == set(), sorted(recorded - registered)


def test_the_hook_leaves_an_error_result_alone(tmp_path):
    plan = BuildPlan(workdir=tmp_path)
    hook = ext.gate_rulings_hook(plan)
    call = ToolCall(id="x", name="build", arguments={})
    assert hook(call, ToolResult(content="build failed: no traces", is_error=True)) is None
    assert hook(call, ToolResult(content="nothing", details={"produced": []})) is None


# --- the tool_call hook --------------------------------------------------------

class _WriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    text: str


class _Written(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str


def _write_tool(record: list) -> AgentTool:
    async def write(args: _WriteArgs) -> _Written:
        record.append(args.path)
        return _Written(path=args.path)
    return AgentTool("write_file", "Write a file (test only).", _WriteArgs, _Written, write)


@pytest.mark.parametrize("path", ["kullback/gates/artifacts.py", "kullback/runner/loop.py", "gates/stages.py",
                                  "runner/verdict.py", "/repo/kullback/gates/__init__.py",
                                  "../kullback/gates/__init__.py", "kullback/builder/../runner/loop.py",
                                  "./gates/stages.py", "kullback\\gates\\stages.py",
                                  "kullback/gates", "kullback/runner"])
def test_a_tool_call_naming_a_path_under_gates_or_runner_is_blocked_and_becomes_an_error_result(tmp_path, path):
    written = []
    plan = BuildPlan(workdir=tmp_path)
    model = TestModel([_reply(None, ("write_file", {"path": path, "text": "x"})), _reply("done")])
    harness = builder_agent.build_harness(plan, agent_model=model)
    harness.register_tool(_write_tool(written))
    events = _collect(harness.prompt("write it"))
    end = next(e for e in events if isinstance(e, ToolExecutionEnd))
    assert end.is_error and "no_agent_writes_gates_or_runner" in end.result.content and "D122" in end.result.content
    assert written == [], "the dummy tool never ran"
    assert harness.messages[2].is_error and harness.messages[2].tool_name == "write_file"
    # The driver applies the same hooks the loop does, so the code path agrees with the model path.
    driven = builder_agent.drive_tool(harness, "write_file", {"path": path, "text": "x"})
    assert driven.is_error and "no_agent_writes_gates_or_runner" in driven.content and written == []


@pytest.mark.parametrize("path", ["gates.json", "runs/task_1/run.jsonl", "runner_version.json", "kullback/builder/mine.py"])
def test_a_tool_call_naming_anything_else_goes_through(tmp_path, path):
    written = []
    plan = BuildPlan(workdir=tmp_path)
    harness = builder_agent.build_harness(plan)
    harness.register_tool(_write_tool(written))
    result = builder_agent.drive_tool(harness, "write_file", {"path": path, "text": "x"})
    assert not result.is_error and written == [path]


def test_the_hook_looks_inside_lists_and_nested_dicts(tmp_path):
    assert ext.names_protected_path({"edits": [{"file": "kullback/runner/route.py"}]}) == "kullback/runner/route.py"
    assert ext.names_protected_path({"edits": [{"file": "kullback/builder/mine.py"}], "note": "gates.json"}) is None
    # The package directory itself, named with nothing after it, is the form a delete takes; a bare
    # word `gates` or `runner` in an argument is an ordinary string and stays one.
    assert ext.names_protected_path({"path": "kullback/gates"}) == "kullback/gates"
    assert ext.names_protected_path({"command": "rm -r kullback/runner"}) == "rm -r kullback/runner"
    assert ext.names_protected_path({"role": "runner", "stage": "gates"}) is None
    assert ext.names_protected_path({"note": "the runner is frozen"}) is None
    with pytest.raises(PermissionError, match="D122"):
        ext.no_agent_writes_gates_or_runner(ToolCall(id="1", name="x", arguments={"paths": ["a", "gates/x.py"]}))


# --- the prompt and the tools -------------------------------------------------

def test_the_extension_registers_the_six_tools_and_the_three_sections_and_no_repair_verb(tmp_path):
    harness = builder_agent.build_harness(BuildPlan(workdir=tmp_path))
    assert isinstance(harness, AgentHarness), "the Builder is an extension on the core, not a harness of its own"
    assert harness.registry.names() == ["build", "recluster", "grow", "compile_tool", "replay", "reroll"]
    assert [s.name for s in harness.sections] == ["builder", "builder_rules", "builder_targets"]
    assert "no repair verb" in harness.system and "kullback/gates" in harness.system
    assert "`environment` is the whole build" in harness.system
    assert "compile_tools" in harness.system and "bodies" in harness.system
    assert "repair" not in harness.registry.names()
    schema = harness.registry.get("grow").schema()
    assert schema["input_schema"]["properties"]["count"]["minimum"] == 1
    assert [h.hook_name for h in harness.hooks.tool_call] == ["no_agent_writes_gates_or_runner"]
    assert [h.hook_name for h in harness.hooks.tool_result] == ["gate_rulings"]


def test_invalid_tool_arguments_are_an_error_result_not_a_crash(tmp_path):
    harness = builder_agent.build_harness(BuildPlan(workdir=tmp_path))
    result = builder_agent.drive_tool(harness, "grow", {"table": "users", "count": 0})
    assert result.is_error and "count" in result.content
    result = builder_agent.drive_tool(harness, "build", {"target": "environment", "extra": 1})
    assert result.is_error
    unknown = builder_agent.drive_tool(harness, "compile_tool", {"name": "no_such_tool"})
    assert unknown.is_error and ("no traces" in unknown.content.lower() or "no mined tool" in unknown.content)


def test_a_stage_target_after_the_build_is_served_from_the_cache_and_only_runs_upstream(driven):
    plan = BuildPlan(workdir=driven["workdir"], iterate=True, model=Bodies(), max_attempts=0)
    harness = builder_agent.build_harness(plan)
    result = builder_agent.drive_tool(harness, "build", {"target": "cluster"})
    assert not result.is_error, result.content
    stages = {s["name"]: s for s in result.details["stages"]}
    assert set(stages) == {"mine", "cluster"}, "no files to ingest, so the traces come off disk and mine is first"
    # The first build mined and clustered before the anchor existed; the anchor is in every key now
    # (D81), so the two run once more, without a model, and are served from the cache from then on.
    assert not any(s["cached"] for s in stages.values())
    assert result.details["produced"] == ["sigs", "schema", "categories", "tasks"]
    assert "gate rulings: cluster pass" in result.content
    again = builder_agent.drive_tool(harness, "build", {"target": "cluster"})
    assert all(s["cached"] for s in again.details["stages"])
    assert again.details["produced"] == result.details["produced"]
    recluster = builder_agent.drive_tool(harness, "recluster", {})
    assert not recluster.is_error and recluster.details["target"] == "cluster"


def test_compile_tool_recompiles_one_body_and_releases_every_body(driven):
    plan = BuildPlan(workdir=driven["workdir"], iterate=True, model=Bodies(), max_attempts=0)
    harness = builder_agent.build_harness(plan)
    before = json.loads((driven["workdir"] / "bodies.json").read_text(encoding="utf-8"))
    result = builder_agent.drive_tool(harness, "compile_tool", {"name": "get_user_details"})
    assert not result.is_error, result.content
    stages = {s["name"]: s for s in result.details["stages"]}
    assert stages["compile_tools"]["cached"] is False and stages["starting_state"]["cached"] is True
    after = json.loads((driven["workdir"] / "bodies.json").read_text(encoding="utf-8"))
    assert after == before, "one tool recompiled by the same model, the rest read back: every body is still there"
    assert set(plan.store["bodies"]) == set(before)
    gates = {g["stage"] for g in json.loads((driven["workdir"] / "gates.json").read_text(encoding="utf-8"))}
    assert {"parses", "intent", "derive_verifier"} <= gates, "the sandbox rulings were appended, the rest kept"


def test_replay_one_task_keeps_the_other_tasks_replays(driven):
    plan = BuildPlan(workdir=driven["workdir"], iterate=True, model=Bodies(), max_attempts=0)
    harness = builder_agent.build_harness(plan)
    replays = json.loads((driven["workdir"] / "replays.json").read_text(encoding="utf-8"))
    task = sorted(replays)[0]
    result = builder_agent.drive_tool(harness, "replay", {"task": task})
    assert not result.is_error, result.content
    assert {s["name"] for s in result.details["stages"] if not s["cached"]} == {"replay_reference"}
    assert set(plan.store["replays"]) == set(replays)
    assert "replay_reference" in result.content
    missing = builder_agent.drive_tool(harness, "replay", {"task": "no_such_task"})
    assert missing.is_error and "no Task is named" in missing.content


# --- the model-driven path ---------------------------------------------------

def test_a_scripted_model_driving_the_session_calls_build_and_reads_the_rulings(model_driven):
    result, model = model_driven["result"], model_driven["agent_model"]
    assert result["status"] == "complete" and result["tool_result"]["is_error"] is False
    assert len(model.calls) == 2
    tools = [t["name"] for t in model.calls[0]["tools"]]
    assert tools == ["build", "recluster", "grow", "compile_tool", "replay", "reroll"]
    system = model.calls[0]["messages"][0]
    assert "You are the Builder" in json.dumps(system)
    second = json.dumps(model.calls[1]["messages"])
    assert "gate rulings:" in second and "build environment: complete" in second
    events = model_driven["events"]
    assert [e.name for e in events if isinstance(e, StageStart)][:3] == ["ingest", "mine", "cluster"]
    assert [e.type for e in events][0] == "agent_start" and events[-1].type == "agent_end"
    kinds = [e.type for e in events]
    assert kinds.index("stage_start") < kinds.index("tool_execution_end")


def test_the_model_driven_build_leaves_the_same_artifacts_as_the_driver(driven, model_driven):
    # The trees as each build left them, before the targeted tools below touched the driver's workdir.
    left, right = driven["tree"], model_driven["tree"]
    assert set(left) == set(right)
    different = [name for name in left if left[name] != right[name]]
    assert different == [], different
    assert driven["result"]["env_id"] == model_driven["result"]["env_id"]
    assert driven["result"]["rulings"] == model_driven["result"]["rulings"]


def test_a_model_that_never_calls_build_is_a_build_error(tmp_path, request):
    model = TestModel([_reply("I would rather not.")])
    with pytest.raises(build_module.BuildError, match="never called build"):
        builder_agent.run_builder(tmp_path, model=Bodies(), files=[_fixture(request)], agent_model=model,
                                  max_attempts=0)


def test_a_driver_build_with_nothing_to_build_raises_the_way_the_cli_expects(tmp_path):
    with pytest.raises(build_module.BuildError, match="no Traces"):
        builder_agent.run_builder(tmp_path)


def test_build_tools_render_the_summary_and_keep_the_payload_in_the_model(tmp_path):
    result = builder_tools.BuildResult(summary="build x: complete; 1 stages", target="x", status="complete", passed=True,
                                       stage_gates=[builder_tools.Ruling(stage="g", passed=False, failures=["why"])],
                                       stages=[builder_tools.StageReport(name="s", status="cached", cached=True)],
                                       payload={"tasks": ["task_1"]})
    text = builder_tools.render(result)
    assert text.splitlines() == ["build x: complete; 1 stages", "stages: s (cached)", "rulings: g fail (why)"]
    assert "task_1" not in text


def test_the_harness_of_the_driver_refuses_a_model_turn(tmp_path):
    harness = builder_agent.build_harness(BuildPlan(workdir=tmp_path))
    assert isinstance(harness.model, builder_agent.DriverModel)
    with pytest.raises(RuntimeError, match="no model turn"):
        harness.model.query([])
