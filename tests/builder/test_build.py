"""build.py is the one wiring cli build and cli run go through; this runs it over the fixture with no live model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conftest import PTR
from kullback import sampling
from kullback.ai.provider import TestModel
from kullback.builder import build as build_module
from kullback.builder import pipeline
from kullback.builder import repair as repair_module
from kullback.builder import tools as builder_tools
from kullback.builder.build import BuildPlan
from kullback.gates import tool_runs
from kullback.runner.records import Task, ToolCall, ToolSig, Trace, Turn, Verifier
from test_e2e import TOOL_BODIES


class Bodies(TestModel):
    """A Builder model that answers each tool-body request with the body that tool needs.

    compile_env asks for one body at a time and names the tool in the prompt, so the reply is
    picked by name rather than by call order.
    """

    def __init__(self) -> None:
        super().__init__(["return None"], loop=True)
        self.by_name = {name: TestModel([body]).replies[0] for name, body in TOOL_BODIES.items()}
        self.fallback = self.replies[0]

    def query(self, messages, tools=None, config=None):
        # The system message now names every tool in the build (docs/prompt-caching.md item 1), so
        # a bare substring search would always match the first name in self.by_name. "Tool: <name>"
        # is the one line compile_env.py writes only for the tool actually being asked about.
        text = " ".join(str(m.get("content") or "") for m in messages)
        chosen = next((reply for name, reply in self.by_name.items() if f"Tool: {name}" in text), self.fallback)
        self.replies, self.index = [chosen], 0
        return super().query(messages, tools=tools, config=config)


@pytest.fixture(scope="module")
def built(tmp_path_factory, request) -> Path:
    workdir = tmp_path_factory.mktemp("build")
    fixture = Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"
    build_module.build(workdir, model=Bodies(), files=[fixture], max_attempts=0)
    return workdir


def test_the_build_writes_the_records_the_report_reads(built):
    for name in ("schema.json", "tool_sigs.json", "tasks.json", "tasks_frozen.json", "anchor.json",
                 "canon-rules.json", "environment.json", "constraints.json", "policy_coverage.json",
                 "scorecard.json", "bodies.json"):
        assert (built / name).is_file(), name
    assert list((built / "tasks").glob("*.json"))
    assert (built / "env" / "db.json").is_file()
    assert (built / "pipeline" / "state.json").is_file()


def test_the_anchor_is_chosen_before_the_first_builder_stage(built):
    """D81: a Builder stage that ran with nothing held out would be a stage fitted to every Run."""
    anchor = json.loads((built / "anchor.json").read_text(encoding="utf-8"))
    tasks = json.loads((built / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    assert set(anchor["held_out"]) == {t["id"] for t in tasks}


def test_the_env_id_covers_the_emitted_files(built):
    """Two worlds holding different rows must not share one env_id (design section 5)."""
    environment = json.loads((built / "environment.json").read_text(encoding="utf-8"))
    assert set(environment["files"]) >= {"db.json", "tasks.json", "tools.py", "policy.md"}


def test_the_canonicalizer_rules_are_learned_and_saved(built):
    """Every caller reads the customer's rules from one file rather than the module defaults (D39)."""
    rules = json.loads((built / "canon-rules.json").read_text(encoding="utf-8"))
    assert rules["id_patterns"]


def test_a_second_build_is_served_from_the_cache(built, tmp_path):
    """The stage cache is what makes `build --iterate` cheap (design section 8): with no new file
    and no code change, every stage but the three that could not yet have a settled key comes back
    from the cache. mine and cluster ran on the first build with no anchor.json on disk, and the
    anchor is part of every stage's key (D81), so those two are the ones that run once more.

    intent is the third, and for a different reason: it declares `intents/` as an input path so a
    repair_intent is seen by the next full build, and the first build wrote that folder, so the
    second build's key has moved. compile_tools is the fourth and it is the same reason: it declares
    `replays.json` as an input path so the calls the replay of the References failed on are evidence
    for the next body (D191), and the first build's replay wrote that file after compile_tools had
    run. starting_state is the fifth and the same again: it declares `bodies.json`, because a column
    a Task first touches with a write is pinned by running that tool's body (D202), and the first
    build wrote the bodies after the Starting state had been built. All of them settle on the run
    after, which is what the third build here shows: the stages read the same records back, write
    the same bytes, and the keys stop moving.

    The rebuild goes into a copy of the built workdir, so the module's shared fixture is left as
    the first build wrote it and the tests that read it do not depend on running after this one."""
    workdir = tmp_path / "again"
    shutil.copytree(built, workdir)
    result = build_module.build(workdir, iterate=True, model=Bodies())
    assert result["status"] == "complete"
    statuses = json.loads((workdir / "pipeline" / "state.json").read_text(encoding="utf-8"))["statuses"]
    # ingest is the previous build's own record, carried over because this build had no file to
    # ingest and so ran no ingest stage at all.
    assert {name for name, status in statuses.items() if status != "cached"} == {
        "ingest", "mine", "readers", "cluster", "intent", "compile_tools", "starting_state"}
    assert statuses["ingest"] == "ran"

    third = build_module.build(workdir, iterate=True, model=Bodies())
    assert third["status"] == "complete"
    statuses = json.loads((workdir / "pipeline" / "state.json").read_text(encoding="utf-8"))["statuses"]
    assert {name for name, status in statuses.items() if status != "cached"} == {"ingest"}


def test_a_new_tool_lesson_makes_compile_tools_run_again_instead_of_hitting_the_cache(built, tmp_path):
    """The live build asked for the same recompile seven times and was answered "5 stages, 5 from
    cache" every time: the lesson the request had recorded was in nothing the stage declared, so
    the key never moved and the narrowed rerun handed back the body it already had. The lesson file
    is a declared input path now, and the lesson itself reaches that tool's prompt."""
    from kullback.builder import memory

    workdir = tmp_path / "lesson"
    shutil.copytree(built, workdir)
    name = sorted(json.loads((workdir / "bodies.json").read_text(encoding="utf-8")))[0]
    model = Bodies()
    plan = BuildPlan(workdir=workdir, iterate=True, model=model, max_attempts=0)
    build_module.execute(plan, "compile_tools", tools=[name])
    again = build_module.execute(plan, "compile_tools", tools=[name])
    assert again.reports["compile_tools"].cached is True, "nothing changed, so the cache is right to answer"

    memory.record_lesson(workdir, name, ["executes: the body raised KeyError on every recorded call"])
    after = build_module.execute(plan, "compile_tools", tools=[name])
    assert after.status == "complete"
    assert after.reports["compile_tools"].cached is False, "a new lesson is a new question for the stage"
    sent = [" ".join(str(m.get("content") or "") for m in call["messages"]) for call in model.calls]
    assert any("raised KeyError on every recorded call" in text for text in sent), \
        "the lesson has to reach the compiler prompt, or the recompile asks the failed question again"
    assert all(f"Tool: {name}" in text for text in sent), "the narrowed rerun compiles that tool alone"


def test_a_kept_body_that_ignores_its_arguments_is_marked_hardcoded_where_the_next_round_reads_it(
    built, tmp_path
):
    """A body no attempt got through is assisted, and the gates say which one it fell at; none of
    them says the body never read its arguments, so a hint reaches nothing."""
    from kullback.builder import compile_env, memory

    workdir = tmp_path / "hardcoded"
    shutil.copytree(built, workdir)
    outcomes = json.loads((workdir / "tool_call_outcomes.json").read_text(encoding="utf-8"))
    name = max(sorted(outcomes), key=lambda tool: len(outcomes[tool]))
    bodies = json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))
    bodies.pop(name)  # so D174 has no kept body to decline this one against
    (workdir / "bodies.json").write_text(json.dumps(bodies), encoding="utf-8")
    plan = BuildPlan(workdir=workdir, iterate=True, max_attempts=0,
                     model=TestModel(['return {"answer": "the same one every time"}'], loop=True))
    build_module.execute(plan, "compile_tools", tools=[name])

    builds = json.loads((workdir / "tool_builds.json").read_text(encoding="utf-8"))
    written = json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))
    assert builds[name]["hardcoded"] is True
    assert written[name].startswith(compile_env.HARDCODED_MARK)
    assert compile_env.HARDCODED_LESSON in memory.lesson_for(workdir, name)


def test_the_hardcoded_lesson_is_written_once_so_the_stage_key_does_not_move_on_every_run(tmp_path):
    from kullback.builder import compile_env, memory

    build_module._record_hardcoded_lesson(tmp_path, "post_entry")
    build_module._record_hardcoded_lesson(tmp_path, "post_entry")
    assert memory.load_tool_lessons(tmp_path)["post_entry"] == [[compile_env.HARDCODED_LESSON]]


# --- the Intent stage's ratchet, and the folder it declares ---


def _clinic_inputs(run_ids: list[str]) -> dict:
    """One Task in an invented clinic domain, with a Trace per Run and the signatures the stage reads."""
    traces = [
        Trace(trace_id=run_id, raw_hash="raw", ingest_version="0", source="test",
              turns=[Turn(idx=0, role="user", content="please reschedule appointment a1", raw_ptr=PTR)],
              tool_calls=[ToolCall(id=f"{run_id}-0", name="reschedule_appointment",
                                   args={"appointment_id": "a1"}, result={"status": "moved"}, raw_ptr=PTR)],
              raw_ptr=PTR)
        for run_id in run_ids
    ]
    return {"tasks": [Task(id="task_1", category_id="cat_1", run_ids=list(run_ids))],
            "traces": traces,
            "sigs": [ToolSig(name="reschedule_appointment", kind="write")]}


def _record_intent(workdir: Path, task_id: str, text: str, run_ids: list[str], grounded: bool = True) -> None:
    """Put one Intent under intents/ the way an earlier run of the stage, or a repair, would leave it."""
    path = workdir / "intents" / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_id": task_id, "text": text, "grounded": grounded,
                                "run_coverage": {text: sorted(run_ids)}}), encoding="utf-8")


def _run_intent_stage(stage, workdir: Path, inputs: dict, anchor=None) -> dict:
    """One stage driven on its own, the way the scheduler drives it: the context knows the inputs,
    so a Builder stage's ctx.evidence is the seed set of exactly these (D220)."""
    ctx = pipeline.StageContext(stage, workdir, anchor, lambda usd, item="": None, inputs=inputs)
    return stage.fn(ctx, pipeline.withhold(stage, inputs) if stage.filtered else inputs)


def test_a_full_intent_run_keeps_a_grounded_record_and_never_asks_the_model_for_it(tmp_path):
    """The ratchet: a stage never replaces a passing artifact with a failing one. A repaired Intent
    has to survive the next full build, and an iterated build must not pay to write a line again
    that already grounds on every one of its Task's Runs."""
    inputs = _clinic_inputs(["r1", "r2"])
    _record_intent(tmp_path, "task_1", "reschedule appointment a1", ["r1", "r2"])
    model = TestModel([])  # any call is recorded before it raises, so calls is the evidence either way
    out = _run_intent_stage(build_module._intent_stage(model), tmp_path, inputs)
    assert model.calls == []
    assert out["intents"]["task_1"]["text"] == "reschedule appointment a1"
    assert out["intents"]["task_1"]["grounded"] is True


def test_a_full_intent_run_writes_again_for_a_task_whose_member_runs_changed(tmp_path):
    """A grounded line says what every member Run showed at the time. A Task that has gained or
    lost a Run is a different question of the evidence, so the record no longer stands for it."""
    inputs = _clinic_inputs(["r1", "r2", "r3"])
    _record_intent(tmp_path, "task_1", "reschedule appointment a1", ["r1", "r2"])
    model = TestModel(["reschedule appointment a1 to friday"], loop=True)
    out = _run_intent_stage(build_module._intent_stage(model), tmp_path, inputs)
    assert model.calls, "the Task's members changed, so its Intent is written again"
    assert out["intents"]["task_1"]["text"] == "reschedule appointment a1 to friday"


def test_a_full_intent_run_writes_again_for_a_record_that_never_grounded(tmp_path):
    """The ratchet holds a passing artifact, not a failing one: an ungrounded line is a Task with no
    Verdict, and every build gets another go at it."""
    inputs = _clinic_inputs(["r1", "r2"])
    _record_intent(tmp_path, "task_1", "collect the parking fee", ["r1", "r2"], grounded=False)
    model = TestModel(["reschedule appointment a1"], loop=True)
    out = _run_intent_stage(build_module._intent_stage(model), tmp_path, inputs)
    assert model.calls, "nothing grounded, so there is nothing to keep"
    assert out["intents"]["task_1"]["text"] == "reschedule appointment a1"
    assert out["intents"]["task_1"]["grounded"] is True


def test_a_narrowed_intent_run_writes_its_target_again_even_when_the_record_grounds(tmp_path):
    """repair_intent(task_id, hint) is an explicit ask. If the ratchet applied to it, a mechanic
    who read a grounded line and disagreed with it would be answered with that same line."""
    inputs = _clinic_inputs(["r1", "r2"])
    _record_intent(tmp_path, "task_1", "reschedule appointment a1", ["r1", "r2"])
    model = TestModel(["reschedule appointment a1 to friday"], loop=True)
    stage = build_module._intent_stage(model, only=["task_1"], hints={"task_1": "name the day"})
    out = _run_intent_stage(stage, tmp_path, inputs)
    assert model.calls, "the repair's target is written again however well it grounded"
    prompts = [" ".join(str(m.get("content") or "") for m in call["messages"]) for call in model.calls]
    assert all("A repair asks for this: name the day" in prompt for prompt in prompts)
    assert out["intents"]["task_1"]["text"] == "reschedule appointment a1 to friday"


def test_a_repaired_intent_file_moves_the_full_builds_key_and_reaches_the_artifact(built, tmp_path):
    """Build 12: the Builder repaired six Intents, then asked for the whole build. The intent stage
    declared intents/ only on a narrowed run, so the full run's key had not moved, the stage came
    back from the cache, and the intents the Examiner derives its Verifiers from were the lines from
    before the repair. The folder is declared on every run now."""
    workdir = tmp_path / "repaired"
    shutil.copytree(built, workdir)
    tasks = json.loads((workdir / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    task = sorted(tasks, key=lambda t: t["id"])[0]
    line = "reschedule appointment a1"
    _record_intent(workdir, task["id"], line, task["run_ids"])

    plan = BuildPlan(workdir=workdir, iterate=True, model=Bodies(), max_attempts=0)
    result = build_module.execute(plan, "intent")
    assert result.reports["intent"].cached is False, "a changed file under intents/ has to move the key"
    assert result.artifacts["intents"][task["id"]]["text"] == line

    again = build_module.execute(plan, "intent")
    assert again.reports["intent"].cached is True, "nothing changed this time, so the cache may answer"
    assert again.artifacts["intents"][task["id"]]["text"] == line


def test_wrap_sets_a_prompt_cache_key_scoped_to_the_build_and_stage(tmp_path):
    """docs/prompt-caching.md item 4: one short string per build and stage."""
    model = build_module._wrap(TestModel(["hi"]), "compile_tools", tmp_path, None, model_id="anthropic/claude-opus-5")
    build_id = build_module._build_id(tmp_path)
    assert model.prompt_cache_key == f"kullback-{build_id}-compile_tools"
    other = build_module._wrap(TestModel(["hi"]), "compile_policy", tmp_path, None, model_id="anthropic/claude-opus-5")
    assert other.prompt_cache_key != model.prompt_cache_key


def test_wrap_memoizes_a_builder_stage_but_not_the_candidate(tmp_path):
    """docs/prompt-caching.md item 3: run_batch's candidate model must stay a fresh sample."""
    from kullback.ai.provider import MemoModel

    stage = build_module._wrap(TestModel(["hi"]), "compile_tools", tmp_path, None, model_id="anthropic/claude-opus-5")
    assert isinstance(stage.inner, MemoModel)

    candidate = build_module._wrap(
        TestModel(["hi"]), "candidate", tmp_path, None, model_id="anthropic/claude-opus-5",
        cap_context=False, memoize=False,
    )
    assert not isinstance(candidate.inner, MemoModel)


def test_a_second_build_in_the_same_workdir_serves_stage_calls_from_the_memo(built):
    """The memo (item 3) is content addressed on the workdir and lives under model_cache/, a
    folder `build()`'s own cache-clear (`_clear_cache`, on iterate=False) never touches, since
    that only drops the pipeline stage cache under cache/. A rebuild with iterate=False forces
    every Builder stage to run its model calls again, which is what turns the surviving memo into
    an actual hit rather than a stage that was merely skipped."""
    model_cache = built / "model_cache"
    assert model_cache.is_dir() and list(model_cache.glob("*.json")), "the first build must have written a memo"
    before = {p.name for p in model_cache.glob("*.json")}

    build_module.build(built, model=Bodies())  # iterate=False: drops the pipeline cache, keeps the memo
    after = {p.name for p in model_cache.glob("*.json")}
    # judge_lessons reads back what the first build wrote to memory_dir, so its prompt (and hence
    # its key) legitimately differs between the two builds and can add new memo entries; nothing
    # the first build wrote may ever be dropped, memo or no memo.
    assert before <= after, "a repeat build must not drop a memo entry the first build wrote"

    totals = json.loads((built / "budget.json").read_text(encoding="utf-8"))
    assert totals["total"]["memo_hits"] > 0, "the rerun's stage calls should have hit the memo"


def test_the_judge_runs_on_the_model_it_was_named_with_and_the_build_model_otherwise(tmp_path):
    """D160: the model that writes the Environment need not be the one that rules on it."""
    plan = BuildPlan(workdir=tmp_path / "own", model=TestModel(["hi"], name="vendor/large"))
    assert plan.models["reference_judge"].model_id == "vendor/large"
    assert plan.models["second_judge"] is None
    assert plan.judge_model_ids() == {"build": "vendor/large"}

    named = BuildPlan(workdir=tmp_path / "named", model=TestModel(["hi"], name="vendor/large"),
                      judge_model=TestModel(["hi"], name="other/small"),
                      second_judge_model=TestModel(["hi"], name="third/tiny"))
    assert named.models["reference_judge"].model_id == "other/small"
    assert named.models["second_judge"].model_id == "third/tiny"
    assert named.judge_model_ids() == {"build": "vendor/large", "judge": "other/small",
                                       "second_judge": "third/tiny"}
    # every stage model goes through budget.py, judges included (D65, D86)
    assert named.models["reference_judge"].stage == "reference_judge"


def test_a_build_with_no_model_has_no_judge_unless_one_is_named(tmp_path):
    """The judge is a model call: without a model there is none, and naming one gives a build that
    judges without a Builder model of its own."""
    assert BuildPlan(workdir=tmp_path / "none").models["reference_judge"] is None
    judged = BuildPlan(workdir=tmp_path / "judge-only", judge_model=TestModel(["hi"], name="other/small"))
    assert judged.models["reference_judge"].model_id == "other/small"


def test_the_user_model_lands_in_the_wrapped_models_and_the_rules_drive_without_it(tmp_path):
    """D232: a build that names a user model wraps it under `user_agent` for the re-rolls; a build
    that names none keeps None there, so every Run is driven by the rule-driven user as before."""
    assert BuildPlan(workdir=tmp_path / "none").models["user_agent"] is None
    code_driven = BuildPlan(workdir=tmp_path / "coded",
                             user_agent_model=TestModel(["hi"], name="other/small"))
    assert code_driven.models["user_agent"].model_id == "other/small"
    assert code_driven.models["user_agent"].stage == "user_agent"
    both = BuildPlan(workdir=tmp_path / "both", model=TestModel(["hi"], name="vendor/large"),
                     user_agent_model=TestModel(["hi"], name="other/small"))
    assert both.models["user_agent"].model_id == "other/small"
    assert both.models["reference_judge"].model_id == "vendor/large"


def test_wrap_refuses_an_unpriced_model_before_building_the_wrapper(tmp_path):
    """D86: an unpriced model under a ceiling must be refused in _wrap itself, not handed
    ceiling=None and left to run completely unmetered."""
    from kullback.runner import budget

    ceiling = budget.Ceiling(usd=10.0)
    with pytest.raises(budget.UnpricedModel):
        build_module._wrap(TestModel(["hi"]), "compile_tools", tmp_path, ceiling, model_id="openai/mystery")


def test_the_build_environment_gate_rules_on_the_referenced_ids_and_the_synthetic_rows(built):
    """Unwired, both halves of the gate default to empty and never fire: a db.json missing an id
    the traces name, and an untagged synthetic row, would both pass."""
    from kullback.runner.records import Environment

    stage = build_module._environment_stage("domain")
    environment = Environment.model_validate(json.loads((built / "environment.json").read_text(encoding="utf-8")))
    ctx = build_module.pipeline.StageContext(stage, built, None, lambda usd, item="": None)
    clean = stage.gate(ctx, {"environment": environment, "referenced_ids": [],
                             "synthetic_rows_tagged": [{"id": "row-1", "synthetic": True}]})
    assert clean.passed, clean.failures
    ruled = stage.gate(ctx, {"environment": environment, "referenced_ids": ["#W-nowhere"],
                             "synthetic_rows_tagged": [{"id": "row-1"}]})
    assert not ruled.passed
    assert any("#W-nowhere" in failure for failure in ruled.failures)
    assert any("not tagged" in failure for failure in ruled.failures)
    # And the build that ran recorded that gate over the ids and rows its own stage produced.
    state = json.loads((built / "pipeline" / "state.json").read_text(encoding="utf-8"))
    assert any(g.get("stage") == "build_environment" and g["pass"] for g in state["gates"])


def test_budget_json_is_written_because_every_model_goes_through_budget_py(built):
    """D65 and D86 only bind if the Builder's model is wrapped; unwrapped, nothing records a call."""
    totals = json.loads((built / "budget.json").read_text(encoding="utf-8"))
    assert totals["total"]["calls"] > 0
    assert "compile_tools" in totals["stages"]


def test_run_batch_writes_one_jsonl_per_candidate_run(built):
    """cli run's whole job: a Candidate over the built Environment, one Run file each (D49, D74)."""
    from kullback.ai.provider import TestModel

    task_id = sorted(p.stem for p in (built / "tasks").glob("*.json"))[0]
    candidate = TestModel([
        {"tool_calls": [{"id": "x1", "name": "get_order_details", "arguments": {"order_id": "#W6390527"}}]},
        {"content": "done ###STOP###"},
    ], loop=True)
    out = build_module.run_batch(built, task_id, candidate, count=2, seed=3)
    assert len(out["runs"]) == 2
    for path in out["runs"]:
        lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines[-1]["task_id"] == task_id
        assert lines[-2]["type"] == "stop" and "end_state" in lines[-2]["payload"]
    routes = [record for path in out["runs"]
              for record in [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
              if record.get("type") == "tool_result"]
    assert routes and all(r["route"] == "code" for r in routes)


def test_an_iterated_build_with_nothing_to_ingest_keeps_the_ingest_record(tmp_path_factory, request):
    """Two things about the one pipeline/state.json on disk, over one build sequence.

    build() runs two Pipelines (ingest/mine/cluster, then the Builder stages) that each write the
    file at the same fixed path, so the file has to hold both runs' statuses and gates and not just
    whichever pipeline wrote last. Then `build --iterate` with no new file runs no ingest stage, so
    the previous ingest's status and gate would drop out of the file; report.py reads it once, so
    they are carried over."""
    def state_of(workdir):
        state = json.loads((workdir / "pipeline" / "state.json").read_text(encoding="utf-8"))
        return state["statuses"], {g.get("stage") for g in state["gates"]}

    workdir = tmp_path_factory.mktemp("iterate-state")
    fixture = Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"
    build_module.build(workdir, model=Bodies(), files=[fixture], max_attempts=0)
    statuses, gate_stages = state_of(workdir)
    for name in ("ingest", "mine", "cluster"):
        assert name in statuses, name
    assert "ingest" in gate_stages

    build_module.build(workdir, iterate=True, model=Bodies(), max_attempts=0)
    statuses, gate_stages = state_of(workdir)
    assert "ingest" in statuses
    assert "ingest" in gate_stages


def test_a_stage_that_delegates_to_a_module_carries_that_module_in_its_code_version():
    """R42: `compile_policy:{model}` never changed when policy.py did, so an edited compiler was
    served its stale cache entry. compile_tools is the stage with the most of its work somewhere
    else: a fix to sandbox.py left every body it had already refused sitting in the cache for
    --iterate to reuse. The derivation is the Examiner's now (phase 5) and has no stage here."""
    from kullback.builder import compile_env, memory, policy, sandbox
    from kullback.gates import verifier_suite
    versions = {name: build_module._module_hash(mod)
                for name, mod in (("policy", policy), ("memory", memory), ("suite", verifier_suite),
                                  ("compile_env", compile_env), ("sandbox", sandbox))}
    assert len(set(versions.values())) == 5
    assert build_module._policy_stage(None).code_version.endswith(versions["policy"])
    tools_version = build_module._tools_stage(None, 3).code_version
    assert versions["compile_env"] in tools_version and versions["sandbox"] in tools_version


def test_the_dag_ends_at_rerolls_and_names_no_verifier_target(built):
    plan = BuildPlan(workdir=built, iterate=True, model=Bodies(), max_attempts=0)
    names = [stage.name for stage in build_module.stages(plan)]
    assert names[-1] == "rerolls" and "derive_verifier" not in names
    assert build_module.TARGET_ALL == "environment"
    with pytest.raises(pipeline.GraphError):
        build_module.execute(plan, target="derive_verifier")


def _cached_plan(built: Path) -> BuildPlan:
    """The fixture build's store read back through the cache: every stage served, nothing rebuilt."""
    plan = BuildPlan(workdir=built, iterate=True, model=Bodies(), max_attempts=0)
    build_module.execute(plan)
    return plan


def test_probe_runner_builds_one_run_in_the_task_world_under_probes_without_the_examiner(built):
    plan = _cached_plan(built)
    run_probe = build_module.probe_runner(plan)
    task = plan.store["tasks"][0]
    verifier = Verifier(task_id=task.id, atoms=[])
    run = run_probe(Bodies(), verifier)
    assert run.run_id == f"probe-{task.id}" and run.task_id == task.id and run.model.startswith("probe:")
    written = list((built / "probes").rglob("*.jsonl"))
    assert written and all(task.id in str(path) for path in written)
    assert not list((built / "runs" / task.id).glob("probe-*")), "a probe is never a Run of the Task"
    assert not (built / "verifiers").exists() and not (built / "task_status.json").exists()
    source = Path(build_module.__file__).read_text(encoding="utf-8")
    assert "from kullback.examiner" not in source and "import kullback.examiner" not in source, \
        "the Builder hands the Runner over; it never imports the Examiner"


def test_reroll_runner_writes_runs_under_the_task_with_the_prefix_given_and_leaves_rerolls_json_alone(built):
    plan = _cached_plan(built)
    run_rerolls = build_module.reroll_runner(plan)
    task = plan.store["tasks"][0]
    before = json.dumps(plan.store["rerolls"], sort_keys=True, default=str)
    state_before = (built / "pipeline" / "state.json").read_bytes()
    rows = run_rerolls(task.id, 2, "reroll-r1")
    assert [row["run_id"] for row in rows] == [f"reroll-r1-{task.id}-0", f"reroll-r1-{task.id}-1"]
    for row in rows:
        assert Path(row["path"]).is_file() and Path(row["path"]).parent == built / "runs" / task.id
        assert row["termination_reason"]
    assert json.dumps(plan.store["rerolls"], sort_keys=True, default=str) == before, \
        "the stage's rows are not the Examiner's to rewrite"
    assert (built / "pipeline" / "state.json").read_bytes() == state_before
    listed = json.loads((built / "runs.json").read_text(encoding="utf-8"))
    ids = {r["run_id"] for r in (listed["runs"] if isinstance(listed, dict) else listed)}
    assert {row["run_id"] for row in rows} <= ids
    with pytest.raises(build_module.BuildError, match="no Task is named"):
        run_rerolls("nobody", 1, "reroll-r1")


def test_a_reroll_carries_the_seed_its_own_run_id_draws_and_draws_it_again_when_it_is_made_again(built):
    """D212: the seed on a Run record is a function of the Run id and the build salt, not of the
    position the Run held in the batch, so a re-roll discarded and made again is the one it replaced."""
    plan = _cached_plan(built)
    run_rerolls = build_module.reroll_runner(plan)
    task = plan.store["tasks"][0]
    salt = sampling.build_salt(built)
    rows = run_rerolls(task.id, 2, "reroll-r1")
    seeds = {row["run_id"]: _seed_of(row["path"]) for row in rows}
    assert seeds == {row["run_id"]: sampling.sample_seed(build_module.RUN_SEED_KIND, row["run_id"], salt)
                     for row in rows}
    assert len(set(seeds.values())) == 2, "two Runs of one batch are two draws, not one"
    for row in rows:
        Path(row["path"]).unlink()
    again = run_rerolls(task.id, 2, "reroll-r1")
    assert {row["run_id"]: _seed_of(row["path"]) for row in again} == seeds


def _seed_of(path: str) -> int:
    """The seed the Run record on disk carries, off the run event the loop wrote first."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        body = json.loads(line)
        if "seed" in body:
            return int(body["seed"])
    raise AssertionError(f"no event of {path} names a seed")


def test_a_task_the_rerolls_stage_skips_loses_the_rerolls_an_earlier_build_left_it(built):
    """A Task with no confirmed re-play is not re-rolled, and its re-rolls from an earlier build go too:
    the second retail build's dead re-rolls sat under 36 Tasks a later build skipped, every one a
    provider error, and every count that globs runs/ read them as this build's."""
    plan = _cached_plan(built)
    task = plan.store["tasks"][0]
    stale = built / "runs" / task.id / f"reroll-{task.id}-7.jsonl"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"idx": 0, "type": "error", "payload": {"class": "env_error"}}\n', encoding="utf-8")
    replays = {tid: {run_id: {**row, "confirmed": False} for run_id, row in rows.items()}
               for tid, rows in plan.store["replays"].items()}
    inputs = {**{name: plan.store[name] for name in ("tasks", "user_rules", "schema", "sigs", "bodies", "db",
                                                     "environment", "canon_rules", "traces", "policy_text")},
              "replays": replays}
    stage = build_module._rerolls_stage(Bodies(), 1)
    out = _run_intent_stage(stage, built, inputs)
    assert out == {"rerolls": {}}, "nothing confirmed, so nothing re-rolled"
    assert not stale.exists()
    assert not list((built / "runs" / task.id).glob(f"reroll-{task.id}-*")), \
        "only the stage's own prefix goes; the Examiner's re-rolls under another prefix stay"


# --- a Task's re-rolls are keyed on that Task's own inputs ---

REROLL_INPUTS = ("tasks", "replays", "user_rules", "schema", "sigs", "bodies", "db",
                 "environment", "canon_rules", "traces", "policy_text")


class Rulings:
    """A ledger that keeps what a stage ruled instead of writing it, so a test can read the counts."""

    def __init__(self) -> None:
        self.results: list = []

    def record(self, stage, result):
        self.results.append(result)
        return result

    def write(self, stage, results):
        self.results.extend(results)

    def rulings(self, stage):
        return [result.stage for result in self.results]


def _run_rerolls(workdir: Path, inputs: dict, model, only=None) -> tuple[dict, dict]:
    """The re-rolls stage over this workdir as a build runs it: its rows, and its ruling's metrics."""
    stage = build_module._rerolls_stage(model, build_module.DEFAULT_REROLLS, only=only)
    ledger = Rulings()
    ctx = pipeline.StageContext(stage, workdir, None, lambda usd, item="": None, ledger=ledger,
                                inputs=inputs)
    rows = stage.fn(ctx, pipeline.withhold(stage, inputs))["rerolls"]
    return rows, ledger.results[-1].metrics


@pytest.fixture
def rerolled(built, tmp_path) -> tuple[Path, dict, dict]:
    """A copy of the fixture build with every Task's re-rolls settled: the workdir, the store the
    stage reads, and the rows the settling run left."""
    workdir = tmp_path / "keyed"
    shutil.copytree(built, workdir)
    inputs = {name: _cached_plan(built).store[name] for name in REROLL_INPUTS}
    rows, _ = _run_rerolls(workdir, inputs, Bodies())
    assert len(rows) >= 2, "the fixture has to re-roll more than one Task for these tests to say anything"
    return workdir, inputs, rows


def test_a_second_run_of_the_rerolls_stage_re_rolls_no_task_when_nothing_moved(rerolled):
    """Re-rolls were 75 to 82 percent of every live build's spend because one changed body or one
    repaired Intent re-ran every Task's Runs: the stage's key covers the whole toolkit and every
    Task. Keyed per Task, a repeat over the same inputs buys nothing and so runs nothing."""
    workdir, inputs, before = rerolled
    silent = TestModel([])  # a call is recorded before it raises, so calls is the evidence either way
    rows, metrics = _run_rerolls(workdir, inputs, silent)
    assert silent.calls == [], "nothing moved, so no Task is put to the model again"
    assert metrics["rerolled"] == 0 and metrics["reused"] == len(rows)
    assert metrics["rerolled_because"] == {}
    assert rows == before, "a reused Task lists the Runs it already has"
    assert all(Path(row["path"]).is_file() for task_rows in rows.values() for row in task_rows)


def test_a_changed_tool_body_re_rolls_only_the_tasks_whose_recordings_call_it(rerolled):
    """The key holds the bodies of the tools this Task's own recordings call (D171's grain). A body
    no recording of the Task calls may move without the Task's Runs going stale: the Run on disk is
    a sample already taken against the toolkit as it stood."""
    workdir, inputs, before = rerolled
    by_id = {trace.trace_id: trace for trace in inputs["traces"]}
    tasks = {task.id: task for task in inputs["tasks"] if task.id in before}
    called = {task_id: set(build_module._tools_called(task, by_id)) for task_id, task in tasks.items()}
    caller, keeper = sorted(tasks)[0], sorted(tasks)[1]
    target = sorted(called[caller] & called[keeper])[0]
    # The fixture's two re-rolled Tasks call the same tools, so the keeper is given a recording that
    # never makes the call; which tool that is comes from the store, never from a name written here.
    kept_runs = set(tasks[keeper].run_ids)
    traces = [trace if trace.trace_id not in kept_runs
              else trace.model_copy(update={"tool_calls": [call for call in trace.tool_calls
                                                           if call.name != target]})
              for trace in inputs["traces"]]
    settled = {**inputs, "traces": traces}
    _run_rerolls(workdir, settled, Bodies())

    bodies = {**inputs["bodies"], target: f"{inputs['bodies'][target]}\n# a body written another way"}
    rows, metrics = _run_rerolls(workdir, {**settled, "bodies": bodies}, Bodies())
    assert metrics["rerolled"] == 1 and metrics["reused"] == len(rows) - 1
    assert metrics["rerolled_because"] == {caller: f"body of tool {target}"}
    assert keeper in rows, "the keeper still lists the Runs it already had"


def test_a_task_whose_user_rules_changed_is_the_only_one_re_rolled(rerolled):
    """The Simulated user drives the Run, so its rules are the Task's own input. The Intent is not:
    nothing on the re-roll path reads it, and keying on it would buy re-rolls that change nothing."""
    workdir, inputs, before = rerolled
    target = sorted(before)[0]
    task = next(task for task in inputs["tasks"] if task.id == target)
    trace_id = next(run_id for run_id in task.run_ids if run_id in inputs["user_rules"])
    rules = inputs["user_rules"][trace_id]
    changed = {**inputs["user_rules"],
               trace_id: rules.model_copy(update={"style_sample": ["a turn the recording never had"]})}
    rows, metrics = _run_rerolls(workdir, {**inputs, "user_rules": changed}, Bodies())
    assert metrics["rerolled"] == 1 and metrics["reused"] == len(rows) - 1
    assert metrics["rerolled_because"] == {target: "user rules"}


def test_an_explicit_reroll_re_rolls_the_task_it_names_however_settled_its_key_is(rerolled):
    """The Builder's `reroll` tool and an `--iterate` naming a Task are explicit asks, the way a
    recompile is: a mechanic who wanted another sample must not be answered with the last one."""
    workdir, inputs, before = rerolled
    target = sorted(before)[0]
    rows, metrics = _run_rerolls(workdir, inputs, Bodies(), only=[target])
    assert set(rows) == {target}
    assert metrics["rerolled"] == 1 and metrics["reused"] == 0
    assert metrics["rerolled_because"] == {target: "an explicit re-roll was asked for"}
    assert [row["run_id"] for row in rows[target]] == [row["run_id"] for row in before[target]], \
        "the stage's re-rolls keep their names; the Examiner's rotate the prefix instead"


def test_a_task_whose_run_files_are_gone_is_re_rolled_however_settled_its_key_is(rerolled):
    """The key alone is not evidence the Runs exist: a workdir a build was killed in half way, or
    one whose Run files were cleared, has to buy them again rather than list files that are not there."""
    workdir, inputs, before = rerolled
    target = sorted(before)[0]
    for row in before[target]:
        Path(row["path"]).unlink()
    rows, metrics = _run_rerolls(workdir, inputs, Bodies())
    assert metrics["rerolled"] == 1
    assert metrics["rerolled_because"] == {target: "the Run files the key names are gone"}
    assert all(Path(row["path"]).is_file() for row in rows[target])


# --- what a tool is gated on (D74) ---

def _trace(trace_id, calls):
    from conftest import PTR
    from kullback.runner.records import Trace
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="1", source="tau2", tool_calls=calls, raw_ptr=PTR)


def _tc(name, args, error=None):
    from conftest import PTR
    from kullback.runner.records import ToolCall
    return ToolCall(name=name, args=args, raw_ptr=PTR, error=error)


def test_a_call_after_a_write_on_the_same_row_is_not_gate_evidence():
    trace = _trace("t", [
        _tc("get_order_details", {"order_id": "#W1"}),
        _tc("modify_pending_order_items", {"order_id": "#W1", "item_ids": ["11"], "new_item_ids": ["22"]}),
        _tc("get_order_details", {"order_id": "#W1"}),
        _tc("get_order_details", {"order_id": "#W2"}),
        _tc("get_product_details", {"product_id": "22"}),  # named by the write's list argument
    ])
    assert build_module.after_write_calls([trace], {"modify_pending_order_items"}) == {("t", 2), ("t", 4)}


def test_a_refused_write_changes_nothing_so_nothing_follows_it():
    from kullback.runner.records import ToolCallError
    refused = _tc("cancel_pending_order", {"order_id": "#W1"},
                  error=ToolCallError(**{"class": "business_error"}, payload="Non-pending order"))
    trace = _trace("t", [refused, _tc("get_order_details", {"order_id": "#W1"})])
    assert build_module.after_write_calls([trace], {"cancel_pending_order"}) == set()


def test_only_a_callers_own_call_is_evidence_for_the_body_of_a_tool():
    """D164: the compile stage writes a body against the calls of the callers the tool answers. A
    call the recording refused says nothing about what the body does."""
    from conftest import PTR
    from kullback.runner.records import ToolCall, ToolSig

    sigs = [ToolSig(name="get_loan_details"),
            ToolSig(name="read_shelf_lamp", callers=["user"], refused_callers=["assistant"])]
    callers = build_module.callers_by_tool(sigs)
    assert callers == {"get_loan_details": {"assistant"}, "read_shelf_lamp": {"user"}}

    def a_call(name, requestor):
        return ToolCall(name=name, args={}, requestor=requestor, raw_ptr=PTR)

    assert build_module.is_evidence_call(a_call("read_shelf_lamp", "user"), callers) is True
    assert build_module.is_evidence_call(a_call("read_shelf_lamp", "assistant"), callers) is False
    assert build_module.is_evidence_call(a_call("get_loan_details", "assistant"), callers) is True
    assert build_module.is_evidence_call(a_call("get_loan_details", "user"), callers) is False


def _outcome(call_id, replayed, detail="", tool="get_loan_details"):
    return {"tool": tool, "call_id": call_id, "replayed": replayed, "detail": detail}


DIFFERS = "get_loan_details({'loan_id': 'L9'}): hard columns differ: due_on: ours 2099-12-31, recorded 2026-01-05"


def test_a_task_whose_own_calls_all_replay_is_not_blocked_by_a_miss_on_another_tasks_call():
    """D171: the corpus counts every recorded call of the tool, a Task only the calls its own
    Traces made, and a tool assisted over the corpus differs on nobody else's Task."""
    outcomes = {"get_loan_details": [_outcome("c1", True), _outcome("c2", True),
                                     _outcome("c3", False, DIFFERS)]}
    out = build_module.attribute_fidelity(outcomes, {"c1": "t1", "c2": "t1", "c3": "t2"},
                                          ["get_loan_details"])
    assert out["tools"]["get_loan_details"] == {"calls": 3, "replayed": 2, "differing": 1, "assisted": True}
    assert out["tasks"]["t1"]["get_loan_details"] == {"replayed": 2, "differing": 0, "reasons": []}
    assert out["tasks"]["t2"]["get_loan_details"]["differing"] == 1
    assert "due_on" in out["tasks"]["t2"]["get_loan_details"]["reasons"][0]


def test_a_recorded_call_no_task_claims_is_evidence_about_the_body_and_about_no_task():
    """A Trace outside every Task still says whether the body is right; it costs no Task a Reference."""
    outcomes = {"get_loan_details": [_outcome("c1", False, DIFFERS), _outcome("c2", True)]}
    out = build_module.attribute_fidelity(outcomes, {"c2": "t1"}, ["get_loan_details"])
    assert out["tools"]["get_loan_details"]["differing"] == 1
    assert out["tasks"] == {"t1": {"get_loan_details": {"replayed": 1, "differing": 0, "reasons": []}}}


def test_the_rerolls_gate_is_not_green_over_runs_that_all_died():
    dead = {"t1": [{"termination_reason": "env_error"}] * 3, "t2": [{"termination_reason": "env_error"}]}
    gate = build_module.rerolls_gate(dead, 3)
    assert not gate.passed and gate.metrics["runs"] == 4 and gate.metrics["finished"] == 0
    assert "no re-roll finished" in gate.failures[0]
    one_alive = {"t1": [{"termination_reason": "env_error"}, {"termination_reason": "stop"}]}
    assert build_module.rerolls_gate(one_alive, 2).passed
    assert build_module.rerolls_gate({}, 3).passed  # no confirmed Task to re-roll is not a failure


def test_a_candidate_run_opens_with_the_system_prompt_the_user_and_the_tools(built):
    """A model asked for a first turn over an empty transcript with no tools is the second retail build's
    597 dead re-rolls; every Candidate Run is seeded like the recorded one instead."""
    from kullback.ai.provider import TestModel

    task_id = sorted(p.stem for p in (built / "tasks").glob("*.json"))[0]
    candidate = TestModel([{"content": "done ###STOP###"}], loop=True)
    out = build_module.run_batch(built, task_id, candidate, count=1)
    first = candidate.calls[0]
    roles = [m["role"] for m in first["messages"]]
    assert roles[0] == "system" and "user" in roles
    assert first["tools"] and {t["name"] for t in first["tools"]} >= {"get_order_details"}
    events = [json.loads(x) for x in Path(out["runs"][0]).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert events[0]["type"] == "user_turn" and events[0]["payload"]["text"]


# --- what a stage's code_version is hashed over, and the tool definitions the loop is handed ---

def test_every_stage_hashes_the_modules_it_delegates_to():
    """R42 for every stage, not only compile_tools: the first live build was served a schema mined before D106."""
    from kullback.builder import cluster, compile_env, mine, synth
    from kullback.gates import fidelity
    from kullback.runner import replay as replay_mod

    # D214: the rule-driven Simulated user lives in kullback/user now, and the stage hashes the
    # module that can actually change, not the name that re-exports it.
    from kullback.user import rules as user_sim
    assert build_module._mine_stage().code_version.endswith(build_module._module_hash(mine))
    assert build_module._module_hash(cluster) in build_module._cluster_stage().code_version
    assert build_module._module_hash(user_sim) in build_module._user_rules_stage().code_version
    assert build_module._module_hash(replay_mod) in build_module._replay_stage().code_version
    assert build_module._module_hash(fidelity) in build_module._replay_stage().code_version
    grown = build_module._state_stage({"users": 10}, 0).code_version
    assert build_module._module_hash(compile_env) in grown and build_module._module_hash(synth) in grown
    assert grown != build_module._state_stage({"users": 20}, 0).code_version
    assert grown == build_module._state_stage({"users": 10}, 0).code_version


def test_the_replay_stage_key_moves_when_the_judge_that_settles_a_column_moves(tmp_path):
    """D219: a replay scored with no judge and one scored with a judge are two readings of the same
    bytes. The judge's identity and the table's version ride in the stage key, so a cache written
    under the first is never handed back for the second."""
    class _Model:
        name = "invented/judge-one"

    class _Other:
        name = "invented/judge-two"

    unjudged = build_module._replay_stage(build_module.SemanticJudging(tmp_path)).code_version
    judged = build_module._replay_stage(build_module.SemanticJudging(tmp_path, _Model())).code_version
    other = build_module._replay_stage(build_module.SemanticJudging(tmp_path, _Other())).code_version
    assert unjudged != judged != other and unjudged != other
    assert "judge=none" in unjudged and "invented/judge-one" in judged
    again = build_module._replay_stage(build_module.SemanticJudging(tmp_path, _Model())).code_version
    assert again == judged, "the same judge and the same table key the same stage"


def test_the_replay_stage_hands_its_comparer_a_judge_rather_than_defaulting_it_away(tmp_path):
    """The wiring itself, which is what nothing asserted before: the object that reaches the Runner
    knows how to settle a semantic column, or the build says out loud that it has no judge."""
    class _Model:
        name = "invented/judge-one"

    judging = build_module.SemanticJudging(tmp_path, _Model())
    assert judging.judge is not None and judging.identity.startswith("invented/judge-one")
    assert build_module.SemanticJudging(tmp_path).judge is None
    comparer = tool_runs.ReplayComparer(judge=judging.judge, equivalence=judging.table)
    assert comparer.judge is not None and comparer.equivalence is judging.table


def test_tool_definitions_speak_json_schema():
    from kullback.runner.records import ToolSig
    sig = ToolSig(name="find_user_id_by_email", kind="read",
                  args_schema={"properties": {"email": {"type": ["str"]}, "n": {"type": ["int", "NoneType"]}},
                               "required": ["email"], "type": "object"})
    [definition] = build_module._tool_definitions([sig])
    assert definition["name"] == "find_user_id_by_email"
    assert definition["parameters"]["properties"]["email"]["type"] == "string"
    assert definition["parameters"]["properties"]["n"]["type"] == ["integer", "null"]
    assert definition["parameters"]["required"] == ["email"]


def test_tool_definitions_show_the_model_the_values_the_corpus_showed_for_an_argument():
    """Build 8: the re-roll model dropped the '#' from 132 order ids over a schema that showed it nothing."""
    from kullback.builder.vocabulary import FieldSpec, Vocabulary
    from kullback.runner.records import ToolSig
    sig = ToolSig(name="get_order_details", kind="read",
                  args_schema={"properties": {"order_id": {"type": ["str"]}}, "required": ["order_id"],
                               "type": "object"})
    vocab = Vocabulary(domain="retail", fields=[
        FieldSpec(field="order_id", kind="reference", examples=["#W5056519", "#W8277957"])])
    [definition] = build_module._tool_definitions([sig], vocab)
    assert definition["parameters"]["properties"]["order_id"]["description"] == "for example #W5056519, #W8277957"
    [plain] = build_module._tool_definitions([sig])
    assert "description" not in plain["parameters"]["properties"]["order_id"]


def test_a_stage_drops_the_run_files_an_earlier_build_left_under_its_name(tmp_path):
    """Build 8: 111 re-roll files from a build four days older sat beside its own, and runs.json counted them."""
    run_dir = tmp_path / "runs" / "t1"
    run_dir.mkdir(parents=True)
    for name in ("reroll-t1-0.jsonl", "reroll-t1-1.jsonl", "replay-abc.jsonl", "reroll-t10-0.jsonl"):
        (run_dir / name).write_text("{}\n")
    assert build_module._discard_runs(run_dir, "reroll-t1-") == 2
    assert sorted(p.name for p in run_dir.iterdir()) == ["replay-abc.jsonl", "reroll-t10-0.jsonl"]
    assert build_module._discard_runs(tmp_path / "runs" / "missing", "reroll-") == 0


# --- D174: a recompile keeps the better body ---


def test_a_recompile_that_scores_no_higher_than_the_kept_body_leaves_it_in_place_and_says_so(built, tmp_path):
    """One live build's third round recompiled a write tool from 65 percent of its recorded calls
    matched to none of them, and lost 41 Tasks of fidelity in the round: the narrowed rerun took
    whatever its attempts produced as the tool's body. A recompile is an attempt at a better body;
    it replaces the kept one only by beating it on the compiler's own key."""
    workdir = tmp_path / "declined"
    shutil.copytree(built, workdir)
    name = sorted(json.loads((workdir / "bodies.json").read_text(encoding="utf-8")))[0]
    good = json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))[name]
    plan = BuildPlan(workdir=workdir, iterate=True, model=Bodies(), max_attempts=0)
    build_module.execute(plan, "compile_tools", tools=[name])  # writes the kept body's score
    row = json.loads((workdir / "tool_builds.json").read_text(encoding="utf-8"))[name]
    assert isinstance(row.get("score"), list) and len(row["score"]) == 2

    from kullback.builder import memory

    # A repair records the lesson it acts on, which is what makes the narrowed rerun recompile
    # rather than answer from the cache; the attempt it then makes crashes on every call.
    memory.record_lesson(workdir, name, ["replay_fidelity: one recorded call differs"])
    worse = TestModel(["raise KeyError('no such row')"], loop=True)
    result = build_module.execute(BuildPlan(workdir=workdir, iterate=True, model=worse, max_attempts=0),
                                  "compile_tools", tools=[name])
    assert result.status == "complete" and result.reports["compile_tools"].cached is False
    assert json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))[name] == good, \
        "the body that crashes on every call must not replace the one that replayed"
    row = json.loads((workdir / "tool_builds.json").read_text(encoding="utf-8"))[name]
    assert row["recompile_declined"]["kept_score"] == row["score"]
    assert row["recompile_declined"]["attempt_score"] < row["score"]
    assert row.get("assisted") is False, "the kept body's assisted ruling stands with it"
    light = next((r for r in builder_tools.red_lights(workdir)
                  if r.target == name and r.stage == "compile_tools"), None)
    assert light is None, "a body that passed every gate is no red light, declined attempt or not"


def test_a_tool_no_recompile_has_moved_twice_over_is_named_stalled_on_its_red_light(tmp_path):
    """The red lights are what the Builder reads before it chooses a verb, so a tool it has already
    spent two rounds on has to read differently there from one it has not tried."""
    workdir = tmp_path / "stalled_light"
    workdir.mkdir()
    (workdir / "tool_builds.json").write_text(json.dumps({
        "lookup_shelf": {"assisted": True, "score": [5, 9]}}), encoding="utf-8")
    (workdir / "kept_bodies.json").write_text(json.dumps({
        "lookup_shelf": {"outcome": "kept", "attempt_score": [5, 9], "kept_score": [5, 9],
                         "unbeaten": 3}}), encoding="utf-8")

    light = next(r for r in builder_tools.red_lights(workdir) if r.target == "lookup_shelf")

    assert "stalled: 3 recompiles in a row scored no higher" in light.failure


def test_a_declined_recompile_is_named_on_the_assisted_tools_red_light(tmp_path):
    workdir = tmp_path / "lights"
    workdir.mkdir()
    (workdir / "tool_builds.json").write_text(json.dumps({
        "lookup_shelf": {"assisted": True, "score": [5, 9],
                         "recompile_declined": {"attempt_score": [2, 0], "kept_score": [5, 9]}}}),
        encoding="utf-8")
    light = next(r for r in builder_tools.red_lights(workdir) if r.target == "lookup_shelf")
    assert "scored [2, 0] against the kept body's [5, 9]" in light.failure and "declined" in light.failure


# --- a body rewritten because the stage's inputs changed replaces the kept one only by beating it ---


def _rerun_the_whole_stage(workdir: Path, model) -> None:
    """Run compile_tools over every tool again, the way a change to one of its inputs makes it run.

    A change to the schema, the Starting state, the readers or the tables block moves the stage's
    key and the run that follows writes every body from scratch. The replay-evidence file is a
    declared input of the stage and each tool's own slice sits in that tool's compile hash, so
    bumping every slice asks for the same full run without a fixture having to fake a new world.
    world_provenance.json cannot carry the bust: starting_state lists bodies.json as an input
    path, so a test that plants a body reruns it and rewrites that file.
    """
    bodies = json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))
    path = workdir / "replay_evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(evidence, dict):
        evidence = {}
    n = int(evidence.get("_compile_bust") or 0) + 1
    evidence["_compile_bust"] = n
    for name in bodies:
        slice_ = evidence.get(name)
        evidence[name] = dict(slice_, compile_bust=n) if isinstance(slice_, dict) else {
            "ids": slice_, "compile_bust": n}
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    plan = BuildPlan(workdir=workdir, iterate=True, model=model, max_attempts=0)
    result = build_module.execute(plan, "compile_tools")
    assert result.reports["compile_tools"].cached is False, "an input moved, so the stage has to run"


def _busiest_tool(workdir: Path) -> str:
    """The tool the recordings call most, so a body planted for it is judged on real evidence."""
    outcomes = json.loads((workdir / "tool_call_outcomes.json").read_text(encoding="utf-8"))
    return max(sorted(outcomes), key=lambda tool: len(outcomes[tool]))


def _plant_body(workdir: Path, name: str, body: str) -> dict:
    """Put `body` in the workdir as the body this tool already has; answers what was there before."""
    path = workdir / "bodies.json"
    before = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(dict(before, **{name: body})), encoding="utf-8")
    return before


def test_a_body_rewritten_because_the_stages_inputs_moved_does_not_replace_one_it_scores_below(
    built, tmp_path
):
    """A live round changed one constant of the world, which rewrote every body from scratch, and a
    write tool nothing about had changed went from 83 of its 139 recorded calls matched to 39; the
    corpus fell from 179 confirmed Tasks to 165. D174 made a narrowed recompile replace a kept body
    only by beating it and this path had no such rule: the rewrite was kept whatever it scored."""
    workdir = tmp_path / "scores_below"
    shutil.copytree(built, workdir)
    before = json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))

    _rerun_the_whole_stage(workdir, TestModel(["raise KeyError('no such row')"], loop=True))

    assert json.loads((workdir / "bodies.json").read_text(encoding="utf-8")) == before, \
        "every body that crashes on every call scored below the one the workdir held"
    rulings = json.loads((workdir / "kept_bodies.json").read_text(encoding="utf-8"))
    assert rulings, "the run has to say per tool what it did with the body that tool already had"
    busiest = rulings[_busiest_tool(workdir)]
    assert busiest["outcome"] == "kept" and busiest["kept_score"] > busiest["attempt_score"]


def test_the_gates_a_run_records_are_the_kept_bodys_and_not_the_losing_attempts(built, tmp_path):
    """gates.json is read as the ruling on the module the stage released, and `red_lights` turns every
    failing row in it into work for the Builder. A run that keeps the body it already had must not
    leave the failures of the attempt that lost to it there: the Builder would repair a body no
    module holds. What the attempt scored is on the tool's row and in the run's own ruling."""
    workdir = tmp_path / "gates_of_the_kept"
    shutil.copytree(built, workdir)
    crash = "this attempt reaches for a row that is not there"

    _rerun_the_whole_stage(workdir, TestModel([f"raise KeyError({crash!r})"], loop=True))

    rulings = json.loads((workdir / "kept_bodies.json").read_text(encoding="utf-8"))
    assert rulings and all(r["outcome"] == "kept" for r in rulings.values())
    rows = json.loads((workdir / "gates.json").read_text(encoding="utf-8"))
    assert rows, "the run still records what it ruled on the bodies it released"
    failed = [row for row in rows if not row.get("pass")]
    assert not [row for row in failed if row.get("stage") == "executes_on_s0"], \
        "the attempt crashed on every call and the bodies released did not"
    assert all(crash not in json.dumps(row) for row in failed), \
        "nothing the attempt raised reaches the file the Builder repairs from"


def test_a_body_rewritten_because_the_stages_inputs_moved_replaces_one_it_scores_above(built, tmp_path):
    """The rule is not that the older body wins; it is that the better one does. A body that matches
    more recorded calls than the one the workdir held takes the tool, and the ruling says so."""
    workdir = tmp_path / "scores_above"
    shutil.copytree(built, workdir)
    name = _busiest_tool(workdir)
    before = _plant_body(workdir, name, "return None")  # runs on every call and matches none of them

    _rerun_the_whole_stage(workdir, Bodies())

    written = json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))
    assert written[name] == before[name], "the body that answers the recordings replaces the one that does not"
    ruling = json.loads((workdir / "kept_bodies.json").read_text(encoding="utf-8"))[name]
    assert ruling["outcome"] == "beaten" and ruling["attempt_score"] > ruling["kept_score"]


def test_a_rewritten_body_that_only_ties_leaves_the_body_the_workdir_already_had(built, tmp_path):
    """A tie goes to the body that is already there. It has been through the Examiner's round and the
    Tasks that were trusted were trusted against it, and an equally scored replacement buys nothing."""
    workdir = tmp_path / "tie"
    shutil.copytree(built, workdir)
    name = _busiest_tool(workdir)
    before = json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))
    variant = before[name] + "\n# the same body again, written another way"
    _plant_body(workdir, name, variant)

    _rerun_the_whole_stage(workdir, Bodies())  # writes exactly the body `before` holds

    assert json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))[name] == variant
    ruling = json.loads((workdir / "kept_bodies.json").read_text(encoding="utf-8"))[name]
    assert ruling["outcome"] == "kept" and ruling["kept_score"] == ruling["attempt_score"]


def test_a_kept_body_that_raises_on_every_call_under_the_new_world_is_reported_and_gives_up_the_tool(
    built, tmp_path
):
    """A schema change can take away the table a body read, and then it answers nothing at all. Such a
    body is no candidate whatever it scores, or the tie would hand the tool back to a body the change
    broke; it is named `could_not_run` and the run's own attempt takes the tool."""
    workdir = tmp_path / "could_not_run"
    shutil.copytree(built, workdir)
    name = _busiest_tool(workdir)
    _plant_body(workdir, name, "raise KeyError('the table this body read is gone')")
    attempt = "raise ValueError('this attempt raises on every call too')"

    _rerun_the_whole_stage(workdir, TestModel([attempt], loop=True))

    ruling = json.loads((workdir / "kept_bodies.json").read_text(encoding="utf-8"))[name]
    assert ruling["outcome"] == "could_not_run"
    assert ruling["kept_score"] == ruling["attempt_score"], "the two tie, and the tie does not save it"
    written = json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))[name]
    assert attempt in written and "the table this body read is gone" not in written


def test_the_writer_is_told_what_the_body_it_has_to_beat_fails_at_and_is_never_shown_that_body(
    built, tmp_path
):
    """Shown the body, the writer copies it, and a copy cannot beat it. What it is shown is what that
    body fails at, grouped by shape with counts the way every other hint is (D181's rule 6)."""
    from kullback.builder import compile_env

    workdir = tmp_path / "hint"
    shutil.copytree(built, workdir)
    name = _busiest_tool(workdir)
    planted = "raise KeyError('the table this body read is gone')"
    _plant_body(workdir, name, planted)
    model = Bodies()

    _rerun_the_whole_stage(workdir, model)

    sent = [" ".join(str(m.get("content") or "") for m in call["messages"]) for call in model.calls]
    asked = [text for text in sent if f"Tool: {name}" in text]
    assert asked, "the stage has to have asked for this tool's body"
    assert all(planted not in text for text in asked), "the body to beat is never in the prompt"
    hints = [text[text.index(compile_env.KEPT_BODY_HEAD):] for text in asked
             if compile_env.KEPT_BODY_HEAD in text]
    assert hints, "what that body fails at is what the writer is given"
    assert all("- gate " in hint and "(1 call" in hint for hint in hints), \
        "and it is given by gate, grouped by shape with the calls counted"


# --- D191: one scorer for a narrowed repair, and the replay's own failures as evidence ---


def _kept_row(workdir: Path, name: str) -> dict:
    return json.loads((workdir / "kept_bodies.json").read_text(encoding="utf-8"))[name]


def _recompile_one(workdir: Path, name: str, model, lesson: str) -> None:
    """One narrowed recompile of one tool, the way `repair_recompile` runs it.

    The lesson is what the repair verb records before the stage runs, and it is a declared input
    path of the narrowed stage, so recording one is what makes the rerun actually recompile rather
    than answer from the cache.
    """
    from kullback.builder import memory

    memory.record_lesson(workdir, name, [lesson])
    result = build_module.execute(BuildPlan(workdir=workdir, iterate=True, model=model, max_attempts=0),
                                  "compile_tools", tools=[name])
    assert result.reports["compile_tools"].cached is False


def test_a_narrowed_recompile_scores_a_tool_the_way_the_full_pass_scores_it(built, tmp_path):
    """The Builder reads a repair's ruling and decides what to spend the next round on, so what a
    narrowed recompile calls a score has to be the number the full pass will re-derive. Four live
    rounds of one build repaired the same four tools on a local reading and lost every one of them
    again on the next full pass."""
    workdir = tmp_path / "one_scorer"
    shutil.copytree(built, workdir)
    name = _busiest_tool(workdir)

    _recompile_one(workdir, name, Bodies(), "replay_fidelity: one recorded call differs")
    narrowed = _kept_row(workdir, name)
    _rerun_the_whole_stage(workdir, Bodies())
    full = _kept_row(workdir, name)

    assert narrowed["kept_score"] == full["kept_score"], \
        "the body that already exists is graded under one world by one function, however narrow the run"
    assert narrowed["attempt_score"] == full["attempt_score"]
    assert narrowed["evidence_calls"] == full["evidence_calls"], \
        "and against the same recorded calls, which is what makes the two scores comparable at all"


def test_a_tool_two_recompiles_in_a_row_bought_nothing_on_is_marked_stalled_where_the_builder_reads_it(
    built, tmp_path
):
    """An attempt that scores no higher leaves the tool exactly as it was, and the ruling for it read
    the same as the first attempt's. One live build spent nine rounds on four tools that way."""
    workdir = tmp_path / "stalled"
    shutil.copytree(built, workdir)
    name = _busiest_tool(workdir)
    worse = TestModel(["return None"], loop=True)  # runs on every call and matches almost none

    _recompile_one(workdir, name, worse, "the first hint")
    first = _kept_row(workdir, name)["unbeaten"]
    assert first >= 1, "a recompile that scored no higher counts against the tool"

    _recompile_one(workdir, name, worse, "the second hint, near enough the first")

    second = _kept_row(workdir, name)["unbeaten"]
    assert second == first + 1, "and each one after it counts once more"
    assert (f"stalled: {second} recompiles in a row scored no higher"
            in repair_module.recompile_ruling(workdir, name))


def test_a_recompile_that_beats_the_body_it_had_clears_the_stalled_count(built, tmp_path):
    workdir = tmp_path / "unstalled"
    shutil.copytree(built, workdir)
    name = _busiest_tool(workdir)
    good = json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))[name]
    _plant_body(workdir, name, "return None")

    _recompile_one(workdir, name, TestModel(["return None"], loop=True), "a hint that changes nothing")
    assert _kept_row(workdir, name)["unbeaten"] >= 1
    _recompile_one(workdir, name, Bodies(), "a hint that lands")

    assert json.loads((workdir / "bodies.json").read_text(encoding="utf-8"))[name] == good
    assert _kept_row(workdir, name)["unbeaten"] == 0, "a repair that landed is not a stall"


def _replay_call(name, call_id, args=None):
    from conftest import PTR
    from kullback.runner.records import ToolCall
    return ToolCall(name=name, args=args or {}, id=call_id, raw_ptr=PTR)


def _kennel_traces():
    """One Run the Builder may learn from and one held out as the anchor, over an invented domain.

    The seed Run books a stay and then reads the row it just changed, which is the D74 skip; the
    held-out Run only reads, which is the D81 hold-out.
    """
    seed = _trace("run_seed", [
        _replay_call("book_kennel_stay", "c_write", {"kennel_id": "k1"}),
        _replay_call("read_kennel_row", "c_after_write", {"kennel_id": "k1"}),
        _replay_call("read_kennel_row", "c_plain", {"kennel_id": "k2"}),
    ])
    anchor = _trace("run_anchor", [_replay_call("read_kennel_row", "c_anchor", {"kennel_id": "k3"})])
    return seed, anchor


def _kennel_evidence(replay_failed=None):
    seed, anchor = _kennel_traces()
    traces = [seed, anchor]
    after_write = build_module.after_write_calls(traces, {"book_kennel_stay"})
    callers = {"book_kennel_stay": {"assistant"}, "read_kennel_row": {"assistant"}}
    return build_module.evidence_calls(traces, {"run_seed"}, after_write, callers, replay_failed)


def test_the_three_filters_still_drop_every_call_no_replay_failure_names():
    calls, skipped, from_replay = _kennel_evidence()
    assert [c.id for c in calls["read_kennel_row"]] == ["c_plain"]
    assert skipped == {"read_kennel_row": 1} and from_replay == {}


def test_a_call_the_reference_replay_failed_on_is_evidence_whatever_a_filter_dropped_it_for():
    """The filters answer what a body may be written from. Whether a Task confirms is a different
    question, and a call the replay of the References failed on is the whole of that question: the
    recording made it, so a Run of that Task cannot confirm until the body answers it."""
    failed = {"read_kennel_row": {"c_after_write": "differs: status: ours \"open\", recorded \"held\""}}
    calls, skipped, from_replay = _kennel_evidence(failed)

    assert sorted(c.id for c in calls["read_kennel_row"]) == ["c_after_write", "c_plain"]
    assert from_replay == {"read_kennel_row": 1}, "and the ruling says how many are there for that reason"
    assert skipped == {}, "a call put back is not also counted as skipped"


def test_a_replay_failure_in_a_held_out_run_is_not_put_back_as_evidence():
    """The held-out split is not one of the filters a replay failure overrides: that Run's failure
    is the corpus fidelity number, and its arguments and result are not the writer's to see."""
    failed = {"read_kennel_row": {"c_after_write": "differs", "c_anchor": "differs"}}
    calls, _skipped, from_replay = _kennel_evidence(failed)

    assert "c_anchor" not in {c.id for c in calls["read_kennel_row"]}
    assert from_replay == {"read_kennel_row": 1}, "only the seed Run's call was put back"


def test_a_call_the_replay_agreed_on_is_left_where_its_filter_put_it(tmp_path):
    replays = {"task_a": {"run_seed": {"confirmed": False, "checks": [
        {"tool": "read_kennel_row", "call_id": "c_after_write", "verdict": "same"},
        {"tool": "read_kennel_row", "call_id": "c_anchor", "verdict": "both_refused"}]}}}
    (tmp_path / "replays.json").write_text(json.dumps(replays), encoding="utf-8")
    assert build_module.replay_failures(tmp_path) == {}


def test_a_confirmed_task_leaves_no_failing_call_and_a_failing_one_names_its_leaf(tmp_path):
    replays = {
        "task_a": {"run_1": {"confirmed": True, "checks": [
            {"tool": "read_kennel_row", "call_id": "c_ok", "verdict": "differs"}]}},
        "task_b": {"run_2": {"confirmed": False, "checks": [
            {"tool": "read_kennel_row", "call_id": "c_bad",
             "verdict": "differs", "difference": {"leaf": 'status: ours "open", recorded "held"'}},
            {"tool": "book_kennel_stay", "call_id": "c_refused", "verdict": "theirs_refused",
             "difference": {"keys_changed": ["nights"], "keys_only_theirs": ["rate"]}}]}}}
    (tmp_path / "replays.json").write_text(json.dumps(replays), encoding="utf-8")

    failed = build_module.replay_failures(tmp_path)

    assert failed == {
        "read_kennel_row": {"c_bad": 'differs: status: ours "open", recorded "held"'},
        "book_kennel_stay": {"c_refused": "theirs_refused: columns that differ: nights; "
                                          "columns the recording has and the body does not: rate"}}


def test_a_workdir_that_has_not_replayed_yet_has_no_replay_failures_to_read(tmp_path):
    assert build_module.replay_failures(tmp_path) == {}


def test_the_lesson_names_the_column_the_replay_parted_on_and_counts_the_calls_that_parted_alike():
    failed = {"c1": 'differs: status: ours "open", recorded "held"',
              "c2": 'differs: status: ours "open", recorded "booked"',
              "c3": "differs: columns that differ: nights"}

    lesson = build_module.replay_lesson(failed, ["c1", "c2", "c3"])

    assert build_module.REPLAY_LESSON_HEAD in lesson
    assert 'status: ours "open", recorded "held" (2 calls)' in lesson, \
        "two calls that parted on the same column in the same way are one shape with a count"
    assert "columns that differ: nights (1 call)" in lesson


def test_a_replay_failing_call_the_writer_is_not_shown_is_counted_and_never_quoted():
    """The masking is the one every other hint already applies: a call held out of the split, or a
    Run held out as the anchor, is a call whose recorded answer the writer must not read."""
    failed = {"c_shown": 'differs: status: ours "open", recorded "held"',
              "c_hidden": 'differs: rate: ours 40, recorded 65'}

    lesson = build_module.replay_lesson(failed, ["c_shown"])

    assert "recorded 65" not in lesson and "rate" not in lesson
    assert "1 more on calls you were not shown" in lesson


def test_a_tool_the_replay_failed_on_nothing_for_gets_no_such_lesson():
    assert build_module.replay_lesson({}, ["c1"]) == ""


def test_a_written_call_path_becomes_a_trace_the_replay_can_drive():
    """D199: the conversation is the Run's own, turn for turn, and each call joins the turn it names,
    so a rewrite varies the calls and neither what the agent asked nor what it told the user."""
    spoken = [{"role": "user", "content": "please stock shelf s1"},
              {"role": "assistant", "content": "let me look"},
              {"role": "assistant", "content": "shelf s1 now holds 4."},
              {"role": "user", "content": "and check it for me"}]
    calls = [{"name": "read_shelf", "args": {"shelf_id": "s1"}, "id": "c0", "turn": 1},
             {"name": "stock_shelf", "args": {"shelf_id": "s1", "units": 4}, "id": "c1", "turn": 1},
             {"name": "read_shelf", "args": {"shelf_id": "s1"}, "id": "c2", "requestor": "user", "turn": 3}]

    trace = build_module.call_trace("t1", calls, spoken, "synth-path-r1-t1-1")

    assert [(turn.role, turn.content, turn.tool_call_ids) for turn in trace.turns] == [
        ("user", "please stock shelf s1", []), ("assistant", "let me look", ["c0", "c1"]),
        ("assistant", "shelf s1 now holds 4.", []), ("user", "and check it for me", ["c2"])]
    assert [(c.id, c.name, c.requestor) for c in trace.tool_calls] == [
        ("c0", "read_shelf", "assistant"), ("c1", "stock_shelf", "assistant"),
        ("c2", "read_shelf", "user")]


def test_a_call_whose_turn_is_gone_joins_the_last_turn_its_speaker_had():
    spoken = [{"role": "assistant", "content": "done"}]
    trace = build_module.call_trace("t1", [{"name": "read_shelf", "args": {}, "turn": 9}], spoken, "synth-1")
    assert [call.id for call in trace.tool_calls] == ["synth-1-0"]
    assert [(turn.role, turn.tool_call_ids) for turn in trace.turns] == [("assistant", ["synth-1-0"])]


def test_a_narrowed_recompile_plus_full_rebuild_does_not_recompile_a_sibling(built, tmp_path):
    """A lesson recorded for one tool used to miss the un-narrowed compile_tools stage for every
    other tool, so D161's rebuild spent a model call on bodies whose evidence had not moved."""
    workdir = tmp_path / "sibling"
    shutil.copytree(built, workdir)
    names = sorted(json.loads((workdir / "bodies.json").read_text(encoding="utf-8")))
    assert len(names) >= 2
    target, sibling = names[0], names[1]
    _recompile_one(workdir, target, Bodies(), "a hint for one tool only")

    model = Bodies()
    result = build_module.execute(BuildPlan(workdir=workdir, iterate=True, model=model, max_attempts=0),
                                  "compile_tools")
    assert result.reports["compile_tools"].cached is False
    sent = [" ".join(str(m.get("content") or "") for m in call["messages"]) for call in model.calls]
    assert not [text for text in sent if f"Tool: {sibling}" in text], \
        "a sibling whose evidence bytes did not change is not sent to the compiler"


def test_a_tools_compile_hash_moves_when_the_compiler_version_moves(tmp_path):
    """A sandbox or gate fix misses the stage cache; reuse must not then keep a body scored under
    the old gates just because this tool's lessons and world bytes did not move."""
    (tmp_path / "world_provenance.json").write_text("{}", encoding="utf-8")
    inputs = {"schema": {}, "db": {}}
    before = build_module._tool_compile_hash(tmp_path, "quote_haulage", inputs, "compile_tools:a")
    after = build_module._tool_compile_hash(tmp_path, "quote_haulage", inputs, "compile_tools:b")
    assert before != after
