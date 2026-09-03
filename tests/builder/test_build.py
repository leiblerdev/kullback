"""build.py is the one wiring cli build and cli run go through; this runs it over the fixture with no live model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kullback.ai.provider import TestModel
from kullback.builder import build as build_module
from kullback.builder import pipeline
from kullback.builder.build import BuildPlan
from kullback.runner.records import Verifier
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
    and no code change, every stage but the two that first ran before the anchor existed comes
    back from the cache. mine and cluster ran on the first build with no anchor.json on disk, and
    the anchor is part of every stage's key (D81), so those two are the ones that run once more.

    The rebuild goes into a copy of the built workdir, so the module's shared fixture is left as
    the first build wrote it and the tests that read it do not depend on running after this one."""
    workdir = tmp_path / "again"
    shutil.copytree(built, workdir)
    result = build_module.build(workdir, iterate=True, model=Bodies())
    assert result["status"] == "complete"
    statuses = json.loads((workdir / "pipeline" / "state.json").read_text(encoding="utf-8"))["statuses"]
    # ingest is the previous build's own record, carried over because this build had no file to
    # ingest and so ran no ingest stage at all.
    assert {name for name, status in statuses.items() if status != "cached"} == {"ingest", "mine", "cluster"}
    assert statuses["ingest"] == "ran"


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
    from kullback.builder import cluster, compile_env, mine, synth, user_sim
    from kullback.gates import fidelity
    from kullback.runner import replay as replay_mod
    assert build_module._mine_stage().code_version.endswith(build_module._module_hash(mine))
    assert build_module._module_hash(cluster) in build_module._cluster_stage().code_version
    assert build_module._module_hash(user_sim) in build_module._user_rules_stage().code_version
    assert build_module._module_hash(replay_mod) in build_module._replay_stage().code_version
    assert build_module._module_hash(fidelity) in build_module._replay_stage().code_version
    grown = build_module._state_stage({"users": 10}, 0).code_version
    assert build_module._module_hash(compile_env) in grown and build_module._module_hash(synth) in grown
    assert grown != build_module._state_stage({"users": 20}, 0).code_version
    assert grown == build_module._state_stage({"users": 10}, 0).code_version


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
