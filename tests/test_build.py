"""build.py is the one wiring cli build and cli run go through; this runs it over the fixture with no live model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.builder import build as build_module
from harness.shared.provider import TestModel
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
                 "task_status.json", "scorecard.json", "bodies.json"):
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


def test_a_second_build_is_served_from_the_cache(built):
    """The stage cache is what makes `build --iterate` cheap (design section 8)."""
    result = build_module.build(built, iterate=True, model=Bodies())
    assert result["status"] == "complete"
    statuses = json.loads((built / "pipeline" / "state.json").read_text(encoding="utf-8"))["statuses"]
    assert "cached" in set(statuses.values())


def test_pipeline_state_on_disk_covers_both_pipeline_runs(tmp_path_factory, request):
    """build() runs two Pipelines (ingest/mine/cluster, then the Builder stages) that each write
    pipeline/state.json to the same fixed path; the file on disk must hold both runs' statuses and
    gates, not just whichever pipeline wrote last. A fresh workdir and a single build() call, so an
    unrelated later rebuild's own overwrite of the file cannot mask the bug this guards against."""
    workdir = tmp_path_factory.mktemp("pipeline-state")
    fixture = Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"
    build_module.build(workdir, model=Bodies(), files=[fixture], max_attempts=0)
    state = json.loads((workdir / "pipeline" / "state.json").read_text(encoding="utf-8"))
    for name in ("ingest", "mine", "cluster"):
        assert name in state["statuses"], name
    gate_stages = {g.get("stage") for g in state["gates"]}
    assert "ingest" in gate_stages


def test_wrap_sets_a_prompt_cache_key_scoped_to_the_build_and_stage(tmp_path):
    """docs/prompt-caching.md item 4: one short string per build and stage."""
    model = build_module._wrap(TestModel(["hi"]), "compile_tools", tmp_path, None, model_id="anthropic/claude-opus-5")
    build_id = build_module._build_id(tmp_path)
    assert model.prompt_cache_key == f"kullback-{build_id}-compile_tools"


def test_wrap_gives_two_stages_of_the_same_build_different_cache_keys(tmp_path):
    a = build_module._wrap(TestModel(["hi"]), "compile_tools", tmp_path, None, model_id="anthropic/claude-opus-5")
    b = build_module._wrap(TestModel(["hi"]), "compile_policy", tmp_path, None, model_id="anthropic/claude-opus-5")
    assert a.prompt_cache_key != b.prompt_cache_key


def test_wrap_memoizes_a_builder_stage_but_not_the_candidate(tmp_path):
    """docs/prompt-caching.md item 3: run_batch's candidate model must stay a fresh sample."""
    from harness.shared.provider import MemoModel

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
    from harness.shared import budget

    ceiling = budget.Ceiling(usd=10.0)
    with pytest.raises(budget.UnpricedModel):
        build_module._wrap(TestModel(["hi"]), "compile_tools", tmp_path, ceiling, model_id="openai/mystery")


def test_environment_gate_is_wired_with_referenced_ids_and_synthetic_rows(tmp_path_factory, request, monkeypatch):
    """Without wiring, both the referenced-ids and synthetic-rows halves of the build_environment
    gate default to empty and never fire."""
    seen: dict = {}
    original = build_module.validate.environment_gate

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(build_module.validate, "environment_gate", spy)
    workdir = tmp_path_factory.mktemp("env-gate")
    fixture = Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"
    build_module.build(workdir, model=Bodies(), files=[fixture], max_attempts=0)
    assert "referenced_ids" in seen and "synthetic_rows" in seen
    assert seen["referenced_ids"], "referenced_ids was not wired into the gate call"


def test_budget_json_is_written_because_every_model_goes_through_budget_py(built):
    """D65 and D86 only bind if the Builder's model is wrapped; unwrapped, nothing records a call."""
    totals = json.loads((built / "budget.json").read_text(encoding="utf-8"))
    assert totals["total"]["calls"] > 0
    assert "compile_tools" in totals["stages"]


def test_run_batch_writes_one_jsonl_per_candidate_run(built):
    """cli run's whole job: a Candidate over the built Environment, one Run file each (D49, D74)."""
    from harness.shared.provider import TestModel

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
    """`build --iterate` with no new file runs no ingest stage, so the previous ingest's status and
    gate would drop out of pipeline/state.json; report.py reads that file once, so it is carried."""
    workdir = tmp_path_factory.mktemp("iterate-state")
    fixture = Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"
    build_module.build(workdir, model=Bodies(), files=[fixture], max_attempts=0)
    build_module.build(workdir, iterate=True, model=Bodies(), max_attempts=0)
    state = json.loads((workdir / "pipeline" / "state.json").read_text(encoding="utf-8"))
    assert "ingest" in state["statuses"]
    assert "ingest" in {g.get("stage") for g in state["gates"]}


def test_a_stage_that_delegates_to_a_module_carries_that_module_in_its_code_version():
    """R42: `compile_policy:{model}` and `derive_verifier:1` never changed when policy.py or
    verifier.py did, so an edited compiler was served its stale cache entry."""
    from harness.builder import memory, policy
    from harness.builder import verifier as verifier_mod
    versions = {name: build_module._module_hash(mod)
                for name, mod in (("policy", policy), ("memory", memory), ("verifier", verifier_mod))}
    assert len(set(versions.values())) == 3
    assert build_module._policy_stage(None).code_version.endswith(versions["policy"])
    assert versions["verifier"] in build_module._verifier_stage().code_version


def test_the_compile_tools_cache_moves_when_the_modules_it_delegates_to_change(monkeypatch, tmp_path):
    """R42, for the stage that has the most of its work somewhere else. Without this, a fix to
    sandbox.py left every body it had already refused sitting in the cache for --iterate to reuse."""
    from harness.builder import build as bd

    first = bd._tools_stage(None, 3).code_version
    monkeypatch.setattr(bd.sandbox, "__file__", str(tmp_path / "different.py"))
    (tmp_path / "different.py").write_text("# not the sandbox we hashed a moment ago\n")
    assert bd._tools_stage(None, 3).code_version != first


# --- what a tool is gated on (D74) ---

def _trace(trace_id, calls):
    from conftest import PTR
    from harness.shared.records import Trace
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="1", source="tau2", tool_calls=calls, raw_ptr=PTR)


def _tc(name, args, error=None):
    from conftest import PTR
    from harness.shared.records import ToolCall
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
    from harness.shared.records import ToolCallError
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
    from harness.shared.provider import TestModel

    task_id = sorted(p.stem for p in (built / "tasks").glob("*.json"))[0]
    candidate = TestModel([{"content": "done ###STOP###"}], loop=True)
    out = build_module.run_batch(built, task_id, candidate, count=1)
    first = candidate.calls[0]
    roles = [m["role"] for m in first["messages"]]
    assert roles[0] == "system" and "user" in roles
    assert first["tools"] and {t["name"] for t in first["tools"]} >= {"get_order_details"}
    events = [json.loads(x) for x in Path(out["runs"][0]).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert events[0]["type"] == "user_turn" and events[0]["payload"]["text"]
