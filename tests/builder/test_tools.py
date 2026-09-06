"""The Builder's tools: `status` over the records the build left, and the repair verbs that act (D135, D138)."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from session_fixtures import fixture_path
from test_build import Bodies

from kullback.builder import agent as builder_agent
from kullback.builder import repair as repair_module
from kullback.builder import tools as builder_tools
from kullback.builder.build import BuildPlan

ASSISTED = "list_all_product_types"


@pytest.fixture(scope="module")
def built(tmp_path_factory, request) -> Path:
    """One code-driven build of the fixture, which leaves red lights of every kind to read."""
    workdir = tmp_path_factory.mktemp("tools")
    builder_agent.run_builder(workdir, model=Bodies(), files=[fixture_path(request)], max_attempts=0)
    return workdir


def _run(tool, arguments):
    return asyncio.run(tool.run(arguments))


def _cached(out, stage: str) -> bool:
    """Whether one stage of a tool result's build came back from the cache."""
    return next(report["cached"] for report in out.details["stages"] if report["name"] == stage)


def _tool(plan: BuildPlan, name: str):
    tools = {t.name: t for t in [*builder_tools.builder_tools(plan), *builder_tools.repair_verb_tools(plan)]}
    return tools[name]


# --- status -------------------------------------------------------------------

def test_status_names_every_failing_gate_with_its_tool_or_task_and_the_verb_that_owns_it(built):
    result = builder_tools.status_of(built)
    assert result.red_lights, "the fixture build leaves failing gates to report"
    by_stage = {light.stage: light for light in result.red_lights}
    assert by_stage["intent"].kind == "task" and by_stage["intent"].target.startswith("task_")
    assert by_stage["intent"].verb == "repair_intent", "an Intent is a line a model wrote, so it is repaired"
    assert by_stage["replay_reference"].kind == "task"
    assert by_stage["replay_reference"].verb == "repair_recompile"
    assisted = next(light for light in result.red_lights if light.target == ASSISTED and "assisted" in light.failure)
    assert assisted.stage == "compile_tools" and assisted.kind == "tool" and assisted.verb == "repair_recompile"
    assert "parses" in result.passing and "intent" in result.failing


def test_status_reads_the_records_and_asks_no_model(built):
    """Every red light comes off gates.json, replays.json and tool_builds.json; nothing else is read."""
    lights = builder_tools.red_lights(built)
    recorded = json.loads((built / "gates.json").read_text(encoding="utf-8"))
    failing = {row["stage"] for row in recorded if not row["pass"]}
    replays = json.loads((built / "replays.json").read_text(encoding="utf-8"))
    unconfirmed = {task for task, rows in replays.items() if not any(r["confirmed"] for r in rows.values())}
    assert failing <= {light.stage for light in lights}
    assert unconfirmed <= {light.target for light in lights if light.stage == "replay_reference"}
    plan = BuildPlan(workdir=built, iterate=True)
    assert plan.model is None, "status takes no model and the plan it reads from need not hold one"
    out = _run(_tool(plan, "status"), {})
    assert not out.is_error and out.content.startswith("status: ")
    assert f"-> {lights[0].verb}" in out.content


def test_status_over_a_workdir_with_nothing_built_says_so(tmp_path):
    result = builder_tools.status_of(tmp_path)
    assert result.red_lights == [] and result.passing == []
    assert "0 red lights" in result.summary
    assert "the gates are green" in builder_tools.render_status(result)


# --- status at the size a live build reaches ------------------------------------

CROWDED_TASKS = [f"task_{i:04x}" for i in range(184)]
CROWDED_TOOLS = [f"tool_{i:02d}" for i in range(30)]
PHRASES = ["the refund", "the exchange", "the open order", "the tracking number"]


def _crowded(workdir: Path) -> Path:
    """A workdir holding the records of a build the size of the live one: 30 tools, 184 Tasks.

    Nothing is mocked: these are `gates.json` and `tool_builds.json` in the shape the gates and the
    compiler write them, which is the only thing `status` ever reads.
    """
    intent = [f"task {t}: noun phrases not evidenced in every Run: {PHRASES[i % 4]} (not in run_a, run_b)"
              if i % 6 == 0 else f"task {t}: noun phrases with no span: {PHRASES[i % 4]}"
              for i, t in enumerate(CROWDED_TASKS)]
    compile_tools = [f"{name}: a recorded success call replays differently; D80 wants 100% on recorded calls"
                     for j, name in enumerate(CROWDED_TOOLS) for _ in range(1 + j % 3)]
    rows = [{"stage": "intent", "pass": False, "failures": intent},
            {"stage": "compile_tools", "pass": False, "failures": compile_tools},
            {"stage": "derive_verifier", "pass": False,
             "failures": [f"task {t}: the D79 suite did not pass" for t in CROWDED_TASKS[:40]]},
            {"stage": "rerolls", "pass": False,
             "failures": ["no re-roll finished: 12 Runs and every one stopped on an error"]}]
    rows += [{"stage": name, "pass": True, "failures": []}
             for name in ("parses", "executes_on_s0", "deterministic", "non_trivial", "confined",
                          "refuses_unknown", "cluster", "vocabulary", "starting_state", "ingest")]
    (workdir / "gates.json").write_text(json.dumps(rows), encoding="utf-8")
    (workdir / "tool_builds.json").write_text(
        json.dumps({name: {"assisted": True} for name in CROWDED_TOOLS[:4]}), encoding="utf-8")
    return workdir


def _flat(text: str) -> str:
    """The rendering with its wrapping undone, so a test can read a logical line as one string."""
    return " ".join(text.split())


def test_status_shows_every_gate_and_every_task_id_and_never_a_truncated_list(tmp_path):
    """A live build rendered 380 red lights as 25 and "355 more, all in details"; the model never
    asked for the details and repaired the four tools it could see."""
    result = builder_tools.status_of(_crowded(tmp_path))
    text = builder_tools.render_status(result)
    flat = _flat(text)
    assert len(result.red_lights) > 250, "the crowded workdir is the size the live build reached"
    for stage in result.failing:
        assert f"{stage}: " in text, f"{stage} is red and has no block"
    for task_id in CROWDED_TASKS:
        assert task_id in flat, f"{task_id} is red and unnamed"
    for name in CROWDED_TOOLS:
        assert name in flat, f"{name} is red and unnamed"
    assert "more, all in details" not in flat and "..." not in flat
    assert len(text) < 12000, "the whole picture still has to fit in what the model reads"
    assert builder_tools.render_status(builder_tools.status_of(tmp_path)) == text, "not deterministic"


def test_status_groups_task_failures_by_kind_and_shows_the_phrases_that_fail(tmp_path):
    """The two ways an Intent is refused are two lines, each naming its Tasks and what their words were."""
    flat = _flat(builder_tools.render_status(builder_tools.status_of(_crowded(tmp_path))))
    with_no_span = [t for i, t in enumerate(CROWDED_TASKS) if i % 6]
    assert f"noun phrases with no span: {len(with_no_span)} Tasks: " + ", ".join(with_no_span) in flat
    assert f"noun phrases not evidenced in every Run: {184 - len(with_no_span)} Tasks: " in flat
    examples = flat.split(f"what fails ({builder_tools.KIND_EXAMPLES} of {len(with_no_span)}): ")[1]
    examples = examples.split("noun phrases not evidenced")[0]
    assert examples.count("task_") == builder_tools.KIND_EXAMPLES
    assert "the exchange" in examples, "the phrase that has no span is what tells the model what to fix"


def test_status_zooms_on_one_gate_or_one_target_and_lists_its_red_lights_in_full(tmp_path):
    """`status()` groups; `status(gate=...)` and `status(target=...)` print every line of one of them."""
    workdir = _crowded(tmp_path)
    plan = BuildPlan(workdir=workdir, iterate=True)
    zoomed = _run(_tool(plan, "status"), {"gate": "intent"})
    assert not zoomed.is_error and zoomed.content.startswith("status(gate=intent): 184 red lights of ")
    assert zoomed.content.count("\n- ") == 184, "a zoom lists every red light of the gate, one per line"
    assert all(t in zoomed.content for t in CROWDED_TASKS)
    on_task = _run(_tool(plan, "status"), {"target": CROWDED_TASKS[0]})
    assert {light["stage"] for light in on_task.details["red_lights"]} == {"intent", "derive_verifier"}
    assert on_task.content.count("\n- ") == 2
    assert on_task.details["failing"] == builder_tools.status_of(workdir).failing, (
        "a zoom still says which gates are red")
    nothing = _run(_tool(plan, "status"), {"gate": "no_such_gate"})
    assert "no red light matches" in nothing.content and "intent" in nothing.content
    assert _run(_tool(plan, "status"), {}).content == builder_tools.render_status(
        builder_tools.status_of(workdir)), "no argument is still the whole grouped picture"


def test_one_very_long_failure_is_shortened_in_the_grouped_view_and_whole_in_the_zoom(tmp_path):
    """The list is never cut, but a single failure that dumps a table's every key would crowd the
    picture out on its own, so the grouped view says how much is left and the zoom shows it."""
    dump = "get_order: hard columns differ: keys differ: " + ", ".join(f"key_{i}" for i in range(200))
    (tmp_path / "gates.json").write_text(
        json.dumps([{"stage": "replay_fidelity", "pass": False, "failures": [dump]}]), encoding="utf-8")
    grouped = builder_tools.render_status(builder_tools.status_of(tmp_path))
    assert dump not in grouped and f"[+{len(dump) - builder_tools.FAILURE_CHARS} characters" in grouped
    assert dump in builder_tools.render_status(builder_tools.status_of(tmp_path, target="get_order"))


def test_the_headline_counts_the_gates_the_tasks_with_no_verdict_and_the_assisted_tools(tmp_path):
    """The first line is the whole picture: how much is red, what it costs in Tasks, what is assisted."""
    result = builder_tools.status_of(_crowded(tmp_path))
    head = result.summary
    assert head.startswith(f"status: {len(result.red_lights)} red lights; ")
    assert "4 of 14 gates red" in head, "four gates red of the fourteen the records name"
    assert "184 Tasks with no Verdict, most of them noun phrases with no span" in head
    assert "4 tools assisted: " + ", ".join(CROWDED_TOOLS[:4]) in head
    assert builder_tools.render_status(result).startswith(head + "\n")


@pytest.mark.parametrize("failure,target,kind", [
    ("task task_7: no Trace of the Task was replayed", "task_7", "task"),
    ("get_order: a recorded success call replays differently", "get_order", "tool"),
    ('get_order({"id": 1}): hard columns differ', "get_order", "tool"),
    ("search_products has no body", "search_products", "tool"),
    ("no re-roll finished: 12 Runs and every one stopped on an error", "", "build"),
])
def test_a_failure_names_the_tool_or_the_task_the_way_the_gate_wrote_it(failure, target, kind):
    assert builder_tools._named_by(failure) == (target, kind)


def test_every_ruling_the_registry_names_has_a_verb_that_acts_or_an_owner():
    """A ruling may lead to a verb that changes the artifact, to a person, or to the agent that owns
    it; a verb that only records a decision is not an answer to a ruling something can repair."""
    import kullback.gates as gates_pkg

    verbs = {"repair_recompile", "repair_grow", "repair_intent", "reroll", "repair_escalate",
             builder_tools.EXAMINER_OWNS}
    for spec in gates_pkg.GATES:
        for ruling in spec.rulings:
            assert builder_tools.verb_for(ruling) in verbs, ruling
    assert builder_tools.verb_for("a_ruling_nobody_registered") == "repair_escalate"
    assert "repair_refuse_task" not in set(builder_tools.REPAIR_VERB_FOR.values()), (
        "refusing a Task repairs nothing, so no ruling points at it")
    assert builder_tools.verb_for("derive_verifier") == builder_tools.EXAMINER_OWNS


# --- the repair verbs ----------------------------------------------------------

def test_the_five_repair_verbs_are_registered_and_the_skill_rewrite_is_not(tmp_path):
    names = [t.name for t in builder_tools.repair_verb_tools(BuildPlan(workdir=tmp_path))]
    assert names == ["repair_recompile", "repair_grow", "repair_intent", "repair_refuse_task",
                     "repair_escalate"]
    assert "repair_rewrite_skill" not in names, "a model rewriting its own prompt stays gated (GEPA caution)"


def test_repair_recompile_keeps_the_hint_as_a_lesson_and_compiles_the_tool_again(built):
    plan = BuildPlan(workdir=built, iterate=True, model=Bodies(), max_attempts=0)
    before = json.loads((built / "bodies.json").read_text(encoding="utf-8"))
    out = _run(_tool(plan, "repair_recompile"), {"name": ASSISTED, "hint": "import decimal before using it"})
    assert not out.is_error, out.content
    assert out.details["target"] == "compile_tools" and "compile_tools" in out.content
    assert repair_module.lesson_for_tool(built, ASSISTED).count("import decimal before using it") == 1
    request = (built / "repairs" / "repair_recompile.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(request[-1])["arguments"]["hint"] == "import decimal before using it"
    after = json.loads((built / "bodies.json").read_text(encoding="utf-8"))
    assert set(after) == set(before), "one tool repaired, every other body still there"


def test_repair_recompile_with_a_hint_recompiles_the_tool_instead_of_hitting_the_cache(built, tmp_path):
    """A recompile that changes nothing is right to be served from the cache. One whose hint became
    a new lesson is a new question, since the lesson reaches that tool's compiler prompt: the live
    build asked seven times running and was answered "5 stages, 5 from cache" every time, because
    the lesson was in nothing the stage declared as an input.

    The copy keeps this test off the module's shared build, whose lessons the tests around it write.
    """
    workdir = tmp_path / "recompile"
    shutil.copytree(built, workdir)
    plan = BuildPlan(workdir=workdir, iterate=True, model=Bodies(), max_attempts=0)
    tool = _tool(plan, "repair_recompile")
    _run(tool, {"name": ASSISTED, "hint": "the body has to import decimal"})
    _run(tool, {"name": ASSISTED})  # the bodies and the cache entry settle on these inputs

    steady = _run(tool, {"name": ASSISTED})
    assert _cached(steady, "compile_tools") is True, "nothing changed, so the cache is right to answer"
    again = _run(tool, {"name": ASSISTED, "hint": "the rows are pydantic models, not dicts"})
    assert _cached(again, "compile_tools") is False, "a new lesson is a new question for the stage"
    assert repair_module.lesson_for_tool(workdir, ASSISTED).count("pydantic models") == 1


def test_repair_recompile_without_a_hint_records_no_lesson(built):
    plan = BuildPlan(workdir=built, iterate=True, model=Bodies(), max_attempts=0)
    before = repair_module.lesson_for_tool(built, "get_user_details")
    out = _run(_tool(plan, "repair_recompile"), {"name": "get_user_details"})
    assert not out.is_error, out.content
    assert repair_module.lesson_for_tool(built, "get_user_details") == before


def test_repair_grow_grows_the_table_and_records_the_request(built):
    plan = BuildPlan(workdir=built, iterate=True, model=Bodies(), max_attempts=0)
    out = _run(_tool(plan, "repair_grow"), {"table": "users", "count": 4})
    assert not out.is_error, out.content
    assert out.details["target"] == "starting_state"
    row = json.loads((built / "repairs" / "repair_grow.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["target"] == "users" and row["arguments"]["count"] == 4


def _intents(workdir: Path) -> dict:
    return {path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((workdir / "intents").glob("*.json"))}


def _intent_prompts(model: Bodies) -> list[str]:
    """Every prompt the Builder's model was asked the Intent question with."""
    texts = [" ".join(str(m.get("content") or "") for m in call["messages"]) for call in model.calls]
    return [text for text in texts if "Write one line saying what the user wanted" in text]


def test_repair_intent_reruns_the_intent_stage_for_one_task_with_the_hint_in_the_prompt(built):
    model = Bodies()
    plan = BuildPlan(workdir=built, iterate=True, model=model, max_attempts=0)
    before = _intents(built)
    task = sorted(before)[0]
    out = _run(_tool(plan, "repair_intent"), {"task_id": task, "hint": "say only what every run of it shows"})
    assert not out.is_error, out.content
    assert out.details["target"] == "intent" and "intent" in out.content
    stages = {s["name"]: s for s in out.details["stages"]}
    assert stages["intent"]["cached"] is False and stages["cluster"]["cached"] is True
    prompts = _intent_prompts(model)
    assert prompts, "the narrowed stage asked the model for that Task's Intent"
    assert all("A repair asks for this: say only what every run of it shows" in p for p in prompts)
    assert set(_intents(built)) == set(before), "the other Tasks keep the Intent they had"
    assert set(plan.store["intents"]) == set(before)
    row = json.loads((built / "repairs" / "repair_intent.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["target"] == task and row["arguments"]["hint"] == "say only what every run of it shows"
    missing = _run(_tool(plan, "repair_intent"), {"task_id": "no_such_task", "hint": "x"})
    assert missing.is_error and "no Task is named" in missing.content


def test_a_repeated_repair_intent_with_a_new_hint_is_not_served_from_the_cache(built):
    """The hint is part of the stage's identity, so a repair with something new to say actually reruns."""
    plan = BuildPlan(workdir=built, iterate=True, model=Bodies(), max_attempts=0)
    task = sorted(_intents(built))[0]
    first = _run(_tool(plan, "repair_intent"), {"task_id": task, "hint": "one hint"})
    again = _run(_tool(plan, "repair_intent"), {"task_id": task, "hint": "one hint"})
    fresh = _run(_tool(plan, "repair_intent"), {"task_id": task, "hint": "another hint"})
    for out in (first, again, fresh):
        assert not out.is_error, out.content
    cached = [{s["name"]: s["cached"] for s in out.details["stages"]}["intent"] for out in (first, again, fresh)]
    assert cached == [False, True, False]


def test_an_intent_red_light_sends_the_builder_to_repair_intent_and_not_to_a_refusal(built):
    """The live build called repair_refuse_task 41 times on ungrounded Intents and nothing changed."""
    lights = [light for light in builder_tools.red_lights(built) if light.stage == "intent"]
    assert lights, "the fixture build leaves ungrounded Intents to report"
    assert {light.verb for light in lights} == {"repair_intent"}
    assert all(light.kind == "task" and light.target for light in lights)
    line = builder_tools.render_status(builder_tools.status_of(built))
    assert "-> repair_intent" in line


def test_a_deciding_verb_records_its_decision_and_produces_no_artifact(tmp_path):
    plan = BuildPlan(workdir=tmp_path)
    out = _run(_tool(plan, "repair_refuse_task"), {"task_id": "task_1", "reason": "no frontier Run finishes it"})
    assert not out.is_error and out.details["verb"] == "repair_refuse_task"
    assert not out.details.get("produced"), "a decision changes no artifact, so no gate rules on it"
    escalated = _run(_tool(plan, "repair_escalate"), {"task_id": "task_1", "queue": "review"})
    assert not escalated.is_error and "repair_escalate task_1" in escalated.content


def test_a_repair_verb_with_bad_arguments_is_an_error_result_not_a_crash(tmp_path):
    plan = BuildPlan(workdir=tmp_path)
    out = _run(_tool(plan, "repair_grow"), {"table": "users", "count": 0})
    assert out.is_error and "count" in out.content
