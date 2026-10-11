"""A synthetic store through the whole chain, every model a scripted one (D360).

The kite world of tests/test_synthesise.py with its toolkit rendered and a policy shipped: one walk is
kept as a store Task whose description an echoing double writes. The walk is only the Task's scenario
source, never its Reference. From there the chain runs: the Candidate opens on the description with no
user, the miner reads the description as one volunteered turn, the writer writes the Verifier, the
Examiner's goal step and review round run, mutants score the compiled expected end state, and trust lands
a tier under its own heading once a fresh Run passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback import synthesise
from kullback.agent.bus import Bus
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.builder import compile_env
from kullback.runner.records import EntitySchema, ToolSig
from kullback.spec import intent_tools
from tests.test_synthesise import _EchoModel, kite_world  # noqa: F401 - the fixture is used by name

POLICY = ("# Kite desk policy\n\n## Changes\n\n"
          "Before you change a kite, confirm the change with the customer.\n")


def call(name: str, **arguments) -> ModelReply:
    return ModelReply(tool_calls=[ToolCallRequest(id=f"{name}-1", name=name, arguments=arguments)])


DONE = ModelReply(content="done")


@pytest.fixture
def kite_store(kite_world: Path):  # noqa: F811 - the imported fixture is the parent workdir
    """The kite world with its toolkit and policy on disk, and one walk kept as a store Task."""
    schema = EntitySchema.model_validate_json((kite_world / "schema.json").read_text())
    sigs = [ToolSig.model_validate(row) for row in json.loads((kite_world / "tool_sigs.json").read_text())]
    bodies = json.loads((kite_world / "bodies.json").read_text())
    env = kite_world / "env"
    env.mkdir()
    (env / "data_model.py").write_text(compile_env.render_data_model(schema))
    (env / "tools.py").write_text(compile_env.render_tools(schema, sigs, bodies))
    (env / "policy.md").write_text(POLICY)
    body = synthesise.build_store(kite_world, {"w1t3p1": 1}, seed="store-chain", model=_EchoModel())
    assert body["counts"]["tasks_kept"] == 1, body["counts"]
    store = synthesise.store_path(kite_world)
    return store, body["tasks"][0]["task_id"]


def _written(store: Path, task_id: str) -> tuple[str, dict]:
    """The row the scenario's write names and the change it asks for, read off the scenario source."""
    scenario = json.loads((store / synthesise.SCENARIOS / f"{task_id}.json").read_text())
    args = next(step["args"] for step in scenario["steps"] if step["tool"] == "retie_kite")
    return args["kite_id"], {"length": args["length"]}


# --- the Task record and the Candidate's opening ----------------------------------------------


def test_the_store_says_its_tasks_are_synthetic_and_the_parent_is_not(kite_store, kite_world):  # noqa: F811
    store, task_id = kite_store
    assert intent_tools.is_synthetic_store(store) and not intent_tools.is_synthetic_store(kite_world)
    marker = json.loads((store / intent_tools.SYNTHETIC_MARKER).read_text())
    assert marker["origin"] == intent_tools.SYNTHETIC_ORIGIN
    index = json.loads((store / synthesise.STORE_INDEX).read_text())
    assert [(row["task_id"], row["origin"]) for row in index["tasks"]] == \
        [(task_id, intent_tools.SYNTHETIC_ORIGIN)]


def test_a_candidate_run_opens_on_the_description_and_no_user_is_started(kite_store):
    from kullback.runner import tool as runner_tool
    from kullback.runner.world.environment import BuiltEnvironment
    from kullback.runner.world.episode import Episode

    store, task_id = kite_store
    env = BuiltEnvironment(store)
    task = env.task(task_id)
    assert task.facts_in_instruction is False and env.opening(task) == task.intent
    started = []
    episode = Episode(env, outdir=store / "episodes",
                      user_factory=lambda *args: started.append(args) or object())
    assert episode.reset(task_id).opening == task.intent
    assert started == [], "a user was started for a Task that has none"
    candidate = TestModel([ModelReply(content="It is done.")])
    report = runner_tool.run(store, task_id, candidate, workdir=store / "fresh")
    [first] = [message for message in candidate.calls[0]["messages"] if message["role"] == "user"]
    assert first["content"] == task.intent and report.task_id == task_id


# --- the chain: miner, writer, goal step, review round, mutants and trust ----------------------


def _mined(store: Path, task_id: str):
    from kullback.spec.intent import mine_intent

    text = json.loads((store / "tasks" / f"{task_id}.json").read_text())["intent"]
    model = TestModel([call("list_recordings", task_id=task_id),
                       call("user_turns", task_id=task_id, recording=intent_tools.DESCRIPTION_SOURCE),
                       call("add_facts", facts=[{"text": text, "recording": intent_tools.DESCRIPTION_SOURCE,
                                                 "turn": 0, "stance": "volunteered"}]),
                       DONE])
    return mine_intent(store, task_id, model, ceiling_usd=1.0)


def _verified(store: Path, task_id: str, confirm: bool = True):
    """The Task mined and its Verifier written from the Intent: one gated row item on the row the
    description names, expecting the change it asks for, and (unless told not to) one gated confirm
    item off the policy."""
    from kullback.spec.stage import write_from_intent

    mined = _mined(store, task_id)
    key, changed = _written(store, task_id)
    item = {"id": "s1", "kind": "row_is", "table": "kites", "find": {"kite_id": key}, "row": key,
            "expect": changed, "gate": True, "weight": 1.0, "fact_ids": ["f1"], "policy_line": None}
    confirm_item = {"id": "c1", "kind": "before", "first": "confirm_turn", "then": "retie_kite", "gate": True,
               "weight": 1.0, "fact_ids": [], "policy_line": POLICY.splitlines()[-1]}
    items = [item, confirm_item] if confirm else [item]
    writer = TestModel([call("add_items", items=items), DONE], loop=True)
    write_from_intent(store, task_id, mined.intent, writer, 1.0, Bus(store / "bus.jsonl", agent="spec"))
    return mined


def test_the_miner_reads_the_description_as_one_volunteered_turn(kite_store):
    store, task_id = kite_store
    mined = _mined(store, task_id)
    [fact] = mined.intent.facts
    assert (fact.stance, fact.source.recording, fact.source.turn) == \
        ("volunteered", intent_tools.DESCRIPTION_SOURCE, 0)
    assert not fact.echo and not fact.after_pushback


def test_the_writer_writes_the_verifier_from_the_intent_and_never_from_the_walk(kite_store):
    from kullback.spec.schema import load_spec
    from kullback.spec.writer import runner_verifier_path

    store, task_id = kite_store
    _verified(store, task_id)
    spec = load_spec(store, task_id)
    assert spec.set_aside is None and runner_verifier_path(store, task_id).is_file()
    assert sorted(check.id for check in spec.checks if check.id in ("s1", "c1")) == ["c1", "s1"]


def _honest(store: Path, task_id: str) -> list[ModelReply]:
    """A Candidate's calls that read what the description leads to before writing: the scenario's steps."""
    scenario = json.loads((store / synthesise.SCENARIOS / f"{task_id}.json").read_text())
    return [call(step["tool"], **step["args"]) for step in scenario["steps"]]


def _fresh(store: Path, task_id: str, *replies: ModelReply):
    """One scripted Candidate Run of the store Task, written under the store's runs/ like any fresh Run."""
    from kullback.runner import tool as runner_tool

    return runner_tool.run(store, task_id, TestModel([*replies, ModelReply(content="It is done.")]),
                           workdir=store, run_id=f"fresh-{len(replies)}")


def test_a_store_task_has_no_reference_and_its_constructed_runs_lie_on_the_starting_state(kite_store):
    from kullback.examiner.rule_tool import verify_rows
    from kullback.spec import scenario
    from kullback.spec.trust import constructed_runs, faithful_references, load_spec_verifier

    store, task_id = kite_store
    _verified(store, task_id)
    verifier = load_spec_verifier(store, task_id)
    assert faithful_references(store) == {}
    assert verify_rows(store, task_id, verifier, None) == [{"run": "reference", "error": "no faithful Reference on disk"}]
    expected = scenario.store_expected_run(store, task_id, verifier)
    built = constructed_runs(verifier, [expected], [])
    stop = built["do_nothing"].events[-1].payload
    assert stop["end_state"] == stop["start_state"] == scenario.starting_state(store, task_id)


def test_the_goal_step_reads_the_description_as_the_users_words(kite_store):
    from kullback.examiner import goal as G

    store, task_id = kite_store
    key, changed = _written(store, task_id)
    text = json.loads((store / "tasks" / f"{task_id}.json").read_text())["intent"]
    assert G.user_words(store, task_id) == {intent_tools.DESCRIPTION_SOURCE: [{"turn": 0, "user": text}]}
    [(field, value)] = changed.items()
    cell = {"table": "kites", "row": key, "field": field, "value": value, "user": value}
    out = G.goals(store, [task_id], TestModel([ModelReply(content=json.dumps({"required": [cell]}))]))
    assert out["written"] == 1
    goal = json.loads(G.goal_path(store, task_id).read_text())
    assert [(row["row"], row["value"]) for row in goal["required"]] == [(key, value)]


def _examined(store: Path, task_id: str, *replies: ModelReply) -> dict:
    from kullback.examiner.session import examine_rounds

    examiner = TestModel([call("verify", task_id=task_id), *replies, ModelReply(content="Nothing more to rule.")],
                         loop=True)
    return examine_rounds(store, [task_id], model=examiner, writer_model=TestModel([DONE], loop=True))


def test_a_review_round_and_a_passing_fresh_run_land_the_task_trusted_in_the_stores_own_tiers(kite_store, kite_world):  # noqa: F811
    from kullback.spec.trust import STRAY_UNTESTED, intent_tiers

    store, task_id = kite_store
    key, changed = _written(store, task_id)
    # No confirm item: the runner logs a Task's opening as a model message and never as a user turn,
    # so no fresh Run of a description-only Task can satisfy a confirm_turn gate (open in the report).
    _verified(store, task_id, confirm=False)
    out = _examined(store, task_id)
    assert out["examined"] == [task_id] and out["held"] == []
    tier = intent_tiers(store)[task_id]
    assert (tier.tier, tier.row["reason"]) == ("untrusted", STRAY_UNTESTED), "no Run passed yet, so nothing tested the stray write"
    _fresh(store, task_id, *_honest(store, task_id))
    tier = intent_tiers(store)[task_id]
    assert tier.tier == "trusted" and tier.row["reference"]["runs"] == [], tier
    assert task_id not in intent_tiers(kite_world), "the parent's tiers read a synthetic Task"


def test_mutants_of_a_store_task_score_the_compiled_expected_end_state_and_not_a_walk(kite_store):
    from kullback.spec import scenario
    from kullback.spec.mutants import workdir_mutants

    store, task_id = kite_store
    _verified(store, task_id)
    report = workdir_mutants(store, [task_id])[task_id]
    assert report["status"] == "scored" and report["base"] == scenario.EXPECTED, report
    assert report["reference"] == f"{task_id}.{scenario.EXPECTED}"
    assert report["killed"] >= 1 and report["built"] == report["killed"] + report["survived"] + report["open"]


def test_a_fails_reference_ruling_never_applies_to_a_store_task(kite_store):
    from kullback.spec.trust import intent_tiers

    store, task_id = kite_store
    _verified(store, task_id)
    _examined(store, task_id, call("rule", task_id=task_id, item="s1", kind="code", code="fails_reference",
                                   reason="The walk wrote another value.", fix="Keep the check.", blocking=True,
                                   wrong_side="reference", ref_step=1))
    tier = intent_tiers(store)[task_id]
    assert tier.row["why"] != "reference_wrong" and tier.row["reason"] != "no_verifier", tier


# --- every count of a store is printed under its own heading -------------------------------------


def test_the_stores_tier_report_and_counts_line_print_under_the_synthetic_heading(kite_store, kite_world):  # noqa: F811
    from kullback import cli
    from kullback.spec import report

    store, task_id = kite_store
    key, changed = _written(store, task_id)
    _verified(store, task_id, confirm=False)
    _examined(store, task_id)
    _fresh(store, task_id, *_honest(store, task_id))
    written = report.write_report(store, corpus="kites")
    assert written["origin"] == intent_tools.SYNTHETIC_ORIGIN and written["tiers"]["trusted"] == 1
    assert report.header_line(written).startswith(report.SYNTHETIC_HEADING + ": ")
    assert "| kites (synthetic) |" in report.table_line(written)
    assert cli._counts_line(store).startswith(report.SYNTHETIC_HEADING + ": ")
    parent = report.write_report(kite_world, corpus="kites")
    assert "origin" not in parent and not report.header_line(parent).startswith(report.SYNTHETIC_HEADING)
    assert not cli._counts_line(kite_world).startswith(report.SYNTHETIC_HEADING)


def test_a_published_store_carries_its_origin_and_the_card_heads_its_counts_apart(kite_store, tmp_path):
    from kullback.hub import card, package

    store, _task_id = kite_store
    manifest = package.export(store, tmp_path / "package", name="kites-synthetic", scan=False)
    assert manifest["origin"] == intent_tools.SYNTHETIC_ORIGIN
    body = card.card_markdown(manifest, "someone/kites-synthetic")
    assert "## Synthetic Tasks" in body and "| Trusted synthetic Tasks, never counted with recorded ones |" in body
    recorded = card.card_markdown(dict(manifest, origin="recorded"), "someone/kites")
    assert "## Synthetic Tasks" not in recorded and "| Trusted |" in recorded


# --- the shortcut audit (synth-add3 item 2) -------------------------------------------------------


def test_a_run_that_calls_none_of_the_scenarios_writes_is_a_shortcut(kite_store):
    from kullback.spec import scenario
    from kullback.spec.trust import runs_on_disk

    store, task_id = kite_store
    _fresh(store, task_id, *_honest(store, task_id))
    [run] = runs_on_disk(store)[task_id]
    text = json.loads((store / "tasks" / f"{task_id}.json").read_text())["intent"]
    assert scenario.shortcut(run, text, {"retie_kite"}) == ""
    assert scenario.shortcut(run, text, {"repaint_kite"}).startswith("called none")


def test_a_passing_run_naming_a_value_it_could_not_have_read_holds_the_task_as_a_shortcut(kite_store):
    from kullback.spec import scenario
    from kullback.spec.report import build_report
    from kullback.spec.trust import intent_tiers

    store, task_id = kite_store
    key, changed = _written(store, task_id)
    task_path = store / "tasks" / f"{task_id}.json"
    task = json.loads(task_path.read_text())
    _verified(store, task_id, confirm=False)
    _examined(store, task_id)
    _fresh(store, task_id, *_honest(store, task_id))
    task["intent"] = "Please sort my kite out. Go ahead."
    task_path.write_text(json.dumps(task))
    tiers = intent_tiers(store)
    assert tiers[task_id].row["reason"] == scenario.SHORTCUT
    assert "argument value" in tiers[task_id].row["shortcuts"][0]["why"]
    assert build_report(tiers)["reasons"][scenario.SHORTCUT] == 1


def test_a_passing_run_that_read_what_it_wrote_is_no_shortcut(kite_store):
    from kullback.spec import scenario
    from kullback.spec.trust import runs_on_disk

    store, task_id = kite_store
    key, changed = _written(store, task_id)
    _fresh(store, task_id, call("get_kite", kite_id=key), call("retie_kite", kite_id=key, **changed))
    [run] = runs_on_disk(store)[task_id]
    text = f"My kite {key} needs retying at {changed['length']}."
    assert scenario.shortcut(run, text, {"retie_kite"}) == ""
    assert scenario.shortcut(run, "Please fix my kite.", {"retie_kite"}) != ""
