"""A synthetic store through the whole chain, every model a scripted one (D360).

The kite world of tests/test_synthesise.py with its toolkit rendered and a policy shipped: one walk is
kept as a store Task whose description an echoing double writes. From there the chain runs as it runs on
a recorded workdir: the Candidate opens on the description with no user, the miner reads the description
as one volunteered turn, the writer writes the Verifier, the Examiner's goal step and review round run,
the walk is the Reference, mutants score its End state, and trust lands a tier under its own heading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback import synthesise
from kullback.agent.bus import Bus
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.builder import compile_env
from kullback.runner.records import EntitySchema, Run, ToolSig
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
    row = body["tasks"][0]
    return store, row["task_id"], row["run_id"]


def _walk(store: Path, task_id: str, run_id: str) -> Run:
    from kullback.spec.trust import runs_on_disk

    return next(run for run in runs_on_disk(store)[task_id] if run.run_id == run_id)


def _written(store: Path, run: Run) -> tuple[str, dict]:
    """The one row the walk changed and its changed fields, read off the walk's own End state."""
    stop = next(event for event in run.events if event.type == "stop").payload
    start, end = stop["start_state"], stop["end_state"]
    for table, rows in end.items():
        for key, row in rows.items():
            changed = {field: value for field, value in row.items()
                       if (start.get(table) or {}).get(key, {}).get(field) != value}
            if changed:
                return key, changed
    raise AssertionError("the walk changed nothing")


# --- the Task record and the Candidate's opening ----------------------------------------------


def test_the_store_says_its_tasks_are_synthetic_and_the_parent_is_not(kite_store, kite_world):  # noqa: F811
    store, task_id, _run_id = kite_store
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

    store, task_id, _run_id = kite_store
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


def _verified(store: Path, task_id: str, run_id: str, expect: dict | None = None):
    """The Task mined and its Verifier written from the Intent: one gated row item on the walk's row,
    expecting what the description asks (the walk's own change unless `expect` says otherwise)."""
    from kullback.spec.stage import write_from_intent

    mined = _mined(store, task_id)
    key, changed = _written(store, _walk(store, task_id, run_id))
    item = {"id": "s1", "kind": "row_is", "table": "kites", "find": {"kite_id": key}, "row": key,
            "expect": expect or changed, "gate": True, "weight": 1.0, "fact_ids": ["f1"], "policy_line": None}
    confirm = {"id": "c1", "kind": "before", "first": "confirm_turn", "then": "retie_kite", "gate": True,
               "weight": 1.0, "fact_ids": [], "policy_line": POLICY.splitlines()[-1]}
    writer = TestModel([call("add_items", items=[item, confirm]), DONE], loop=True)
    write_from_intent(store, task_id, mined.intent, writer, 1.0, Bus(store / "bus.jsonl", agent="spec"))
    return mined


def test_the_miner_reads_the_description_as_one_volunteered_turn(kite_store):
    store, task_id, _run_id = kite_store
    mined = _mined(store, task_id)
    [fact] = mined.intent.facts
    assert (fact.stance, fact.source.recording, fact.source.turn) == \
        ("volunteered", intent_tools.DESCRIPTION_SOURCE, 0)
    assert not fact.echo and not fact.after_pushback


def test_the_writer_writes_the_verifier_from_the_intent_and_never_from_the_walk(kite_store):
    from kullback.spec.schema import load_spec
    from kullback.spec.writer import runner_verifier_path

    store, task_id, run_id = kite_store
    _verified(store, task_id, run_id)
    spec = load_spec(store, task_id)
    assert spec.set_aside is None and runner_verifier_path(store, task_id).is_file()
    assert sorted(check.id for check in spec.checks if check.id in ("s1", "c1")) == ["c1", "s1"]


def test_the_walk_as_reference_passes_the_verifier_and_both_constructed_runs_fail(kite_store):
    from kullback.examiner.rule_tool import verify_rows
    from kullback.spec.trust import load_spec_verifier

    store, task_id, run_id = kite_store
    _verified(store, task_id, run_id)
    verifier = load_spec_verifier(store, task_id)
    rows = {row["run"]: row for row in verify_rows(store, task_id, verifier, _walk(store, task_id, run_id))}
    assert rows["reference"]["verifier_passes_it"] is True
    assert rows["do_nothing"]["verifier_fails_it"] is True and rows["stray_write"]["verifier_fails_it"] is True


def test_the_goal_step_reads_the_description_as_the_users_words(kite_store):
    from kullback.examiner import goal as G

    store, task_id, run_id = kite_store
    key, changed = _written(store, _walk(store, task_id, run_id))
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


def test_a_review_round_that_rules_nothing_lands_the_task_trusted_in_the_stores_own_tiers(kite_store, kite_world):  # noqa: F811
    from kullback.spec.mutants import workdir_mutants
    from kullback.spec.trust import intent_tiers

    store, task_id, run_id = kite_store
    _verified(store, task_id, run_id)
    out = _examined(store, task_id)
    assert out["examined"] == [task_id] and out["held"] == []
    tier = intent_tiers(store)[task_id]
    assert tier.tier == "trusted", tier
    assert task_id not in intent_tiers(kite_world), "the parent's tiers read a synthetic Task"
    report = workdir_mutants(store, [task_id])[task_id]
    assert report["status"] == "scored" and report["reference"] == run_id, report
    assert report["killed"] >= 1 and report["built"] == report["killed"] + report["survived"] + report["open"]


def test_a_walk_that_did_not_do_what_the_description_asks_ends_the_task_with_no_verifier(kite_store):
    from kullback.spec.schema import load_spec
    from kullback.spec.trust import NO_VERIFIER, intent_tiers

    store, task_id, run_id = kite_store
    walk = _walk(store, task_id, run_id)
    key, changed = _written(store, walk)
    [(field, value)] = changed.items()
    path = store / "tasks" / f"{task_id}.json"
    task = json.loads(path.read_text())
    asked = str(int(value) + 7)
    task["intent"] = task["intent"].replace(value, asked)
    path.write_text(json.dumps(task))
    _verified(store, task_id, run_id, expect={field: asked})
    assert load_spec(store, task_id).checks[0].demand["expect"] == {field: asked}
    write = next(event.idx for event in walk.events if event.type == "tool_call"
                 and event.payload.get("args", {}).get(field) == value)
    _examined(store, task_id, call("rule", task_id=task_id, item="s1", kind="code", code="fails_reference",
                                   reason="The walk wrote a value the description never asked for.",
                                   fix="Keep the check; the walk is wrong.", blocking=True,
                                   wrong_side="reference", ref_step=write))
    tier = intent_tiers(store)[task_id]
    assert (tier.tier, tier.row["reason"], tier.row["why"]) == ("untrusted", NO_VERIFIER, "reference_wrong"), tier


# --- every count of a store is printed under its own heading -------------------------------------


def test_the_stores_tier_report_and_counts_line_print_under_the_synthetic_heading(kite_store, kite_world):  # noqa: F811
    from kullback import cli
    from kullback.spec import report

    store, task_id, run_id = kite_store
    _verified(store, task_id, run_id)
    _examined(store, task_id)
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

    store, task_id, run_id = kite_store
    manifest = package.export(store, tmp_path / "package", name="kites-synthetic", scan=False)
    assert manifest["origin"] == intent_tools.SYNTHETIC_ORIGIN
    body = card.card_markdown(manifest, "someone/kites-synthetic")
    assert "## Synthetic Tasks" in body and "| Trusted synthetic Tasks, never counted with recorded ones |" in body
    recorded = card.card_markdown(dict(manifest, origin="recorded"), "someone/kites")
    assert "## Synthetic Tasks" not in recorded and "| Trusted |" in recorded
