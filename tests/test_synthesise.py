"""Synthetic Tasks over an invented world: bound, run, derived and counted apart (D224)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback import graph, synthesise
from kullback.ai import TestModel
from kullback.runner.records import Column, EntitySchema, FieldStat, ToolSig, write_json

# One world nobody's customer has: kites on spools. Two reads and one write, the write refusing a
# kite that is already tied at the length it is asked for, so a walk can be refused by the world.
BODIES = {
    "find_kite": "for kite in self.db.kites.values():\n"
                 "    if kite.colour == colour:\n"
                 "        return {'kite_id': kite.kite_id}\n"
                 "raise ValueError('no kite of that colour')",
    "get_kite": "kite = self.db.kites[kite_id]\n"
                "return {'kite_id': kite.kite_id, 'colour': kite.colour, 'spool_id': kite.spool_id,"
                " 'length': kite.length}",
    "retie_kite": "kite = self.db.kites[kite_id]\n"
                  "if kite.length == length:\n"
                  "    raise ValueError('the kite is already tied at that length')\n"
                  "kite.length = length\n"
                  "return {'kite_id': kite.kite_id, 'length': kite.length}",
}

def _sig(name: str, kind: str, *args: str) -> ToolSig:
    return ToolSig(name=name, kind=kind,
                   args_fields=[FieldStat(name=arg, types=["str"], optional=False) for arg in args],
                   args_schema={"type": "object",
                                "properties": {arg: {"type": ["str"]} for arg in args},
                                "required": list(args)})


SIGS = [_sig("find_kite", "read", "colour"), _sig("get_kite", "read", "kite_id"),
        _sig("retie_kite", "write", "kite_id", "length")]

DB = {"kites": {f"k{n}": {"kite_id": f"k{n}", "colour": f"teal{n}", "spool_id": f"s{n}",
                          "length": f"{30 + n}"}
                for n in range(1, 6)},
      "spools": {f"s{n}": {"spool_id": f"s{n}", "turns": f"{n}"} for n in range(1, 6)}}


def _recording(number: int) -> list[dict]:
    """One recorded Run of this world: find a kite, read it, retie it at another length."""
    kite = f"k{number}"
    return [{"name": "find_kite", "args": {"colour": f"teal{number}"},
             "result": {"kite_id": kite}, "error": None},
            {"name": "get_kite", "args": {"kite_id": kite},
             "result": {"kite_id": kite, "colour": f"teal{number}", "spool_id": f"s{number}",
                        "length": f"{30 + number}"}, "error": None},
            {"name": "retie_kite", "args": {"kite_id": kite, "length": "99"},
             "result": {"kite_id": kite, "length": "99"}, "error": None}]


def _columns() -> list[Column]:
    out = []
    for table, row in (("kites", DB["kites"]["k1"]), ("spools", DB["spools"]["s1"])):
        out += [Column(table=table, name=name, class_="hard") for name in row]
    return out


@pytest.fixture
def kite_world(tmp_path: Path) -> Path:
    """A finished workdir of the kite world: the schema, the tools, the bodies and the graph."""
    workdir = tmp_path / "work"
    schema = EntitySchema(tables=["kites", "spools"], columns=_columns())
    write_json(workdir / "schema.json", json.loads(schema.model_dump_json()))
    write_json(workdir / "tool_sigs.json", [json.loads(sig.model_dump_json()) for sig in SIGS])
    write_json(workdir / "bodies.json", BODIES)
    write_json(workdir / "db.json", DB)
    write_json(workdir / "environment.json", {"env_id": "kites-1"})
    write_json(workdir / "canon-rules.json", {})
    write_json(workdir / "readers.json", [])
    # The colour is what a person says; the length on file is what only the world knows (D210).
    write_json(workdir / "user_facts.json",
               {"facts": [{"run_id": "r1", "field": "colour", "value": "teal1"}]})
    write_json(workdir / graph.FILE_NAME,
               graph.mine([_recording(n) for n in range(1, 6)], write_tools=["retie_kite"]))
    return workdir


def test_a_walk_is_run_in_the_rebuilt_world_and_becomes_a_task_with_a_verifier(kite_world: Path):
    body = synthesise.synthesise(kite_world, {"w1t3p1": 1})
    assert body["tasks"], body["counts"]
    row = body["tasks"][0]
    stored = json.loads((kite_world / synthesise.DIR / "tasks" / f"{row['task_id']}.json").read_text())
    assert stored["synthetic"] is True and stored["walk"]
    verifier = json.loads((kite_world / synthesise.DIR / "verifiers" / f"{row['task_id']}.json").read_text())
    assert verifier["atoms"], "the executed End state derived no claim about itself"
    assert row["writes"] == 1
    written = [atom for atom in verifier["atoms"] if (atom.get("target") or {}).get("kind") == "write"]
    assert written and any(atom["target"].get("tool") == "retie_kite" for atom in written), \
        "the Verifier was not derived from the End state the walk left"


def test_a_walk_the_world_refuses_is_discarded_and_counted(kite_world: Path):
    """Every kite is already tied at the only length the recordings show, so every write is refused."""
    db = json.loads(json.dumps(DB))
    for kite in db["kites"].values():
        kite["length"] = "99"
    write_json(kite_world / "db.json", db)
    body = synthesise.synthesise(kite_world, {"w1t3p1": 2})
    assert body["counts"]["walks_refused"] >= 1
    assert not body["tasks"], "a refused walk was kept anyway"


def test_one_seed_generates_the_same_tasks_twice(kite_world: Path):
    first = synthesise.synthesise(kite_world, {"w1t3p1": 2}, seed="one")
    again = synthesise.synthesise(kite_world, {"w1t3p1": 2}, seed="one")
    assert [row["task_id"] for row in first["tasks"]] == [row["task_id"] for row in again["tasks"]]
    assert [row["bucket"] for row in first["tasks"]] == [row["bucket"] for row in again["tasks"]]


def test_the_intent_states_no_value_only_the_world_knows(kite_world: Path):
    body = synthesise.synthesise(kite_world, {"w1t3p1": 1})
    stored = json.loads(
        (kite_world / synthesise.DIR / "tasks" / f"{body['tasks'][0]['task_id']}.json").read_text())
    text = str(stored["task"]["intent"])
    held = json.loads((kite_world / "db.json").read_text())["kites"]
    for row in held.values():
        assert str(row["length"]) not in text
        assert str(row["kite_id"]) not in text


def test_a_draft_that_states_a_record_value_is_refused_and_not_edited():
    said = [{"field": "colour", "value": "teal1"}]
    held = [{"field": "kite_id", "value": "k1"}]

    leaky = TestModel(["please retie k1 for me"])
    text, why = synthesise.intent_text(leaky, said, held, "the fallback")
    assert text == "the fallback" and "world" in why


def test_the_user_says_the_askable_values_and_holds_the_rest(kite_world: Path):
    calls = _recording(1)
    said, held = synthesise.facts_of(calls, synthesise.askable_fields(kite_world))
    assert {row["field"] for row in said} == {"colour"}
    assert "kite_id" in {row["field"] for row in held}
    rules = synthesise.user_rules_for(said)
    assert [fact.field for fact in rules.facts] == ["colour"]


def test_nothing_synthetic_joins_the_tasks_the_build_counts(kite_world: Path):
    from kullback import report

    write_json(kite_world / "tasks.json", {"tasks": []})
    write_json(kite_world / "task_status.json", {})
    synthesise.synthesise(kite_world, {"w1t3p1": 1})
    assert json.loads((kite_world / "tasks.json").read_text())["tasks"] == []
    assert json.loads((kite_world / "task_status.json").read_text()) == {}
    assert not (kite_world / "verifiers").exists(), "a synthetic Verifier landed where the Tasks' own live"
    counts = synthesise.counts_of(kite_world)
    assert counts["synthetic_tasks"] == 1
    assert "trusted" not in counts and "fidelity" not in counts
    data = report.load(kite_world)
    text = report.render(data)
    assert report.SYNTHETIC in text
    assert "walks tried" in text.lower() or "Walks tried" in text
    assert data.synthetic["tasks"], "the report read no synthetic store"
    # The generated Task is in its own section and not among the Tasks the build counts.
    assert data.synthetic["tasks"][0]["task_id"] not in {task.id for task in data.tasks}


def test_a_bucket_the_graph_cannot_reach_is_answered_with_what_it_did_reach(kite_world: Path):
    body = synthesise.synthesise(kite_world, {"w3t3p1": 1})
    for row in body["tasks"]:
        assert row["bucket_requested"] == "w3t3p1"
        assert row["bucket"] != "w3t3p1", "a band the world cannot fill was reported as filled"


def test_the_difficulty_command_fills_a_bucket_and_says_what_came_out(kite_world: Path):
    from typer.testing import CliRunner

    from kullback import cli

    result = CliRunner().invoke(cli.app, ["difficulty", "--workdir", str(kite_world),
                                          "--no-write", "--fill", "w1t3p1=1"])
    assert result.exit_code == 0, result.output
    assert "Synthetic Tasks" in result.output and "bucket asked" in result.output
    assert synthesise.read_index(kite_world)["tasks"]


def test_a_bucket_nobody_spells_that_way_is_refused_by_the_command(kite_world: Path):
    from typer.testing import CliRunner

    from kullback import cli

    result = CliRunner().invoke(cli.app, ["synthesise", "--workdir", str(kite_world),
                                          "--bucket", "wobble=2"])
    assert result.exit_code == 2 and "not a bucket" in result.output


# --- walks shaped to what the domain's own material attests (D225) ---------------------

def _archetype(**extra) -> dict:
    """One archetype of the kite world, as a domain reading would have left it in the store."""
    row = {"id": "arch_kite_1", "goal": "I want my kite tied at a different length",
           "preconditions": ["I know the colour of my kite"],
           "effects": ["my kite hangs at the length I asked for"], "constraints": [],
           "source": "https://help.kites.invalid/retying", "sources": ["https://help.kites.invalid/retying"],
           "tools": ["find_kite", "retie_kite"], "write_tools": ["retie_kite"],
           "effects_unrealised": []}
    row.update(extra)
    return row


def _read_domain(workdir: Path, *rows: dict) -> None:
    write_json(workdir / "domain" / "archetypes.json",
               {"format": 1, "archetypes": list(rows) or [_archetype()],
                "counts": {"archetypes_extracted": len(rows) or 1}})
    write_json(workdir / "domain" / "gaps.json", {"format": 1, "gaps": []})


def test_a_shaped_walk_visits_the_write_the_archetype_maps_onto(kite_world: Path):
    _read_domain(kite_world)
    body = synthesise.shape(kite_world, per_archetype=1)
    assert body["counts"]["tasks_shaped"] >= 1, body["counts"]
    row = body["tasks"][0]
    assert row["archetype"] == "arch_kite_1" and row["source"].startswith("https://")
    stored = json.loads((kite_world / synthesise.DIR / "tasks" / f"{row['task_id']}.json").read_text())
    assert "retie_kite" in [step["tool"] for step in stored["walk"]]


def test_a_shaped_walk_takes_only_edges_the_recordings_showed(kite_world: Path):
    _read_domain(kite_world)
    body = synthesise.shape(kite_world, per_archetype=1)
    edges = {(edge["from"], edge["to"])
             for edge in json.loads((kite_world / graph.FILE_NAME).read_text())["edges"]}
    stored = json.loads(
        (kite_world / synthesise.DIR / "tasks" / f"{body['tasks'][0]['task_id']}.json").read_text())
    steps = [step["tool"] for step in stored["walk"]]
    for before, after in zip(steps, steps[1:], strict=False):
        assert any(src == before or src in steps[:steps.index(after)] for src, to in edges if to == after)


def test_an_archetype_mapping_onto_no_write_is_never_shaped(kite_world: Path):
    _read_domain(kite_world, _archetype(id="arch_kite_2", write_tools=[], tools=["get_kite"]))
    body = synthesise.shape(kite_world, per_archetype=1)
    assert body["counts"]["tasks_shaped"] == 0 and body["counts"]["archetypes_read"] == 0


def test_the_read_only_band_is_refused_with_a_reason(kite_world: Path):
    _read_domain(kite_world)
    body = synthesise.shape(kite_world, bucket="w0t2p1")
    assert body["tasks"] == [] and "Starting state" in body["refused"]
    plain = synthesise.synthesise(kite_world, {"w0t2p1": 2})
    assert plain["tasks"] == [] and plain["bands_refused"]["w0t2p1"]


def test_a_judge_rejection_removes_a_task_and_is_counted(kite_world: Path):
    _read_domain(kite_world)
    rejection = json.dumps({"reject": True, "citation": "my kite hangs at the length I asked for"})
    rejecting = [TestModel([rejection], loop=True), TestModel([rejection], loop=True)]
    body = synthesise.shape(kite_world, per_archetype=1, judges=rejecting)
    assert body["counts"]["tasks_shaped"] == 0
    assert body["counts"]["fell"]["plausible"] >= 1

    passing = [TestModel([json.dumps({"reject": False})], loop=True) for _ in range(2)]
    judged = synthesise.shape(kite_world, per_archetype=1, seed="one", judges=passing)
    alone = synthesise.shape(kite_world, per_archetype=1, seed="one")
    assert judged["counts"]["tasks_shaped"] == alone["counts"]["tasks_shaped"], \
        "a judge that rejects nothing adds no task"


def test_the_report_carries_the_archetype_counts(kite_world: Path):
    from kullback import report

    _read_domain(kite_world)
    synthesise.shape(kite_world, per_archetype=1)
    counts = synthesise.shaped_counts(kite_world)
    assert counts["tasks_shaped"] >= 1 and counts["shaped_per_source"]
    text = report.render(report.load(kite_world))
    assert "archetypes extracted" in text and "Tasks shaped" in text
    assert "realism rung" in text


# --- binding fixes: list spellings agree, writes name rows the walk read (D360) -----------


def _world_and_graph(kite_world: Path):
    world = synthesise.World(kite_world)
    body = json.loads((kite_world / graph.FILE_NAME).read_text())
    sigs = {sig.name: sig for sig in world.sigs}
    return world, body, sigs


def test_a_list_argument_binds_from_values_stored_under_the_marked_spelling(kite_world: Path):
    world, _body, sigs = _world_and_graph(kite_world)
    sigs = dict(sigs, retie_kite=_sig("retie_kite", "write", "kite_id", "handle"))
    answers = [{"kite_id": "k1"}]
    step = {"tool": "retie_kite",
            "bindings": [{"arg": "kite_id", "column": "kite_id", "from": 0}]}
    observed = {"retie_kite": {"handle[]": {"values": ["h7"]}}}
    args = synthesise.bind(world, step, answers, observed, sigs, "t", is_write=True)
    assert args == {"kite_id": "k1", "handle": "h7"}


def test_a_write_names_the_row_the_walk_read_and_not_the_recorded_pool(kite_world: Path):
    world, _body, sigs = _world_and_graph(kite_world)
    answers = [{"kite_id": "k1", "colour": "teal1"}]
    step = {"tool": "retie_kite", "bindings": []}
    observed = {"retie_kite": {"kite_id": {"values": ["k9"]}, "length": {"values": ["77"]}}}
    args = synthesise.bind(world, step, answers, observed, sigs, "t", is_write=True)
    assert args["kite_id"] == "k1"


def test_a_write_naming_a_row_nothing_read_is_unbound_despite_the_pool(kite_world: Path):
    world, _body, sigs = _world_and_graph(kite_world)
    step = {"tool": "retie_kite", "bindings": []}
    observed = {"retie_kite": {"kite_id": {"values": ["k9"]}, "length": {"values": ["77"]}}}
    assert synthesise.bind(world, step, [], observed, sigs, "t", is_write=True) is None


def test_a_write_with_no_row_to_name_pulls_in_the_read_the_graph_feeds_it_from(kite_world: Path):
    from kullback.sampling import sample_key

    world, _body, _sigs = _world_and_graph(kite_world)
    keys = ["k1", "k2", "k3", "k4", "k5"]
    colour = keys[sample_key("synth-row", "ins-0:0:ins1:kites", "") % 5]
    length = keys[sample_key("synth-row", "ins-0:0:kites", "") % 5]
    assert (colour, length) == ("k5", "k3"), "the draws moved; pick another ident"
    direct = [{"name": "find_kite", "args": {"colour": "teal1"}, "result": {"kite_id": "k1"},
               "error": None},
              {"name": "retie_kite", "args": {"kite_id": "k1", "length": "99"},
               "result": {"kite_id": "k1"}, "error": None}]
    body = graph.mine([direct], write_tools=["retie_kite"])
    calls, why = synthesise.run_walk(world, [{"tool": "retie_kite", "bindings": []}],
                                      body["args"], "ins-0", graph=body,
                                      write_tools=["retie_kite"])
    assert why == "", why
    assert [call["name"] for call in calls] == ["find_kite", "retie_kite"]
    assert calls[1]["args"]["kite_id"] == calls[0]["result"]["kite_id"]
    assert calls[1]["args"]["length"] == "33"


def test_a_write_with_no_feeding_read_is_unbound(kite_world: Path):
    world, _body, _sigs = _world_and_graph(kite_world)
    calls, why = synthesise.run_walk(world, [{"tool": "retie_kite", "bindings": []}], {},
                                      "unbound-t", graph={"edges": []},
                                      write_tools=["retie_kite"])
    assert why == synthesise.UNBOUND and calls == []


def test_a_refused_write_is_retried_with_a_new_draw(kite_world: Path):
    from kullback.sampling import sample_key

    world, _body, _sigs = _world_and_graph(kite_world)
    keys = ["k1", "k2", "k3", "k4", "k5"]
    colour = keys[sample_key("synth-row", "retry-7:0:kites", "") % 5]
    first = keys[sample_key("synth-row", "retry-7:1:kites", "") % 5]
    second = keys[sample_key("synth-row", "retry-7:1:r1:kites", "") % 5]
    assert (colour, first, second) == ("k2", "k2", "k4"), "the draws moved; pick another ident"
    steps = [{"tool": "find_kite", "bindings": []},
             {"tool": "retie_kite",
              "bindings": [{"arg": "kite_id", "column": "kite_id", "from": 0}]}]
    note: dict = {}
    calls, why = synthesise.run_walk(world, steps, {}, "retry-7",
                                      graph={"edges": []}, write_tools=["retie_kite"], note=note)
    assert why == "", why
    assert [call["name"] for call in calls] == ["find_kite", "retie_kite"]
    assert calls[1]["args"] == {"kite_id": "k2", "length": "34"}
    assert set(note["refused"]) == {"k2", "32"}


# --- the synthetic store: a description over a kept walk, chain unchanged (D360) -----------


class _EchoModel:
    """A test double for the description writer only: it states back the values it is given."""

    def __init__(self):
        self.calls = []

    def query(self, messages, tools=None, config=None):
        from kullback.ai.provider import ModelReply

        self.calls.append(messages)
        values = []
        for line in messages[0]["content"].splitlines():
            if line.startswith("- ") and ": " in line:
                value = line.split(": ", 1)[1].strip()
                if value and value != "nothing" and value not in values:
                    values.append(value)
        return ModelReply(content="Please sort this out: " + ", ".join(values)
                                  + ". You have my go-ahead.")


def _kept_store(kite_world: Path):
    """One store Task built with the echoing description writer, with its task id and run id."""
    model = _EchoModel()
    body = synthesise.build_store(kite_world, {"w1t3p1": 1}, seed="store-e2e", model=model)
    assert body["counts"]["tasks_kept"] == 1, body["counts"]
    assert model.calls, "the description was never written by the model"
    store = kite_world / synthesise.DIR / synthesise.STORE
    return store, body["tasks"][0]["task_id"], body["tasks"][0]["run_id"]


def test_build_store_keeps_a_described_walk_apart_from_the_d224_store(kite_world: Path):
    from kullback.spec import intent_tools

    store, task_id, _run_id = _kept_store(kite_world)
    assert (store / intent_tools.SYNTHETIC_MARKER).is_file()
    assert (store / "db.json").is_symlink(), "the database must be linked, never copied"
    assert not (store / "verifiers").exists(), "the store path never derives a Verifier"
    assert not (kite_world / synthesise.DIR / "tasks").exists(), \
        "the D224 store stays as it is"
    index = json.loads((store / synthesise.STORE_INDEX).read_text())
    assert index["counts"]["tasks_kept"] == 1 and index["counts"]["per_bucket"] == {"w1t3p1": 1}


def test_a_store_task_reads_through_the_chain_unchanged(kite_world: Path):
    from kullback.runner.records import Task as TaskRecord
    from kullback.spec import intent_tools, trust
    from kullback.spec.schema import SpecIntent
    from kullback.spec.writer import load_inputs

    store, task_id, run_id = _kept_store(kite_world)
    task = TaskRecord.model_validate(json.loads((store / "tasks" / f"{task_id}.json").read_text()))
    assert task.intent and task.facts_in_instruction is False
    found = intent_tools.recordings_of(store, task_id)
    assert [turn.text for turn in found.recordings["description"]] == [task.intent]
    inputs = load_inputs(store, task_id, intent=SpecIntent(task_id=task_id, text=task.intent))
    assert inputs.tools and inputs.state is not None
    assert task_id in trust.intent_tiers(store)
    assert trust.faithful_references(store)[task_id] == [run_id] == task.run_ids


def test_the_synthesise_command_builds_a_store_with_the_store_flag(kite_world: Path):
    from typer.testing import CliRunner

    from kullback import cli

    result = CliRunner().invoke(cli.app, ["synthesise", "--workdir", str(kite_world),
                                          "--bucket", "w1t3p1=1", "--store"])
    assert result.exit_code == 0, result.output
    assert "kept" in result.output
    assert (kite_world / synthesise.DIR / synthesise.STORE / synthesise.STORE_INDEX).is_file()


def _id_first_walk() -> list[dict]:
    """A walk whose first call names a row by id: no earlier answer carried it."""
    return [{"name": "get_kite", "args": {"kite_id": "k2"},
             "result": {"kite_id": "k2", "colour": "teal2", "spool_id": "s2", "length": "32"},
             "error": None},
            {"name": "retie_kite", "args": {"kite_id": "k2", "length": "40"},
             "result": {"kite_id": "k2", "length": "40"}, "error": None}]


def test_an_id_no_answer_carried_is_stated_in_the_description_and_the_walk_kept(kite_world: Path):
    store = synthesise._link_store(kite_world)
    world = synthesise.World(store)
    references: dict = {}
    row, drop = synthesise._store_task(world, kite_world, store, _id_first_walk(), {}, "w1t2p1",
                                       "id-first", _EchoModel(), [], False, references, {}, {})
    assert drop == "" and row is not None
    task = json.loads((store / "tasks" / f"{row['task_id']}.json").read_text())
    assert "k2" in task["intent"], "the id the customer gives never reached the description"


def test_a_walk_naming_an_id_the_description_leaves_out_fails_the_fairness_check():
    from kullback.runner.world import birth

    calls = _id_first_walk()
    walk = [{"args": call["args"]} for call in calls]
    results = [{"id": f"call-{n + 1}", "result": call["result"]} for n, call in enumerate(calls)]
    held = birth.check_arguments_trace(walk=walk, results=results,
                                       intent="Please retie kite k2 at 40.", user_facts=[])
    dropped = birth.check_arguments_trace(walk=walk, results=results,
                                          intent="Please retie my kite at 40.", user_facts=[])
    assert held.outcome == birth.HELD
    assert dropped.outcome == birth.FAILED and [item.value for item in dropped.untraced] == ["k2"]


def test_the_description_never_states_an_id_an_earlier_answer_carried(kite_world: Path):
    walk = [{"name": "find_kite", "args": {"colour": "teal5"}, "result": {"kite_id": "k5"},
             "error": None},
            {"name": "retie_kite", "args": {"kite_id": "k5", "length": "32"},
             "result": {"kite_id": "k5", "length": "32"}, "error": None}]
    facts, ask, _refusals = synthesise._description_inputs(walk, [], {"retie_kite"})
    assert [row["value"] for row in facts] == ["teal5", "32"]
    assert ask == [{"field": "length", "value": "32"}]
