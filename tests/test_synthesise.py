"""Synthetic Tasks over an invented world: bound, run, derived and counted apart (D224)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback import graph, synthesise
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


def test_the_end_state_the_walk_left_is_what_the_verifier_was_derived_from(kite_world: Path):
    body = synthesise.synthesise(kite_world, {"w1t3p1": 1})
    row = body["tasks"][0]
    verifier = json.loads((kite_world / synthesise.DIR / "verifiers" / f"{row['task_id']}.json").read_text())
    written = [atom for atom in verifier["atoms"] if (atom.get("target") or {}).get("kind") == "write"]
    assert written and any(atom["target"].get("tool") == "retie_kite" for atom in written)


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

    class Leaky:
        def query(self, messages, tools=None, config=None):
            return type("Reply", (), {"content": "please retie k1 for me"})()

    text, why = synthesise.intent_text(Leaky(), said, held, "the fallback")
    assert text == "the fallback" and "world" in why


def test_the_user_says_the_askable_values_and_holds_the_rest(kite_world: Path):
    calls = _recording(1)
    said, held = synthesise.facts_of(calls, synthesise.askable_fields(kite_world))
    assert {row["field"] for row in said} == {"colour"}
    assert "kite_id" in {row["field"] for row in held}
    rules = synthesise.user_rules_for(said)
    assert [fact.field for fact in rules.facts] == ["colour"]


def test_nothing_synthetic_joins_the_tasks_the_build_counts(kite_world: Path):
    write_json(kite_world / "tasks.json", {"tasks": []})
    write_json(kite_world / "task_status.json", {})
    synthesise.synthesise(kite_world, {"w1t3p1": 1})
    assert json.loads((kite_world / "tasks.json").read_text())["tasks"] == []
    assert json.loads((kite_world / "task_status.json").read_text()) == {}
    assert not (kite_world / "verifiers").exists(), "a synthetic Verifier landed where the Tasks' own live"
    counts = synthesise.counts_of(kite_world)
    assert counts["synthetic_tasks"] == 1
    assert "trusted" not in counts and "fidelity" not in counts


def test_a_bucket_the_graph_cannot_reach_is_answered_with_what_it_did_reach(kite_world: Path):
    body = synthesise.synthesise(kite_world, {"w3t3p1": 1})
    for row in body["tasks"]:
        assert row["bucket_requested"] == "w3t3p1"
        assert row["bucket"] != "w3t3p1", "a band the world cannot fill was reported as filled"


def test_the_report_shows_the_synthetic_rows_under_their_own_heading(kite_world: Path):
    from kullback import report

    synthesise.synthesise(kite_world, {"w1t3p1": 1})
    data = report.load(kite_world)
    text = report.render(data)
    assert report.SYNTHETIC in text
    assert "walks tried" in text.lower() or "Walks tried" in text
    assert data.synthetic["tasks"], "the report read no synthetic store"
    # The generated Task is in its own section and not among the Tasks the build counts.
    assert data.synthetic["tasks"][0]["task_id"] not in {task.id for task in data.tasks}


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
