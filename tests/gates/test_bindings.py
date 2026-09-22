"""Tests for kullback/gates/bindings.py: path-bound gates whose refusals name rows.

The world here is an invented widget shelf: widgets standing in rows, each with a
status, and one write tool that moves a widget to a new status. No name here is a
customer name.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import PTR
from gates.examiner_fixtures import (
    SIGS,
    TASK,
    base,
    history,
    replay_row,
    reroll_row,
    tighten,
    version,
)
from gates.verifier_fixtures import (
    WRITE_TOOLS,
    alt_path_run,
    derive,
    extra_write_run,
    other_reason_run,
    reference_run,
    wrong_run,
)
from kullback.builder import compile_env as ce
from kullback.builder import sandbox as sb
from kullback.gates import verifier_suite as S
from kullback.gates.bindings import BINDINGS, binding_for, rows_for, rulings_for
from kullback.gates.loosening import false_rejection_gate, loosening_gate
from kullback.gates.tool_runs import MEMORISED_STAGE, SensitivityPair
from kullback.runner.records import (
    Column,
    Constraint,
    EntitySchema,
    FieldStat,
    ToolCall,
    ToolSig,
    as_dict,
)

DB = {
    "widgets": {
        "W-1": {"widget_id": "W-1", "status": "new"},
        "W-2": {"widget_id": "W-2", "status": "new"},
    },
}

RIGHT = ("row = self.db.widgets[widget_id]\n"
         "row.status = new_status\n"
         "return {\"widget_id\": row.widget_id, \"status\": row.status}\n")
WRONG_ON_W2 = ("row = self.db.widgets[widget_id]\n"
               "row.status = new_status\n"
               "if widget_id == \"W-2\":\n"
               "    return {\"widget_id\": row.widget_id, \"status\": \"new\"}\n"
               "return {\"widget_id\": row.widget_id, \"status\": row.status}\n")


def _schema() -> EntitySchema:
    columns = [Column(table="widgets", name=name, **{"class": "hard"}) for name in ("widget_id", "status")]
    return EntitySchema(tables=["widgets"], columns=columns, id_patterns={"widgets.widget_id": r"^W-\d+$"})


def _sig() -> ToolSig:
    return ToolSig(name="set_status", description="Move one widget to a new status.",
                   args_fields=[FieldStat(name="widget_id", types=["str"], optional=False),
                                FieldStat(name="new_status", types=["str"], optional=False)],
                   kind="write", unclassified=False)


def _call(call_id: str, widget: str, recorded: str) -> ToolCall:
    return ToolCall(id=call_id, name="set_status", args={"widget_id": widget, "new_status": "done"},
                    result={"widget_id": widget, "status": recorded}, raw_ptr=PTR)


CALLS = [_call("c1", "W-1", "done"), _call("c2", "W-2", "done")]


def _source(body: str) -> str:
    return ce.module_source(_schema(), [_sig()], {"set_status": body})


def _execute(box_dir: Path):
    def run(source: str, calls: list[ToolCall], db: dict) -> list[dict]:
        return sb.Sandbox(source, db, box_dir).run(list(calls))

    return run


def _write(root: Path, rel: str, text: str) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return rel


def _evidence(**overrides) -> dict:
    evidence: dict = {"calls": CALLS, "schema": _schema(), "db": dict(DB)}
    evidence.update(overrides)
    return evidence


# --- the bindings themselves ---

def test_tools_writes_draw_the_nine_body_gates_in_order():
    binding = binding_for("tools/set_status.py")
    assert binding is not None
    assert binding.gates == ("confined", "parses", "executes_on_s0", "deterministic", "non_trivial",
                             "sensitivity", "replay_fidelity", "refuses_unknown", MEMORISED_STAGE)


def test_verifier_refusal_intent_and_policy_writes_each_draw_their_gates():
    assert binding_for("verifiers/t1.json").gates == (
        "gate_a_oracle_replay", "verifier_suite", "loosening", "false_rejection", "trusted")
    assert binding_for("refusals/t1.json").gates == ("refuse",)
    assert binding_for("intents/t1.json").gates == ("intent",)
    assert binding_for("policy/predicates.py").gates == ("compile_policy.confined",)


def test_a_write_outside_every_binding_draws_no_ruling(tmp_path):
    _write(tmp_path / "root", "notes/scratch.py", "x = 1\n")
    assert rulings_for(tmp_path / "root", "notes/scratch.py", tmp_path / "work") == []
    assert binding_for("notes/scratch.py") is None


# --- acceptance: one wrong answer names its call and its column ---

def test_a_body_answering_one_recorded_call_wrongly_is_refused_at_replay_with_rows(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    rel = _write(root, "tools/set_status.py", _source(WRONG_ON_W2))
    rulings = rulings_for(root, rel, work, execute=_execute(tmp_path / "box"), evidence=_evidence())
    stages = [ruling.stage for ruling in rulings]
    assert stages[:5] == ["confined", "parses", "executes_on_s0", "deterministic", "non_trivial"]
    refused = next(ruling for ruling in rulings if not ruling.passed)
    assert refused.stage == "replay_fidelity"
    assert refused.rows and len(refused.rows) == 1
    row = refused.rows[0]
    assert row["call"] == "c2" and row["column"] == "status"
    assert row["recorded"]["status"] == "done" and row["ours"]["status"] == "new"
    assert row["tool"] == "set_status" and row["args"]


def test_a_body_answering_every_call_right_passes_replay(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    rel = _write(root, "tools/set_status.py", _source(RIGHT))
    rulings = rulings_for(root, rel, work, execute=_execute(tmp_path / "box"), evidence=_evidence())
    assert [ruling.stage for ruling in rulings if not ruling.passed] == []


def test_a_write_read_off_the_workdir_files_rules_the_same_way(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    rel = _write(root, "tools/set_status.py", _source(WRONG_ON_W2))
    (work / "traces").mkdir(parents=True)
    (work / "traces" / "t1.json").write_text(
        json.dumps({"tool_calls": [as_dict(call) for call in CALLS]}), encoding="utf-8")
    (work / "schema.json").write_text(json.dumps(as_dict(_schema())), encoding="utf-8")
    (work / "db.json").write_text(json.dumps(DB), encoding="utf-8")
    rulings = rulings_for(root, rel, work, execute=_execute(tmp_path / "box"))
    refused = next(ruling for ruling in rulings if not ruling.passed)
    assert refused.stage == "replay_fidelity"
    assert refused.rows and refused.rows[0]["call"] == "c2"


# --- every refusal names its rows ---

def _refusals(root: Path, work: Path, rel: str, **kwargs) -> list:
    rulings = rulings_for(root, rel, work, **kwargs)
    return [ruling for ruling in rulings if not ruling.passed]


def test_every_refusal_names_its_rows(tmp_path):
    schema, db = _schema(), dict(DB)
    pair = SensitivityPair(first=CALLS[0], second=CALLS[1], columns=("status",), tasks=("t1", "t2"))
    same = [{"ok": True, "value": {"widget_id": "W-1", "status": "done"}}]
    varied = [_call("c1", "W-1", "done"), _call("c2", "W-2", "held")]
    cases = {
        "tools/confined.py": ({"source": "class DomainTools:\n"
                                          "    def set_status(self, widget_id, new_status):\n"
                                          "        return os.getcwd()\n"}, {}),
        "tools/parses.py": ({"source": "def broken(:"}, {}),
        "tools/executes.py": ({"source": "x = 1\n", "calls": CALLS},
                              {"execute": lambda s, c, d: [
                                  {"ok": False, "error": "NameError", "message": "boom"} for _ in c]}),
        "tools/deterministic.py": ({"source": "x = 1\n", "calls": CALLS},
                                  {"execute": _flip_flop()}),
        "tools/non_trivial.py": ({"source": "x = 1\n", "calls": varied},
                                 {"execute": lambda s, c, d: [
                                     {"ok": True, "value": {"status": "done"}} for _ in c]}),
        "tools/sensitivity.py": ({"source": "x = 1\n", "pairs": [pair], "first": same, "second": same,
                                  "schema": schema}, {}),
        "tools/refuses.py": ({"source": "x = 1\n",
                              "unknown_probes": [("widget_id", "W-9", CALLS[0])]},
                             {"execute": lambda s, c, d: [
                                 {"ok": True, "value": {"widget_id": "W-9"}} for _ in c]}),
        "tools/memorised.py": ({"source": "hold = \"W-1\"\nx = 1\n", "schema": schema, "db": db,
                                "calls": CALLS}, {}),
        "policy/predicates.py": ({"source": "import os\n"}, {}),
        "intents/t1.json": ({"document": {"t1": {"grounded": False, "reason": "no verdict"}}}, {}),
        "refusals/t1.json": ({"document": {"t1": {"reason": "no path"}},
                              "replays": {"t1": {"a": {"confirmed": True, "run_id": "r1"}}},
                              "rerolls": {}}, {}),
        "verifiers/oracle.json": ({"document": {"verifier": "v1"},
                                   "replays": [{"run_id": "r1", "held_out": False,
                                                "writes": [{"expected": {"a": 1},
                                                            "actual": {"a": 2}}]}]}, {}),
    }
    for rel, (made, kwargs) in cases.items():
        root, work = tmp_path / "root", tmp_path / "work"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".json"):
            path.write_text(json.dumps(made.pop("document", {})), encoding="utf-8")
        else:
            path.write_text(made.pop("source", "x = 1\n"), encoding="utf-8")
        refused = _refusals(root, work, rel, evidence=made, **kwargs)
        assert refused, rel
        for ruling in refused:
            assert ruling.rows, (rel, ruling.stage)


def _flip_flop():
    state = {"n": 0}

    def run(source: str, calls: list, db: dict) -> list[dict]:
        state["n"] += 1
        return [{"ok": True, "value": {"n": state["n"]}} for _ in calls]

    return run


def test_the_memorised_refusal_names_the_literal_and_what_it_matched(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    rel = _write(root, "tools/memorised.py", "hold = \"W-1\"\nx = 1\n")
    refused = _refusals(root, work, rel, evidence={"schema": _schema(), "db": dict(DB), "calls": CALLS})
    assert [ruling.stage for ruling in refused] == [MEMORISED_STAGE]
    assert refused[0].rows[0]["literal"] == "'W-1'"
    assert "widgets" in refused[0].rows[0]["match"]


def test_the_refusal_names_the_runs_that_finished(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    rel = _write(root, "refusals/t1.json", json.dumps({"t1": {"reason": "no path"}}))
    refused = _refusals(root, work, rel,
                        evidence={"replays": {"t1": {"a": {"confirmed": True, "run_id": "r1"}}},
                                  "rerolls": {}})
    assert refused and refused[0].stage == "refuse"
    assert refused[0].rows[0] == {"task": "t1", "finished": ["r1"]}


def test_the_verifier_suite_refusal_names_the_check_and_the_atom(tmp_path):
    rule = Constraint(id="k1", text="never cancel an order that is not pending", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return pre_state['orders']['#W123']['status'] == 'pending'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    results = S.validate_verifier(verifier, reference_run())
    oracle = next(result for result in results if result.stage == "verifier_oracle")
    assert not oracle.passed
    rows = rows_for(oracle, {"verifier": verifier})
    assert rows and rows[0]["check"] == "verifier_oracle"
    assert rows[0]["atom"] and rows[0]["text"]

    root, work = tmp_path / "root", tmp_path / "work"
    rel = _write(root, "verifiers/t1.json", json.dumps({"verifier": "v1"}))
    refused = _refusals(root, work, rel,
                        evidence={"verifier": verifier, "reference": reference_run(),
                                  "write_tools": set(WRITE_TOOLS)})
    assert refused
    for ruling in refused:
        assert ruling.rows, ruling.stage


def _suite_evidence(verifier, **overrides) -> dict:
    """What a verifiers write carries when its Task has a Reference: every D79 input, check 6 on a
    hand-built probe Run, and a status row whose suite did not pass."""
    evidence = {"reference": reference_run(), "alt_path_run": alt_path_run(), "seed_runs": [alt_path_run()],
                "wrong_run": wrong_run(), "write_tools": set(WRITE_TOOLS), "probe_model": object(),
                "run_probe": lambda model, ruled: wrong_run(), "verifiers": [verifier],
                "history": history(version(verifier)),
                "task_status": {verifier.task_id: {"verifier_passed": False}}}
    evidence.update(overrides)
    return evidence


def test_a_verifiers_write_with_a_reference_draws_the_suite_rulings(tmp_path):
    verifier = derive(tmp_path)
    root = tmp_path / "root"
    rel = _write(root, "verifiers/t1.json", json.dumps(as_dict(verifier)))
    out = rulings_for(root, rel, tmp_path / "work", evidence=_suite_evidence(verifier))
    assert [r.stage for r in out if r.stage in S.D79_STAGES] == list(S.D79_STAGES)


def test_the_trusted_ruling_reads_the_suite_that_just_ran_not_the_status_row(tmp_path):
    verifier = derive(tmp_path)
    root = tmp_path / "root"
    rel = _write(root, "verifiers/t1.json", json.dumps(as_dict(verifier)))
    out = rulings_for(root, rel, tmp_path / "work", evidence=_suite_evidence(verifier))
    trusted = next(r for r in out if r.stage == "trusted")
    assert not any("suite" in failure for failure in trusted.failures), trusted.failures

    no_suite = rulings_for(root, rel, tmp_path / "work", evidence=_suite_evidence(verifier, reference=None))
    trusted = next(r for r in no_suite if r.stage == "trusted")
    assert any("D79 suite did not pass" in failure for failure in trusted.failures), trusted.failures


def test_loosening_and_false_rejection_refusals_name_their_runs(tmp_path):
    strict, plain = tighten(base(tmp_path)), base(tmp_path)
    runs = {TASK: [reference_run(), extra_write_run(), other_reason_run()]}
    replays = {TASK: {"tr1": replay_row("tr1", True, run_id="ref")}}
    unfinished = {TASK: [reroll_row("rr2", "max_steps")]}
    hist = history(version(strict), version(plain, 2, by="repair", accepted=False))
    out = loosening_gate(hist, runs, replays, unfinished, None, SIGS)
    assert not out.passed
    rows = rows_for(out, {})
    assert rows and rows[0] == {"task": TASK, "run": "rr2"}

    strict = tighten(base(tmp_path)).model_copy(update={"seed_run_ids": ["ref"]})
    out = false_rejection_gate([strict], {TASK: [reference_run(), other_reason_run()]},
                               {TASK: {"tr1": replay_row("tr1", True, run_id="ref")}},
                               {TASK: [reroll_row("rr2", "success")]}, None, SIGS, task_status={})
    assert not out.passed
    rows = rows_for(out, {})
    assert rows and {row["run"] for row in rows if "run" in row} == {"rr2"}


def test_every_bound_gate_lives_in_the_registry():
    from kullback.gates import GATES

    names = {spec.name for spec in GATES}
    for binding in BINDINGS:
        for gate in binding.gates:
            assert gate in names, gate
    assert len(GATES) == 43
