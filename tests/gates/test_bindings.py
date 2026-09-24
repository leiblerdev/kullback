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
    write_events_jsonl,
    wrong_run,
)
from kullback.builder import compile_env as ce
from kullback.builder import sandbox as sb
from kullback.gates import verifier_suite as S
from kullback.gates.bindings import BINDINGS, binding_for, load_trace_calls, rows_for, rulings_for
from kullback.gates.loosening import false_rejection_gate, loosening_gate
from kullback.gates.tool_runs import MEMORISED_STAGE, SensitivityPair
from kullback.runner.canon import CanonRules
from kullback.runner.records import (
    Column,
    Constraint,
    EntitySchema,
    FieldStat,
    RawPtr,
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

def test_tools_writes_draw_the_static_body_gates_before_the_execution_gates():
    binding = binding_for("tools/set_status.py")
    assert binding is not None
    assert binding.gates == ("confined", "parses", MEMORISED_STAGE, "executes_on_s0", "deterministic",
                             "non_trivial", "sensitivity", "replay_fidelity", "refuses_unknown")


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

def test_a_body_answering_every_call_right_passes_and_one_answering_a_call_wrongly_is_refused_with_rows(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    right = _write(root, "tools/right.py", _source(RIGHT))
    rulings = rulings_for(root, right, work, execute=_execute(tmp_path / "box_right"), evidence=_evidence())
    assert [ruling.stage for ruling in rulings if not ruling.passed] == []
    rel = _write(root, "tools/set_status.py", _source(WRONG_ON_W2))
    rulings = rulings_for(root, rel, work, execute=_execute(tmp_path / "box"), evidence=_evidence())
    stages = [ruling.stage for ruling in rulings]
    assert stages[:6] == ["confined", "parses", MEMORISED_STAGE, "executes_on_s0", "deterministic",
                          "non_trivial"]
    refused = next(ruling for ruling in rulings if not ruling.passed and ruling.stage != MEMORISED_STAGE)
    assert refused.stage == "replay_fidelity"
    assert refused.rows and len(refused.rows) == 1
    row = refused.rows[0]
    assert row["call"] == "c2" and row["column"] == "status"
    assert row["recorded"]["status"] == "done" and row["ours"]["status"] == "new"
    assert row["tool"] == "set_status" and row["args"]


def test_a_write_read_off_the_workdir_files_rules_the_same_way(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    rel = _write(root, "tools/set_status.py", _source(WRONG_ON_W2))
    (work / "traces").mkdir(parents=True)
    (work / "traces" / "t1.json").write_text(
        json.dumps({"tool_calls": [as_dict(call) for call in CALLS]}), encoding="utf-8")
    (work / "schema.json").write_text(json.dumps(as_dict(_schema())), encoding="utf-8")
    (work / "db.json").write_text(json.dumps(DB), encoding="utf-8")
    rulings = rulings_for(root, rel, work, execute=_execute(tmp_path / "box"))
    refused = next(ruling for ruling in rulings if not ruling.passed and ruling.stage != MEMORISED_STAGE)
    assert refused.stage == "replay_fidelity"
    assert refused.rows and refused.rows[0]["call"] == "c2"


# --- static gates rule first, together (F41) ---

def test_a_body_with_a_memorised_literal_and_a_replay_difference_draws_both_refusals_memorised_first(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    rel = _write(root, "tools/set_status.py", _source(WRONG_ON_W2))
    rulings = rulings_for(root, rel, work, execute=_execute(tmp_path / "box"), evidence=_evidence())
    refused = [ruling.stage for ruling in rulings if not ruling.passed]
    assert refused == [MEMORISED_STAGE, "replay_fidelity"]


def test_a_body_that_does_not_parse_draws_every_static_ruling_and_no_execution_ruling(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    rel = _write(root, "tools/set_status.py", "def broken(:\n")
    ran = []

    def execute(source: str, calls: list, db: dict) -> list[dict]:
        ran.append(source)
        return [{"ok": True, "value": {}} for _ in calls]

    rulings = rulings_for(root, rel, work, execute=execute, evidence=_evidence())
    assert "parses" in [ruling.stage for ruling in rulings if not ruling.passed]
    assert all(ruling.stage in ("confined", "parses", MEMORISED_STAGE) for ruling in rulings)
    assert ran == []


# --- the trace table reaches an executor that takes it (F50) ---

def _traced_call(call_id: str, trace: str, at: int, name: str, args: dict, result: dict) -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args, result=result, trace_id=trace,
                    raw_ptr=RawPtr(file_hash="f", sim_index=0, msg_index=at))


def test_an_executor_taking_the_trace_table_is_handed_every_traced_call_of_the_workdir(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    rel = _write(root, "tools/set_status.py", _source(RIGHT))
    first = _traced_call("a1", "t1", 1, "read_widget", {"widget_id": "W-1"}, {"widget_id": "W-1"})
    second = _traced_call("a2", "t1", 2, "set_status", {"widget_id": "W-1", "new_status": "done"},
                          {"widget_id": "W-1", "status": "done"})
    (work / "traces").mkdir(parents=True)
    (work / "traces" / "t1.json").write_text(
        json.dumps({"tool_calls": [as_dict(call) for call in (first, second)]}), encoding="utf-8")
    handed = []

    def execute(source: str, calls: list, db: dict, trace_calls=None) -> list[dict]:
        handed.append(trace_calls)
        return sb.Sandbox(source, DB, tmp_path / "box", trace_calls=trace_calls).run(list(calls))

    rulings_for(root, rel, work, execute=execute, evidence={"db": dict(DB), "schema": _schema()})
    assert handed
    assert {call.id for call in handed[0]["t1"]} == {"a1", "a2"}


def test_a_workdir_with_only_calls_files_yields_one_table_across_tools_in_recorded_order(tmp_path):
    work = tmp_path / "work"
    (work / "env" / "calls").mkdir(parents=True)
    read = _traced_call("b1", "t1", 1, "read_widget", {"widget_id": "W-1"}, {"widget_id": "W-1"})
    write = _traced_call("b2", "t1", 4, "set_status", {"widget_id": "W-1", "new_status": "done"},
                         {"widget_id": "W-1", "status": "done"})
    again = _traced_call("b3", "t1", 7, "read_widget", {"widget_id": "W-1"}, {"widget_id": "W-1"})
    for name, calls in (("read_widget", [again, read]), ("set_status", [write])):
        (work / "env" / "calls" / f"{name}.jsonl").write_text(
            "".join(json.dumps(as_dict(call)) + "\n" for call in calls), encoding="utf-8")
    table = load_trace_calls(work, "tools/set_status.py")
    assert list(table) == ["t1"]
    assert [call.id for call in table["t1"]] == ["b1", "b2", "b3"]


def test_an_executor_without_the_trace_table_keyword_is_called_as_before(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    rel = _write(root, "tools/set_status.py", _source(RIGHT))
    second = _traced_call("a2", "t1", 2, "set_status", {"widget_id": "W-1", "new_status": "done"},
                          {"widget_id": "W-1", "status": "done"})
    rulings = rulings_for(root, rel, work, execute=_execute(tmp_path / "box"),
                          evidence=_evidence(calls=[second], trace_calls={"t1": [second]}))
    assert [ruling.stage for ruling in rulings if not ruling.passed] == []


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
    refused_by_rel = {}
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
        refused_by_rel[rel] = refused
    memorised = refused_by_rel["tools/memorised.py"]
    assert [ruling.stage for ruling in memorised] == [MEMORISED_STAGE]
    assert memorised[0].rows[0]["literal"] == "'W-1'"
    assert "widgets" in memorised[0].rows[0]["match"]
    refusal = refused_by_rel["refusals/t1.json"]
    assert refusal[0].stage == "refuse"
    assert refusal[0].rows[0] == {"task": "t1", "finished": ["r1"]}


def _flip_flop():
    state = {"n": 0}

    def run(source: str, calls: list, db: dict) -> list[dict]:
        state["n"] += 1
        return [{"ok": True, "value": {"n": state["n"]}} for _ in calls]

    return run


def test_the_verifier_suite_refusal_names_the_check_and_the_atom(tmp_path):
    rule = Constraint(id="k1", text="never cancel an order that is not pending", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return pre_state['orders']['#W123']['status'] == 'pending'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    results = S.validate_verifier(verifier, reference_run(), canon=CanonRules())
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


def test_a_single_reference_task_with_a_waived_row_is_trusted_on_a_passing_suite(tmp_path):
    verifier = derive(tmp_path)
    root = tmp_path / "root"
    rel = _write(root, "verifiers/t1.json", json.dumps(as_dict(verifier)))
    waived_row = {verifier.task_id: {"verifier_passed": True, "second_path_waived": True}}
    seeds = tmp_path / "work" / "runs" / TASK
    seeds.mkdir(parents=True)
    for run in (reference_run(), alt_path_run(), other_reason_run()):
        write_events_jsonl(run, seeds / f"{run.run_id}.jsonl")  # the seeds, as Runs of this Task (D281)
    out = rulings_for(root, rel, tmp_path / "work",
                      evidence=_suite_evidence(verifier, alt_path_run=None, task_status=waived_row))
    alt = next(r for r in out if r.stage == "verifier_alt_path")
    assert alt.passed and alt.rows[0]["second_path_waived"] == S.ALT_PATH_NOT_RUN
    trusted = next(r for r in out if r.stage == "trusted")
    assert trusted.passed, trusted.failures
    assert all(r.passed for r in out), [(r.stage, r.failures) for r in out if not r.passed]


def test_a_single_reference_task_without_the_waiver_is_refused_at_the_alt_path_stage(tmp_path):
    verifier = derive(tmp_path)
    root = tmp_path / "root"
    rel = _write(root, "verifiers/t1.json", json.dumps(as_dict(verifier)))
    out = rulings_for(root, rel, tmp_path / "work", evidence=_suite_evidence(verifier, alt_path_run=None))
    assert out[-1].stage == "verifier_alt_path" and not out[-1].passed
    assert S.ALT_PATH_NOT_RUN in out[-1].failures[0]
    assert not any(r.stage == "trusted" for r in out)


def test_loosening_and_false_rejection_refusals_name_their_runs(tmp_path):
    strict, plain = tighten(base(tmp_path)), base(tmp_path)
    runs = {TASK: [reference_run(), extra_write_run(), other_reason_run()]}
    replays = {TASK: {"tr1": replay_row("tr1", True, run_id="ref")}}
    unfinished = {TASK: [reroll_row("rr2", "max_steps")]}
    hist = history(version(strict), version(plain, 2, by="repair", accepted=False))
    out = loosening_gate(hist, runs, replays, unfinished, CanonRules(), SIGS)
    assert not out.passed
    rows = rows_for(out, {})
    assert rows and rows[0] == {"task": TASK, "run": "rr2"}

    strict = tighten(base(tmp_path)).model_copy(update={"seed_run_ids": ["ref"]})
    out = false_rejection_gate([strict], {TASK: [reference_run(), other_reason_run()]},
                               {TASK: {"tr1": replay_row("tr1", True, run_id="ref")}},
                               {TASK: [reroll_row("rr2", "success")]}, CanonRules(), SIGS, task_status={})
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


# --- F27: a write's confinement ruling is about the write ---

_TWO_BODIES = ("class DomainTools:\n"
               "    def cancel_order(self, order_id):\n"
               "        setattr(self, 'x', order_id)\n"
               "        return order_id\n"
               "    def show_order(self, order_id):\n"
               "        return order_id\n")
_CLEAN_SHOW = "def show_order(self, order_id):\n    return order_id\n"
_BAD_CANCEL = "def cancel_order(self, order_id):\n    setattr(self, 'x', order_id)\n    return order_id\n"


def _confined_of(tmp_path, rel: str, written: str, module: str):
    from kullback.gates.hook import ruling_line

    root = tmp_path / "root"
    (root / "tools").mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(written, encoding="utf-8")
    rulings = rulings_for(root, rel, tmp_path / "work", evidence={"render": lambda source: module})
    ruling = next(r for r in rulings if r.stage == "confined")
    return ruling, ruling_line(ruling)


def test_the_write_of_a_clean_body_passes_confined_and_notes_the_other_body_s_failure(tmp_path):
    ruling, line = _confined_of(tmp_path, "tools/show_order.py", _CLEAN_SHOW, _TWO_BODIES)
    assert ruling.passed and ruling.failures == []
    assert line.splitlines() == ["confined pass",
                                 "the module still fails confinement on: cancel_order uses setattr"]


def test_the_write_of_the_failing_body_fails_confined_on_its_own_line(tmp_path):
    ruling, line = _confined_of(tmp_path, "tools/cancel_order.py", _BAD_CANCEL, _TWO_BODIES)
    assert not ruling.passed
    assert ruling.failures == ["cancel_order uses setattr"]
    assert line == "confined fail (cancel_order uses setattr) [1 rows]"
    assert ruling.note == ""


def test_a_module_where_only_the_written_body_fails_carries_no_note(tmp_path):
    module = _TWO_BODIES.replace("setattr(self, 'x', order_id)", "order_id = str(order_id)")
    module = module.replace("    def show_order(self, order_id):\n        return order_id\n",
                            "    def show_order(self, order_id):\n        return eval(order_id)\n")
    written = "def show_order(self, order_id):\n    return eval(order_id)\n"
    ruling, line = _confined_of(tmp_path, "tools/show_order.py", written, module)
    assert not ruling.passed and ruling.failures == ["show_order uses eval"]
    assert ruling.note == "" and "\n" not in line
