"""Golden invariants for verdict.py: the Verdict is code only, computed on End state only (D43, D46, D94)."""

from __future__ import annotations

import json

import pytest

from kullback.gates.artifacts import regrade_gate
from kullback.runner.canon import canon_value
from kullback.runner.records import Atom, Column, EntitySchema, Environment, Verifier
from kullback.runner.verdict import VERDICT_VERSION, load_run, verdict

CANCEL = "cancel_pending_order"
READ = "get_order_details"
WRITE_TOOLS = {CANCEL, "delete_order", "modify_order"}


# --- run JSONL builders ---

def user(idx, text):
    return {"idx": idx, "type": "user_turn", "payload": {"content": text}}


def say(idx, text):
    return {"idx": idx, "type": "model_call", "payload": {"content": text}}


def call(idx, name, args=None, cid=None, **payload):
    body = {"id": cid or f"c{idx}", "name": name, "args": args or {}}
    body.update(payload)
    return {"idx": idx, "type": "tool_call", "payload": body}


def result(idx, cid, value=None, error=None):
    return {"idx": idx, "type": "tool_result", "payload": {"id": cid, "result": value, "error": error}}


def stop(idx, reason="done", **payload):
    body = {"termination_reason": reason}
    body.update(payload)
    return {"idx": idx, "type": "stop", "payload": body}


def header(run_id="r1", **extra):
    body = {"run_id": run_id, "env_id": "e1", "task_id": "t1", "model": "test/model"}
    body.update(extra)
    return body


def oracle_lines(reason="no longer needed", order_id="W123"):
    return [
        header(),
        user(0, "I want to stop order W123"),
        say(1, "Sure. May I ask why you want to stop it?"),
        user(2, "It arrived late elsewhere. Yes, go ahead."),
        call(3, READ, {"order_id": "W123"}, cid="c3"),
        result(4, "c3", {"status": "pending"}),
        call(5, CANCEL, {"order_id": order_id, "reason": reason}, cid="c5"),
        result(6, "c5", {"status": "cancelled"}),
        say(7, "Done. The refund goes back to your original card in 5 days."),
        stop(8),
    ]


@pytest.fixture
def verifier():
    """One Task's Verifier: a required write, an allowed value, a question, a message, a forbidden write, a Hard
    constraint."""
    return Verifier(
        task_id="t1",
        verifier_version="v1",
        atoms=[
            Atom(
                id="a_cancel",
                kind="required",
                provenance="user_stated",
                description="order W123 is cancelled",
                predicate_src='wrote("cancel_pending_order", order_id="W123")',
            ),
            Atom(
                id="a_reason",
                kind="allowed",
                provenance="user_elicited",
                description="any reason the user gave is fine",
                predicate_src='wrote("cancel_pending_order")',
            ),
            Atom(
                id="a_ask_why",
                kind="question",
                description="the agent asked why",
                predicate_src='asked("why")',
            ),
            Atom(
                id="a_tell_refund",
                kind="communicate",
                description="the user was told about the original card",
                predicate_src='communicated("original card")',
            ),
            Atom(
                id="a_no_delete",
                kind="forbidden",
                description="an order is never deleted",
                predicate_src='called("delete_order")',
            ),
            Atom(
                id="a_confirm_first",
                kind="hard",
                description="no cancel without a prior confirmation",
                predicate_src='user_confirmed_before("cancel_pending_order", "yes", "confirm")',
            ),
        ],
    )


@pytest.fixture
def write_run(tmp_path):
    def _write(lines, name="run.jsonl"):
        path = tmp_path / name
        with path.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line) + "\n")
        return path

    return _write


# --- the golden invariants from design section 9 ---

def test_oracle_run_passes(verifier, write_run):
    out = verdict(write_run(oracle_lines()), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is True
    assert out.class_ == "pass"
    assert out.failing_atom is None
    assert out.judge_used is False
    assert out.environment_suspected is False
    assert out.cause is None


def test_empty_run_fails(verifier, write_run):
    lines = [header(), user(0, "I want to stop order W123"), say(1, "Noted."), stop(2)]
    out = verdict(write_run(lines), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.failing_atom == "a_cancel"
    assert out.class_ in {"fail", "transferred_without_acting"}


def test_plausible_wrong_run_fails(verifier, write_run):
    """The right shape, the wrong order id: a user-stated value is required (D42)."""
    out = verdict(write_run(oracle_lines(order_id="W999")), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.failing_atom == "a_cancel"
    assert out.class_ == "fail"


def test_two_valid_orders_both_pass(verifier, write_run):
    """Reads and their order are path, never End state (D94)."""
    first = [
        header(),
        user(0, "I want to stop order W123"),
        call(1, READ, {"order_id": "W123"}, cid="c1"),
        result(2, "c1", {"status": "pending"}),
        say(3, "Why do you want to stop it?"),
        user(4, "Yes, cancel it, it came late."),
        call(5, CANCEL, {"order_id": "W123", "reason": "late"}, cid="c5"),
        result(6, "c5", {"status": "cancelled"}),
        say(7, "Refunded to your original card."),
        stop(8),
    ]
    second = [
        header(run_id="r2"),
        user(0, "I want to stop order W123"),
        say(1, "Why?"),
        user(2, "Yes please, it came late."),
        call(3, READ, {"order_id": "W123"}, cid="c3"),
        result(4, "c3", {"status": "pending"}),
        call(5, CANCEL, {"order_id": "W123", "reason": "came late"}, cid="c5"),
        result(6, "c5", {"status": "cancelled"}),
        say(7, "The money returns to your original card."),
        stop(8),
    ]
    a = verdict(write_run(first, "a.jsonl"), verifier, canon_value, write_tools=WRITE_TOOLS)
    b = verdict(write_run(second, "b.jsonl"), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert a.passed is True
    assert b.passed is True


def test_skipped_required_question_fails(verifier, write_run):
    """D43 reverses the older default: a required question that is not asked fails."""
    lines = [line for line in oracle_lines() if line.get("idx") != 1]
    lines.insert(1, say(1, "Right away."))
    out = verdict(write_run(lines), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.failing_atom == "a_ask_why"


def test_different_wording_of_an_allowed_value_passes(verifier, write_run):
    """The reason is user-elicited, so any plausible wording passes (D42)."""
    out = verdict(
        write_run(oracle_lines(reason="I simply changed my mind")),
        verifier,
        canon_value,
        write_tools=WRITE_TOOLS,
    )
    assert out.passed is True


# --- the rest of the checks verdict() owns ---

def test_canonicalization_is_applied_to_write_arguments(verifier, write_run):
    out = verdict(write_run(oracle_lines(order_id="  w123 ")), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is True


def test_forbidden_write_fails(verifier, write_run):
    lines = oracle_lines()
    lines.insert(-1, call(75, "delete_order", {"order_id": "W123"}, cid="c75"))
    lines.insert(-1, result(76, "c75", {"ok": True}))
    out = verdict(write_run(lines), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.failing_atom == "a_no_delete"


def test_hard_constraint_on_the_transcript_fails(verifier, write_run):
    """D43 case 3: a write with no prior confirmation in the transcript."""
    lines = [
        header(),
        user(0, "I want to stop order W123"),
        say(1, "Why do you want that?"),
        user(2, "It came late."),
        call(3, CANCEL, {"order_id": "W123", "reason": "late"}, cid="c3"),
        result(4, "c3", {"status": "cancelled"}),
        say(5, "Refunded to your original card."),
        stop(6),
    ]
    out = verdict(write_run(lines), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.failing_atom == "a_confirm_first"


def test_extra_write_is_a_spurious_side_effect(verifier, write_run):
    lines = oracle_lines()
    lines.insert(-1, call(70, "modify_order", {"order_id": "W900", "note": "x"}, cid="c70"))
    lines.insert(-1, result(71, "c70", {"ok": True}))
    out = verdict(write_run(lines), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.failing_atom.startswith("extra_write:modify_order")


def test_side_effect_count_is_reported(verifier, write_run):
    out = verdict(write_run(oracle_lines()), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert "side_effects=1" in out.notes
    assert "tool_calls=2" in out.notes


def test_extra_write_check_is_skipped_without_a_write_tool_set(verifier, write_run):
    """Without ToolSig kinds the Verifier says nothing about other writes, so nothing is invented."""
    lines = oracle_lines()
    lines.insert(-1, call(70, "modify_order", {"order_id": "W900"}, cid="c70"))
    lines.insert(-1, result(71, "c70", {"ok": True}))
    out = verdict(write_run(lines), verifier, canon_value)
    assert out.passed is True
    assert "side_effect_check_skipped" in out.notes


@pytest.mark.parametrize(
    "write_tools",
    [WRITE_TOOLS, WRITE_TOOLS | {"transfer_to_human_agents"}],
    ids=["transfer_not_a_write", "transfer_classed_as_write"],
)
def test_transferred_without_acting_is_its_own_class(verifier, write_run, write_tools):
    """D46: mine.py may class the transfer tool as a write, and it still changed nothing."""
    lines = [
        header(),
        user(0, "I want to stop order W123"),
        say(1, "Let me pass you to a person."),
        call(2, "transfer_to_human_agents", {"summary": "cancel"}, cid="c2"),
        result(3, "c2", {"ok": True}),
        stop(4, reason="agent_transfer"),
    ]
    out = verdict(write_run(lines), verifier, canon_value, write_tools=write_tools)
    assert out.passed is False
    assert out.class_ == "transferred_without_acting"
    assert out.failing_atom == "a_cancel"


def test_an_environment_error_event_classes_the_run_env_error_and_names_the_environment_as_cause(
    verifier, write_run
):
    lines = [
        header(),
        user(0, "I want to stop order W123"),
        {"idx": 1, "type": "error", "payload": {"environment": True, "message": "tool crashed"}},
        stop(2, reason="env_error"),
    ]
    out = verdict(write_run(lines), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.class_ == "env_error"
    assert out.cause == "environment"
    assert out.environment_suspected is True


# --- D88 code-first cause marks ---

def _wrong_run_with(extra_event):
    lines = oracle_lines(order_id="W999")
    lines.insert(-1, extra_event)
    return lines


@pytest.mark.parametrize(
    "event,mark",
    [
        ({"idx": 60, "type": "tool_call", "payload": {"name": "lookup", "args": {}}, "assisted": True},
         "env_mark:assisted"),
        (
            {
                "idx": 60,
                "type": "user_turn",
                "payload": {"content": "I do not have that", "fact_unavailable": "card_last_four"},
            },
            "env_mark:fact_unavailable",
        ),
        ({"idx": 60, "type": "tool_call", "payload": {"name": "lookup", "args": {}, "overlay_miss": True}},
         "env_mark:overlay_miss"),
    ],
)
def test_code_marks_environment_suspected(verifier, write_run, event, mark):
    out = verdict(write_run(_wrong_run_with(event)), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.environment_suspected is True
    assert mark in out.notes
    assert out.cause is None  # code marks it, the judge names the cause (D88)


def test_flagged_tool_marks_environment_suspected(verifier, write_run):
    out = verdict(
        write_run(oracle_lines(order_id="W999")),
        verifier,
        canon_value,
        write_tools=WRITE_TOOLS,
        flagged_tools={READ},
    )
    assert out.environment_suspected is True
    assert "env_mark:flagged_tool:get_order_details" in out.notes


def test_unmarked_failure_waits_for_the_judge(verifier, write_run):
    out = verdict(write_run(oracle_lines(order_id="W999")), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.environment_suspected is False
    assert out.cause is None
    assert "cause_pending_judge" in out.notes


# --- judge atoms (D76) ---

def _judge_verifier(verifier):
    atoms = list(verifier.atoms) + [
        Atom(id="a_polite", kind="required", judge=True, description="the tone matched the policy")
    ]
    return Verifier(task_id="t1", verifier_version="v2", atoms=atoms)


def test_judge_used_only_when_judge_results_are_supplied(verifier, write_run):
    path = write_run(oracle_lines())
    judged = _judge_verifier(verifier)
    without = verdict(path, judged, canon_value, write_tools=WRITE_TOOLS)
    assert without.judge_used is False
    assert without.passed is False  # a required atom nobody answered is not a pass (D76)
    # An unanswered atom is an immature Verifier, not a broken Environment (design section 6).
    assert without.class_ == "not_verdicted"
    assert "judge_atom_unevaluated:a_polite" in without.notes

    with_pass = verdict(path, judged, canon_value, {"a_polite": True}, write_tools=WRITE_TOOLS)
    assert with_pass.judge_used is True
    assert with_pass.passed is True


# --- same path (D46) ---

@pytest.mark.parametrize(
    "reference_path, expected",
    [([READ, CANCEL], True), ([CANCEL, READ], False), (None, None)],
    ids=["reference_order", "a_different_order", "no_reference"],
)
def test_same_path_compares_the_write_order_against_the_reference_and_is_unknown_without_one(
    verifier, write_run, reference_path, expected
):
    out = verdict(
        write_run(oracle_lines()),
        verifier,
        canon_value,
        write_tools=WRITE_TOOLS,
        reference_path=reference_path,
    )
    assert out.same_path is expected


# --- versions (D97) ---

def test_all_versions_are_copied_onto_the_verdict(verifier, write_run):
    env = Environment(
        env_id="env-abc", schema_version="s3", tools_version="t7", policy_version="p2", version=4
    )
    out = verdict(
        write_run(oracle_lines()),
        verifier,
        canon_value,
        environment=env,
        runner_version="runner-1",
        write_tools=WRITE_TOOLS,
    )
    assert out.env_id == "env-abc"
    assert (out.schema_version, out.tools_version, out.policy_version) == ("s3", "t7", "p2")
    assert out.verifier_version == "v1"
    assert out.runner_version == "runner-1"
    assert out.verdict_version == VERDICT_VERSION
    assert out.run_id == "r1"


def test_a_verdict_scored_without_an_environment_leaves_those_versions_absent(verifier, write_run):
    """A placeholder version string is truthy, so it walks past regrade's presence check (D97)."""
    out = verdict(write_run(oracle_lines()), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert (out.schema_version, out.tools_version, out.policy_version) == (None, None, None)
    assert regrade_gate([out]).passed is False


# --- end state diff ---

def test_end_state_diff_ignores_exempt_columns(verifier, write_run):
    lines = oracle_lines()
    lines[-1] = stop(
        8,
        start_state={"orders": {"W123": {"status": "pending", "updated_at": "1", "id": "W123"}}},
        end_state={"orders": {"W123": {"status": "cancelled", "updated_at": "2", "id": "W123"}}},
    )
    schema = EntitySchema(
        tables=["orders"],
        columns=[
            Column(table="orders", name="status", **{"class": "hard"}),
            Column(table="orders", name="updated_at", **{"class": "exempt"}),
            Column(table="orders", name="id", **{"class": "exempt"}),
        ],
    )
    extra = Atom(
        id="a_status",
        kind="required",
        predicate_src='value("orders", "W123", "status") == "cancelled" and list(diff()) == ["orders.W123"]',
    )
    with_state = Verifier(task_id="t1", verifier_version="v1", atoms=list(verifier.atoms) + [extra])
    out = verdict(write_run(lines), with_state, canon_value, write_tools=WRITE_TOOLS, schema=schema)
    assert out.passed is True


def _semantic_setup(write_run):
    """A Run whose only doubtful change is one semantic column, and the schema that says so."""
    lines = oracle_lines()
    lines[-1] = stop(
        8,
        start_state={"orders": {"W123": {"status": "pending", "note": "cancelled by the user"}}},
        end_state={"orders": {"W123": {"status": "cancelled", "note": "cancelled by user"}}},
    )
    schema = EntitySchema(tables=["orders"], columns=[
        Column(table="orders", name="status", **{"class": "hard"}),
        Column(table="orders", name="note", **{"class": "semantic"}),
    ])
    atom = Atom(id="a_note", kind="required",
                predicate_src='"note" not in diff()["orders.W123"]["fields"]')
    return write_run(lines), Verifier(task_id="t1", atoms=[atom]), schema


def test_a_semantic_column_is_settled_by_the_equivalence_table_not_by_string_equality(write_run):
    """D84: two wordings of one field are the same End state only where the table says they are."""
    from kullback.runner import canon as canon_module

    path, semantic, schema = _semantic_setup(write_run)
    table = canon_module.EquivalenceTable()
    canon_module.put(table, "orders.note",
                     canon_module.canon_value("cancelled by the user"),
                     canon_module.canon_value("cancelled by user"),
                     True, classified_by="human")
    out = verdict(path, semantic, schema=schema, equivalence=table)
    assert out.passed is True, out.notes
    assert not [note for note in out.notes if note.startswith("semantic_unresolved")]


def test_a_semantic_pair_nobody_settled_is_named_and_never_read_as_equal(write_run, tmp_path):
    """verdict.py never calls a model (D91), so an unsettled pair is left open, not judged here."""
    path, semantic, schema = _semantic_setup(write_run)
    out = verdict(path, semantic, schema=schema, workdir=tmp_path)
    assert out.passed is False
    assert any(note.startswith("semantic_unresolved:") for note in out.notes)
    used = (tmp_path / "equivalence_uses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(used) == 1 and json.loads(used[0])["run_id"] == "r1"


# --- loading and the code-only property ---

def test_load_run_reads_header_events_and_footer(write_run):
    path = write_run([header(run_id="rx"), user(0, "hi"), stop(1, reason="done")])
    run = load_run(path)
    assert run.run_id == "rx"
    assert run.task_id == "t1"
    assert [event.type for event in run.events] == ["user_turn", "stop"]


def test_load_run_accepts_a_whole_run_on_one_line(write_run):
    path = write_run([{"run_id": "ry", "events": [user(0, "hi")]}])
    assert load_run(path).run_id == "ry"


def import_closure(module_name: str) -> set:
    """Every kullback module reachable from this one by import, however deep."""
    import ast
    import importlib
    from pathlib import Path

    seen: set = set()
    stack = [module_name]
    while stack:
        name = stack.pop()
        if name in seen or not name.startswith("kullback"):
            continue
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue  # a "from package import Name" that names an object, not a module
        seen.add(name)
        source = Path(module.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                stack.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                stack.append(node.module)
                stack.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return seen


def test_verdict_never_calls_a_model():
    """The deterministic half of the Verdict has no import path to a provider (D76).

    A grep for the word would pass on `import kullback.runner.judge`, which pulls the provider in
    two hops, so this walks the import graph instead.
    """
    closure = import_closure("kullback.runner.verdict")
    assert "kullback.runner.verdict" in closure
    assert "kullback.ai.provider" not in closure
    assert not [name for name in closure if name.startswith("kullback.builder")]  # D89, D91


def test_an_atom_without_a_predicate_leaves_the_run_not_verdicted(verifier, write_run):
    naked = Verifier(task_id="t1", atoms=[Atom(id="a_bare", kind="required")])
    out = verdict(write_run(oracle_lines()), naked, canon_value)
    assert out.passed is False
    assert out.class_ == "not_verdicted"
    assert out.failing_atom == "a_bare"
    assert "atom_without_predicate:a_bare" in out.notes


def test_an_atom_that_cannot_be_checked_never_counts_as_a_pass(write_run):
    """An allowed atom is not required to hold, so only the required one stops the Verdict."""
    broken = Verifier(task_id="t1", atoms=[
        Atom(id="a_loose", kind="allowed", predicate_src="wrote("),
        Atom(id="a_bad", kind="required", predicate_src="1 / 0"),
    ])
    out = verdict(write_run(oracle_lines()), broken, canon_value)
    assert any(note.startswith("atom_rejected:a_loose") for note in out.notes)
    assert any(note.startswith("atom_error:a_bad:ZeroDivisionError") for note in out.notes)
    assert out.passed is False
    assert out.class_ == "not_verdicted"
    assert out.failing_atom == "a_bad"


def test_a_predicate_using_a_certified_builtin_is_evaluated_not_skipped(tmp_path):
    from kullback.runner.records import Atom, Verifier
    from kullback.runner.verdict import verdict as run_verdict

    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"run_id": "r", "termination_reason": "user_stop"}) + "\n"
                    + json.dumps({"idx": 0, "type": "stop", "payload": {}}) + "\n", encoding="utf-8")
    verifier = Verifier(task_id="t", atoms=[
        Atom(id="rounds", kind="required", predicate_src="round(2.4) == 2 and len(list(zip([1], [2]))) == 1")])
    out = run_verdict(path, verifier)
    assert out.passed is True
    assert not [n for n in out.notes if n.startswith("atom_error")]


# --- the atom vocabulary is a fence, not a doorway (design section 7, code without exception) ---

def test_a_predicate_that_reaches_module_globals_is_rejected_before_it_runs(write_run, tmp_path):
    """wrote.__globals__ used to hand an atom this module's Path and json, so an atom could write files."""
    marker = tmp_path / "escaped.txt"
    escape = Verifier(task_id="t1", atoms=[Atom(
        id="esc", kind="required",
        predicate_src=f"wrote.__globals__['Path']({str(marker)!r}).write_text('pwned') or True")])
    out = verdict(write_run(oracle_lines()), escape, canon_value)
    assert marker.exists() is False
    assert any(note.startswith("atom_rejected:esc") for note in out.notes)
    assert out.passed is False


def test_a_predicate_that_walks_subclasses_or_a_bound_self_is_rejected(write_run):
    escape = Verifier(task_id="t1", atoms=[Atom(
        id="esc", kind="required",
        predicate_src="len(().__class__.__base__.__subclasses__()) > 10 and canon.__self__.run.run_id == 'r1'")])
    out = verdict(write_run(oracle_lines()), escape, canon_value)
    assert any(note.startswith("atom_rejected:esc") for note in out.notes)
    assert out.passed is False


def test_a_predicate_that_imports_is_rejected(write_run):
    escape = Verifier(task_id="t1", atoms=[Atom(
        id="esc", kind="required",
        predicate_src="def check():\n    import os\n    return os.getpid() > 0\n")])
    out = verdict(write_run(oracle_lines()), escape, canon_value)
    assert any("imports" in note for note in out.notes if note.startswith("atom_rejected:esc"))
    assert out.passed is False


def test_the_gate_lets_the_predicates_the_builder_writes_through(write_run, verifier):
    """The fence is only a fence: every atom of the golden Verifier still evaluates."""
    out = verdict(write_run(oracle_lines()), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is True
    assert not [note for note in out.notes if note.startswith(("atom_rejected", "atom_error"))]


# --- a judge result is read by its verdict word, never by bool() (D76) ---

def _failing_judge_answers():
    """Every shape a failing judge answer arrives in: a bare dict, a JudgeResult, its dump."""
    from kullback.runner.judge import JudgeResult

    answer = JudgeResult(use="policy_atom", verdict="fail", judge="a", cited_spans=["turn 7"])
    return [
        ({"pass": False, "spans": ["turn 7"]}, "bare_dict"),
        (answer, "judge_result"),
        (answer.model_dump(), "judge_result_dump"),
    ]


@pytest.mark.parametrize(
    "answer", [shape for shape, _ in _failing_judge_answers()],
    ids=[name for _, name in _failing_judge_answers()],
)
def test_a_judge_atom_that_says_fail_fails_the_run(verifier, write_run, answer):
    judged = _judge_verifier(verifier)
    out = verdict(write_run(oracle_lines()), judged, canon_value, {"a_polite": answer},
                  write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.failing_atom == "a_polite"
    assert out.judge_used is True


def test_every_judge_use_maps_its_own_verdict_words(verifier, write_run):
    judged = _judge_verifier(verifier)
    holds = ("pass", "equivalent", "acceptable", "good_reference")
    fails = ("fail", "not_equivalent", "unacceptable", "bad_reference")
    for word in holds:
        out = verdict(write_run(oracle_lines()), judged, canon_value,
                      {"a_polite": {"verdict": word}}, write_tools=WRITE_TOOLS)
        assert out.passed is True, word
    for word in fails:
        out = verdict(write_run(oracle_lines()), judged, canon_value,
                      {"a_polite": {"verdict": word}}, write_tools=WRITE_TOOLS)
        assert out.passed is False and out.failing_atom == "a_polite", word


def test_an_abstaining_judge_atom_leaves_the_run_not_verdicted(verifier, write_run):
    """D76 sends an unsure judge atom to a person, so it is never a counted pass."""
    judged = _judge_verifier(verifier)
    for word in ("abstain", "undetermined"):
        out = verdict(write_run(oracle_lines()), judged, canon_value,
                      {"a_polite": {"verdict": word}}, write_tools=WRITE_TOOLS)
        assert out.passed is False
        assert out.class_ == "not_verdicted"
        assert any(note.startswith("judge_abstained:a_polite") for note in out.notes)


# --- side effects are counted whenever the write-tool set is known (D43) ---

@pytest.mark.parametrize(
    "wants_none",
    [
        Verifier(task_id="t1", atoms=[
            Atom(id="c0", kind="communicate", predicate_src='communicated("original card")')]),
        Verifier(task_id="t1"),
    ],
    ids=["communicate_only", "no_atoms_at_all"],
)
def test_a_write_on_a_task_whose_verifier_wants_none_is_an_extra_write(write_run, wants_none):
    out = verdict(write_run(oracle_lines()), wants_none, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.failing_atom == "extra_write:cancel_pending_order"
    assert "side_effect_check_skipped" not in out.notes


# --- an atom that cannot be evaluated is a Verifier defect, never a pass (D76, D79) ---

def test_a_required_atom_that_cannot_be_evaluated_is_not_a_pass(write_run):
    holed = Verifier(task_id="t1", atoms=[
        Atom(id="j", kind="required", judge=True),
        Atom(id="b", kind="required", predicate_src="wrote("),
        Atom(id="n", kind="required"),
    ])
    out = verdict(write_run(oracle_lines()), holed, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.class_ == "not_verdicted"
    assert out.failing_atom == "j"
    # The Verifier could not answer, so nothing here points at the Environment (design section 6).
    assert out.environment_suspected is False
    assert out.cause == "undetermined"


def test_a_definite_failure_wins_over_an_unevaluable_atom(write_run):
    holed = Verifier(task_id="t1", atoms=[
        Atom(id="a_cancel", kind="required",
             predicate_src='wrote("cancel_pending_order", order_id="W999")'),
        Atom(id="j", kind="required", judge=True),
    ])
    out = verdict(write_run(oracle_lines()), holed, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is False
    assert out.class_ == "fail"
    assert out.failing_atom == "a_cancel"


# --- reading the End state (D39) ---

def test_value_returns_the_raw_value_and_eq_canonicalizes(write_run):
    """The default canonicalizer turns every value into a string, so a literal must compare raw."""
    lines = oracle_lines()
    lines[-1] = stop(8,
                     start_state={"orders": {"W123": {"status": "pending", "total": 25, "paid": False}}},
                     end_state={"orders": {"W123": {"status": "cancelled", "total": 25, "paid": True}}})
    typed = Verifier(task_id="t1", atoms=[
        Atom(id="total", kind="required", predicate_src='value("orders", "W123", "total") == 25'),
        Atom(id="paid", kind="required", predicate_src='value("orders", "W123", "paid") is True'),
        Atom(id="same", kind="required",
             predicate_src='eq(value("orders", "W123", "status"), " Cancelled ")'),
    ])
    out = verdict(write_run(lines), typed)  # the default canon is canon.canon_value
    assert out.passed is True, out.notes


# --- the transcript a predicate sees (D43 case 3, D45) ---

def test_inline_tool_calls_keep_the_position_of_their_model_call(write_run):
    lines = [
        header(),
        user(0, "Cancel W123"),
        user(1, "yes go ahead"),
        {"idx": 2, "type": "model_call", "payload": {"content": "", "tool_calls": [
            {"id": "c2", "name": CANCEL, "args": {"order_id": "W123"}}]}},
        stop(3),
    ]
    confirmed = Verifier(task_id="t1", atoms=[Atom(
        id="h", kind="hard", predicate_src='user_confirmed_before("cancel_pending_order")')])
    out = verdict(write_run(lines), confirmed, canon_value)
    assert out.passed is True


def test_inline_tool_calls_are_read_from_a_reply_nested_payload(write_run):
    """loop.py always nests the reply (and its tool_calls) under payload["reply"] (D90)."""
    lines = [
        header(),
        user(0, "Cancel W123"),
        user(1, "yes go ahead"),
        {"idx": 2, "type": "model_call", "payload": {"reply": {"content": "", "tool_calls": [
            {"id": "c2", "name": CANCEL, "arguments": {"order_id": "W123"}}]}}},
        stop(3),
    ]
    confirmed = Verifier(task_id="t1", atoms=[Atom(
        id="h", kind="hard", predicate_src='wrote("cancel_pending_order", order_id="W123")')])
    out = verdict(write_run(lines), confirmed, canon_value)
    assert out.passed is True, out.notes


def test_a_call_that_errored_is_not_a_call_that_happened(verifier, write_run):
    """D45: a rejected call had no effect, so a forbidden atom must not fire on it."""
    lines = oracle_lines()
    lines.insert(-1, call(70, "delete_order", {"order_id": "W123"}, cid="c70"))
    lines.insert(-1, result(71, "c70", None, error={"class": "invalid_arguments", "payload": "bad"}))
    out = verdict(write_run(lines), verifier, canon_value, write_tools=WRITE_TOOLS)
    assert out.passed is True
    assert out.failing_atom is None


def test_a_hard_atom_sees_every_write_whatever_its_place_in_the_verifier(write_run):
    """Without a write-tool set the write list is what the other atoms covered, so hard atoms run last."""
    lines = [
        header(),
        user(0, "Cancel W123"),
        call(1, CANCEL, {"order_id": "W123"}, cid="c1"),
        result(2, "c1", {"status": "cancelled"}),
        stop(3),
    ]
    ordered = Verifier(task_id="t1", atoms=[
        Atom(id="h", kind="hard", predicate_src="def check():\n    return all(False for _ in write_calls())\n"),
        Atom(id="a_cancel", kind="required",
             predicate_src='wrote("cancel_pending_order", order_id="W123")'),
    ])
    out = verdict(write_run(lines), ordered, canon_value)
    assert out.passed is False
    assert out.failing_atom == "h"


# --- the judge names the cause of a failure (D88) ---

def test_a_judge_cause_lands_on_the_verdict(verifier, write_run):
    out = verdict(write_run(oracle_lines(order_id="W999")), verifier, canon_value,
                  write_tools=WRITE_TOOLS,
                  cause_result={"use": "cause", "verdict": "simulated_user",
                                "cited_spans": ["the user gave the wrong id"]})
    assert out.passed is False
    assert out.cause == "simulated_user"
    assert "cause_pending_judge" not in out.notes
    assert any("the user gave the wrong id" in note for note in out.notes)


def test_an_undetermined_cause_leaves_the_failure_for_a_person(verifier, write_run):
    out = verdict(write_run(oracle_lines(order_id="W999")), verifier, canon_value,
                  write_tools=WRITE_TOOLS, cause_result={"use": "cause", "verdict": "undetermined"})
    assert out.cause == "undetermined"


def test_a_cause_is_never_invented_for_a_run_that_passed(verifier, write_run):
    out = verdict(write_run(oracle_lines()), verifier, canon_value, write_tools=WRITE_TOOLS,
                  cause_result={"use": "cause", "verdict": "candidate"})
    assert out.passed is True
    assert out.cause is None
