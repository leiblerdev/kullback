"""One scorer (G3): the structured target is the only meaning for non-Hard atoms.

Invented tools and tables only. Skipped unless the one-scorer patch is applied.
"""

from __future__ import annotations

import importlib.util

import pytest

from kullback.gates import verifier_suite as S
from kullback.runner.atom_context import _evaluate
from kullback.runner.canon import CanonRules
from kullback.runner.records import Atom, Event, Run, Verifier
from kullback.runner.verdict import verdict

try:
    from kullback.runner import target as T
except ImportError:  # without the one-scorer patch the module is absent and every test skips
    T = None  # type: ignore

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kullback.runner.target") is None,
    reason="needs the one-scorer patch",
)

WRITE_TOOLS = {"archive_entry"}


def _ev(type_: str, **payload) -> dict:
    return {"type": type_, "payload": payload}


def user(text: str) -> dict:
    return _ev("user_turn", content=text)


def assistant(text: str) -> dict:
    return _ev("model_call", reply={"content": text})


def assistant_with_call(text: str, name: str, args: dict, cid: str) -> dict:
    return {"type": "model_call",
            "payload": {"reply": {"content": text, "tool_calls": [{"name": name, "arguments": args, "id": cid}]}}}


def call(name: str, args: dict, kind: str = "write", cid: str = "c1") -> dict:
    return _ev("tool_call", id=cid, name=name, args=args, kind=kind)


def result(data, cid: str = "c1") -> dict:
    return _ev("tool_result", id=cid, result=data)


def make_run(run_id: str, events: list[dict]) -> Run:
    return Run(run_id=run_id, task_id="t9", termination_reason="success",
               events=[Event(idx=i, type=e["type"], payload=e["payload"]) for i, e in enumerate(events)])


def _key(value) -> str:
    return T._key(T.canon_fn(None), value)


def test_a_fact_stated_but_never_read_fails_the_one_scorer():
    """Cause one: the answer states E-999 but no tool result ever held it."""
    key = _key("E-999")
    atom = S.make_atom("c0", "communicate", {"kind": "communicate", "value": key, "text": "E-999"})
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[atom])
    run = make_run("r1", [
        user("Please archive entry E-100."),
        call("fetch_entry", {"entry_id": "E-100"}, kind="read", cid="c0"),
        result({"entry_id": "E-100", "title": "ledger"}, cid="c0"),
        call("archive_entry", {"entry_id": "E-100"}, cid="c1"),
        result({"entry_id": "E-100", "archived": True}, cid="c1"),
        assistant("Done, entry E-999 is archived."),
    ])
    assert T.communicate_values(run, T.canon_fn(None)) == {}
    assert S.check_run(verifier, run, write_tools=WRITE_TOOLS) == (False, "c0")
    out = verdict(run, verifier, write_tools=WRITE_TOOLS)
    assert out.passed is False and out.failing_atom == "c0"


def test_a_value_hidden_inside_a_longer_word_fails_the_one_scorer():
    """Cause three: E-12 passes a substring test inside XE-123 but is no token of any message."""
    key = _key("E-12")
    atom = S.make_atom("c0", "communicate", {"kind": "communicate", "value": key, "text": "E-12"})
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[atom])
    run = make_run("r1", [
        user("Please archive entry XE-123."),
        call("fetch_entry", {"entry_id": "XE-123"}, kind="read", cid="c0"),
        result({"entry_id": "XE-123", "title": "ledger"}, cid="c0"),
        assistant("Done, entry XE-123 is archived."),
    ])
    assert S.check_run(verifier, run, write_tools=WRITE_TOOLS) == (False, "c0")
    out = verdict(run, verifier, write_tools=WRITE_TOOLS)
    assert out.passed is False and out.failing_atom == "c0"


def test_a_fact_stated_in_a_tool_call_turn_fails_the_one_scorer():
    """Smaller cause: the fact is stated only in a turn that also called a tool."""
    key = _key("E-441")
    atom = S.make_atom("c0", "communicate", {"kind": "communicate", "value": key, "text": "E-441"})
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[atom])
    run = make_run("r1", [
        user("Please archive entry E-441."),
        call("fetch_entry", {"entry_id": "E-441"}, kind="read", cid="c0"),
        result({"entry_id": "E-441", "title": "ledger"}, cid="c0"),
        assistant_with_call("Entry E-441 is archived.", "archive_entry", {"entry_id": "E-441"}, "c1"),
        result({"entry_id": "E-441", "archived": True}, cid="c1"),
    ])
    assert S.check_run(verifier, run, write_tools=WRITE_TOOLS) == (False, "c0")
    out = verdict(run, verifier, write_tools=WRITE_TOOLS)
    assert out.passed is False and out.failing_atom == "c0"


def test_a_raising_hard_rule_is_a_defect_on_both_sides():
    """A Hard rule that raises is never a Candidate failure: None in the gates, not verdicted here."""
    atom = S.make_atom("hard.k1", "hard", {"kind": "hard", "constraint_id": "k1",
                                           "predicate_src": "def check(pre_state, write_call, transcript):\n    return None.foo\n",
                                           "write_tools": ["archive_entry"], "read_tools": []})
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[atom])
    run = make_run("r1", [
        user("Please archive entry E-100."),
        call("archive_entry", {"entry_id": "E-100"}, cid="c1"),
        result({"entry_id": "E-100", "archived": True}, cid="c1"),
        assistant("Done."),
    ])
    assert T.hard_holds(atom, run, WRITE_TOOLS) is None
    assert S.check_run(verifier, run, write_tools=None) == (True, None)
    out = verdict(run, verifier, write_tools=None)
    assert out.passed is False and out.class_ == "not_verdicted"


def test_customer_canon_rules_change_the_answer_where_the_default_would_not():
    """With case kept, ABC and abc differ; under the module default they are the same."""
    rules = CanonRules(lowercase=False)
    fn_rules, fn_default = T.canon_fn(rules), T.canon_fn(None)
    assert fn_rules("AB12") != fn_rules("ab12")
    assert fn_default("AB12") == fn_default("ab12")
    key = T._key(fn_rules, "AB12")
    atom = S.make_atom("c0", "communicate", {"kind": "communicate", "value": key, "text": "AB12"})
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[atom])
    run = make_run("r1", [
        user("Please read tag AB12."),
        call("fetch_entry", {"entry_id": "AB12"}, kind="read", cid="c0"),
        result({"entry_id": "AB12", "title": "AB12"}, cid="c0"),
        assistant("The tag is AB12."),
    ])
    assert S.check_run(verifier, run, rules, write_tools=WRITE_TOOLS) == (True, None)
    assert S.check_run(verifier, run, None, write_tools=WRITE_TOOLS) == (False, "c0")
    out = verdict(run, verifier, rules=rules, write_tools=WRITE_TOOLS)
    assert out.passed is True


def test_a_defective_hard_rule_fails_admission_but_not_the_run():
    """D79 admits no Verifier whose policy rule cannot run; the Candidate Run still passes."""
    fn = T.canon_fn(None)
    entity = T.text_of(fn("E-100"))
    write = S.make_atom("w0", "required", {"kind": "write", "tool": "archive_entry",
                                              "entity": entity, "entity_raw": "E-100",
                                              "id_field": "entry_id"})
    hard = S.make_atom("hard.k1", "hard", {"kind": "hard", "constraint_id": "k1",
                                               "predicate_src": "def check(pre_state, write_call, transcript):\n    return None.foo\n",
                                               "write_tools": ["archive_entry"], "read_tools": []})
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[write, hard])
    run = make_run("r1", [
        user("Please archive entry E-100."),
        call("archive_entry", {"entry_id": "E-100"}, cid="c1"),
        result({"entry_id": "E-100", "archived": True}, cid="c1"),
        assistant("Done."),
    ])
    assert T.hard_defect(hard, run, WRITE_TOOLS) is True
    assert S.check_run(verifier, run, write_tools=WRITE_TOOLS) == (True, None)
    gates = {g.stage: g for g in S.validate_verifier(verifier, run, write_tools=WRITE_TOOLS)}
    assert gates["verifier_oracle"].passed is False


def test_the_gates_answers_on_plain_write_atoms_are_unchanged():
    """No Hard defect and no grounding gap: the moved scorer answers as the suite always did."""
    fn = T.canon_fn(None)
    entity = T.text_of(fn("E-100"))
    atom = S.make_atom("w0", "required", {"kind": "write", "tool": "archive_entry",
                                          "entity": entity, "entity_raw": "E-100",
                                          "id_field": "entry_id"})
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[atom])
    good = make_run("good", [
        user("Please archive entry E-100."),
        call("archive_entry", {"entry_id": "E-100"}, cid="c1"),
        result({"entry_id": "E-100", "archived": True}, cid="c1"),
        assistant("Done."),
    ])
    bad = make_run("bad", [
        user("Please archive entry E-100."),
        call("archive_entry", {"entry_id": "E-777"}, cid="c1"),
        result({"entry_id": "E-777", "archived": True}, cid="c1"),
        assistant("Done."),
    ])
    assert S.check_run(verifier, good, write_tools=WRITE_TOOLS) == (True, None)
    assert S.check_run(verifier, bad, write_tools=WRITE_TOOLS) == (False, "w0")
    assert verdict(good, verifier, write_tools=WRITE_TOOLS).passed is True
    assert verdict(bad, verifier, write_tools=WRITE_TOOLS).passed is False
    assert _evaluate(atom.predicate_src, {"__builtins__": {}, "wrote": lambda *a, **k: True}) is True


def test_a_judged_run_still_names_and_records_its_unsettled_semantic_pair(tmp_path):
    """No short-circuit: with judge_used already True, the comparisons pass must still name
    unsettled pairs and record their use, in the original note order."""
    from kullback.runner.records import Column, EntitySchema

    run = make_run("r-combined", [
        user("Please cancel order W123."),
        assistant("Done."),
        _ev("stop", termination_reason="done",
            start_state={"orders": {"W123": {"status": "pending", "note": "cancelled by the user"}}},
            end_state={"orders": {"W123": {"status": "cancelled", "note": "cancelled by user"}}}),
    ])
    schema = EntitySchema(tables=["orders"], columns=[
        Column(table="orders", name="status", **{"class": "hard"}),
        Column(table="orders", name="note", **{"class": "semantic"})])
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[
        Atom(id="j_tone", kind="required", judge=True, description="tone fit"),
        Atom(id="a_note", kind="required",
             predicate_src='"note" not in diff()["orders.W123"]["fields"]'),
    ])
    out = verdict(run, verifier, judge_results={"j_tone": {"verdict": "abstain"}},
                  schema=schema, workdir=tmp_path)
    assert out.judge_used is True
    assert "judge_abstained:j_tone" in out.notes
    unresolved = [note for note in out.notes if note.startswith("semantic_unresolved:")]
    assert len(unresolved) == 1
    assert out.notes.index("judge_abstained:j_tone") < out.notes.index(unresolved[0])
    uses = (tmp_path / "equivalence_uses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(uses) == 1
