"""The Spec writes the whole Verifier; the Examiner rules, the writer applies or rebuts, in rounds (D320, D331).

A tiny invented workdir: one Task whose user asks to move one item to a new slot and rename it; the
Spec's one state item carries both values, said by the user, so its expected end state has two
sourced cells. The Reference on disk is a witness only: nothing here derives from it.
"""

from __future__ import annotations

import json

import pytest

from kullback.ai.provider import ModelReply, TestModel
from kullback.examiner.rule_tool import verify_line, verify_rows
from kullback.runner.records import Run, Verifier, write_json
from kullback.spec import review as RV
from kullback.spec import rounds as RD
from kullback.spec import rulings as RL
from kullback.spec import writer as W
from kullback.spec.canfail import judge_run
from kullback.spec.schema import FactSource, IntentFact, intent_text, load_spec
from kullback.spec.trust import NOT_EXAMINED, constructed_runs, tier_of_task
from tests.spec.fixtures import FACTS, check, fn
from tests.spec.fixtures import spec as _spec
from tests.spec.test_writer import _workdir as _environment
from tests.spec.test_writer import call
from tests.spec.test_writer import inputs as _inputs

TURN = "Please move item A1 to slot seven and call it blue."
START = {"items": {"A1": {"item_id": "A1", "slot": "two", "label": "red"},
                   "B2": {"item_id": "B2", "slot": "four", "label": "green"}}}
END = {"items": {"A1": {"item_id": "A1", "slot": "seven", "label": "blue"},
                 "B2": {"item_id": "B2", "slot": "four", "label": "green"}}}
ASKED = IntentFact(id="f3", text=TURN, stance="volunteered", source=FactSource(recording="rec1", turn=0))
ROW = {"id": "s1", "kind": "row_is", "table": "items", "find": {"item_id": "A1"}, "row": "A1",
       "expect": {"slot": "seven", "label": "blue"}, "gate": True, "weight": 1.0, "fact_ids": ["f3"],
       "policy_line": None}


def inputs():
    return _inputs(facts=[*FACTS, ASKED])


def spec(checks=None):
    """The Spec whose one item moves and renames A1, beside any older checks given; its Intent has f3."""
    out = _spec(checks=[W.check_of(ROW, inputs())] + list(checks or []))
    facts = [*FACTS, ASKED]
    return out.model_copy(update={"intent": out.intent.model_copy(update={"facts": facts, "text": intent_text(facts)}),
                                  "gaps": ["f1", "f2"]})


SAID = {"id": "said_slot", "kind": "required",
        "payload": {"kind": "communicate", "text": "slot seven", "id": "said_slot"}}


def _reference() -> Run:
    events = [{"idx": 0, "type": "user_turn", "payload": {"content": TURN}},
              {"idx": 1, "type": "tool_call", "payload": {"id": "c1", "name": "update_item", "kind": "write",
                                                          "args": {"item_id": "A1", "slot": "seven",
                                                                   "label": "blue"}}},
              {"idx": 2, "type": "tool_result", "payload": {"id": "c1", "result": END["items"]["A1"]}},
              {"idx": 3, "type": "model_call", "payload": {"reply": {"content": "Item A1 is in slot seven."}}},
              {"idx": 4, "type": "stop", "payload": {"start_state": START, "end_state": END}}]
    return Run.model_validate({"run_id": "ref", "task_id": "t1", "trace_id": "rec1",
                               "termination_reason": "success", "events": events})


def _workdir(root):
    """A minimal Environment with the Task's Reference on disk the way the Reference stage leaves it."""
    _environment(root)
    (root / "verifiers" / "t1.json").unlink()
    path = root / "runs" / "t1" / "ref.jsonl"
    path.write_text(_reference().model_dump_json() + "\n")
    write_json(root / "references.json", {"t1": {"references": [{"kind": "recording", "run_id": "ref",
                                                                  "trace_id": "rec1"}]}})
    write_json(root / "replays.json", {"t1": {"rec1": {"run_id": "ref", "trace_id": "rec1", "confirmed": True,
                                                       "path": "runs/t1/ref.jsonl"}}})
    return root


def _written(tmp_path):
    root = _workdir(tmp_path)
    W.write_verifier(spec(), root, inputs())
    return root, Verifier.model_validate_json(W.runner_verifier_path(root, "t1").read_text())


def test_write_verifier_puts_the_items_two_sourced_cells_in_expected_and_no_write_predicate(tmp_path):
    root, verifier = _written(tmp_path)
    [state] = verifier.expected
    cells = {cell.field: cell for cell in state.cells}
    assert set(cells) == {"slot", "label"}
    assert all(cell.source is not None and cell.source.kind == "user_turn" for cell in cells.values())
    assert not any(atom.target.get("kind") in {"write_value", "entity_count"} for atom in verifier.atoms)
    assert not any(getattr(atom, "predicate_src", None) for atom in verifier.atoms)
    counts = load_spec(root, "t1").end_state
    assert (counts["end_states"], counts["cells"], counts["gate"]) == (1, 2, 2) and "reference" not in counts


def test_a_verifier_with_end_state_gates_passes_its_reference_and_fails_both_constructed_runs(tmp_path):
    _, verifier = _written(tmp_path)
    assert judge_run(verifier, _reference()) == (True, None)
    built = constructed_runs(verifier, [_reference()], [_reference().run_id])
    assert sorted(built) == ["do_nothing", "stray_write"] and all(built.values())
    assert not any(judge_run(verifier, run)[0] for run in built.values())


TOLD = check("c2", demand={"demand": "say", "text": "slot seven"}, because="move item A1 to slot seven")


def _told(tmp_path):
    """The workdir with a Verifier of two cells and one fact-told atom, i0, made by the older check c2."""
    root = _workdir(tmp_path)
    W.write_verifier(spec(checks=[TOLD]), root, inputs())
    return root


def _both_copies(root) -> Verifier:
    text = W.runner_verifier_path(root, "t1").read_text()
    assert (root / "spec" / "verifiers" / "t1.json").read_text() == text
    return Verifier.model_validate_json(text)


def _ruling(root, number=1, item="c2", kind="derivation", code="unasked_item", blocking=True, round_number=1):
    ruling = RL.Ruling(task_id="t1", number=number, round=round_number, item=item, kind=kind, code=code,
                       reason="No Intent fact asks the agent to tell the slot.", fix="Drop the check c2.",
                       blocking=blocking)
    RL.save_ruling(root, ruling)
    RL.sync_spec(root, "t1")
    return ruling


def test_a_ruling_names_the_checks_the_writer_may_change_for_a_check_an_atom_or_a_fact(tmp_path):
    root = _told(tmp_path)
    spec_now, verifier = load_spec(root, "t1"), _both_copies(root)
    [atom] = [a.id for a in verifier.atoms if a.id.startswith("i")]
    assert RV.ruled_checks(spec_now, verifier, "c2") == ["c2"]
    assert RV.ruled_checks(spec_now, verifier, atom) == ["c2"]
    assert RV.ruled_checks(spec_now, verifier, "f1") == ["c2"]
    assert RV.ruled_checks(spec_now, verifier, "f3") == ["s1"]
    assert RV.ruled_checks(spec_now, verifier, RL.TASK_ITEM) == []


def test_an_applied_ruling_rewrites_both_verifier_copies_and_a_rebutted_one_keeps_them(tmp_path):
    root = _told(tmp_path)
    _ruling(root, 1)
    model = TestModel([call("drop_items", ids=["c2"]), ModelReply(content="done")])
    before = _both_copies(root)
    assert RV.answer_rulings(root, "t1", model, 2)["applied"] == 1
    _ruling(root, 2, item="s1", code="fixed_value")
    out = RV.answer_rulings(root, "t1", TestModel([ModelReply(content="rebut: The user named slot seven.")]), 2)
    assert out == {"task_id": "t1", "answered": 1, "applied": 0, "rebutted": 1, "skipped": 0}
    first, second = RL.load_rulings(root, "t1")
    assert first.answer["action"] == "applied" and first.answer["changed"] == ["c2"]
    assert second.answer["action"] == "rebutted" and second.answer["why"] == "The user named slot seven."
    assert second.answer["changed"] == []
    assert [c.id for c in load_spec(root, "t1").checks] == ["s1"]
    assert _both_copies(root).atoms != before.atoms
    assert load_spec(root, "t1").rulings_open == 2  # answered is not closed: the Examiner closes


def test_an_empty_rebuttal_keeps_what_code_refused_so_it_reads_apart_from_a_reasoned_one(tmp_path):
    root = _told(tmp_path)
    _ruling(root, 1)
    RV.answer_rulings(root, "t1", TestModel([call("drop_items", ids=["c2"]), ModelReply(content="done")]), 2)
    _ruling(root, 2, item="s1", code="fixed_value")
    unfound = dict(ROW, expect={"slot": "nine_42"})
    twice = [call("add_items", items=[unfound]), ModelReply(content="done")] * 2
    RV.answer_rulings(root, "t1", TestModel(twice), 2)
    ruling = RL.load_rulings(root, "t1")[1]
    assert ruling.answer["action"] == "rebutted" and ruling.answer["why"] == RV.REBUT_NONE
    assert ruling.answer["writer"]["refused"] == 2 and ruling.answer["writer"]["retried"] == 1
    assert ruling.answer["writer"]["refusals"] == [{"id": "s1", "kind": "row_is"}] * 2


def test_a_ruling_that_the_reference_is_wrong_asks_nothing_of_the_writer(tmp_path):
    root = _told(tmp_path)
    RL.save_ruling(root, RL.Ruling(task_id="t1", number=1, round=1, item="c1", kind="code", code="fails_reference",
                                   reason="The Reference wrote a slot the user never asked for.",
                                   fix="Keep the check; the Reference is wrong.", blocking=False,
                                   wrong_side="reference"))
    out = RV.answer_rulings(root, "t1", TestModel([]), 2)
    assert out["answered"] == 0 and out["skipped"] == 1


def test_verify_says_the_do_nothing_run_fails_and_the_reference_passes(tmp_path):
    root, verifier = _written(tmp_path)
    rows = verify_rows(root, "t1", verifier, _reference())
    by_run = {row["run"]: row for row in rows}
    assert by_run["do_nothing"]["verifier_fails_it"] is True and by_run["stray_write"]["verifier_fails_it"] is True
    assert by_run["reference"]["verifier_passes_it"] is True
    assert verify_line(rows).startswith("do_nothing: the Verifier fails 1 of 1;") and "reference: passes" in verify_line(rows)
    assert verify_rows(root, "t1", verifier, None) == [{"run": "reference", "error": "no faithful Reference on disk"}]


def test_the_loop_runs_examiner_writer_examiner_and_a_ruling_kept_open_holds_the_task(tmp_path):
    root = _told(tmp_path)
    seen = []

    def examine(workdir, task_ids, *, model, round_number):
        seen.append(("examiner", round_number))
        if round_number == 1:
            _ruling(workdir, 1)
            _ruling(workdir, 2, item="f2", code="uncovered_fact", blocking=False)
        else:
            RL.close(workdir, "t1", 1, True, "The check still tells a value nobody asked for.", round_number)
        return {"capped": False, "counts": RL.counts(r for r in RL.load_rulings(workdir, "t1")
                                                      if r.round == round_number)}

    def answer(workdir, task_id, model, round_number, ceiling_usd=None):
        seen.append(("writer", round_number))
        for ruling in RL.unanswered(RL.load_rulings(workdir, task_id)):
            RL.answer(workdir, task_id, ruling.number, "rebutted", "The policy asks it.", round_number)
        return {"answered": 2, "applied": 0, "rebutted": 2, "skipped": 0}

    out = RD.run_rounds(root, ["t1"], examine=examine, model=None, answer=answer)
    assert seen == [("examiner", 1), ("writer", 2), ("examiner", 2)]
    assert out["held"] == ["t1"] and out["tasks"]["t1"]["open_blocking"] == 1
    assert out["total"]["by_kind"] == {"derivation": 2} and out["total"]["kept_open"] == 1
    spec_now = load_spec(root, "t1")
    assert spec_now.round == 2 and spec_now.rulings_open == 1
    logged = [json.loads(line) for line in RD.log_path(root).read_text().splitlines()]
    assert [row["side"] for row in logged] == ["examiner", "writer", "examiner", "summary"]


def test_the_loop_stops_after_one_examiner_round_when_nothing_is_ruled(tmp_path):
    root = _told(tmp_path)
    calls = []
    out = RD.run_rounds(root, ["t1"], model=None,
                        examine=lambda w, t, *, model, round_number: calls.append(round_number) or {"counts": {}},
                        answer=lambda *a, **k: calls.append("writer"))
    assert calls == [1] and out["held"] == []



def test_the_confirm_question_is_asked_before_the_examiners_first_round(tmp_path):
    root = _told(tmp_path)
    calls = []
    RD.run_rounds(root, ["t1"], model="m",
                  confirm=lambda w, t, model, ceiling_usd=None: calls.append(("confirm", t)) or {"asked": True, "added": 1},
                  examine=lambda w, t, *, model, round_number: calls.append(("examiner", round_number)) or {"counts": {}},
                  answer=lambda *a, **k: calls.append("writer"))
    assert calls == [("confirm", "t1"), ("examiner", 1)]
    logged = [json.loads(line) for line in RD.log_path(root).read_text().splitlines()]
    assert logged[0] == {"side": "writer", "confirm_asked": ["t1"], "confirm_added": 1}


def _trust_tier(root, verifier):
    """The Task tier under the Spec file as the last round left it, scored over its Reference."""
    return tier_of_task(load_spec(root, "t1"), verifier, [_reference()], (), canon=fn,
                        write_tools={"update_item"}, references=("ref",))


def test_a_task_no_round_examined_is_untrusted_as_not_examined(tmp_path):
    root, verifier = _written(tmp_path)
    assert load_spec(root, "t1").examined_rounds == []
    tier, row = _trust_tier(root, verifier)
    assert tier == "untrusted" and row["reason"] == NOT_EXAMINED


def test_an_examined_task_with_no_open_ruling_keeps_its_tier(tmp_path):
    root, verifier = _written(tmp_path)
    RL.sync_spec(root, "t1", 1)
    assert load_spec(root, "t1").examined_rounds == [1]
    tier, row = _trust_tier(root, verifier)
    assert tier == "trusted" and row["reason"] is None


def _unfounded_ruling(root):
    return _ruling(root, 1, item="f3", kind="scope", code=RL.UNFOUNDED_CODE)


def test_an_applied_intent_unfounded_ruling_stops_the_fact_backing_a_gate(tmp_path):
    root, _ = _written(tmp_path)
    _unfounded_ruling(root)
    regate = [call("drop_items", ids=["s1"]), call("add_items", items=[ROW]), ModelReply(content="done")]
    assert RV.answer_rulings(root, "t1", TestModel(regate), 2)["applied"] == 1
    after = load_spec(root, "t1")
    assert [f.id for f in after.intent.facts if f.unfounded] == ["f3"] and after.checks == []
    store = W.store_of(inputs()._replace(intent=after.intent))
    assert "unfounded facts" in store.add([ROW]) and not store.items
    assert store.add([dict(ROW, gate=False, expect={"slot": "seven"})]) == "s1: kept (row A1)"


def test_a_rebutted_intent_unfounded_ruling_leaves_the_fact_founded(tmp_path):
    root, _ = _written(tmp_path)
    _unfounded_ruling(root)
    RV.answer_rulings(root, "t1", TestModel([ModelReply(content="rebut: The user said it in turn 0.")]), 2)
    assert not any(f.unfounded for f in load_spec(root, "t1").intent.facts)


def _no_truth(root, blocking=False):
    from kullback.examiner.exam_files import ExamRoot
    from kullback.examiner.rule_tool import RuleArgs, file_ruling

    return file_ruling(ExamRoot(workdir=root, round=1), RuleArgs(
        task_id="t1", item=RL.TASK_ITEM, kind="scope", code=RL.NO_TRUTH_CODE, blocking=blocking,
        reason="No recording shows what the user wants once the slot is taken.",
        fix="Gate the move to slot seven anyway."))


def test_an_open_not_enough_truth_ruling_leaves_the_task_untrusted_with_that_reason(tmp_path):
    from kullback.spec.report import build_report
    from kullback.spec.trust import NOT_ENOUGH_TRUTH

    root, verifier = _written(tmp_path)
    filed = _no_truth(root)
    assert filed["blocking"] is True and filed["fix"] == RL.NO_TRUTH_FIX
    RL.sync_spec(root, "t1", 1)
    assert load_spec(root, "t1").truth_open == 1
    tier, row = _trust_tier(root, verifier)
    assert tier == "untrusted" and row["reason"] == NOT_ENOUGH_TRUTH
    assert build_report({"t1": _trust_tier(root, verifier)})["reasons"][NOT_ENOUGH_TRUTH] == 1
    RL.answer(root, "t1", 1, "rebutted", "The user says in turn 0 any free slot will do.", 2)
    RL.close(root, "t1", 1, False, "The cited turn settles what the user wants.", 2)
    RL.sync_spec(root, "t1", 2)
    assert _trust_tier(root, verifier)[1]["reason"] is None


def test_the_writer_cannot_apply_not_enough_truth(tmp_path):
    root, _ = _written(tmp_path)
    _no_truth(root)
    before = _both_copies(root)
    out = RV.answer_rulings(root, "t1", TestModel([call("drop_items", ids=["s1"]), ModelReply(content="done")]), 2)
    assert out["applied"] == 0 and out["rebutted"] == 1
    [ruling] = RL.load_rulings(root, "t1")
    assert ruling.answer["action"] == "rebutted" and ruling.answer["changed"] == [] and ruling.status == "open"
    assert [c.id for c in load_spec(root, "t1").checks] == ["s1"] and _both_copies(root) == before
    with pytest.raises(ValueError, match="cannot be applied"):
        RL.answer(root, "t1", 1, "applied", "Kept the move as asked.", 2)
