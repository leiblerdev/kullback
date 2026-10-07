"""The Examiner's ruling tools: a ruling names an item, a kind and code, a reason and a fix; it never edits (D331)."""

from __future__ import annotations

import asyncio
import json

import pytest

from kullback.agent.bus import Bus
from kullback.examiner import domain_tools as D
from kullback.examiner import prompt as P
from kullback.examiner import rule_tool as R
from kullback.examiner.exam_files import ExamRoot
from kullback.spec import events
from kullback.spec import rulings as RL
from kullback.spec.schema import load_spec, save_spec
from tests.spec.fixtures import WRITE_TOOLS, reference_run, spec

SIGS = [{"name": name, "kind": "write"} for name in WRITE_TOOLS]
TURN = "Please move item A1 to slot seven."


def keep_case(value):
    """A canoniser that keeps case: the Spec compiles entities as written (see the report)."""
    return str(value).strip()


def _root(tmp_path, runs=None, with_spec=True):
    runs = [reference_run()] if runs is None else runs
    rows = {}
    for run in runs:
        path = tmp_path / "runs" / f"{run.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run.model_dump(mode="json", by_alias=True)) + "\n")
        rows[run.run_id] = {"run_id": run.run_id, "path": f"runs/{run.run_id}.json"}
    if with_spec:
        save_spec(tmp_path, spec())
    return ExamRoot(workdir=tmp_path, replays={"t1": rows}, sigs=SIGS, canon_rules=keep_case,
                    bus=Bus(tmp_path / "bus.jsonl", agent="examiner"))


GOOD = {"item": "c1", "kind": "derivation", "code": "fixed_value", "blocking": True,
        "reason": "The check fixes one slot where the policy allows any free slot.",
        "fix": "Accept any free slot as a value set instead of one value."}


def _rule(root, **kwargs):
    return asyncio.run(R._rule(root)(R.RuleArgs(task_id="t1", **dict(GOOD, **kwargs))))


def _close(root, **kwargs):
    return asyncio.run(R._close(root)(R.CloseArgs(task_id="t1", **kwargs)))


def _filed(root):
    return [r.event.payload for r in root.bus.replay() if getattr(r.event, "name", "") == events.RULING_FILED]


def test_a_ruling_is_written_whole_counted_on_the_spec_and_published_without_its_sentences(tmp_path):
    root = _root(tmp_path)
    result = _rule(root)
    assert result.number == 1 and "blocking" in result.summary
    body = json.loads((tmp_path / "exam" / "rulings" / "t1" / "1.json").read_text())
    assert body["reason"] == GOOD["reason"] and body["fix"] == GOOD["fix"] and body["status"] == "open"
    assert load_spec(tmp_path, "t1").rulings_open == 1
    [payload] = _filed(root)
    assert payload == {"task_id": "t1", "number": 1, "round": 1, "item": "c1", "kind": "derivation",
                       "code": "fixed_value", "blocking": True}


def test_a_note_never_counts_as_open_and_rulings_number_up_per_task(tmp_path):
    root = _root(tmp_path)
    _rule(root, blocking=False)
    _rule(root, item="f2", code="uncovered_fact")
    assert [r["number"] for r in R.read_rulings(tmp_path, "t1")] == [1, 2]
    assert load_spec(tmp_path, "t1").rulings_open == 1


@pytest.mark.parametrize("kwargs, message", [
    ({"item": "c9"}, "is not on the Task"),
    ({"code": "anchor_wrong"}, "a derivation ruling's code is one of"),
    ({"reason": "Bad."}, "the reason is one sentence"),
    ({"fix": "Drop it.\nThen add another."}, "the fix is one sentence"),
    ({"kind": "code", "code": "passes_do_nothing"}, "call verify on task t1 first"),
])
def test_a_ruling_missing_what_it_needs_is_refused_and_counted(tmp_path, kwargs, message):
    root = _root(tmp_path)
    with pytest.raises(ValueError, match=message):
        _rule(root, **kwargs)
    rows = (tmp_path / "exam" / "rulings" / R.REFUSED_FILE).read_text().splitlines()
    assert len(rows) == 1 and GOOD["reason"] not in rows[0]
    assert not _filed(root)


def test_a_task_without_a_spec_takes_no_ruling(tmp_path):
    with pytest.raises(ValueError, match="no Spec"):
        _rule(_root(tmp_path, with_spec=False))


def test_a_reference_ruled_wrong_is_a_note_carrying_the_verify_rows(tmp_path):
    root = _root(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "i0"}]
    with pytest.raises(ValueError, match="wrong_side"):
        _rule(root, kind="code", code="fails_reference")
    _rule(root, kind="code", code="fails_reference", wrong_side="reference")
    [ruling] = RL.load_rulings(tmp_path, "t1")
    assert ruling.blocking is False and ruling.verified == root.verified["t1"]
    assert load_spec(tmp_path, "t1").rulings_open == 0


def test_close_needs_a_later_round_and_keeping_open_holds_the_task(tmp_path):
    root = _root(tmp_path)
    _rule(root)
    _rule(root, item="f2", code="uncovered_fact")
    with pytest.raises(ValueError, match="filed this round"):
        _close(root, number=1, keep_open=False, why="The writer accepted the value set as asked.")
    RL.answer(tmp_path, "t1", 1, "applied", "value set", 2, ["c1"])
    RL.answer(tmp_path, "t1", 2, "rebutted", "no fact asks it", 2)
    root.round = 2
    _close(root, number=1, keep_open=False, why="The writer accepted the value set as asked.")
    _close(root, number=2, keep_open=True, why="The second fact still has no item checking it.")
    rulings = RL.load_rulings(tmp_path, "t1")
    assert [r.status for r in rulings] == ["closed", "open"] and rulings[1].closed["kept_open"]
    assert load_spec(tmp_path, "t1").rulings_open == 1
    with pytest.raises(ValueError, match="already closed"):
        _close(root, number=1, keep_open=True, why="Second thoughts about the closed one.")


def test_the_view_holds_the_items_with_provenance_the_uncovered_facts_and_the_rulings(tmp_path):
    root = _root(tmp_path)
    _rule(root)
    view = R.spec_view(root, "t1")
    assert [f["stance"] for f in view["facts"]] == ["volunteered", "accepted"]
    assert view["items"][0]["fact_ids"] == ["f1"] and view["items"][0]["because"]
    assert view["facts_without_item"] == ["f2"]
    assert [r["number"] for r in view["rulings"]] == [1] and "runs" not in view


def test_the_examiner_offers_its_ruling_tools_and_no_edit_tool(tmp_path):
    names = [tool.name for tool in D.domain_tools(_root(tmp_path))]
    assert names == list(D.tool_names())
    assert names[:5] == ["rule", "close", "verify", "lookup_rows", "search_rows"]
    assert not {"reject_reference", "edit_verifier"} & set(names)


def test_the_prompt_names_every_offered_tool_every_ruling_kind_and_no_edit_tool():
    text = P.render(P.sections(D.tool_names()))
    for name in D.tool_names():
        assert f"{name}:" in text
    for kind, codes in RL.RULING_KINDS.items():
        assert kind in text and all(code in text for code in codes)
    assert not any(f"{name}:" in text for name in ("edit_verifier", "propose_verifier", "try_atoms", "probe",
                                                    "reroll", "reject_reference"))
    assert chr(0x2014) not in text and chr(0x2013) not in text
