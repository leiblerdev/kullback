"""The Examiner's rule tool: a ruling quotes its evidence, lands in its file and on the bus, or is refused."""

from __future__ import annotations

import asyncio
import json

import pytest

from kullback.agent.bus import Bus
from kullback.examiner import domain_tools as D
from kullback.examiner import prompt as P
from kullback.examiner import rule_tool as R
from kullback.examiner.exam_files import ExamRoot
from kullback.runner.records import Event, Run
from kullback.spec import events
from kullback.spec.schema import save_spec
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


def _rule(root, **kwargs):
    return asyncio.run(R._rule(root)(R.RuleArgs(task_id="t1", **kwargs)))


def _filed(root):
    return [r.event.payload for r in root.bus.replay() if getattr(r.event, "name", "") == events.RULING_FILED]


def test_a_ruling_quoting_a_because_is_written_whole_and_published_redacted(tmp_path):
    root = _root(tmp_path)
    result = _rule(root, target="check", check_ids=["c1"], code="check_unpassable",
                   reason='The check rests on "move item A1 to slot seven" alone.')
    assert result.number == 1 and result.quoted == ["because c1"]
    body = json.loads((tmp_path / "exam" / "rulings" / "t1" / "1.json").read_text())
    assert "move item A1 to slot seven" in body["reason"]
    [payload] = _filed(root)
    assert payload["target"] == "check" and payload["check_ids"] == ["c1"] and payload["round"] == 0
    assert payload["reason"] == "check_unpassable" and payload["quoted"] == ["[because c1]"]


def test_a_ruling_on_a_run_may_quote_one_of_its_turns(tmp_path):
    root = _root(tmp_path)
    result = _rule(root, target="run", run_id="ref", reason=f'The user said "{TURN}" and it did more.')
    assert result.quoted == ["turn 0"]


def test_a_reason_without_a_verbatim_quote_is_refused_and_counted(tmp_path):
    root = _root(tmp_path)
    with pytest.raises(ValueError, match="quotes no evidence"):
        _rule(root, target="check", check_ids=["c1"], reason="this check is simply wrong")
    assert not _filed(root)
    refused = (tmp_path / "exam" / "rulings" / R.REFUSED_FILE).read_text().splitlines()
    [row] = [json.loads(line) for line in refused]
    assert row["code"] == events.QUOTE_MISSING and row["check_ids"] == ["c1"] and len(row["reason_sha256"]) == 64


def test_no_sentence_of_a_reason_reaches_the_bus_or_the_refusal_log(tmp_path):
    root = _root(tmp_path)
    sentence = "Zebras hum quietly behind the ninth lantern"
    _rule(root, target="check", check_ids=["c1"], code="other",
          reason=f'{sentence}; the because is "move item A1 to slot seven".')
    with pytest.raises(ValueError):
        _rule(root, target="intent", reason=sentence)
    assert sentence in (tmp_path / "exam" / "rulings" / "t1" / "1.json").read_text()
    for path in (tmp_path / "bus.jsonl", tmp_path / "exam" / "rulings" / R.REFUSED_FILE):
        text = path.read_text()
        assert "Zebras" not in text and "lantern" not in text and "slot seven" not in text


@pytest.mark.parametrize("kwargs, message", [
    ({"target": "run", "reason": TURN}, "names it in run_id"),
    ({"target": "run", "run_id": "nope", "reason": TURN}, "not a Run of this Task"),
    ({"target": "check", "reason": "move item A1 to slot seven"}, "names it in check_ids"),
    ({"target": "check", "check_ids": ["c9"], "reason": "move item A1 to slot seven"}, "not on the Task's Spec"),
    ({"target": "intent", "excerpt": "words nobody wrote down", "reason": "move item A1 to slot seven"},
     "excerpt is not verbatim"),
])
def test_a_ruling_that_names_what_the_task_lacks_is_refused(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _rule(_root(tmp_path), **kwargs)


def test_a_task_without_a_spec_takes_no_ruling(tmp_path):
    with pytest.raises(ValueError, match="no Spec"):
        _rule(_root(tmp_path, with_spec=False), target="intent", reason="move item A1 to slot seven")


def test_rulings_number_up_per_task(tmp_path):
    root = _root(tmp_path)
    for _ in range(2):
        _rule(root, target="intent", reason="move item A1 to slot seven")
    assert [r["number"] for r in R.read_rulings(tmp_path, "t1")] == [1, 2]


def test_the_view_names_the_checks_each_run_fails_with_the_expected_values(tmp_path):
    empty = Run(run_id="empty", task_id="t1", termination_reason="success",
                events=[Event(idx=0, type="user_turn", payload={"content": TURN})])
    root = _root(tmp_path, runs=[reference_run(), empty])
    view = R.spec_view(root, "t1")
    assert [f["stance"] for f in view["facts"]] == ["volunteered", "accepted"]
    assert view["checks"][0]["because"] and view["checks"][0]["tier"] == "critical"
    rows = {row["run_id"]: row for row in view["runs"]}
    assert rows["ref"]["passed"] is True
    [failed] = rows["empty"]["failed"]
    assert failed["check"] == "c1" and {"field": "slot", "expected": "seven"} in \
        [{k: v for k, v in value.items() if k != "atom"} for value in failed["values"]]


def test_the_examiner_offers_its_review_tools_and_no_edit_tool(tmp_path):
    names = [tool.name for tool in D.domain_tools(_root(tmp_path))]
    assert names == list(D.tool_names())
    assert names == ["rule", "finding", "no_finding", "check_reference", "reject_reference"]


def test_reject_reference_files_a_ruling_on_the_run(tmp_path):
    root = _root(tmp_path)
    reject = next(t for t in D.domain_tools(root) if t.name == "reject_reference")
    from kullback.examiner.reference_check import RejectReferenceArgs

    asyncio.run(reject.execute(RejectReferenceArgs(task_id="t1", run_id="ref", why=f"the user said \"{TURN}\"")))
    [payload] = _filed(root)
    assert payload["target"] == "run" and payload["run_id"] == "ref" and payload["reason"] == "reference_wrong"


def test_the_prompt_names_every_offered_tool_and_no_edit_tool():
    text = P.render(P.sections())
    for name in D.tool_names():
        assert f"{name}:" in text
    assert not any(f"{name}:" in text for name in ("edit_verifier", "propose_verifier", "try_atoms", "probe",
                                                    "reroll"))
    assert chr(0x2014) not in text and chr(0x2013) not in text
