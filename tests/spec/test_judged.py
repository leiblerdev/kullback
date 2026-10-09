"""Trust reads the judge answers kept for each Run: verify's for the Reference, the score stage's Verdict
for a fresh Run (D347)."""

from __future__ import annotations

from kullback.runner.records import Atom, Run, Verifier, write_json
from kullback.spec import judged as J

ITEM = Atom(id="j1", kind="allowed", judge=True, gate=True,
            target={"question": "Did the agent name the slot?", "anchor": {"answer": "seven"}})
VERIFIER = Verifier(task_id="t1", atoms=[ITEM])


def _run(run_id):
    return Run.model_validate({"run_id": run_id, "task_id": "t1", "events": []})


def _saved_verdict(tmp_path):
    item = {"id": "j1", "kind": "judge", "gate": True, "score": 1, "why": "named"}
    other = {"id": "s1", "kind": "state", "gate": True, "holds": True}
    write_json(tmp_path / "verdicts" / "t1" / "fresh-0.aaaa.json", {"run_id": "fresh-0", "items": [item, other]})
    return tmp_path / "verdicts" / "t1"


def test_a_fresh_run_reads_the_judge_items_of_its_newest_saved_verdict(tmp_path):
    J.record_scored(_saved_verdict(tmp_path), ["fresh-0"], VERIFIER)
    assert J.judge_results_of(tmp_path, "t1", VERIFIER, [_run("fresh-0"), _run("fresh-1")]) == {
        "fresh-0": {"j1": {"score": 1, "why": "named"}}}


def test_the_reference_reads_verifys_kept_answers_and_a_changed_verifier_reads_none(tmp_path):
    def judge(verifier, run):
        return {"j1": {"score": 0, "why": "not named"}}

    run = _run("ref")
    first = J.judge_reference(tmp_path, "t1", VERIFIER, run, judge)
    assert first["j1"]["score"] == 0 and J.judge_results_of(tmp_path, "t1", VERIFIER, [run])["ref"] == first
    changed = VERIFIER.model_copy(update={"atoms": [ITEM.model_copy(update={"id": "j2"})]})
    assert J.judge_results_of(tmp_path, "t1", changed, [run]) == {}


def test_a_failed_judge_call_is_not_kept_so_the_next_verify_asks_again(tmp_path):
    answers = J.judge_reference(tmp_path, "t1", VERIFIER, _run("ref"), lambda v, r: {"j1": {"score": None}})
    assert answers["j1"]["score"] is None and J.cached_judge(tmp_path, "t1", VERIFIER, "ref") is None


def test_a_saved_verdict_scored_under_another_verifier_lends_trust_no_judge_answer(tmp_path):
    asked_before = VERIFIER.model_copy(update={"atoms": [ITEM.model_copy(update={"target": {
        "question": "Did the agent name the old slot?", "anchor": {"answer": "two"}}})]})
    folder = _saved_verdict(tmp_path)
    J.record_scored(folder, ["fresh-0"], asked_before)
    assert J.judge_results_of(tmp_path, "t1", VERIFIER, [_run("fresh-0")]) == {}


def test_a_saved_verdict_with_no_record_of_its_verifier_lends_trust_no_judge_answer(tmp_path):
    _saved_verdict(tmp_path)
    assert J.saved_verdict_judges(tmp_path, "t1", "fresh-0", VERIFIER) is None
