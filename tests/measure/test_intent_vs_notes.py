"""The note measurement counts recall, precision, rule 5 and steering leaks, and prints no note text."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kullback.spec.schema import FactSource, IntentFact, Spec, SpecIntent, save_spec
from tests.spec.test_intent import RECORDINGS, _trace

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure" / "intent_vs_notes.py"
spec_ = importlib.util.spec_from_file_location("intent_vs_notes", SCRIPT)
M = importlib.util.module_from_spec(spec_)
sys.modules["intent_vs_notes"] = M
spec_.loader.exec_module(M)

NOTE = {"persona": "Calm and brief.",
        "instructions": {"domain": "d", "reason_for_call": "You want the blue lamp order moved to Oak street.",
                         "known_info": "Your card ends in 44.", "unknown_info": "You do not know the order id.",
                         "task_instructions": "Refuse any store credit."}}


def _fact(fid, text, recording="rec-a", turn=1, stance="volunteered", witnesses=()):
    return IntentFact(id=fid, text=text, stance=stance, witnesses=list(witnesses),
                      source=FactSource(recording=recording, turn=turn))


@pytest.fixture
def workdir(tmp_path):
    for folder in ("tasks", "traces", "raw", "grader"):
        (tmp_path / folder).mkdir()
    (tmp_path / "tasks" / "t1.json").write_text(json.dumps({"id": "t1", "run_ids": list(RECORDINGS)}))
    for trace_id, turns in RECORDINGS.items():
        (tmp_path / "traces" / f"{trace_id}.json").write_text(json.dumps(_trace(trace_id, turns)))
        (tmp_path / "grader" / f"{trace_id}.json").write_text(
            json.dumps({"trace_id": trace_id, "fields": {"task_id": "7"}}))
    (tmp_path / "raw" / "r.json").write_text(json.dumps({"tasks": [{"id": "7", "user_scenario": NOTE}]}))
    facts = [_fact("f1", "move the blue lamp order to Oak street"),
             _fact("f2", "refund the shipping fee to the card", turn=5, stance="accepted"),
             _fact("f3", "refuse any store credit", recording="rec-c", turn=0)]
    save_spec(tmp_path, Spec(task_id="t1", intent=SpecIntent(task_id="t1", facts=facts, refusals=4)))
    return tmp_path


def test_a_note_sentence_is_recalled_when_a_fact_carries_half_its_words_and_counted_by_field(workdir):
    row = M.score(workdir, ["t1"])["per_task"][0]
    assert row["recalled"]["reason_for_call"] == 1 and row["note_sentences"]["reason_for_call"] == 1
    assert row["recalled"]["known_info"] == 0 and row["note_sentences"]["known_info"] == 1


def test_a_kept_fact_its_own_turn_does_not_carry_is_left_out_as_the_miner_would_have_refused_it(workdir):
    row = M.score(workdir, ["t1"])["per_task"][0]
    assert row["dropped_by_carried"] == 1 and row["facts"] == 2
    assert row["recalled"]["task_instructions"] == 0


def test_witnesses_are_found_again_at_the_witness_share_and_a_higher_share_finds_fewer(workdir):
    assert M.score(workdir, ["t1"])["rates"]["witnessed"] == 2
    assert M.score(workdir, ["t1"], M.Shares(witness=0.9))["rates"]["witnessed"] == 1


def test_an_accepted_fact_no_other_recording_volunteers_is_counted_apart_by_whether_the_note_has_it(workdir):
    (workdir / "tasks" / "t1.json").write_text(json.dumps({"id": "t1", "run_ids": ["rec-a", "rec-b"]}))
    row = M.score(workdir, ["t1"])["per_task"][0]
    assert row["facts_in_note"] == 1
    assert row["unwitnessed_accepted_not_in_note"] == 1 and row["unwitnessed_accepted_in_note"] == 0


def test_a_fact_the_agent_said_before_any_user_did_is_a_steering_leak(workdir):
    row = M.score(workdir, ["t1"])["per_task"][0]
    assert row["leaked"] == 1 and row["leaked_volunteered"] == 0


def test_rates_carry_denominators_wilson_intervals_and_the_refusal_total_and_no_note_text(workdir, tmp_path):
    out = tmp_path / "score.json"
    M.main(["score", str(workdir), "--split", str(_split(tmp_path)), "--corpus", "c1", "--out", str(out)])
    body = out.read_text()
    rates = json.loads(body)["rates"]
    assert (rates["precision"]["k"], rates["precision"]["n"]) == (1, 2) and len(rates["precision"]["wilson95"]) == 2
    assert rates["refusals"] == 4
    for sentence in ("Oak street", "store credit", "card ends", "Calm and brief"):
        assert sentence not in body


def test_the_seeded_draw_repeats_for_one_seed_and_never_holds_an_excluded_id():
    ids = [f"t{i}" for i in range(30)]
    first = M.draw(ids, 10, 0, {"t1", "t2"})
    assert first == M.draw(ids, 10, 0, {"t1", "t2"}) and len(first) == 10
    assert not {"t1", "t2"} & set(first) and first != M.draw(ids, 10, 1, {"t1", "t2"})


def test_a_test_id_listed_among_the_dev_ids_is_refused(workdir, tmp_path):
    split = _split(tmp_path, test=["t1"])
    with pytest.raises(SystemExit):
        M.main(["score", str(workdir), "--split", str(split), "--corpus", "c1", "--out", str(tmp_path / "o.json")])


def _split(tmp_path, test=()):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"dev": {"c1": ["t1"]}, "test": {"c1": list(test)}}))
    return path
