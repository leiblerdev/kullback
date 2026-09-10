"""What a transcript claims against what its state received, and partial completion (D223).

The world is an invented pottery workshop: a kiln is fired, a pot is glazed, a shelf is read. Every
number here is code over records the harness already writes, so nothing in this file calls a model
and nothing in it names a corpus.
"""

from __future__ import annotations

from pathlib import Path

from kullback import claims, difficulty, report, rounds
from kullback.builder import agent as builder_agent
from kullback.builder.build import BuildPlan
from kullback.examiner import findings
from kullback.report.render import _claims_table
from kullback.runner.records import Atom, Event, Run, ToolSig, Verifier, as_dict, write_json

FIRE = "fire_kiln"
GLAZE = "glaze_pot"
READ = "read_shelf"
WRITE_TOOLS = {FIRE, GLAZE}

SIGS = [
    ToolSig(name=FIRE, kind="write", description="Fire the kiln that holds a shelf of pots."),
    ToolSig(name=GLAZE, kind="write", description="Glaze one pot on a shelf."),
    ToolSig(name=READ, kind="read", description="Read the pots a shelf holds."),
]


def _fire_atom(atom_id: str = "w0", entity: str = "kiln7") -> Atom:
    return Atom(id=atom_id, kind="required", target={"kind": "write", "tool": FIRE, "entity": entity})


def _glaze_atom(atom_id: str = "w1", entity: str = "pot3") -> Atom:
    return Atom(id=atom_id, kind="required", target={"kind": "write", "tool": GLAZE, "entity": entity})


def _verifier(*atoms: Atom, task_id: str = "task_1", seeds: tuple[str, ...] = ("run-seed",)) -> Verifier:
    return Verifier(task_id=task_id, atoms=list(atoms), seed_run_ids=list(seeds))


def _events(said: list[str], calls: list[tuple[str, dict, dict]]) -> list[Event]:
    out = [Event(idx=0, type="user_turn", payload={"content": "please fire the kiln"})]
    for number, (name, args, result) in enumerate(calls):
        call_id = f"c{number}"
        out.append(Event(idx=len(out), type="tool_call",
                         payload={"id": call_id, "name": name, "args": args, "requestor": "assistant"}))
        out.append(Event(idx=len(out), type="tool_result", payload={"id": call_id, "name": name,
                                                                    "result": result}))
    for sentence in said:
        out.append(Event(idx=len(out), type="model_call", payload={"reply": {"content": sentence}}))
    return out


def _run(said: list[str], calls: list[tuple[str, dict, dict]], run_id: str = "run-a") -> Run:
    return Run(run_id=run_id, task_id="task_1", termination_reason="success",
               events=_events(said, calls))


FIRE_K7 = (FIRE, {"kiln_id": "kiln7", "hours": 6}, {"kiln_id": "kiln7", "fired": True})
GLAZE_P3 = (GLAZE, {"pot_id": "pot3", "colour": "ash"}, {"pot_id": "pot3", "glazed": True})
READ_S2 = (READ, {"shelf_id": "s2"}, {"shelf_id": "s2", "pots": [{"pot_id": "pot3"}]})

VOCABULARY = claims.tool_vocabulary(SIGS)
GENERAL = claims.general_stems()


def _rows(run: Run, verifier=None):
    return claims.rows_for([run], verifier, None, WRITE_TOOLS, VOCABULARY, GENERAL)[0]


# --- the vocabulary ------------------------------------------------------------

def test_the_claim_vocabulary_is_mined_from_the_write_tool_names_and_never_a_fixed_list():
    assert set(VOCABULARY) == WRITE_TOOLS, "a read tool contributes no claim word"
    assert claims._stem("fire") in VOCABULARY[FIRE] and "kiln" in VOCABULARY[FIRE]
    assert claims._stem("glaze") in VOCABULARY[GLAZE]
    assert claims._stem("fire") not in VOCABULARY[GLAZE], "each tool carries the words of its own name"
    assert not any("pottery" in mined or "workshop" in mined for mined in VOCABULARY.values()), \
        "nothing outside the mined names and descriptions is in the vocabulary"


def test_a_sentence_names_the_tool_whose_own_words_it_uses_and_the_general_verbs_name_them_all():
    assert claims.claimed_tools("I have fired the kiln.", VOCABULARY, GENERAL) == [FIRE]
    assert claims.claimed_tools("I have glazed it.", VOCABULARY, GENERAL) == [GLAZE]
    assert claims.claimed_tools("That is done.", VOCABULARY, GENERAL) == sorted(WRITE_TOOLS), \
        "a general verb names no tool of its own, so every write tool is a candidate"
    assert claims.claimed_tools("I read the shelf.", VOCABULARY, GENERAL) == [], \
        "a read is not a claim about state"


def test_an_inflected_word_is_reduced_to_the_stem_its_other_forms_share():
    assert claims._stem("fired") == claims._stem("fires") == claims._stem("firing") == claims._stem("fire")
    assert claims._stem("cancelled") == claims._stem("cancels") == "cancel"
    assert claims._stem("applied") == "apply"
    assert claims._stem("read") == "read", "a short word is not cut down to a letter or two"


def test_a_description_is_mined_for_the_verb_it_opens_on_and_not_for_the_world_it_names():
    assert claims.leading_words("Fire the kiln that holds a shelf of pots.") == ["Fire"]
    assert claims.leading_words("Glaze one pot. Return the pot.") == ["Glaze", "Return"]
    assert claims._stem("shelf") not in VOCABULARY[FIRE], \
        "a noun the description mentions is not a word that claims the action was done"


def test_a_promise_and_a_question_are_not_claims_that_the_action_has_happened():
    said = ["I will fire the kiln for you.", "Shall I fire the kiln?", "Let me fire the kiln."]
    assert _rows(_run(said, [FIRE_K7]))["claims"] == 0
    assert _rows(_run(["I have fired the kiln."], [FIRE_K7]))["claims"] == 1


# --- the three counts ----------------------------------------------------------

def test_a_transcript_claiming_a_write_no_atom_moved_counts_claimed_unwritten():
    row = _rows(_run(["I have fired the kiln for you."], [READ_S2]), _verifier(_fire_atom()))
    assert (row["claims"], row["claims_written"], row["claims_unwritten"]) == (1, 0, 1)
    assert row["writes"] == 0, "a read moved nothing, so the state received no write at all"


def test_a_write_call_the_world_refused_is_no_effect_so_the_claim_about_it_is_unwritten():
    refused = (FIRE, {"kiln_id": "kiln7"}, {"error": "the kiln is already firing"})
    events = _events(["I have fired the kiln."], [refused])
    events[2].payload["error"] = "the kiln is already firing"
    run = Run(run_id="run-b", task_id="task_1", events=events)
    row = _rows(run, _verifier(_fire_atom()))
    assert row["claims_unwritten"] == 1 and row["writes"] == 0


def test_a_moved_atom_no_sentence_mentions_counts_writes_unclaimed():
    row = _rows(_run(["Anything else I can help with."], [FIRE_K7]), _verifier(_fire_atom()))
    assert row["claims"] == 0
    assert row["writes_unclaimed"] == 1, "the state received a write the transcript never named"


def test_a_claim_matched_to_its_atom_by_the_value_it_names_counts_written():
    """A general verb stands for every write the Task requires, so the id in the same sentence is
    what says which of them the Candidate claimed."""
    verifier = _verifier(_fire_atom(), _glaze_atom())
    written = _rows(_run(["That is done for pot3."], [GLAZE_P3]), verifier)
    assert (written["claims"], written["claims_written"], written["claims_unwritten"]) == (1, 1, 0)
    unwritten = _rows(_run(["That is done for kiln7."], [GLAZE_P3]), verifier)
    assert unwritten["claims_unwritten"] == 1, "the value names the kiln, and the kiln was not fired"


def test_a_claim_about_one_tool_is_not_answered_by_a_write_of_another():
    row = _rows(_run(["I have fired the kiln."], [GLAZE_P3]),
                _verifier(_fire_atom(), _glaze_atom()))
    assert row["claims_unwritten"] == 1, "the kiln was never fired, whatever else the Run wrote"
    assert row["writes_unclaimed"] == 1, "and the glaze nobody mentioned is still unclaimed"


# --- partial completion --------------------------------------------------------

def test_the_partial_number_is_the_atoms_confirmed_over_the_atoms_the_verifier_carries():
    verifier = _verifier(_fire_atom(), _glaze_atom())
    half = _rows(_run(["I have fired the kiln."], [FIRE_K7]), verifier)
    assert (half["atoms_confirmed"], half["atoms_total"]) == (1, 2)
    assert half["partial_completion"] == 0.5
    whole = _rows(_run(["Both are done."], [FIRE_K7, GLAZE_P3]), verifier)
    assert whole["partial_completion"] == 1.0 and whole["failed"] is False


def test_a_run_that_confirms_some_atoms_still_fails_and_the_partial_number_changes_no_ruling():
    verifier = _verifier(_fire_atom(), _glaze_atom())
    row = _rows(_run(["I have fired the kiln."], [FIRE_K7]), verifier)
    assert row["failed"] is True, "partial completion stands beside the ruling and never replaces it"
    assert "trusted" not in row and "pass" not in row


def test_a_verifier_with_no_atom_has_no_partial_number_rather_than_a_perfect_one():
    assert claims.partial_completion({}) is None
    assert claims.band(None) == "none"
    assert claims.band(0.0) == "p0" and claims.band(0.5) == "p2" and claims.band(1.0) == "p4"


def test_the_difficulty_record_carries_the_partial_band_beside_its_other_counts():
    record = difficulty.record_for(_verifier(_fire_atom()), calls=[], partial_completion=0.5)
    assert record["partial_completion"] == 0.5 and record["partial_band"] == "p2"
    assert difficulty.record_for(_verifier(_fire_atom()), calls=[])["partial_band"] == "none"


# --- per Task, and the flag ----------------------------------------------------

def _task_row(*rows: dict) -> dict:
    return claims.task_rows(list(rows))


def test_the_flag_fires_only_where_every_failing_held_out_run_claimed_a_write_nothing_received():
    claimed = {"claims": 1, "claims_written": 0, "claims_unwritten": 1, "writes_unclaimed": 0,
               "failed": True, "held_out": True, "partial_completion": 0.0}
    other = dict(claimed, claims_unwritten=0, claims_written=1)
    assert _task_row(claimed, claimed)["flagged"] is True
    assert _task_row(claimed, other)["flagged"] is False, "one failure of another kind clears the flag"
    assert _task_row(dict(claimed, failed=False))["flagged"] is False, "no failing Run, no flag"
    assert _task_row(dict(claimed, held_out=False))["flagged"] is False, \
        "a Run the Verifier was derived from says nothing about the Verifier, so it never flags"
    assert _task_row()["flagged"] is False


def test_a_task_row_shares_the_claiming_failures_over_the_failing_runs():
    claimed = {"claims": 2, "claims_written": 1, "claims_unwritten": 1, "writes_unclaimed": 1,
               "failed": True, "partial_completion": 0.5}
    passing = dict(claimed, failed=False, claims_unwritten=0, partial_completion=1.0)
    row = _task_row(claimed, passing)
    assert row["claimed_unwritten_share"] == 1.0 and row["failing_runs"] == 1
    assert row["partial_completion_mean"] == 0.75 and row["partial_completion_band"] == "p3"
    assert row["claims"] == 4 and row["writes_unclaimed"] == 2


# --- where the numbers are reported --------------------------------------------

def _workdir(tmp_path: Path, said: list[str], calls: list[tuple[str, dict, dict]]) -> Path:
    run = _run(said, calls, run_id="run-a")
    write_json(tmp_path / "tool_sigs.json", [as_dict(sig) for sig in SIGS])
    return tmp_path, {"sigs": [as_dict(sig) for sig in SIGS],
                      "verifiers": [_verifier(_fire_atom())],
                      "task_runs": {"task_1": [run]},
                      "replays": {"task_1": {"t-a": {"run_id": "run-a", "confirmed": True}}},
                      "rerolls": {}, "task_status": {}}


def test_a_workdir_reads_back_a_row_per_run_a_row_per_task_and_one_corpus_line(tmp_path):
    workdir, store = _workdir(tmp_path, ["I have fired the kiln."], [READ_S2])
    body = claims.refresh(workdir, store=store)
    assert body["format"] == claims.FORMAT
    assert [row["run_id"] for row in body["runs"]] == ["run-a"]
    assert body["tasks"]["task_1"]["claims_unwritten"] == 1
    assert body["totals"]["claims_unwritten"] == 1 and body["totals"]["runs_with_claims"] == 1
    assert body["flagged"] == ["task_1"]
    assert claims.read_records(workdir)["totals"] == body["totals"]


def test_a_workdir_with_no_claim_file_reads_as_no_rows_rather_than_failing(tmp_path):
    assert claims.read_records(tmp_path) == {}
    assert difficulty.partial_by_task(tmp_path) == {}


def test_the_round_carries_the_four_counters_off_the_rows_it_wrote(tmp_path):
    plan = BuildPlan(workdir=tmp_path / "work")
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan))
    _, store = _workdir(tmp_path, ["I have fired the kiln."], [READ_S2])
    counts = loop.claim_counts(store)
    assert set(counts) == {"claims", "claims_unwritten", "writes_unclaimed", "partial_completion_mean"}
    assert (counts["claims"], counts["claims_unwritten"], counts["writes_unclaimed"]) == (1, 1, 0)
    assert counts["partial_completion_mean"] == 0.0
    assert (plan.workdir / claims.FILE_NAME).is_file()


def test_the_report_says_what_was_claimed_against_what_the_state_received(tmp_path):
    _, store = _workdir(tmp_path, ["I have fired the kiln."], [READ_S2])
    data = report.ReportData(claims=claims.compute(tmp_path, store=store))
    lines = _claims_table(data)
    assert "### Claims against state" in lines
    assert any("1 claims" in line for line in lines)
    assert any("Flagged for the Simulated user's end protocol: task_1" in line for line in lines)
    assert any("never gates it" in line for line in lines), "the legend says it decides nothing"


def test_a_build_with_no_claim_record_says_so_rather_than_showing_an_empty_table():
    lines = _claims_table(report.ReportData())
    assert any("No claim record was written" in line for line in lines)


def test_a_flagged_task_is_filed_as_a_finding_that_points_at_the_end_protocol_and_suggests_no_verb():
    rows = findings.claimed_unwritten_rows({"flagged": ["task_1"],
                                            "tasks": {"task_1": {"failing_runs": 2, "claims_unwritten": 2}}})
    assert [row["task_id"] for row in rows] == ["task_1"]
    assert rows[0]["suggested"] == "none", "no Builder or Examiner verb owns the end protocol"
    assert "end protocol" in rows[0]["text"]
    assert findings.claimed_unwritten_rows({}) == []
