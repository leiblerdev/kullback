import json

from kullback.runner.records import EndState, Event, ExpectedCell, ValueSource, Verifier
from kullback.spec import mutant_rulings as MR
from kullback.spec import rulings as R
from kullback.spec import trust as T
from kullback.spec.router import ROUNDS_CAP, is_set_aside
from kullback.spec.schema import load_spec, save_spec
from tests.spec.fixtures import WRITE_TOOLS, check, fn, reference_run
from tests.spec.fixtures import spec as base_spec

START = {"items": {"A1": {"slot": "two", "label": "red"}, "B2": {"slot": "four", "label": "blue"}}}
END = {"items": {"A1": {"slot": "seven", "label": "red"}, "B2": {"slot": "four", "label": "blue"}}}
SAID = ValueSource(kind="user_turn", ptr={"turn": 0})
STATE = {"kind": "row_is", "table": "items", "row": "A1", "expect": {"slot": "seven"}, "gate": True}


def _base(start=START, end=END, run_id="fresh-1", model="some/model"):
    """A fresh Run (a live model) ending in `end`; model "recorded" makes it a replay of the recording."""
    run = reference_run()
    stop = Event(idx=len(run.events), type="stop", payload={"start_state": start, "end_state": end})
    return run.model_copy(update={"events": run.events + [stop], "run_id": run_id, "model": model})


def _spec(fact_ids=("f1",), gaps=()):
    """The Spec with one gated row_is item citing `fact_ids`; f1 is an ask (kind do), f2 an accepted detail."""
    the_spec = base_spec(checks=[check("s1", demand=dict(STATE, fact_ids=list(fact_ids)), fact_ids=fact_ids)],
                         gaps=gaps)
    facts = [f.model_copy(update={"kind": "do" if f.id == "f1" else "detail"}) for f in the_spec.intent.facts]
    return the_spec.model_copy(update={"intent": the_spec.intent.model_copy(update={"facts": facts})})


def _row_only():
    """A Verifier that allows the written row to change but never holds its slot: every value there passes."""
    cell = ExpectedCell(table="items", row_id="A1", field="label", value="red", row_source=SAID, source=SAID)
    return Verifier(task_id="t1", expected=[EndState(cells=[cell], allowed=[{"table": "items", "row_id": "A1"}])])


def _file(tmp_path, the_spec, verifier, round_number=2):
    save_spec(tmp_path, the_spec)
    return MR.mutant_rulings(tmp_path, verifier, round_number, runs=[_base()])


def test_a_value_survivor_on_a_field_a_volunteered_fact_names_files_one_ruling_with_no_value_in_it(tmp_path):
    out = _file(tmp_path, _spec(), _row_only())
    field = [r for r in R.load_rulings(tmp_path, "t1") if r.code == MR.UNCHECKED_FIELD]
    assert len(field) == 1 and field[0].item == "s1" and field[0].kind == "derivation" and field[0].blocking
    assert field[0].round == 2 and field[0].number in out["filed"]
    for text in (field[0].reason, field[0].fix, str(field[0].verified)):
        assert "seven" not in text and "two" not in text and "four" not in text
    assert "items.slot" in field[0].reason and R.sentence_refusal(field[0].reason, "reason") is None


def test_a_value_survivor_on_a_field_no_volunteered_fact_names_files_no_ruling_and_is_a_note(tmp_path):
    the_spec = _spec(fact_ids=("f2",), gaps=["f1"])
    accepted = [f.model_copy(update={"kind": "do"}) for f in the_spec.intent.facts]
    _file(tmp_path, the_spec.model_copy(update={"intent": the_spec.intent.model_copy(update={"facts": accepted})}),
          _row_only())
    assert not [r for r in R.load_rulings(tmp_path, "t1") if r.code == MR.UNCHECKED_FIELD]
    report = (tmp_path / "exam" / "mutants" / "t1.json").read_text()
    assert '"ruling": "note"' in report


def test_a_value_survivor_on_a_field_only_a_side_fact_names_is_a_note_since_a_side_fact_is_never_gated(tmp_path):
    the_spec = _spec()
    side = [f.model_copy(update={"side": True}) if f.id == "f1" else f for f in the_spec.intent.facts]
    the_spec = the_spec.model_copy(update={"intent": the_spec.intent.model_copy(update={"facts": side})})
    _file(tmp_path, the_spec, _row_only())
    assert not [r for r in R.load_rulings(tmp_path, "t1") if r.code == MR.UNCHECKED_FIELD]


def test_a_write_no_gate_notices_files_one_unchecked_write_on_the_check_that_reads_the_row(tmp_path):
    _file(tmp_path, _spec(), _row_only())
    write, = [r for r in R.load_rulings(tmp_path, "t1") if r.code == MR.UNCHECKED_WRITE]
    assert write.item == "s1" and all(row["operator"] == "undone" for row in write.verified)


def test_each_survivor_in_the_report_names_the_ruling_it_became_and_a_second_verify_files_nothing_new(tmp_path):
    first = _file(tmp_path, _spec(), _row_only())
    report = json.loads((tmp_path / "exam" / "mutants" / "t1.json").read_text())
    survivors = [row for row in report["mutants"] if row["outcome"] == "survived"]
    assert survivors and all(row["ruling"] == "note" or row["ruling"] in first["filed"] for row in survivors)
    again = MR.mutant_rulings(tmp_path, _row_only(), 2, runs=[_base()])
    assert again["filed"] == [] and len(R.load_rulings(tmp_path, "t1")) == len(first["filed"])


def test_a_verifier_that_gates_the_named_cell_files_no_ruling(tmp_path):
    held = ExpectedCell(table="items", row_id="A1", field="slot", value="seven", row_source=SAID, source=SAID)
    out = _file(tmp_path, _spec(), Verifier(task_id="t1", expected=[EndState(cells=[held])]))
    assert out["status"] == "scored" and out["filed"] == [] and R.load_rulings(tmp_path, "t1") == []


def test_rulings_from_mutants_are_open_blocking_and_count_against_the_round_cap(tmp_path):
    _file(tmp_path, _spec().model_copy(update={"round": ROUNDS_CAP}), _row_only(), round_number=ROUNDS_CAP)
    spec = load_spec(tmp_path, "t1")
    assert spec.rulings_open == len(R.load_rulings(tmp_path, "t1")) > 0
    assert is_set_aside(spec)


# --- trust: the stray-write gate reads the untouched sample ---

def _tier(verifier, runs=None):
    the_spec = _spec(gaps=["f2"]).model_copy(update={"examined_rounds": [1]})
    return T.tier_of_task(the_spec, verifier, runs or [_base()], canon=fn, write_tools=WRITE_TOOLS)


def _held_cell(field, value):
    return ExpectedCell(table="items", row_id="A1", field=field, value=value, row_source=SAID, source=SAID)


def test_a_task_whose_untouched_mutants_are_all_killed_is_not_held_stray_untested():
    every = Verifier(task_id="t1", expected=[EndState(cells=[_held_cell("slot", "seven")])])
    tier, row = _tier(every)
    sample = row["constructed"]["stray_write"]["sample"]
    assert sample["built"] > 1 and sample["not_killed"] == 0
    assert tier == "trusted" and row["reason"] is None and row["gates"]["stray_write_fails"] is True


def test_one_untouched_survivor_keeps_the_task_stray_untested_though_the_one_stray_write_fails():
    loose = Verifier(task_id="t1", expected=[EndState(cells=[_held_cell("slot", "seven")],
                                                      allowed=[{"table": "items", "row_id": "A1"}])])
    tier, row = _tier(loose)
    stray = row["constructed"]["stray_write"]
    assert stray["atom"].startswith("gate:sanity:") and stray["sample"]["not_killed"] >= 1
    assert tier == "untrusted" and row["reason"] == "stray_untested" and row["detail"] == "stray_untested"


# --- the Examiner's verify step ---

def _runs_on_disk(root, *runs):
    folder = root / "runs" / "t1"
    folder.mkdir(parents=True, exist_ok=True)
    for run in runs:
        (folder / f"{run.run_id}.jsonl").write_text(run.model_dump_json() + "\n")


def test_verify_files_the_survivors_of_the_passing_fresh_run_in_the_examiners_round_and_says_so(tmp_path):
    from kullback.examiner.exam_files import ExamRoot
    from kullback.examiner.rule_tool import _mutants

    save_spec(tmp_path, _spec())
    _runs_on_disk(tmp_path, _base())
    line = _mutants(ExamRoot(workdir=tmp_path, round=2), _row_only())
    filed = R.load_rulings(tmp_path, "t1")
    assert filed and {r.round for r in filed} == {2} and line.startswith("mutants: ")
    assert ", ".join(str(r.number) for r in filed) in line


def test_verify_with_only_a_passing_replay_builds_no_mutant_files_nothing_and_says_no_passing_run(tmp_path):
    from kullback.examiner.exam_files import ExamRoot
    from kullback.examiner.rule_tool import _mutants

    save_spec(tmp_path, _spec())
    _runs_on_disk(tmp_path, _base(run_id="replay-1", model="recorded"))
    line = _mutants(ExamRoot(workdir=tmp_path, round=2), _row_only())
    assert line == "mutants: not built (no_passing_run)" and R.load_rulings(tmp_path, "t1") == []


def test_a_task_whose_only_passing_run_is_a_replay_is_untrusted_as_no_passing_run():
    every = Verifier(task_id="t1", expected=[EndState(cells=[_held_cell("slot", "seven")])])
    tier, row = _tier(every, [_base(run_id="replay-1", model="recorded")])
    assert tier == "untrusted" and row["reason"] == T.NO_PASSING_RUN and row["reference_passes"] is None


def test_a_review_session_that_never_called_verify_gets_the_verify_step_and_its_mutants_by_code(tmp_path):
    from kullback.examiner.exam_files import ExamRoot
    from kullback.examiner.session import _verify_once
    from kullback.runner.records import write_json
    from tests.spec.test_roles import _written

    root, _ = _written(tmp_path)
    write_json(root / "replays.json", {"t1": {"rec1": {"task_id": "t1", "run_id": "ref", "trace_id": "rec1",
                                                       "confirmed": True, "fidelity": 1}}})
    exam_root = ExamRoot(workdir=root, round=1)
    _verify_once(exam_root, "t1")
    assert "t1" in exam_root.verified
    assert json.loads((root / "exam" / "mutants" / "t1.json").read_text())["status"] == "no_passing_run"


def test_a_review_session_on_a_task_with_no_verifier_skips_the_verify_step(tmp_path):
    from kullback.examiner.exam_files import ExamRoot
    from kullback.examiner.session import _verify_once

    exam_root = ExamRoot(workdir=tmp_path, round=1)
    _verify_once(exam_root, "t1")
    assert exam_root.verified == {}


def test_verify_files_one_unchecked_ask_per_ask_no_gate_cites_once_and_trust_holds_the_task_for_it(tmp_path):
    from kullback.examiner.exam_files import ExamRoot
    from kullback.examiner.rule_tool import _mutants

    weighed = dict(STATE, fact_ids=["f1"], gate=False)
    the_spec = _spec().model_copy(update={"checks": [check("s1", demand=weighed, fact_ids=("f1",))]})
    save_spec(tmp_path, the_spec)
    _runs_on_disk(tmp_path, _base())
    line = _mutants(ExamRoot(workdir=tmp_path, round=2), _row_only())
    asked = [r for r in R.load_rulings(tmp_path, "t1") if r.code == MR.UNCHECKED_ASK]
    assert len(asked) == 1 and asked[0].verified == [{"fact_id": "f1"}] and asked[0].blocking
    assert line.startswith(f"unchecked asks filed: {asked[0].number}; mutants: ")
    assert MR.ask_rulings(tmp_path, the_spec, 3) == []
    every = Verifier(task_id="t1", expected=[EndState(cells=[_held_cell("slot", "seven")])])
    tier, row = T.tier_of_task(the_spec.model_copy(update={"examined_rounds": [1], "gaps": ["f2"]}), every,
                               [_base()], canon=fn, write_tools=WRITE_TOOLS)
    assert tier == "untrusted" and row["reason"] == T.UNCHECKED_ASK


def test_an_ask_filed_as_a_gap_or_a_reaction_needs_no_gate():
    from kullback.spec.schema import unchecked_asks

    the_spec = _spec().model_copy(update={"checks": []})
    assert unchecked_asks(the_spec) == ["f1"]
    assert unchecked_asks(the_spec.model_copy(update={"gaps": ["f1"]})) == []
    facts = [f.model_copy(update={"condition": "agent"}) for f in the_spec.intent.facts]
    assert unchecked_asks(the_spec.model_copy(update={"intent": the_spec.intent.model_copy(update={"facts": facts})})) == []
