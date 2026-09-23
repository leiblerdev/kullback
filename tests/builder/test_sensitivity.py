"""Tests for the sensitivity ruling: a body that reads state answers differently when state differs (D195).

The world here is an invented plant nursery: trays standing on benches, each tray at some growing
stage, and a tool that reports one tray. Two Tasks saw the same tray at two stages, which is the
whole evidence the gate needs and the only evidence it uses: a body that answers both Tasks alike
never read the column that moved, whatever its answers happen to match.
"""

from __future__ import annotations

import pytest

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.builder import sandbox as sb
from kullback.runner.records import Column, EntitySchema, FieldStat, ToolCall, ToolSig

NURSERY = {
    "benches": {
        "BN-1": {"bench_id": "BN-1", "zone": "north"},
        "BN-2": {"bench_id": "BN-2", "zone": "south"},
    },
    "trays": {
        "TR-100": {"tray_id": "TR-100", "bench_id": "BN-1", "stage": "seedling", "seen_at": "day 3"},
        "TR-200": {"tray_id": "TR-200", "bench_id": "BN-2", "stage": "sprout", "seen_at": "day 3"},
    },
}
SPRING = NURSERY  # the world the first Task saw
SUMMER = {
    "benches": NURSERY["benches"],
    "trays": {
        "TR-100": dict(NURSERY["trays"]["TR-100"], stage="flowering", seen_at="day 60"),
        "TR-200": dict(NURSERY["trays"]["TR-200"], seen_at="day 60"),
    },
}

READS_THE_COLUMN = ('tray = self.db.trays[tray_id]\n'
                    'return {"tray_id": tray.tray_id, "stage": tray.stage}\n')
WRITES_A_CONSTANT = 'return {"tray_id": tray_id, "stage": "seedling"}\n'
READS_THE_WRONG_ROW = ('return {"tray_id": tray_id, "stage": self.db.trays["TR-200"].stage}\n')


def _schema(exempt: tuple[str, ...] = ()) -> EntitySchema:
    columns = [Column(table=table, name=name,
                      **{"class": "exempt" if name in exempt else "hard"})
               for table, rows in NURSERY.items()
               for name in sorted({key for row in rows.values() for key in row})]
    return EntitySchema(tables=sorted(NURSERY), columns=columns,
                        id_patterns={"trays.tray_id": r"^TR-\d{3}$", "benches.bench_id": r"^BN-\d$"})


def _sig(name: str = "tray_status", argument: str = "tray_id") -> ToolSig:
    return ToolSig(name=name, description="Report one tray.",
                   args_fields=[FieldStat(name=argument, types=["str"], optional=False)],
                   kind="read", unclassified=False)


def _call(call_id: str, args: dict, result, name: str = "tray_status") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args, result=result, raw_ptr=PTR)


def _box(body: str, workdir, calls, sig=None, schema=None, states=None, tasks=None) -> sb.Sandbox:
    sig = sig or _sig()
    schema = schema if schema is not None else _schema()
    source = ce.module_source(schema, [sig], {sig.name: body})
    box = sb.Sandbox(source, NURSERY, workdir, call_states=states, call_tasks=tasks)
    box.run(calls)  # the gates before this one have already answered every call
    return box


@pytest.fixture
def seen_twice():
    """One tray read under two Tasks, its stage moved between them; identical arguments."""
    calls = [_call("c1", {"tray_id": "TR-100"}, {"tray_id": "TR-100", "stage": "seedling"}),
             _call("c2", {"tray_id": "TR-100"}, {"tray_id": "TR-100", "stage": "flowering"})]
    states = {"c1": SPRING, "c2": SUMMER}
    tasks = {"c1": "task-spring", "c2": "task-summer"}
    return calls, states, tasks


# --- the ruling ---


def test_a_body_that_reads_the_column_answers_the_two_tasks_differently_and_passes(tmp_path, seen_twice):
    calls, states, tasks = seen_twice
    ruling = sb.gate_sensitivity(_box(READS_THE_COLUMN, tmp_path, calls, states=states, tasks=tasks),
                                 calls, _schema())
    assert ruling.passed is True, ruling.failures
    assert ruling.stage == "sensitivity"
    assert ruling.metrics["pairs"] == 1
    assert ruling.metrics["failed"] == 0
    assert ruling.metrics["same_args_pairs"] == 1


def test_a_body_that_writes_the_column_as_a_constant_fails_naming_the_column_and_both_tasks(
        tmp_path, seen_twice):
    calls, states, tasks = seen_twice
    ruling = sb.gate_sensitivity(_box(WRITES_A_CONSTANT, tmp_path, calls, states=states, tasks=tasks),
                                 calls, _schema())
    assert ruling.passed is False
    assert ruling.metrics["failed"] == 1
    assert ruling.metrics["columns"] == ["stage"]
    line = ruling.failures[0]
    assert "tray_status" in line and "stage" in line
    assert "task-spring" in line and "task-summer" in line


def test_a_body_that_reads_another_tasks_row_fails_naming_the_column(tmp_path, seen_twice):
    """Both worlds hold TR-200 at one stage, so a body reading it answers both Tasks alike."""
    calls, states, tasks = seen_twice
    ruling = sb.gate_sensitivity(_box(READS_THE_WRONG_ROW, tmp_path, calls, states=states, tasks=tasks),
                                 calls, _schema())
    assert ruling.passed is False
    assert ruling.metrics["columns"] == ["stage"]
    assert "stage" in ruling.failures[0]


def test_a_failure_carries_the_shape_of_the_two_answers_and_no_value_out_of_them(tmp_path, seen_twice):
    calls, states, tasks = seen_twice
    ruling = sb.gate_sensitivity(_box(WRITES_A_CONSTANT, tmp_path, calls, states=states, tasks=tasks),
                                 calls, _schema())
    line = ruling.failures[0]
    assert "an object with stage, tray_id" in line
    assert "seedling" not in line and "flowering" not in line


def test_a_tool_the_corpus_never_showed_twice_records_no_pairs_and_passes(tmp_path):
    """One Task, one reading: nothing says whether this body reads anything, so nothing is claimed."""
    calls = [_call("c1", {"tray_id": "TR-100"}, {"tray_id": "TR-100", "stage": "seedling"}),
             _call("c2", {"tray_id": "TR-200"}, {"tray_id": "TR-200", "stage": "sprout"})]
    states = {"c1": SPRING, "c2": SPRING}
    ruling = sb.gate_sensitivity(_box(WRITES_A_CONSTANT, tmp_path, calls, states=states), calls, _schema())
    assert ruling.passed is True
    assert ruling.metrics["no_pairs"] is True
    assert ruling.metrics["pairs"] == 0


# --- which calls make a pair ---


def test_two_calls_that_differ_only_in_the_row_they_name_are_a_pair_even_when_the_id_sits_in_an_object(tmp_path):
    """A tool called with one tray per Task still says what the body has to read, more weakly."""
    calls = [_call("c1", {"tray_id": "TR-100"}, {"tray_id": "TR-100", "stage": "seedling"}),
             _call("c2", {"tray_id": "TR-200"}, {"tray_id": "TR-200", "stage": "sprout"})]
    states, tasks = {"c1": SPRING, "c2": SUMMER}, {"c1": "task-spring", "c2": "task-summer"}
    box = _box(WRITES_A_CONSTANT, tmp_path, calls, states=states, tasks=tasks)
    pairs = sb.sensitivity_pairs(box, calls, _schema())
    assert [pair.columns for pair in pairs] == [("stage",)]
    assert pairs[0].same_args is False
    # the homing rule the pinner uses finds an id nested inside an object argument too
    sig = _sig(argument="tray")
    calls = [_call("c1", {"tray": {"tray_id": "TR-100"}}, {"tray_id": "TR-100", "stage": "seedling"}),
             _call("c2", {"tray": {"tray_id": "TR-200"}}, {"tray_id": "TR-200", "stage": "sprout"})]
    states, tasks = {"c1": SPRING, "c2": SUMMER}, {"c1": "task-spring", "c2": "task-summer"}
    body = 'return {"tray_id": tray["tray_id"], "stage": "seedling"}\n'
    box = _box(body, tmp_path / "nested", calls, sig=sig, states=states, tasks=tasks)
    assert [pair.columns for pair in sb.sensitivity_pairs(box, calls, _schema())] == [("stage",)]


# --- what the ruling is worth to the rest of the stage ---


def test_the_chain_goes_on_past_a_failed_sensitivity_ruling_so_the_replay_count_still_ties_it(
        tmp_path, seen_twice):
    """Every body that fails this gate is level on gates passed; how much it replays breaks the tie."""
    calls, states, tasks = seen_twice
    schema, sig = _schema(), _sig()
    source = ce.module_source(schema, [sig], {sig.name: WRITES_A_CONSTANT})
    box = sb.Sandbox(source, NURSERY, tmp_path, call_states=states, call_tasks=tasks)
    stages = [gate.stage for gate in sb.run_gates(source, box, calls, [], schema, sig=sig)]
    assert stages[-2:] == ["sensitivity", "replay_fidelity"]


def _worlds_alike():
    """The recordings part in `stage` and both worlds hold one stage for every tray, so no body
    that reads the world can answer the two calls differently. That is the Starting state missing a
    pin, which the replay ruling already blocks the Task for, and blaming the body for it would be
    an accusation the evidence does not support."""
    flat = {"benches": NURSERY["benches"], "trays": NURSERY["trays"]}
    other = {"benches": NURSERY["benches"],
             "trays": {key: dict(row, seen_at="day 60") for key, row in NURSERY["trays"].items()}}
    calls = [_call("c1", {"tray_id": "TR-100"}, {"tray_id": "TR-100", "stage": "seedling"}),
             _call("c2", {"tray_id": "TR-100"}, {"tray_id": "TR-100", "stage": "flowering"})]
    return dict(calls=calls, states={"c1": flat, "c2": other}, tasks={"c1": "task-one", "c2": "task-two"})


def _same_recorded_answer():
    """Two worlds, one recorded answer: the tool answered them alike, so a body may too."""
    calls = [_call("c1", {"tray_id": "TR-200"}, {"tray_id": "TR-200", "stage": "sprout"}),
             _call("c2", {"tray_id": "TR-200"}, {"tray_id": "TR-200", "stage": "sprout"})]
    return dict(calls=calls, states={"c1": SPRING, "c2": SUMMER}, tasks={"c1": "task-spring", "c2": "task-summer"})


def _exempt_column():
    """An exempt column is equal whatever it holds, so a body cannot be asked to move it."""
    calls = [_call("c1", {"tray_id": "TR-200"}, {"tray_id": "TR-200", "seen_at": "day 3"}),
             _call("c2", {"tray_id": "TR-200"}, {"tray_id": "TR-200", "seen_at": "day 60"})]
    return dict(calls=calls, states={"c1": SPRING, "c2": SUMMER}, tasks={"c1": "task-spring", "c2": "task-summer"},
                schema=_schema(exempt=("seen_at",)))


def _one_world():
    """A body that answers two calls of one Task alike has answered one question once."""
    calls = [_call("c1", {"tray_id": "TR-100"}, {"tray_id": "TR-100", "stage": "seedling"}),
             _call("c2", {"tray_id": "TR-200"}, {"tray_id": "TR-200", "stage": "sprout"})]
    return dict(calls=calls, states={"c1": SPRING, "c2": SPRING})


def _refused_call():
    """An error has no columns to differ in; matching error classes is the replay ruling's job."""
    from kullback.runner.records import ToolCallError

    refused = ToolCall(id="c2", name="tray_status", args={"tray_id": "TR-100"},
                       error=ToolCallError(**{"class": "not_found_entity"}, payload="No such tray"),
                       raw_ptr=PTR)
    calls = [_call("c1", {"tray_id": "TR-100"}, {"tray_id": "TR-100", "stage": "seedling"}), refused]
    return dict(calls=calls, run=[calls[0]], body=READS_THE_COLUMN, states={"c1": SPRING, "c2": SUMMER})


@pytest.mark.parametrize("case", [_worlds_alike, _same_recorded_answer, _exempt_column, _one_world, _refused_call])
def test_a_column_the_two_worlds_hold_alike_is_no_pair_however_the_recordings_differ(tmp_path, case):
    """Calls the state cannot tell apart, or that recorded no difference, make no pair and pass."""
    c = case()
    schema = c.get("schema") or _schema()
    box = _box(c.get("body", WRITES_A_CONSTANT), tmp_path, c.get("run", c["calls"]), schema=schema,
               states=c["states"], tasks=c.get("tasks"))
    assert sb.sensitivity_pairs(box, c["calls"], schema) == []
    ruling = sb.gate_sensitivity(box, c["calls"], schema)
    assert ruling.passed is True
    assert ruling.metrics["no_pairs"] is True


def test_a_prose_result_no_reader_reads_is_counted_and_not_ruled_on(tmp_path):
    """Without a reader there is no column to name, no world to ask about it and no lesson to give."""
    calls = [_call("c1", {"tray_id": "TR-100"}, "Tray TR-100 is a seedling."),
             _call("c2", {"tray_id": "TR-100"}, "Tray TR-100 is flowering.")]
    states, tasks = {"c1": SPRING, "c2": SUMMER}, {"c1": "task-spring", "c2": "task-summer"}
    body = 'return "Tray " + tray_id + " is a seedling."\n'
    ruling = sb.gate_sensitivity(_box(body, tmp_path, calls, states=states, tasks=tasks), calls, _schema())
    assert ruling.passed is True
    assert ruling.metrics["unread_pairs"] == 1
    assert ruling.metrics["pairs"] == 0
