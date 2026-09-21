"""The tool context: a body takes time, randomness and new ids from one context.

Needs docs/frozen-patches/tool-context.patch on top of the confinement imports
patch. Skips on a tree without them. Every tool and table here is invented.
"""

from __future__ import annotations

import pytest

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.gates import confinement as confinement_module
from kullback.gates.confinement import gate_confined, source_confinement
from kullback.runner.records import Column, EntitySchema, FieldStat, RawPtr, ToolCall, ToolSig


def _patch_applied() -> bool:
    """Whether the tool context patch is applied: the four modules are gone and the
    refusal names the new name the bodies use instead."""
    if "random" in confinement_module.ALLOWED_IMPORTS:
        return False
    source = ("class DomainTools:\n    def fetch_widget(self, x):\n"
              "        import random\n        return random.random()\n")
    return any("self.ctx" in line for line in source_confinement(source))


pytestmark = pytest.mark.skipif(not _patch_applied(), reason="needs tool-context.patch")


LOANS_SCHEMA = EntitySchema(
    tables=["loans"],
    columns=[
        Column(table="loans", name="loan_id", **{"class": "hard"}, classified_by="rule"),
        Column(table="loans", name="opened", **{"class": "hard"}, classified_by="rule"),
    ],
    id_patterns={"loans.loan_id": r"^L\d+$"},
)
OPEN_SIG = ToolSig(
    name="open_loan",
    description="Open a loan for a patron.",
    args_fields=[FieldStat(name="patron", types=["str"], optional=False)],
    kind="write",
    unclassified=False,
)
LOANS_DB = {"loans": {"L100": {"loan_id": "L100", "opened": "2024-01-02T10:00:00"}}}

OPEN_BODY = (
    "loan_id = self.ctx.new_id(\"loans\")\n"
    "opened = self.ctx.now()\n"
    "self.db.loans[loan_id] = {\"loan_id\": loan_id, \"opened\": opened}\n"
    "return {\"loan_id\": loan_id, \"opened\": opened}\n"
)


def _call(call_id, result):
    return ToolCall(id=call_id, name="open_loan", args={"patron": "ann"},
                    result=result, raw_ptr=PTR)


def test_the_four_modules_left_the_allowed_imports():
    assert {"datetime", "time", "random", "uuid"} - confinement_module.ALLOWED_IMPORTS == {
        "datetime", "time", "random", "uuid"}


@pytest.mark.parametrize("module,context", [
    ("datetime", "self.ctx.now()"),
    ("time", "self.ctx.now()"),
    ("random", "self.ctx.random()"),
    ("uuid", "self.ctx.new_id(table)"),
])
def test_a_body_importing_a_removed_module_is_refused_naming_the_context(module, context):
    source = ("class DomainTools:\n    def fetch_widget(self, x):\n"
              f"        import {module}\n        return {{'id': x}}\n")
    refused = source_confinement(source)
    assert refused == [f"fetch_widget imports {module}; take it from the tool context "
                       f"instead ({context})"]
    assert gate_confined(source).passed is False


def test_a_creating_body_gets_the_recorded_id_at_replay(tmp_path):
    """The gates feed each call what its recording witnessed, so the body replays it."""
    recorded = {"loan_id": "L101", "opened": "2024-03-04T05:06:07"}
    call = _call("c1", recorded)
    source = ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY})
    box = ce.Sandbox(source, LOANS_DB, tmp_path,
                     call_context=ce.recorded_call_contexts([call], LOANS_SCHEMA))
    assert box.run([call]) == [{"ok": True, "value": recorded}]
    assert gate_confined(source).passed is True


def test_the_recorded_id_lands_in_the_world_the_next_call_reads():
    """The row is inserted under the recorded id, which is what a later read replays."""
    call = _call("c1", {"loan_id": "L101", "opened": "2024-03-04T05:06:07"})
    source = ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY})
    toolkit = ce.load_toolkit(source, {table: dict(rows) for table, rows in LOANS_DB.items()})
    toolkit.ctx.attach_recorded(ce.recorded_run_context([call], LOANS_SCHEMA))
    assert toolkit.open_loan(patron="ann") == {"loan_id": "L101", "opened": "2024-03-04T05:06:07"}
    assert set(toolkit.db.loans) == {"L100", "L101"}
    assert toolkit.ctx.usage() == {"recorded": 2, "seeded": 0}


def test_the_same_body_off_the_path_mints_a_shaped_id_never_a_colliding_one():
    toolkit = ce.load_toolkit(
        ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY}),
        {table: dict(rows) for table, rows in LOANS_DB.items()})
    toolkit.ctx.reseed(7)
    first = toolkit.open_loan(patron="ann")
    assert first["loan_id"] == "L101" and "L101" not in LOANS_DB["loans"]
    assert first["opened"].startswith("2024-")
    assert toolkit.ctx.usage() == {"recorded": 0, "seeded": 2}


def test_the_same_seed_draws_the_same_run():
    def run_once():
        toolkit = ce.load_toolkit(
            ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY}),
            {table: dict(rows) for table, rows in LOANS_DB.items()})
        toolkit.ctx.reseed(7)
        return toolkit.open_loan(patron="ann")

    assert run_once() == run_once()


def test_two_creations_in_one_run_mint_two_ids():
    toolkit = ce.load_toolkit(
        ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY}),
        {table: dict(rows) for table, rows in LOANS_DB.items()})
    toolkit.ctx.reseed(7)
    first = toolkit.open_loan(patron="ann")
    second = toolkit.open_loan(patron="bea")
    assert first["loan_id"] != second["loan_id"]
    assert set(toolkit.db.loans) == {"L100", first["loan_id"], second["loan_id"]}


def test_now_at_replay_is_the_recorded_time_and_off_the_path_a_function_of_seed_and_step():
    recorded = _call("c1", {"loan_id": "L101", "opened": "2024-03-04T05:06:07"})
    toolkit = ce.load_toolkit(
        ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY}),
        {table: dict(rows) for table, rows in LOANS_DB.items()})
    toolkit.ctx.attach_recorded(ce.recorded_run_context([recorded], LOANS_SCHEMA))
    assert toolkit.ctx.now() == "2024-03-04T05:06:07"

    fresh = ce.load_toolkit(
        ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY}),
        {table: dict(rows) for table, rows in LOANS_DB.items()})
    fresh.ctx.reseed(11)
    first, second = fresh.ctx.now(), fresh.ctx.now()
    assert first.startswith("2024-") and "T" in first
    # Built at a later wall clock under the same seed, the same steps answer alike.
    later = ce.load_toolkit(
        ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY}),
        {table: dict(rows) for table, rows in LOANS_DB.items()})
    later.ctx.reseed(11)
    assert (later.ctx.now(), later.ctx.now()) == (first, second)
    assert fresh.ctx.usage()["seeded"] == 2


def test_a_table_with_no_rows_mints_under_a_fixed_prefix():
    toolkit = ce.load_toolkit(
        ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY}), {"loans": {}})
    toolkit.ctx.reseed(3)
    first, second = toolkit.ctx.new_id("loans"), toolkit.ctx.new_id("loans")
    assert first.startswith("new_") and second.startswith("new_") and first != second


def test_a_body_that_never_touches_the_context_answers_as_before(tmp_path):
    """The context rides along unheard: answers with and without a feed are identical."""
    read_sig = ToolSig(
        name="get_loan",
        description="Read a loan.",
        args_fields=[FieldStat(name="loan_id", types=["str"], optional=False)],
        kind="read",
        unclassified=False,
    )
    body = ("loan = self.db.loans[loan_id]\n"
            "return {\"loan_id\": loan.loan_id, \"opened\": loan.opened}\n")
    source = ce.module_source(LOANS_SCHEMA, [read_sig], {"get_loan": body})
    call = ToolCall(id="c9", name="get_loan", args={"loan_id": "L100"},
                    result={"loan_id": "L100", "opened": "2024-01-02T10:00:00"}, raw_ptr=PTR)
    plain = ce.Sandbox(source, LOANS_DB, tmp_path / "plain").run([call])
    fed = ce.Sandbox(source, LOANS_DB, tmp_path / "fed",
                     call_context=ce.recorded_call_contexts([call], LOANS_SCHEMA)).run([call])
    assert plain == fed == [{"ok": True,
                             "value": {"loan_id": "L100", "opened": "2024-01-02T10:00:00"}}]
    assert ce.load_toolkit(source, LOANS_DB).ctx.usage() == {"recorded": 0, "seeded": 0}


def test_a_read_before_a_creation_does_not_shift_the_creation_feed():
    """The run feed lists only what a body could not know: a read echoing a held row and an
    update echoing its own id contribute nothing, so the creation is served its own values."""
    read = ToolCall(id="r1", name="get_loan", args={"loan_id": "L100"},
                    result={"loan_id": "L100", "opened": "2024-01-02T10:00:00"}, raw_ptr=PTR)
    update = ToolCall(id="u1", name="open_loan", args={"patron": "ann", "loan_id": "L100"},
                      result={"loan_id": "L100", "opened": "2024-01-02T10:00:00"}, raw_ptr=PTR)
    create = _call("c1", {"loan_id": "L101", "opened": "2024-03-04T05:06:07"})
    run = ce.recorded_run_context([read, update, create], LOANS_SCHEMA, ["open_loan"])
    assert run == {"ids": {"loans": ["L101"]}, "times": ["2024-03-04T05:06:07"]}
    toolkit = ce.load_toolkit(
        ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY}),
        {table: dict(rows) for table, rows in LOANS_DB.items()})
    toolkit.ctx.attach_recorded(run)
    assert toolkit.open_loan(patron="ann") == {"loan_id": "L101",
                                                    "opened": "2024-03-04T05:06:07"}


def test_an_unrelated_earlier_value_spelling_the_new_id_does_not_discard_it():
    """Newness is read off entity ids alone: a note echoing the string must not cost the
    creation its recorded id."""
    note = ToolCall(id="r1", name="get_loan", args={"loan_id": "L100"},
                    result={"loan_id": "L100", "note": "L101"}, raw_ptr=PTR)
    create = _call("c1", {"loan_id": "L101", "opened": "2024-03-04T05:06:07"})
    run = ce.recorded_run_context([note, create], LOANS_SCHEMA, ["open_loan"])
    assert run["ids"] == {"loans": ["L101"]}


def test_the_prompt_names_the_context_and_the_hint_points_at_it():
    system = ce.body_messages(OPEN_SIG, [], schema=LOANS_SCHEMA)[0]["content"]
    assert "self.ctx.now()" in system and "self.ctx.random()" in system
    assert "self.ctx.new_id(\"loans\")" in system
    failures = ["open_loan({'patron': 'ann'}) raised NameError: name 'random' is not defined"]
    hints = ce._import_hints([confinement_module.GateResult(stage="x", passed=False,
                                                            failures=failures)])
    assert hints == ["- `random` is not importable; take it from the tool context instead, "
                     "as in `self.ctx.random()`"]


UPDATE_SIG = ToolSig(
    name="touch_loan",
    description="Touch a loan row.",
    args_fields=[FieldStat(name="loan_id", types=["str"], optional=False)],
    kind="write",
    unclassified=False,
)
WRONG_TOUCH_BODY = (
    "loan_id = self.ctx.new_id(\"loans\")\n"
    "opened = self.ctx.now()\n"
    "self.db.loans[loan_id] = {\"loan_id\": loan_id, \"opened\": opened}\n"
    "return {\"loan_id\": loan_id, \"opened\": opened}\n"
)


def _update_call():
    return ToolCall(id="u1", name="touch_loan", args={"loan_id": "L100"},
                    result={"loan_id": "L100", "opened": "2024-01-02T10:00:00"}, raw_ptr=PTR)


def test_an_update_echoing_its_own_id_feeds_no_new_id():
    feeds = ce.recorded_call_contexts([_update_call()], LOANS_SCHEMA)
    assert feeds["u1"]["new_ids"] == {}


def test_a_wrong_update_body_is_not_certified_from_the_fed_old_id(tmp_path):
    call = _update_call()
    source = ce.module_source(LOANS_SCHEMA, [UPDATE_SIG], {"touch_loan": WRONG_TOUCH_BODY})
    box = ce.Sandbox(source, LOANS_DB, tmp_path,
                     call_context=ce.recorded_call_contexts([call], LOANS_SCHEMA))
    [outcome] = box.run([call])
    assert outcome["ok"] is True
    assert outcome["value"]["loan_id"] != "L100"


def test_an_id_observed_earlier_in_the_recording_is_withheld():
    create = ToolCall(id="c1", name="open_loan", args={"patron": "ann"},
                      result={"loan_id": "L101", "opened": "2024-03-04T05:06:07"},
                      raw_ptr=PTR, trace_id="t1")
    echo = ToolCall(id="r2", name="get_loan", args={},
                    result={"loan_id": "L101"}, raw_ptr=PTR, trace_id="t1")
    feeds = ce.recorded_call_contexts([create, echo], LOANS_SCHEMA)
    assert feeds["c1"]["new_ids"] == {"loans": "L101"}
    assert feeds["r2"]["new_ids"] == {}


def test_a_scalar_spelling_does_not_suppress_per_call_newness():
    note = ToolCall(id="r1", name="get_loan", args={"loan_id": "L100"},
                    result={"note": "L101"}, raw_ptr=PTR)
    create = _call("c1", {"loan_id": "L101", "opened": "2024-03-04T05:06:07"})
    feeds = ce.recorded_call_contexts([note, create], LOANS_SCHEMA)
    assert feeds["c1"]["new_ids"] == {"loans": "L101"}


def test_separate_recordings_never_share_prior_observation():
    first = ToolCall(id="a1", name="open_loan", args={"patron": "ann"},
                     result={"loan_id": "L102"}, raw_ptr=PTR, trace_id="ta")
    second = ToolCall(id="b1", name="open_loan", args={"patron": "bea"},
                      result={"loan_id": "L102"}, raw_ptr=PTR, trace_id="tb")
    feeds = ce.recorded_call_contexts([first, second], LOANS_SCHEMA)
    assert feeds["a1"]["new_ids"] == {"loans": "L102"}
    assert feeds["b1"]["new_ids"] == {"loans": "L102"}


def test_new_id_refuses_a_witnessed_id_the_table_already_holds():
    toolkit = ce.load_toolkit(
        ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY}),
        {table: dict(rows) for table, rows in LOANS_DB.items()})
    toolkit.ctx.feed_call({"now": None, "new_ids": {"loans": "L100"}})
    with pytest.raises(ValueError):
        toolkit.ctx.new_id("loans")
    toolkit.ctx.feed_call({"now": None, "new_ids": {"loans": "L101"}})
    assert toolkit.ctx.new_id("loans") == "L101"


def test_a_reseeded_context_matches_a_fresh_one_over_the_same_database():
    source = ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY})
    used = ce.load_toolkit(source, {table: dict(rows) for table, rows in LOANS_DB.items()})
    used.ctx.attach_recorded({"ids": {"loans": ["L999"]}, "times": ["1999-01-01T00:00:00"]})
    used.ctx.feed_call({"now": "1999-01-01T00:00:00", "new_ids": {"loans": "L999"}})
    assert used.open_loan(patron="ann") == {"loan_id": "L999", "opened": "1999-01-01T00:00:00"}
    used.ctx.reseed(7)
    got = (used.ctx.now(), used.ctx.random(), used.ctx.new_id("loans"), used.ctx.usage())
    assert "L999" not in got and "1999" not in got[0]
    fresh = ce.load_toolkit(source, {t: {k: (v.model_dump() if hasattr(v, "model_dump") else dict(v))
                                           for k, v in getattr(used.db, t).items()}
                                   for t in LOANS_SCHEMA.tables})
    fresh.ctx.reseed(7)
    want = (fresh.ctx.now(), fresh.ctx.random(), fresh.ctx.new_id("loans"), fresh.ctx.usage())
    assert got == want
    assert used.ctx._db is used.db


NONID_UPDATE_SIG = ToolSig(
    name="retarget_loan",
    description="Rewrite a loan row named elsewhere.",
    args_fields=[FieldStat(name="spec", types=["dict"], optional=False)],
    kind="write",
    unclassified=False,
)
NONID_WRONG_BODY = (
    "loan_id = self.ctx.new_id(\"loans\")\n"
    "opened = self.ctx.now()\n"
    "self.db.loans[loan_id] = {\"loan_id\": loan_id, \"opened\": opened}\n"
    "return {\"loan_id\": loan_id, \"opened\": opened}\n"
)


def test_a_wrong_body_with_the_id_under_a_non_id_key_is_not_certified(tmp_path):
    call = ToolCall(id="u1", name="retarget_loan", args={"spec": {"target": "L200"}},
                    result={"loan_id": "L200", "opened": "2024-01-02T10:00:00"}, raw_ptr=PTR)
    source = ce.module_source(LOANS_SCHEMA, [NONID_UPDATE_SIG], {"retarget_loan": NONID_WRONG_BODY})
    box = ce.Sandbox(source, LOANS_DB, tmp_path,
                     call_context=ce.recorded_call_contexts([call], LOANS_SCHEMA))
    [outcome] = box.run([call])
    assert outcome["ok"] is True
    assert outcome["value"]["loan_id"] != "L200"


def test_the_recordings_own_order_decides_newness_not_the_handover_order():
    first = ToolCall(id="c1", name="open_loan", args={"patron": "ann"},
                     result={"loan_id": "L101", "opened": "2024-03-04T05:06:07"},
                     raw_ptr=RawPtr(file_hash="testfile", sim_index=0, msg_index=0), trace_id="t1")
    second = ToolCall(id="c2", name="get_loan", args={},
                      result={"loan_id": "L101"},
                      raw_ptr=RawPtr(file_hash="testfile", sim_index=0, msg_index=1), trace_id="t1")
    ordered = ce.recorded_call_contexts([first, second], LOANS_SCHEMA)
    assert ce.recorded_call_contexts([second, first], LOANS_SCHEMA) == ordered
    assert ordered["c1"]["new_ids"] == {"loans": "L101"}
    assert ordered["c2"]["new_ids"] == {}


def test_a_positioned_call_decides_newness_before_unpositioned_calls_in_its_trace():
    placed = ToolCall(id="p0", name="open_loan", args={"patron": "ann"},
                      result={"loan_id": "L101", "opened": "2024-03-04T05:06:07"},
                      raw_ptr=RawPtr(file_hash="testfile", sim_index=0, msg_index=0), trace_id="t1")
    earlier = ToolCall(id="u1", name="get_loan", args={},
                       result={"loan_id": "L101"}, raw_ptr=PTR, trace_id="t1")
    later = ToolCall(id="u2", name="get_loan", args={},
                     result={"loan_id": "L101"}, raw_ptr=PTR, trace_id="t1")
    feeds = ce.recorded_call_contexts([later, earlier, placed], LOANS_SCHEMA)
    assert feeds == ce.recorded_call_contexts([placed, earlier, later], LOANS_SCHEMA)
    assert feeds["p0"]["new_ids"] == {"loans": "L101"}
    assert feeds["u1"]["new_ids"] == {}
    assert feeds["u2"]["new_ids"] == {}


def test_dict_calls_in_one_trace_withhold_a_reobserved_id():
    first = {"id": "c1", "name": "open_loan", "args": {"patron": "ann"},
             "result": {"loan_id": "L101", "opened": "2024-03-04T05:06:07"}, "trace_id": "t1"}
    second = {"id": "c2", "name": "get_loan", "args": {},
              "result": {"loan_id": "L101"}, "trace_id": "t1"}
    feeds = ce.recorded_call_contexts([first, second], LOANS_SCHEMA)
    assert feeds["c1"]["new_ids"] == {"loans": "L101"}
    assert feeds["c2"]["new_ids"] == {}


def test_a_reseeded_context_matches_a_fresh_one_after_a_row_is_deleted():
    source = ce.module_source(LOANS_SCHEMA, [OPEN_SIG], {"open_loan": OPEN_BODY})
    used = ce.load_toolkit(source, {table: dict(rows) for table, rows in LOANS_DB.items()})
    del used.db.loans["L100"]
    used.ctx.reseed(7)
    used.ctx.feed_call({"now": None, "new_ids": {"loans": "L100"}})
    fresh = ce.load_toolkit(source, {t: {k: (v.model_dump() if hasattr(v, "model_dump") else dict(v))
                                          for k, v in getattr(used.db, t).items()}
                                  for t in LOANS_SCHEMA.tables})
    fresh.ctx.reseed(7)
    fresh.ctx.feed_call({"now": None, "new_ids": {"loans": "L100"}})
    assert used.ctx.new_id("loans") == fresh.ctx.new_id("loans") == "L100"
    assert (used.ctx.now(), used.ctx.random(), used.ctx.usage()) == (fresh.ctx.now(), fresh.ctx.random(), fresh.ctx.usage())
