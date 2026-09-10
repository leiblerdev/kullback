"""Tests for D215: a write is judged on every row it changed, not only on its own result.

The world here is an invented lending library. A member has a credit balance and a ledger of the
lines charged against it; a title has a fee; a loan names the member holding it and the title it is
against. `borrow_title` swaps the title on a loan, and the customer's tool does two more things its
own answer never mentions: it debits the member's credit by the difference between the two titles'
fees, and it appends a line to that member's ledger. The loan is named by the call's arguments; the
member is not, and is reached inside the tool through the loan's own foreign key.

That shape is the whole point. Nothing about it is any customer's, and every mechanism under test
here is about the shape and not about the names: which rows two sightings show moving, which of
them the call addressed, what relation the numbers hold, and what the replay does when a rebuilt
body answers the call correctly and leaves the rest of the world alone.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from conftest import PTR
from kullback.builder import effects
from kullback.gates.tool_runs import body_memorised_values_gate
from kullback.runner import replay
from kullback.runner.records import Column, EntitySchema, FieldStat, ToolCall, ToolSig, Trace, Turn
from kullback.runner.route import Router

WRITE = "borrow_title"

WORLD = {
    "titles": {"TT01": {"title_id": "TT01", "fee": 30},
               "TT09": {"title_id": "TT09", "fee": 12}},
    "members": {"MB01": {"member_id": "MB01", "credit": 100, "ledger": []}},
    "loans": {"LN01": {"loan_id": "LN01", "member_id": "MB01", "title_id": "TT01",
                       "status": "open"}},
}


def schema() -> EntitySchema:
    columns = [Column(table=table, name=name, **{"class": "hard"})
               for table, rows in WORLD.items()
               for name in sorted({key for row in rows.values() for key in row})]
    return EntitySchema(tables=sorted(WORLD), columns=columns,
                        id_patterns={"titles.title_id": r"^TT\d{2}$",
                                     "members.member_id": r"^MB\d{2}$",
                                     "loans.loan_id": r"^LN\d{2}$"})


def call(call_id: str, name: str, args: dict, result, requestor: str = "assistant") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args, result=result, requestor=requestor,
                    raw_ptr=PTR, trace_id="tr1")


def loan_row(title_id: str = "TT01") -> dict:
    return {"loan_id": "LN01", "member_id": "MB01", "title_id": title_id, "status": "open"}


def borrowing_calls() -> list[ToolCall]:
    """Read the loan, both titles and the member; swap the title; read the member again."""
    return [
        call("c1", "get_loan", {"loan_id": "LN01"}, loan_row()),
        call("c2", "get_title", {"title_id": "TT01"}, {"title_id": "TT01", "fee": 30}),
        call("c3", "get_title", {"title_id": "TT09"}, {"title_id": "TT09", "fee": 12}),
        call("c4", "get_member", {"member_id": "MB01"},
             {"member_id": "MB01", "credit": 100, "ledger": []}),
        call("c5", WRITE, {"loan_id": "LN01", "title_id": "TT09"}, loan_row("TT09")),
        call("c6", "get_member", {"member_id": "MB01"},
             {"member_id": "MB01", "credit": 82, "ledger": [{"amount": 18}]}),
    ]


def trace_of(calls: list[ToolCall], trace_id: str = "tr1") -> Trace:
    turns = [Turn(idx=0, role="assistant", content="How can I help?", raw_ptr=PTR),
             Turn(idx=1, role="user", content="Swap my loan to another title.", raw_ptr=PTR)]
    for one in calls:
        turns.append(Turn(idx=len(turns), role="assistant", content=None,
                          tool_call_ids=[one.id], raw_ptr=PTR))
        turns.append(Turn(idx=len(turns), role="tool", content=json.dumps(one.result),
                          tool_call_ids=[one.id], raw_ptr=PTR))
    turns.append(Turn(idx=len(turns), role="assistant", content="Done.", raw_ptr=PTR))
    turns.append(Turn(idx=len(turns), role="user", content="Thanks. ###STOP###", raw_ptr=PTR))
    for one in calls:
        one.trace_id = trace_id
    return Trace(trace_id=trace_id, raw_hash="r" * 64, ingest_version="1", source="test",
                 turns=turns, tool_calls=calls, raw_ptr=PTR, system_prompt="You are the agent.")


def observed(traces=None, db=None):
    return effects.observe_effects(traces or [trace_of(borrowing_calls())], schema(), {WRITE},
                                   db=db)


def column_of(seen, table: str, path: str):
    rows = [c for effect in seen[WRITE] for c in effect.columns
            if c.table == table and c.path == path]
    return rows[0] if rows else None


# --- rule 1: what two sightings of one row say the write did -----------------------


def test_a_column_a_later_read_shows_moved_is_credited_to_the_write_between_the_two_sightings():
    seen = observed()
    credit = column_of(seen, "members", "credit")
    assert credit is not None
    assert (credit.before, credit.after) == (100, 82)
    assert [effect.call_id for effect in seen[WRITE]] == ["c5"]


def test_a_list_a_write_appended_to_is_credited_at_its_length():
    """An append moves nothing but the length, and the length is a leaf of its own."""
    ledger = column_of(observed(), "members", "ledger[]")
    assert ledger is not None
    assert (ledger.before, ledger.after) == (0, 1)


def test_a_row_the_call_addressed_is_kept_apart_from_one_it_never_named():
    seen = observed()
    assert column_of(seen, "loans", "title_id").named is True
    assert column_of(seen, "members", "credit").named is False


def test_a_column_no_call_read_before_the_write_takes_its_before_from_the_starting_state():
    """Nothing read the member first, so the pin is the only evidence of what it held."""
    calls = [call("c1", "get_loan", {"loan_id": "LN01"}, loan_row()),
             call("c2", WRITE, {"loan_id": "LN01", "title_id": "TT09"}, loan_row("TT09")),
             call("c3", "get_member", {"member_id": "MB01"},
                  {"member_id": "MB01", "credit": 82, "ledger": [{"amount": 18}]})]
    seen = observed([trace_of(calls)], db=WORLD)
    credit = column_of(seen, "members", "credit")
    assert (credit.before, credit.after) == (100, 82)
    assert credit.ambiguous is False


def test_a_column_that_did_not_move_leaves_no_evidence():
    calls = borrowing_calls()
    calls[-1] = call("c6", "get_member", {"member_id": "MB01"},
                     {"member_id": "MB01", "credit": 100, "ledger": []})
    seen = effects.observe_effects([trace_of(calls)], schema(), {WRITE})
    assert column_of(seen, "members", "credit") is None


def test_the_evidence_a_writer_sees_is_the_runs_it_was_handed_and_no_others():
    """The held-out filter is the caller's, and it is the whole of what a writer is shown."""
    anchor = borrowing_calls()
    for index, one in enumerate(anchor):
        one.id = f"held{index}"
    seed, held = trace_of(borrowing_calls(), "tr1"), trace_of(anchor, "tr2")
    shown = effects.effects_block(WRITE, effects.observe_effects([seed], schema(), {WRITE})[WRITE])
    assert "c5" in shown and "held4" not in shown
    both = effects.observe_effects([seed, held], schema(), {WRITE})
    # The replay check is code and is allowed every Run, which is why it is read from its own call.
    assert sorted(effects.replay_evidence(both)) == ["c5", "held4"]


# --- rule 2: the formula, and the section of the prompt it sits in -----------------


def test_the_formula_names_the_two_title_rows_and_the_column_that_moved():
    credit = column_of(observed(), "members", "credit")
    found = [line for line in credit.formulas if "changes by" in line]
    assert found, credit.formulas
    assert found[0].startswith("members.credit changes by")
    assert found[0].count("titles.fee") == 2


def test_the_formula_names_columns_and_never_the_values_it_was_found_from():
    credit = column_of(observed(), "members", "credit")
    for line in credit.formulas:
        assert "18" not in line and "82" not in line and "30" not in line


def test_the_block_shows_the_row_the_call_never_named_and_leaves_out_the_one_it_did():
    block = effects.effects_block(WRITE, observed()[WRITE])
    assert "members.credit: 100 became 82" in block
    assert "members.ledger[]" in block
    assert "loans.title_id" not in block


def test_a_tool_nothing_was_observed_of_gets_no_section_at_all():
    assert effects.effects_block("get_member", []) == ""
    assert effects.effects_block(WRITE, observed()[WRITE]) != ""


def test_two_calls_that_agree_on_a_formula_state_it_as_the_rule():
    second = [one.model_copy(update={"id": f"{one.id}b"}) for one in borrowing_calls()]
    block = effects.effects_block(
        WRITE, effects.observe_effects([trace_of(borrowing_calls(), "tr1"),
                                        trace_of(second, "tr2")], schema(), {WRITE})[WRITE])
    assert "on every call of this tool that shows the column moving" in block


def test_two_calls_that_disagree_on_a_formula_say_so_and_show_both():
    """One Run charges the difference between the two fees and the other charges the new fee."""
    other = borrowing_calls()
    for one in other:
        one.id = f"{one.id}b"
    other[5] = call("c6b", "get_member", {"member_id": "MB01"},
                    {"member_id": "MB01", "credit": 88, "ledger": [{"amount": 12}]})
    block = effects.effects_block(
        WRITE, effects.observe_effects([trace_of(borrowing_calls(), "tr1"),
                                        trace_of(other, "tr2")], schema(), {WRITE})[WRITE])
    assert "the calls do not agree on members.credit" in block


# --- rule 5: an after value is a value the body has to work out --------------------


def test_a_body_that_writes_down_the_value_a_write_was_seen_to_leave_is_refused():
    source = ("loan = self.db.loans[loan_id]\n"
              "self.db.members[loan.member_id].credit = 82\n"
              "return loan\n")
    values = effects.effect_values(observed()[WRITE])
    ruling = body_memorised_values_gate(source, schema(), WORLD, [], _write_sig(),
                                        effect_values=values)
    assert ruling.passed is False
    assert "members.credit" in ruling.failures[0]


def test_the_same_body_computing_the_value_passes():
    source = ("loan = self.db.loans[loan_id]\n"
              "member = self.db.members[loan.member_id]\n"
              "member.credit = member.credit - (self.db.titles[title_id].fee\n"
              "                                 - self.db.titles[loan.title_id].fee)\n"
              "return loan\n")
    ruling = body_memorised_values_gate(source, schema(), WORLD, [], _write_sig(),
                                        effect_values=effects.effect_values(observed()[WRITE]))
    assert ruling.passed is True, ruling.failures


def _write_sig() -> ToolSig:
    return ToolSig(name=WRITE, description="Swap the title on a loan.", kind="write",
                   unclassified=False,
                   args_fields=[FieldStat(name="loan_id", types=["str"], optional=False),
                                FieldStat(name="title_id", types=["str"], optional=False)])


# --- rule 3: the check at replay ---------------------------------------------------


class Title(BaseModel):
    title_id: str
    fee: int


class Member(BaseModel):
    member_id: str
    credit: int
    ledger: list = []


class Loan(BaseModel):
    loan_id: str
    member_id: str
    title_id: str
    status: str


class DB(BaseModel):
    titles: dict[str, Title] = {}
    members: dict[str, Member] = {}
    loans: dict[str, Loan] = {}


class ForgetfulToolkit:
    """Answers every call the way the recording did and never touches the member's row."""

    def __init__(self, db: dict):
        self.db = DB.model_validate(db)

    def get_loan(self, loan_id):
        return self.db.loans[loan_id].model_dump()

    def get_title(self, title_id):
        return self.db.titles[title_id].model_dump()

    def get_member(self, member_id):
        return self.db.members[member_id].model_dump()

    def borrow_title(self, loan_id, title_id):
        self.db.loans[loan_id].title_id = title_id
        return self.db.loans[loan_id].model_dump()


class WholeToolkit(ForgetfulToolkit):
    """The same, and it moves the two things the recording shows the write moving."""

    def borrow_title(self, loan_id, title_id):
        loan = self.db.loans[loan_id]
        member = self.db.members[loan.member_id]
        charge = self.db.titles[title_id].fee - self.db.titles[loan.title_id].fee
        member.credit = member.credit + charge
        member.ledger = [*member.ledger, {"amount": -charge}]
        loan.title_id = title_id
        return loan.model_dump()


def sigs() -> list[ToolSig]:
    return [ToolSig(name="get_loan", kind="read"), ToolSig(name="get_title", kind="read"),
            ToolSig(name="get_member", kind="read"), ToolSig(name=WRITE, kind="write")]


def do_replay(tmp_path, toolkit_class, evidence=None) -> replay.Replay:
    trace = trace_of(borrowing_calls())
    world = json.loads(json.dumps(WORLD))
    router = Router(env_tools_module=toolkit_class(json.loads(json.dumps(WORLD))),
                    starting_state=world, tool_sigs=sigs())
    return replay.replay_trace(trace, router, workdir=tmp_path / "runs" / "t1", task_id="t1",
                               write_tools={WRITE}, effects=evidence)


def evidence_for() -> dict:
    return effects.replay_evidence(observed(db=WORLD))


def test_a_body_that_forgets_the_debit_fails_the_write_check_with_the_effect_reason(tmp_path):
    result = do_replay(tmp_path, ForgetfulToolkit, evidence_for())
    assert result.confirmed is False
    write = next(c for c in result.checks if c["tool"] == WRITE)
    assert write["verdict"] in replay.AGREES, "the write answered exactly what the recording did"
    assert write["effect_failures"], "and left the rows the recording shows it moving"
    assert any(f"{WRITE} write: effect members.credit" in line for line in result.reasons)
    assert result.counts["effect_failures"] >= 2
    assert result.counts["writes_matched"] == 0


def test_the_read_that_saw_the_stale_value_is_marked_as_the_writes_doing(tmp_path):
    result = do_replay(tmp_path, ForgetfulToolkit, evidence_for())
    reads = [c for c in result.checks if c["tool"] == "get_member"
             and c["verdict"] not in replay.AGREES]
    assert reads and reads[0][replay.DOWNSTREAM] == "c5"
    assert reads[0]["downstream_tool"] == WRITE
    assert result.counts["effects_downstream"] == 1


def test_a_body_that_moves_every_row_the_recording_moved_confirms(tmp_path):
    result = do_replay(tmp_path, WholeToolkit, evidence_for())
    assert result.confirmed is True, result.reasons
    assert result.counts["effect_checks"] > 0
    assert result.counts["effect_failures"] == 0


def test_without_the_evidence_the_forgetful_body_is_only_caught_by_the_later_read(tmp_path):
    """What the harness scored before D215: the write passes and the read carries the blame."""
    result = do_replay(tmp_path, ForgetfulToolkit, None)
    assert result.counts["writes_matched"] == 1
    assert result.counts["effect_checks"] == 0
    assert any("get_member read" in line for line in result.reasons)


# --- D234: an ambiguous effect is credited to the write that owes it, not to the last call ---

CREATE, LOOKUP, ADJUST, SETTLE = "open_loan", "read_ledger", "adjust_credit", "settle_loan"


def two_writer_calls() -> list[ToolCall]:
    """Read the member; a create names it, a lookup only names the loan; read the member again."""
    return [
        call("c1", "get_member", {"member_id": "MB01"},
             {"member_id": "MB01", "credit": 100, "ledger": []}),
        call("c2", CREATE, {"member_id": "MB01"}, loan_row()),
        call("c3", LOOKUP, {"loan_id": "LN01"},
             {"member_id": "MB01", "credit": 82, "ledger": [{"amount": 18}]}),
    ]


def naming_row_seen():
    """A create names the member row; a later lookup names only the loan."""
    return effects.observe_effects([trace_of(two_writer_calls())], schema(), {CREATE, LOOKUP})


def test_the_write_naming_the_row_takes_the_credit():
    seen = naming_row_seen()
    assert sorted(effect.call_id for effect in seen[CREATE]) == ["c2"]
    assert LOOKUP not in seen
    evidence = effects.replay_evidence(seen)
    assert [row["path"] for row in evidence["c2"] if row["table"] == "members"] != []
    assert all(row["table"] != "members" for row in evidence.get("c3", []))


def test_the_named_row_credit_is_checked_and_owned():
    seen = naming_row_seen()
    credit = next(column for effect in seen[CREATE] for column in effect.columns
                  if column.table == "members" and column.path == "credit")
    assert (credit.before, credit.after) == (100, 82)
    assert credit.ambiguous is True
    assert credit.checked is True
    assert credit.unattributed is False


def test_a_tool_seen_moving_the_column_alone_elsewhere_keeps_the_credit():
    first = [
        call("c1", "get_member", {"member_id": "MB01"},
             {"member_id": "MB01", "credit": 100, "ledger": []}),
        call("c2", ADJUST, {"loan_id": "LN01"}, loan_row()),
        call("c3", "get_member", {"member_id": "MB01"},
             {"member_id": "MB01", "credit": 90, "ledger": []}),
    ]
    second = [
        call("d1", "get_member", {"member_id": "MB01"},
             {"member_id": "MB01", "credit": 100, "ledger": []}),
        call("d2", ADJUST, {"loan_id": "LN01"}, loan_row()),
        call("d3", SETTLE, {"loan_id": "LN01"}, loan_row()),
        call("d4", "get_member", {"member_id": "MB01"},
             {"member_id": "MB01", "credit": 90, "ledger": []}),
    ]
    seen = effects.observe_effects([trace_of(first, "tr1"), trace_of(second, "tr2")],
                                   schema(), {ADJUST, SETTLE})
    owned = [column for effect in seen[ADJUST] for column in effect.columns
             if column.table == "members" and column.path == "credit"
             and not column.unattributed]
    assert sorted((column.after, column.checked) for column in owned) == [(90, True), (90, True)]
    assert SETTLE not in seen
    evidence = effects.replay_evidence(seen)
    assert [row["path"] for row in evidence["d2"] if row["table"] == "members"] != []
    assert all(row["table"] != "members" for row in evidence.get("d3", []))


def nowhere_owned_calls() -> list[ToolCall]:
    """Two writes neither of which names the member, and no lone move anywhere in the trace."""
    return [
        call("c1", "get_member", {"member_id": "MB01"},
             {"member_id": "MB01", "credit": 100, "ledger": []}),
        call("c2", ADJUST, {"loan_id": "LN01"}, loan_row()),
        call("c3", SETTLE, {"loan_id": "LN01"}, loan_row()),
        call("c4", "get_member", {"member_id": "MB01"},
             {"member_id": "MB01", "credit": 90, "ledger": []}),
    ]


def nowhere_owned_seen():
    return effects.observe_effects([trace_of(nowhere_owned_calls())], schema(), {ADJUST, SETTLE})


def test_a_column_no_write_owes_is_unattributed_and_not_charged_to_the_last_call():
    seen = nowhere_owned_seen()
    evidence = effects.replay_evidence(seen)
    stray = evidence.get(effects.UNATTRIBUTED, [])
    assert [(row["table"], row["path"]) for row in stray] == [("members", "credit")]
    assert all(row["table"] != "members" for row in evidence.get("c3", []))
    assert effects.counts(seen)["effects_unattributed"] == 1


class StillWorld:
    """Answers every call the way the recording did and moves nothing at all."""

    def __init__(self, db: dict):
        self.db = db

    def get_member(self, member_id):
        return dict(self.db["members"][member_id])

    def adjust_credit(self, loan_id):
        return dict(self.db["loans"][loan_id])

    def settle_loan(self, loan_id):
        return dict(self.db["loans"][loan_id])


def test_an_unattributed_column_fails_no_call_at_replay(tmp_path):
    evidence = effects.replay_evidence(nowhere_owned_seen())
    world = json.loads(json.dumps(WORLD))
    router = Router(env_tools_module=StillWorld(json.loads(json.dumps(WORLD))),
                    starting_state=world,
                    tool_sigs=[ToolSig(name="get_member", kind="read"),
                               ToolSig(name=ADJUST, kind="write"),
                               ToolSig(name=SETTLE, kind="write")])
    result = replay.replay_trace(trace_of(nowhere_owned_calls()), router,
                                 workdir=tmp_path / "runs" / "t1", task_id="t1",
                                 write_tools={ADJUST, SETTLE}, effects=evidence)
    assert result.counts["effect_failures"] == 0
    assert not any(check.get("effect_failures") for check in result.checks)
    last = next(check for check in result.checks if check.get("call_id") == "c3")
    assert last["verdict"] in replay.AGREES


def test_a_recorded_call_claiming_the_reserved_key_loses_no_evidence():
    owned = effects.WriteEffect(tool=WRITE, call_id=effects.UNATTRIBUTED, trace_id="tr1", columns=[
        effects.EffectColumn(table="loans", row_id="LN01", path="title_id",
                             before="TT01", after="TT09")])
    stray = effects.EffectColumn(table="members", row_id="MB01", path="credit",
                                 before=100, after=82, ambiguous=True,
                                 checked=False, unattributed=True)
    holder = effects.WriteEffect(tool=WRITE, call_id="c9", trace_id="tr1", columns=[stray])
    evidence = effects.replay_evidence({WRITE: [owned, holder]})
    assert [row["path"] for row in evidence[effects.UNATTRIBUTED]] == ["title_id"]
    bumped = effects.UNATTRIBUTED + "\x00"
    assert [row["path"] for row in evidence[bumped] if row["unattributed"]] == ["credit"]


def two_owner_seen():
    """Two writes both naming the member row, read again after each."""
    calls = [
        call("c1", "get_member", {"member_id": "MB01"},
             {"member_id": "MB01", "credit": 100, "ledger": []}),
        call("c2", ADJUST, {"member_id": "MB01"}, loan_row()),
        call("c3", SETTLE, {"member_id": "MB01"}, loan_row()),
        call("c4", "get_member", {"member_id": "MB01"},
             {"member_id": "MB01", "credit": 90, "ledger": []}),
    ]
    return effects.observe_effects([trace_of(calls)], schema(), {ADJUST, SETTLE})


def test_two_owners_check_only_the_later_write():
    seen = two_owner_seen()
    first = next(column for effect in seen[ADJUST] for column in effect.columns
                 if column.table == "members" and column.path == "credit")
    second = next(column for effect in seen[SETTLE] for column in effect.columns
                  if column.table == "members" and column.path == "credit")
    assert first.checked is False
    assert second.checked is True


def test_two_owner_evidence_and_counts_stay_with_the_last():
    seen = two_owner_seen()
    evidence = effects.replay_evidence(seen)
    assert all(row["table"] != "members" for row in evidence.get("c2", []))
    assert [row["path"] for row in evidence["c3"] if row["table"] == "members"] == ["credit"]
    assert effects.counts(seen)["effects_ambiguous"] == 2


def test_a_single_write_span_is_credited_and_checked_as_before():
    credit = column_of(observed(), "members", "credit")
    assert credit.ambiguous is False
    assert credit.checked is True
    assert credit.unattributed is False


def test_unattributed_columns_are_shown_to_no_writer():
    seen = nowhere_owned_seen()
    assert effects.effects_block(SETTLE, seen.get(SETTLE, [])) == ""
    assert effects.effects_block(ADJUST, seen.get(ADJUST, [])) == ""
