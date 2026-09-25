"""The agent user's two readings: its choices, from what was recorded and tagged by source, and its
own account, read only and cut to its own rows and the columns recorded users stated.

The domain is invented: a seed library that lends seed packets, with members, loans and a pickup
option. Nothing here names any customer's world.
"""

from __future__ import annotations

from kullback.ai.provider import TestModel
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, UserFact, UserRules
from kullback.user import account as account_mod
from kullback.user import context as context_mod
from kullback.user import guards as guards_mod
from kullback.user.agent import AgentUser
from kullback.user.simulated import SimulatedUser
from kullback.user.tools import Toolbox, user_tools

PTR = RawPtr(file_hash="seeds", sim_index=0)
WRITES = ["reserve_packet"]


def _trace(trace_id: str, said: list[str], args: dict, result: dict | None = None) -> Trace:
    turns = [Turn(idx=i, role="user" if i % 2 == 0 else "assistant", content=text, raw_ptr=PTR)
             for i, text in enumerate(said)]
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="v", source="test", turns=turns,
                 tool_calls=[ToolCall(id=f"{trace_id}-c1", name="reserve_packet", args=args,
                                      result=result if result is not None else dict(args), raw_ptr=PTR)],
                 raw_ptr=PTR)


def _rules(email: str) -> UserRules:
    return UserRules(facts=[UserFact(field="email", value=email), UserFact(field="name", value="Ada Lind")])


# --- choices ----------------------------------------------------------------------------------

def _corpus() -> tuple[dict, dict]:
    traces = {
        "own": _trace("own", ["Reserve me a packet."], {"pickup": "counter"}),
        "same_customer": _trace("same_customer", ["A packet please."], {"pickup": "counter",
                                                                        "wrapping": "paper"}),
        "stranger_a": _trace("stranger_a", ["Hi."], {"pickup": "locker", "wrapping": "cloth",
                                                     "size": "large"}),
        "stranger_b": _trace("stranger_b", ["Hi."], {"pickup": "locker", "size": "large"}),
        "stranger_c": _trace("stranger_c", ["Hi."], {"size": "small", "voucher": "VCH-40017"}),
    }
    rules = {"own": _rules("ada@seeds.example"), "same_customer": _rules("ada@seeds.example"),
             "stranger_a": _rules("bo@seeds.example"), "stranger_b": _rules("cy@seeds.example"),
             "stranger_c": _rules("di@seeds.example")}
    return traces, rules


def _book() -> account_mod.ChoiceBook:
    traces, rules = _corpus()
    return account_mod.choice_book(
        traces["own"], WRITES, customer=account_mod.customer_keys(rules["own"], ["email", "name"]),
        traces=traces, rules=rules, identity_fields=["email", "name"])


def test_a_choice_is_answered_from_this_tasks_recording_first():
    value, source, line = _book().answer("pickup")
    assert (value, source) == ("counter", account_mod.RECORDING)
    assert "source: recording" in line


def test_a_choice_this_recording_lacks_is_answered_from_the_same_customers_other_recording():
    value, source, _ = _book().answer("wrapping")
    assert (value, source) == ("paper", account_mod.OTHER_RECORDING)


def test_a_choice_no_recording_of_this_customer_holds_is_what_most_recorded_users_chose():
    value, source, line = _book().answer("size")
    assert (value, source) == ("large", account_mod.POPULATION)
    assert "2 of 3 recorded users" in line


def test_a_choice_nobody_recorded_says_so_and_invents_nothing():
    assert _book().answer("gift_message") == (None, None, account_mod.NO_CHOICE)


def test_another_customers_id_is_never_offered_as_a_population_choice():
    assert _book().answer("voucher") == (None, None, account_mod.NO_CHOICE)


# --- the account ------------------------------------------------------------------------------

TABLES = {
    "members": {
        "M-1001": {"member_id": "M-1001", "email": "ada@seeds.example", "name": "Ada Lind",
                   "plot": "PLOT-7731", "pin": "4455"},
        "M-2002": {"member_id": "M-2002", "email": "bo@seeds.example", "name": "Bo Park",
                   "plot": "PLOT-8842", "pin": "9911"},
    },
    "loans": {
        "L-1": {"loan_id": "L-1", "member_id": "M-1001", "due": "2031-04-02"},
        "L-2": {"loan_id": "L-2", "member_id": "M-2002", "due": "2031-05-09"},
    },
}


def _stated() -> frozenset:
    """A recording whose user said its plot and a due date, and never its pin."""
    trace = _trace("seen", ["My plot is PLOT-5555, and when is it due?", "ok",
                            "So 2030-01-01 then."],
                   {"member_id": "M-9"}, {"plot": "PLOT-5555", "pin": "1234", "due": "2030-01-01"})
    return account_mod.stated_columns({"seen": trace})


def _account() -> account_mod.AccountView:
    return account_mod.account_view(TABLES, {"email": "ada@seeds.example", "name": "Ada Lind"},
                                    _stated())


def test_only_columns_some_recorded_user_stated_are_ever_mined_as_readable():
    assert _stated() == frozenset({"plot", "due"})


def test_the_account_shows_this_customers_rows_and_never_another_customers():
    text, given = _account().read()
    assert "PLOT-7731" in text and "2031-04-02" in text
    assert "PLOT-8842" not in text and "2031-05-09" not in text
    assert all(tag in text for tag in ("source: account",))
    assert set(given) == {"PLOT-7731", "2031-04-02"}


def test_the_account_never_shows_a_column_no_recorded_user_stated():
    text, _ = _account().read("pin")
    assert text == account_mod.NO_ACCOUNT
    everything, _ = _account().read()
    assert "4455" not in everything and "ada@seeds.example" not in everything


def test_a_customer_the_identity_does_not_match_has_no_account():
    view = account_mod.account_view(TABLES, {"email": "nobody@seeds.example", "name": "No One"},
                                    _stated())
    assert view.read() == (account_mod.NO_ACCOUNT, {})


def test_the_account_is_a_snapshot_the_run_cannot_write_through():
    tables = {"members": {"M-1001": dict(TABLES["members"]["M-1001"])}}
    view = account_mod.account_view(tables, {"email": "ada@seeds.example", "name": "Ada Lind"},
                                    _stated())
    tables["members"]["M-1001"]["plot"] = "PLOT-0000"
    assert "PLOT-7731" in view.read("plot")[0]


# --- the tools and the guard in a turn --------------------------------------------------------

def _agent(replies: list[str]) -> AgentUser:
    rules = UserRules(facts=[UserFact(field="email", value="ada@seeds.example"),
                             UserFact(field="name", value="Ada Lind"),
                             UserFact(field="goal", value="I want to reserve a packet.")])
    ctx = context_mod.curate("t", rules)
    return AgentUser(ctx, SimulatedUser(rules), TestModel(replies, loop=True),
                     choices=_book(), account=_account())


def test_the_two_readings_are_tools_only_where_the_caller_built_them():
    rules = UserRules(facts=[UserFact(field="email", value="ada@seeds.example")])
    bare = [tool.name for tool in user_tools(Toolbox(context_mod.curate("t", rules)))]
    assert "my_choices" not in bare and "my_account" not in bare
    built = [tool.name for tool in user_tools(Toolbox(context_mod.curate("t", rules),
                                                      choices=_book(), account=_account()))]
    assert built[-2:] == ["my_choices", "my_account"]


def test_a_turn_stating_an_account_value_nobody_asked_for_falls_back_to_the_floor():
    user = _agent(["Hello, my plot is PLOT-7731."])
    user.box.my_account("plot")
    said = user.reply([{"role": "assistant", "content": "How can I help you today?"}])
    assert "PLOT-7731" not in said
    assert user.counts["fallback_turns"] == 1
    assert user.guards.counts[guards_mod.ACCOUNT_UNASKED] == 1
    assert user.events[-1].payload["agent_turn_dropped"] == guards_mod.ACCOUNT_UNASKED


def test_a_turn_stating_an_account_value_that_was_asked_for_goes_through_tagged_account():
    user = _agent(["It is PLOT-7731."])
    user.box.my_account("plot")
    said = user.reply([{"role": "assistant", "content": "Which plot is yours?"}])
    assert "PLOT-7731" in said
    assert user.counts["agent_user_turns"] == 1
    assert user.events[-1].payload["sources"] == {"plot": account_mod.ACCOUNT}


def test_a_chosen_value_is_tagged_with_the_layer_that_answered_it():
    user = _agent(["Paper wrapping, please."])
    user.box.my_choices("wrapping")
    user.reply([{"role": "assistant", "content": "Paper or cloth for the wrapping?"}])
    assert user.events[-1].payload["sources"] == {"wrapping": account_mod.OTHER_RECORDING}
