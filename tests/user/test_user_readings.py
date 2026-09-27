"""The agent user's two readings: its choices, from what was recorded and tagged by source, and its
own account, read only and cut to its own rows and the columns recorded users stated.

Every name here is abstract: one write, two tables, option kinds and columns named by letter, so
the tests prove the mechanism and not any world.
"""

from __future__ import annotations

from kullback.ai.provider import ModelReply, TestModel
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, UserFact, UserRules
from kullback.user import account as account_mod
from kullback.user import context as context_mod
from kullback.user import guards as guards_mod
from kullback.user.agent import AgentUser
from kullback.user.simulated import SimulatedUser
from kullback.user.tools import Toolbox, user_tools

PTR = RawPtr(file_hash="abstract", sim_index=0)
WRITES = ["write_a"]


def _trace(trace_id: str, said: list[str], args: dict, result: dict | None = None) -> Trace:
    turns = [Turn(idx=i, role="user" if i % 2 == 0 else "assistant", content=text, raw_ptr=PTR)
             for i, text in enumerate(said)]
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="v", source="test", turns=turns,
                 tool_calls=[ToolCall(id=f"{trace_id}-c1", name="write_a", args=args,
                                      result=result if result is not None else dict(args), raw_ptr=PTR)],
                 raw_ptr=PTR)


def _rules(key: str) -> UserRules:
    return UserRules(facts=[UserFact(field="ident_a", value=key), UserFact(field="ident_b", value="Ann Aa")])


# --- choices ----------------------------------------------------------------------------------

def _corpus() -> tuple[dict, dict]:
    traces = {
        "own": _trace("own", ["I want the thing done."], {"kind_a": "option_one"}),
        "same_customer": _trace("same_customer", ["The thing, please."], {"kind_a": "option_one",
                                                                        "kind_b": "option_two"}),
        "stranger_a": _trace("stranger_a", ["Hi."], {"kind_a": "option_three", "kind_b": "option_four",
                                                     "kind_c": "option_five"}),
        "stranger_b": _trace("stranger_b", ["Hi."], {"kind_a": "option_three", "kind_c": "option_five"}),
        "stranger_c": _trace("stranger_c", ["Hi."], {"kind_c": "option_six", "kind_d": "CODE-40017"}),
    }
    rules = {"own": _rules("ada@x.example"), "same_customer": _rules("ada@x.example"),
             "stranger_a": _rules("bo@x.example"), "stranger_b": _rules("cy@x.example"),
             "stranger_c": _rules("di@x.example")}
    return traces, rules


def _book() -> account_mod.ChoiceBook:
    traces, rules = _corpus()
    return account_mod.choice_book(
        traces["own"], WRITES, customer=account_mod.customer_keys(rules["own"], ["ident_a", "ident_b"]),
        traces=traces, rules=rules, identity_fields=["ident_a", "ident_b"])


def test_a_choice_is_answered_from_this_tasks_recording_first():
    value, source, line = _book().answer("kind_a")
    assert (value, source) == ("option_one", account_mod.RECORDING)
    assert "source: recording" in line


def test_a_choice_this_recording_lacks_is_answered_from_the_same_customers_other_recording():
    value, source, _ = _book().answer("kind_b")
    assert (value, source) == ("option_two", account_mod.OTHER_RECORDING)


def test_a_choice_no_recording_of_this_customer_holds_is_what_most_recorded_users_chose():
    value, source, line = _book().answer("kind_c")
    assert (value, source) == ("option_five", account_mod.POPULATION)
    assert "2 of 3 recorded users" in line


def test_an_unmatched_kind_lists_the_kinds_held_nearest_layer_first():
    value, source, line = _book().answer("kind_e")
    assert (value, source) == (None, None)
    assert line.startswith(account_mod.NO_CHOICE)
    assert "kind_a (source: recording)" in line
    assert "kind_b (source: other_recording)" in line
    assert "kind_c (source: population)" in line
    assert line.index("kind_a") < line.index("kind_b") < line.index("kind_c")


def test_an_unmatched_kind_on_an_empty_book_says_so_and_invents_nothing():
    assert account_mod.ChoiceBook().answer("kind_e") == (None, None, account_mod.NO_CHOICE)


def test_another_customers_id_is_never_offered_as_a_population_choice():
    value, source, line = _book().answer("kind_d")
    assert (value, source) == (None, None)
    assert line.startswith(account_mod.NO_CHOICE) and "kind_a" in line



# --- the account ------------------------------------------------------------------------------

TABLES = {
    "table_a": {
        "M-1001": {"key_a": "M-1001", "ident_a": "ada@x.example", "ident_b": "Ann Aa",
                   "col_x": "REF-7731", "col_hidden": "4455"},
        "M-2002": {"key_a": "M-2002", "ident_a": "bo@x.example", "ident_b": "Ben Bb",
                   "col_x": "REF-8842", "col_hidden": "9911"},
    },
    "table_b": {
        "L-1": {"key_b": "L-1", "key_a": "M-1001", "col_y": "2031-04-02"},
        "L-2": {"key_b": "L-2", "key_a": "M-2002", "col_y": "2031-05-09"},
    },
}


def _stated() -> frozenset:
    """A recording whose user said two of the returned values, and never the third."""
    trace = _trace("seen", ["It is REF-5555, and when is it?", "ok",
                            "So 2030-01-01 then."],
                   {"key_a": "M-9"}, {"col_x": "REF-5555", "col_hidden": "1234", "col_y": "2030-01-01"})
    return account_mod.stated_columns({"seen": trace})


def _account() -> account_mod.AccountView:
    return account_mod.account_view(TABLES, {"ident_a": "ada@x.example", "ident_b": "Ann Aa"},
                                    _stated())


def test_only_columns_some_recorded_user_stated_are_ever_mined_as_readable():
    assert _stated() == frozenset({"col_x", "col_y"})


def test_the_account_shows_this_customers_rows_and_never_another_customers():
    text, given = _account().read()
    assert "REF-7731" in text and "2031-04-02" in text
    assert "REF-8842" not in text and "2031-05-09" not in text
    assert all(tag in text for tag in ("source: account",))
    assert set(given) == {"REF-7731", "2031-04-02"}


def test_the_account_never_shows_a_column_no_recorded_user_stated():
    text, _ = _account().read("col_hidden")
    assert text == account_mod.NO_ACCOUNT
    everything, _ = _account().read()
    assert "4455" not in everything and "ada@x.example" not in everything


def test_a_customer_the_identity_does_not_match_has_no_account():
    view = account_mod.account_view(TABLES, {"ident_a": "nobody@x.example", "ident_b": "No One"},
                                    _stated())
    assert view.read() == (account_mod.NO_ACCOUNT, {})


def _id_rules() -> UserRules:
    return UserRules(facts=[UserFact(field="key_a", value="M-1001"),
                            UserFact(field="goal", value="I want the thing done.")])


def test_id_keys_are_the_ids_outside_the_person_shaped_identity():
    assert account_mod.id_keys(_id_rules(), ["ident_a", "ident_b"]) == ["M-1001"]
    assert account_mod.customer_keys(_rules("ada@x.example"), ["ident_a", "ident_b"]) == [
        "ada@x.example"]


def test_a_customer_keyed_by_an_id_alone_gets_their_own_rows_only():
    view = account_mod.account_view(TABLES, {}, _stated(),
                                    keys=account_mod.id_keys(_id_rules(), ["ident_a", "ident_b"]))
    text, given = view.read()
    assert "REF-7731" in text and "2031-04-02" in text
    assert "REF-8842" not in text and "2031-05-09" not in text
    assert set(given) == {"REF-7731", "2031-04-02"}


def test_two_ids_naming_two_unrelated_rows_are_no_account_never_a_guess():
    view = account_mod.account_view(TABLES, {}, _stated(), keys=["M-1001", "M-2002"])
    assert view.read() == (account_mod.NO_ACCOUNT, {})


def test_of_two_named_rows_the_one_another_row_plainly_carries_is_the_owner():
    view = account_mod.account_view(TABLES, {}, _stated(), keys=["M-1001", "L-1"])
    text, _ = view.read()
    assert "REF-7731" in text and "REF-8842" not in text


def test_the_account_is_a_snapshot_the_run_cannot_write_through():
    tables = {"table_a": {"M-1001": dict(TABLES["table_a"]["M-1001"])}}
    view = account_mod.account_view(tables, {"ident_a": "ada@x.example", "ident_b": "Ann Aa"},
                                    _stated())
    tables["table_a"]["M-1001"]["col_x"] = "REF-0000"
    assert "REF-7731" in view.read("col_x")[0]


# --- the tools and the guard in a turn --------------------------------------------------------

def _agent(replies: list[str]) -> AgentUser:
    rules = UserRules(facts=[UserFact(field="ident_a", value="ada@x.example"),
                             UserFact(field="ident_b", value="Ann Aa"),
                             UserFact(field="goal", value="I want the thing done.")])
    ctx = context_mod.curate("t", rules)
    return AgentUser(ctx, SimulatedUser(rules), TestModel(replies, loop=True),
                     choices=_book(), account=_account())


def test_the_two_readings_are_tools_only_where_the_caller_built_them():
    rules = UserRules(facts=[UserFact(field="ident_a", value="ada@x.example")])
    bare = [tool.name for tool in user_tools(Toolbox(context_mod.curate("t", rules)))]
    assert "my_choices" not in bare and "my_account" not in bare
    built = [tool.name for tool in user_tools(Toolbox(context_mod.curate("t", rules),
                                                      choices=_book(), account=_account()))]
    assert built[-2:] == ["my_choices", "my_account"]


def test_a_turn_stating_an_account_value_nobody_asked_for_falls_back_to_the_floor():
    user = _agent(["Hello, my col x is REF-7731."])
    user.box.my_account("col_x")
    said = user.reply([{"role": "assistant", "content": "How can I help you today?"}])
    assert "REF-7731" not in said
    assert user.counts["fallback_turns"] == 1
    assert user.guards.counts[guards_mod.ACCOUNT_UNASKED] == 1
    assert user.events[-1].payload["agent_turn_dropped"] == guards_mod.ACCOUNT_UNASKED


def test_a_turn_stating_an_account_value_that_was_asked_for_goes_through_tagged_account():
    user = _agent(["It is REF-7731."])
    user.box.my_account("col_x")
    said = user.reply([{"role": "assistant", "content": "What is your col x?"}])
    assert "REF-7731" in said
    assert user.counts["agent_user_turns"] == 1
    assert user.events[-1].payload["sources"] == {"col_x": account_mod.ACCOUNT}


def test_a_chosen_value_is_tagged_with_the_layer_that_answered_it():
    user = _agent(["Option_two, please."])
    user.box.my_choices("kind_b")
    user.reply([{"role": "assistant", "content": "Which kind_b would you like?"}])
    assert user.events[-1].payload["sources"] == {"kind_b": account_mod.OTHER_RECORDING}


def test_the_prompt_names_every_tool_the_harness_registers():
    from kullback.user import skills as skills_mod
    rules = UserRules(facts=[UserFact(field="ident_a", value="ada@x.example")])
    box = Toolbox(context_mod.curate("t", rules), choices=_book(), account=_account())
    for tool in user_tools(box):
        assert tool.name in skills_mod.TOOLS, tool.name
    assert "my_choices" in skills_mod.TOOLS and "my_account" in skills_mod.TOOLS
    assert "my_choices" in skills_mod.RULES
    assert "my_account" in skills_mod.EXAMPLES


def test_a_turn_records_the_tools_it_called_as_names_and_argument_keys():
    user = _agent(["Option_two, please."])
    user.box.my_choices("kind_b")
    user.box.my_account("col_x")
    user.reply([{"role": "assistant", "content": "Which kind_b would you like?"}])
    assert user.events[-1].payload["user_tools"] == [
        {"tool": "my_choices", "args": ["kind"]},
        {"tool": "my_account", "args": ["what"]},
    ]


def test_a_turn_that_fell_to_the_floor_still_records_what_was_called():
    user = _agent(["Hello, my col x is REF-7731."])
    user.box.my_account()
    user.reply([{"role": "assistant", "content": "How can I help you today?"}])
    payload = user.events[-1].payload
    assert payload["agent_turn_dropped"] == guards_mod.ACCOUNT_UNASKED
    assert payload["user_tools"] == [{"tool": "my_account", "args": []}]


def test_a_facts_call_logs_asked_only_when_fields_were_passed():
    user = _agent(["Thanks, noted."])
    user.box.facts([])
    user.box.facts(["ident_a"])
    user.reply([{"role": "assistant", "content": "How can I help you today?"}])
    assert user.events[-1].payload["user_tools"] == [
        {"tool": "my_facts", "args": []},
        {"tool": "my_facts", "args": ["asked"]},
    ]


def test_a_turn_records_what_the_model_weighed_before_it_wrote():
    user = _agent([ModelReply(content="Thanks, noted.",
                              thinking="nothing was asked for, so no read fits")])
    user.reply([{"role": "assistant", "content": "How can I help you today?"}])
    assert user.events[-1].payload["user_thinking"] == "nothing was asked for, so no read fits"


def test_a_turn_with_no_reasoning_summary_records_an_empty_one():
    user = _agent(["Thanks, noted."])
    user.reply([{"role": "assistant", "content": "How can I help you today?"}])
    assert user.events[-1].payload["user_thinking"] == ""


def test_a_turn_that_fell_to_the_floor_keeps_what_the_model_weighed():
    user = _agent([ModelReply(content="Hello, my col x is REF-7731.",
                              thinking="I answered from my head instead of reading")])
    user.box.my_account()
    user.reply([{"role": "assistant", "content": "How can I help you today?"}])
    payload = user.events[-1].payload
    assert payload["agent_turn_dropped"] == guards_mod.ACCOUNT_UNASKED
    assert payload["user_thinking"] == "I answered from my head instead of reading"
