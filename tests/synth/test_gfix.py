"""The generator's prompt and refusal fixes from the observation of the live runs (D400), on scripted data only."""

from __future__ import annotations

import asyncio
import json

import pytest

from kullback.synth import generate_prompt
from kullback.synth import generate_tools as T
from kullback.synth import plan as P
from kullback.synth import records as R
from kullback.synth import world_tools as W
from kullback.synth.rule import ID_SHAPE, World
from tests.synth.fixture import BOUND, STATE, make_workdir
from tests.synth.test_review import _fix, _read, _root, _rule


def _run(tools, name, **args):
    tool = next(tool for tool in tools if tool.name == name)
    return asyncio.run(tool.execute(tool.args_model(**args)))


def test_reuse_across_sessions_is_allowed_and_list_scenarios_marks_the_scenarios_of_this_session(tmp_path):
    workdir = make_workdir(tmp_path)
    earlier = R.born("Ada Byron wants a widget moved", R.Source(kind="recordings", task_ids=["task-a1"]), frequency=1)
    R.write_all(workdir, [earlier])
    now = R.born("Ada Byron wants a sealed widget swapped", R.Source(kind="recordings", task_ids=["task-b2"]),
                 frequency=1)
    root = T.GenerateRoot(world=World(workdir=workdir), kind="recordings", found=[now])
    listed = _run(T.generate_tools(root), "list_scenarios")
    assert listed.summary == "2 scenarios, 1 of this session"
    assert [(row["id"], row["this_session"]) for row in listed.rows] == [(earlier.id, False), (now.id, True)]
    assert "used in earlier sessions may be used again" in generate_prompt.RULE


def test_a_search_says_how_many_keys_match_and_a_missing_key_shows_the_key_shape_and_two_samples(tmp_path):
    rows = {f"R{n:03d}": {"place": "north"} for n in range(13)}
    tools = W.world_tools(World(workdir=tmp_path, state={"spots": rows}))
    found = json.loads(_run(tools, "search_rows", table="spots", field="place", value="north").summary)
    assert found == {"keys": sorted(rows)[:10], "matched": 13, "not_shown": 3}
    missed = _run(tools, "lookup_rows", table="spots", key="R999").summary
    assert missed == "no such row; keys of this table look like A999, e.g. R000, R001"
    assert json.loads(_run(tools, "lookup_rows", table="spots", key="R001").summary) == {"place": "north"}


def test_the_rule_asks_for_requires_only_when_a_search_shows_no_row_fits_and_binds_whoever_holds_the_rows():
    assert "Write requires only when a search shows no row of the world fits" in generate_prompt.RULE
    assert "whichever customer holds them" in generate_prompt.RULE
    assert "What the world lacks for this situation goes into requires" not in generate_prompt.RULE


def test_the_id_rule_spares_a_value_of_a_bound_row_and_a_refusal_names_the_word_that_matched(tmp_path):
    state = {"orders": {"O1": {"item": "earbuds", "options": {"rating": "IPX4"}, "paid": 16005.0}}, **STATE}
    root = T.GenerateRoot(world=World(workdir=tmp_path, state=state), kind="recordings")
    text = "Ada Byron paid 16005 for earbuds rated IPX4"
    assert T._statement_why(root, text, {"orders": "O1"}) == ("", "")
    assert T._statement_why(root, text, {}) == (ID_SHAPE, '"16005" matched')
    assert T._statement_why(root, "Ada Byron wants widget W1001 moved", BOUND) == (ID_SHAPE, '"W1001" matched')


def test_a_kind_sharing_its_stem_with_a_kind_of_the_plan_is_refused_naming_it_in_both_sessions(tmp_path):
    world = World(workdir=tmp_path)
    plan = P.Plan(situations=[{"id": "si_a", "statement": "a holder books a slot", "kind": "book", "openings": [],
                               "variants": []}])
    with pytest.raises(ValueError, match="kind twin.*holds the kind book"):
        P.write_situation(plan, world, [], "a holder wants a second slot held", "booking", [], [])
    plan.situations.append({"id": "si_b", "statement": "a holder pays twice", "kind": "booking", "openings": [],
                            "variants": []})
    with pytest.raises(ValueError, match="kind twin.*holds the kind book"):
        P.write_searched(plan, world, "a holder wants a slot held for a week", "booking", {}, "span", 0.7)
    assert P.stem("cancellation") == P.stem("cancel") and P.kind_twin(plan, "refund") == ""


def test_copy_slips_are_quoted_and_a_send_back_may_quote_the_ask_end_or_requires(tmp_path):
    with pytest.raises(ValueError, match='hedged statement: "or"'):
        P.write_situation(P.Plan(), World(workdir=tmp_path), [], "a holder wants one or two slots", "change", [], [])
    root, [record] = _root(tmp_path / "r", "a holder wants a widget moved", requires=["a tool that paints a widget"])
    _read(root)
    with pytest.raises(ValueError, match='nearest line of the page is "Customers often ask to move a widget'):
        _rule(root, record, occurs_span="Customers often ask to move a widget after sealing.")
    assert _fix(root, record, "paints a widget").summary == f"{record.id} goes back to the generator"
    policy = "Every holder must be verified before any change."
    assert T._policy_refusal(["Every holder is verified before a change."], policy + "\nOther line.")[1] == (
        f'the nearest policy line is "{policy}"')


def test_plan_search_writes_a_situation_only_where_a_tool_acts_or_the_policy_refuses():
    choice = dict(generate_prompt.plan_search_texts(["list_plan", "write_situation"]))["choice"]
    assert "only where the tool list holds an action for it or the policy says to refuse it" in choice
