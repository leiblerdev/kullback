"""The Examiner's scenario review as a role on the shared harness (D373, D382), on scripted models only."""

from __future__ import annotations

import asyncio
import json

import pytest

from kullback.agent.extensions import load_extensions
from kullback.agent.harness import AgentHarness
from kullback.ai.provider import ModelReply, TestModel
from kullback.ai.usage import Usage
from kullback.runner import budget
from kullback.runner.records import read_json
from kullback.synth import generate as M
from kullback.synth import prompt as P
from kullback.synth import records as R
from kullback.synth import review as V
from kullback.synth import search as S
from kullback.synth import world_tools as W
from tests.synth.fixture import BOUND, POLICY, REFUSE, STATE, TOOLS, URL, call, make_workdir, web

LINE = POLICY.splitlines()[0]
SPAN = "Customers often ask to move a widget after it has been sealed."
CHECKED = ["tool:move_widget", "row:widget_rows:W1001"]


def _root(tmp_path, *statements, requires=()):
    workdir = make_workdir(tmp_path)
    records = [R.born(text, R.Source(kind="recordings", task_ids=["task-a1"]), frequency=3).model_copy(
        update={"binding": R.listed(BOUND), "requires": list(requires)}) for text in statements]
    R.write_all(workdir, records)
    root = V.ReviewRoot(world=M.world_of(workdir), reader=S.Crawled(web(workdir)[0], workdir), refuse=REFUSE)
    return root, records


def _tool(root, name):
    return next(tool for tool in V.scenario_tools(root) if tool.name == name)


def _web(root, name, arguments):
    """One call of a shared web tool as the review session registers it."""
    harness = AgentHarness(model=TestModel([]))
    load_extensions(harness, [V.scenario_extension(root)])
    return asyncio.run(harness.registry.get(name).run(arguments))


def _rule(root, record, **verdict):
    args = {"id": record.id, "occurs": "yes", "occurs_url": URL, "occurs_span": SPAN, "feasible": "yes",
            "checked": CHECKED, "text": "it happens and the world serves it", **verdict}
    tool = _tool(root, "rule_scenario")
    return asyncio.run(tool.execute(tool.args_model(**args)))


def _read(root):
    _web(root, "read_page", {"url": URL})


def _fix(root, record, quote, why="names a value the rows do not hold", change="say the slot the row holds"):
    tool = _tool(root, "fix_scenario")
    return asyncio.run(tool.execute(V.FixArgs(id=record.id, quote=quote, why=why, change=change)))


def _rule_reply(record, call_id="r1"):
    return call("rule_scenario", {"id": record.id, "occurs": "yes", "occurs_url": URL, "occurs_span": SPAN,
                                  "feasible": "yes", "checked": CHECKED, "text": "common and served"}, call_id)


def test_the_session_holds_the_review_tools_and_reads_only_no_tool_writes_a_file(tmp_path):
    root, _ = _root(tmp_path, "a holder wants a widget put in another slot")
    harness = AgentHarness(model=TestModel([]))
    load_extensions(harness, [V.scenario_extension(root)])
    names = set(harness.registry.names())
    assert set(V.TOOL_NAMES) <= names
    assert not names & {"write", "edit", "bash"}


def test_the_view_hides_where_a_scenario_came_from_and_how_often_and_shows_its_binding_and_requires(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    R.write_scenario(root.workdir, record.model_copy(update={
        "situation": "si_0123456789ab", "binding": {"widget_rows": ["W1001"]}, "requires": ["a tool that seals"]}))
    paths = V.write_views(root, [record.id])
    body = json.loads((root.dir / paths[record.id]).read_text())
    assert not set(body) & {"source", "also_from", "frequency", "situation"}
    assert body["binding"] == {"widget_rows": ["W1001"]} and body["requires"] == ["a tool that seals"]
    assert body["statement"] == record.statement
    assert "recordings" not in json.dumps(body) and "task-a1" not in json.dumps(body)


def test_a_verdict_of_occurs_yes_without_a_cite_or_feasible_no_without_missing_is_refused(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    _read(root)
    with pytest.raises(ValueError, match="occurs yes needs"):
        _rule(root, record, occurs_url="", occurs_span="")
    with pytest.raises(ValueError, match="missing"):
        _rule(root, record, feasible="no")


def test_a_cite_is_a_page_read_in_this_session_and_a_span_of_its_text_and_its_evidence_sits_on_that_scenario(tmp_path):
    root, [record, other] = _root(tmp_path, "a holder wants to move a sealed widget", "a holder wants things")
    _web(root, "search", {"query": "sealed widget"})
    with pytest.raises(ValueError, match="search result alone is no cite"):
        _rule(root, record)
    with pytest.raises(ValueError, match="policy line is no cite"):
        _rule(root, record, occurs_url="", occurs_span=LINE)
    _read(root)
    with pytest.raises(ValueError, match="not on the page"):
        _rule(root, record, occurs_span="Widgets are painted on request.")
    with pytest.raises(ValueError, match="not on the page"):
        _rule(root, record, occurs_span=LINE)
    assert _rule(root, record).rows[0]["occurs_url"] == URL
    [row] = R.load_scenario(root.workdir, record.id).evidence
    assert row == {"tool": "read_page", "url": URL, "span": SPAN, "start": row["start"], "end": row["end"]}
    assert row["end"] - row["start"] == len(SPAN.encode("utf-8"))
    assert R.load_scenario(root.workdir, other.id).evidence == []


def test_a_page_read_in_an_earlier_session_is_no_cite_in_the_next(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants to move a sealed widget")
    _read(root)
    root.read = {}
    with pytest.raises(ValueError, match="not read with read_page in this session"):
        _rule(root, record)


@pytest.mark.parametrize(("verdict", "status"), [
    ({}, "passed"),
    ({"occurs": "no", "occurs_url": "", "occurs_span": ""}, "struck"),
    ({"feasible": "unsure", "checked": []}, "unsure"),
    ({"feasible": "no", "checked": [], "missing": "a tool that paints a widget"}, "infeasible"),
])
def test_the_outcome_is_read_off_the_verdict_by_code(tmp_path, verdict, status):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    _read(root)
    _rule(root, record, **verdict)
    settled = V.settle(root, R.load_scenario(root.workdir, record.id))
    assert settled.status == status
    assert (status == "unsure") == (R.store_dir(root.workdir) / R.HUMAN).is_file()


def test_feasible_yes_needs_checked_entries_code_finds_and_no_open_requires(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    _, [needing] = _root(tmp_path / "b", "a holder wants a widget sealed", requires=["a tool that seals a widget"])
    R.write_scenario(root.workdir, needing)
    _read(root)
    with pytest.raises(ValueError, match="needs checked"):
        _rule(root, record, checked=[])
    with pytest.raises(ValueError, match="tool:paint_widget"):
        _rule(root, record, checked=["tool:paint_widget"])
    with pytest.raises(ValueError, match="row:widget_rows:W9999"):
        _rule(root, record, checked=["tool:move_widget", "row:widget_rows:W9999"])
    with pytest.raises(ValueError, match="requires what the world lacks"):
        _rule(root, needing)
    assert _rule(root, record).rows[0]["checked"] == CHECKED


def test_a_checked_row_outside_the_binding_that_the_world_holds_passes(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    _read(root)
    checked = ["tool:move_widget", "row:widget_rows:W1001", "row:widget_rows:W1002"]
    assert _rule(root, record, checked=checked).rows[0]["checked"] == checked


def test_a_checked_row_the_world_lacks_is_refused_naming_it(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    _read(root)
    with pytest.raises(ValueError, match="does not hold: row:holder_rows:holder_9;"):
        _rule(root, record, checked=["tool:move_widget", "row:widget_rows:W1002", "row:holder_rows:holder_9"])


def test_a_repeat_ruled_by_the_review_ends_struck_naming_the_scenario_it_repeats(tmp_path):
    root, [first, second] = _root(tmp_path, "a holder wants a widget moved", "a holder asks to move a widget")
    _rule(root, second, occurs="unsure", feasible="unsure", occurs_url="", occurs_span="", repeat_of=first.id)
    settled = V.settle(root, R.load_scenario(root.workdir, second.id))
    assert settled.status == "struck" and settled.review[-1]["repeat_of"] == first.id
    assert not (R.store_dir(root.workdir) / R.HUMAN).exists()
    with pytest.raises(ValueError, match="itself"):
        _rule(root, first, repeat_of=first.id)


def test_a_scenario_this_world_cannot_serve_files_one_finding_carrying_its_requires_and_binding(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget painted", requires=["a tool that paints a widget"])
    _read(root)
    _rule(root, record, feasible="no", checked=[], missing="a tool that paints a widget")
    V.settle(root, R.load_scenario(root.workdir, record.id))
    [finding] = read_json(root.workdir / "examiner" / "findings.json")
    assert (finding["kind"], finding["change"]) == ("environment", "add a tool that paints a widget")
    [order] = finding["edits"]
    assert order["requires"] == ["a tool that paints a widget"] and order["binding"] == R.listed(BOUND)
    assert R.load_scenario(root.workdir, record.id).status == "infeasible"


def test_a_send_back_quotes_words_of_the_statement_and_a_second_one_ends_it_unsure_on_the_human_queue(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants things moved")
    with pytest.raises(ValueError, match="quote is not in the scenario"):
        _fix(root, record, "wants stuff")
    _fix(root, record, "things")
    held = R.load_scenario(root.workdir, record.id)
    assert held.status == "fixing" and held.review[-1] == {
        "kind": "fix_requested", "quote": "things", "why": "names a value the rows do not hold",
        "change": "say the slot the row holds"}
    R.write_scenario(root.workdir, held.model_copy(update={"status": "born", "fixes": 1}))
    _fix(root, record, "moved", why="still too broad")
    assert R.load_scenario(root.workdir, record.id).status == "unsure"
    [row] = read_json(R.store_dir(root.workdir) / R.HUMAN)
    assert row["id"] == record.id and "still too broad" in row["question"]


def test_an_end_left_to_a_choice_the_customer_makes_is_sent_back_by_the_ruling_quoting_the_end(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget moved to another slot")
    R.write_scenario(root.workdir, record.model_copy(update={
        "ask": "Please move my widget.", "end": "The widget moves to slot seven, or stays if the holder declines."}))
    _read(root)
    _rule(root, record, end_outcome="choice")
    held = R.load_scenario(root.workdir, record.id)
    assert held.status == "fixing" and held.review[-1]["kind"] == "fix_requested"
    assert held.review[-1]["quote"] == held.end and held.review[-1]["why"] == V.CHOICE_WHY


def test_a_review_built_with_no_reader_injected_and_live_off_has_no_reader(tmp_path):
    root, _ = _root(tmp_path, "a holder wants to move a sealed widget")
    assert S.reader(root.workdir) is None


def test_with_no_reader_the_shared_review_tools_answer_off_and_no_page_becomes_citable(tmp_path):
    from kullback.agent.web_tools import OFF

    root, [record] = _root(tmp_path, "a holder wants to move a sealed widget")
    root.reader = None
    searched = _web(root, "search", {"query": "sealed widget"})
    read = _web(root, "read_page", {"url": URL})
    assert searched.details["summary"] == read.details["summary"] == OFF
    assert root.read == {} and R.load_scenario(root.workdir, record.id).evidence == []
    with pytest.raises(ValueError, match="not read"):
        _rule(root, record)
    assert not (root.workdir / "web_cache").exists()


def test_the_lookups_read_the_starting_state_and_find_tools_reads_the_tool_list(tmp_path):
    root, _ = _root(tmp_path, "a holder wants a widget put in another slot")
    table = next(iter(STATE))
    looked = asyncio.run(_tool(root, "lookup_rows").execute(W.LookupArgs(table=table)))
    assert json.loads(looked.summary)["rows"] == len(STATE[table])
    found = asyncio.run(_tool(root, "find_tools").execute(W.FindArgs(query="move slot")))
    assert found.rows[0]["name"] == TOOLS[0]["name"]


def test_the_prompt_runs_in_gepa_order_says_it_is_blind_and_names_nothing_of_the_environment():
    names = [name for name, _ in P.sections()]
    assert names == ["receives", "tools", "examples", "choice", "feedback", "stop"]
    text = "\n".join(body for _, body in P.sections())
    assert "blind" in P.receives_section()
    fixture_names = ({tool["name"] for tool in TOOLS} | set(STATE)
                     | {key for rows in STATE.values() for row in rows.values() for key in row})
    assert not {name for name in fixture_names if name in text}
    assert all(len(line) < 150 for line in text.splitlines())


def test_a_review_session_plays_on_the_shared_bus_and_books_its_calls_under_its_stage(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    model = TestModel([call("read_page", {"url": URL}, "r0"), _rule_reply(record),
                       ModelReply(content="Ruled one: passed.", usage=Usage(input=10, output=2))])
    out = V.review(root.workdir, model=model, engine=web(root.workdir)[0], refuse=REFUSE)
    assert R.load_scenario(root.workdir, record.id).status == "passed" and out["capped"] == []
    bus = (root.workdir / "bus.jsonl").read_text()
    assert "scenario_verdict" in bus
    assert budget.load_totals(root.workdir)["stages"][V.STAGE]["calls"] >= 1


def test_a_spent_ceiling_reviews_nothing_and_the_review_says_it_stopped_on_the_ceiling(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    ceiling = budget.Ceiling(usd=0.5)
    ceiling.spent = 0.6
    model = TestModel([_rule_reply(record)])
    out = V.review(root.workdir, model=model, engine=web(root.workdir)[0], refuse=REFUSE, ceiling=ceiling)
    assert out["stopped_on_ceiling"] is True and model.calls == []
    assert R.load_scenario(root.workdir, record.id).status == "born"


def test_a_ceiling_with_room_reviews_as_without_one(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    model = TestModel([call("read_page", {"url": URL}, "r0"), _rule_reply(record),
                       ModelReply(content="Ruled one: passed.", usage=Usage(input=10, output=2))])
    out = V.review(root.workdir, model=model, engine=web(root.workdir)[0], refuse=REFUSE,
                   ceiling=budget.Ceiling(usd=0.5))
    assert out["stopped_on_ceiling"] is False
    assert R.load_scenario(root.workdir, record.id).status == "passed"


def test_the_fix_session_reads_the_quote_why_and_change_and_keeps_the_binding(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants things moved")
    _fix(root, record, "things")
    fixer = TestModel([call("rewrite_scenario", {"id": record.id, "statement": "Ada Byron wants her widget moved"},
                            "f0"), ModelReply(content="Rewrote one.")])
    assert V._fix_all(root, fixer) == [record.id]
    sent = json.dumps(fixer.calls[0]["messages"])
    assert 'quote: \\"things\\"' in sent and "why: names a value" in sent and "change: say the slot" in sent
    fixed = R.load_scenario(root.workdir, record.id)
    assert fixed.statement == "Ada Byron wants her widget moved" and fixed.binding == R.listed(BOUND)
