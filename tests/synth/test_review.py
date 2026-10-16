"""The Examiner's scenario review as a role on the shared harness (D373), on scripted models only."""

from __future__ import annotations

import asyncio
import json

import pytest

from kullback.agent.extensions import load_extensions
from kullback.agent.harness import AgentHarness
from kullback.ai.provider import TestModel
from kullback.runner.records import read_json
from kullback.synth import miner as M
from kullback.synth import prompt as P
from kullback.synth import records as R
from kullback.synth import review as V
from kullback.synth import search as S
from tests.synth.fixture import POLICY, STATE, TOOLS, URL, call, make_workdir, web

LINE = POLICY.splitlines()[0]


def _root(tmp_path, *statements):
    workdir = make_workdir(tmp_path)
    records = [R.born(text, R.Source(kind="recordings", task_ids=["task-a1"]), frequency=3) for text in statements]
    R.write_all(workdir, records)
    root = V.ReviewRoot(world=M.world_of(workdir), engine=web(workdir)[0])
    return root, records


def _tool(root, name):
    return next(tool for tool in V.scenario_tools(root) if tool.name == name)


def _rule(root, record, **verdict):
    args = {"id": record.id, "occurs": "yes", "occurs_cite": LINE, "feasible": "yes",
            "text": "it happens and the world serves it", **verdict}
    tool = _tool(root, "rule_scenario")
    return asyncio.run(tool.execute(tool.args_model(**args)))


def test_the_session_holds_the_review_tools_and_reads_only_no_tool_writes_a_file(tmp_path):
    root, _ = _root(tmp_path, "a holder wants a widget put in another slot")
    harness = AgentHarness(model=TestModel([]))
    load_extensions(harness, [V.scenario_extension(root)])
    names = set(harness.registry.names())
    assert set(V.TOOL_NAMES) <= names
    assert not names & {"write", "edit", "bash"}


def test_the_view_hides_where_a_scenario_came_from_and_how_often(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    paths = V.write_views(root, [record.id])
    body = json.loads((root.dir / paths[record.id]).read_text())
    assert not set(body) & {"source", "also_from", "frequency"}
    assert body["statement"] == record.statement
    assert "recordings" not in json.dumps(body) and "task-a1" not in json.dumps(body)


def test_a_verdict_of_occurs_yes_without_a_cite_or_feasible_no_without_missing_is_refused(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    with pytest.raises(ValueError, match="occurs_cite"):
        _rule(root, record, occurs_cite="")
    with pytest.raises(ValueError, match="missing"):
        _rule(root, record, feasible="no")


def test_a_cite_is_a_url_this_scenarios_own_search_returned_or_a_policy_line_and_nothing_else(tmp_path):
    root, [record, other] = _root(tmp_path, "a holder wants to move a sealed widget", "a holder wants things")
    with pytest.raises(ValueError, match="neither"):
        _rule(root, record, occurs_cite=URL)
    with pytest.raises(ValueError, match="neither"):
        _rule(root, record, occurs_cite="task-a1")
    asyncio.run(_tool(root, "search").execute(V.SearchArgs(id=record.id, query="sealed widget")))
    assert _rule(root, record, occurs_cite=URL).rows[0]["occurs_cite"] == URL
    assert _rule(root, other, occurs_cite=LINE).rows[0]["occurs_cite"] == LINE
    with pytest.raises(ValueError, match="neither"):
        _rule(root, other, occurs_cite=URL)


@pytest.mark.parametrize(("verdict", "status"), [
    ({}, "passed"),
    ({"occurs": "no", "occurs_cite": ""}, "struck"),
    ({"feasible": "unsure"}, "unsure"),
    ({"feasible": "no", "missing": "a tool that paints a widget"}, "infeasible"),
])
def test_the_outcome_is_read_off_the_verdict_by_code(tmp_path, verdict, status):
    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    _rule(root, record, **verdict)
    settled = V.settle(root, R.load_scenario(root.workdir, record.id))
    assert settled.status == status
    assert (status == "unsure") == (R.store_dir(root.workdir) / R.HUMAN).is_file()


def test_a_repeat_ruled_by_the_review_ends_struck_naming_the_scenario_it_repeats(tmp_path):
    root, [first, second] = _root(tmp_path, "a holder wants a widget moved", "a holder asks to move a widget")
    _rule(root, second, occurs="unsure", feasible="unsure", occurs_cite="", repeat_of=first.id)
    settled = V.settle(root, R.load_scenario(root.workdir, second.id))
    assert settled.status == "struck" and settled.review[-1]["occurs_cite"] == first.id
    assert not (R.store_dir(root.workdir) / R.HUMAN).exists()
    with pytest.raises(ValueError, match="itself"):
        _rule(root, first, repeat_of=first.id)


def test_a_scenario_this_world_cannot_serve_files_one_environment_finding_and_is_never_dropped(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants a widget painted")
    _rule(root, record, feasible="no", missing="a tool that paints a widget")
    V.settle(root, R.load_scenario(root.workdir, record.id))
    findings = read_json(root.workdir / "examiner" / "findings.json")
    assert [(f["kind"], f["change"]) for f in findings] == [("environment", "add a tool that paints a widget")]
    assert R.load_scenario(root.workdir, record.id).status == "infeasible"


def test_a_second_send_back_ends_the_scenario_unsure_on_the_human_queue(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants things")
    tool = _tool(root, "fix_scenario")
    asyncio.run(tool.execute(V.FixArgs(id=record.id, why="narrow it to one situation")))
    assert R.load_scenario(root.workdir, record.id).status == "fixing"
    R.write_scenario(root.workdir, R.load_scenario(root.workdir, record.id).model_copy(
        update={"status": "born", "fixes": 1}))
    asyncio.run(tool.execute(V.FixArgs(id=record.id, why="still too broad")))
    assert R.load_scenario(root.workdir, record.id).status == "unsure"
    [row] = read_json(R.store_dir(root.workdir) / R.HUMAN)
    assert row["id"] == record.id and "still too broad" in row["question"]


def test_a_search_is_kept_on_the_scenario_as_evidence_with_the_lines_it_read(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants to move a sealed widget")
    tool = _tool(root, "search")
    result = asyncio.run(tool.execute(V.SearchArgs(id=record.id, query="move widget after sealed")))
    assert result.rows[0]["url"] == URL and result.rows[0]["lines"]
    [evidence] = R.load_scenario(root.workdir, record.id).evidence
    assert evidence["urls"] == [URL] and evidence["query"] == "move widget after sealed"


def test_a_review_built_with_no_reader_injected_and_live_off_has_no_reader(tmp_path):
    root, _ = _root(tmp_path, "a holder wants to move a sealed widget")
    assert S.reader(root.workdir) is None


def test_with_nothing_live_the_review_search_reads_nothing_and_its_evidence_names_the_switch(tmp_path):
    root, [record] = _root(tmp_path, "a holder wants to move a sealed widget")
    root.engine = None
    result = asyncio.run(_tool(root, "search").execute(V.SearchArgs(id=record.id, query="sealed widget")))
    assert result.rows == [] and result.summary == S.OFF_NOTE
    assert R.load_scenario(root.workdir, record.id).evidence[0]["note"] == S.OFF_NOTE


def test_the_lookups_read_the_starting_state_and_find_tools_reads_the_tool_list(tmp_path):
    root, _ = _root(tmp_path, "a holder wants a widget put in another slot")
    table = next(iter(STATE))
    looked = asyncio.run(_tool(root, "lookup_rows").execute(V.LookupArgs(table=table)))
    assert json.loads(looked.summary)["rows"] == len(STATE[table])
    found = asyncio.run(_tool(root, "find_tools").execute(V.FindArgs(query="move slot")))
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
    from kullback.ai.provider import ModelReply
    from kullback.ai.usage import Usage
    from kullback.runner import budget

    root, [record] = _root(tmp_path, "a holder wants a widget put in another slot")
    model = TestModel([call("rule_scenario", {"id": record.id, "occurs": "yes", "occurs_cite": LINE,
                                              "feasible": "yes", "text": "common and served"}),
                       ModelReply(content="Ruled one: passed.", usage=Usage(input=10, output=2))])
    out = V.review(root.workdir, model=model, engine=root.engine)
    assert R.load_scenario(root.workdir, record.id).status == "passed" and out["capped"] == []
    bus = (root.workdir / "bus.jsonl").read_text()
    assert "scenario_verdict" in bus
    assert budget.load_totals(root.workdir)["stages"][V.STAGE]["calls"] >= 1
