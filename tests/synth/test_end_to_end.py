"""The scenario layer end to end on scripted models: mine from all three sources, review blind, fix once,
settle every outcome, print the report (D372 to D374)."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from kullback import cli
from kullback.ai.provider import ModelReply, TestModel
from kullback.runner.records import read_json
from kullback.synth import records as R
from kullback.synth import report as T
from kullback.synth import review as V
from kullback.synth.miner import mine
from tests.synth.fixture import PAGE, call, make_workdir

KEY = {"KULLBACK_SEARCH_KEY": "k"}
URL = "https://help.example/a"
SPAN = "Customers often ask to move a widget after it has been sealed."
PLAIN = "a holder wants a widget moved to another slot"
VAGUE = "a holder wants things"
SEALED = "a holder wants to move a widget that is already sealed"
WHICH = "someone asks which slot their widget sits in"
PAINT = "a holder wants their widget painted"
FIXED = "a holder wants a sealed widget swapped for a new one"
# Two lines in a customer's own shape; the miner rewrites them as WHICH and PAINT.
TYPED = ("I want to see what happens when someone asks which slot their widget sits in",
         "test whether the agent will paint a widget")


def _miner_model() -> TestModel:
    return TestModel([
        json.dumps({"scenarios": [{"statement": PLAIN, "openings": [0]}, {"statement": VAGUE, "openings": [1, 2]}]}),
        json.dumps({"queries": ["what holders ask about sealed widgets"]}),
        json.dumps({"scenarios": [{"statement": SEALED, "span": SPAN}]}),
        json.dumps({"scenarios": [{"line": 0, "statement": WHICH}, {"line": 1, "statement": PAINT}]})])


def _rule(scenario_id: str, call_id: str, **verdict) -> ModelReply:
    args = {"id": scenario_id, "occurs": "yes", "occurs_cite": URL, "feasible": "yes", "text": "common, served"}
    return call("rule_scenario", {**args, **verdict}, call_id)


def _review_model(ids: dict) -> TestModel:
    return TestModel([
        call("read_scenario", {"id": ids[PLAIN]}, "r0"),
        call("search", {"id": ids[PLAIN], "query": "widget put in another slot"}, "r1"),
        _rule(ids[PLAIN], "r2"),
        _rule(ids[SEALED], "r3", occurs="no", occurs_cite="nothing searched shows it"),
        _rule(ids[WHICH], "r4", feasible="unsure"),
        _rule(ids[PAINT], "r5", feasible="no", missing="a tool that paints a widget"),
        call("fix_scenario", {"id": ids[VAGUE], "why": "too broad: narrow it to one situation"}, "r6"),
        ModelReply(content="Ruled four, sent one back."),
        _rule(ids[VAGUE], "r7"),
        ModelReply(content="Ruled one: passed.")])


def _mined(tmp_path):
    workdir = make_workdir(tmp_path)
    (workdir / R.DIR).mkdir()
    (workdir / R.DIR / R.ASKS).write_text("\n".join(TYPED) + "\n", encoding="utf-8")
    mine(workdir, model=_miner_model(), env=KEY, search=lambda q, k: [URL], fetch=lambda url: PAGE)
    return workdir, R.load_scenarios(workdir)


def _reviewed(tmp_path):
    workdir, born = _mined(tmp_path)
    ids = {record.statement: record.id for record in born}
    model = _review_model(ids)
    out = V.review(workdir, model=model, fixer=TestModel([json.dumps({"statement": FIXED})]), env=KEY,
                   search=lambda q, k: [URL], fetch=lambda url: PAGE)
    return workdir, ids, model, out


def test_the_miner_writes_records_from_all_three_sources(tmp_path):
    _, born = _mined(tmp_path)
    assert sorted(record.source.kind for record in born) == ["ask", "ask", "recordings", "recordings", "search"]


def test_every_outcome_settles_from_the_review_and_the_fix_loop(tmp_path):
    workdir, ids, _, out = _reviewed(tmp_path)
    status = {record.id: record.status for record in R.load_scenarios(workdir)}
    assert status == {ids[PLAIN]: "passed", ids[SEALED]: "struck", ids[WHICH]: "unsure",
                      ids[PAINT]: "infeasible", ids[VAGUE]: "passed"}
    fixed = R.load_scenario(workdir, ids[VAGUE])
    assert fixed.statement == FIXED and fixed.fixes == 1
    [finding] = read_json(workdir / "examiner" / "findings.json")
    assert finding["key"] == f"environment:scenario:{ids[PAINT]}" and out["findings"]
    assert [row["id"] for row in read_json(R.store_dir(workdir) / R.HUMAN)] == [ids[WHICH]]
    assert R.load_scenario(workdir, ids[PLAIN]).evidence[0]["urls"] == [URL]


def test_the_review_never_sees_a_source_kind_a_task_id_or_a_frequency(tmp_path):
    workdir, _, model, _ = _reviewed(tmp_path)
    views = sorted((R.store_dir(workdir) / V.ROOT / V.VIEWS).glob("*.json"))
    assert views and all(not set(json.loads(path.read_text())) & set(V.HIDDEN) for path in views)
    sent = json.dumps([message for call_ in model.calls for message in call_["messages"]])
    kinds = "|".join(R.SOURCE_KINDS)
    assert not re.search(rf'kind\W+({kinds})\W', sent) and "task-a1" not in sent and "frequency" not in sent
    assert "test whether" not in sent.lower() and "i want to see" not in sent.lower()


def test_the_report_prints_every_count_under_the_scenarios_heading(tmp_path):
    workdir, _, _, _ = _reviewed(tmp_path)
    printed = T.lines(workdir)
    assert printed[0] == T.HEADING
    assert "| recordings | 0 | 0 | 2 | 0 | 0 | 0 |" in printed
    assert "| search | 0 | 0 | 0 | 0 | 0 | 1 |" in printed
    assert "| ask | 0 | 0 | 0 | 1 | 1 | 0 |" in printed
    assert "findings filed: 1" in printed and "human queue: 1" in printed


def test_both_scenario_verbs_refuse_without_a_model_and_name_the_option(tmp_path):
    workdir = make_workdir(tmp_path)
    for verb in ("mine", "review"):
        result = CliRunner().invoke(cli.app, ["synth", verb, str(workdir)])
        assert result.exit_code == 2 and "--model" in result.output
