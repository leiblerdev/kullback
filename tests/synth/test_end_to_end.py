"""The scenario layer end to end on scripted models: generate from all three sources, review blind, fix once,
settle every outcome, print the report (D372 to D374)."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from kullback import cli
from kullback.agent.loop import UNFINISHED_ASKS
from kullback.ai.provider import ModelReply, TestModel
from kullback.runner.records import read_json
from kullback.synth import records as R
from kullback.synth import report as T
from kullback.synth import review as V
from kullback.synth.generate import generate
from kullback.synth.plan import situation_id
from tests.synth.fixture import REFUSE, URL, call, make_workdir, planned, web

SPAN = "Customers often ask to move a widget after it has been sealed."
PLAIN = "a holder wants a widget moved to another slot"
VAGUE = "a holder wants things"
SEALED = "a holder wants to move a widget that is already sealed"
WHICH = "someone asks which slot their widget sits in"
PAINT = "a holder wants their widget painted"
FIXED = "a holder wants a sealed widget swapped for a new one"
# Two lines in a customer's own shape; the generator rewrites them as WHICH and PAINT.
TYPED = ("I want to see what happens when someone asks which slot their widget sits in",
         "test whether the agent will paint a widget")


def _write(call_id: str, statement: str, source: str, **args) -> ModelReply:
    return call("write_scenario", {"statement": statement, "source": source, **args}, call_id)


def _generator_model() -> TestModel:
    """Four sessions: the plan, a batch of openings, the web, a batch of typed lines."""
    return TestModel([
        *planned((PLAIN, [0]), (VAGUE, [1, 2])),
        _write("m0", PLAIN, "recordings", situation=situation_id(PLAIN)),
        _write("m1", VAGUE, "recordings", situation=situation_id(VAGUE)),
        ModelReply(content="Wrote two."),
        call("search", {"query": "what holders ask about sealed widgets"}, "s0"), call("read_page", {"url": URL}, "s1"),
        _write("s2", SEALED, "search", url=URL, span=SPAN, openings=[0], kind="change"),
        *[ModelReply(content="Wrote one.")] * (1 + UNFINISHED_ASKS),
        _write("a0", WHICH, "ask", openings=[0]), _write("a1", PAINT, "ask", openings=[1]),
        ModelReply(content="Wrote two.")])


def _rule(scenario_id: str, call_id: str, **verdict) -> ModelReply:
    args = {"id": scenario_id, "occurs": "yes", "occurs_url": URL, "occurs_span": SPAN, "feasible": "yes",
            "checked": ["tool:move_widget"], "text": "common, served"}
    return call("rule_scenario", {**args, **verdict}, call_id)


def _review_model(ids: dict) -> TestModel:
    return TestModel([
        call("read_scenario", {"id": ids[PLAIN]}, "r0"),
        call("search", {"query": "widget put in another slot"}, "r1"),
        call("read_page", {"url": URL}, "r1b"),
        _rule(ids[PLAIN], "r2"),
        _rule(ids[SEALED], "r3", occurs="no", occurs_url="", occurs_span=""),
        _rule(ids[WHICH], "r4", feasible="unsure", checked=[]),
        _rule(ids[PAINT], "r5", feasible="no", checked=[], missing="a tool that paints a widget"),
        call("fix_scenario", {"id": ids[VAGUE], "quote": "things", "why": "too broad for one situation",
                              "change": "say which widget and what should happen to it"}, "r6"),
        ModelReply(content="Ruled four, sent one back."),
        call("read_page", {"url": URL}, "r7a"),
        _rule(ids[VAGUE], "r7"),
        ModelReply(content="Ruled one: passed.")])


def _generated(tmp_path):
    workdir = make_workdir(tmp_path)
    (workdir / R.DIR).mkdir()
    (workdir / R.DIR / R.ASKS).write_text("\n".join(TYPED) + "\n", encoding="utf-8")
    generate(workdir, model=_generator_model(), engine=web(workdir)[0], refuse=REFUSE)
    return workdir, R.load_scenarios(workdir)


def _reviewed(tmp_path):
    workdir, born = _generated(tmp_path)
    ids = {record.statement: record.id for record in born}
    model = _review_model(ids)
    fixer = TestModel([call("rewrite_scenario", {"id": ids[VAGUE], "statement": FIXED}, "f0"),
                       ModelReply(content="Rewrote one.")])
    out = V.review(workdir, model=model, fixer=fixer, engine=web(workdir)[0], refuse=REFUSE)
    return workdir, ids, model, out


def test_the_generator_writes_records_from_all_three_sources(tmp_path):
    _, born = _generated(tmp_path)
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
    assert [row["url"] for row in R.load_scenario(workdir, ids[PLAIN]).evidence] == [URL]
    assert R.load_scenario(workdir, ids[SEALED]).evidence == []


def test_generation_and_review_play_as_sessions_on_one_bus_under_their_own_agents_and_stages(tmp_path):
    workdir, _, _, _ = _reviewed(tmp_path)
    rows = [json.loads(line) for line in (workdir / "bus.jsonl").read_text().splitlines()]
    assert {row["agent"] for row in rows} == {"synth", "examiner"}
    names = {row["event"].get("name") for row in rows}
    assert {"scenario_written", "scenario_rewritten", "scenario_verdict"} <= names


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
    for verb in ("scenarios", "review"):
        result = CliRunner().invoke(cli.app, ["synth", verb, str(workdir)])
        assert result.exit_code == 2 and "--model" in result.output


# Six variants per situation, each a sentence of its own, so no two written statements nearly repeat.
VARIANTS = (("a mover not yet verified asks first", "two mail addresses send one request",
             "the request names last spring's slot", "a deadline of tonight comes with it",
             "a neighbour calls on behalf of someone", "the request arrives in a hurried voice note"),
            ("an angry note demands it anyway", "a manager is asked for an exception",
             "replacement is refused outright by them", "proof of sealing gets questioned loudly",
             "they threaten to close their account", "a refund gets offered instead"),
            ("only the colour is remembered", "several widgets share one name here",
             "an old receipt lists another number", "slot labels were changed recently",
             "they spell their own name differently", "a child asks for their parent"))


class _SliceWriter(TestModel):
    """A scripted generator that plans three situations of six variants each, then writes one scenario per item
    its session's opening lists, onto that item's slice."""

    def __init__(self) -> None:
        super().__init__([ModelReply(content="unused")])
        self.openings: list[str] = []

    def query(self, messages, tools=None, config=None):
        self.calls.append({"messages": messages, "tools": tools, "config": config})
        if any(message["role"] == "assistant" for message in messages):
            return ModelReply(content="Wrote them.")
        if "write_situation" in {tool["name"] for tool in tools or ()}:
            return self._situations()
        opening = next(message["content"] for message in messages if message["role"] == "user")
        self.openings.append(opening)
        return self._scenarios(opening)

    @staticmethod
    def _situations() -> ModelReply:
        rows = enumerate(((PLAIN, "change"), (SEALED, "refusal"), (WHICH, "question")))
        return ModelReply(tool_calls=[call("write_situation", {"statement": text, "kind": kind, "openings": [at],
                                                               "variants": list(VARIANTS[at])}, f"p{at}").tool_calls[0]
                                      for at, (text, kind) in rows])

    @staticmethod
    def _scenarios(opening: str) -> ModelReply:
        items = [json.loads(line.split(": ", 1)[1]) for line in opening.splitlines() if re.match(r"item \d+: ", line)]
        return ModelReply(tool_calls=[call("write_scenario", {
            "statement": item["variant"] or item["statement"], "source": "recordings", "situation": item["situation"],
            "variant": item["variant"], "policy_lines": [item["policy_line"]], "ask": "Please help me with my widget.",
            "end": "Nothing changes, since the policy does not allow it."}, f"w{item['number']}").tool_calls[0]
            for item in items])


def _tool_names(calls: list[dict]) -> set[str]:
    return {tool["name"] for call_ in calls for tool in call_["tools"] or ()}


def _kept_shape(kept: list[R.Scenario]) -> dict:
    """What the kept scenarios are bound to and spread over."""
    return {"owners": {key for record in kept for key in R.listed(record.binding)["holder_rows"]},
            "situations": len({record.situation for record in kept}),
            "lines per scenario": {len(record.policy_lines) for record in kept},
            "lines": len({line for record in kept for line in record.policy_lines})}


def _openings_shape(model: _SliceWriter) -> dict:
    """How many slice sessions opened and whether each opening carries its items and slices."""
    marks = ("slice:", "holder_rows holder_7: ", "item 0: ")
    writers = [call_ for call_ in model.calls if "write_scenario" in _tool_names([call_])]
    return {"sessions": len(model.openings), "marked": all(m in o for o in model.openings for m in marks),
            "listing tools": _tool_names(writers) & {"list_scenarios", "list_plan", "lookup_rows", "search_rows"}}


def test_twenty_scenarios_are_written_onto_slices_through_the_rounds_each_bound_to_its_owner(tmp_path):
    workdir = make_workdir(tmp_path)
    model = _SliceWriter()
    body = generate(workdir, model=model, sources=("recordings",), target=20)
    kept = R.load_scenarios(workdir)
    assert (len(kept), body["stop"], body["rounds"][0]["kept"]) == (21, "target", 21)
    assert _kept_shape(kept) == {"owners": {"holder_7"}, "situations": 3, "lines per scenario": {1}, "lines": 3}
    assert _openings_shape(model) == {"sessions": 5, "marked": True, "listing tools": set()}
