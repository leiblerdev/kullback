"""The scenario miner and the search reader (D372, D374), on scripted models only."""

from __future__ import annotations

import json
from collections import Counter

from kullback import domain
from kullback.ai.provider import TestModel
from kullback.synth import miner as M
from kullback.synth import records as R
from kullback.synth import search as S
from tests.synth.fixture import DESCRIPTION, OPENINGS, PAGE, make_workdir

KEY = {"KULLBACK_SEARCH_KEY": "secret-value"}


def _reply(body: dict) -> str:
    return json.dumps(body)


def test_every_recording_gives_its_first_user_turn_as_the_opening_ask(tmp_path):
    rows = M.openings(make_workdir(tmp_path))
    assert [(row["task_id"], row["text"]) for row in rows] == list(OPENINGS.items())


def test_a_statement_naming_a_tool_an_id_or_an_identifying_row_value_is_refused(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    assert M.statement_refusal("a holder asks to have a widget put in another slot", world) == ""
    assert M.statement_refusal("a holder wants get_widget run", world) == M.TOOL_NAME
    assert M.statement_refusal("a holder wants widget W1001 put elsewhere", world) == M.ID_SHAPE
    assert M.statement_refusal("Ada Byron wants a widget moved", world) == M.ROW_VALUE
    assert M.statement_refusal("a holder within the 30-day window wants a sealed widget moved", world) == ""


def test_recordings_group_into_statements_with_their_frequency_and_checked_policy_lines(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    model = TestModel([_reply({"scenarios": [
        {"statement": "a holder wants a widget moved to another slot", "openings": [0, 1],
         "policy_lines": ["A widget may be moved to another slot only before it is sealed.", "made up line"]},
        {"statement": "a holder wants a replacement for a widget that arrived broken", "openings": [2]}]})])
    dropped = Counter()
    found = M.from_recordings(world, model, dropped=dropped)
    assert [(r.frequency, r.source.task_ids) for r in found] == [(2, ["task-a1", "task-b2"]), (1, ["task-c3"])]
    assert found[0].policy_lines == ["A widget may be moved to another slot only before it is sealed."]
    assert dropped["policy_lines_dropped"] == 1 and len(model.calls) == 1


def test_a_statement_breaking_the_rule_is_sent_back_once_then_dropped_and_counted(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    model = TestModel([
        _reply({"scenarios": [{"statement": "a holder wants move_widget called", "openings": [0]},
                              {"statement": "a holder wants widget W1001 moved", "openings": [1]}]}),
        _reply({"statements": [{"index": 0, "statement": "a holder wants a widget moved to another slot"},
                               {"index": 1, "statement": "a holder wants widget W1001 moved now"}]})])
    dropped = Counter()
    found = M.from_recordings(world, model, dropped=dropped)
    assert [r.statement for r in found] == ["a holder wants a widget moved to another slot"]
    assert dropped == Counter({M.ID_SHAPE: 1}) and len(model.calls) == 2
    assert "move_widget" not in model.calls[1]["messages"][0]["content"].split("why:")[1]


def test_the_search_reader_posts_the_query_with_the_key_header_and_caches_the_links(tmp_path):
    sent = []

    def post(url, headers, body):
        sent.append((url, headers, body))
        return {"organic": [{"link": "https://help.example/a"}, {"link": "https://help.example/a"},
                            {"link": "https://help.example/b"}]}

    search = S.reader(tmp_path, post=post, env={S.URL_VAR: "https://search.example/q"})
    assert search("move a sealed widget", "k1") == ["https://help.example/a", "https://help.example/b"]
    assert search("move a sealed widget", "k1") == ["https://help.example/a", "https://help.example/b"]
    assert sent == [("https://search.example/q", {S.KEY_HEADER: "k1"}, {"q": "move a sealed widget"})]
    assert "k1" not in S.cache_path(tmp_path, "move a sealed widget").read_text()
    assert S.endpoint({}) == S.DEFAULT_URL


def test_without_a_search_key_no_search_runs_and_the_note_names_the_variable_only(tmp_path):
    found, note = S.run("anything", tmp_path, search=lambda q, k: ["https://x.example"], env={})
    assert found == [] and domain.SEARCH_KEY_VAR in note


def _search_model(span: str) -> TestModel:
    return TestModel([
        _reply({"queries": ["what holders ask about sealed widgets"]}),
        _reply({"scenarios": [
            {"statement": "a holder wants to move a widget that is already sealed", "span": span,
             "policy_lines": ["A sealed widget cannot be moved; offer a replacement instead."]},
            {"statement": "a holder wants their widget painted", "span": "We paint widgets on request."}]})])


def test_a_search_statement_cites_the_byte_range_of_a_span_the_page_holds_and_unverified_ones_drop(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    span = "Customers often ask to move a widget after it has been sealed."
    dropped = Counter()
    found = M.from_search(world, _search_model(span), dropped=dropped, env=KEY,
                          search=lambda q, k: ["https://help.example/a"], fetch=lambda url: PAGE)
    [record] = found
    raw = domain.cached_text(world.workdir, record.source.key).encode("utf-8")
    assert raw[record.source.start:record.source.end].decode("utf-8") == span
    assert record.source.url == "https://help.example/a" and record.frequency == 0
    assert dropped[M.NO_SPAN] == 1


def test_search_without_a_key_reads_no_page_and_says_why(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    notes: list = []
    model = TestModel([_reply({"queries": ["what holders ask"]})])
    assert M.from_search(world, model, env={}, notes=notes, fetch=lambda url: PAGE) == []
    assert any(domain.SEARCH_KEY_VAR in note for note in notes)


def test_ask_lines_become_records_as_typed_and_a_line_naming_a_tool_is_dropped(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    dropped = Counter()
    found = M.from_asks(world, ["test whether the agent refuses to move a sealed widget",
                                "call move_widget twice"], dropped=dropped)
    assert [(r.source.kind, r.source.line) for r in found] == [
        ("ask", "test whether the agent refuses to move a sealed widget")]
    assert dropped == Counter({M.TOOL_NAME: 1})


def test_mining_folds_an_ask_into_the_recordings_record_and_writes_coverage(tmp_path):
    workdir = make_workdir(tmp_path)
    statement = "a holder wants a widget moved to another slot"
    (workdir / R.DIR).mkdir()
    (workdir / R.DIR / R.ASKS).write_text(f"{statement}\nsomeone asks which slot their widget sits in\n")
    model = TestModel([_reply({"scenarios": [{"statement": statement, "openings": [0, 1]}]})])
    body = M.mine(workdir, model=model, searching=False)
    records = R.load_scenarios(workdir)
    kinds = sorted(r.source.kind for r in records)
    assert kinds == ["ask", "recordings"]
    held = next(r for r in records if r.source.kind == "recordings")
    assert [source.kind for source in held.also_from] == ["ask"]
    assert body["coverage"]["asks_not_covered"] == 1 and body["coverage"]["folded"] == {"ask": 1}
    assert "search turned off" in body["notes"]


def test_mining_again_keeps_the_status_and_review_of_a_record_already_held(tmp_path):
    workdir = make_workdir(tmp_path)
    (workdir / R.DIR).mkdir()
    (workdir / R.DIR / R.ASKS).write_text("someone asks which slot their widget sits in\n")
    M.mine(workdir, model=None, searching=False)
    [record] = R.load_scenarios(workdir)
    R.write_scenario(workdir, record.model_copy(update={"status": "passed"}))
    M.mine(workdir, model=None, searching=False)
    assert [r.status for r in R.load_scenarios(workdir)] == ["passed"]


def test_a_fix_rewrites_the_statement_once_from_the_same_source_and_keeps_the_id(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    record = R.born("a holder wants things", R.Source(kind="recordings", task_ids=["task-a1"]), frequency=1)
    model = TestModel([_reply({"statement": "a holder wants a widget moved to another slot"})])
    fixed = M.fix(world, record, "too vague: say what the holder wants", model)
    assert fixed.id == record.id and fixed.fixes == 1 and fixed.status == "born"
    assert OPENINGS["task-a1"] in model.calls[0]["messages"][0]["content"]
    bad = M.fix(world, record, "narrow it", TestModel([_reply({"statement": "run get_widget"})]))
    assert bad.status == "unsure" and bad.review[-1]["refused"] == M.TOOL_NAME


def test_the_miner_prompts_hold_no_tool_table_or_field_name_of_the_environment(tmp_path):
    from tests.synth.fixture import STATE, TOOLS

    names = {tool["name"] for tool in TOOLS} | set(STATE) | {f for rows in STATE.values() for row in rows.values()
                                                              for f in row}
    texts = [M.RULE, M.RECORDINGS_PROMPT, M.SEND_BACK_PROMPT, M.QUERY_PROMPT, M.PAGE_PROMPT, M.FIX_PROMPT]
    assert not {name for name in names for text in texts if name in text}
    assert DESCRIPTION not in "".join(texts)
    assert all(len(line) < 150 for text in texts for line in text.splitlines())
