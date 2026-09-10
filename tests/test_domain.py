"""Reading an invented domain's public material into task archetypes and gaps (D225).

The domain is a lantern rental shop nobody runs. Its help centre is three HTML files on disk, served
by a reader function the crawl is handed, so nothing here opens a socket: what is under test is what
the harness does with pages, and where a page came from is the caller's business.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback import domain, graph
from kullback.runner.records import Column, EntitySchema, FieldStat, ToolSig, write_json

HOST = "https://help.lantern-rental.invalid"
OTHER = "https://payments.elsewhere.invalid"

INDEX = f"""<html><body>
<h1>Lantern rental help</h1>
<p>Everything about hiring a lantern, keeping it longer and giving it back.</p>
<a href="{HOST}/keep-it-longer">Keeping a lantern longer</a>
<a href="{HOST}/broken-wick">A wick that will not light</a>
<a href="{OTHER}/fees">How our payment partner charges</a>
</body></html>"""

LONGER = f"""<html><body>
<h1>Keeping a lantern longer</h1>
<p>A hire runs for seven nights. You can extend a hire that has not ended yet.</p>
<p>Extending a hire that has already ended is not possible and a new hire is needed.</p>
<a href="{HOST}/deep-link">Deeper still</a>
</body></html>"""

WICK = """<html><body>
<h1>A wick that will not light</h1>
<p>A lantern whose wick will not light can be swapped at any depot on the same day.</p>
</body></html>"""

DEEP = """<html><body><h1>Deeper still</h1><p>Nothing anyone needs is written here.</p></body></html>"""

PAGES = {f"{HOST}/": INDEX, f"{HOST}/keep-it-longer": LONGER, f"{HOST}/broken-wick": WICK,
         f"{HOST}/deep-link": DEEP, f"{OTHER}/fees": "<html><body><p>Fees.</p></body></html>"}


def fetch(url: str) -> str:
    """The invented help centre, served off the dictionary above. No socket is opened in any test."""
    if str(url) not in PAGES:
        raise FileNotFoundError(url)
    return PAGES[str(url)]


class Scripted:
    """A model handed the JSON it should answer, keyed by a phrase of the prompt it is given.

    Not a mock of the module's own logic: the module's whole contract with a model is a prompt in and
    a string out, and this is the string, so every rule under test still runs in full.
    """

    def __init__(self, replies: dict, default: str = "{}") -> None:
        self.replies, self.default, self.prompts = replies, default, []

    def query(self, messages, tools=None, config=None):
        prompt = str(messages[-1]["content"])
        self.prompts.append(prompt)
        for phrase, reply in self.replies.items():
            if phrase in prompt:
                return type("Reply", (), {"content": reply, "tool_calls": []})()
        return type("Reply", (), {"content": self.default, "tool_calls": []})()


def _archetype(goal: str, effects: list[str], **extra) -> dict:
    return {"goal": goal, "preconditions": extra.get("preconditions", ["my hire has not ended"]),
            "effects": effects, "constraints": extra.get("constraints", [])}


READER = json.dumps({"archetypes": [_archetype("I want to keep my lantern for a few more nights",
                                               ["my hire runs longer than it did"])]})


# --- the world the archetypes are mapped onto -------------------------------------------

BODIES = {
    "find_hire": "for hire in self.db.hires.values():\n"
                 "    if hire.holder == holder:\n"
                 "        return {'hire_id': hire.hire_id}\n"
                 "raise ValueError('no hire for that holder')",
    "get_hire": "hire = self.db.hires[hire_id]\n"
                "return {'hire_id': hire.hire_id, 'holder': hire.holder, 'nights': hire.nights}",
    "extend_hire": "hire = self.db.hires[hire_id]\n"
                   "if hire.nights == nights:\n"
                   "    raise ValueError('the hire already runs that long')\n"
                   "hire.nights = nights\n"
                   "return {'hire_id': hire.hire_id, 'nights': hire.nights}",
    "scrap_depot": "depot = self.db.depots[depot_id]\n"
                   "depot.open = 'no'\n"
                   "return {'depot_id': depot.depot_id}",
}


def _sig(name: str, kind: str, *args: str) -> ToolSig:
    return ToolSig(name=name, kind=kind,
                   args_fields=[FieldStat(name=arg, types=["str"], optional=False) for arg in args],
                   args_schema={"type": "object",
                                "properties": {arg: {"type": ["str"]} for arg in args},
                                "required": list(args)})


SIGS = [_sig("find_hire", "read", "holder"), _sig("get_hire", "read", "hire_id"),
        _sig("extend_hire", "write", "hire_id", "nights"), _sig("scrap_depot", "write", "depot_id")]

DB = {"hires": {f"h{n}": {"hire_id": f"h{n}", "holder": f"holder{n}", "nights": f"{n + 1}"}
                for n in range(1, 6)},
      "depots": {f"d{n}": {"depot_id": f"d{n}", "open": "yes"} for n in range(1, 4)}}


def _recording(number: int) -> list[dict]:
    hire = f"h{number}"
    return [{"name": "find_hire", "args": {"holder": f"holder{number}"},
             "result": {"hire_id": hire}, "error": None},
            {"name": "get_hire", "args": {"hire_id": hire},
             "result": {"hire_id": hire, "holder": f"holder{number}", "nights": f"{number + 1}"},
             "error": None},
            {"name": "extend_hire", "args": {"hire_id": hire, "nights": "9"},
             "result": {"hire_id": hire, "nights": "9"}, "error": None}]


UNREACHED = [{"name": "find_hire", "args": {"holder": "holder1"},
              "result": {"hire_id": "h1"}, "error": None},
             {"name": "scrap_depot", "args": {"depot_id": "d3"},
              "result": {"depot_id": "d3"}, "error": None}]
"""One recorded Run that ends in a write nothing led to: no answer of the read carried the depot and
no row of the read is the row the write touched, so the graph holds `scrap_depot` as a node with no
edge into it. A mapping that names it is a mapping no walk can realise."""


@pytest.fixture
def shop(tmp_path: Path) -> Path:
    """A finished workdir of the lantern world, with the graph mined off six recorded Runs."""
    workdir = tmp_path / "work"
    columns = [Column(table=table, name=name, class_="hard")
               for table, row in (("hires", DB["hires"]["h1"]), ("depots", DB["depots"]["d1"]))
               for name in row]
    schema = EntitySchema(tables=["hires", "depots"], columns=columns)
    write_json(workdir / "schema.json", json.loads(schema.model_dump_json()))
    write_json(workdir / "tool_sigs.json", [json.loads(sig.model_dump_json()) for sig in SIGS])
    write_json(workdir / "bodies.json", BODIES)
    write_json(workdir / "db.json", DB)
    write_json(workdir / "environment.json", {"env_id": "lanterns-1"})
    write_json(workdir / "canon-rules.json", {})
    write_json(workdir / "readers.json", [])
    write_json(workdir / "user_facts.json",
               {"facts": [{"run_id": "r1", "field": "holder", "value": "holder1"}]})
    write_json(workdir / graph.FILE_NAME,
               graph.mine([_recording(n) for n in range(1, 6)] + [UNREACHED],
                          write_tools=["extend_hire", "scrap_depot"]))
    return workdir


# --- the crawl ---------------------------------------------------------------------------

def test_pages_on_the_same_host_are_followed_to_the_depth_and_no_further(shop: Path):
    read = domain.crawl(shop, [{"url": f"{HOST}/", "depth": 0}], fetch=fetch, depth=1)
    urls = {row["url"] for row in read["pages"]}
    assert f"{HOST}/keep-it-longer" in urls, "a link on the same host was not followed"
    assert f"{HOST}/deep-link" not in urls, "a link two hops away was followed past the depth"
    assert f"{OTHER}/fees" not in urls, "a link off the source's own host was followed"


def test_a_source_under_the_excluded_corpus_url_is_refused_and_counted(shop: Path):
    corpus = f"{HOST}/benchmark"
    read = domain.crawl(shop, [{"url": f"{corpus}/tasks", "depth": 0}, {"url": f"{HOST}/broken-wick",
                                                                       "depth": 0}],
                        fetch=fetch, depth=0, corpus_url=corpus)
    assert [row["url"] for row in read["pages"]] == [f"{HOST}/broken-wick"]
    assert [row["reason"] for row in read["refused"]] == [domain.CORPUS_URL]


def test_a_sibling_path_that_only_spells_the_excluded_one_is_not_under_it():
    """A path sits under another at a segment boundary and nowhere else, so /paperwork is not the
    publication at /paper and the pages under it keep their archetypes."""
    corpus = f"{HOST}/paper"
    assert domain.under(f"{corpus}/tasks", corpus) and domain.under(corpus, corpus)
    assert not domain.under(f"{HOST}/paperwork/help", corpus)
    assert domain.refusal(f"{HOST}/paperwork/help", corpus, []) == ""
    assert domain.refusal(f"{corpus}/tasks", corpus, []) == domain.CORPUS_URL


def test_a_page_lands_in_the_workdir_cache_and_never_in_the_package(shop: Path):
    import kullback

    domain.crawl(shop, [{"url": f"{HOST}/broken-wick", "depth": 0}], fetch=fetch, depth=0)
    kept = sorted(domain.cache_dir(shop).glob("*.txt"))
    assert kept and "wick" in kept[0].read_text()
    package = Path(kullback.__file__).parent
    assert not any("wick will not light" in path.read_text(encoding="utf-8", errors="ignore")
                   for path in package.rglob("*") if path.is_file())


# --- the records --------------------------------------------------------------------------

def test_an_archetype_record_holds_no_sentence_of_the_page(shop: Path):
    text = domain.page_text(LONGER)
    copied = {"goal": "I want to keep it longer",
              "effects": ["A hire runs for seven nights"],
              "preconditions": ["Extending a hire that has already ended is not possible"]}
    assert domain.copies_page(copied, text)
    written = {"goal": "I want to keep my lantern a few more nights", "effects": ["it runs longer"]}
    assert not domain.copies_page(written, text)


def test_a_record_repeating_a_corpus_string_is_dropped_and_counted(shop: Path):
    write_json(shop / "tasks.json",
               {"tasks": [{"id": "t1", "name": None,
                           "intent": "I want to keep my lantern for a few more nights"}]})
    reader = Scripted({"Read it as evidence": json.dumps({"archetypes": [
        _archetype("I want to keep my lantern for a few more nights", ["my hire runs longer"]),
        _archetype("I need a lantern swapped today", ["my hire runs longer"])]})})
    body = domain.read(shop, sources=[f"{HOST}/keep-it-longer"], fetch=fetch, depth=0,
                       reader=reader, mapper=Scripted({}))
    assert body["counts"]["archetypes_contaminated"] == 1
    assert [row["goal"] for row in body["archetypes"]] == ["I need a lantern swapped today"]


def test_a_goal_written_at_the_customer_rather_than_by_them_is_dropped_and_counted(shop: Path):
    reader = Scripted({"Read it as evidence": json.dumps({"archetypes": [
        _archetype("Process the extension request and update the hire", ["the hire runs longer"]),
        _archetype("I want to keep my lantern for more nights", ["my hire runs longer"])]})})
    body = domain.read(shop, sources=[f"{HOST}/keep-it-longer"], fetch=fetch, depth=0,
                       reader=reader, mapper=Scripted({}))
    assert body["counts"]["archetypes_operator_side"] == 1
    assert [row["goal"] for row in body["archetypes"]] == [
        "I want to keep my lantern for more nights"]
    assert all(domain.customer_voice(row["goal"]) for row in body["archetypes"])


def test_two_goals_are_folded_only_where_the_harness_judge_settles_the_pair_as_equal(shop: Path):
    """The pair goes through the one comparison the harness settles a semantic pair with (D219),
    so the citation rule it enforces is that judge's and not a second copy kept here."""
    first, second = "I want to keep my lantern longer", "I would like my lantern kept longer"
    uncited = Scripted({"Value A": json.dumps({"verdict": "equivalent"})})
    assert domain.same_goal(uncited, first, second) == (False, "")
    cited = Scripted({"Value A": json.dumps({"verdict": "equivalent",
                                             "evidence": ["value_a", "value_b"]})})
    assert domain.same_goal(cited, first, second) == (True, "judge")


def test_a_pair_nobody_settled_leaves_both_archetypes_standing(shop: Path):
    """Unresolved is not agreement (D219): with no judge at all the two goals stay two."""
    records = [_archetype("I want to keep my lantern longer", ["my hire runs longer"]),
               _archetype("I would like my lantern kept longer", ["my hire runs longer"])]
    kept, folded = domain.dedup(records, None)
    assert len(kept) == 2 and folded == []


def test_a_fold_records_the_route_that_settled_it(shop: Path):
    """Two goals that read the same after normalising need no judge, and the fold says so."""
    records = [_archetype("I want to keep my lantern longer", ["my hire runs longer"]),
               _archetype("I want to keep my lantern longer", ["my hire runs longer"])]
    kept, folded = domain.dedup(records, None)
    assert len(kept) == 1 and [row["settled_by"] for row in folded] == ["canon"]


# --- the mapping and the gaps ---------------------------------------------------------------

def test_mapping_keeps_only_tools_that_exist_and_writes_the_graph_can_reach(shop: Path):
    body = json.loads((shop / graph.FILE_NAME).read_text())
    kept, reasons = domain.check_mapping(["extend_hire", "polish_lantern", "scrap_depot"], body)
    assert kept == ["extend_hire"]
    assert any("polish_lantern" in reason for reason in reasons)
    assert any("scrap_depot" in reason and "reachable" in reason for reason in reasons)


def test_an_archetype_no_tool_realises_lands_in_the_gaps(shop: Path):
    reader = Scripted({"Read it as evidence": json.dumps({"archetypes": [
        _archetype("I want to keep my lantern for more nights", ["my hire runs longer"]),
        _archetype("I want someone to come and carry it back for me", ["a person collects it"])]})})
    mapper = Scripted({"my hire runs longer": json.dumps(
        {"effects": [{"effect": "my hire runs longer", "tools": ["extend_hire"]}]}),
        "a person collects it": json.dumps(
            {"effects": [{"effect": "a person collects it", "tools": []}]})})
    body = domain.read(shop, sources=[f"{HOST}/keep-it-longer"], fetch=fetch, depth=0,
                       reader=reader, mapper=mapper)
    gaps = domain.read_gaps(shop)
    assert body["counts"]["archetypes_mapped"] == 1
    assert [row["goal"] for row in gaps] == ["I want someone to come and carry it back for me"]


def test_the_model_names_sources_for_a_one_line_description_and_the_corpus_ones_are_refused(shop: Path):
    corpus = f"{HOST}/benchmark"
    namer = Scripted({"public web pages": json.dumps({"pages": [
        {"url": f"{HOST}/broken-wick", "why": "what people write in about"},
        {"url": f"{corpus}/tasks", "why": "a task list"},
        {"url": f"{HOST}/nowhere", "why": "does not resolve"}]})})
    found = domain.named_sources(namer, "a shop that rents lanterns by the night", fetch=fetch,
                                 corpus_url=corpus)
    assert found["used"] == [f"{HOST}/broken-wick"]
    assert {row["reason"] for row in found["dropped"]} == {domain.CORPUS_URL, domain.UNREACHABLE}


def test_a_search_runs_only_where_a_key_is_set_and_the_value_is_never_returned():
    found, note = domain.search_sources("lantern hire help", env={})
    assert found == [] and domain.SEARCH_KEY_VAR in note
    found, note = domain.search_sources("lantern hire help", env={domain.SEARCH_KEY_VAR: "secret"},
                                        search=lambda query, key: [f"{HOST}/broken-wick"])
    assert found == [f"{HOST}/broken-wick"] and not note
    assert "secret" not in note


# --- the judges ---------------------------------------------------------------------------

def test_a_judge_removes_a_task_only_by_citing_a_line_of_the_archetype(shop: Path):
    record = _archetype("I want to keep my lantern for more nights", ["my hire runs longer"])
    uncited = Scripted({"Your only power": json.dumps({"reject": True, "citation": "made up"})})
    assert domain.rejection(uncited, domain.SHAPE_JUDGE, record, "keep it longer", ["extend_hire"]) == ""
    cited = Scripted({"Your only power": json.dumps(
        {"reject": True, "citation": "my hire runs longer"})})
    assert domain.rejection(cited, domain.SHAPE_JUDGE, record, "keep it longer",
                            ["extend_hire"]) == "my hire runs longer"


def test_the_archetype_lines_reach_a_judge_as_the_checks_run_before_it_is_asked(shop: Path):
    """D222 rule 1: the reads the question needs are run first and named in the prompt, so the judge
    is answering over evidence the harness gathered rather than over prose it was handed."""
    record = _archetype("I want to keep my lantern for more nights", ["my hire runs longer"])
    model = Scripted({"Your only power": json.dumps({"reject": False})})
    domain.rejection(model, domain.SHAPE_JUDGE, record, "keep it longer", ["extend_hire"])
    prompt = model.prompts[-1]
    assert "Checks already run for you" in prompt
    assert "my hire runs longer" in prompt and "effects" in prompt


def test_an_archetype_with_no_line_to_cite_is_never_put_to_a_judge(shop: Path):
    """A question with no check behind it is refused the way D222 refuses one, and a refusal removes
    nothing: these judges hold the power to reject and nothing else."""
    empty = {"goal": "", "preconditions": [], "effects": [], "constraints": []}
    model = Scripted({"Your only power": json.dumps({"reject": True, "citation": "anything"})})
    assert domain.archetype_checks(empty) == []
    assert domain.rejection(model, domain.SHAPE_JUDGE, empty, "keep it longer", ["extend_hire"]) == ""
    assert domain.judged([model, model], empty, "keep it longer", ["extend_hire"]) == []
    assert model.prompts == []


def test_a_judge_that_would_pass_a_task_adds_nothing(shop: Path):
    record = _archetype("I want to keep my lantern for more nights", ["my hire runs longer"])
    passing = Scripted({"Your only power": json.dumps({"reject": False, "citation": "anything"})})
    assert domain.judged([passing, passing], record, "keep it longer", ["extend_hire"]) == []
