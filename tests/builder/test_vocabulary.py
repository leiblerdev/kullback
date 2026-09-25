"""builder/vocabulary.py: what users state comes from the corpus in code; the web adds wording only (D115)."""

from __future__ import annotations

import re

from kullback.ai.provider import TestModel
from kullback.builder import user_sim
from kullback.builder import vocabulary as vb
from kullback.builder.search import TestSearch as ScriptedSearch
from kullback.runner.records import EntitySchema, RawPtr, ToolCall, ToolSig, Trace, Turn


def _ptr(i: int) -> RawPtr:
    return RawPtr(file_hash="raw", sim_index=0, msg_index=i)


def trace(trace_id: str, user_says: str, calls: list[tuple[str, dict]], said_first: bool = True) -> Trace:
    """One Run: the user speaks, then the agent makes `calls` in order (or the other way round)."""
    turns, tool_calls = [], []
    if said_first:
        turns.append(Turn(idx=0, role="user", content=user_says, raw_ptr=_ptr(0)))
    for n, (name, args) in enumerate(calls):
        idx = len(turns)
        call_id = f"{trace_id}-c{n}"
        turns.append(Turn(idx=idx, role="assistant", content=None, tool_call_ids=[call_id], raw_ptr=_ptr(idx)))
        tool_calls.append(ToolCall(id=call_id, name=name, args=args, result={}, raw_ptr=_ptr(idx), trace_id=trace_id))
    if not said_first:
        turns.append(Turn(idx=len(turns), role="user", content=user_says, raw_ptr=_ptr(len(turns))))
    return Trace(trace_id=trace_id, raw_hash="raw", ingest_version="0", source="test", turns=turns,
                 tool_calls=tool_calls, raw_ptr=_ptr(0))


def sig(name: str, *args: str) -> ToolSig:
    return ToolSig(name=name, args_schema={"properties": {a: {"type": ["str"]} for a in args}})


LOAN_SCHEMA = EntitySchema(tables=["loans", "members"], id_patterns={"loans.loan_id": r"^#L\d{7}$"})
LOAN_SIGS = [sig("find_member_id_by_email", "email"), sig("get_loan_details", "loan_id"),
             sig("cancel_hold", "loan_id", "reason")]


def loan_traces(n: int = 4) -> list[Trace]:
    return [trace(f"t{i}", f"Hi, my email is u{i}@x.com and I want to renew loan #L123456{i}, please extend it.",
                  [("find_member_id_by_email", {"email": f"u{i}@x.com"}),
                   ("get_loan_details", {"loan_id": f"#L123456{i}"}),
                   ("cancel_hold", {"loan_id": f"#L123456{i}", "reason": "leaving town"})])
            for i in range(n)]


def test_an_argument_the_users_state_is_a_field_with_its_shape_prefix_and_kind():
    vocab = vb.derive(loan_traces(), LOAN_SCHEMA, LOAN_SIGS, "# Library agent policy")
    loan = vocab.get("loan_id")
    assert loan is not None and loan.kind == "reference" and loan.prefix == "#"
    assert re.search(loan.pattern, "it is L1234560") and re.search(loan.pattern, "#L1234560")
    assert "schema:loans.loan_id" in loan.sources and "signature:get_loan_details" in loan.sources
    assert any(re.search(cue, "which loan number is it?") for cue in loan.cues)
    assert vocab.get("reason") is None, "the user never says 'leaving town'"
    assert vocab.domain == "Library agent policy"


def test_an_opener_argument_a_generic_field_already_asks_for_folds_into_it():
    vocab = vb.derive(loan_traces(), LOAN_SCHEMA, LOAN_SIGS)
    email = vocab.get("email")
    assert email.kind == "identity" and "signature:find_member_id_by_email" in email.sources
    assert [f.field for f in vocab.fields if f.field == "email"] == ["email"]


def test_the_argument_of_the_tool_that_opens_a_run_is_an_identity_field():
    schema = EntitySchema(tables=["members", "holds"], id_patterns={"members.member_id": r"^[a-z]+_[a-z]+_\d+$"})
    sigs = [sig("get_member_details", "member_id"), sig("get_hold_details", "hold_id")]
    traces = [trace(f"a{i}", f"Hello, my member id is mia_li_{3660 + i} and I want to check hold HFA0{i}X.",
                    [("get_member_details", {"member_id": f"mia_li_{3660 + i}"}),
                     ("get_hold_details", {"hold_id": f"HFA0{i}X"})]) for i in range(4)]
    vocab = vb.derive(traces, schema, sigs)
    assert vocab.get("member_id").kind == "identity" and "member_id" in vocab.by_kind("identity")
    hold = vocab.get("hold_id")
    assert hold.kind == "value" and hold.prefix == ""
    assert any(re.search(cue, "what is your hold code?") for cue in hold.cues)


def test_a_value_shape_that_matches_ordinary_words_gives_no_scanning_pattern():
    """Six plain letters is what many words look like; such a pattern must not scan every turn."""
    schema = EntitySchema(tables=["holds"], id_patterns={"holds.hold_id": r"^.{6}$"})
    words = "hi, can you help me, i am not sure what to do, it is about my visit and i need an answer today, " \
            "so if you can see it on your end that would be good, and also the hold code please"
    codes = ["ABCDEF", "GHIJKL", "MNOPQR", "STUVWX"]
    traces = [trace(f"r{i}", f"{words}, the hold is {codes[i]}",
                    [("get_hold_details", {"hold_id": codes[i]})]) for i in range(4)]
    vocab = vb.derive(traces, schema, [sig("get_hold_details", "hold_id")])
    spec = vocab.get("hold_id")
    assert spec.pattern is None and spec.asked_only is not None
    assert user_sim.extracted_values("GHIJKL", asked=["hold_id"], vocab=vocab) == [("hold_id", "GHIJKL")]
    assert user_sim.extracted_values("please GHIJKL", vocab=vocab) == []
    states = ["TX", "NY", "CA", "WA"]
    traces = [trace(f"s{i}", f"No, I do not want insurance. Ship it to {states[i]} please, no rush.",
                    [("book", {"insurance": "no", "state": states[i]})]) for i in range(4)]
    traces += [trace(f"n{i}", "No thanks, nothing else today.", []) for i in range(6)]
    vocab = vb.derive(traces, EntitySchema(tables=[]), [sig("book", "insurance", "state")])
    state = vocab.get("state")
    assert state is not None and state.pattern is None and state.asked_only is None, "two letters is any word"
    assert vocab.get("insurance").pattern is None, '"no" goes with every Run, not with the hold'
    assert user_sim.extracted_values("No, thanks.", vocab=vocab) == []


def test_a_few_distinct_values_are_matched_by_listing_them():
    traces = [trace(f"c{i}", f"Seat me in {section.replace('_', ' ')} please.",
                    [("place_hold", {"section": section})]) for i, section in enumerate(["quiet", "group", "quiet_corner", "quiet"])]
    vocab = vb.derive(traces, EntitySchema(tables=[]), [sig("place_hold", "section")])
    section = vocab.get("section")
    assert section.kind == "value" and re.search(section.pattern, "I want quiet corner")
    assert user_sim.extracted_values("Group area, please.", vocab=vocab) == [("section", "Group")]


def test_an_id_shape_of_letters_and_digits_only_matches_where_a_digit_is():
    schema = EntitySchema(tables=["holds"], id_patterns={"holds.hold_id": r"^[A-Z0-9]{6}$"})
    traces = [trace(f"d{i}", f"my library hold is HFA{i}4B thanks",
                    [("get_hold_details", {"hold_id": f"HFA{i}4B"})]) for i in range(4)]
    vocab = vb.derive(traces, schema, [sig("get_hold_details", "hold_id")])
    spec = vocab.get("hold_id")
    pattern = spec.pattern or spec.asked_only
    assert re.search(pattern, "my library hold is HFA14B").group(0) == "HFA14B"
    assert user_sim.extracted_values("my library hold is HFA14B", asked=["hold_id"], vocab=vocab) \
        == [("hold_id", "HFA14B")]


def test_the_derived_vocabulary_drives_user_sim():
    vocab = vb.derive(loan_traces(), LOAN_SCHEMA, LOAN_SIGS)
    assert user_sim.extracted_values("Loan L1234567 please.", vocab=vocab) == [("loan_id", "#L1234567")]
    assert user_sim.asked_fields("Could you give me the loan number?", vocab=vocab) == ["loan_id"]
    assert user_sim.asked_fields("Could you give me the loan number?") == [], "the generic core knows no loans"


# --- the web ----------------------------------------------------------------

def _hold_vocab() -> vb.Vocabulary:
    schema = EntitySchema(tables=["holds"], id_patterns={"holds.hold_id": r"^[A-Z0-9]{6}$"})
    traces = [trace(f"a{i}", f"My hold is HFA{i}4B.", [("get_hold_details", {"hold_id": f"HFA{i}4B"})])
              for i in range(4)]
    return vb.derive(traces, schema, [sig("get_hold_details", "hold_id")], "# Library agent policy")


def test_web_aliases_become_cues_only_when_a_fetched_page_carries_them():
    search = ScriptedSearch(hits={"hold id": [{"url": "https://help.example/booking", "title": "Booking help"}]},
                        pages={"https://help.example/booking": "Have your hold reference or confirmation slip ready."})
    model = TestModel(['{"aliases": ["hold reference", "confirmation slip", "call number", "the"]}'])
    vocab = vb.enrich(_hold_vocab(), search, model)
    spec = vocab.get("hold_id")
    assert "hold reference" in spec.aliases and "confirmation slip" in spec.aliases
    assert "call number" not in spec.aliases, "no fetched page says it"
    assert "the" not in spec.aliases
    assert any(re.search(cue, "what is your hold reference?") for cue in spec.cues)
    assert "web:https://help.example/booking" in spec.sources
    assert vocab.searched[0]["aliases"] == ["hold reference", "confirmation slip"]
    assert all("Library agent policy" in q and "hold id" in q for q in search.queries) and len(search.queries) == 2
    assert not any("HFA" in q for q in search.queries), "a corpus value never goes onto the web"
    assert "help.example/booking" in model.calls[0]["messages"][0]["content"]


def test_generic_fields_the_web_being_down_and_unreadable_alias_replies_add_nothing():
    down = ScriptedSearch()  # nothing scripted: every search answers with no hits, every fetch fails
    vocab = vb.enrich(_hold_vocab(), down, TestModel([]))
    assert [r["field"] for r in vocab.searched] == ["hold_id"] and vocab.searched[0]["urls"] == []
    assert vocab.get("email").aliases == ["email"]

    class Broken:
        name = "broken"

        def search(self, query, limit=8):
            raise RuntimeError("no route to host")

        def fetch(self, urls):
            raise RuntimeError("no route to host")

    vocab = vb.enrich(_hold_vocab(), Broken(), TestModel([]))
    assert vocab.notes == ["search unavailable: RuntimeError: no route to host"]
    assert vb.enrich(_hold_vocab(), None, None).notes == ["no web search: the cues are the field names alone"]
    assert vb.parse_aliases("sure, here: hold reference") == []
    assert vb.parse_aliases('{"aliases": "hold reference"}') == []
    assert vb.parse_aliases('```json\n{"aliases": ["Hold Reference"]}\n```') == ["hold reference"]


def test_a_value_of_the_field_is_never_an_alias():
    vocab = vb.derive([trace(f"c{i}", f"Seat me in {c} please.", [("place_hold", {"section": c})])
                       for i, c in enumerate(["quiet", "group", "quiet", "group"])],
                      EntitySchema(tables=[]), [sig("place_hold", "section")], "# Library")
    search = ScriptedSearch(hits={"section": [{"url": "https://a/", "title": "A"}]},
                            pages={"https://a/": "Choose a seating section: quiet or group."})
    out = vb.enrich(vocab, search, TestModel(['{"aliases": ["seating section", "quiet", "group"]}']))
    assert out.get("section").aliases == ["section", "seating section"]
