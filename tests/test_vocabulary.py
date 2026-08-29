"""builder/vocabulary.py: what users state comes from the corpus in code; the web adds wording only (D115)."""

from __future__ import annotations

import re

from harness.builder import user_sim
from harness.builder import vocabulary as vb
from harness.shared.provider import TestModel
from harness.shared.records import EntitySchema, RawPtr, ToolCall, ToolSig, Trace, Turn
from harness.shared.search import TestSearch as ScriptedSearch


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


RETAIL_SCHEMA = EntitySchema(tables=["orders", "users"], id_patterns={"orders.order_id": r"^#W\d{7}$"})
RETAIL_SIGS = [sig("find_user_id_by_email", "email"), sig("get_order_details", "order_id"),
               sig("cancel_pending_order", "order_id", "reason")]


def retail_traces(n: int = 4) -> list[Trace]:
    return [trace(f"t{i}", f"Hi, my email is u{i}@x.com and I want to cancel order #W123456{i}, I changed my mind.",
                  [("find_user_id_by_email", {"email": f"u{i}@x.com"}),
                   ("get_order_details", {"order_id": f"#W123456{i}"}),
                   ("cancel_pending_order", {"order_id": f"#W123456{i}", "reason": "no longer needed"})])
            for i in range(n)]


def test_an_argument_the_users_state_is_a_field_with_its_shape_prefix_and_kind():
    vocab = vb.derive(retail_traces(), RETAIL_SCHEMA, RETAIL_SIGS, "# Retail agent policy")
    order = vocab.get("order_id")
    assert order is not None and order.kind == "reference" and order.prefix == "#"
    assert re.search(order.pattern, "it is W1234560") and re.search(order.pattern, "#W1234560")
    assert "schema:orders.order_id" in order.sources and "signature:get_order_details" in order.sources
    assert any(re.search(cue, "which order number is it?") for cue in order.cues)
    assert vocab.get("reason") is None, "the user never says 'no longer needed'"
    assert vocab.domain == "Retail agent policy"


def test_an_opener_argument_a_generic_field_already_asks_for_folds_into_it():
    vocab = vb.derive(retail_traces(), RETAIL_SCHEMA, RETAIL_SIGS)
    email = vocab.get("email")
    assert email.kind == "identity" and "signature:find_user_id_by_email" in email.sources
    assert [f.field for f in vocab.fields if f.field == "email"] == ["email"]


def test_the_argument_of_the_tool_that_opens_a_run_is_an_identity_field():
    schema = EntitySchema(tables=["users", "reservations"], id_patterns={"users.user_id": r"^[a-z]+_[a-z]+_\d+$"})
    sigs = [sig("get_user_details", "user_id"), sig("get_reservation_details", "reservation_id")]
    traces = [trace(f"a{i}", f"Hello, my user id is mia_li_{3660 + i} and I want to check reservation ZFA0{i}X.",
                    [("get_user_details", {"user_id": f"mia_li_{3660 + i}"}),
                     ("get_reservation_details", {"reservation_id": f"ZFA0{i}X"})]) for i in range(4)]
    vocab = vb.derive(traces, schema, sigs)
    assert vocab.get("user_id").kind == "identity" and "user_id" in vocab.by_kind("identity")
    reservation = vocab.get("reservation_id")
    assert reservation.kind == "value" and reservation.prefix == ""
    assert any(re.search(cue, "what is your reservation code?") for cue in reservation.cues)


def test_a_value_shape_that_matches_ordinary_words_is_read_only_when_asked():
    """Six plain letters is what many words look like; such a pattern must not scan every turn."""
    schema = EntitySchema(tables=["reservations"], id_patterns={"reservations.reservation_id": r"^.{6}$"})
    words = "hi, can you help me, i am not sure what to do, it is about my trip and i need an answer today, " \
            "so if you can see it on your end that would be good, and also the flight code please"
    codes = ["ABCDEF", "GHIJKL", "MNOPQR", "STUVWX"]
    traces = [trace(f"r{i}", f"{words}, the reservation is {codes[i]}",
                    [("get_reservation_details", {"reservation_id": codes[i]})]) for i in range(4)]
    vocab = vb.derive(traces, schema, [sig("get_reservation_details", "reservation_id")])
    spec = vocab.get("reservation_id")
    assert spec.pattern is None and spec.asked_only is not None
    assert user_sim.extracted_values("GHIJKL", asked=["reservation_id"], vocab=vocab) == [("reservation_id", "GHIJKL")]
    assert user_sim.extracted_values("please GHIJKL", vocab=vocab) == []


def test_a_few_distinct_values_are_matched_by_listing_them():
    traces = [trace(f"c{i}", f"Book me in {cabin.replace('_', ' ')} please.",
                    [("book_reservation", {"cabin": cabin})]) for i, cabin in enumerate(["economy", "business", "basic_economy", "economy"])]
    vocab = vb.derive(traces, EntitySchema(tables=[]), [sig("book_reservation", "cabin")])
    cabin = vocab.get("cabin")
    assert cabin.kind == "value" and re.search(cabin.pattern, "I want basic economy")
    assert user_sim.extracted_values("Business class, please.", vocab=vocab) == [("cabin", "Business")]


def test_a_shape_that_is_any_word_gives_no_pattern_and_a_value_said_everywhere_is_not_enumerated():
    states = ["TX", "NY", "CA", "WA"]
    traces = [trace(f"s{i}", f"No, I do not want insurance. Ship it to {states[i]} please, no rush.",
                    [("book", {"insurance": "no", "state": states[i]})]) for i in range(4)]
    traces += [trace(f"n{i}", "No thanks, nothing else today.", []) for i in range(6)]
    vocab = vb.derive(traces, EntitySchema(tables=[]), [sig("book", "insurance", "state")])
    state = vocab.get("state")
    assert state is not None and state.pattern is None and state.asked_only is None, "two letters is any word"
    assert vocab.get("insurance").pattern is None, '"no" goes with every Run, not with the booking'
    assert user_sim.extracted_values("No, thanks.", vocab=vocab) == []


def test_an_id_shape_of_letters_and_digits_only_matches_where_a_digit_is():
    schema = EntitySchema(tables=["reservations"], id_patterns={"reservations.reservation_id": r"^[A-Z0-9]{6}$"})
    traces = [trace(f"d{i}", f"my flight reservation is ZFA{i}4B thanks",
                    [("get_reservation_details", {"reservation_id": f"ZFA{i}4B"})]) for i in range(4)]
    vocab = vb.derive(traces, schema, [sig("get_reservation_details", "reservation_id")])
    spec = vocab.get("reservation_id")
    pattern = spec.pattern or spec.asked_only
    assert re.search(pattern, "my flight reservation is ZFA14B").group(0) == "ZFA14B"
    assert user_sim.extracted_values("my flight reservation is ZFA14B", asked=["reservation_id"], vocab=vocab) \
        == [("reservation_id", "ZFA14B")]


def test_the_derived_vocabulary_drives_user_sim():
    vocab = vb.derive(retail_traces(), RETAIL_SCHEMA, RETAIL_SIGS)
    assert user_sim.extracted_values("Order W1234567 please.", vocab=vocab) == [("order_id", "#W1234567")]
    assert user_sim.asked_fields("Could you give me the order number?", vocab=vocab) == ["order_id"]
    assert user_sim.asked_fields("Could you give me the order number?") == [], "the generic core knows no orders"


# --- the web ----------------------------------------------------------------

def _airline_vocab() -> vb.Vocabulary:
    schema = EntitySchema(tables=["reservations"], id_patterns={"reservations.reservation_id": r"^[A-Z0-9]{6}$"})
    traces = [trace(f"a{i}", f"My reservation is ZFA{i}4B.", [("get_reservation_details", {"reservation_id": f"ZFA{i}4B"})])
              for i in range(4)]
    return vb.derive(traces, schema, [sig("get_reservation_details", "reservation_id")], "# Airline agent policy")


def test_web_aliases_become_cues_only_when_a_fetched_page_carries_them():
    search = ScriptedSearch(hits={"reservation id": [{"url": "https://help.example/booking", "title": "Booking help"}]},
                        pages={"https://help.example/booking": "Have your booking reference or confirmation number ready."})
    model = TestModel(['{"aliases": ["booking reference", "confirmation number", "ticket locator", "the"]}'])
    vocab = vb.enrich(_airline_vocab(), search, model)
    spec = vocab.get("reservation_id")
    assert "booking reference" in spec.aliases and "confirmation number" in spec.aliases
    assert "ticket locator" not in spec.aliases, "no fetched page says it"
    assert "the" not in spec.aliases
    assert any(re.search(cue, "what is your booking reference?") for cue in spec.cues)
    assert "web:https://help.example/booking" in spec.sources
    assert vocab.searched[0]["aliases"] == ["booking reference", "confirmation number"]
    assert all("Airline agent policy" in q and "reservation id" in q for q in search.queries) and len(search.queries) == 2
    assert not any("ZFA" in q for q in search.queries), "a corpus value never goes onto the web"
    assert "help.example/booking" in model.calls[0]["messages"][0]["content"]


def test_generic_fields_are_not_searched_and_the_web_being_down_is_a_note():
    down = ScriptedSearch()  # nothing scripted: every search answers with no hits, every fetch fails
    vocab = vb.enrich(_airline_vocab(), down, TestModel([]))
    assert [r["field"] for r in vocab.searched] == ["reservation_id"] and vocab.searched[0]["urls"] == []
    assert vocab.get("email").aliases == ["email"]

    class Broken:
        name = "broken"

        def search(self, query, limit=8):
            raise RuntimeError("no route to host")

        def fetch(self, urls):
            raise RuntimeError("no route to host")

    vocab = vb.enrich(_airline_vocab(), Broken(), TestModel([]))
    assert vocab.notes == ["search unavailable: RuntimeError: no route to host"]
    assert vb.enrich(_airline_vocab(), None, None).notes == ["no web search: the cues are the field names alone"]


def test_a_value_of_the_field_is_never_an_alias():
    vocab = vb.derive([trace(f"c{i}", f"Book me in {c} please.", [("book_reservation", {"cabin": c})])
                       for i, c in enumerate(["economy", "business", "economy", "business"])],
                      EntitySchema(tables=[]), [sig("book_reservation", "cabin")], "# Airline")
    search = ScriptedSearch(hits={"cabin": [{"url": "https://a/", "title": "A"}]},
                            pages={"https://a/": "Choose a cabin class: economy or business."})
    out = vb.enrich(vocab, search, TestModel(['{"aliases": ["cabin class", "economy", "business"]}']))
    assert out.get("cabin").aliases == ["cabin", "cabin class"]


def test_an_unreadable_alias_reply_adds_nothing():
    assert vb.parse_aliases("sure, here: booking reference") == []
    assert vb.parse_aliases('{"aliases": "booking reference"}') == []
    assert vb.parse_aliases('```json\n{"aliases": ["Booking Reference"]}\n```') == ["booking reference"]


def test_the_vocabulary_round_trips_as_a_record():
    vocab = vb.derive(retail_traces(), RETAIL_SCHEMA, RETAIL_SIGS)
    again = vb.Vocabulary.model_validate(vocab.model_dump(mode="json"))
    assert again == vocab and again.get("order_id").prefix == "#"
