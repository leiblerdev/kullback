"""Tests for shared/canon.py: canonical values, column classes, and the cached equivalence table."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from harness.shared.canon import (
    EXEMPT,
    CanonRules,
    Comparison,
    EquivalenceTable,
    canon_record,
    canon_value,
    compare,
    equal,
    load_rules,
    load_table,
    lookup,
    overturn,
    pair_key,
    put,
    record_hash,
    save_rules,
    save_table,
)
from harness.shared.records import Column, EntitySchema

# --- numbers, the tau3 case from R22 ---


def test_tau3_twenty_five_equals_twenty_five_point_zero():
    """R22: tau3 canonicalizes 25 and 25.0 to the same value; so do we (D39)."""
    assert canon_value(25) == canon_value(25.0) == "25"
    assert canon_value("25") == canon_value("25.00") == "25"
    assert canon_value(Decimal("25.000")) == "25"
    assert equal(25, 25.0, "hard")
    assert equal("25.00", 25, "hard")


def test_numbers_keep_the_digits_that_matter():
    assert canon_value(25.10) == "25.1"
    assert canon_value(-0.0) == "0"
    assert canon_value(1000.0) == "1000"
    assert canon_value(0.5) == "0.5"
    assert not equal(25, 25.5, "hard")


def test_number_precision_is_a_rule():
    rules = CanonRules(number_precision=2)
    assert canon_value(25.004, rules=rules) == "25"
    assert canon_value(25.006, rules=rules) == "25.01"
    assert equal(25.004, 25.0, "hard", rules)


def test_currency_strings_canonicalize_the_amount_and_keep_the_currency():
    """Only the spelling of the amount is noise; the currency is part of the value (D39)."""
    assert canon_value("$25.00") == "25 usd"
    assert canon_value("25 USD") == "25 usd"
    assert canon_value("USD 25.0") == "25 usd"
    assert canon_value("1,234.50") == "1234.5"
    assert equal("$25.00", "25.0 USD", "hard")
    assert not equal("$25.00", 25, "hard"), "a bare 25 is not an amount in dollars"


def test_currency_can_be_turned_off():
    rules = CanonRules(currency=False)
    assert canon_value("$25.00", rules=rules) == "$25.00"


def test_leading_zero_strings_are_not_numbers():
    """0025 is an id shape, not a number; canonicalizing it to 25 would merge two rows."""
    assert canon_value("0025") == "0025"
    assert not equal("0025", 25, "hard")
    assert canon_value("0") == "0"
    assert canon_value("0.5") == "0.5"


def test_booleans_and_none():
    assert canon_value(True) == "true"
    assert canon_value(False) == "false"
    assert canon_value(None) == "null"
    assert not equal(True, 1, "hard")
    assert equal(None, None, "hard")


# --- whitespace and case, the tau3 case from R22 ---


def test_tau3_whitespace_and_case():
    assert canon_value("  Order   Cancelled \n") == "order cancelled"
    assert equal("Order Cancelled", "order   cancelled", "hard")
    assert canon_value("a\tb") == "a b"


def test_case_folding_is_a_rule():
    rules = CanonRules(lowercase=False)
    assert canon_value("Order", rules=rules) == "Order"
    assert not equal("Order", "order", "hard", rules)


def test_whitespace_collapse_is_a_rule():
    rules = CanonRules(collapse_whitespace=False)
    assert canon_value(" a  b ", rules=rules) == "a  b"


# --- timestamps ---


def test_timestamps_go_to_iso_utc():
    same = {
        canon_value("2024-05-01T12:00:00Z"),
        canon_value("2024-05-01 12:00:00"),
        canon_value("2024-05-01T14:00:00+02:00"),
        canon_value("2024-05-01T12:00:00.000Z"),
    }
    assert same == {"2024-05-01T12:00:00Z"}
    assert equal("2024-05-01 12:00:00", "2024-05-01T14:00:00+02:00", "hard")


def test_date_only_stays_a_date():
    assert canon_value("2024-05-01") == "2024-05-01"


def test_extra_timestamp_formats_are_data():
    rules = CanonRules(timestamp_formats=["%d/%m/%Y %H:%M"])
    assert canon_value("01/05/2024 12:00", rules=rules) == "2024-05-01T12:00:00Z"
    assert canon_value("01/05/2024 12:00") == "01/05/2024 12:00"


def test_timestamps_can_be_turned_off():
    rules = CanonRules(timestamps=False)
    assert canon_value("2024-05-01T12:00:00Z", rules=rules) == "2024-05-01t12:00:00z"


# --- id formats ---


def test_id_patterns_normalize_case_but_keep_shape():
    rules = CanonRules(id_patterns={"sku": r"[A-Za-z]{2}-\d+"})
    assert canon_value("ab-100", rules=rules) == "AB-100"
    assert equal("ab-100", "AB-100", "hard", rules)
    assert not equal("AB-0100", "AB-100", "hard", rules)


def test_id_strip_chars_are_a_rule():
    rules = CanonRules(id_patterns={"order": r"#?W\d+"}, id_strip_chars="#")
    assert canon_value("#W123", rules=rules) == canon_value("W123", rules=rules) == "W123"


def test_id_patterns_win_over_the_number_rule():
    rules = CanonRules(id_patterns={"account": r"\d{4}"})
    assert canon_value("0025", rules=rules) == "0025"
    assert canon_value("2500", rules=rules) == "2500"


# --- containers and list order ---


def test_dict_key_order_does_not_matter_and_values_canonicalize():
    left = canon_value({"amount": 25.0, "name": " Ada "})
    right = canon_value({"name": "ada", "amount": "25.00"})
    assert left == right
    assert json.loads(left) == {"amount": "25", "name": "ada"}


def test_list_order_matters_by_default():
    assert canon_value(["b", "a"]) != canon_value(["a", "b"])


def test_list_order_is_ignored_where_the_schema_says_so():
    rules = CanonRules(unordered_lists=["items"])
    assert canon_value(["b", "a"], rules=rules, path="items") == canon_value(
        ["a", "b"], rules=rules, path="items"
    )
    assert canon_value(["b", "a"], rules=rules, path="steps") != canon_value(
        ["a", "b"], rules=rules, path="steps"
    )


def test_unordered_list_paths_match_nested_fields():
    rules = CanonRules(unordered_lists=["order.items"])
    left = canon_value({"items": [2, 1]}, rules=rules, path="order")
    right = canon_value({"items": [1, 2]}, rules=rules, path="order")
    assert left == right


def test_unordered_all_is_a_rule():
    rules = CanonRules(unordered_all=True)
    assert canon_value([2, 1], rules=rules) == canon_value([1, 2], rules=rules)


# --- column classes: exempt, hard, semantic ---


def test_tau3_reads_never_hashed_is_our_exempt_class():
    """R22: tau3 keeps read observations out of the hash. Our exempt columns do the same (D73)."""
    assert canon_value("anything at all", "exempt") == EXEMPT
    assert canon_value({"nested": 1}, "exempt") == EXEMPT
    assert equal("2024-05-01T12:00:00Z", "1999-01-01T00:00:00Z", "exempt")
    assert compare("a", "b", "exempt").route == "exempt"


def test_hard_columns_compare_canonical_strings_only():
    result = compare("Cancelled", "  cancelled ", "hard")
    assert result.equal is True
    assert result.route == "canon"
    assert result.judge_used is False
    assert compare("cancelled", "pending", "hard").equal is False


def test_a_hard_column_never_reaches_the_judge():
    """D84 fences the model to semantic columns; hard columns are code, without exception."""
    calls = []
    result = compare("cancelled", "voided", "hard", judge=lambda *args: calls.append(args) or True)
    assert result.equal is False
    assert result.judge_used is False
    assert calls == []


def test_semantic_columns_that_canonicalize_the_same_need_no_judge():
    calls = []

    def judge(column, a, b):
        calls.append((column, a, b))
        return True

    result = compare("Broken screen", "broken  screen", "semantic", judge=judge, column="t.reason")
    assert result.equal is True
    assert result.route == "canon"
    assert result.judge_used is False
    assert calls == []


def test_semantic_columns_without_a_judge_are_unresolved_and_not_equal():
    result = compare("screen is broken", "cracked display", "semantic", column="t.reason")
    assert result.equal is False
    assert result.route == "unresolved"
    assert result.judge_used is False
    assert equal("screen is broken", "cracked display", "semantic") is False


def test_free_text_is_still_graded_unlike_tau3():
    """D84 deviation from R22: we compare semantic columns with the judge, we do not drop them."""
    table = EquivalenceTable()
    asked = []

    def judge(column, a, b):
        asked.append((column, a, b))
        return {"verdict": "equivalent" if b in a else "not_equivalent"}

    same = compare("cancelled by user", "cancelled", "semantic",
                   judge=judge, table=table, column="orders.reason")
    assert same.equal is True
    assert same.route == "judge"
    different = compare("cancelled by user", "shipped", "semantic",
                        judge=judge, table=table, column="orders.reason")
    assert different.equal is False
    assert asked == [
        ("orders.reason", "cancelled by user", "cancelled"),
        ("orders.reason", "cancelled by user", "shipped"),
    ]
    assert len(table.entries) == 2, "both answers are cached, so free text is graded once"


# --- the equivalence table ---


def test_judge_decides_once_and_the_pair_is_cached():
    table = EquivalenceTable()
    calls = []

    def judge(column, a, b):
        calls.append((column, a, b))
        return True

    first = compare(
        "screen is broken", "cracked display", "semantic",
        judge=judge, table=table, column="tickets.reason",
    )
    assert first.equal is True
    assert first.route == "judge"
    assert first.judge_used is True
    assert len(calls) == 1

    second = compare(
        "screen is broken", "cracked display", "semantic",
        judge=judge, table=table, column="tickets.reason",
    )
    assert second.equal is True
    assert second.route == "cache"
    assert second.judge_used is True, "this Verdict still rests on a model decision (D84 audit)"
    assert second.judge_called is False, "but no model was called for it"
    assert len(calls) == 1, "a cached pair must not cost a second judge call"


def test_the_cache_is_symmetric_and_per_column():
    table = EquivalenceTable()
    judge_calls = []

    def judge(column, a, b):
        judge_calls.append(column)
        return True

    compare("a b", "c d", "semantic", judge=judge, table=table, column="tickets.reason")
    flipped = compare("c d", "a b", "semantic", judge=judge, table=table, column="tickets.reason")
    assert flipped.route == "cache"
    other_column = compare("a b", "c d", "semantic", judge=judge, table=table, column="orders.note")
    assert other_column.route == "judge"
    assert judge_calls == ["tickets.reason", "orders.note"]


def test_judge_may_return_a_reason():
    table = EquivalenceTable()

    def judge(column, a, b):
        return {"equal": False, "reason": "different part of the order"}

    result = compare("item red", "item blue", "semantic", judge=judge, table=table, column="c")
    assert result.equal is False
    entry = lookup(table, "c", "item red", "item blue")
    assert entry is not None
    assert entry.note == "different part of the order"
    assert entry.classified_by == "llm"


def test_entries_carry_classified_by_and_the_judge_version():
    table = EquivalenceTable()
    compare(
        "a", "b", "semantic",
        judge=lambda column, x, y: True, table=table, column="c", judge_version="j2",
    )
    entry = lookup(table, "c", "a", "b")
    assert entry.classified_by == "llm"
    assert entry.judge_version == "j2"
    assert entry.equal is True
    assert entry.key == pair_key("c", "a", "b") == pair_key("c", "b", "a")


def test_a_human_can_overturn_an_entry_and_the_llm_cannot_overwrite_it():
    table = EquivalenceTable()
    put(table, "c", "a", "b", True)
    entry = overturn(table, "c", "a", "b", False, note="these are different orders")
    assert entry.equal is False
    assert entry.classified_by == "human"

    put(table, "c", "b", "a", True)
    assert lookup(table, "c", "a", "b").equal is False
    assert lookup(table, "c", "a", "b").classified_by == "human"
    assert len(table.entries) == 1

    result = compare("a", "b", "semantic", judge=lambda *_: True, table=table, column="c")
    assert result.equal is False
    assert result.route == "cache"
    assert result.classified_by == "human"


def test_put_replaces_a_machine_entry():
    table = EquivalenceTable()
    put(table, "c", "a", "b", True)
    put(table, "c", "a", "b", False, note="second look")
    assert len(table.entries) == 1
    assert lookup(table, "c", "a", "b").equal is False


def test_the_table_is_a_file_a_human_can_open(workdir):
    path = workdir / "equivalence.json"
    table = EquivalenceTable()
    put(table, "tickets.reason", "screen is broken", "cracked display", True)
    save_table(table, path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["entries"][0]["column"] == "tickets.reason"
    assert raw["entries"][0]["classified_by"] == "llm"

    back = load_table(path)
    assert back.entries == table.entries
    assert lookup(back, "tickets.reason", "cracked display", "screen is broken").equal is True


def test_load_table_on_a_missing_file_is_empty(workdir):
    assert load_table(workdir / "nope.json").entries == []


def test_compare_can_take_a_table_path_and_writes_it(workdir):
    path = workdir / "equivalence.json"
    result = compare(
        "a", "b", "semantic", judge=lambda column, x, y: True, table=path, column="c",
    )
    assert result.equal is True
    assert path.exists()
    assert lookup(load_table(path), "c", "a", "b").equal is True

    calls = []
    again = compare(
        "a", "b", "semantic",
        judge=lambda column, x, y: calls.append(1) or True, table=path, column="c",
    )
    assert again.route == "cache"
    assert calls == []


def test_the_cached_pair_is_the_canonical_pair():
    """The table keys on canonical strings, so wording noise does not grow it (D84)."""
    table = EquivalenceTable()
    compare("Screen  broken", "cracked display", "semantic",
            judge=lambda *_: True, table=table, column="c")
    result = compare("screen broken ", "cracked display", "semantic",
                     judge=lambda *_: True, table=table, column="c")
    assert result.route == "cache"
    assert len(table.entries) == 1
    assert table.entries[0].a == "screen broken"


# --- records and schemas ---


def make_schema() -> EntitySchema:
    return EntitySchema(
        tables=["orders"],
        columns=[
            Column(table="orders", name="id", **{"class": "hard"}),
            Column(table="orders", name="total", **{"class": "hard"}),
            Column(table="orders", name="updated_at", **{"class": "exempt"}),
            Column(table="orders", name="reason", **{"class": "semantic"}),
        ],
    )


def test_canon_record_applies_the_class_per_column():
    row = {"id": "W123", "total": 25.0, "updated_at": "2024-05-01T12:00:00Z", "reason": " Broken "}
    out = canon_record(row, make_schema(), table="orders")
    assert out == {"id": "w123", "total": "25", "updated_at": EXEMPT, "reason": "broken"}


def test_canon_record_takes_a_plain_column_map_too():
    out = canon_record({"a": 1.0, "b": "x"}, {"a": "hard", "b": "exempt"})
    assert out == {"a": "1", "b": EXEMPT}


def test_canon_record_takes_a_table_keyed_column_map():
    schema = {"orders": {"note": "exempt"}, "tickets": {"note": "hard"}}
    assert canon_record({"note": "x"}, schema, table="orders") == {"note": EXEMPT}
    assert canon_record({"note": "x"}, schema, table="tickets") == {"note": "x"}


def test_canon_record_uses_the_default_class_for_unknown_columns():
    out = canon_record({"surprise": " New  Field "}, make_schema(), table="orders")
    assert out == {"surprise": "new field"}
    rules = CanonRules(default_class="exempt")
    assert canon_record({"surprise": "x"}, make_schema(), table="orders", rules=rules) == {
        "surprise": EXEMPT
    }


def test_canon_record_reads_the_table_the_column_belongs_to():
    schema = EntitySchema(
        columns=[
            Column(table="orders", name="note", **{"class": "exempt"}),
            Column(table="tickets", name="note", **{"class": "hard"}),
        ]
    )
    assert canon_record({"note": "x"}, schema, table="orders") == {"note": EXEMPT}
    assert canon_record({"note": "x"}, schema, table="tickets") == {"note": "x"}


def test_record_hash_ignores_exempt_columns():
    """The tau3 rule again, at row level: a changed timestamp is not a changed End state."""
    schema = make_schema()
    base = {"id": "W123", "total": 25, "updated_at": "2024-05-01T12:00:00Z", "reason": "broken"}
    later = dict(base, updated_at="2025-09-09T09:09:09Z")
    assert record_hash(base, schema, "orders") == record_hash(later, schema, "orders")

    changed = dict(base, total=26)
    assert record_hash(base, schema, "orders") != record_hash(changed, schema, "orders")


def test_record_hash_is_stable_across_key_order_and_number_spelling():
    schema = make_schema()
    left = {"id": "W123", "total": "25.00"}
    right = {"total": 25, "id": "w123"}
    assert record_hash(left, schema, "orders") == record_hash(right, schema, "orders")


# --- rules as data ---


def test_rules_round_trip_through_a_file(workdir):
    path = workdir / "canon-rules.json"
    rules = CanonRules(lowercase=False, unordered_lists=["orders.items"], id_patterns={"o": r"W\d+"})
    save_rules(rules, path)
    back = load_rules(path)
    assert back == rules
    assert json.loads(path.read_text(encoding="utf-8"))["unordered_lists"] == ["orders.items"]


def test_load_rules_on_a_missing_file_gives_the_defaults(workdir):
    assert load_rules(workdir / "nope.json") == CanonRules()


def test_default_rules_are_not_shared_state():
    """A caller mutating its own rules must not change what the next caller canonicalizes."""
    from harness.shared.canon import DEFAULT_RULES

    mine = CanonRules()
    mine.lowercase = False
    mine.unordered_lists.append("items")
    mine.id_patterns["x"] = r"\d+"
    assert canon_value(" A ", rules=mine) == "A"
    assert canon_value(" A ") == "a"
    assert DEFAULT_RULES.lowercase is True
    assert DEFAULT_RULES.unordered_lists == []
    assert DEFAULT_RULES.id_patterns == {}
    assert canon_value(["b", "a"], path="items") != canon_value(["a", "b"], path="items")


def test_equal_is_compare_dot_equal_on_every_route():
    table = EquivalenceTable()
    put(table, "c", "yes", "yeah", True, "llm", "1")
    cases = [
        (("a", "a", "hard"), {}, True, "canon"),
        (("a", "b", "hard"), {}, False, "canon"),
        (("a", "b", "exempt"), {}, True, "exempt"),
        (("a", "b", "semantic"), {}, False, "unresolved"),
        (("yes", "yeah", "semantic"), {"table": table, "column": "c"}, True, "cache"),
        (("a", "b", "semantic"),
         {"judge": lambda c, x, y: {"equal": False}, "column": "c"}, False, "judge"),
    ]
    for args, kwargs, expected, route in cases:
        result = compare(*args, **kwargs)
        assert isinstance(result, Comparison)
        assert result.route == route
        assert result.equal is expected
        assert equal(*args, **kwargs) is expected


def test_unknown_column_class_is_refused():
    with pytest.raises(ValueError):
        canon_value("x", "mystery")


def test_an_overturned_entry_survives_the_file_and_still_outranks_the_judge(workdir):
    """The overturn a person made must still be there, and still win, after a reload (D84)."""
    path = workdir / "equivalence.json"
    table = EquivalenceTable()
    overturn(table, "c", "a", "b", False, note="two different orders")
    save_table(table, path)

    back = load_table(path)
    entry = lookup(back, "c", "b", "a")
    assert entry.classified_by == "human"
    assert entry.equal is False
    assert entry.note == "two different orders"

    calls = []
    result = compare("a", "b", "semantic",
                     judge=lambda *_: calls.append(1) or True, table=back, column="c")
    assert result.equal is False
    assert result.classified_by == "human"
    assert calls == []


# --- D84: overturning a cached pair queues a regrade ---

def test_overturning_a_pair_queues_every_run_that_used_it(tmp_path):
    from harness.shared.canon import (
        clear_regrade_queue,
        overturn,
        put,
        queued_regrades,
        record_use,
    )

    table = EquivalenceTable()
    put(table, "reason", "no longer needed", "changed my mind", True, "llm", "1")
    used = compare("no longer needed", "changed my mind", "semantic", table=table, column="reason")
    assert used.route == "cache"
    record_use(tmp_path, used, run_id="r1", task_id="t1")
    record_use(tmp_path, used, run_id="r2", task_id="t1")
    record_use(tmp_path, used, run_id="r1", task_id="t1")

    entry = overturn(table, "reason", "no longer needed", "changed my mind", False,
                     note="different reasons", workdir=tmp_path)
    assert entry.classified_by == "human"
    assert entry.equal is False
    assert queued_regrades(tmp_path) == ["r1", "r2"]
    clear_regrade_queue(tmp_path)
    assert queued_regrades(tmp_path) == []


def test_only_the_runs_a_batch_rescored_leave_the_queue(tmp_path):
    """D84: a queued Run the batch did not re-score stays queued, or it keeps its stale Verdict."""
    from harness.shared.canon import (
        clear_regrade_queue,
        overturn,
        put,
        queued_regrades,
        record_use,
    )

    table = EquivalenceTable()
    put(table, "reason", "no longer needed", "changed my mind", True, "llm", "1")
    used = compare("no longer needed", "changed my mind", "semantic", table=table, column="reason")
    for run_id in ("r1", "r2", "r3"):
        record_use(tmp_path, used, run_id=run_id, task_id="t1")
    overturn(table, "reason", "no longer needed", "changed my mind", False, workdir=tmp_path)
    assert queued_regrades(tmp_path) == ["r1", "r2", "r3"]

    clear_regrade_queue(tmp_path, {"r1", "r3"})
    assert queued_regrades(tmp_path) == ["r2"]
    clear_regrade_queue(tmp_path, ["r2"])
    assert queued_regrades(tmp_path) == []


def test_an_overturn_without_a_workdir_still_corrects_the_table(tmp_path):
    from harness.shared.canon import overturn, queued_regrades

    table = EquivalenceTable()
    overturn(table, "reason", "a", "b", False)
    assert table.entries[0].classified_by == "human"
    assert queued_regrades(tmp_path) == []


def test_a_comparison_with_no_cache_key_records_no_use(tmp_path):
    from harness.shared.canon import queued_regrades, record_use

    record_use(tmp_path, compare("a", "a", "hard"), run_id="r1")
    assert queued_regrades(tmp_path) == []


# --- D84: the judge's answer is read, never coerced (canon-1, canon-13) ---


class FakeJudgeResult:
    """The shape judge.py returns for the equivalence use: a verdict string, not a truthy object."""

    def __init__(self, verdict: str, reason: str | None = None, refused: bool = False):
        self.use = "equivalence"
        self.verdict = verdict
        self.reason = reason
        self.refused = refused


def test_a_judge_result_verdict_decides_the_comparison():
    """D84: judge.py answers with equivalent, not_equivalent or abstain; bool() of it is always True."""
    table = EquivalenceTable()
    said_no = compare(
        "screen is broken", "coffee is cold", "semantic",
        judge=lambda c, a, b: FakeJudgeResult("not_equivalent", "different complaints"),
        table=table, column="tickets.reason",
    )
    assert said_no.equal is False
    assert said_no.route == "judge"
    assert lookup(table, "tickets.reason", "screen is broken", "coffee is cold").equal is False

    said_yes = compare(
        "a b", "c d", "semantic",
        judge=lambda c, x, y: FakeJudgeResult("equivalent"), table=table, column="c",
    )
    assert said_yes.equal is True
    assert lookup(table, "c", "a b", "c d").equal is True


def test_an_abstaining_judge_leaves_the_pair_unresolved_and_uncached():
    table = EquivalenceTable()
    result = compare(
        "a b", "c d", "semantic",
        judge=lambda c, x, y: FakeJudgeResult("abstain"), table=table, column="c",
    )
    assert result.equal is False
    assert result.route == "unresolved"
    assert result.judge_used is False
    assert result.judge_called is True
    assert table.entries == [], "an abstain decides nothing, so nothing may be cached"


def test_a_refused_judge_answer_is_unresolved():
    table = EquivalenceTable()
    result = compare(
        "a b", "c d", "semantic",
        judge=lambda c, x, y: FakeJudgeResult("equivalent", refused=True), table=table, column="c",
    )
    assert result.route == "unresolved"
    assert table.entries == []


def test_a_judge_answer_that_is_not_a_verdict_is_refused_loudly():
    """A string 'no' or 'false' used to canonicalize to equal; a silent pass is the worst outcome."""
    for answer in ("no", "false", "not_equivalent_at_all", 0, 1, None, ["equivalent"]):
        with pytest.raises(TypeError):
            compare("a b", "c d", "semantic", judge=lambda c, x, y, answer=answer: answer, column="c")


def test_a_dict_answer_must_carry_a_real_bool():
    assert compare(
        "a b", "c d", "semantic", judge=lambda c, x, y: {"equal": False}, column="c"
    ).equal is False
    with pytest.raises(TypeError):
        compare("a b", "c d", "semantic", judge=lambda c, x, y: {"equal": "false"}, column="c")


def test_a_dict_answer_may_carry_the_verdict_word():
    result = compare(
        "a b", "c d", "semantic",
        judge=lambda c, x, y: {"verdict": "not_equivalent", "reason": "unrelated"}, column="c",
    )
    assert result.equal is False
    assert result.note == "unrelated"


def test_a_judge_that_raises_leaves_the_pair_unresolved(workdir):
    """One failed judge call must not abort a whole Verdict (D84)."""
    path = workdir / "equivalence.json"

    def judge(column, a, b):
        raise RuntimeError("the provider timed out")

    result = compare("a b", "c d", "semantic", judge=judge, table=path, column="c")
    assert result.equal is False
    assert result.route == "unresolved"
    assert result.judge_used is False
    assert "the provider timed out" in (result.note or "")
    assert load_table(path).entries == []


# --- D84: a Verdict resting on a cached model decision is still judge_used (canon-9) ---


def test_a_cached_llm_decision_still_counts_as_judge_used():
    table = EquivalenceTable()
    put(table, "c", "a b", "c d", True, "llm", "1")
    hit = compare("a b", "c d", "semantic", table=table, column="c")
    assert hit.route == "cache"
    assert hit.judge_used is True, "the audit must see the Verdicts resting on a model decision"
    assert hit.judge_called is False, "a cache hit costs no model call"

    put(table, "c", "e f", "g h", True, "human", "1")
    human = compare("e f", "g h", "semantic", table=table, column="c")
    assert human.route == "cache"
    assert human.judge_used is False
    assert human.classified_by == "human"


# --- D39: currency is part of the value, not noise (canon-2) ---


def test_the_currency_stays_in_the_canonical_string():
    assert canon_value("25 USD") != canon_value("25 EUR")
    assert not equal("25 USD", "25 EUR", "hard")
    assert not equal("$25", "€25", "hard")
    assert equal("$25.00", "25 usd", "hard")
    assert equal("USD 25.0", "25 USD", "hard")


# --- D39: no digit is ever rounded away (canon-5, canon-11) ---


def test_long_numeric_ids_keep_every_digit():
    left, right = "12345678901234567890123456789012", "12345678901234567890123456789013"
    assert canon_value(left) == left
    assert canon_value(right) == right
    assert not equal(left, right, "hard")
    assert not equal(int(left), int(right), "hard")


def test_big_and_non_finite_numbers_do_not_crash_canonicalization():
    rules = CanonRules(number_precision=2)
    assert canon_value(Decimal("1e30"), rules=rules) == "1000000000000000000000000000000"
    assert canon_value(float("inf"), rules=rules) == "infinity"
    assert canon_value(float("-inf"), rules=rules) == "-infinity"
    assert canon_value(Decimal("1e30")) == "1000000000000000000000000000000"


# --- D39: a digit string is not a timestamp (canon-6) ---


def test_digit_only_strings_are_not_timestamps():
    assert canon_value("20240501") == "20240501"
    assert equal("20240501", 20240501, "hard")
    assert not equal("20240501", "2024-05-01T00:00:00Z", "hard")
    assert canon_value("2024-W18-3") == "2024-w18-3"
    assert canon_value("20240501T120000") == "20240501t120000"
    assert canon_value("2024-05-01T12:00:00Z") == "2024-05-01T12:00:00Z"


# --- D39: a case-sensitive id column keeps its case (canon-7) ---


def test_case_sensitive_columns_are_not_folded():
    rules = CanonRules(case_sensitive_paths=["customers.id"])
    assert canon_value("cus_AbC123", rules=rules, path="customers.id") == "cus_AbC123"
    assert not equal("cus_AbC123", "cus_ABC123", "hard", rules, column="customers.id")
    assert canon_value("cus_AbC123") == "cus_abc123", "the default still folds"


def test_a_case_sensitive_column_beats_the_id_upper_rule():
    rules = CanonRules(id_patterns={"cus": r"cus_[A-Za-z0-9]+"}, case_sensitive_paths=["id"])
    assert canon_value("cus_AbC123", rules=rules, path="id") == "cus_AbC123"
    assert canon_value("cus_AbC123", rules=rules, path="other") == "CUS_ABC123"


def test_learned_rules_carry_the_mined_id_patterns_and_case_sensitive_ids():
    from harness.shared.canon import learn_rules

    schema = EntitySchema(
        tables=["customers"],
        columns=[Column(table="customers", name="id", **{"class": "hard"})],
        id_patterns={"customer": r"cus_[A-Za-z0-9]+"},
    )
    rules = learn_rules(schema)
    assert rules.id_patterns == {"customer": r"cus_[A-Za-z0-9]+"}
    assert "customers.id" in rules.case_sensitive_paths
    assert not equal("cus_AbC123", "cus_ABC123", "hard", rules, column="customers.id")


def test_learned_rules_call_a_list_unordered_when_the_traces_show_it_reordered():
    from harness.shared.canon import learn_rules

    rows = [
        {"table": "orders", "items": ["a", "b"], "steps": ["one", "two"]},
        {"table": "orders", "items": ["b", "a"], "steps": ["one", "two"]},
    ]
    rules = learn_rules(None, [{k: v for k, v in row.items() if k != "table"} for row in rows],
                        table="orders")
    assert "orders.items" in rules.unordered_lists
    assert "orders.steps" not in rules.unordered_lists
    assert equal(["a", "b"], ["b", "a"], "hard", rules, column="orders.items")


# --- D39: a string never spells a typed value (canon-8, canon-12) ---


def test_a_string_does_not_collide_with_a_typed_value():
    assert not equal(None, "null", "hard")
    assert not equal(None, "NULL", "hard")
    assert not equal(True, "TRUE", "hard")
    assert not equal(False, "false", "hard")
    assert not equal(["a"], '["a"]', "hard")
    assert not equal({"a": 1}, '{"a":"1"}', "hard")
    assert canon_value(True) == "true" and canon_value(None) == "null"
    assert equal("25.00", 25, "hard"), "R22 still holds: a number spelled as a string is that number"


def test_the_exempt_sentinel_is_unreachable_from_a_string():
    assert canon_value("<EXEMPT>", "hard") != EXEMPT
    assert canon_value("<exempt>", "hard") != EXEMPT
    assert not equal("<exempt>", canon_value("anything", "exempt"), "hard")


def test_a_set_canonicalizes_in_one_order_whatever_its_iteration_order():
    assert canon_value({"b", "a"}) == canon_value({"a", "b"}) == json.dumps(["a", "b"], separators=(",", ":"))
    assert equal({"a", "b"}, frozenset({"b", "a"}), "hard")


def test_a_type_canon_has_no_rule_for_is_refused():
    class Opaque:
        pass

    with pytest.raises(TypeError):
        canon_value(Opaque())


# --- D39: the two entry points agree on list order (canon-10) ---


def test_canon_record_honours_a_table_qualified_unordered_rule():
    rules = CanonRules(unordered_lists=["orders.items"])
    left = canon_record({"items": ["b", "a"]}, table="orders", rules=rules)
    right = canon_record({"items": ["a", "b"]}, table="orders", rules=rules)
    assert left == right
    assert left == {"items": canon_value(["a", "b"], rules=rules, path="orders.items")}
    assert canon_record({"items": ["b", "a"]}, table="tickets", rules=rules) != canon_record(
        {"items": ["a", "b"]}, table="tickets", rules=rules
    )
