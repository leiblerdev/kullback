"""Tests for the memorised_values ruling: a tool body reads the world, it does not remember it (D162).

The world here is an invented lending library: books on shelves, loans against them, and a tool
that renews one loan. Nothing about any customer's corpus enters, which is the point of the gate:
it knows an id by the shape the schema mined, by the rows the Starting state holds and by the
values the recorded calls passed, and never by a name someone wrote into the code.
"""

from __future__ import annotations

from conftest import PTR
from kullback.gates.tool_runs import body_literals, body_memorised_values_gate, load_readers
from kullback.runner.records import Column, EntitySchema, FieldStat, ToolCall, ToolSig

LIBRARY = {
    "books": {
        "BK4821": {"book_id": "BK4821", "title": "Tide Tables", "shelf": "north", "copies": 3},
        "BK5109": {"book_id": "BK5109", "title": "Kite Repair", "shelf": "south", "copies": 1},
    },
    "loans": {
        "LN0031": {"loan_id": "LN0031", "book_id": "BK4821", "days_left": 14, "term": "standard"},
        "LN0044": {"loan_id": "LN0044", "book_id": "BK5109", "days_left": 2, "term": "short"},
    },
}


def _schema() -> EntitySchema:
    columns = [Column(table=table, name=name, **{"class": "hard"})
               for table, rows in LIBRARY.items()
               for name in sorted({key for row in rows.values() for key in row})]
    return EntitySchema(tables=sorted(LIBRARY), columns=columns,
                        id_patterns={"books.book_id": r"^BK\d{4}$", "loans.loan_id": r"^LN\d{4}$"})


def _sig(**kw) -> ToolSig:
    base = dict(name="renew_loan",
                description="Renew one loan for another term.",
                args_fields=[FieldStat(name="loan_id", types=["str"], optional=False),
                             FieldStat(name="term", types=["str"], optional=False)],
                kind="write", unclassified=False)
    base.update(kw)
    return ToolSig(**base)


def _calls() -> list[ToolCall]:
    return [ToolCall(id=f"c{i}", name="renew_loan", args=args, result={"loan_id": args["loan_id"]},
                     raw_ptr=PTR)
            for i, args in enumerate([{"loan_id": "LN0031", "term": "standard"},
                                      {"loan_id": "LN0044", "term": "short"}])]


def _rule(source, sig=None, calls=None, schema=None):
    return body_memorised_values_gate(source, schema or _schema(), LIBRARY,
                                      _calls() if calls is None else calls, sig)


# --- the body that reads the world ---


def test_a_body_that_looks_the_row_up_by_its_argument_passes():
    source = ("loan = self.db.loans[loan_id]\n"
              "book = self.db.books[loan.book_id]\n"
              "loan.days_left = loan.days_left + 21\n"
              "return {\"loan_id\": loan.loan_id, \"title\": book.title}\n")
    result = _rule(source, _sig())
    assert result.passed is True, result.failures
    assert result.stage == "compile_tools.memorised_values"


def test_the_column_and_table_names_a_body_has_to_say_are_never_memorised_values():
    """A row is addressed by name: `books`, `loan_id`, `days_left` are structure, not data."""
    source = "return {\"loan_id\": loan_id, \"days_left\": self.db.loans[loan_id].days_left}\n"
    assert _rule(source, _sig()).passed is True


# --- rule (a): the shape the schema mined for a table's ids ---


def test_a_literal_with_a_mined_id_shape_fails_and_the_failure_names_it_and_the_pattern():
    source = "if loan_id == \"LN9999\":\n    raise ValueError(\"Loan not found\")\nreturn self.db.loans[loan_id]\n"
    result = _rule(source, _sig())
    assert result.passed is False
    assert len(result.failures) == 1
    assert "'LN9999'" in result.failures[0]
    assert "loans.loan_id" in result.failures[0] and r"^LN\d{4}$" in result.failures[0]


def test_a_literal_with_an_id_shape_fails_even_where_no_recorded_call_carried_it():
    """The shape is the rule, not the sighting: a body that invents a well-shaped id is memorising
    the corpus's vocabulary just as surely as one that copies a value out of a call."""
    source = "return self.db.books[\"BK0001\"]\n"
    result = body_memorised_values_gate(source, _schema(), {}, [], _sig())
    assert result.passed is False and "books.book_id" in result.failures[0]


def test_an_id_pattern_that_accepts_an_ordinary_word_is_not_read_as_an_id_shape():
    """`mine.id_pattern` falls back to a bare character class when a column's values share nothing.
    Read as an id shape, that would refuse every alphanumeric literal a body writes."""
    schema = _schema()
    schema.id_patterns = {"books.shelf": r"^[A-Za-z0-9]+$"}
    source = "return {\"shelf\": \"north\"}\n"
    assert _rule(source, _sig(), schema=schema).passed is True


# --- rule (b): a row id the Starting state holds ---


def test_a_literal_equal_to_a_starting_state_row_id_fails_and_names_the_table():
    schema = EntitySchema(tables=sorted(LIBRARY), columns=_schema().columns)  # no mined id shapes
    source = "return self.db.books[\"BK5109\"]\n"
    result = _rule(source, _sig(), schema=schema)
    assert result.passed is False
    assert "'BK5109'" in result.failures[0] and "row id of books" in result.failures[0]


def test_a_row_id_that_lives_inside_another_table_is_still_a_row_id():
    """A corpus that keeps its rows nested under a parent leaves the child table's own dict empty;
    reading only the top level would find none of the ids a body could memorise."""
    world = {"books": {"BK4821": {"book_id": "BK4821", "chapters": {"CH77": {"chapter_id": "CH77"}}}},
             "chapters": {}}
    schema = EntitySchema(tables=["books", "chapters"], homes={"chapters": "books.chapters"},
                          columns=[Column(table="chapters", name="chapter_id", **{"class": "hard"})])
    result = body_memorised_values_gate("return \"CH77\"\n", schema, world, [], _sig())
    assert result.passed is False and "row id of chapters" in result.failures[0]


# --- rule (c): a value the recorded calls passed, unless the tool itself names it ---


def test_a_literal_equal_to_a_recorded_argument_value_fails_and_names_the_argument():
    schema = EntitySchema(tables=[], columns=[])  # no shapes and no world: only the recordings
    source = "return {\"note\": \"standard\"}\n"
    result = body_memorised_values_gate(source, schema, {}, _calls(), _sig())
    assert result.passed is False
    assert "'standard'" in result.failures[0] and "passed as term" in result.failures[0]


def test_a_recorded_argument_value_the_signatures_enum_lists_is_the_tools_own_vocabulary():
    schema = EntitySchema(tables=[], columns=[])
    sig = _sig(args_schema={"properties": {"term": {"type": "string",
                                                    "enum": ["standard", "short"]}}})
    source = "return {\"note\": \"standard\"}\n"
    assert body_memorised_values_gate(source, schema, {}, _calls(), sig).passed is True


def test_a_recorded_argument_value_the_description_lists_is_allowed_too():
    """A mined signature often carries the enum only in the sentence that introduced it."""
    schema = EntitySchema(tables=[], columns=[])
    sig = _sig(description="Renew one loan. term is standard or short.")
    source = "return {\"note\": \"standard\"}\n"
    assert body_memorised_values_gate(source, schema, {}, _calls(), sig).passed is True


def test_a_value_inside_a_recorded_list_argument_counts_as_a_recorded_value():
    schema = EntitySchema(tables=[], columns=[])
    calls = [ToolCall(id="c0", name="renew_loan", args={"terms": ["standard", "short"]}, raw_ptr=PTR)]
    result = body_memorised_values_gate("return \"short\"\n", schema, {}, calls, _sig())
    assert result.passed is False and "passed as terms" in result.failures[0]


# --- what is code and not data ---


def test_small_integers_short_strings_and_the_empty_string_never_fail():
    schema = EntitySchema(tables=[], columns=[])
    calls = [ToolCall(id="c0", name="renew_loan", args={"loan_id": "LN", "days": 7, "note": ""},
                      raw_ptr=PTR)]
    source = ("total = 0\n"
              "for i in range(7):\n"
              "    total = total + 1\n"
              "return {\"id\": \"LN\", \"note\": \"\", \"total\": total}\n")
    assert body_memorised_values_gate(source, schema, {}, calls, _sig()).passed is True


def test_the_code_owned_docstring_and_the_data_model_are_not_the_models_literals():
    """Only the toolkit's own methods are read: the module skeleton is bytes no model wrote."""
    source = ('class DomainDB:\n'
              '    """LN0031 lives in this docstring and belongs to the skeleton."""\n'
              '    table = "LN0044"\n'
              '\n'
              'class DomainTools:\n'
              '    def __init__(self, db):\n'
              '        self.marker = "LN0031"\n'
              '\n'
              '    def renew_loan(self, loan_id):\n'
              '        """Renew one loan, LN0031 for example."""\n'
              '        return self.db.loans[loan_id]\n')
    assert _rule(source, _sig()).passed is True


def test_dict_keys_count_as_literals():
    """A memorising body writes its table as a dict display more often than any other way; the keys
    of that display are where the recorded ids sit."""
    source = "days = {\"LN0031\": 14, \"LN0044\": 2}\nreturn days[loan_id]\n"
    result = _rule(source, _sig())
    assert result.passed is False
    assert [f for f in result.failures if "'LN0031'" in f]
    assert [f for f in result.failures if "'LN0044'" in f]


def test_body_literals_reads_a_repeated_literal_once_and_in_one_fixed_order():
    """The order is what the failures come out in, so it has to be the same on every run."""
    source = "sizes = {\"north\": 100, \"south\": 100}\nreturn \"north\"\n"
    assert body_literals(source) == ["north", "south", 100]


def test_a_body_that_does_not_parse_is_refused_rather_than_read():
    result = _rule("return self.db.loans[loan_id\n", _sig())
    assert result.passed is False and "does not parse" in result.failures[0]


def test_the_failure_names_the_tool_so_a_red_light_can_be_attributed_to_it():
    result = _rule("return self.db.loans[\"LN0031\"]\n", _sig())
    assert result.passed is False
    assert result.failures[0].startswith("renew_loan: ")
    assert result.metrics["memorised"] == 1


# --- rule (a) again: a pattern that describes no shape is not a shape (D167) ---


def _wildcard_schema() -> EntitySchema:
    """The library's schema, plus a shelves table whose codes the miner could give no shape.

    `mine.id_pattern` falls back to a bare character class when a column's values share nothing, and
    six characters of any kind is what it wrote for one build's ids.
    """
    schema = _schema()
    return schema.model_copy(update={
        "id_patterns": dict(schema.id_patterns, **{"shelves.shelf_code": r"^.{6}$"})})


def test_an_ordinary_word_is_not_a_memorised_id_under_a_wildcard_pattern_of_its_length():
    """D167: the three fixed probes were none of them six letters long, so `^.{6}$` read as a shape
    and the gate refused a dict key, a status word and two place names as memorised ids. Five tools
    stayed assisted for four attempts each on that alone."""
    source = ('totals = {"amount": 0}\n'
              'state = "loaned"\n'
              'return {"branch": "Marlow", "held_at": "Barrow", "state": state, **totals}\n')
    result = _rule(source, _sig(), schema=_wildcard_schema())
    assert result.passed is True, result.failures


def test_a_literal_with_a_real_id_shape_is_still_refused_beside_a_shapeless_pattern():
    """Widening the probes may not blunt the gate: the mined shape that is a shape still rules."""
    source = 'if loan_id == "LN9999":\n    raise ValueError("Loan not found")\nreturn self.db.loans[loan_id]\n'
    result = _rule(source, _sig(), schema=_wildcard_schema())
    assert result.passed is False
    assert "'LN9999'" in result.failures[0] and "loans.loan_id" in result.failures[0]


# --- rule (d): a value the recordings answered, not only one they passed in (D187) ---

def _answering_calls(result: object) -> list[ToolCall]:
    """One recorded call of the renewal tool, with the result the recording says it answered."""
    return [ToolCall(id="c0", name="renew_loan", args={"loan_id": "LN0031", "term": "standard"},
                     result=result, raw_ptr=PTR)]


def test_a_literal_the_recorded_results_carried_is_refused_even_though_no_call_passed_it():
    calls = _answering_calls({"loan_id": "LN0031", "fee": 1275, "term": "standard"})
    body = "def renew_loan(self, loan_id, term):\n    return {'loan_id': loan_id, 'fee': 1275}\n"
    ruling = body_memorised_values_gate(body, _schema(), LIBRARY, calls, _sig(), class_name="none")
    assert ruling.passed is False
    assert "1275" in ruling.failures[0] and "recorded results of renew_loan" in ruling.failures[0]


def test_a_word_the_recorded_results_carried_is_the_tools_vocabulary_and_is_kept():
    calls = _answering_calls({"loan_id": "LN0031", "state": "renewed", "term": "standard"})
    body = "def renew_loan(self, loan_id, term):\n    return {'loan_id': loan_id, 'state': 'renewed'}\n"
    ruling = body_memorised_values_gate(body, _schema(), LIBRARY, calls, _sig(), class_name="none")
    assert ruling.passed is True


def test_a_value_a_reader_reads_out_of_a_prose_result_is_refused_the_same_way():
    reader = ("def read(result):\n"
              "    return {'days_left': int(result.split(' ')[2])}\n")
    readers = load_readers({"proposals": {"borrower": {
        "table": "loans", "columns": ["days_left"],
        "readers": [{"tool": "renew_loan", "source": reader}]}}})
    calls = _answering_calls("Renewed. Now 137 days left")
    body = "def renew_loan(self, loan_id, term):\n    return 'Renewed. Now %d days left' % 137\n"
    kept = body_memorised_values_gate(body, _schema(), LIBRARY, calls, _sig(), class_name="none")
    refused = body_memorised_values_gate(body, _schema(), LIBRARY, calls, _sig(), class_name="none",
                                         readers=readers)
    # 137 is a leaf of no recorded result until the reader is asked what the sentence asserts.
    assert kept.passed is True
    assert refused.passed is False
    assert any("137" in line for line in refused.failures)
