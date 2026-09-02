"""Tests for builder/policy.py: sentence spans, predicate compilation, the sandbox, the D76 order."""

from __future__ import annotations

import json

import pytest

from kullback.ai.provider import TestModel
from kullback.builder.policy import (
    HELPERS_SRC,
    accept_rewrite,
    compile_policy,
    compile_rule,
    pending_review,
    reference_violations,
    reject_rule,
    residual,
    run_constraint_tests,
    split_policy,
)
from kullback.runner.records import Constraint, ConstraintTests, Event, Run

# --- the three real retail policy sentences the module is exercised on ---

CONFIRM = "explicit user confirmation"
AUTH = "authenticate the user identity"
MAKEUP = "should not make up any information"


CONFIRM_PRED = '''WRITE_TOOLS = ("cancel_pending_order", "modify_pending_order_address", "return_delivered_order_items")


def check(pre_state, write_call, transcript):
    if write_call.get("name") not in WRITE_TOOLS:
        return True
    return user_confirmed(transcript)
'''

CONFIRM_TESTS = {
    "pos": [
        {
            "pre_state": {},
            "write_call": {"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}},
            "transcript": [
                {"role": "assistant", "content": "I will cancel order #W1 for a refund. Shall I proceed?"},
                {"role": "user", "content": "yes please"},
            ],
        }
    ],
    "neg": [
        {
            "pre_state": {},
            "write_call": {"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}},
            "transcript": [{"role": "user", "content": "how much do I get back for it?"}],
        }
    ],
}

AUTH_BAD_PRED = '''def check(pre_state, write_call, transcript):
    return called_before(transcript, "find_user_id_by_email")
'''

AUTH_GOOD_PRED = '''LOOKUPS = ("find_user_id_by_email", "find_user_id_by_name_zip")


def check(pre_state, write_call, transcript):
    if write_call.get("name") in LOOKUPS:
        return True
    return called_before(transcript, *LOOKUPS)
'''

AUTH_TESTS = {
    "pos": [
        {
            "pre_state": {},
            "write_call": {"name": "get_order_details", "arguments": {"order_id": "#W1"}},
            "transcript": [
                {"role": "assistant", "tool_calls": [{"name": "find_user_id_by_name_zip", "arguments": {}}]}
            ],
        }
    ],
    "neg": [
        {
            "pre_state": {},
            "write_call": {"name": "get_order_details", "arguments": {"order_id": "#W1"}},
            "transcript": [],
        }
    ],
}


def _reply(obj) -> str:
    return json.dumps(obj)


def _compiled_reply(src, tests) -> str:
    return _reply({"compilable": True, "predicate_src": src, "tests": tests})


def _sentence(policy_md, needle):
    for item in split_policy(policy_md):
        if needle in item.text:
            return item
    raise AssertionError(f"sentence not found: {needle}")


@pytest.fixture(scope="session")
def policy_md(tau2_retail_dir) -> str:
    return (tau2_retail_dir / "policy.md").read_text(encoding="utf-8")


# --- split_policy ---


def test_split_policy_finds_the_three_sentences(policy_md):
    texts = [s.text for s in split_policy(policy_md)]
    assert any(CONFIRM in t for t in texts)
    assert any(AUTH in t for t in texts)
    assert any(MAKEUP in t for t in texts)
    assert len(texts) > 30


def test_split_policy_spans_point_back_at_the_source(policy_md):
    for item in split_policy(policy_md):
        assert 0 <= item.start < item.end <= len(policy_md)
        assert policy_md[item.start : item.end].split() == item.text.split()
        assert item.span().file_hash == item.file_hash


def test_split_policy_drops_headings_and_keeps_the_section(policy_md):
    items = split_policy(policy_md)
    assert not any(s.text.startswith("#") for s in items)
    assert any(s.section == "User" for s in items)
    confirm = _sentence(policy_md, CONFIRM)
    assert confirm.section == "Retail agent policy"  # the H1, the last heading above it


def test_split_policy_keeps_one_sentence_per_rule(policy_md):
    confirm = _sentence(policy_md, CONFIRM)
    assert confirm.text.startswith("Before taking any action")
    assert confirm.text.endswith("to proceed.")
    assert "e.g." not in confirm.text


def test_split_policy_does_not_break_on_abbreviations_or_times(policy_md):
    texts = [s.text for s in split_policy(policy_md)]
    assert any("e.g." in t and len(t.split()) > 6 for t in texts)
    assert not any(t.strip() in {"g.", "e.", "30:00\" means 2:30 AM EST."} for t in texts)


def test_split_policy_drops_a_marker_with_no_rule_in_it():
    """An inline numbered list must not leave a constraint whose whole text is "2."."""
    assert [s.text for s in split_policy("1. First rule. 2. Second rule.")] == ["First rule.", "Second rule."]


def test_split_policy_ends_a_sentence_after_a_closing_quote():
    items = split_policy('The user must say "yes." Then the agent may proceed.')
    assert [s.text for s in items] == ['The user must say "yes."', "Then the agent may proceed."]


def test_split_policy_takes_a_plain_string():
    items = split_policy("One thing. Two things happen here.\n\n- A bullet rule.\n")
    assert [s.text for s in items] == ["One thing.", "Two things happen here.", "A bullet rule."]
    assert items[2].start == len("One thing. Two things happen here.\n\n- ")


# --- run_constraint_tests ---


def _constraint(src, tests=None, cid="c1") -> Constraint:
    return Constraint(
        id=cid,
        text="a rule",
        predicate_src=src,
        tests=ConstraintTests(**(tests or CONFIRM_TESTS)),
    )


def test_run_constraint_tests_passes_a_good_predicate():
    gate = run_constraint_tests(_constraint(CONFIRM_PRED))
    assert gate.passed is True
    assert gate.failures == []
    assert gate.metrics["pos"] == 1 and gate.metrics["neg"] == 1
    assert gate.metrics["ok"] == 2


def test_run_constraint_tests_fails_when_the_negative_case_is_allowed():
    """The sequence rule must actually read the transcript, not return a constant."""
    tests = json.loads(json.dumps(CONFIRM_TESTS))
    tests["neg"][0]["transcript"] = [
        {"role": "assistant", "content": "Shall I cancel order #W1?"},
        {"role": "user", "content": "yes, go ahead"},
    ]
    gate = run_constraint_tests(_constraint(CONFIRM_PRED, tests))
    assert gate.passed is False
    assert any("neg[0]" in f for f in gate.failures)


def test_run_constraint_tests_needs_both_a_positive_and_a_negative_case():
    gate = run_constraint_tests(_constraint(CONFIRM_PRED, {"pos": CONFIRM_TESTS["pos"], "neg": []}))
    assert gate.passed is False
    assert any("negative" in f for f in gate.failures)


def test_run_constraint_tests_refuses_an_import():
    src = "import os\n\n\ndef check(pre_state, write_call, transcript):\n    return os.getcwd() != ''\n"
    gate = run_constraint_tests(_constraint(src))
    assert gate.passed is False
    assert any("import" in f for f in gate.failures)
    assert gate.metrics["ran"] == 0


def test_run_constraint_tests_refuses_open_and_dunders():
    src = "def check(pre_state, write_call, transcript):\n    return open('/etc/passwd').read() != ''\n"
    assert any("open" in f for f in run_constraint_tests(_constraint(src)).failures)
    src2 = "def check(pre_state, write_call, transcript):\n    return pre_state.__class__ is dict\n"
    assert any("__class__" in f for f in run_constraint_tests(_constraint(src2)).failures)


def test_run_constraint_tests_refuses_a_frame_walk_that_carries_no_denied_name(tmp_path):
    """The certification is an allowlist: a bypass with no import, no dunder and no denied name is still refused."""
    marker = tmp_path / "escaped.marker"
    src = (
        "def check(pre_state, write_call, transcript):\n"
        "    def gen():\n"
        "        yield gen_obj.gi_frame.f_back.f_back\n"
        "    gen_obj = gen()\n"
        "    for outer in gen_obj:\n"
        "        break\n"
        "    real = outer.f_builtins\n"
        f"    with real['open']({str(marker)!r}, 'w') as fh:\n"
        "        fh.write('escaped')\n"
        "    return bool(write_call)\n"
    )
    gate = run_constraint_tests(_constraint(src))
    assert gate.passed is False
    assert gate.metrics["ran"] == 0, "the predicate must never reach the sandbox"
    assert not marker.exists()
    assert any("gi_frame" in f for f in gate.failures)


def test_run_constraint_tests_refuses_an_attribute_outside_the_allowlist():
    src = "def check(pre_state, write_call, transcript):\n    return pre_state.setdefault('a', 1) == 1\n"
    assert any("setdefault" in f for f in run_constraint_tests(_constraint(src)).failures)


def test_run_constraint_tests_refuses_a_name_it_cannot_account_for():
    src = "def check(pre_state, write_call, transcript):\n    return bool(helper_that_was_never_defined)\n"
    gate = run_constraint_tests(_constraint(src))
    assert gate.passed is False
    assert any("helper_that_was_never_defined" in f for f in gate.failures)


def test_run_constraint_tests_keeps_the_module_constants_a_real_predicate_needs():
    """The allowlist must not refuse the ordinary shape: a module tuple, the three parameters, a helper."""
    assert run_constraint_tests(_constraint(CONFIRM_PRED)).passed is True
    assert run_constraint_tests(_constraint(AUTH_GOOD_PRED, AUTH_TESTS)).passed is True


def test_run_constraint_tests_ignores_an_expect_the_model_put_in_its_own_case():
    """A constant predicate must not pass by declaring the positive case false (the gate owns expect)."""
    src = "def check(pre_state, write_call, transcript):\n    return False\n"
    tests = {
        "pos": [{"write_call": {"name": "cancel_pending_order"}, "expect": False}],
        "neg": [{"write_call": {"name": "cancel_pending_order"}}],
    }
    gate = run_constraint_tests(_constraint(src, tests))
    assert gate.passed is False
    assert any("pos[0]" in f for f in gate.failures)


def test_run_constraint_tests_refuses_a_missing_check_function():
    src = "def other(a, b, c):\n    return True\n"
    assert any("check" in f for f in run_constraint_tests(_constraint(src)).failures)


def test_run_constraint_tests_reports_a_syntax_error():
    gate = run_constraint_tests(_constraint("def check(pre_state, write_call transcript):\n    return True\n"))
    assert gate.passed is False
    assert any("parse" in f for f in gate.failures)


def test_run_constraint_tests_reports_a_raising_predicate():
    src = "def check(pre_state, write_call, transcript):\n    return write_call['nope'] == 1\n"
    gate = run_constraint_tests(_constraint(src))
    assert gate.passed is False
    assert any("KeyError" in f for f in gate.failures)


def test_run_constraint_tests_kills_a_loop():
    src = "def check(pre_state, write_call, transcript):\n    while True:\n        pass\n"
    gate = run_constraint_tests(_constraint(src), timeout_s=1.0)
    assert gate.passed is False
    assert any("timed out" in f for f in gate.failures)


def test_run_constraint_tests_says_so_with_no_predicate():
    gate = run_constraint_tests(Constraint(id="c0", text="a vague rule"))
    assert gate.passed is False
    assert any("predicate" in f for f in gate.failures)


def _helpers() -> dict:
    namespace: dict = {}
    exec(HELPERS_SRC, namespace)  # noqa: S102, the helpers are our own source, not the model's
    return namespace


def test_helpers_are_available_to_the_predicate():
    gate = run_constraint_tests(_constraint(AUTH_GOOD_PRED, AUTH_TESTS))
    assert gate.passed is True
    tests = json.loads(json.dumps(AUTH_TESTS))
    tests["pos"][0]["transcript"] = [{"role": "assistant", "tool_calls": [{"name": "get_order_details"}]}]
    assert run_constraint_tests(_constraint(AUTH_GOOD_PRED, tests)).passed is False, "called_before must read names"
    said_before = _helpers()["said_before"]
    turns = [{"role": "assistant", "content": "Your refund goes to the ORIGINAL card."}]
    assert said_before(turns, "original card") is True
    assert said_before(turns, "gift card") is False
    assert said_before([{"role": "user", "content": "original card"}], "original card") is False


def test_user_confirmed_needs_a_yes_to_something_that_was_proposed():
    """D43 case 3: a Hard constraint that reads the transcript must not read a refusal as consent."""
    user_confirmed = _helpers()["user_confirmed"]
    ask = {"role": "assistant", "content": "I will cancel order #W1 for a refund. Shall I proceed?"}

    def reply(text):
        return user_confirmed([ask, {"role": "user", "content": text}])

    assert reply("yes please") is True
    assert reply("Go ahead.") is True
    assert reply("No, do not proceed.") is False
    assert reply("That is not correct, please don't.") is False
    assert reply("No thanks, it's ok as it is.") is False
    assert reply("Hmm, which order is that?") is False
    # an opening turn nobody asked for is not consent
    assert user_confirmed([{"role": "user", "content": "Yes hello, I want to ask about my order."}]) is False
    # the proposal has to come before the yes, and it has to be a proposal
    assert user_confirmed([{"role": "assistant", "content": "I cancelled it."},
                           {"role": "user", "content": "yes"}]) is False
    assert user_confirmed([]) is False


# --- compile_rule ---


def test_compile_rule_compiles_a_sequence_rule(policy_md):
    sentence = _sentence(policy_md, CONFIRM)
    model = TestModel([_compiled_reply(CONFIRM_PRED, CONFIRM_TESTS)])
    constraint = compile_rule(model, sentence)
    assert constraint.compiled is True
    assert constraint.judge_atom is False
    assert constraint.residual_reason is None
    assert "user_confirmed(transcript)" in constraint.predicate_src
    assert len(constraint.tests.pos) == 1 and len(constraint.tests.neg) == 1
    assert constraint.span.file_hash == sentence.file_hash
    assert constraint.span_text == sentence.text
    assert len(model.calls) == 1


def test_compile_rule_ids_are_stable_and_content_addressed(policy_md):
    sentence = _sentence(policy_md, CONFIRM)
    first = compile_rule(TestModel([_compiled_reply(CONFIRM_PRED, CONFIRM_TESTS)]), sentence)
    second = compile_rule(TestModel([_compiled_reply(CONFIRM_PRED, CONFIRM_TESTS)]), sentence)
    other = compile_rule(TestModel([_compiled_reply(AUTH_GOOD_PRED, AUTH_TESTS)]), _sentence(policy_md, AUTH))
    assert first.id == second.id
    assert first.id != other.id


def test_compile_rule_rewrites_when_the_first_predicate_fails_its_tests(policy_md):
    """D76: a rule that does not compile is rewritten for review, not silently dropped."""
    sentence = _sentence(policy_md, AUTH)
    rewritten = (
        "Before any tool call other than a user lookup, the transcript must already contain "
        "a call to find_user_id_by_email or find_user_id_by_name_zip."
    )
    model = TestModel(
        [
            _compiled_reply(AUTH_BAD_PRED, AUTH_TESTS),
            _reply({"rewritten_text": rewritten, "predicate_src": AUTH_GOOD_PRED, "tests": AUTH_TESTS}),
        ]
    )
    constraint = compile_rule(model, sentence)
    assert constraint.compiled is False, "a rewrite waits for the setup review"
    assert constraint.judge_atom is False
    assert constraint.rewritten_text == rewritten
    assert constraint.text == sentence.text, "the original stays beside the rewrite"
    assert constraint.predicate_src == AUTH_GOOD_PRED.strip()
    assert len(model.calls) == 2


def test_the_rewrite_call_keeps_the_first_two_messages_byte_identical(policy_md):
    """docs/prompt-caching.md item 2: the rewrite is a retry, not a fresh ask, so the system
    message (_CONTRACT plus the policy) and the first user turn must be the same bytes both calls."""
    sentence = _sentence(policy_md, AUTH)
    model = TestModel(
        [
            _compiled_reply(AUTH_BAD_PRED, AUTH_TESTS),
            _reply({"rewritten_text": "checkable form", "predicate_src": AUTH_GOOD_PRED, "tests": AUTH_TESTS}),
        ]
    )
    compile_rule(model, sentence, policy_text="Authenticate the user first.")
    assert len(model.calls) == 2
    first_two = model.calls[0]["messages"][:2]
    assert model.calls[1]["messages"][:2] == first_two
    assert len(model.calls[1]["messages"]) > 2


def test_accept_rewrite_turns_it_into_a_normal_constraint(policy_md):
    sentence = _sentence(policy_md, AUTH)
    model = TestModel(
        [
            _compiled_reply(AUTH_BAD_PRED, AUTH_TESTS),
            _reply({"rewritten_text": "checkable form", "predicate_src": AUTH_GOOD_PRED, "tests": AUTH_TESTS}),
        ]
    )
    accepted = accept_rewrite(compile_rule(model, sentence))
    assert accepted.compiled is True
    assert accepted.judge_atom is False
    assert accepted.rewritten_text == "checkable form"


def test_accept_rewrite_falls_back_to_a_judge_atom_when_the_predicate_stops_working(policy_md):
    sentence = _sentence(policy_md, AUTH)
    constraint = compile_rule(
        TestModel(
            [
                _compiled_reply(AUTH_BAD_PRED, AUTH_TESTS),
                _reply({"rewritten_text": "checkable form", "predicate_src": AUTH_GOOD_PRED, "tests": AUTH_TESTS}),
            ]
        ),
        sentence,
    )
    constraint.tests.pos[0]["transcript"] = []  # the reviewer edited the case out from under it
    accepted = accept_rewrite(constraint)
    assert accepted.compiled is False
    assert accepted.judge_atom is True


def test_compile_rule_makes_a_judge_atom_when_the_rule_must_stay_natural_language(policy_md):
    sentence = _sentence(policy_md, MAKEUP)
    model = TestModel(
        [
            _reply({"compilable": False, "reason": "no state or transcript test decides made-up knowledge"}),
            _reply({"judge_atom": True, "rewritten_text": "State no fact that no tool result supports."}),
        ]
    )
    constraint = compile_rule(model, sentence)
    assert constraint.compiled is False
    assert constraint.judge_atom is True
    assert constraint.predicate_src is None
    assert constraint.rewritten_text == "State no fact that no tool result supports."
    assert len(model.calls) == 2


def test_compile_rule_makes_a_judge_atom_when_the_rewrite_has_no_predicate(policy_md):
    sentence = _sentence(policy_md, MAKEUP)
    model = TestModel([_reply({"compilable": False}), _reply({"rewritten_text": "still vague"})])
    constraint = compile_rule(model, sentence)
    assert constraint.judge_atom is True
    assert constraint.compiled is False


def test_compile_rule_makes_a_judge_atom_when_the_rewrites_predicate_also_fails(policy_md):
    sentence = _sentence(policy_md, AUTH)
    model = TestModel(
        [
            _compiled_reply(AUTH_BAD_PRED, AUTH_TESTS),
            _reply({"rewritten_text": "second try", "predicate_src": AUTH_BAD_PRED, "tests": AUTH_TESTS}),
        ]
    )
    constraint = compile_rule(model, sentence)
    assert constraint.judge_atom is True
    assert constraint.compiled is False
    assert constraint.predicate_src is None
    assert constraint.rewritten_text == "second try"


def test_unparseable_model_output_is_a_residual_not_a_judge_atom(policy_md):
    """D76: a judge atom is a decision about the rule; an unreadable Builder reply is not that decision."""
    sentence = _sentence(policy_md, CONFIRM)
    constraint = compile_rule(TestModel(["not json at all", "still not json"]), sentence)
    assert constraint.compiled is False
    assert constraint.judge_atom is False
    assert constraint.rewritten_text is None
    assert "unparseable" in (constraint.residual_reason or "")
    assert residual([constraint]) == [constraint]


def test_a_rewrite_with_neither_a_predicate_nor_a_rewritten_text_is_a_residual(policy_md):
    sentence = _sentence(policy_md, CONFIRM)
    constraint = compile_rule(TestModel([_reply({"compilable": False}), _reply({"reason": "cannot"})]), sentence)
    assert constraint.judge_atom is False
    assert constraint.residual_reason is not None


def test_compile_rule_reads_json_out_of_a_fenced_block(policy_md):
    sentence = _sentence(policy_md, CONFIRM)
    fenced = "```json\n" + _compiled_reply(CONFIRM_PRED, CONFIRM_TESTS) + "\n```"
    assert compile_rule(TestModel([fenced]), sentence).compiled is True


def test_compile_rule_puts_a_model_failure_in_residual(policy_md):
    sentence = _sentence(policy_md, CONFIRM)
    constraint = compile_rule(TestModel([]), sentence)  # empty TestModel replies with ""
    assert constraint.judge_atom is False
    assert constraint.residual_reason is not None

    class Broken(TestModel):
        def query(self, messages, tools=None, config=None):
            raise RuntimeError("provider down")

    broken = compile_rule(Broken([]), sentence)
    assert broken.compiled is False
    assert broken.judge_atom is False
    assert "provider down" in broken.residual_reason


# --- residual list ---


def test_reject_rule_goes_to_residual_and_never_into_a_verdict(policy_md):
    sentence = _sentence(policy_md, CONFIRM)
    constraint = compile_rule(TestModel([_compiled_reply(CONFIRM_PRED, CONFIRM_TESTS)]), sentence)
    rejected = reject_rule(constraint, "the reviewer says this is not our rule")
    assert rejected.compiled is False
    assert rejected.judge_atom is False
    assert rejected.predicate_src is None
    assert rejected.residual_reason == "the reviewer says this is not our rule"
    assert residual([constraint, rejected]) == [rejected]
    assert pending_review([constraint, rejected]) == []


def test_pending_review_lists_a_rewrite_nobody_has_accepted_yet(policy_md):
    """D76 and D48: until the reviewer accepts it, a rewrite is checked by nobody and must be reported as such."""
    sentence = _sentence(policy_md, AUTH)
    model = TestModel(
        [
            _compiled_reply(AUTH_BAD_PRED, AUTH_TESTS),
            _reply({"rewritten_text": "lookup first", "predicate_src": AUTH_GOOD_PRED, "tests": AUTH_TESTS}),
        ]
    )
    constraint = compile_rule(model, sentence)
    assert constraint.compiled is False and constraint.judge_atom is False
    assert residual([constraint]) == [], "it is not residual, it is waiting"
    assert pending_review([constraint]) == [constraint]
    assert pending_review([accept_rewrite(constraint)]) == []
    assert pending_review([reject_rule(constraint, "no")]) == []


# --- the Reference's own path (compile policy gate, second half) ---


def _run(pairs) -> Run:
    return Run(run_id="ref", events=[Event(idx=i, type=t, payload=p) for i, (t, p) in enumerate(pairs)])


CONFIRM_ONLY_PRED = (
    "def check(pre_state, write_call, transcript):\n"
    "    if write_call.get('name') != 'cancel_pending_order':\n"
    "        return True\n"
    "    return user_confirmed(transcript)\n"
)


def test_reference_violations_finds_a_rule_the_gold_path_itself_breaks():
    rule = Constraint(id="k1", text="never cancel without confirmation", compiled=True,
                      predicate_src=CONFIRM_ONLY_PRED)
    unconfirmed = _run([
        ("user_turn", {"content": "cancel #W1"}),
        ("tool_call", {"name": "cancel_pending_order", "args": {"order_id": "#W1"}}),
        ("stop", {"start_state": {}, "end_state": {}}),
    ])
    hits = reference_violations([rule], [unconfirmed])
    assert [(h["run_id"], h["constraint_id"], h["tool"]) for h in hits] == [
        ("ref", "k1", "cancel_pending_order")
    ]


def test_reference_violations_reads_the_transcript_before_the_write_only():
    """A confirmation given after the write is not a prior confirmation (D43 case 3)."""
    rule = Constraint(id="k1", text="never cancel without confirmation", compiled=True,
                      predicate_src=CONFIRM_ONLY_PRED)
    before = _run([
        ("model_call", {"content": "I will cancel order #W1. Shall I proceed?"}),
        ("user_turn", {"content": "yes"}),
        ("tool_call", {"name": "cancel_pending_order", "args": {"order_id": "#W1"}}),
        ("stop", {"start_state": {}, "end_state": {}}),
    ])
    after = _run([
        ("user_turn", {"content": "cancel #W1"}),
        ("tool_call", {"name": "cancel_pending_order", "args": {"order_id": "#W1"}}),
        ("model_call", {"content": "Cancelled. Was that right?"}),
        ("user_turn", {"content": "yes"}),
        ("stop", {"start_state": {}, "end_state": {}}),
    ])
    assert reference_violations([rule], [before]) == []
    assert len(reference_violations([rule], [after])) == 1


def test_reference_violations_skips_rules_that_are_not_compiled_code():
    judged = Constraint(id="k2", text="be polite", judge_atom=True)
    pending = Constraint(id="k3", text="lookup first", rewritten_text="lookup first",
                         predicate_src=CONFIRM_ONLY_PRED)
    run = _run([
        ("tool_call", {"name": "cancel_pending_order", "args": {"order_id": "#W1"}}),
        ("stop", {"start_state": {}, "end_state": {}}),
    ])
    assert reference_violations([judged, pending], [run]) == []


# --- compile_policy over the three sentences ---


def test_compile_policy_returns_a_compiled_a_rewritten_and_a_judge_constraint_one_per_sentence(policy_md):
    sentences = [_sentence(policy_md, n) for n in (CONFIRM, AUTH, MAKEUP)]
    model = TestModel(
        [
            _compiled_reply(CONFIRM_PRED, CONFIRM_TESTS),
            _compiled_reply(AUTH_BAD_PRED, AUTH_TESTS),
            _reply({"rewritten_text": "lookup first", "predicate_src": AUTH_GOOD_PRED, "tests": AUTH_TESTS}),
            _reply({"compilable": False, "reason": "not decidable by code"}),
            _reply({"judge_atom": True, "rewritten_text": "State no unsupported fact."}),
        ]
    )
    constraints = compile_policy(model, sentences, timeout_s=5.0)
    assert [c.compiled for c in constraints] == [True, False, False]
    assert [c.judge_atom for c in constraints] == [False, False, True]
    assert constraints[1].rewritten_text == "lookup first"
    assert residual(constraints) == []
    assert len({c.id for c in constraints}) == 3


def test_compile_policy_accepts_raw_markdown_and_a_limit(policy_md):
    model = TestModel([_compiled_reply(CONFIRM_PRED, CONFIRM_TESTS)], loop=True)
    constraints = compile_policy(model, policy_md, limit=2)
    sentences = split_policy(policy_md)[:2]
    assert len(constraints) == 2
    assert [c.text for c in constraints] == [s.text for s in sentences]
    assert [c.span.msg_index for c in constraints] == [s.index for s in sentences]
    assert all(c.span.file_hash == sentences[0].file_hash for c in constraints)
    assert len(model.calls) == 2, "one model call per sentence, in order"
