"""Tests for builder/cluster.py: Categories by write-tool signature, Tasks by intent similarity."""

from __future__ import annotations

import itertools
import json

import pytest

from conftest import PTR
from kullback.builder.cluster import (
    category_signature,
    cluster_runs,
    confirmed_write_calls,
    idf_weights,
    name_task,
    run_tokens,
    similarity,
    tokens,
    write_tool_names,
)
from kullback.runner.records import ToolCall, ToolCallError, ToolSig, Trace, Turn

SIGS = [
    ToolSig(name="cancel_order", kind="write"),
    ToolSig(name="modify_address", kind="write"),
    ToolSig(name="get_order", kind="read"),
]
WRITES = {"cancel_order", "modify_address"}


def make_trace(trace_id: str, user_turns: list[str], calls: list[dict]) -> Trace:
    turns = [Turn(idx=i, role="user", content=c, raw_ptr=PTR) for i, c in enumerate(user_turns)]
    tool_calls = []
    for i, call in enumerate(calls):
        error = None
        if call.get("error"):
            error = ToolCallError(**{"class": "business_error", "payload": call["error"]})
        tool_calls.append(
            ToolCall(
                id=f"{trace_id}-{i}",
                name=call["name"],
                args=call.get("args", {}),
                result=call.get("result"),
                error=error,
                requestor=call.get("requestor", "assistant"),
                raw_ptr=PTR,
            )
        )
    return Trace(
        trace_id=trace_id,
        raw_hash="raw",
        ingest_version="0",
        source="tau2",
        turns=turns,
        tool_calls=tool_calls,
        raw_ptr=PTR,
    )


def cancel_trace(trace_id: str, order: str, reason: str = "delivery was late") -> Trace:
    return make_trace(
        trace_id,
        [f"i want to cancel order {order}", f"cancel it, the {reason}"],
        [
            {"name": "get_order", "args": {"order_id": order}, "result": {"status": "pending"}},
            {"name": "cancel_order", "args": {"order_id": order, "reason": reason}, "result": {"status": "cancelled"}},
        ],
    )


def address_trace(trace_id: str, order: str) -> Trace:
    return make_trace(
        trace_id,
        [f"please change the shipping address on order {order}", "the new street number is 42"],
        [{"name": "modify_address", "args": {"order_id": order, "street": "42 elm"}, "result": {"ok": True}}],
    )


# --- write tools and the Category signature ---


def test_write_tool_names_from_tool_sigs():
    assert write_tool_names(SIGS) == WRITES


def test_write_tool_names_accepts_names_a_mapping_or_nothing():
    assert write_tool_names(["cancel_order"]) == {"cancel_order"}
    assert write_tool_names({"cancel_order": "write", "get_order": "read"}) == {"cancel_order"}
    assert write_tool_names(None) == set()


def test_category_signature_is_the_confirmed_write_tools_only():
    trace = cancel_trace("t1", "W1")
    assert category_signature(trace, WRITES) == ("cancel_order",)


def test_a_user_requestor_write_call_does_not_count_towards_the_signature():
    """Telecom's simulated user runs its own tool calls against its own phone
    (docs/cross-domain-check.md, Judgement); those must not enter the write signature. Retail and
    airline never carry a user-requestor call, so this is new coverage, not a change to their result."""
    trace = make_trace(
        "t1",
        ["cancel order W1"],
        [
            {"name": "cancel_order", "args": {"order_id": "W1"}, "result": {"status": "cancelled"},
             "requestor": "user"},
            {"name": "modify_address", "args": {"order_id": "W1"}, "result": {"ok": True}},
        ],
    )
    assert [c.name for c in confirmed_write_calls(trace, WRITES)] == ["modify_address"]
    assert category_signature(trace, WRITES) == ("modify_address",)


def test_a_failed_write_call_does_not_count_towards_the_signature():
    trace = make_trace(
        "t1",
        ["cancel order W1"],
        [{"name": "cancel_order", "args": {"order_id": "W1"}, "error": "order already delivered"}],
    )
    assert confirmed_write_calls(trace, WRITES) == []
    assert category_signature(trace, WRITES) == ()


def test_same_write_set_is_one_category_different_write_sets_are_two():
    traces = [cancel_trace("t1", "W1"), cancel_trace("t2", "W2"), address_trace("t3", "W3")]
    categories, tasks = cluster_runs(traces, SIGS)
    assert len(categories) == 2
    by_tools = {tuple(c.write_tools): c for c in categories}
    assert set(by_tools) == {("cancel_order",), ("modify_address",)}
    cancel_runs = {rid for t in tasks if t.category_id == by_tools[("cancel_order",)].id for rid in t.run_ids}
    assert cancel_runs == {"t1", "t2"}


def test_every_task_is_listed_by_its_category():
    traces = [cancel_trace("t1", "W1"), address_trace("t2", "W2")]
    categories, tasks = cluster_runs(traces, SIGS)
    listed = [tid for c in categories for tid in c.task_ids]
    assert sorted(listed) == sorted(t.id for t in tasks)
    for task in tasks:
        assert task.id in {tid for c in categories if c.id == task.category_id for tid in c.task_ids}


# --- similarity and Task membership ---


def test_similarity_is_token_jaccard_and_zero_when_a_side_is_empty():
    assert similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert similarity({"a", "b", "c", "d"}, {"a", "e", "f", "g"}) == pytest.approx(1 / 7)
    assert similarity(set(), {"a"}) == 0.0
    assert similarity({"a"}, set()) == 0.0
    assert similarity(set(), set()) == 0.0


def test_similarity_weighted_by_idf_counts_a_rare_token_above_a_common_one():
    """A token every Run says weighs almost nothing; a token one Run says weighs the most (D83)."""
    weights = idf_weights([{"help", f"rare_{i}"} for i in range(20)])
    assert weights["rare_0"] > 3 * weights["help"]
    shared_boilerplate = similarity({"help", "rare_0"}, {"help", "rare_1"}, weights)
    shared_rare = similarity({"help", "rare_0"}, {"help", "rare_0", "rare_1"}, weights)
    assert shared_boilerplate < 0.2
    assert shared_rare > 0.5
    # unweighted, the two look the same, which is what makes the raw bag unsound
    assert similarity({"help", "rare_0"}, {"help", "rare_1"}) == pytest.approx(1 / 3)


def test_a_three_run_corpus_weights_almost_evenly_so_the_rule_still_works_small():
    """The smoothing has to degrade to plain Jaccard, or every small corpus falls apart."""
    weights = idf_weights([{"a", "b"}, {"a", "c"}, {"a", "d"}])
    assert 1.0 <= weights["a"] < weights["b"] < 1.7
    assert similarity({"a", "b"}, {"a", "c"}, weights) == pytest.approx(0.23, abs=0.03)


def test_run_tokens_use_the_first_two_user_turns_and_not_the_write_arg_keys():
    """The arg keys are the same for every Run in a Category, so they say nothing about intent."""
    trace = make_trace(
        "t1",
        ["cancel my order", "it was late", "third turn ignored"],
        [{"name": "cancel_order", "args": {"order_id": "W1"}, "result": {}}],
    )
    toks = run_tokens(trace)
    assert "cancel" in toks and "late" in toks
    assert "third" not in toks and "ignored" not in toks
    assert not any(t.startswith("arg:") for t in toks)


def test_two_runs_with_no_shared_user_words_are_two_tasks():
    """They wrote through the same tool with the same argument keys, and that is not a shared intent."""
    args = {"order_id": "W1", "reason": "asked"}
    t1 = make_trace("t1", ["swap the jacket for a bigger size"], [{"name": "cancel_order", "args": args, "result": {}}])
    t2 = make_trace(
        "t2", ["wrong colour on my boots, need black"], [{"name": "cancel_order", "args": args, "result": {}}]
    )
    assert run_tokens(t1) & run_tokens(t2) == set()
    _, tasks = cluster_runs([t1, t2], SIGS)
    assert sorted(sorted(t.run_ids) for t in tasks) == [["t1"], ["t2"]]


def test_a_contraction_is_one_token_and_no_letter_fragments_survive():
    assert tokens("I don't remember, I'd like to swap it") == ["remember", "like", "swap"]
    assert tokens("I don’t remember") == ["remember"]
    assert "t" not in tokens("don't") and "d" not in tokens("I'd rather")


def test_words_every_run_says_do_not_merge_two_tasks():
    """Politeness and authentication chatter is in every Run, so it must not decide membership (D83)."""
    boiler = "hello i need help with my account, another way to look up my email"
    traces = [
        make_trace(
            f"t{i}", [f"{boiler}, {extra}"], [{"name": "cancel_order", "args": {"order_id": "W1"}, "result": {}}]
        )
        for i, extra in enumerate(
            [
                "the jacket arrived damaged",
                "the damaged jacket must go back",
                "i ordered two sofas by mistake",
                "the boots are the wrong size",
                "my flight was moved so i do not need the tent",
                "the price dropped the day after",
                "the colour is nothing like the photo",
                "i found the same speaker cheaper",
            ]
        )
    ]
    _, tasks = cluster_runs(traces, SIGS)
    membership = sorted(sorted(t.run_ids) for t in tasks)
    assert ["t0", "t1"] in membership  # the two that share "damaged jacket"
    assert all(len(m) <= 2 for m in membership), membership


def test_similar_runs_share_a_task_and_a_dissimilar_run_is_its_own_task():
    traces = [
        cancel_trace("t1", "W1"),
        cancel_trace("t2", "W2"),
        make_trace(
            "t3",
            ["the jacket arrived broken", "i changed my mind about the colour anyway"],
            [{"name": "cancel_order", "args": {"order_id": "W9"}, "result": {}}],
        ),
    ]
    _, tasks = cluster_runs(traces, SIGS)
    membership = sorted(sorted(t.run_ids) for t in tasks)
    assert membership == [["t1", "t2"], ["t3"]]


def test_a_high_threshold_splits_every_run_into_its_own_task():
    traces = [cancel_trace("t1", "W1"), cancel_trace("t2", "W2")]
    assert sorted(sorted(t.run_ids) for t in cluster_runs(traces, SIGS)[1]) == [["t1", "t2"]]
    _, tasks = cluster_runs(traces, SIGS, threshold=0.99)
    assert sorted(sorted(t.run_ids) for t in tasks) == [["t1"], ["t2"]]


# --- D81 unguarded ---


def test_a_task_with_fewer_than_three_runs_is_unguarded():
    traces = [cancel_trace("t1", "W1"), cancel_trace("t2", "W2")]
    _, tasks = cluster_runs(traces, SIGS)
    assert [t.unguarded for t in tasks] == [True]


def test_a_task_with_three_runs_is_guarded():
    traces = [cancel_trace(f"t{i}", f"W{i}") for i in range(3)]
    _, tasks = cluster_runs(traces, SIGS)
    assert len(tasks) == 1
    assert tasks[0].unguarded is False
    assert tasks[0].run_ids == ["t0", "t1", "t2"]


# --- determinism ---


UNEVEN_CHAIN = [  # A and B overlap more than B and C do, and A and C not at all
    "alpha beta gamma delta epsilon",
    "gamma delta epsilon zeta eta",
    "zeta eta theta iota kappa",
]
EVEN_CHAIN = [  # A to B is worth exactly what B to C is worth, so only the words can break the tie
    "alpha beta gamma delta",
    "gamma delta epsilon zeta",
    "epsilon zeta eta theta",
]


def chain_grouping(names: tuple[str, str, str], texts: list[str]) -> list[list[str]]:
    traces = [
        make_trace(name, [text], [{"name": "cancel_order", "args": {"order_id": "W1"}, "result": {}}])
        for name, text in zip(names, texts, strict=False)
    ]
    label = dict(zip(names, "ABC", strict=False))
    _, tasks = cluster_runs(traces, SIGS, threshold=0.3)
    return sorted(sorted(label[r] for r in task.run_ids) for task in tasks)


@pytest.mark.parametrize("texts", [UNEVEN_CHAIN, EVEN_CHAIN])
def test_membership_does_not_depend_on_how_the_trace_ids_sort(texts):
    """Leader clustering gave {A,B},{C} or {A,B,C} or {A},{B,C} for the same three Runs (D83, section 8)."""
    first = chain_grouping(("a", "b", "c"), texts)
    assert first == chain_grouping(("b", "a", "c"), texts)
    assert first == chain_grouping(("c", "b", "a"), texts)
    assert first == chain_grouping(("m", "z", "a"), texts)
    assert first == [["A", "B"], ["C"]]


def test_ids_and_membership_do_not_depend_on_input_order():
    traces = [cancel_trace("t1", "W1"), address_trace("t2", "W2"), cancel_trace("t3", "W3")]
    first = cluster_runs(traces, SIGS)
    second = cluster_runs(list(reversed(traces)), SIGS)
    assert [c.model_dump() for c in first[0]] == [c.model_dump() for c in second[0]]
    assert [t.model_dump() for t in first[1]] == [t.model_dump() for t in second[1]]


def test_a_category_id_is_the_same_for_the_same_write_set_in_another_corpus():
    one, _ = cluster_runs([cancel_trace("t1", "W1")], SIGS)
    two, _ = cluster_runs([cancel_trace("z9", "W7")], SIGS)
    assert one[0].id == two[0].id


def test_no_traces_gives_no_categories_and_no_tasks():
    assert cluster_runs([], SIGS) == ([], [])


def test_runs_with_no_confirmed_writes_still_form_a_category():
    traces = [make_trace("t1", ["where is my order"], [{"name": "get_order", "args": {"order_id": "W1"}}])]
    categories, tasks = cluster_runs(traces, SIGS)
    assert categories[0].write_tools == []
    assert tasks[0].run_ids == ["t1"]


# --- the naming hook only names ---


def test_name_task_returns_one_trimmed_line_from_the_model(make_test_model):
    model = make_test_model(["Cancel a late order\nand nothing else"])
    assert name_task(model, [cancel_trace("t1", "W1")]) == "Cancel a late order"
    assert len(model.calls) == 1


def test_name_task_without_a_model_returns_none():
    assert name_task(None, [cancel_trace("t1", "W1")]) is None


def test_naming_changes_only_the_name(make_test_model):
    traces = [cancel_trace("t1", "W1"), cancel_trace("t2", "W2")]
    _, unnamed = cluster_runs(traces, SIGS)
    _, named = cluster_runs(traces, SIGS, model=make_test_model(["Cancel a late order"], loop=True))
    assert [t.name for t in named] == ["Cancel a late order"]
    assert [t.model_dump(exclude={"name"}) for t in named] == [t.model_dump(exclude={"name"}) for t in unnamed]


def test_the_naming_prompt_carries_the_runs_own_words(make_test_model):
    model = make_test_model(["Cancel a late order"])
    name_task(model, [cancel_trace("t1", "W1")])
    prompt = model.calls[0]["messages"][-1]["content"]
    assert "cancel order W1" in prompt
    assert "cancel_order" in prompt


def test_an_empty_model_reply_leaves_the_name_unset(make_test_model):
    assert name_task(make_test_model(["   \n  "]), [cancel_trace("t1", "W1")]) is None


# --- the real tau2 traces (no ingest here: the messages are read as the raw file stores them) ---

TAU2_WRITES = {"exchange_delivered_order_items", "return_delivered_order_items"}


def tau2_traces(tau2_small) -> list[Trace]:
    out = []
    for sim in tau2_small["simulations"]:
        messages = sim["messages"]
        out.append(
            Trace(
                trace_id=sim["id"][:8],
                raw_hash="raw",
                ingest_version="0",
                source="tau2",
                turns=[Turn(idx=i, role=m["role"], content=m.get("content"), raw_ptr=PTR)
                       for i, m in enumerate(messages)],
                tool_calls=[
                    ToolCall(id=tc.get("id"), name=tc["name"], args=tc.get("arguments") or {}, raw_ptr=PTR)
                    for m in messages
                    for tc in (m.get("tool_calls") or [])
                ],
                raw_ptr=PTR,
            )
        )
    return out


def test_tau2_runs_split_into_an_exchange_category_and_a_return_category(tau2_small):
    categories, _ = cluster_runs(tau2_traces(tau2_small), TAU2_WRITES)
    assert sorted(tuple(c.write_tools) for c in categories) == [
        ("exchange_delivered_order_items",),
        ("return_delivered_order_items",),
    ]


def test_the_three_tau2_runs_are_three_tasks_as_tau2_says_they_are(tau2_small):
    """tau2 calls them tasks 6, 2 and 1; the two exchanges differ (bottle and lamp, keyboard).

    They used to merge, at a threshold picked so that they would, because they share the
    authentication chatter every Run of this corpus repeats: "another way", "look up", "email".
    """
    assert sorted(s["task_id"] for s in tau2_small["simulations"]) == ["1", "2", "6"]
    traces = {t.trace_id: t for t in tau2_traces(tau2_small)}
    shared = run_tokens(traces["4bec2b80"]) & run_tokens(traces["526bdc8f"])
    assert {"another", "way", "look", "email"} <= shared  # the words that used to merge them
    _, tasks = cluster_runs(list(traces.values()), TAU2_WRITES)
    membership = sorted(sorted(t.run_ids) for t in tasks)
    assert membership == [["4bec2b80"], ["526bdc8f"], ["ee6707bb"]]
    assert all(t.unguarded for t in tasks)


# --- the whole corpus, when it is on disk (the raw traces are never committed) ---

# tau2 retail's write tools, from vendor/tau2-bench retail tools.py; mine.py derives the same set.
TAU2_RETAIL_WRITES = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}


def corpus_traces_and_truth(raw_dir):
    """Traces straight off a raw file, with tau2's own task_id kept aside as the test's ground truth.

    The task_id is a grader field (D66): it is the yardstick this test measures against and is never
    handed to `cluster_runs`, which sees only turns and tool calls.
    """
    paths = sorted(raw_dir.glob("*retail*4trials.json"))
    if not paths:
        pytest.skip("no tau2 retail raw file")
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    traces, truth = [], {}
    for sim in data["simulations"]:
        messages = sim["messages"]
        failed = {m.get("id") for m in messages if m.get("role") == "tool" and m.get("error")}
        traces.append(
            Trace(
                trace_id=sim["id"],
                raw_hash="raw",
                ingest_version="0",
                source="tau2",
                turns=[Turn(idx=i, role=m["role"], content=m.get("content"), raw_ptr=PTR)
                       for i, m in enumerate(messages)],
                tool_calls=[
                    ToolCall(
                        id=tc.get("id"),
                        name=tc["name"],
                        args=tc.get("arguments") or {},
                        error=(
                            ToolCallError(**{"class": "business_error", "payload": ""})
                            if tc.get("id") in failed
                            else None
                        ),
                        raw_ptr=PTR,
                    )
                    for m in messages
                    for tc in (m.get("tool_calls") or [])
                ],
                raw_ptr=PTR,
            )
        )
        truth[sim["id"]] = str(sim["task_id"])
    return traces, truth


def pair_f1(tasks, truth) -> float:
    """Pair-counting F1 of the Task membership against tau2's task_ids."""
    task_of = {run_id: task.id for task in tasks for run_id in task.run_ids}
    tp = fp = fn = 0
    for a, b in itertools.combinations(sorted(truth), 2):
        same_task, same_truth = task_of[a] == task_of[b], truth[a] == truth[b]
        tp += same_task and same_truth
        fp += same_task and not same_truth
        fn += same_truth and not same_task
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def test_the_whole_corpus_clusters_close_to_tau2s_own_tasks(raw_dir):
    """The rule has to hold on 456 Runs, not only on three: pair F1 against tau2's task_ids."""
    traces, truth = corpus_traces_and_truth(raw_dir)
    assert len(traces) > 100
    _, tasks = cluster_runs(traces, TAU2_RETAIL_WRITES)
    assert pair_f1(tasks, truth) >= 0.65


def test_the_corpus_score_does_not_hang_on_the_exact_threshold(raw_dir):
    """The shipped rule scored 0.28 at its own default and 0.72 at 0.6; that cliff is what idf removes."""
    traces, truth = corpus_traces_and_truth(raw_dir)
    scores = {th: pair_f1(cluster_runs(traces, TAU2_RETAIL_WRITES, threshold=th)[1], truth) for th in (0.3, 0.5)}
    assert min(scores.values()) >= 0.65, scores


# --- D74: one starting world per Task ---------------------------------------

def _bare_trace(trace_id: str) -> Trace:
    """The record split_by_world is handed by the pipeline; it reads only the trace_id."""
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="1", source="tau2", raw_ptr=PTR)


def test_runs_that_saw_a_row_in_two_versions_before_writing_are_different_tasks():
    from kullback.builder.cluster import split_by_world

    worlds = {"a": {("orders", "o1"): "v_pending"}, "b": {("orders", "o1"): "v_pending", ("users", "u1"): "v1"},
              "c": {("orders", "o1"): "v_delivered"}, "d": {}}
    traces = [_bare_trace(t) for t in ("c", "a", "d", "b")]
    parts = split_by_world(traces, worlds)
    assert [[t.trace_id for t in part] for part in parts] == [["a", "b", "d"], ["c"]]


def test_a_group_with_no_world_information_stays_one_task():
    from kullback.builder.cluster import split_by_world

    parts = split_by_world([_bare_trace("b"), _bare_trace("a")], {})
    assert [[t.trace_id for t in p] for p in parts] == [["a", "b"]]
