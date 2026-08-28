"""Tests for the Builder's memory: the version tree (D64, D82) and the lessons file (D87)."""

from __future__ import annotations

import inspect
import json
import types

import pytest
from pydantic import ValidationError

from harness.builder import memory
from harness.builder.memory import (
    AnonymizationError,
    LessonError,
    ReadOnlyEditError,
    RetiredLessonError,
    Lesson,
    LessonApplication,
    MemoryConfig,
    Node,
    NodeNotFound,
    OpenProposalError,
    TreeError,
    accept,
    acceptance_rates,
    accepted_single_rounds,
    active_lessons,
    batch_allowed,
    bisect,
    children,
    check_anonymized,
    customer_vocabulary,
    evaluate,
    evidence_in_material,
    grep_nodes,
    head,
    init_tree,
    iter_nodes,
    judge_lessons,
    judge_relevance,
    lessons_path,
    load_lessons,
    load_node,
    load_vocabulary,
    node_path,
    open_proposals,
    path_to_root,
    propose,
    propose_batch,
    rates_are_stable,
    record_application,
    reject,
    restore,
    retire_lesson,
    retirement_candidates,
    save_lesson,
    save_vocabulary,
    snapshot_files,
    tree_dir,
)
from harness.shared.records import FieldStat, ToolSig, content_hash
from harness.shared.report import SetAsideLesson

# --- helpers ---


def _round(workdir, description="edit", prediction="anchor holds", scorecard=None, ok=True, **kw):
    """One full single-change round: propose, evaluate, accept or reject."""
    node = propose(workdir, description, prediction, **kw)
    evaluate(workdir, node, scorecard or {"anchor_pass_rate": 0.9})
    return accept(workdir, node) if ok else reject(workdir, node, reason="worse")


def _lesson(**kw):
    data = {
        "pattern": "a tool that returns a list was mined with a scalar result schema",
        "fix": "take the union of every observed result field before writing the schema",
        "confirming_result": "anchor pass rate rose from 0.71 to 0.86 on the next build",
        "relevance_condition": "applies when a tool returns a list of records",
    }
    data.update(kw)
    return Lesson(**data)


# --- tree layout and round trip ---


def test_init_tree_writes_a_root_node_under_builder_tree(workdir):
    root = init_tree(workdir, files_hash="h0")
    assert root.parent_id is None
    assert root.accepted is True
    assert root.seq == 0
    assert node_path(workdir, root.id).is_file()
    assert node_path(workdir, root.id).parent == tree_dir(workdir) / "nodes"
    assert tree_dir(workdir) == workdir / "builder_tree"


def test_node_round_trips_through_disk(workdir):
    node = propose(workdir, "widen the result schema", "replay fidelity goes up", edit_kind="mine")
    again = load_node(workdir, node.id)
    assert again == node
    assert again.edit_kind == "mine"
    assert again.accepted is None
    stored = json.loads(node_path(workdir, node.id).read_text())
    assert stored["prediction"] == "replay fidelity goes up"


def test_propose_creates_a_child_of_the_head(workdir):
    root = init_tree(workdir, files_hash="h0")
    first = _round(workdir, "one")
    second = propose(workdir, "two", "still holds")
    assert first.parent_id == root.id
    assert second.parent_id == first.id
    assert [n.id for n in path_to_root(workdir, second.id)] == [root.id, first.id, second.id]
    assert [n.id for n in children(workdir, root.id)] == [first.id]


def test_propose_without_init_creates_the_root_itself(workdir):
    node = propose(workdir, "first edit", "nothing breaks")
    root = load_node(workdir, node.parent_id)
    assert root.parent_id is None
    assert head(workdir).id == root.id


def test_node_ids_are_content_addressed_not_time_based(workdir, tmp_path):
    other = tmp_path / "work2"
    other.mkdir()
    a = propose(workdir, "same edit", "same prediction", files_hash="hX")
    b = propose(other, "same edit", "same prediction", files_hash="hX")
    assert a.id == b.id


def test_missing_node_raises(workdir):
    with pytest.raises(NodeNotFound):
        load_node(workdir, "nope")


def test_iter_nodes_is_lazy_and_grep_does_not_load_the_whole_tree(workdir, node_parses):
    _round(workdir, "widen result schema for list tools")
    _round(workdir, "loosen the timestamp canonicalizer")
    node_parses()
    assert isinstance(iter_nodes(workdir), types.GeneratorType)
    next(iter_nodes(workdir))
    assert node_parses() == 1, "iter_nodes read more than the node it yielded"
    hits = grep_nodes(workdir, "canonicaliz")
    assert [n.edit_description for n in hits] == ["loosen the timestamp canonicalizer"]
    assert node_parses() == 1, "grep parsed a node file it did not match"
    assert grep_nodes(workdir, "CANONICALIZ") == hits
    assert grep_nodes(workdir, "nothing here") == []


def test_one_proposal_does_not_read_the_whole_tree(workdir, node_parses):
    """D65: head, the open list and the counters come from the index, not from every node file."""
    for i in range(5):
        _round(workdir, f"round {i}")
    node_parses()
    propose(workdir, "next", "p")
    parsed = node_parses()
    assert parsed <= 3, f"propose parsed {parsed} node files for a tree of six nodes"


def test_iter_nodes_yields_in_seq_order(workdir):
    for i in range(13):
        _round(workdir, f"r{i}")
    seqs = [n.seq for n in iter_nodes(workdir)]
    assert seqs == sorted(seqs) == list(range(len(seqs)))


# --- one change per round (D82) ---


def test_second_open_proposal_is_refused(workdir):
    propose(workdir, "one", "p")
    with pytest.raises(OpenProposalError):
        propose(workdir, "two", "p")


def test_open_proposal_clears_on_accept_and_on_reject(workdir):
    first = propose(workdir, "one", "p")
    evaluate(workdir, first, {"anchor_pass_rate": 0.9})
    accept(workdir, first)
    assert open_proposals(workdir) == []
    second = propose(workdir, "two", "p")
    reject(workdir, second, reason="anchor dropped")
    assert open_proposals(workdir) == []
    propose(workdir, "three", "p")
    assert [n.edit_description for n in open_proposals(workdir)] == ["three"]


def test_evaluate_records_the_scorecard_and_the_outcome(workdir):
    node = propose(workdir, "one", "anchor pass rate rises above 0.8")
    done = evaluate(workdir, node, {"anchor_pass_rate": 0.62}, outcome="it fell instead")
    assert done.scorecard == {"anchor_pass_rate": 0.62}
    assert done.outcome == "it fell instead"
    assert done.accepted is None
    assert load_node(workdir, node.id).scorecard == {"anchor_pass_rate": 0.62}


def test_accept_needs_a_scorecard_but_reject_does_not(workdir):
    node = propose(workdir, "one", "p")
    with pytest.raises(TreeError):
        accept(workdir, node)
    rejected = reject(workdir, node, reason="the gate crashed")
    assert rejected.accepted is False
    assert "the gate crashed" in (rejected.note or "")


def test_accepted_node_becomes_the_head_and_a_rejected_one_does_not(workdir):
    root = init_tree(workdir)
    kept = _round(workdir, "kept")
    assert head(workdir).id == kept.id
    dropped = _round(workdir, "dropped", ok=False)
    assert head(workdir).id == kept.id
    assert dropped.parent_id == kept.id
    assert root.id != kept.id


def test_batch_mode_needs_both_the_config_flag_and_enough_accepted_rounds(workdir):
    config = MemoryConfig(batch_mode=True, single_rounds_before_batch=3)
    assert batch_allowed(workdir, config) is False
    for i in range(2):
        _round(workdir, f"round {i}")
    assert batch_allowed(workdir, config) is False
    propose(workdir, "open one", "p", config=config)
    with pytest.raises(OpenProposalError):
        propose(workdir, "open two", "p", config=config)
    accept(workdir, evaluate(workdir, open_proposals(workdir)[0], {"anchor_pass_rate": 0.9}))
    assert batch_allowed(workdir, config) is True
    batched = propose_batch(workdir, ["batch one", "batch two"], "p", config=config)
    assert batched.batch is True
    assert batched.edits == ["batch one", "batch two"]
    assert len(open_proposals(workdir)) == 1


def test_a_batch_is_one_node_so_it_is_decided_as_a_whole(workdir):
    """D82: no member of a batch can be kept on its own, because there are no members."""
    config = MemoryConfig(batch_mode=True, single_rounds_before_batch=2)
    for i in range(2):
        _round(workdir, f"single {i}")
    batched = propose_batch(workdir, ["a", "b", "c"], "p", config=config)
    with pytest.raises(OpenProposalError):
        propose(workdir, "another edit while the batch is open", "p", config=config)
    accept(workdir, evaluate(workdir, batched, {"anchor_pass_rate": 0.9}))
    assert [n.id for n in path_to_root(workdir, head(workdir).id)][-1] == batched.id
    assert head(workdir).edits == ["a", "b", "c"]


def test_propose_batch_is_refused_before_the_switch_and_for_a_single_edit(workdir):
    config = MemoryConfig(batch_mode=True, single_rounds_before_batch=2)
    with pytest.raises(TreeError):
        propose_batch(workdir, ["a", "b"], "p", config=config)
    for i in range(2):
        _round(workdir, f"single {i}")
    with pytest.raises(TreeError):
        propose_batch(workdir, ["only one"], "p", config=config)
    with pytest.raises(TreeError):
        propose_batch(workdir, ["a", "b"], "p", config=MemoryConfig())


def test_batch_rounds_do_not_count_towards_the_switch(workdir):
    config = MemoryConfig(batch_mode=True, single_rounds_before_batch=2)
    for i in range(2):
        _round(workdir, f"single {i}")
    batched = propose_batch(workdir, ["batched a", "batched b"], "p", config=config)
    accept(workdir, evaluate(workdir, batched, {"anchor_pass_rate": 0.9}))
    assert MemoryConfig().batch_mode is False
    assert batch_allowed(workdir, MemoryConfig(batch_mode=True, single_rounds_before_batch=3)) is False


def test_default_config_never_allows_a_batch(workdir):
    for i in range(25):
        _round(workdir, f"round {i}")
    assert batch_allowed(workdir, MemoryConfig()) is False
    propose(workdir, "open", "p")
    with pytest.raises(OpenProposalError):
        propose(workdir, "second", "p")


# --- lessons file (D87) ---


def test_save_and_load_a_lesson_round_trips_every_field(workdir):
    saved = save_lesson(workdir, _lesson(), vocabulary=["search_products", "orders"])
    assert lessons_path(workdir) == workdir / "lessons.md"
    text = lessons_path(workdir).read_text(encoding="utf-8")
    assert "relevance_condition" in text
    loaded = load_lessons(workdir)
    assert len(loaded) == 1
    assert loaded[0].pattern == saved.pattern
    assert loaded[0].fix == saved.fix
    assert loaded[0].confirming_result == saved.confirming_result
    assert loaded[0].relevance_condition == saved.relevance_condition
    assert loaded[0].id == saved.id
    assert loaded[0].retired is False


def test_two_lessons_append_and_keep_their_order(workdir):
    save_lesson(workdir, _lesson(), vocabulary=[])
    save_lesson(workdir, _lesson(pattern="policy text with 'unless' was compiled as one predicate"),
                vocabulary=[])
    assert [l.pattern[:6] for l in load_lessons(workdir)] == ["a tool", "policy"]


def test_anonymization_gate_rejects_a_customer_tool_name(workdir):
    lesson = _lesson(fix="always call search_products before writing")
    with pytest.raises(AnonymizationError) as err:
        save_lesson(workdir, lesson, vocabulary=["search_products", "orders"])
    assert "search_products" in str(err.value)
    assert not lessons_path(workdir).exists()


def test_anonymization_gate_is_case_insensitive_and_covers_ids_and_tables(workdir):
    vocab = ["orders", "#W0000000", "get_order_details"]
    for bad in ["the Orders table", "row #w0000000 was stale", "Get_Order_Details returned null"]:
        with pytest.raises(AnonymizationError):
            save_lesson(workdir, _lesson(confirming_result=bad), vocabulary=vocab)


def test_anonymization_gate_ignores_very_short_vocabulary_entries(workdir):
    lesson = _lesson(pattern="an id field was treated as a hard column")
    saved = save_lesson(workdir, lesson, vocabulary=["id", "a", "orders"])
    assert saved.pattern.startswith("an id field")


def test_anonymization_gate_reads_every_field_including_applications(workdir):
    lesson = _lesson(applications=[LessonApplication(build_id="b1", outcome="fixed orders replay")])
    with pytest.raises(AnonymizationError):
        save_lesson(workdir, lesson, vocabulary=["orders"])


def test_customer_vocabulary_collects_tool_field_table_and_id_names():
    sigs = [
        ToolSig(name="search_products",
                result_schema=[FieldStat(name="product_sku_code")],
                args_fields=[FieldStat(name="warehouse_zone")],
                args_schema={"properties": {"buyer_tier": {"type": "string"}}}),
        ToolSig(name="cancel_order"),
    ]
    vocab = customer_vocabulary(toolsigs=sigs, tables=["orders", "users"], entity_ids=["#W1"],
                                extra=["gold_member"])
    assert {"search_products", "cancel_order", "orders", "users", "#w1", "gold_member"} <= vocab
    assert {"product_sku_code", "warehouse_zone", "buyer_tier"} <= vocab, \
        "the gate never sees the field names of the customer's tools"
    for named in ["the product_sku_code was empty", "row #W1 was stale", "buyer_tier was null"]:
        assert check_anonymized(_lesson(fix=named), vocab), f"gate passed {named!r}"


def test_record_application_appends_to_the_entry_and_survives_a_reload(workdir):
    saved = save_lesson(workdir, _lesson(), vocabulary=[])
    record_application(workdir, saved.id, build_id="build-2", benefit=True,
                       outcome="anchor pass rate rose by four points", vocabulary=[])
    record_application(workdir, saved.id, build_id="build-3", benefit=False, outcome="no change",
                       vocabulary=[])
    loaded = load_lessons(workdir)[0]
    assert [a.build_id for a in loaded.applications] == ["build-2", "build-3"]
    assert [a.benefit for a in loaded.applications] == [True, False]
    assert loaded.applications[0].outcome == "anchor pass rate rose by four points"


def test_record_application_runs_the_anonymization_gate_too(workdir):
    saved = save_lesson(workdir, _lesson(), vocabulary=[])
    with pytest.raises(AnonymizationError):
        record_application(workdir, saved.id, build_id="b", outcome="orders replay fixed",
                           vocabulary=["orders"])
    assert load_lessons(workdir)[0].applications == []


def test_retire_lesson_marks_it_and_keeps_it_in_the_file(workdir):
    saved = save_lesson(workdir, _lesson(), vocabulary=[])
    retired = retire_lesson(workdir, saved.id, reason="six applications, no confirmed benefit",
                            vocabulary=[])
    assert retired.retired is True
    loaded = load_lessons(workdir)[0]
    assert loaded.retired is True
    assert "no confirmed benefit" in (loaded.retired_reason or "")


def test_record_application_on_an_unknown_lesson_raises(workdir):
    save_lesson(workdir, _lesson(), vocabulary=[])
    with pytest.raises(KeyError):
        record_application(workdir, "not-a-lesson", build_id="b", vocabulary=[])


def test_load_lessons_on_a_missing_file_is_empty(workdir):
    assert load_lessons(workdir) == []


# --- relevance judging (D87) ---


def test_judge_relevance_asks_the_model_once_with_the_condition_and_the_evidence(test_model_factory):
    model = test_model_factory(['{"relevant": true, "evidence": "search_products returns a list"}'])
    sigs = [ToolSig(name="search_products", description="find products")]
    relevant, evidence = judge_relevance(model, _lesson(), toolsigs=sigs,
                                         policy_spans=["never cancel a delivered order"])
    assert relevant is True
    assert evidence == "search_products returns a list"
    assert len(model.calls) == 1
    prompt = json.dumps(model.calls[0]["messages"])
    assert "applies when a tool returns a list of records" in prompt
    assert "search_products" in prompt
    assert "never cancel a delivered order" in prompt


def test_judge_relevance_reads_a_no(test_model_factory):
    model = test_model_factory(['{"relevant": false, "evidence": "no tool returns a list"}'])
    relevant, evidence = judge_relevance(model, _lesson(), toolsigs=[], policy_spans=[])
    assert relevant is False
    assert evidence == "no tool returns a list"


def test_judge_relevance_needs_evidence_before_it_says_yes(test_model_factory):
    model = test_model_factory(['{"relevant": true, "evidence": ""}'])
    relevant, evidence = judge_relevance(model, _lesson(), toolsigs=[], policy_spans=[])
    assert relevant is False
    assert "evidence" in evidence


def test_judge_relevance_sets_aside_an_unparseable_reply(test_model_factory):
    model = test_model_factory(["I am not sure what you mean"])
    relevant, evidence = judge_relevance(model, _lesson(), toolsigs=[], policy_spans=[])
    assert relevant is False
    assert "I am not sure" in evidence


def test_judge_relevance_accepts_a_quoted_policy_span_and_refuses_a_pointer_to_one(test_model_factory):
    spans = [{"text": "refund only within 30 days"}]
    model = test_model_factory(['{"relevant": true, "evidence": "refund only within 30 days"}'])
    relevant, _ = judge_relevance(model, _lesson(), toolsigs=[], policy_spans=spans)
    assert relevant is True
    assert "refund only within 30 days" in json.dumps(model.calls[0]["messages"])
    pointer = test_model_factory(['{"relevant": true, "evidence": "policy line 3"}'])
    relevant, reason = judge_relevance(pointer, _lesson(), toolsigs=[], policy_spans=spans)
    assert relevant is False
    assert "quotes nothing" in reason


def test_judge_relevance_never_builds_its_own_model(test_model_factory):
    """The module takes a Model and never constructs one (build brief rule 2)."""
    source = inspect.getsource(memory)
    assert "provider" not in source
    for name in ("Model(", "AnthropicModel", "OpenAIModel", "TestModel"):
        assert name not in source
    model = test_model_factory(['{"relevant": true, "evidence": "get_user"}'])
    relevant, evidence = judge_relevance(model, _lesson(), toolsigs=[ToolSig(name="get_user")],
                                         policy_spans=[])
    assert (relevant, evidence) == (True, "get_user")
    assert len(model.calls) == 1


# --- shapes ---


def test_node_and_lesson_carry_the_fields_the_tree_and_the_gate_need(workdir):
    node = Node(id="n1", parent_id=None, seq=0, files_hash="h", edit_description="d",
                edits=["d"], edit_paths=["builder/mine.py"], prediction="p", scorecard={"x": 1})
    assert Node.model_validate(json.loads(node.model_dump_json())) == node
    assert node.accepted is None and node.batch is False and node.batch_id is None
    assert node.files_dir is None
    assert content_hash(node) == content_hash(node.model_copy(deep=True))
    assert content_hash(node) != content_hash(node.model_copy(update={"scorecard": {"x": 2}}))
    lesson = _lesson(applications=[LessonApplication(build_id="b", benefit=True)])
    assert Lesson.model_validate(json.loads(lesson.model_dump_json())) == lesson
    assert Lesson.model_validate(json.loads(lesson.model_dump_json())).applications[0].benefit is True
    with pytest.raises(ValidationError):
        Node.model_validate({"id": "n2", "unknown_field": 1})


@pytest.fixture
def test_model_factory(make_test_model):
    return make_test_model


@pytest.fixture
def node_parses(monkeypatch):
    """Counts how many node files were parsed since the last call (D65: never the whole tree)."""
    count = {"n": 0}
    original = memory.Node.model_validate_json

    def counting(text, *args, **kwargs):
        count["n"] += 1
        return original(text, *args, **kwargs)

    monkeypatch.setattr(memory.Node, "model_validate_json", counting)

    def since():
        seen = count["n"]
        count["n"] = 0
        return seen

    return since


# --- what the improvement agent may not touch (D69, design section 4 item 21) ---


def test_proposal_that_touches_a_read_only_file_is_refused(workdir):
    for described in ["edit runner/loop.py to relax the stop rule",
                      "tweak route.py", "raise the ceiling in budget.py",
                      "reorder the checks in verdict.py", "loosen validate.py",
                      "delete a Run from the runs directory", "rewrite the evaluator"]:
        with pytest.raises(ReadOnlyEditError):
            propose(workdir, described, "faster")
    with pytest.raises(ReadOnlyEditError):
        propose(workdir, "widen the schema", "p", edit_paths=["src/harness/runner/loop.py"])
    with pytest.raises(ReadOnlyEditError):
        propose(workdir, "widen the schema", "p", edit_kind="loop.py")
    assert open_proposals(workdir) == []


def test_the_read_only_list_is_the_one_the_design_names(workdir):
    assert set(memory.READ_ONLY_PATHS) == {
        "runner/loop.py", "runner/route.py", "runner/verdict.py", "runner/validate.py",
        "shared/budget.py", "runs/", "data/raw/",
    }
    assert memory.read_only_hits("edit builder/mine.py") == []
    assert memory.read_only_hits(["src/harness/runner/route.py"]) == ["route.py"]


def test_an_edit_under_the_builder_directory_is_allowed(workdir):
    node = propose(workdir, "widen the result schema in mine.py", "p",
                   edit_paths=["src/harness/builder/mine.py"], edit_kind="mine")
    assert node.edit_paths == ["src/harness/builder/mine.py"]


# --- the evaluator's record is the evaluator's (D64) ---


def test_evaluate_cannot_rewrite_a_node_the_evaluator_decided(workdir):
    kept = _round(workdir, "kept", scorecard={"anchor_pass_rate": 0.9})
    with pytest.raises(TreeError):
        evaluate(workdir, kept, {"anchor_pass_rate": 0.1}, outcome="rewritten after acceptance")
    assert load_node(workdir, kept.id).scorecard == {"anchor_pass_rate": 0.9}
    dropped = _round(workdir, "dropped", ok=False)
    with pytest.raises(TreeError):
        evaluate(workdir, dropped, {"anchor_pass_rate": 0.99})


def test_propose_refuses_a_rejected_or_still_open_parent(workdir):
    init_tree(workdir, files_hash="h0")
    bad = _round(workdir, "bad edit", ok=False)
    with pytest.raises(TreeError):
        propose(workdir, "child of a rejected edit", "p", parent_id=bad.id)
    still_open = propose(workdir, "open", "p")
    with pytest.raises(TreeError):
        propose(workdir, "child of an open edit", "p", parent_id=still_open.id)


def test_the_accepted_lineage_stays_one_chain(workdir):
    """Head's path to root is every accepted edit, so it describes the files the Builder has."""
    first = _round(workdir, "first")
    second = propose(workdir, "second", "p")
    reject(workdir, second, reason="worse")
    sibling = propose(workdir, "sibling of the rejected one", "p")
    evaluate(workdir, sibling, {"anchor_pass_rate": 0.9})
    accept(workdir, sibling)
    chain = [n.edit_description for n in path_to_root(workdir, head(workdir).id)]
    assert chain == ["root", "first", "sibling of the rejected one"]
    another = propose(workdir, "another sibling", "p", parent_id=first.id)
    evaluate(workdir, another, {"anchor_pass_rate": 0.9})
    with pytest.raises(TreeError):
        accept(workdir, another)


# --- bisection of a rejected batch (D82) ---


def test_a_rejected_batch_is_bisected_into_halves(workdir):
    config = MemoryConfig(batch_mode=True, single_rounds_before_batch=2)
    for i in range(2):
        _round(workdir, f"single {i}")
    parent = head(workdir)
    batched = propose_batch(workdir, ["a", "b", "c", "d"], "p", config=config, edit_kind="mine")
    reject(workdir, batched, reason="anchor dropped")
    halves = bisect(workdir, batched)
    assert [h.edits for h in halves] == [["a", "b"], ["c", "d"]]
    assert all(h.parent_id == parent.id for h in halves)
    assert all(h.batch_id == batched.id for h in halves)
    assert all(h.accepted is None for h in halves)
    assert {h.id for h in halves} == {n.id for n in open_proposals(workdir)}
    evaluate(workdir, halves[0], {"anchor_pass_rate": 0.9})
    accept(workdir, halves[0])
    assert head(workdir).edits == ["a", "b"]


def test_bisect_refuses_an_undecided_node_and_a_single_edit(workdir):
    node = _round(workdir, "one edit", ok=False)
    with pytest.raises(TreeError):
        bisect(workdir, node)
    still_open = propose(workdir, "open", "p")
    with pytest.raises(TreeError):
        bisect(workdir, still_open)


# --- the switch rule's second half (D82) ---


def test_batch_needs_steady_per_edit_kind_acceptance_rates(workdir):
    config = MemoryConfig(batch_mode=True, single_rounds_before_batch=2,
                          stability_window=4, max_rate_drift=0.25)
    for i in range(2):
        _round(workdir, f"mine {i}", edit_kind="mine")
    for i in range(2):
        _round(workdir, f"policy {i}", edit_kind="mine", ok=False)
    assert accepted_single_rounds(workdir) == 2
    assert acceptance_rates(workdir, window=4) == {"mine": 0.5}
    assert rates_are_stable(workdir, config) is False, "the rate for 'mine' fell from 1.0 to 0.0"
    assert batch_allowed(workdir, config) is False
    for i in range(4):
        _round(workdir, f"steady {i}", edit_kind="mine")
    assert rates_are_stable(workdir, config) is True
    assert batch_allowed(workdir, config) is True


# --- a node's files, not only a hash (design section 4 item 21) ---


def test_a_node_snapshots_the_builder_files_and_can_restore_them(workdir, tmp_path):
    builder = tmp_path / "builder"
    (builder / "prompts").mkdir(parents=True)
    (builder / "mine.py").write_text("first version", encoding="utf-8")
    (builder / "prompts" / "compile.md").write_text("the prompt", encoding="utf-8")
    node = propose(workdir, "widen the result schema", "p", files_dir=builder)
    assert node.files_dir and (tree_dir(workdir) / node.files_dir / "mine.py").is_file()
    assert node.files_hash and node.files_hash in node.files_dir
    (builder / "mine.py").write_text("edited and worse", encoding="utf-8")
    restored = restore(workdir, node, builder)
    assert (builder / "mine.py").read_text(encoding="utf-8") == "first version"
    assert (builder / "prompts" / "compile.md").read_text(encoding="utf-8") == "the prompt"
    assert len(restored) == 2
    same, _ = snapshot_files(workdir, builder)
    assert same == node.files_hash
    with pytest.raises(TreeError):
        restore(workdir, init_tree(workdir), builder)


def test_an_edit_node_never_claims_its_parents_files_hash(workdir):
    root = init_tree(workdir, files_hash="h0")
    node = propose(workdir, "change mine.py", "p")
    assert node.files_hash != root.files_hash
    assert node.files_hash == ""
    reject(workdir, node, reason="worse")
    given = _round(workdir, "measured", files_hash="h1")
    assert given.files_hash == "h1"


def test_init_tree_refuses_a_second_root_with_a_different_files_hash(workdir):
    root = init_tree(workdir, files_hash="h0")
    assert init_tree(workdir, files_hash="h0").id == root.id
    assert init_tree(workdir).id == root.id
    with pytest.raises(TreeError):
        init_tree(workdir, files_hash="h1")


def test_path_to_root_refuses_a_cycle(workdir):
    init_tree(workdir, files_hash="h0")
    memory._write_node(workdir, Node(id="loop", parent_id="loop", seq=1))
    with pytest.raises(TreeError):
        path_to_root(workdir, "loop")


# --- the anonymization gate is on by default (D87) ---


def test_save_lesson_without_a_vocabulary_is_refused_not_waved_through(workdir):
    with pytest.raises(AnonymizationError) as err:
        save_lesson(workdir, _lesson(fix="call search_products first"))
    assert "vocabulary" in str(err.value)
    assert not lessons_path(workdir).exists()
    saved = save_lesson(workdir, _lesson(), vocabulary=[])
    with pytest.raises(AnonymizationError):
        record_application(workdir, saved.id, build_id="b")
    with pytest.raises(AnonymizationError):
        retire_lesson(workdir, saved.id, reason="no benefit")
    assert load_lessons(workdir)[0].applications == []


def test_a_stored_vocabulary_runs_the_gate_without_being_passed(workdir):
    save_vocabulary(workdir, ["search_products", "orders"])
    assert load_vocabulary(workdir) == {"search_products", "orders"}
    with pytest.raises(AnonymizationError):
        save_lesson(workdir, _lesson(fix="call search_products first"))
    saved = save_lesson(workdir, _lesson())
    with pytest.raises(AnonymizationError):
        record_application(workdir, saved.id, build_id="b", outcome="orders replay fixed")
    with pytest.raises(AnonymizationError):
        retire_lesson(workdir, saved.id, reason="the orders tool went away")


def test_the_gate_catches_hyphen_space_and_camel_case_spellings(workdir):
    for bad in ["call search-products first", "the search products tool ran",
                "searchProducts returned nothing"]:
        assert check_anonymized(_lesson(fix=bad), ["search_products"]) == ["search_products"]
        with pytest.raises(AnonymizationError):
            save_lesson(workdir, _lesson(fix=bad), vocabulary=["search_products"])
    assert check_anonymized(_lesson(fix="the list result was empty"), ["search_products"]) == []


def test_the_gate_covers_short_entity_ids(workdir):
    vocab = customer_vocabulary(entity_ids=["#W1"])
    assert check_anonymized(_lesson(fix="row #W1 was stale"), vocab) == ["#w1"]


def test_the_gate_covers_the_lesson_id(workdir):
    with pytest.raises(AnonymizationError):
        save_lesson(workdir, _lesson(id="search_products"), vocabulary=["search_products"])
    assert not lessons_path(workdir).exists()


def test_the_gate_covers_the_retirement_reason(workdir):
    saved = save_lesson(workdir, _lesson(), vocabulary=[])
    with pytest.raises(AnonymizationError):
        retire_lesson(workdir, saved.id, reason="the orders table went away", vocabulary=["orders"])
    assert load_lessons(workdir)[0].retired is False


def test_a_lesson_without_a_relevance_condition_is_refused(workdir):
    for empty in [{"relevance_condition": ""}, {"pattern": " "}, {"fix": ""}]:
        with pytest.raises(LessonError):
            save_lesson(workdir, _lesson(**empty), vocabulary=[])


# --- retirement belongs to the evaluator (D87) ---


def test_re_saving_a_lesson_does_not_un_retire_it_or_drop_its_applications(workdir):
    saved = save_lesson(workdir, _lesson(), vocabulary=[])
    record_application(workdir, saved.id, "b1", benefit=False, outcome="no change", vocabulary=[])
    retire_lesson(workdir, saved.id, reason="no benefit", vocabulary=[])
    again = save_lesson(workdir, _lesson(fix="a different fix"), vocabulary=[])
    loaded = load_lessons(workdir)
    assert len(loaded) == 1
    assert again.id == saved.id
    assert loaded[0].fix == "a different fix"
    assert loaded[0].retired is True
    assert [a.build_id for a in loaded[0].applications] == ["b1"]


def test_retirement_candidates_are_the_ones_with_applications_and_no_benefit(workdir):
    dead = save_lesson(workdir, _lesson(), vocabulary=[])
    alive = save_lesson(workdir, _lesson(pattern="policy text with 'unless' became one predicate"),
                        vocabulary=[])
    for i in range(3):
        record_application(workdir, dead.id, f"b{i}", benefit=False, vocabulary=[])
    record_application(workdir, alive.id, "b9", benefit=True, vocabulary=[])
    assert [l.id for l in retirement_candidates(workdir, min_applications=3)] == [dead.id]
    assert retirement_candidates(workdir, min_applications=4) == []
    with pytest.raises(LessonError):
        retire_lesson(workdir, alive.id, reason="not paying off", vocabulary=[])
    retire_lesson(workdir, alive.id, reason="the evaluator overrode it", vocabulary=[], force=True)
    assert load_lessons(workdir)[1].retired is True


def test_a_retired_lesson_takes_no_new_applications_and_is_not_active(workdir):
    saved = save_lesson(workdir, _lesson(), vocabulary=[])
    retire_lesson(workdir, saved.id, reason="six applications, no benefit", vocabulary=[])
    with pytest.raises(RetiredLessonError):
        record_application(workdir, saved.id, "b9", benefit=True, vocabulary=[])
    assert active_lessons(workdir) == []
    assert load_lessons(workdir)[0].applications == []
    assert active_lessons(load_lessons(workdir)) == []


# --- the judge's answer goes through a code gate (D87) ---


def test_a_retired_lesson_is_never_judged_relevant_and_costs_no_model_call(test_model_factory):
    model = test_model_factory(['{"relevant": true, "evidence": "get_user"}'])
    relevant, reason = judge_relevance(model, _lesson(retired=True, retired_reason="no benefit"),
                                       toolsigs=[ToolSig(name="get_user")], policy_spans=[])
    assert relevant is False
    assert reason == "retired: no benefit"
    assert model.calls == []


def test_a_lesson_with_no_relevance_condition_is_set_aside_without_a_model_call(test_model_factory):
    model = test_model_factory(['{"relevant": true, "evidence": "get_user"}'])
    relevant, reason = judge_relevance(model, _lesson(relevance_condition=""),
                                       toolsigs=[ToolSig(name="get_user")], policy_spans=[])
    assert relevant is False
    assert "relevance condition" in reason
    assert model.calls == []


def test_evidence_that_quotes_nothing_from_the_material_is_refused(test_model_factory):
    model = test_model_factory(['{"relevant": true, "evidence": "trust me, it applies"}'])
    sigs = [ToolSig(name="get_user", description="fetch one user")]
    relevant, reason = judge_relevance(model, _lesson(), toolsigs=sigs,
                                       policy_spans=["never refund twice"])
    assert relevant is False
    assert "quotes nothing" in reason
    assert evidence_in_material("trust me, it applies", sigs, ["never refund twice"]) is False
    assert evidence_in_material("get_user returns one record", sigs, []) is True
    assert evidence_in_material("never refund twice", [], ["never refund twice"]) is True


def test_evidence_may_quote_a_result_field_name(test_model_factory):
    sigs = [ToolSig(name="get_user", result_schema=[FieldStat(name="membership_tier")])]
    model = test_model_factory(['{"relevant": true, "evidence": "membership_tier is a list"}'])
    relevant, _ = judge_relevance(model, _lesson(), toolsigs=sigs, policy_spans=[])
    assert relevant is True


def test_only_a_json_true_counts_as_relevant(test_model_factory):
    for answer in ['"false"', '"true"', '1', 'null', '"yes"']:
        model = test_model_factory(['{"relevant": %s, "evidence": "get_user"}' % answer])
        relevant, reason = judge_relevance(model, _lesson(), toolsigs=[ToolSig(name="get_user")],
                                           policy_spans=[])
        assert relevant is False, f"relevant: {answer} was read as true"
        if answer != '"false"':
            assert "non-boolean" in reason


def test_a_set_aside_lesson_always_carries_a_reason(test_model_factory):
    model = test_model_factory(['{"relevant": false}'])
    relevant, reason = judge_relevance(model, _lesson(), toolsigs=[], policy_spans=[])
    assert relevant is False
    assert reason.strip()


def test_judge_lessons_splits_applied_from_set_aside_for_the_report(workdir, test_model_factory):
    keep = _lesson()
    drop = _lesson(pattern="policy text with 'unless' became one predicate",
                   relevance_condition="applies when policy text contains 'unless'")
    retired = _lesson(pattern="a third pattern", retired=True, retired_reason="no benefit")
    model = test_model_factory([
        '{"relevant": true, "evidence": "search_products returns a list"}',
        '{"relevant": false, "evidence": "no policy text says unless"}',
    ])
    sigs = [ToolSig(name="search_products")]
    applied, set_aside = judge_lessons(model, [keep, drop, retired], toolsigs=sigs, policy_spans=[])
    assert [l.pattern for l in applied] == [keep.pattern]
    assert all(isinstance(s, SetAsideLesson) for s in set_aside)
    assert [s.reason for s in set_aside] == ["no policy text says unless", "retired: no benefit"]
    assert all(s.reason for s in set_aside)
    assert len(model.calls) == 2
