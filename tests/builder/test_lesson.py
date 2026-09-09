"""What a stalled tool body is told: the full diff, the relation across failing calls, the lines
no recorded call reaches, and the change of strategy past the stall limit (D211).

The domain is invented: a depot that files parcels into crates and quotes a haulage price. Nothing
here names a corpus, a customer tool or a customer column.
"""

from __future__ import annotations

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.builder import lesson
from kullback.runner.records import Column, EntitySchema, FieldStat, ToolCall, ToolSig


def _triple(call_id, args=None, theirs=None, ours=None, their_error="", our_error="", matched=False):
    return lesson.Triple(call_id=call_id, args=args or {}, theirs=theirs, ours=ours,
                         their_error=their_error, our_error=our_error, matched=matched)


# --- 1. the full diff ---------------------------------------------------------------


def test_the_diff_lists_every_leaf_two_answers_part_on_not_only_the_first():
    theirs = {"crates": [{"depot": "north", "weight": 4}, {"depot": "north", "weight": 4}]}
    ours = {"crates": [{"depot": "north", "weight": 4}, {"depot": "south", "weight": 9}]}
    leaves, dropped = lesson.differing_leaves(theirs, ours)
    assert dropped == 0
    assert [where for where, _, _ in leaves] == ["crates[1].depot", "crates[1].weight"]
    assert leaves[0][1:] == ('"north"', '"south"')


def test_the_diff_caps_the_leaves_it_carries_and_says_how_many_it_left_out():
    theirs = {f"slot_{i}": i for i in range(12)}
    ours = {f"slot_{i}": i + 1 for i in range(12)}
    leaves, dropped = lesson.differing_leaves(theirs, ours, cap=4)
    assert len(leaves) == 4 and dropped == 8
    line = lesson.full_diff_line(_triple("c1", theirs=theirs, ours=ours), cap=4)
    assert "parts at 12 leaves" in line and "8 more leaves differ" in line


def test_a_call_the_two_sides_answered_alike_leaves_no_diff_line():
    same = {"crates": [{"depot": "north"}]}
    assert lesson.full_diff_line(_triple("c1", theirs=same, ours=same)) == ""


def test_a_call_one_side_refused_is_reported_as_a_refusal_and_not_as_leaves():
    line = lesson.full_diff_line(_triple("c1", theirs=None, ours={"ok": True}, their_error="NoSuchCrate"))
    assert "the recording raised NoSuchCrate" in line and "the body answered" in line


# --- 2. the relation step -------------------------------------------------------------


def _broadcast_triple(call_id):
    """The recording puts the first parcel's collection day on every parcel; ours works one out each."""
    theirs = {"parcels": [{"day": "monday", "code": "a"}, {"day": "monday", "code": "b"}]}
    ours = {"parcels": [{"day": "monday", "code": "a"}, {"day": "friday", "code": "b"}]}
    return _triple(call_id, args={"crate": "k1"}, theirs=theirs, ours=ours)


def test_broadcast_is_found_when_the_recording_carries_one_element_value_on_every_element():
    found = lesson.relations_over([_broadcast_triple("c1"), _broadcast_triple("c2")])
    broadcast = [r for r in found if r.kind == "broadcast"]
    assert [r.where for r in broadcast] == ["parcels.day"]
    assert broadcast[0].on_all and broadcast[0].of == 2
    assert "first element" in broadcast[0].detail
    assert "holds on all 2 failing calls" in broadcast[0].sentence()


def test_broadcast_is_absent_when_the_two_sides_vary_together():
    theirs = {"parcels": [{"day": "monday"}, {"day": "friday"}]}
    ours = {"parcels": [{"day": "monday"}, {"day": "sunday"}]}
    found = lesson.relations_over([_triple("c1", theirs=theirs, ours=ours)])
    assert not [r for r in found if r.kind == "broadcast"]


def test_shared_value_names_the_side_that_flattened_what_the_other_distinguishes():
    theirs = {"parcels": [{"fee": 3}, {"fee": 7}]}
    ours = {"parcels": [{"fee": 5}, {"fee": 5}]}
    found = lesson.relations_over([_triple("c1", theirs=theirs, ours=ours)])
    shared = [r for r in found if r.kind == "shared_value"]
    assert [r.where for r in shared] == ["parcels.fee"]
    assert shared[0].detail.startswith("the body puts one value")


def test_a_relation_a_passing_call_also_holds_is_not_what_tells_the_failing_calls_apart():
    passing = _broadcast_triple("c2")
    found = lesson.relations_over([_broadcast_triple("c1")], [passing])
    assert not [r for r in found if r.kind == "broadcast"]


def test_a_relation_most_failing_calls_hold_is_stated_with_its_exceptions():
    odd = _triple("c9", theirs={"parcels": [{"day": "monday"}, {"day": "friday"}]},
                  ours={"parcels": [{"day": "monday"}, {"day": "sunday"}]})
    found = lesson.relations_over([_broadcast_triple(f"c{i}") for i in range(4)] + [odd])
    broadcast = [r for r in found if r.kind == "broadcast"]
    assert broadcast and not broadcast[0].on_all
    assert broadcast[0].holds_on == 4 and broadcast[0].of == 5
    assert "it does not hold on c9" in broadcast[0].sentence()


def test_order_is_found_when_the_recorded_list_is_a_sort_the_body_did_not_make():
    theirs = {"stops": [{"rank": 1}, {"rank": 2}, {"rank": 3}]}
    ours = {"stops": [{"rank": 3}, {"rank": 1}, {"rank": 2}]}
    found = lesson.relations_over([_triple("c1", theirs=theirs, ours=ours)])
    order = [r for r in found if r.kind == "order"]
    assert order and "ascending order of rank" in order[0].detail


def test_order_is_absent_when_the_two_lists_are_not_the_same_elements():
    theirs = {"stops": [{"rank": 1}, {"rank": 2}]}
    ours = {"stops": [{"rank": 1}, {"rank": 9}]}
    found = lesson.relations_over([_triple("c1", theirs=theirs, ours=ours)])
    assert not [r for r in found if r.kind == "order"]


def test_arithmetic_is_found_when_a_recorded_number_is_a_product_of_the_call_numbers():
    triples = [_triple(f"c{i}", args={"crates": 3 + i, "rate": 4},
                       theirs={"price": (3 + i) * 4}, ours={"price": 0}) for i in range(3)]
    found = lesson.relations_over(triples)
    arithmetic = [r for r in found if r.kind == "arithmetic"]
    assert arithmetic and arithmetic[0].where == "price"
    assert any("times" in r.detail for r in arithmetic)


def test_arithmetic_is_absent_when_no_one_rule_holds_across_the_failing_calls():
    triples = [_triple("c1", args={"crates": 3, "rate": 4}, theirs={"price": 12}, ours={"price": 0}),
               _triple("c2", args={"crates": 5, "rate": 4}, theirs={"price": 99}, ours={"price": 0})]
    found = lesson.relations_over(triples)
    assert not [r for r in found if r.kind == "arithmetic"]


def test_one_recorded_answer_over_a_run_of_calls_the_body_tells_apart_is_stated_once():
    """The shape a hint written per call can never see: the recording answers a run of calls alike."""
    triples = [_triple(f"c{i}", args={"crate": f"k{i}"}, theirs={"note": "closed for collection"},
                       ours={"note": f"open until {i}"}) for i in range(5)]
    found = lesson.relations_over(triples)
    shared = [r for r in found if r.kind == "shared_value" and r.where == "note"]
    assert shared and shared[0].on_all
    assert "the recording answers every one of these calls with one value here" in shared[0].detail


def test_one_body_answer_over_calls_the_recording_tells_apart_is_stated_the_other_way_round():
    triples = [_triple(f"c{i}", args={"crate": f"k{i}"}, theirs={"note": f"open until {i}"},
                       ours={"note": "closed for collection"}) for i in range(5)]
    found = lesson.relations_over(triples)
    shared = [r for r in found if r.kind == "shared_value" and r.where == "note"]
    assert shared and "the body answers every one of these calls with one value here" in shared[0].detail


def test_no_relation_across_calls_when_both_sides_vary_freely():
    triples = [_triple(f"c{i}", theirs={"note": f"a{i}"}, ours={"note": f"b{i}"}) for i in range(5)]
    assert not [r for r in lesson.relations_over(triples) if r.kind == "shared_value"]


def test_a_refusal_predicate_states_what_the_refused_calls_share_that_accepted_ones_lack():
    refused = [_triple(f"r{i}", args={"crate": "k1", "override": True}, their_error="NotPermitted")
               for i in range(3)]
    accepted = [_triple(f"a{i}", args={"crate": "k2"}, theirs={"price": 1}, ours={"price": 1},
                        matched=True) for i in range(3)]
    found = lesson.relations_over(refused, accepted)
    predicates = [r for r in found if r.kind == "refusal_predicate"]
    assert any(r.where == "override" for r in predicates)
    assert any("refused every call that carries override" in r.detail for r in predicates)


def test_no_refusal_predicate_is_stated_when_the_recording_refused_nothing():
    accepted = [_triple(f"a{i}", args={"crate": "k2"}, theirs={"price": 1}, ours={"price": 2})
                for i in range(3)]
    assert not [r for r in lesson.relations_over(accepted) if r.kind == "refusal_predicate"]


# --- 3. the witnessed-branch read -------------------------------------------------------


SOURCE = '''
class DepotTools:
    def quote_haulage(self, crate, express=False):
        rows = self.db.crates[crate]
        total = 0
        if express:
            total = total + 5
        for parcel in rows:
            total = total + 1
        return total
'''


def test_every_branch_loop_and_assignment_of_one_function_is_found_with_the_line_to_reach():
    constructs = lesson.constructs_of(SOURCE, "quote_haulage")
    kinds = sorted({c.kind for c in constructs})
    assert kinds == ["assignment", "branch", "loop"]
    assert all(c.line > 0 for c in constructs)


def test_a_branch_no_recorded_call_reached_is_named_as_unsupported():
    constructs = lesson.constructs_of(SOURCE, "quote_haulage")
    branch = next(c for c in constructs if c.kind == "branch")
    reached = {c.line for c in constructs} - {branch.line}
    dead = lesson.unwitnessed(constructs, reached)
    assert [c.line for c in dead] == [branch.line]
    assert "express" in dead[0].text


def test_a_construct_every_call_reached_is_not_named():
    constructs = lesson.constructs_of(SOURCE, "quote_haulage")
    assert lesson.unwitnessed(constructs, {c.line for c in constructs}) == []


def test_a_function_the_module_does_not_hold_yields_no_constructs():
    assert lesson.constructs_of(SOURCE, "book_collection") == []
    assert lesson.constructs_of("def broken(:", "broken") == []


def test_a_refusal_the_recording_shows_and_the_body_never_raises_is_named_as_missing():
    triples = [_triple(f"r{i}", their_error="NotPermitted", ours={"price": 1}) for i in range(4)]
    note = lesson.missing_refusal(triples, SOURCE, "quote_haulage")
    assert "refused 4 of these calls (NotPermitted)" in note
    assert "the body holds no raise at all" in note


def test_no_missing_refusal_is_named_when_the_body_refuses_the_same_calls():
    triples = [_triple(f"r{i}", their_error="NotPermitted", our_error="NotPermitted") for i in range(4)]
    assert lesson.missing_refusal(triples, SOURCE, "quote_haulage") == ""


# --- 4. the stall limit -------------------------------------------------------------------


def test_the_stall_limit_switches_the_ask_from_patching_to_rewriting():
    triples = [_broadcast_triple("c1"), _broadcast_triple("c2")]
    patched = lesson.diagnose("quote_haulage", triples, unbeaten=lesson.STALL_LIMIT - 1)
    rewritten = lesson.diagnose("quote_haulage", triples, unbeaten=lesson.STALL_LIMIT)
    assert not patched.rewrite and lesson.REWRITE_HEAD[:20] not in patched.lesson()
    assert rewritten.rewrite and "Write a new body from the recorded calls" in rewritten.lesson()
    assert rewritten.counts()["rewrites_forced"] == 1
    assert patched.counts()["rewrites_forced"] == 0


def test_a_tie_where_both_bodies_fall_at_one_gate_before_fidelity_is_reported_as_blocked():
    kept = [{"stage": "parses", "pass": True}, {"stage": "executes_on_s0", "pass": False}]
    attempt = [{"stage": "parses", "pass": True}, {"stage": "executes_on_s0", "pass": False}]
    assert lesson.blocked_gate(kept, attempt) == "executes_on_s0"
    told = lesson.diagnose("quote_haulage", [], blocked="executes_on_s0").lesson()
    assert "fail the executes_on_s0 gate" in told and "before the replay ruling" in told


def test_a_tie_at_the_fidelity_ruling_itself_is_not_reported_as_blocked_by_a_gate():
    both = [{"stage": "parses", "pass": True}, {"stage": "replay_fidelity", "pass": False}]
    assert lesson.blocked_gate(both, both) == ""


def test_two_bodies_that_fall_at_different_gates_are_not_blocked_by_one():
    kept = [{"stage": "executes_on_s0", "pass": False}]
    attempt = [{"stage": "deterministic", "pass": False}]
    assert lesson.blocked_gate(kept, attempt) == ""


# --- what one round records -----------------------------------------------------------------


def test_the_diagnosis_counts_relations_by_kind_and_lines_nothing_witnesses():
    constructs = lesson.constructs_of(SOURCE, "quote_haulage")
    branch = next(c for c in constructs if c.kind == "branch")
    found = lesson.diagnose("quote_haulage", [_broadcast_triple("c1"), _broadcast_triple("c2")],
                            source=SOURCE, function="quote_haulage",
                            executed={c.line for c in constructs} - {branch.line},
                            unbeaten=lesson.STALL_LIMIT)
    counts = found.counts()
    assert counts["relations_found"]["broadcast"] == 1
    assert counts["unwitnessed_lines"] == 1
    assert counts["rewrites_forced"] == 1 and counts["blocked_by_gate"] == 0
    told = found.lesson()
    assert lesson.RELATION_HEAD in told and lesson.DIFF_HEAD in told
    assert lesson.UNWITNESSED_HEAD in told


def test_the_round_adds_the_per_tool_counts_up():
    one = {"relations_found": {"broadcast": 1}, "unwitnessed_lines": 3, "rewrites_forced": 1,
           "blocked_by_gate": 0}
    other = {"relations_found": {"broadcast": 2, "order": 1}, "unwitnessed_lines": 0,
             "rewrites_forced": 0, "blocked_by_gate": 1}
    total = lesson.merge_counts([one, other])
    assert total["relations_found"]["broadcast"] == 3
    assert total["relations_found"]["order"] == 1
    assert total["unwitnessed_lines"] == 3
    assert total["rewrites_forced"] == 1 and total["blocked_by_gate"] == 1


def test_a_tool_with_nothing_found_says_nothing():
    assert lesson.diagnose("quote_haulage", []).lesson() == ""
    assert lesson.diagnose("quote_haulage", []).counts()["unwitnessed_lines"] == 0


# --- the sandbox reports which lines a recorded call ran, and grade_body reads them ------------


DEPOT_DB = {"crates": {"k1": {"crate_id": "k1", "parcels": 2, "express": False},
                       "k2": {"crate_id": "k2", "parcels": 5, "express": False}}}

DEPOT_BODY = """
crate = self.db.crates[crate_id]
total = crate.parcels
if crate.express:
    total = total * 3
return {"crate_id": crate_id, "total": total}
"""


def _depot_schema():
    columns = [Column(table="crates", name=name, **{"class": "hard"}, classified_by="rule")
               for name in ("crate_id", "parcels", "express")]
    return EntitySchema(tables=["crates"], columns=columns, id_patterns={"crates": r"^k\d+$"})


def _depot_sig():
    return ToolSig(name="quote_haulage", description="Quote the haulage for one crate.",
                   args_fields=[FieldStat(name="crate_id", types=["str"], optional=False)],
                   kind="read", unclassified=False)


def _depot_calls():
    return [ToolCall(id=f"d{i}", name="quote_haulage", args={"crate_id": key},
                     result={"crate_id": key, "total": DEPOT_DB["crates"][key]["parcels"]}, raw_ptr=PTR)
            for i, key in enumerate(sorted(DEPOT_DB["crates"]))]


def test_the_sandbox_reports_the_lines_the_recorded_calls_actually_ran(tmp_path):
    schema, sig = _depot_schema(), _depot_sig()
    source = ce.module_source(schema, [sig], {"quote_haulage": DEPOT_BODY})
    box = ce.Sandbox(source, DEPOT_DB, tmp_path)
    ran = box.executed_lines(_depot_calls())
    constructs = lesson.constructs_of(source, "quote_haulage")
    dead = lesson.unwitnessed(constructs, ran)
    # No recorded crate is an express one, so the multiplying branch is a rule the evidence never
    # showed; every other construct of the body ran.
    assert [c.kind for c in dead] == ["branch"]
    assert "express" in dead[0].text


def test_a_traced_run_leaves_the_untraced_results_as_they_were(tmp_path):
    schema, sig = _depot_schema(), _depot_sig()
    source = ce.module_source(schema, [sig], {"quote_haulage": DEPOT_BODY})
    box = ce.Sandbox(source, DEPOT_DB, tmp_path)
    plain = box.run(_depot_calls())
    box.executed_lines(_depot_calls())
    assert all("lines" not in result for result in plain)
    assert box.run(_depot_calls()) == plain


def test_grade_body_reads_the_unreached_branch_off_a_body_that_still_fails(tmp_path):
    schema, sig = _depot_schema(), _depot_sig()
    wrong = DEPOT_BODY.replace("total = crate.parcels", "total = 1")
    calls = _depot_calls()
    build = ce.grade_body(sig, wrong, calls, schema, DEPOT_DB, tmp_path,
                          unbeaten=lesson.STALL_LIMIT)
    assert build.assisted and build.diagnosis is not None
    told = build.diagnosis.lesson()
    assert "Write a new body from the recorded calls" in told
    assert lesson.UNWITNESSED_HEAD in told and "express" in told
    assert build.diagnosis.counts()["unwitnessed_lines"] >= 1


def test_a_body_that_clears_the_gates_is_read_for_nothing(tmp_path):
    schema, sig = _depot_schema(), _depot_sig()
    build = ce.grade_body(sig, DEPOT_BODY, _depot_calls(), schema, DEPOT_DB, tmp_path)
    assert not build.assisted and build.diagnosis is None
