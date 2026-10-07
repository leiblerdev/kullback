import sys
from pathlib import Path

from kullback.runner.records import Event, Run
from kullback.runner.target import atom_holds, check_run, hard_holds, text_of
from kullback.spec.compile import compile_spec
from tests.spec.fixtures import WRITE, WRITE_TOOLS, check, fn, reference_run, spec

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))

from intentv import compile_verifier  # noqa: E402

DEMANDS = [
    WRITE,
    {"demand": "say", "text": "slot seven"},
    {"demand": "ask", "field": "slot", "tool": "update_item"},
    {"demand": "no_write"},
    {"demand": "cap", "count": 1},
    {"demand": "shape", "tool": "update_item", "field": "slot", "id_field": "item_id"},
]
KINDS = ["required", "communicate", "question", "forbidden", "required", "hard"]


def test_compile_yields_the_same_atoms_as_the_intentv_experiment():
    the_spec = spec([check(f"c{n}", kind, demand) for n, (kind, demand) in enumerate(zip(KINDS, DEMANDS, strict=True))])
    compiled = compile_spec(the_spec, WRITE_TOOLS, fn)
    ported, dropped = compile_verifier([dict(d, kind=k, because="move item A1 to slot seven")
                                        for k, d in zip(KINDS, DEMANDS, strict=True)], the_spec.intent.text, [], WRITE_TOOLS, fn)
    assert dropped == compiled.dropped == 0
    # intentv wrote the raw id as the entity; the Spec writes the id as the write effect keys it (sp-layer r1).
    def keyed(target):
        return dict(target, entity=text_of(fn(target["entity_raw"]))) if "entity_raw" in target else target

    assert [(a.id, a.kind, a.target) for a in compiled.verifier.atoms] == \
        [(a.id, a.kind, keyed(a.target)) for a in ported.atoms]
    assert compiled.verifier.task_id == "t1"


def test_every_atom_description_ends_with_its_because():
    compiled = compile_spec(spec(), WRITE_TOOLS, fn)
    assert compiled.verifier.atoms
    assert all(a.description.endswith(" because: move item A1 to slot seven") for a in compiled.verifier.atoms)


def test_a_check_with_an_ungrounded_because_is_dropped_and_counted():
    compiled = compile_spec(spec([check(), check("c2", because="the agent decided this")]), WRITE_TOOLS, fn)
    assert compiled.dropped == 1
    assert {a.id.split(".")[0] for a in compiled.verifier.atoms} == {"i0"}


def test_a_write_id_whose_canonical_form_differs_from_its_raw_form_passes_its_own_run():
    verifier = compile_spec(spec(), WRITE_TOOLS, fn).verifier
    write = next(a for a in verifier.atoms if a.target["kind"] == "write")
    assert write.target["entity"] != write.target["entity_raw"] == "A1"
    assert check_run(verifier, reference_run(), canon=fn)[0]


def test_a_write_demand_without_an_entity_yields_no_atom_and_one_named_drop():
    no_entity = {key: value for key, value in WRITE.items() if key != "entity"}
    compiled = compile_spec(spec([check("c9", demand=no_entity)]), WRITE_TOOLS, fn)
    assert compiled.verifier.atoms == []
    assert compiled.dropped == 1
    assert compiled.reasons == ("c9: write without entity",)


TWO_WRITES = set(WRITE_TOOLS) | {"delete_item"}


def _writing(*calls) -> Run:
    """A Run making each (tool, item id) write once, each answered without an error."""
    events = []
    for n, (tool, item) in enumerate(calls):
        events += [Event(idx=2 * n, type="tool_call", payload={"id": f"c{n}", "name": tool, "args": {"item_id": item}}),
                   Event(idx=2 * n + 1, type="tool_result", payload={"id": f"c{n}", "result": {"item_id": item}})]
    return Run(run_id="r", task_id="t1", termination_reason="success", events=events)


def _holds(demand, run) -> bool:
    (atom,) = compile_spec(spec([check("c0", "required", demand)]), TWO_WRITES, fn).verifier.atoms
    if atom.kind == "hard":
        return hard_holds(atom, run, TWO_WRITES, fn)
    return atom_holds(atom, run, fn, TWO_WRITES)


def test_a_no_write_naming_a_tool_forbids_that_tool_only():
    named = {"demand": "no_write", "tool": "update_item"}
    assert _holds(named, _writing(("delete_item", "B2")))
    assert not _holds(named, _writing(("update_item", "A1")))
    assert not _holds({"demand": "no_write"}, _writing(("delete_item", "B2")))


def test_a_cap_naming_a_tool_counts_that_tools_writes_only():
    named = {"demand": "cap", "count": 1, "tool": "update_item"}
    assert _holds(named, _writing(("update_item", "A1"), ("delete_item", "B2"), ("delete_item", "C3")))
    assert not _holds(named, _writing(("update_item", "A1"), ("update_item", "B2")))
    assert not _holds({"demand": "cap", "count": 1}, _writing(("update_item", "A1"), ("delete_item", "B2")))


def _unsatisfiable(*checks):
    return compile_spec(spec(list(checks)), TWO_WRITES, fn).unsatisfiable


def test_a_hard_no_write_beside_a_required_write_on_a_tool_it_covers_is_unsatisfiable():
    write = check("c0", "required", WRITE)
    assert _unsatisfiable(write, check("c1", "required", {"demand": "no_write"})) == \
        (("c0", "no_write_beside_required_write"), ("c1", "no_write_beside_required_write"))
    assert _unsatisfiable(write, check("c1", "required", {"demand": "no_write", "tool": "delete_item"})) == ()


def test_a_forbidden_demand_equal_to_a_required_one_is_unsatisfiable():
    assert _unsatisfiable(check("c0", "required", WRITE), check("c1", "forbidden", WRITE)) == \
        (("c0", "forbidden_equals_required"), ("c1", "forbidden_equals_required"))


def test_a_required_write_with_no_id_field_or_an_object_entity_is_unsatisfiable():
    no_field = {key: value for key, value in WRITE.items() if key != "id_field"}
    as_object = dict(WRITE, entity={"item_id": "A1"})
    assert _unsatisfiable(check("c0", "required", no_field), check("c1", "required", as_object)) == \
        (("c0", "required_write_without_row"), ("c1", "required_write_without_row"))
    assert _unsatisfiable(check("c0", "allowed", no_field)) == ()


def test_a_say_demand_listing_values_compiles_to_one_communicate_atom_per_value():
    said = {"demand": "say", "values": ["slot seven", "A1"]}
    atoms = compile_spec(spec([check(demand=said)]), WRITE_TOOLS, fn).verifier.atoms
    assert [(atom.kind, atom.target["text"]) for atom in atoms] == [("communicate", "slot seven"),
                                                                     ("communicate", "A1")]
    assert [atom.description.split(" because:")[0] for atom in atoms] == \
        ["the final answer states slot seven", "the final answer states A1"]
    assert len({atom.id for atom in atoms}) == 2
    assert len(compile_spec(spec([check(demand={"demand": "say", "text": "slot seven"})]), WRITE_TOOLS,
                            fn).verifier.atoms) == 1
