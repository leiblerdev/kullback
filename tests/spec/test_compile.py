import sys
from pathlib import Path

from kullback.spec.compile import compile_spec
from tests.spec.fixtures import WRITE, WRITE_TOOLS, check, fn, spec

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
    assert [(a.id, a.kind, a.target) for a in compiled.verifier.atoms] == \
        [(a.id, a.kind, a.target) for a in ported.atoms]
    assert compiled.verifier.task_id == "t1"


def test_every_atom_description_ends_with_its_because():
    compiled = compile_spec(spec(), WRITE_TOOLS, fn)
    assert compiled.verifier.atoms
    assert all(a.description.endswith(" because: move item A1 to slot seven") for a in compiled.verifier.atoms)


def test_a_check_with_an_ungrounded_because_is_dropped_and_counted():
    compiled = compile_spec(spec([check(), check("c2", because="the agent decided this")]), WRITE_TOOLS, fn)
    assert compiled.dropped == 1
    assert {a.id.split(".")[0] for a in compiled.verifier.atoms} == {"i0"}
