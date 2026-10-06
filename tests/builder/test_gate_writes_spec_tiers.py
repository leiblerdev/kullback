"""A Builder write of a Verifier file draws the trusted ruling from the Spec's tiers (D322), end to end.

The write goes through `gate_writes` with the Spec loader `builder/session.py` hands it. The workdir is
tests/spec/test_trust.py's: one Task whose Reference moves item A1 to slot seven, kept and faithfully
replayed, with a fresh Run that reaches the same end state.
"""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

from kullback.gates.hook import gate_writes
from kullback.runner.records import Verifier
from kullback.spec.trust import spec_tiers_of
from tests.spec.test_trust import CODE_CHECKS, _fresh, _intent_workdir, _replay, _verifier, spec


def _workdir(root, sourced=True, with_spec=True):
    """The Spec workdir with one Task, and the Builder's copy of its Verifier under env/."""
    verifier = _verifier(spec(gaps=["f2"]), sourced=sourced)
    _intent_workdir(root, [_replay(), _fresh()], verifier=verifier)
    (root / "task_status.json").write_text(json.dumps({"t1": CODE_CHECKS}))
    if not with_spec:
        shutil.rmtree(root / "spec")
        verifier = Verifier(task_id="t1", atoms=verifier.atoms)  # written before end states (legacy)
    (root / "env" / "verifiers").mkdir(parents=True)
    (root / "env" / "verifiers" / "t1.json").write_text(verifier.model_dump_json())
    return root, verifier


def _trusted_ruling(root, verifier):
    """The trusted record the hook attaches to the Builder's write of the Task's Verifier file.

    The Spec loader is the session's own. The hook loads no task status and no Verifier list for the
    trusted gate, so the test hands both as the session would have to.
    """
    hook = gate_writes(root=root / "env", workdir=root,
                       evidence={"spec_tiers": lambda: spec_tiers_of(root),
                                 "task_status": json.loads((root / "task_status.json").read_text()),
                                 "verifiers": [json.loads(verifier.model_dump_json())]})
    written = SimpleNamespace(content="wrote verifiers/t1.json", details={}, is_error=False)
    out = hook(SimpleNamespace(name="write", arguments={"path": "verifiers/t1.json"}), written)
    return next(record for record in out.details["rulings"] if record["name"] == "trusted")


def test_a_verifier_write_on_a_fully_sourced_spec_draws_a_trusted_ruling(tmp_path):
    root, verifier = _workdir(tmp_path)
    assert spec_tiers_of(root)["t1"].tier == "trusted"
    ruling = _trusted_ruling(root, verifier)
    assert ruling["accepted"] is True and ruling["line"] == "trusted pass" and ruling["rows"] == []


def test_a_verifier_write_with_one_unsupported_cell_is_unconfirmed_naming_the_cell_and_not_trusted(tmp_path):
    root, verifier = _workdir(tmp_path, sourced=False)
    ruling = _trusted_ruling(root, verifier)
    assert ruling["accepted"] is False
    assert ruling["rows"] == [{"task": "t1", "reason": "unconfirmed: unsupported cell items.A1.slot"}]
    assert ruling["line"].startswith("trusted fail (task t1: unconfirmed: unsupported cell items.A1.slot)")


def test_a_verifier_write_in_a_workdir_without_specs_keeps_the_gates_legacy_ruling(tmp_path):
    root, verifier = _workdir(tmp_path, with_spec=False)
    assert spec_tiers_of(root) is None
    ruling = _trusted_ruling(root, verifier)
    assert ruling["accepted"] is False
    assert ruling["rows"] == [{"task": "t1", "reason": "the D79 suite did not pass"}]
