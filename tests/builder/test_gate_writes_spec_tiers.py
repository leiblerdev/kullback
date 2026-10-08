"""A Builder write of a Verifier file draws the trusted ruling from the Spec's tiers (D322), end to end.

The write goes through `gate_writes` wired as `builder/session.py` wires it, the Spec loader the only
evidence. The workdir is tests/spec/test_trust.py's, laid out like a live one (task_status.json, one
verifiers/<task>.json per Task, no verifiers.json, no reference.json): one Task whose Reference moves item
A1 to slot seven, kept and faithfully replayed, with a fresh Run that reaches the same end state;
the untrusted case is the same Task with a ruling open.
"""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

from kullback.builder import env_files
from kullback.gates.hook import gate_writes
from kullback.runner.records import Verifier
from kullback.spec.schema import save_spec
from kullback.spec.trust import spec_tiers_of
from tests.spec.test_trust import CODE_CHECKS, _fresh, _intent_workdir, _replay, _verifier, spec


def _workdir(root, ruling_open=False, with_spec=True):
    """The Spec workdir with one Task, and the Builder's copy of its Verifier under env/."""
    verifier = _verifier(spec(gaps=["f2"]))
    _intent_workdir(root, [_replay(), _fresh()], verifier=verifier)
    if ruling_open:
        save_spec(root, spec(gaps=["f2"]).model_copy(update={"rulings_open": 1}))
    (root / "task_status.json").write_text(json.dumps({"t1": CODE_CHECKS}))
    if not with_spec:
        shutil.rmtree(root / "spec")
        verifier = Verifier(task_id="t1", atoms=verifier.atoms)  # written before end states (legacy)
    for folder in (root / "verifiers", root / "env" / "verifiers"):
        folder.mkdir(parents=True)
        (folder / "t1.json").write_text(verifier.model_dump_json())
    return root


def _old_layout(root):
    """The same workdir with the Verifiers in one verifiers.json, as older workdirs keep them."""
    verifiers = [json.loads(path.read_text()) for path in sorted((root / "verifiers").glob("*.json"))]
    shutil.rmtree(root / "verifiers")
    (root / "verifiers.json").write_text(json.dumps(verifiers))
    return root


def _trusted_ruling(root):
    """The trusted record the hook attaches to the Builder's write of the Task's Verifier file."""
    env = root / "env"
    hook = gate_writes(root=env, workdir=root, execute=env_files.executor(root, env),
                       evidence={"spec_tiers": lambda: spec_tiers_of(root)})
    written = SimpleNamespace(content="wrote verifiers/t1.json", details={}, is_error=False)
    out = hook(SimpleNamespace(name="write", arguments={"path": "verifiers/t1.json"}), written)
    return next(record for record in out.details["rulings"] if record["name"] == "trusted")


def test_a_verifier_write_on_a_fully_sourced_spec_draws_a_trusted_ruling(tmp_path):
    root = _workdir(tmp_path)
    assert spec_tiers_of(root)["t1"].tier == "trusted"
    ruling = _trusted_ruling(root)
    assert ruling["accepted"] is True and ruling["line"] == "trusted pass" and ruling["rows"] == []


def test_a_verifier_write_on_a_spec_with_a_ruling_open_is_not_trusted_and_names_the_reason(tmp_path):
    root = _workdir(tmp_path, ruling_open=True)
    ruling = _trusted_ruling(root)
    assert ruling["accepted"] is False
    assert ruling["rows"] == [{"task": "t1", "reason": "open_ruling"}]
    assert ruling["line"].startswith("trusted fail (task t1: open_ruling)")


def test_a_verifier_write_in_a_workdir_without_specs_keeps_the_gates_legacy_ruling(tmp_path):
    root = _workdir(tmp_path, with_spec=False)
    assert spec_tiers_of(root) is None
    ruling = _trusted_ruling(root)
    assert ruling["accepted"] is False
    assert ruling["rows"] == [{"task": "t1", "reason": "the D79 suite did not pass"}]


def test_a_verifier_write_in_a_workdir_keeping_verifiers_json_still_rules_from_the_spec(tmp_path):
    root = _old_layout(_workdir(tmp_path, ruling_open=True))
    assert not (root / "verifiers").exists()
    ruling = _trusted_ruling(root)
    assert ruling["rows"] == [{"task": "t1", "reason": "open_ruling"}]
