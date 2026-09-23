"""Store specs registry tests: every declared workdir artifact writes, reads, and refuses as one store (D267)."""

from __future__ import annotations

import pytest

from kullback import store_specs
from kullback.store import ArtifactSpec, WorkdirStore

EXAMPLES: dict[str, object] = {
    "rounds": [{"round": 1}],
    "report_config": {"kind": "batch"},
    "gates": [{"stage": "compile_tools", "pass": True}],
    "gates_by_round": [{"round": 1, "rulings": [], "tasks": {}}],
    "compile_snapshot": {"format": 1, "round": 1, "stage": "compile_tools", "rows": []},
    "tasks_frozen": {"format": 2, "task_ids": [], "tasks": [], "hash": "h"},
    "scorecard": {},
    "tool_sigs": [{"name": "t"}],
    "unknown_tools": [],
    "row_homes": {},
    "world_constants": {"table": "t", "row": "r", "columns": []},
    "schema": {},
    "readers": {},
    "canon_rules": {},
    "equivalence": {},
    "tasks_index": {"tasks": [{"id": "t1"}]},
    "task_split": {},
    "grouping": {"format": 2, "fingerprint": "f"},
    "db": {},
    "world_provenance": {"witnesses": {}, "holdout_columns": {}, "holdout_rows": 0, "holdout_columns_total": 0},
    "synthetic": {},
    "assumptions": [],
    "overlay_pins": {"totals": {}, "tasks": {}},
    "bodies": {},
    "tool_builds": {},
    "tool_call_outcomes": {},
    "kept_bodies": {},
    "tool_lesson_counts": {},
    "tool_fidelity": {},
    "replays": {},
    "runs_index": {"runs": []},
    "holdout_answers": {"totals": {}, "columns": [], "tasks": {}, "tools": {}},
    "replay_evidence": {},
    "write_effects": {"totals": {}, "per_tool": {}},
    "semantic_counts": {},
    "readmission_blocked": {},
    "constraints": [{"id": "c", "text": "t"}],
    "policy_coverage": {"exercised": []},
    "policy": {"items": 0, "compiled": 0},
    "lessons_set_aside": [{}],
    "vocabulary": {},
    "user_facts": {"facts": []},
    "environment": {"env_id": "e"},
    "batch_gates": [{"stage": "candidate", "pass": True}],
    "pipeline_state": {"status": "ok", "statuses": {}, "attempts": {}, "log": [], "gates": []},
    "anchor": {},
    "evidence_counts": {},
    "intake_ruling": {"files": {}, "counts": {}, "reasons": {}, "floor": 0.0, "eligible_share": 1.0, "passed": True},
    "claims": {"format": 1, "runs": [], "tasks": {}, "totals": {}, "flagged": [], "write_tools": []},
    "difficulty": {"format": 3, "tasks": [], "no_record": {}, "buckets": []},
    "user_lessons": {"format": 1, "lessons": []},
    "user_fidelity": {"format": 1, "round": 0, "tasks": [], "summary": {}},
    "domain_gaps": {"format": 1, "gaps": []},
    "domain_archetypes": {"format": 1, "archetypes": []},
    "synth_index": {"format": 1, "requested": {}, "salt": "s", "counts": {}, "tasks": []},
    "synth_shaped": {"format": 1, "salt": "s", "shaped": True, "counts": {}, "tasks": []},
    "graph": {"format": 1, "runs": 0, "nodes": [], "edges": []},
    "budget": {"stages": {}, "total": {}},
    "sample_salt": {"salt": "s"},
    "examiner_findings": [{"finding_id": "f", "kind": "other", "text": "t"}],
    "examiner_refusals": {},
    "examiner_auto_loosen": [],
    "examiner_rerolls": {},
    "task_status": {},
    "references": {},
    "constraints_check": {"rates": {}, "demoted": [], "constraints": []},
}

WRONG_SHAPE: dict[str, object] = {
    "rounds": {},
    "report_config": [],
    "gates": {},
    "gates_by_round": [{}],
    "compile_snapshot": {},
    "tasks_frozen": {},
    "scorecard": [],
    "tool_sigs": {},
    "unknown_tools": {},
    "row_homes": [],
    "world_constants": {},
    "schema": [],
    "readers": [],
    "canon_rules": [],
    "equivalence": [],
    "tasks_index": {"tasks": {}},
    "task_split": [],
    "grouping": {},
    "db": [],
    "world_provenance": {},
    "synthetic": [],
    "assumptions": {},
    "overlay_pins": {},
    "bodies": [],
    "tool_builds": [],
    "tool_call_outcomes": [],
    "kept_bodies": [],
    "tool_lesson_counts": [],
    "tool_fidelity": [],
    "replays": [],
    "runs_index": {},
    "holdout_answers": {},
    "replay_evidence": [],
    "write_effects": {},
    "semantic_counts": [],
    "readmission_blocked": [],
    "constraints": {},
    "policy_coverage": {},
    "policy": {},
    "lessons_set_aside": {},
    "vocabulary": [],
    "user_facts": {},
    "environment": [],
    "batch_gates": {},
    "pipeline_state": {},
    "anchor": [],
    "evidence_counts": [],
    "intake_ruling": {},
    "claims": {},
    "difficulty": {},
    "user_lessons": {},
    "user_fidelity": {},
    "domain_gaps": {},
    "domain_archetypes": {},
    "synth_index": {},
    "synth_shaped": {},
    "graph": {},
    "budget": {},
    "sample_salt": {},
    "examiner_findings": {},
    "examiner_refusals": [],
    "examiner_auto_loosen": {},
    "examiner_rerolls": [],
    "task_status": [],
    "references": [],
    "constraints_check": {},
}

NAMES = sorted(EXAMPLES)


def _store(root):
    return WorkdirStore(root, list(store_specs.STORE_SPECS))


def test_every_declared_artifact_has_a_valid_and_a_wrong_shape_example():
    assert sorted(spec.name for spec in store_specs.STORE_SPECS) == NAMES
    assert set(WRONG_SHAPE) == set(EXAMPLES)


def test_no_two_artifacts_share_a_path_even_by_letter_case(tmp_path):
    paths = [spec.path.casefold() for spec in store_specs.STORE_SPECS]
    assert len(set(paths)) == len(paths)
    with pytest.raises(ValueError):
        WorkdirStore(tmp_path / "w", [ArtifactSpec(name="a1", path="data.json", format=1, owner="o", validate=int),
                                      ArtifactSpec(name="a2", path="DATA.json", format=1, owner="o", validate=int)])


def test_every_declared_artifact_is_found_by_its_unique_name():
    names = [spec.name for spec in store_specs.STORE_SPECS]
    assert len(set(names)) == len(names)
    assert set(store_specs.BY_NAME) == set(EXAMPLES)
    for spec in store_specs.STORE_SPECS:
        assert store_specs.BY_NAME[spec.name] is spec


@pytest.mark.parametrize("name", NAMES)
def test_every_declared_artifact_reads_back_what_was_written(tmp_path, name):
    store = _store(tmp_path)
    value = EXAMPLES[name]
    store.write(name, value)
    result = store.read(name)
    assert result.status == "ok"
    assert result.value == value


@pytest.mark.parametrize("name", NAMES)
def test_a_value_of_the_wrong_shape_is_refused_before_anything_is_written(tmp_path, name):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.write(name, WRONG_SHAPE[name])
    assert not (tmp_path / store_specs.BY_NAME[name].path).exists()
