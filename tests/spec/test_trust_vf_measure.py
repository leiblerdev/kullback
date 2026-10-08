"""vf_measure reads trust off the build's tier report where the build wrote one (sp-trust)."""

import json

from tests.test_vf_measure import OTHER, TASK, M, write_workdir


def test_a_tier_report_replaces_the_round_rows_as_the_trusted_set(tmp_path):
    workdir = write_workdir(tmp_path, {TASK: 0.0, OTHER: 1.0}, {TASK: True, OTHER: True})
    rows = [{"task_id": TASK, "tier": "replay_only"}, {"task_id": OTHER, "tier": "trusted"}]
    (workdir / "tiers.json").write_text(json.dumps({"verifier_from": "intent", "rows": rows}))
    result = M.measure(workdir, workdir / "grader", [TASK, OTHER], 7, 20)
    assert result["trusted"] == 1 and result["cost_trusted"] == 1
    assert (result["trusted_wrong"]["k"], result["caught"]["ids"]) == (0, [TASK])
