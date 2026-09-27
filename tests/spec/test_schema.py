from kullback.spec.schema import load_spec, save_spec, spec_path, tier_of
from tests.spec.fixtures import WRITE, spec


def test_a_required_write_is_critical_a_forbidden_check_is_sanity_a_question_is_important():
    assert tier_of("required", WRITE) == "critical"
    assert tier_of("forbidden", {"demand": "no_write"}) == "sanity"
    assert tier_of("required", {"demand": "ask", "field": "slot"}) == "important"


def test_a_spec_saved_to_the_workdir_loads_back_the_same(tmp_path):
    original = spec(gaps=["f2"])
    path = save_spec(tmp_path, original)
    assert path == spec_path(tmp_path, "t1") == tmp_path / "spec" / "t1.json"
    assert load_spec(tmp_path, "t1") == original


def test_a_task_without_a_spec_loads_none(tmp_path):
    assert load_spec(tmp_path, "t9") is None
