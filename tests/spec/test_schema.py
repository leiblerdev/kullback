import json

from kullback.spec.schema import is_empty, load_spec, save_spec, spec_path, tier_of
from tests.spec.fixtures import WRITE, check, spec


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


def test_a_spec_file_written_before_the_writer_counts_still_loads_with_none(tmp_path):
    save_spec(tmp_path, spec())
    path = spec_path(tmp_path, "t1")
    data = json.loads(path.read_text())
    data.pop("writer")
    path.write_text(json.dumps(data))
    assert load_spec(tmp_path, "t1").writer == {}


def test_a_spec_with_no_check_or_every_fact_a_gap_is_empty_unless_it_keeps_a_no_write_or_cap_check():
    assert is_empty(spec([]))
    assert is_empty(spec([check(fact_ids=())], gaps=["f1", "f2"]))
    assert not is_empty(spec())
    assert not is_empty(spec([check("c1", "required", {"demand": "no_write"}, fact_ids=())], gaps=["f1", "f2"]))
    assert not is_empty(spec([check("c1", "required", {"demand": "cap", "count": 0}, fact_ids=())]))
