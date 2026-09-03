"""Phase 6 skills: SKILL.md files, prompt listing, content-hash edits, D132 gate."""

from __future__ import annotations

import pytest

from kullback.builder import skills


def test_write_load_list_skill(tmp_path):
    assert skills.list_skills(tmp_path) == []
    written = skills.write_skill(tmp_path, "compile", "# compile\nRules.")
    assert written["hash"] and written["prev"] is None
    assert skills.load_skill(tmp_path, "compile") == "# compile\nRules."
    assert skills.list_skills(tmp_path) == ["compile"]
    section = skills.skill_prompt_section(tmp_path)
    assert "`compile`" in section


def test_skill_edit_records_hash_chain(tmp_path):
    first = skills.write_skill(tmp_path, "compile", "v1")
    second = skills.write_skill(tmp_path, "compile", "v2")
    assert second["prev"] == first["hash"]
    edits = skills.skill_edits(tmp_path)
    assert [e["hash"] for e in edits] == [first["hash"], second["hash"]]


def test_load_missing_skill_raises(tmp_path):
    with pytest.raises(KeyError):
        skills.load_skill(tmp_path, "nope")


def test_skill_gate_tentative_then_promote():
    early = skills.skill_gate_decision([1.0, 1.0])
    assert early["decision"] == "tentative"
    decisive = skills.skill_gate_decision([1.0] * 6)
    assert decisive["decision"] == "promote"


def test_skill_gate_revert_and_demote():
    bad = skills.skill_gate_decision([-1.0] * 6)
    assert bad["decision"] == "revert"
    demote = skills.recheck_promoted([-1.0] * 6)
    assert demote["decision"] == "demote"
    mixed = skills.skill_gate_decision([1.0, -1.0, 0.0, 1.0, -1.0, 0.0])
    assert mixed["decision"] == "tentative"


def test_skill_name_traversal_raises_and_writes_nothing(tmp_path):
    import pytest

    for bad in ("../../evil", "a/b", "/abs", "..", "", "."):
        with pytest.raises(ValueError):
            skills.write_skill(tmp_path, bad, "evil")
        with pytest.raises(ValueError):
            skills.load_skill(tmp_path, bad)
    assert list(tmp_path.iterdir()) == [] or skills.list_skills(tmp_path) == []
    assert (tmp_path / "evil").exists() is False


def test_skill_edits_skips_corrupt_lines(tmp_path):
    skills.write_skill(tmp_path, "compile", "v1")
    edits = tmp_path / "skills" / "edits.jsonl"
    with edits.open("a", encoding="utf-8") as fh:
        fh.write('{"name": "compile", "hash": "broken"\n')
        fh.write('"just a string"\n')
    skills.write_skill(tmp_path, "compile", "v2")
    rows = skills.skill_edits(tmp_path)
    assert [r["hash"] for r in rows if "hash" in r and r.get("name") == "compile"] == [
        r["hash"] for r in rows if isinstance(r, dict) and r.get("name") == "compile"]
    assert len([r for r in rows if isinstance(r, dict) and "hash" in r]) == 2
