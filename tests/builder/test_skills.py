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
    with pytest.raises(KeyError):
        skills.load_skill(tmp_path, "nope")


def test_skill_edit_records_hash_chain(tmp_path):
    first = skills.write_skill(tmp_path, "compile", "v1")
    second = skills.write_skill(tmp_path, "compile", "v2")
    assert second["prev"] == first["hash"]
    edits = skills.skill_edits(tmp_path)
    assert [e["hash"] for e in edits] == [first["hash"], second["hash"]]


@pytest.mark.parametrize(("gate", "scores", "decision"), [
    (skills.skill_gate_decision, [1.0, 1.0], "tentative"),
    (skills.skill_gate_decision, [1.0] * 6, "promote"),
    (skills.skill_gate_decision, [-1.0] * 6, "revert"),
    (skills.skill_gate_decision, [1.0, -1.0, 0.0, 1.0, -1.0, 0.0], "tentative"),
    (skills.recheck_promoted, [-1.0] * 6, "demote"),
])
def test_the_skill_gate_decides_from_the_scores(gate, scores, decision):
    assert gate(scores)["decision"] == decision


def test_a_skill_path_that_leaves_the_workdir_raises_and_writes_nothing(tmp_path):
    """A traversing name, or a `skills` symlink pointing outside the workdir, fails closed.
    Resolving both sides through the same symlink once made containment vacuous; the anchor is
    the workdir itself, so nothing model-supplied lands outside it."""
    for bad in ("../../evil", "a/b", "/abs", "..", "", "."):
        with pytest.raises(ValueError):
            skills.write_skill(tmp_path, bad, "evil")
        with pytest.raises(ValueError):
            skills.load_skill(tmp_path, bad)
    assert list(tmp_path.iterdir()) == [] or skills.list_skills(tmp_path) == []
    assert (tmp_path / "evil").exists() is False

    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "skills").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes the workdir"):
        skills.write_skill(tmp_path / "work", "compile", "# evil\nRules.")
    with pytest.raises(ValueError, match="escapes the workdir"):
        skills.load_skill(tmp_path / "work", "compile")
    with pytest.raises(ValueError, match="escapes the workdir"):
        skills.list_skills(tmp_path / "work")
    assert list(outside.iterdir()) == [], "no skill content and no edit log may land outside"
    assert list((tmp_path / "work").iterdir()) == [tmp_path / "work" / "skills"]


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
