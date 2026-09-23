"""Skills: files read from a directory by file stem."""

from __future__ import annotations

from kullback.agent.skills import skills_from_dir


def test_skills_are_read_from_a_directory_by_file_stem_in_name_order(tmp_path):
    assert skills_from_dir(tmp_path / "none") == {}
    (tmp_path / "bodies.md").write_text("Write a body from the recording.", encoding="utf-8")
    (tmp_path / "answers.txt").write_text("Answer from the world.", encoding="utf-8")
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("hidden", encoding="utf-8")
    found = skills_from_dir(tmp_path)
    assert list(found) == ["answers", "bodies"]
    assert found["bodies"] == "Write a body from the recording."
