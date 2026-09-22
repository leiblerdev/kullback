"""Skills: files read from a directory, the catalog block a prompt section shows, and the text of
a loaded skill standing in the system prompt."""

from __future__ import annotations

from kullback.agent.harness import AgentHarness
from kullback.agent.skills import content_hash, skills_from_dir
from kullback.ai.provider import TestModel


def harness() -> AgentHarness:
    return AgentHarness(TestModel(["done"]), system="sys")


def test_skills_are_read_from_a_directory_by_file_stem_in_name_order(tmp_path):
    (tmp_path / "bodies.md").write_text("Write a body from the recording.", encoding="utf-8")
    (tmp_path / "answers.txt").write_text("Answer from the world.", encoding="utf-8")
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("hidden", encoding="utf-8")
    found = skills_from_dir(tmp_path)
    assert list(found) == ["answers", "bodies"]
    assert found["bodies"] == "Write a body from the recording."


def test_a_directory_that_is_not_there_is_no_skills_and_not_an_error(tmp_path):
    assert skills_from_dir(tmp_path / "none") == {}


def test_the_skills_section_lists_every_catalogued_skill_and_whether_it_is_loaded():
    agent = harness()
    agent.context.catalog_skill("bodies", "Write a body.\nMore detail.")
    agent.context.catalog_skill("answers", "Answer from the world.", loaded=True)
    section = agent.context.skills_section()
    assert section.startswith("<skills>") and section.endswith("</skills>")
    assert "- bodies (not loaded): Write a body." in section
    assert "- answers (loaded, its text is the <skill> block below): Answer from the world." in section


def test_the_section_says_so_when_no_skill_is_catalogued():
    assert "No skills are catalogued for this session." in harness().context.skills_section()


def test_loading_a_skill_puts_its_text_in_the_system_prompt_and_unloading_takes_it_out():
    agent = harness()
    digest = agent.context.catalog_skill("bodies", "Write a body.", loaded=True)
    assert digest == content_hash("Write a body.")
    assert '<skill name="bodies">\nWrite a body.\n</skill>' in agent.system
    agent.context.skills.unload("bodies")
    assert "Write a body." not in agent.system
    assert "- bodies (not loaded)" in agent.context.skills_section()


def test_the_skills_section_in_the_prompt_is_refreshed_when_the_catalog_changes():
    agent = harness()
    agent.add_prompt_section("skills", agent.context.skills_section())
    agent.context.catalog_skill("bodies", "Write a body.")
    assert "- bodies (not loaded): Write a body." in agent.system
    agent.context.catalog_skill("answers", "Answer from the world.")
    assert "- answers (not loaded)" in agent.system and "- bodies (not loaded)" in agent.system


def test_a_skill_that_was_never_added_as_a_section_leaves_the_prompt_alone():
    agent = harness()
    agent.context.catalog_skill("bodies", "Write a body.")
    assert agent.system == "sys"
    assert agent.context.catalog_skills == {"bodies": "Write a body."}
    assert agent.context.loaded_skills == set()
