"""Skills: files read from a directory, the catalog block a prompt section shows, and the text of
a loaded skill standing in the system prompt."""

from __future__ import annotations

from kullback.agent.harness import AgentHarness
from kullback.agent.session import SessionStore
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
    # a directory that is not there is no skills and not an error
    assert skills_from_dir(tmp_path / "none") == {}


def test_the_skills_section_lists_every_catalogued_skill_and_whether_it_is_loaded():
    agent = harness()
    assert "No skills are catalogued for this session." in agent.context.skills_section()
    agent.context.catalog_skill("bodies", "Write a body.\nMore detail.")
    digest = agent.context.catalog_skill("answers", "Answer from the world.", loaded=True)
    section = agent.context.skills_section()
    assert section.startswith("<skills>") and section.endswith("</skills>")
    assert "- bodies (not loaded): Write a body." in section
    assert "- answers (loaded, its text is the <skill> block below): Answer from the world." in section
    # a loaded skill's text stands in the system prompt under its hash
    assert digest == content_hash("Answer from the world.")
    assert '<skill name="answers">\nAnswer from the world.\n</skill>' in agent.system


def test_the_skills_section_in_the_prompt_is_refreshed_when_the_catalog_changes():
    agent = harness()
    agent.add_prompt_section("skills", agent.context.skills_section())
    agent.context.catalog_skill("bodies", "Write a body.")
    assert "- bodies (not loaded): Write a body." in agent.system
    agent.context.catalog_skill("answers", "Answer from the world.")
    assert "- answers (not loaded)" in agent.system and "- bodies (not loaded)" in agent.system
    # a skill catalogued where no section was ever added leaves the prompt alone
    bare = harness()
    bare.context.catalog_skill("bodies", "Write a body.")
    assert bare.system == "sys"
    assert bare.context.catalog_skills == {"bodies": "Write a body."}
    assert bare.context.loaded_skills == set()


def test_the_session_records_which_skill_text_was_in_context_by_hash(tmp_path):
    store = SessionStore(tmp_path / "session.jsonl")
    agent = AgentHarness(TestModel(["done"]), system="sys", session=store)
    agent.context.catalog_skill("bodies", "Write a body.", loaded=True)
    agent.context.skills.unload("bodies")
    changes = [e for e in store.entries if e.type == "skill_change"]
    assert [(e.name, e.action) for e in changes] == [("bodies", "load"), ("bodies", "unload")]
    assert {e.content_hash for e in changes} == {content_hash("Write a body.")}
    # cataloguing alone puts no text in front of the model, so it records nothing
    agent.context.catalog_skill("answers", "Answer from the world.")
    assert len([e for e in store.entries if e.type == "skill_change"]) == 2
