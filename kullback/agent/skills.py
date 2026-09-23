"""Skills: texts that teach one way of working, put in front of the model as prompt sections.

tau_coding discovers skills as files in a directory and puts the ones in play into the system
prompt; this is that, for an extension of this core. A skill is a name and a text. Cataloguing one
lists it in the `<skills>` block; loading one adds its text to the prompt as its own section, and
the session records which text was in context by hash (D125).

There is no `load` or `unload` tool any more: an extension decides what its agent carries, and
compaction is the core's, not the model's.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Optional, Union

__all__ = ["Skills", "content_hash", "first_line", "prompt_block", "skills_from_dir"]

SKILLS_SECTION = "skills"
SKILL_SUFFIXES = (".md", ".txt")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def first_line(text: str, width: int = 80) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= width else line[: width - 3] + "..."
    return ""


def prompt_block(tag: str, text: str, **attrs: str) -> str:
    """One tagged block of a system prompt: `<tools>` ... `</tools>`. The tag names what the block is
    (task, tools, skills, examples, rules, stop) so the model can point at it and a reader can find it."""
    attributes = "".join(f' {key}="{value}"' for key, value in attrs.items())
    return f"<{tag}{attributes}>\n{text}\n</{tag}>"


def skills_from_dir(directory: Union[str, Path]) -> dict[str, str]:
    """Every skill file in a directory, by name: the file stem and its text, sorted by name.

    A workdir may carry skills (text) and never an extension (code), so this reads files and
    nothing else. A directory that is not there is no skills, not an error.
    """
    base = Path(directory)
    if not base.is_dir():
        return {}
    found: dict[str, str] = {}
    for path in sorted(base.iterdir()):
        if path.is_file() and path.suffix in SKILL_SUFFIXES and not path.name.startswith("."):
            found[path.stem] = path.read_text(encoding="utf-8")
    return found


class Skills:
    """What a harness's skills are: the catalog, which are loaded, and the block that says so.

    `record` is called with the name, the action and the hash of the text whenever a skill enters
    or leaves the prompt, so the session says which text was in context (D125); the harness's
    context manager passes it, and a Skills without one simply keeps no record.
    """

    def __init__(self, harness: Any, record: Optional[Callable[[str, str, str], None]] = None):
        self.harness = harness
        self.record = record
        self.catalog: dict[str, str] = {}
        self.loaded: set[str] = set()

    def catalog_skill(self, name: str, text: str, loaded: bool = False) -> Optional[str]:
        """List a skill; with `loaded`, put its text in the prompt now. Returns the text's hash."""
        self.catalog[name] = text
        digest = None
        if loaded and name not in self.loaded:
            digest = self.load(name)
        self.refresh_section()
        return digest

    def load(self, name: str) -> str:
        text = self.catalog[name]
        self.loaded.add(name)
        self.harness.add_prompt_section(f"skill:{name}", prompt_block("skill", text, name=name))
        self.refresh_section()
        return self._changed(name, "load", text)

    def unload(self, name: str) -> str:
        text = self.catalog.get(name, "")
        self.loaded.discard(name)
        self.harness.remove_prompt_section(f"skill:{name}")
        self.refresh_section()
        return self._changed(name, "unload", text)

    def _changed(self, name: str, action: str, text: str) -> str:
        digest = content_hash(text)
        if self.record is not None:
            self.record(name, action, digest)
        return digest

    def section(self) -> str:
        """The `<skills>` block: every catalogued skill and whether its text is in the prompt."""
        lines = []
        for name, text in self.catalog.items():
            state = "loaded, its text is the <skill> block below" if name in self.loaded else "not loaded"
            lines.append(f"- {name} ({state}): {first_line(text)}")
        body = "\n".join(lines) if lines else "No skills are catalogued for this session."
        return prompt_block(
            SKILLS_SECTION, "Skills are texts that teach one way of working; the tools stay the same.\n" + body
        )

    def refresh_section(self) -> None:
        if any(section.name == SKILLS_SECTION for section in self.harness.sections):
            self.harness.add_prompt_section(SKILLS_SECTION, self.section())
