"""Every model-facing prompt stays general: no Environment tool, table or field name, no corpus word.

The names come from the Environment fixtures under tests/fixtures (their tables, the fields of their rows and
the tools their tasks call); the corpus names from scripts/branch_checks.py; the domain words from the list
below. Every string literal of a prompt module is read, so a tool description or a repair line counts too.
"""

import ast
import importlib.util
import json
import re
import typing
from pathlib import Path

import pytest

from kullback.spec import intent, items, writer
from kullback.spec.schema import Condition, FactKind

ROOT = Path(__file__).resolve().parents[1]
# The modules whose strings a model reads: miner, writer and its repair prompt, Examiner, judge, user agent, Builder.
PROMPT_MODULES = (
    "kullback/spec/intent.py", "kullback/spec/intent_tools.py", "kullback/spec/writer.py",
    "kullback/spec/writer_tools.py", "kullback/spec/review.py", "kullback/examiner/prompt.py",
    "kullback/examiner/session.py", "kullback/examiner/rule_tool.py", "kullback/examiner/skills.py",
    "kullback/examiner/judge.py", "kullback/judge/items.py", "kullback/user/skills.py", "kullback/user/context.py",
    "kullback/user/extension.py", "kullback/builder/prompt.py",
)
# Words of customer-service domains that are no harness word; a prompt says <record>, <field>, <value> instead.
DOMAIN_WORDS = (
    "membership", "reservation", "flight", "baggage", "cabin", "passenger", "insurance", "gift card", "roaming",
    "data plan", "sim card", "voucher", "refund", "shipping", "payment", "airport", "invoice",
    "subscription", "zip code",
)


def _corpus_names() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location("branch_checks", ROOT / "scripts" / "branch_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.FORBIDDEN_NAMES)


def _keys(value, out: set) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            out.add(str(key))
            _keys(inner, out)
    elif isinstance(value, list):
        for inner in value:
            _keys(inner, out)


def _called(value, out: set) -> None:
    if isinstance(value, dict):
        if isinstance(value.get("name"), str) and ("arguments" in value or "args" in value):
            out.add(value["name"])
        for inner in value.values():
            _called(inner, out)
    elif isinstance(value, list):
        for inner in value:
            _called(inner, out)


def _environment_names() -> set[str]:
    """The fixtures' tables, row fields and called tools that are identifiers (snake_case): plain words such
    as a name or a status are English too, so only the identifiers are held against a prompt."""
    names: set[str] = set()
    for db in sorted((ROOT / "tests" / "fixtures").glob("*/db.json")):
        tables = json.loads(db.read_text())
        names |= set(tables)
        for rows in tables.values():
            _keys(list(rows.values()) if isinstance(rows, dict) else rows, names)
        tasks = db.parent / "tasks.json"
        if tasks.is_file():
            _called(json.loads(tasks.read_text()), names)
    return {name for name in names if "_" in name and re.fullmatch(r"[a-z][a-z0-9_]+", name)}


def _strings(path: str) -> list[tuple[int, str]]:
    tree = ast.parse((ROOT / path).read_text())
    return [(node.lineno, node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def test_the_fixtures_give_names_to_hold_against_the_prompts():
    assert len(_environment_names()) >= 10 and _corpus_names()


@pytest.mark.parametrize("path", PROMPT_MODULES)
def test_no_prompt_string_names_an_environment_identifier_a_corpus_or_a_domain_word(path):
    names = _environment_names()
    identifier = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True)))
                            + r")(?![A-Za-z0-9_])")
    word = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, _corpus_names() + DOMAIN_WORDS))
                      + r")s?(?![A-Za-z0-9])", re.IGNORECASE)
    hits = [f"{path}:{line} {found.group(0)}" for line, text in _strings(path)
            for found in (identifier.search(text), word.search(text)) if found]
    assert hits == []


def test_the_miner_examples_are_placeholders_one_per_fact_kind_and_condition():
    text = " ".join(intent.EXAMPLES.split())
    assert "Placeholders in angle brackets" in text and "<record>" in text
    for kind in typing.get_args(FactKind):
        assert re.search(rf"\b{kind}\b", text), kind
    for condition in typing.get_args(Condition):
        if condition != "none":
            assert f"condition {condition}" in text, condition


def test_the_writer_examples_are_placeholders_one_per_item_kind_and_condition():
    text = " ".join(writer.EXAMPLES.split())
    assert "Placeholders in angle brackets" in text and "<record>" in text
    for kind in items.ITEM_KINDS:
        assert f"- {kind}:" in text, kind
    for condition in ("world_or_policy", "agent"):
        assert f"- condition {condition}:" in text, condition
