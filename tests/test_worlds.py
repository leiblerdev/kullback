"""The two invented worlds keep the shape the builder tests are written against."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from kullback.builder import ingest


def _load_generator(worlds_dir: Path):
    """The generator module from its file, without putting fixtures on sys.path."""
    spec = importlib.util.spec_from_file_location(
        "make_worlds", worlds_dir / "make_worlds.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_the_generator_writes_byte_identical_files_when_run_twice(tmp_path, worlds_dir):
    """Determinism the brief demands: two runs agree byte for byte."""
    make_worlds = _load_generator(worlds_dir)
    first, second = tmp_path / "first", tmp_path / "second"
    make_worlds.build(first)
    make_worlds.build(second)
    assert _snapshot(first) == _snapshot(second)


def test_each_worlds_domain_files_parse(world):
    """db.json, tasks.json and the traces load, and the policy is not empty."""
    db = json.loads((world["dir"] / "db.json").read_text(encoding="utf-8"))
    tasks = json.loads((world["dir"] / "tasks.json").read_text(encoding="utf-8"))
    assert isinstance(db, dict) and db
    assert isinstance(tasks, list) and len(tasks) == 3
    assert len(world["traces"]["simulations"]) == 3
    assert (world["dir"] / "policy.md").read_text(encoding="utf-8").strip()


def test_each_world_carries_its_grader_fields_so_ingest_can_be_tested_stripping_them(world):
    """Like the shipped small fixture: reward checks and criteria stay in until ingest."""
    for sim in world["traces"]["simulations"]:
        assert {"reward_info", "task_id", "trial"} <= set(sim)
        assert "action_checks" in sim["reward_info"]
    for task in world["traces"]["tasks"]:
        assert "evaluation_criteria" in task


def test_each_world_ingests_with_every_call_paired(world, tmp_path):
    """Each world passes the intake gate with tools found and nothing rejected."""
    result = ingest.dry_run(world["traces_path"], tmp_path)
    assert result["passed"] is True, result["failures"]
    assert result["traces"] == 3
    assert result["tools"] > 0


def _nested(node) -> bool:
    """True when a value holds a list or a dict below its top level."""
    if isinstance(node, dict):
        return any(isinstance(value, (dict, list)) for value in node.values())
    if isinstance(node, list):
        return any(isinstance(value, (dict, list)) for value in node)
    return False


def _db_rows(db: dict) -> list:
    """Every record row in a world db."""
    return [row for rows in db.values() for row in rows.values()]


def _calls(traces: dict) -> list:
    """Every tool call in a world's simulations."""
    return [
        call
        for sim in traces["simulations"]
        for message in sim["messages"]
        for call in message.get("tool_calls") or []
    ]


def _read_db(world_dir, name: str) -> dict:
    """The parsed db.json of one world."""
    return json.loads((world_dir(name) / "db.json").read_text(encoding="utf-8"))


def test_world_a_stays_flat_in_records_and_arguments(world_dir, world_traces):
    """World_a keeps flat rows and scalar args, the shape most tests assume."""
    for row in _db_rows(_read_db(world_dir, "world_a")):
        assert not _nested(row)
    for call in _calls(world_traces("world_a")):
        assert not _nested(call["arguments"])


def test_world_b_nests_records_and_arguments(world_dir, world_traces):
    """World_b nests lists of objects in records and in tool arguments."""
    assert any(_nested(row) for row in _db_rows(_read_db(world_dir, "world_b")))
    assert any(
        _nested(call["arguments"]) for call in _calls(world_traces("world_b"))
    )


def test_world_bs_transcript_rows_carry_tool_calls_with_results(world_dir, tmp_path):
    """The second shipped format is detected, it holds every call with its answer,
    and it passes the intake gate."""
    from kullback.builder.sources import detect_format

    rows = [
        json.loads(line)
        for line in (world_dir("world_b") / "traces_small.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    assert detect_format(rows, jsonl=True).winner == "claude_code_jsonl"
    uses = [block for row in rows for block in row["message"]["content"]
            if block.get("type") == "tool_use"]
    results = [block for row in rows for block in row["message"]["content"]
               if block.get("type") == "tool_result"]
    assert len(uses) == len(results) > 0
    assert {block["tool_use_id"] for block in results} == {
        block["id"] for block in uses}
    result = ingest.dry_run(world_dir("world_b") / "traces_small.jsonl", tmp_path)
    assert result["passed"] is True, result["failures"]
    assert result["traces"] == 3
