"""The world the Builder's session tests share: the fixture trace file, scripted replies, and the
build tree two workdirs are compared on.

`tests/builder/test_extension.py`, `test_agent.py` and `test_tools.py` all drive one Builder
session over the same tau2 retail fixture, so the helpers live here rather than in whichever test
file happened to write them first.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from kullback.ai.provider import ModelReply, ToolCallRequest

COMPARED = ("bodies.json", "constraints.json", "gates.json", "environment.json", "replays.json",
            "tasks.json", "schema.json", "tool_sigs.json", "user_facts.json", "vocabulary.json")


def fixture_path(request) -> Path:
    """The small tau2 retail export every Builder session test builds from."""
    return Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"


def reply(content, *calls) -> ModelReply:
    """One scripted assistant turn: some text, and the tool calls it asks for in order."""
    return ModelReply(content=content, tool_calls=[ToolCallRequest(id=f"c{i}", name=n, arguments=a)
                                                  for i, (n, a) in enumerate(calls)])


def collect(aiter) -> list:
    """Every event of one harness run, in order."""
    async def go():
        return [event async for event in aiter]
    return asyncio.run(go())


def tree(workdir: Path) -> dict:
    """The build's artifacts by relative name, with the workdir's own absolute path (which
    replays.json records for every Run) replaced, so two workdirs compare on what was built."""
    def read(path: Path) -> bytes:
        return path.read_bytes().replace(str(workdir.resolve()).encode(), b"<workdir>").replace(
            str(workdir).encode(), b"<workdir>")
    out = {}
    for name in COMPARED:
        path = workdir / name
        if path.is_file():
            out[name] = read(path)
    for folder in ("intents", "tasks", "verifiers", "user_rules"):
        for path in sorted((workdir / folder).glob("*.json")):
            out[f"{folder}/{path.name}"] = read(path)
    for path in sorted((workdir / "runs").rglob("*.jsonl")):
        out[str(path.relative_to(workdir))] = read(path)
    return out
