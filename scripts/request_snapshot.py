"""Snapshot the compile_policy and compile_tools request envelopes (D246).

Assembles the request messages and tool lists for ten invented inputs and
writes them to a file, so the bytes on origin/main and on the head compare
with a plain diff. Run from the worktree root:

    uv run python scripts/request_snapshot.py /tmp/request_snapshot.json

Inputs are invented harbor ledger names, never customer data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kullback.builder import compile_env as ce  # noqa: E402
from kullback.builder.policy import _stable_system as _policy_head  # noqa: E402
from kullback.builder.policy import split_policy  # noqa: E402
from kullback.runner.records import Column, EntitySchema, FieldStat, RawPtr, ToolCall, ToolSig  # noqa: E402

WORDS = ("alder", "birch", "cedar", "dune", "elm", "fern", "grove", "heath", "iris", "juniper")


def _policy_snapshot() -> dict:
    policy = " ".join(f"The harbor ledger {word} rule holds." for word in WORDS)
    sentences = split_policy(policy)
    head = _policy_head(policy)
    rows = []
    for sentence in sentences:
        rows.append({
            "index": sentence.index,
            "system_sha256": hashlib.sha256(head.encode("utf-8")).hexdigest(),
            "system_chars": len(head),
            "user": f"Section: {sentence.section or 'top level'}\nRule: {sentence.text}",
        })
    return {"policy_head_sha256": hashlib.sha256(head.encode("utf-8")).hexdigest(),
            "policy_head_chars": len(head),
            "sentences": rows}


def _schema() -> EntitySchema:
    return EntitySchema(
        tables=["berths", "vessels"],
        columns=[
            Column(table="berths", name="berth_id", **{"class": "hard"}),
            Column(table="berths", name="status", **{"class": "hard"}),
            Column(table="vessels", name="hull", **{"class": "semantic"}),
        ],
    )


def _tools_snapshot() -> dict:
    schema = _schema()
    names = [f"check_berth_{word}" for word in WORDS]
    sigs = [ToolSig(name=name, kind="read", unclassified=False,
                    args_fields=[FieldStat(name="berth_id", types=["str"])],
                    result_schema=[]) for name in names]
    head = ce._stable_system(schema, names, True, "")
    tools = [{"name": spec["name"], "description": spec["description"],
              "parameters": spec["parameters"]} for spec in ce.BUILDER_TOOLS]
    rows = []
    for sig in sigs:
        call = ToolCall(id="c0", name=sig.name, args={"berth_id": "b01"},
                        result={"berth_id": "b01", "status": "open"},
                        raw_ptr=RawPtr(file_hash="snapshot", sim_index=0))
        messages = ce.body_messages(sig, [call], schema=schema, tool_names=names,
                                    builder_tools=True)
        system = messages[0]["content"]
        rows.append({
            "tool": sig.name,
            "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "system_chars": len(system),
            "system_is_head": system == head,
        })
    return {"tools_head_sha256": hashlib.sha256(head.encode("utf-8")).hexdigest(),
            "tools_head_chars": len(head),
            "helper_order": [spec["name"] for spec in ce.BUILDER_TOOLS],
            "helper_specs": tools,
            "tools": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", help="file the snapshot JSON is written to")
    args = parser.parse_args()
    snapshot = {"compile_policy": _policy_snapshot(), "compile_tools": _tools_snapshot()}
    Path(args.out).write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
