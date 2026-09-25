"""Real-corpus checks, kept as smoke: they need the full raw corpus size.

Unit tests in tests/builder prove the same rules on the invented worlds. This tier keeps
the checks that only make sense on the real data: clustering pair F1 against the corpus's
own task ids over hundreds of runs, at two thresholds, and mining every tool, table,
id pattern and column class off the whole corpus.
"""

from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path

import pytest

from conftest import PTR
from kullback.builder.cluster import cluster_runs
from kullback.builder.mine import gate_tools as mine_gate_tools
from kullback.builder.mine import mine_schema, mine_tools
from kullback.runner.records import RawPtr, ToolCall, ToolCallError, Trace, Turn

# The corpus's write tools, from vendor/tau2-bench retail tools.py; mine.py derives the same set.
TAU2_RETAIL_WRITES = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}


def corpus_traces_and_truth(raw_dir):
    """Traces straight off a raw file, with tau2's own task_id kept aside as the test's ground truth.

    The task_id is a grader field (D66): it is the yardstick this test measures against and is never
    handed to `cluster_runs`, which sees only turns and tool calls.
    """
    paths = sorted(raw_dir.glob("*retail*4trials.json"))
    if not paths:
        pytest.skip("no tau2 retail raw file")
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    traces, truth = [], {}
    for sim in data["simulations"]:
        messages = sim["messages"]
        failed = {m.get("id") for m in messages if m.get("role") == "tool" and m.get("error")}
        traces.append(
            Trace(
                trace_id=sim["id"],
                raw_hash="raw",
                ingest_version="0",
                source="tau2",
                turns=[Turn(idx=i, role=m["role"], content=m.get("content"), raw_ptr=PTR)
                       for i, m in enumerate(messages)],
                tool_calls=[
                    ToolCall(
                        id=tc.get("id"),
                        name=tc["name"],
                        args=tc.get("arguments") or {},
                        error=(
                            ToolCallError(**{"class": "business_error", "payload": ""})
                            if tc.get("id") in failed
                            else None
                        ),
                        raw_ptr=PTR,
                    )
                    for m in messages
                    for tc in (m.get("tool_calls") or [])
                ],
                raw_ptr=PTR,
            )
        )
        truth[sim["id"]] = str(sim["task_id"])
    return traces, truth


def pair_f1(tasks, truth) -> float:
    """Pair-counting F1 of the Task membership against tau2's task_ids."""
    task_of = {run_id: task.id for task in tasks for run_id in task.run_ids}
    tp = fp = fn = 0
    for a, b in itertools.combinations(sorted(truth), 2):
        same_task, same_truth = task_of[a] == task_of[b], truth[a] == truth[b]
        tp += same_task and same_truth
        fp += same_task and not same_truth
        fn += same_truth and not same_task
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


@pytest.mark.smoke
def test_the_whole_corpus_clusters_close_to_tau2s_own_tasks(raw_dir):
    """The rule has to hold on 456 Runs, not only on three: pair F1 against tau2's task_ids."""
    traces, truth = corpus_traces_and_truth(raw_dir)
    assert len(traces) > 100
    _, tasks = cluster_runs(traces, TAU2_RETAIL_WRITES)
    assert pair_f1(tasks, truth) >= 0.65


@pytest.mark.smoke
def test_the_corpus_score_does_not_hang_on_the_exact_threshold(raw_dir):
    """The shipped rule scored 0.28 at its own default and 0.72 at 0.6; that cliff is what idf removes."""
    traces, truth = corpus_traces_and_truth(raw_dir)
    scores = {th: pair_f1(cluster_runs(traces, TAU2_RETAIL_WRITES, threshold=th)[1], truth) for th in (0.3, 0.5)}
    assert min(scores.values()) >= 0.65, scores


# --- moved from tests/builder/test_mine.py: needs the full raw corpus size ---


def retail_tool_names() -> set[str]:
    # Kullback keeps vendor/ under the package root; the brain keeps it one level up, at
    # monitoring-tool/vendor; mutmut runs this file from a copy under mutants/tests. Walk up
    # from here and take the first ancestor that has it, same idea as raw_dir in conftest.
    rel = Path("vendor") / "tau2-bench" / "src" / "tau2" / "domains" / "retail" / "tools.py"
    for root in Path(__file__).resolve().parents:
        path = root / rel
        if path.is_file():
            break
    else:
        pytest.skip("vendor/tau2-bench not present")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    names.add(item.name)
    assert len(names) >= 16
    return names


def traces_from_raw(raw: dict, file_hash: str = "fixture") -> list[Trace]:
    """Minimal tau2 raw -> Trace conversion, so these tests do not depend on ingest.py."""
    traces = []
    for sim_index, sim in enumerate(raw["simulations"]):
        calls: dict[str, ToolCall] = {}
        order: list[str] = []
        for msg_index, msg in enumerate(sim["messages"]):
            for raw_call in msg.get("tool_calls") or []:
                calls[raw_call["id"]] = ToolCall(
                    id=raw_call["id"],
                    name=raw_call["name"],
                    args=raw_call.get("arguments") or {},
                    requestor=raw_call.get("requestor") or "assistant",
                    raw_ptr=RawPtr(file_hash=file_hash, sim_index=sim_index, msg_index=msg_index),
                )
                order.append(raw_call["id"])
            if msg.get("role") == "tool" and msg.get("id") in calls:
                call = calls[msg["id"]]
                call.result = msg.get("content")
                if msg.get("error"):
                    call.error = ToolCallError(class_="business_error", payload=msg.get("content"))
        traces.append(
            Trace(
                trace_id=sim["id"],
                raw_hash=file_hash,
                ingest_version="test",
                source="tau2",
                tool_calls=[calls[i] for i in order],
                raw_ptr=RawPtr(file_hash=file_hash, sim_index=sim_index),
            )
        )
    return traces


def sig_by_name(sigs, name):
    return next(s for s in sigs if s.name == name)


def col(schema, table, name):
    return next(c for c in schema.columns if c.table == table and c.name == name)


@pytest.mark.smoke
@pytest.mark.slow
def test_mining_the_whole_corpus_finds_every_tool_table_id_pattern_and_column_class(retail_raw_files, raw_dir, tau2_retail_dir):
    # One corpus only: raw_dir also holds other corpora's traces, checked separately.
    traces: list[Trace] = []
    for path in retail_raw_files:
        traces += traces_from_raw(json.loads(path.read_text(encoding="utf-8")), file_hash=path.name)
    assert len(traces) > 100
    sigs = mine_tools(traces)
    mined = {s.name for s in sigs}
    missing = retail_tool_names() - mined
    assert not missing, f"tools in tau2's retail tools.py that the traces never showed: {sorted(missing)}"
    assert mine_gate_tools(sigs).passed is True
    order = sig_by_name(sigs, "get_order_details")
    assert order.evidence_strength.call_count > 1000
    assert {f.name for f in order.result_schema} >= {"order_id", "status", "items"}
    writes = {s.name for s in sigs if s.kind == "write"}
    assert "cancel_pending_order" in writes and "modify_user_address" in writes
    traces: list[Trace] = []
    for path in sorted(raw_dir.glob("*.json")):
        traces += traces_from_raw(json.loads(path.read_text(encoding="utf-8")), file_hash=path.name)
    schema = mine_schema(traces, db_json_path=tau2_retail_dir / "db.json")
    # "items" joins the three db.json tables: get_item_details returns rows keyed by item_id
    assert set(schema.tables) >= {"orders", "products", "users"}
    assert col(schema, "orders", "status").class_ == "hard"
    assert set(schema.id_patterns) >= {"orders.order_id", "users.user_id", "products.product_id"}
