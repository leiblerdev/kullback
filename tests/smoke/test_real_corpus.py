"""Real-export checks, kept as smoke: they need the full recorded export size.

Unit tests in tests/builder prove the same rules on the invented worlds. This tier keeps
the checks that only make sense on the real data: clustering pair F1 against each discovered
export's own task ids over hundreds of runs, at two thresholds, and mining every tool, table,
id pattern and column class off each discovered export. Discovery finds the exports and the
vendored domain files; nothing here names one.
"""

from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path

import pytest

from conftest import PTR
from kullback.builder.cluster import cluster_runs
from kullback.builder.ingest import dry_run
from kullback.builder.mine import gate_tools as mine_gate_tools
from kullback.builder.mine import mine_schema, mine_tools
from kullback.runner.records import RawPtr, ToolCall, ToolCallError, Trace, Turn

_MAPPED: dict[str, list[Path]] = {}


def _vendor_dirs() -> list[Path]:
    """Every vendor/ checkout above this file; the harness and the brain keep it in different places."""
    found = []
    for root in Path(__file__).resolve().parents:
        vendor = root / "vendor"
        if vendor.is_dir() and vendor not in found:
            found.append(vendor)
    return found


def _domain_tools_files() -> list[Path]:
    """Every domain tools.py in the vendor checkouts, in path order."""
    paths = []
    for vendor in _vendor_dirs():
        paths.extend(
            path for path in sorted(vendor.rglob("tools.py")) if path.parent.parent.name == "domains"
        )
    return paths


def _tool_names_of(path: Path) -> set[str]:
    """The public tool functions one domain tools file defines, parsed, never imported."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    names.add(item.name)
    assert len(names) >= 2
    return names


def _recorded_sims(body: object) -> list | None:
    """The simulations of a recorded export body, or None when the body is another shape."""
    sims = body.get("simulations") if isinstance(body, dict) else None
    if not isinstance(sims, list) or not sims:
        return None
    for sim in sims:
        if not isinstance(sim, dict):
            return None
        if not isinstance(sim.get("messages"), list) or "id" not in sim or "task_id" not in sim:
            return None
    return sims


def _discover(raw_dir: Path, workdir: Path) -> list[Path]:
    """Every raw_dir JSON the intake maps that carries recorded simulations, in name order."""
    kept = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _recorded_sims(body) is None:
            continue
        try:
            mapped = dry_run(path, workdir)
        except Exception:
            continue
        if isinstance(mapped, dict) and mapped.get("passed"):
            kept.append(path)
    return kept


def _mapped_export_files(raw_dir: Path, workdir: Path) -> list[Path]:
    """The discovered exports, found once per raw_dir; every smoke test shares them."""
    key = str(raw_dir)
    if key not in _MAPPED:
        _MAPPED[key] = _discover(raw_dir, workdir)
    return _MAPPED[key]


def _traces_and_truth(path: Path) -> tuple[list[Trace], dict]:
    """Traces off one export file, with its own task ids kept aside as ground truth.

    The task_id is a grader field (D66): it is the yardstick this test measures against and is never
    handed to `cluster_runs`, which sees only turns and tool calls.
    """
    sims = _recorded_sims(json.loads(path.read_text(encoding="utf-8"))) or []
    traces, truth = [], {}
    for sim in sims:
        messages = sim["messages"]
        failed = {m.get("id") for m in messages if m.get("role") == "tool" and m.get("error")}
        traces.append(
            Trace(
                trace_id=sim["id"],
                raw_hash="raw",
                ingest_version="0",
                source="recorded",
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
    """Pair-counting F1 of the Task membership against the recorded task ids."""
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


def _mined_writes(traces: list[Trace]) -> tuple[list, set[str]]:
    """Tool sigs mined off traces and the write names among them, which clustering groups by."""
    sigs = mine_tools(traces)
    return sigs, {s.name for s in sigs if s.kind == "write"}


@pytest.mark.smoke
def test_the_whole_corpus_clusters_close_to_the_exports_own_tasks(raw_dir, workdir):
    """The rule has to hold on hundreds of Runs, not only on three: pair F1 per discovered export."""
    files = _mapped_export_files(raw_dir, workdir)
    if not files:
        pytest.skip("no mapped trace export in raw_dir")
    for path in files:
        traces, truth = _traces_and_truth(path)
        assert len(traces) > 100
        _, writes = _mined_writes(traces)
        _, tasks = cluster_runs(traces, writes)
        assert pair_f1(tasks, truth) >= 0.65


@pytest.mark.smoke
def test_the_corpus_score_does_not_hang_on_the_exact_threshold(raw_dir, workdir):
    """The shipped rule scored 0.28 at its own default and 0.72 at 0.6; that cliff is what idf removes."""
    files = _mapped_export_files(raw_dir, workdir)
    if not files:
        pytest.skip("no mapped trace export in raw_dir")
    for path in files:
        traces, truth = _traces_and_truth(path)
        _, writes = _mined_writes(traces)
        scores = {th: pair_f1(cluster_runs(traces, writes, threshold=th)[1], truth) for th in (0.3, 0.5)}
        assert min(scores.values()) >= 0.65, scores


# --- moved from tests/builder/test_mine.py: needs the full raw corpus size ---


def traces_from_raw(raw: dict, file_hash: str = "fixture") -> list[Trace]:
    """Minimal recorded raw -> Trace conversion, so these tests do not depend on ingest.py."""
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
                source="recorded",
                tool_calls=[calls[i] for i in order],
                raw_ptr=RawPtr(file_hash=file_hash, sim_index=sim_index),
            )
        )
    return traces


def _check_export_mining(path: Path, tools_files: list[Path]) -> None:
    """One export's traces mine every tool of each linked domain file, with gated evidence."""
    traces, _ = _traces_and_truth(path)
    assert len(traces) > 100
    sigs, writes = _mined_writes(traces)
    assert sigs
    assert mine_gate_tools(sigs).passed is True
    assert writes, f"no write tool mined off {path.name}"
    top = max(sigs, key=lambda s: s.evidence_strength.call_count)
    assert top.evidence_strength.call_count > 1000
    assert top.result_schema, f"top mined tool carries no result schema off {path.name}"
    mined = {s.name for s in sigs}
    linked = [item for item in tools_files if _tool_names_of(item) & mined]
    assert linked, f"no vendored domain file matches the tools mined off {path.name}"
    for item in linked:
        missing = _tool_names_of(item) - mined
        assert not missing, f"tools in {item} never shown by {path.name}: {sorted(missing)}"


def _check_store_schema(traces: list[Trace], db_json_path: Path) -> None:
    """The mined store schema covers every table the store fixture declares, with ids and hard columns."""
    schema = mine_schema(traces, db_json_path=db_json_path)
    tables = set(json.loads(db_json_path.read_text(encoding="utf-8")))
    assert tables, f"no tables declared in {db_json_path}"
    assert set(schema.tables) >= tables
    id_tables = {pattern.split(".")[0] for pattern in schema.id_patterns}
    assert id_tables >= tables
    assert any(column.class_ == "hard" for column in schema.columns)


@pytest.mark.smoke
@pytest.mark.slow
def test_mining_the_whole_corpus_finds_every_tool_table_id_pattern_and_column_class(raw_dir, workdir, tau2_retail_dir):
    """Every discovered export mines its domain tools; the whole folder mines the store schema."""
    files = _mapped_export_files(raw_dir, workdir)
    if not files:
        pytest.skip("no mapped trace export in raw_dir")
    tools_files = _domain_tools_files()
    if not tools_files:
        pytest.skip("no vendored domain tools found")
    for path in files:
        _check_export_mining(path, tools_files)
    traces: list[Trace] = []
    for path in files:
        traces += traces_from_raw(json.loads(path.read_text(encoding="utf-8")), file_hash=path.name)
    _check_store_schema(traces, tau2_retail_dir / "db.json")
