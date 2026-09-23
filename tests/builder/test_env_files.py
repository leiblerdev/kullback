"""The Builder's files: bodies round-trip through per-tool files, expose hides held-out calls."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from kullback.builder import compile_env, env_files
from kullback.gates.bindings import rulings_for
from kullback.gates.confinement import ALLOWED_IMPORTS, CONTEXT_READS, DENIED_BUILTINS, PROVIDED_HELPERS
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, read_json, write_json
from kullback.runner.world.loading import SandboxError, load_toolkit
from tests.episode.invented import write_env

PTR = RawPtr(file_hash="f" * 64, sim_index=0, msg_index=0)

RENAME_BODY = (
    "row = self.db.widgets.get(widget_id)\n"
    "if row is None:\n"
    "    raise KeyError(widget_id)\n"
    "row.label = label\n"
    'return {"widget_id": widget_id, "label": label}'
)

DESCRIBE_BODY = (
    "row = self.db.widgets.get(widget_id)\n"
    "if row is None:\n"
    "    raise KeyError(widget_id)\n"
    'return {"widget_id": widget_id, "label": row.label}'
)


def _trace(trace_id: str, calls: list[ToolCall]) -> Trace:
    turns = [Turn(idx=0, role="assistant", content="Done.", raw_ptr=PTR)]
    return Trace(trace_id=trace_id, raw_hash="r" * 64, ingest_version="1", source="test",
                 turns=turns, tool_calls=calls, raw_ptr=PTR)


def _call(call_id: str, name: str, trace_id: str, args: dict, result: dict) -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args, result=result, raw_ptr=PTR, trace_id=trace_id)


def test_body_of_reads_the_function_body_statements_dedented():
    source = "from typing import Any\n\ndef rename_widget(self, widget_id, label):\n" + "\n".join(
        f"    {line}" for line in RENAME_BODY.splitlines()) + "\n"
    name, body = env_files.body_of(source)
    assert name == "rename_widget"
    assert body == RENAME_BODY


def test_body_of_refuses_a_file_with_no_function():
    try:
        env_files.body_of("x = 1\n")
    except ValueError:
        return
    raise AssertionError("a file with no function definition must be refused")


def test_explode_writes_missing_tool_files_and_leaves_edited_ones(tmp_path):
    root = write_env(tmp_path / "work")
    write_json(root / "bodies.json", {"rename_widget": RENAME_BODY, "describe_widget": DESCRIBE_BODY})
    written = env_files.explode(root)
    assert sorted(written) == ["describe_widget", "rename_widget"]
    for name in ("rename_widget", "describe_widget"):
        path = root / "env" / "tools" / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert tree.body[-1].name == name
    edited = root / "env" / "tools" / "rename_widget.py"
    edited.write_text(edited.read_text(encoding="utf-8") + "# the model was here\n", encoding="utf-8")
    assert env_files.explode(root) == []
    assert edited.read_text(encoding="utf-8").endswith("# the model was here\n")


def test_regenerate_round_trips_bodies_json_text_and_module_source(tmp_path):
    root = write_env(tmp_path / "work")
    bodies = {"rename_widget": RENAME_BODY, "describe_widget": DESCRIBE_BODY}
    write_json(root / "bodies.json", bodies)
    before_text = (root / "bodies.json").read_text(encoding="utf-8")
    env_files.explode(root)
    report = env_files.regenerate(root)
    assert (root / "bodies.json").read_text(encoding="utf-8") == before_text
    assert report["bodies"] == bodies
    schema = env_files._read_schema(root)
    sigs = env_files._read_sigs(root)
    assert compile_env.module_source(schema, sigs, report["bodies"]) == \
        compile_env.module_source(schema, sigs, bodies)
    assert (root / "env" / "tools.py").read_text(encoding="utf-8")
    assert (root / "env" / "data_model.py").read_text(encoding="utf-8")


def test_expose_copies_the_read_surface_but_no_held_out_call_and_no_verifier(tmp_path):
    root = write_env(tmp_path / "work")
    traces = root / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    shown = _trace("shown1", [_call("s1", "rename_widget", "shown1",
                                    {"widget_id": "w1", "label": "striped"},
                                    {"widget_id": "w1", "label": "striped"})])
    held = _trace("held1", [_call("h1", "rename_widget", "held1",
                                  {"widget_id": "w1", "label": "dotted"},
                                  {"widget_id": "w1", "label": "dotted"})])
    for trace in (shown, held):
        (traces / f"{trace.trace_id}.json").write_text(
            json.dumps(trace.model_dump(mode="json", by_alias=True)), encoding="utf-8")
    (root / "anchor.json").write_text(json.dumps({"held_out": {"widget_task": ["held1"]}}),
                                      encoding="utf-8")
    report = env_files.expose(root)
    assert "schema.json" in report["files"]
    assert (root / "env" / "schema.json").is_file()
    assert (root / "env" / "tasks" / "widget_task.json").is_file()
    lines = (root / "env" / "calls" / "rename_widget.jsonl").read_text(encoding="utf-8").splitlines()
    call_ids = [json.loads(line)["id"] for line in lines]
    assert call_ids == ["s1"]
    assert not (root / "env" / "verifiers").exists()
    assert "verifier" not in (root / "env" / "calls" / "rename_widget.jsonl").read_text()


def test_regenerate_reads_back_an_edited_body(tmp_path):
    root = write_env(tmp_path / "work")
    write_json(root / "bodies.json", {"rename_widget": RENAME_BODY})
    env_files.explode(root)
    path = root / "env" / "tools" / "rename_widget.py"
    path.write_text(path.read_text(encoding="utf-8").replace('row.label = label', 'row.label = label or row.label'),
                    encoding="utf-8")
    report = env_files.regenerate(root)
    assert "label or row.label" in report["bodies"]["rename_widget"]
    assert "label or row.label" in read_json(root / "bodies.json")["rename_widget"]


def test_expose_without_traces_still_copies_the_world_files(tmp_path):
    root = write_env(tmp_path / "work")
    report = env_files.expose(root)
    assert set(report["files"]) >= {"schema.json", "tool_sigs.json", "db.json"}
    assert Path(root / "env" / "db.json").is_file()


def test_explode_seeds_a_stub_per_signature_without_a_body_and_regenerate_renders_the_toolkit(tmp_path):
    root = write_env(tmp_path / "work")
    (root / "env" / "tools.py").unlink()
    (root / "env" / "data_model.py").unlink()
    write_json(root / "bodies.json", {"rename_widget": RENAME_BODY})
    assert sorted(env_files.explode(root)) == ["describe_widget", "rename_widget"]
    stub = root / "env" / "tools" / "describe_widget.py"
    assert env_files.body_of(stub.read_text(encoding="utf-8")) == ("describe_widget", env_files.STUB_BODY)
    assert env_files.is_stub(stub)
    assert not env_files.is_stub(root / "env" / "tools" / "rename_widget.py")
    env_files.regenerate(root)
    assert (root / "env" / "tools.py").is_file() and (root / "env" / "data_model.py").is_file()
    assert read_json(root / "bodies.json", {})["describe_widget"] == env_files.STUB_BODY


def test_an_edited_stub_survives_a_second_explode(tmp_path):
    root = write_env(tmp_path / "work")
    env_files.explode(root)
    stub = root / "env" / "tools" / "describe_widget.py"
    stub.write_text(stub.read_text(encoding="utf-8").replace(env_files.STUB_BODY, DESCRIBE_BODY.replace(
        "\n", "\n    ")), encoding="utf-8")
    edited = stub.read_text(encoding="utf-8")
    assert env_files.explode(root) == []
    assert stub.read_text(encoding="utf-8") == edited
    assert not env_files.is_stub(stub)


# --- the write gate reads the module the Runner loads ---


def _mined_args(root: Path) -> None:
    """The mined arguments, as derive writes them: the rendered module takes its signature from here."""
    sigs = read_json(root / "tool_sigs.json", [])
    for sig in sigs:
        sig["args_fields"] = [{"name": name, "types": ["str"], "optional": False}
                              for name in sig.get("args_schema", {}).get("properties", {})]
    write_json(root / "tool_sigs.json", sigs)


def _write_rename(root: Path, body: str) -> dict:
    """Write one rename body as the model would, and rule on it the way the session hook does."""
    _mined_args(root)
    env_files.explode(root)
    path = root / "env" / "tools" / "rename_widget.py"
    sig = next(sig for sig in env_files._read_sigs(root) if sig.name == "rename_widget")
    path.write_text(env_files._tool_file("rename_widget", sig, body), encoding="utf-8")
    execute = env_files.executor(root)
    rulings = rulings_for(root / "env", "tools/rename_widget.py", root, execute=execute)
    return {"confined": next(r for r in rulings if r.stage == "confined"),
            "module": execute.render(path.read_text(encoding="utf-8"))}


def _runner_refusal(module: str) -> str:
    """What the Runner's loader says about this module: its refusal, or "" when it loads."""
    try:
        load_toolkit(module, {})
    except SandboxError as exc:
        return str(exc)
    except Exception:  # noqa: BLE001, a module past confinement may still fail on an empty db
        return ""
    return ""


def test_a_body_importing_a_module_off_the_list_fails_the_write_gate_as_the_runner_would(tmp_path):
    root = write_env(tmp_path / "work")
    ruled = _write_rename(root, "import ast\n" + RENAME_BODY)
    refusal = _runner_refusal(ruled["module"])
    assert not ruled["confined"].passed
    assert "rename_widget imports ast" in ruled["confined"].failures
    assert refusal and all(failure in refusal for failure in ruled["confined"].failures)


def test_a_body_naming_eval_fails_the_write_gate_as_the_runner_would(tmp_path):
    root = write_env(tmp_path / "work")
    ruled = _write_rename(root, 'label = eval("label")\n' + RENAME_BODY)
    refusal = _runner_refusal(ruled["module"])
    assert not ruled["confined"].passed
    assert "rename_widget uses eval" in ruled["confined"].failures
    assert refusal and all(failure in refusal for failure in ruled["confined"].failures)


def test_a_body_calling_the_provided_helper_passes_the_write_gate_and_the_runner(tmp_path):
    root = write_env(tmp_path / "work")
    body = RENAME_BODY.replace("row.label = label", "row.label = str(evaluate_arithmetic(label))")
    ruled = _write_rename(root, body)
    assert ruled["confined"].passed, ruled["confined"].failures
    assert _runner_refusal(ruled["module"]) == ""


def test_every_stub_header_names_the_gate_s_allowed_imports_helpers_and_context_reads(tmp_path):
    root = write_env(tmp_path / "work")
    env_files.explode(root)
    text = (root / "env" / "tools" / "describe_widget.py").read_text(encoding="utf-8")
    header = text.split("from typing")[0]
    assert ", ".join(sorted(ALLOWED_IMPORTS)) in header
    assert all(helper in header for helper in PROVIDED_HELPERS)
    assert all(read in header for read in CONTEXT_READS)
    refused = next(line for line in header.splitlines() if line.startswith("# Names refused: "))
    assert refused == f"# Names refused: {', '.join(sorted(DENIED_BUILTINS))}."
    assert len(header.splitlines()) <= 8
