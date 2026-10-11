"""World work a bound scenario carries beside its rows: grown rows and the customer's files (synth-add5).

A scenario read off the real world may need rows this world lacks (a second record to tell apart, a
past one, a note) and files the customer would hold (a confirmation, an invoice, a message thread,
an export). Rows are grown by code through the Starting state's own growth (builder/synth.py, D107),
so they obey the mined rules, and each is tagged under the Task as a grown row. Files are written by
one model step each, from the bound rows' values and the description only: a file carrying a value
neither holds is dropped, and so is one that repeats the corpus (D225's contamination scan).

Both reach the Task through its own overlay, never the shared world: grown rows as pinned rows, files
as rows of FILE_TABLE that the store's `read_file` tool lists and reads. A Task with neither keeps the
overlay, scenario source and tool set it had before, byte for byte. Nothing here names a corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from kullback.builder import compile_env, synth
from kullback.runner.records import Column, EntitySchema, FieldStat, ToolSig, read_json, write_json

# Where a store keeps each Task's files, as files/<task_id>/<name>, and the table its overlay pins them in.
FILES = "files"
FILE_TABLE = "task_files"
READ_FILE = "read_file"
# The formats a file may take; pdf is asked for by name but skipped while no pdf writer is in the project.
FORMATS = ("txt", "md", "csv", "json", "eml")
SKIPPED_FORMATS = ("pdf",)
# How many files one scenario carries at most, and how many rows of one table it may grow.
MAX_FILES = 3
MAX_GROWN = 2

READ_FILE_BODY = (
    "name = str(path or '').strip()\n"
    "if name.startswith('files/'):\n"
    "    name = name[len('files/'):]\n"
    "if name in ('', '.', 'files'):\n"
    "    return {'files': sorted(self.db.task_files)}\n"
    "if '/' in name or '\\\\' in name or name.startswith('.'):\n"
    "    raise ValueError('only a file under files/ can be read')\n"
    "if name not in self.db.task_files:\n"
    "    raise ValueError('no such file under files/')\n"
    "return {'name': name, 'text': self.db.task_files[name].text}")

FILE_PROMPT = (
    "A customer is about to write this request to a support agent:\n{description}\n\n"
    "They hold a file named {name} ({kind}). Write its full content as the {fmt} file would read. Use only "
    "these values, every number, id, date, amount, name and address in the file must be one of them:\n"
    "{values}\n\n"
    "Write the file content and nothing else.")

# A value-like token: a run holding a digit or an at sign, its edge punctuation trimmed.
VALUE_TOKEN = re.compile(r"[\w.@+:/-]*[\d@][\w.@+:/-]*")


def _tokens(text: str) -> set[str]:
    return {token.strip(".,:;/-+") for token in VALUE_TOKEN.findall(str(text or ""))} - {""}


def _leaf_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaf_values(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _leaf_values(item)]
    return [] if value is None else [str(value)]


def unfounded(text: str, rows: Iterable[Any], description: str) -> list[str]:
    """The value-like tokens of a file that no bound row and not the description carries."""
    allowed = set(_tokens(description))
    for row in rows or ():
        for leaf in _leaf_values(row):
            allowed |= _tokens(leaf) | {leaf.strip()}
    return sorted(token for token in _tokens(text) if token not in allowed)


def grow_needed(world: Any, needs: Iterable[dict], seed: int = 0) -> tuple[dict, dict[str, dict]]:
    """Grow the rows a scenario needs into a copy of the world's db: (the grown db, {table: {id: row}}).

    Each need names a table of this world and how many more rows of it the scenario wants, at most
    MAX_GROWN; a homed table grows with its parent, so a need on it is left out. The rows come from
    `synth.grow`, so they obey the rules mined off the observed rows; none is ever in the shared world."""
    db = json.loads(json.dumps(world.db))
    targets = _targets(world, db, needs)
    if not targets:
        return db, {}
    grown = synth.grow(db, world.schema, targets, seed=seed)
    rows = {table: {row_id: db[table][row_id] for row_id in ids if row_id in (db.get(table) or {})}
            for table, ids in grown.added.items()}
    return db, {table: found for table, found in rows.items() if found}


def _targets(world: Any, db: dict, needs: Iterable[dict]) -> dict[str, int]:
    """Each needed table of this world, not homed, to the row count it should reach."""
    homed = set(world.schema.homes or {})
    tables = set(world.schema.tables or ())
    targets = {}
    for need in needs or ():
        table, count = str(need.get("table") or ""), min(int(need.get("count") or 0), MAX_GROWN)
        if table in tables and table not in homed and count > 0:
            targets[table] = len(synth.table_rows(db, world.schema, table)) + count
    return targets


def write_files(model: Any, scenario: dict, rows: list[Any], description: str,
                strings: list[str]) -> tuple[dict[str, str], dict[str, int]]:
    """Each file the scenario names, written by one model call from the rows and the description only;
    ({name: text}, {reason: count}) with the files dropped as unfounded, contaminated, skipped or unwritten."""
    files: dict[str, str] = {}
    dropped: dict[str, int] = {}
    values = sorted({leaf for row in rows for leaf in _leaf_values(row) if leaf})
    for spec in list(scenario.get("files") or ())[:MAX_FILES]:
        name, text, reason = _one_file(model, spec, rows, description, strings, values)
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
        else:
            files[name] = text
    return files, dropped


def _one_file(model: Any, spec: Any, rows: list[Any], description: str, strings: list[str],
              values: list[str]) -> tuple[str, str, str]:
    """(name, text, drop reason or ""): skipped, unnamed, unwritten, unfounded or contaminated."""
    from kullback import domain as domain_mod

    name, fmt = _file_name(spec)
    if fmt in SKIPPED_FORMATS:
        return name, "", "skipped"
    if not name:
        return name, "", "unnamed"
    text = _file_text(model, description, name, str(spec.get("kind") or fmt), fmt, values)
    if not text:
        return name, "", "unwritten"
    if unfounded(text, rows, description):
        return name, text, "unfounded"
    if domain_mod.contaminated({"goal": text}, strings):
        return name, text, "contaminated"
    return name, text, ""


def _file_name(spec: Any) -> tuple[str, str]:
    """A file's plain name and format, or ("", format) where the name is a path or the format unknown."""
    name = str((spec or {}).get("name") or "").strip() if isinstance(spec, dict) else ""
    fmt = name.rsplit(".", 1)[-1].casefold() if "." in name else ""
    if fmt in SKIPPED_FORMATS:
        return name, fmt
    if not name or "/" in name or "\\" in name or name.startswith(".") or fmt not in FORMATS:
        return "", fmt
    return name, fmt


def _file_text(model: Any, description: str, name: str, kind: str, fmt: str, values: list[str]) -> str:
    try:
        reply = model.query([{"role": "user", "content": FILE_PROMPT.format(
            description=description, name=name, kind=kind, fmt=fmt, values="\n".join(f"- {v}" for v in values))}])
    except Exception:  # noqa: BLE001 - a file the step cannot write is one file fewer, and is counted
        return ""
    return str(getattr(reply, "content", "") or "").strip()


def _version(table: str, row_id: str, row: Any) -> str:
    return hashlib.sha256(json.dumps([table, row_id, row], sort_keys=True).encode()).hexdigest()[:16]


def task_overlay(task_id: str, grown: dict[str, dict], files: dict[str, str]) -> dict:
    """The Task's overlay: its grown rows and its files pinned for it alone, nothing where it has neither."""
    pinned = [(table, row_id, row) for table, rows in sorted(grown.items()) for row_id, row in sorted(rows.items())]
    pinned += [(FILE_TABLE, name, {"name": name, "text": text}) for name, text in sorted(files.items())]
    rows, values = [], {}
    for table, row_id, row in pinned:
        version = _version(table, row_id, row)
        rows.append({"table": table, "id": row_id, "version_hash": version})
        values[version] = row
    return {"overlay": {"task_id": task_id, "rows": rows}, "values": values}


def write_task_files(store: Path, task_id: str, files: dict[str, str]) -> None:
    """The Task's files on disk under files/<task_id>/, where the miner and the goal step read them."""
    for name, text in sorted(files.items()):
        path = Path(store) / FILES / task_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _read_file_sig() -> ToolSig:
    return ToolSig(name=READ_FILE, kind="read",
                   args_fields=[FieldStat(name="path", types=["str"], optional=False)],
                   args_schema={"type": "object", "properties": {"path": {"type": ["str"]}}, "required": ["path"]})


def install_read_file(store: Path) -> bool:
    """Give the store its own tool set with `read_file` and FILE_TABLE, once some Task has files.

    The store's schema, signatures and bodies become its own files in place of the links to the parent,
    and its env/ its own folder: every parent file linked but the rendered toolkit, which is rendered
    again so the Runner serves `read_file`. The parent is never written. True where it was installed."""
    store = Path(store)
    sigs = read_json(store / "tool_sigs.json", []) or []
    if any(str(row.get("name")) == READ_FILE for row in sigs):
        return False
    schema = EntitySchema.model_validate(read_json(store / "schema.json", {}) or {})
    schema.tables = sorted(set(schema.tables or ()) | {FILE_TABLE})
    schema.columns = [*(schema.columns or ()), *(Column(table=FILE_TABLE, name=name, class_="hard")
                                                 for name in ("name", "text"))]
    sig_rows = [*sigs, json.loads(_read_file_sig().model_dump_json())]
    bodies = dict(read_json(store / "bodies.json", {}) or {}, **{READ_FILE: READ_FILE_BODY})
    for name, body in (("schema.json", json.loads(schema.model_dump_json())), ("tool_sigs.json", sig_rows),
                       ("bodies.json", bodies)):
        _unlink(store / name)
        write_json(store / name, body)
    _own_env(store, schema, [ToolSig.model_validate(row) for row in sig_rows], bodies)
    return True


def _unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()


def _own_env(store: Path, schema: EntitySchema, sigs: list[ToolSig], bodies: dict) -> None:
    env = store / "env"
    parent_env = env.resolve() if env.is_symlink() else None
    if parent_env is not None:
        env.unlink()
        env.mkdir()
        for entry in sorted(parent_env.iterdir()):
            if entry.name not in ("tools.py", "data_model.py"):
                (env / entry.name).symlink_to(os.path.relpath(entry, env), target_is_directory=entry.is_dir())
    elif env.is_dir():
        for name in ("tools.py", "data_model.py"):
            _unlink(env / name)
    else:
        env.mkdir(parents=True)
    (env / "data_model.py").write_text(compile_env.render_data_model(schema), encoding="utf-8")
    (env / "tools.py").write_text(compile_env.render_tools(schema, sigs, bodies), encoding="utf-8")


__all__ = ["FILES", "FILE_PROMPT", "FILE_TABLE", "FORMATS", "MAX_FILES", "MAX_GROWN", "READ_FILE",
           "READ_FILE_BODY", "SKIPPED_FORMATS", "grow_needed", "install_read_file",
           "task_overlay", "unfounded", "write_files", "write_task_files"]
