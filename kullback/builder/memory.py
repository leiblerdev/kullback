"""The Builder's memory: a tree of Builder versions on disk (D64, D69, D82) and the cross-customer
lessons file with its anonymization gate (D87)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from pydantic import Field

from kullback.runner.records import Record, SetAsideLesson, content_hash

# --- errors ---


class TreeError(Exception):
    """Anything the memory tree refuses to do."""


class NodeNotFound(TreeError):
    """No node with that id under this workdir."""


class OpenProposalError(TreeError):
    """One change per round (D82): the open proposal must be accepted or rejected first."""


class ReadOnlyEditError(TreeError):
    """The improvement agent may not edit the Runner, the runs directory, the raw store or the evaluator (D69)."""


class LessonError(TreeError):
    """A lesson the file refuses to hold in that shape (D87)."""


class RetiredLessonError(LessonError):
    """A retired lesson has no standing: it is not applied and takes no new applications (D87)."""


class AnonymizationError(LessonError):
    """A lesson named something of the customer's, so it did not save (D87)."""


# --- what the improvement agent may not touch (D69, design section 4 item 21) ---

READ_ONLY_PATHS: tuple[str, ...] = (
    "kullback/gates/",
    "kullback/runner/",
    "runner/loop.py",
    "runner/route.py",
    "runner/verdict.py",
    "runner/validate.py",
    "runner/budget.py",
    "runs/",
    "data/raw/",
)
_READ_ONLY_NEEDLES: tuple[str, ...] = (
    "kullback/gates", "kullback/runner", "gates/",
    "loop.py", "route.py", "verdict.py", "validate.py", "budget.py",
    "runs/", "runs directory", "raw/", "raw store", "evaluator",
)


def read_only_hits(*texts: Any) -> list[str]:
    """Every read-only name these strings mention; empty means the edit may go ahead."""
    blob = " ".join(str(t or "") for t in _flatten(texts)).lower()
    return sorted({needle for needle in _READ_ONLY_NEEDLES if needle in blob})


def _flatten(items: Any) -> list[Any]:
    out: list[Any] = []
    for item in items or []:
        if isinstance(item, (list, tuple, set)):
            out += _flatten(item)
        elif item is not None:
            out.append(item)
    return out


# --- records ---


class Node(Record):
    """One Builder version: the edits that made it, the prediction, the scorecard, the verdict on it."""
    id: str
    parent_id: Optional[str] = None
    seq: int = 0
    files_hash: str = ""
    files_dir: Optional[str] = None
    edit_kind: Optional[str] = None
    edit_description: str = ""
    edits: list[str] = Field(default_factory=list)
    edit_paths: list[str] = Field(default_factory=list)
    batch: bool = False
    batch_id: Optional[str] = None
    prediction: str = ""
    scorecard: dict = Field(default_factory=dict)
    outcome: Optional[str] = None
    accepted: Optional[bool] = None
    note: Optional[str] = None


class LessonApplication(Record):
    """One build that applied a lesson, and what the anchor said about it (D87)."""
    build_id: str
    benefit: Optional[bool] = None
    outcome: Optional[str] = None


class Lesson(Record):
    """A pattern the Builder carries between customers, with no customer data in it (D87)."""
    id: str = ""
    pattern: str
    fix: str
    confirming_result: str = ""
    relevance_condition: str = ""
    applications: list[LessonApplication] = Field(default_factory=list)
    retired: bool = False
    retired_reason: Optional[str] = None


class MemoryConfig(Record):
    """The D82 switch rule: batches only after enough single-change rounds with steady acceptance rates."""
    batch_mode: bool = False
    single_rounds_before_batch: int = 20
    stability_window: int = 10
    max_rate_drift: float = 0.25


# --- tree on disk ---


def tree_dir(workdir: str | Path) -> Path:
    return Path(workdir) / "builder_tree"


def _nodes_dir(workdir: str | Path) -> Path:
    return tree_dir(workdir) / "nodes"


def _file_name(node: Node) -> str:
    """Zero-padded seq first, so a sorted glob is seq order (D65)."""
    return f"{node.seq:06d}_{node.id}.json"


def node_path(workdir: str | Path, node_id: str) -> Path:
    directory = _nodes_dir(workdir)
    matches = sorted(directory.glob(f"*_{node_id}.json")) if directory.is_dir() else []
    return matches[0] if matches else directory / f"000000_{node_id}.json"


def _write_node(workdir: str | Path, node: Node) -> Node:
    directory = _nodes_dir(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob(f"*_{node.id}.json"):
        if stale.name != _file_name(node):
            stale.unlink()
    (directory / _file_name(node)).write_text(node.model_dump_json(indent=2), encoding="utf-8")
    _index_put(workdir, node)
    return node


def load_node(workdir: str | Path, node_id: str | None) -> Node:
    path = node_path(workdir, str(node_id))
    if not path.is_file():
        raise NodeNotFound(f"no node {node_id!r} under {tree_dir(workdir)}")
    return Node.model_validate_json(path.read_text(encoding="utf-8"))


def iter_nodes(workdir: str | Path) -> Iterator[Node]:
    """One node file at a time, in seq order; the tree is never loaded whole (D65)."""
    directory = _nodes_dir(workdir)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        yield Node.model_validate_json(path.read_text(encoding="utf-8"))


def grep_nodes(workdir: str | Path, pattern: str) -> list[Node]:
    """Search the tree the way D65 wants it searched: match the file text, keep only the hits."""
    regex = re.compile(pattern, re.IGNORECASE)
    hits = []
    directory = _nodes_dir(workdir)
    if not directory.is_dir():
        return hits
    for path in sorted(directory.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        if regex.search(text):
            hits.append(Node.model_validate_json(text))
    return sorted(hits, key=lambda n: (n.seq, n.id))


# --- the index: head, open proposals and the counters, without parsing the tree (D65) ---


def _index_path(workdir: str | Path) -> Path:
    return tree_dir(workdir) / "index.json"


def _entry(node: Node) -> dict:
    return {"seq": node.seq, "parent": node.parent_id, "kind": node.edit_kind,
            "accepted": node.accepted, "batch": bool(node.batch)}


def _read_index(workdir: str | Path) -> dict:
    path = _index_path(workdir)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("nodes"), dict):
            return data
    return _rebuild_index(workdir)


def _rebuild_index(workdir: str | Path) -> dict:
    index = {"nodes": {node.id: _entry(node) for node in iter_nodes(workdir)}}
    if _nodes_dir(workdir).is_dir():
        _save_index(workdir, index)
    return index


def _save_index(workdir: str | Path, index: dict) -> None:
    tree_dir(workdir).mkdir(parents=True, exist_ok=True)
    _index_path(workdir).write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def _index_put(workdir: str | Path, node: Node) -> None:
    index = _read_index(workdir)
    index["nodes"][node.id] = _entry(node)
    _save_index(workdir, index)


def _sorted_entries(index: dict) -> list[tuple[str, dict]]:
    return sorted(index["nodes"].items(), key=lambda item: (item[1]["seq"], item[0]))


def _next_seq(index: dict) -> int:
    return max((e["seq"] for e in index["nodes"].values()), default=-1) + 1


def _node_id(parent_id: Optional[str], seq: int, files_hash: str, edits: list[str],
             prediction: str) -> str:
    key = {"parent_id": parent_id, "seq": seq, "files_hash": files_hash,
           "edits": edits, "prediction": prediction}
    return content_hash(key)[:16]


def init_tree(workdir: str | Path, files_hash: str = "", note: Optional[str] = None) -> Node:
    """Create the root node (the Builder as it stands) if the tree has none yet."""
    index = _read_index(workdir)
    for node_id, entry in _sorted_entries(index):
        if entry["parent"] is None:
            root = load_node(workdir, node_id)
            if files_hash and root.files_hash and files_hash != root.files_hash:
                raise TreeError(
                    f"the tree already has root {root.id} with files_hash {root.files_hash!r}; "
                    f"{files_hash!r} describes different files, so propose an edit instead"
                )
            return root
    root = Node(
        id=_node_id(None, 0, files_hash, ["root"], ""),
        parent_id=None, seq=0, files_hash=files_hash,
        edit_description="root", edits=["root"], prediction="", accepted=True, note=note,
    )
    return _write_node(workdir, root)


def head(workdir: str | Path) -> Optional[Node]:
    """The newest accepted node: what a new proposal branches from."""
    accepted = [i for i, e in _sorted_entries(_read_index(workdir)) if e["accepted"] is True]
    return load_node(workdir, accepted[-1]) if accepted else None


def children(workdir: str | Path, node_id: str) -> list[Node]:
    ids = [i for i, e in _sorted_entries(_read_index(workdir)) if e["parent"] == node_id]
    return [load_node(workdir, i) for i in ids]


def path_to_root(workdir: str | Path, node_id: str) -> list[Node]:
    """The node and its ancestors, root first; a cycle in the files is refused, not walked."""
    chain: list[Node] = []
    seen: set[str] = set()
    current: Optional[str] = node_id
    while current is not None:
        if current in seen:
            raise TreeError(f"node {current} is its own ancestor; the tree file is corrupt")
        seen.add(current)
        node = load_node(workdir, current)
        chain.append(node)
        current = node.parent_id
    return list(reversed(chain))


def open_proposals(workdir: str | Path) -> list[Node]:
    """Nodes the evaluator has not decided yet."""
    ids = [i for i, e in _sorted_entries(_read_index(workdir)) if e["accepted"] is None]
    return [load_node(workdir, i) for i in ids]


def accepted_single_rounds(workdir: str | Path) -> int:
    """Accepted one-change rounds, root and batched nodes excluded (D82 switch rule)."""
    return sum(1 for _, e in _sorted_entries(_read_index(workdir))
               if e["accepted"] is True and e["parent"] is not None and not e["batch"])


def _decided(workdir: str | Path, window: Optional[int] = None) -> list[dict]:
    entries = [e for _, e in _sorted_entries(_read_index(workdir))
               if e["accepted"] is not None and e["parent"] is not None]
    return entries[-window:] if window else entries


def _rates(entries: list[dict]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for kind in {str(e["kind"]) for e in entries}:
        same = [e for e in entries if str(e["kind"]) == kind]
        rates[kind] = sum(1 for e in same if e["accepted"] is True) / len(same)
    return rates


def acceptance_rates(workdir: str | Path, window: Optional[int] = None) -> dict[str, float]:
    """Accepted share per edit kind over the last `window` decided rounds (D82 switch rule)."""
    return _rates(_decided(workdir, window))


def rates_are_stable(workdir: str | Path, config: Optional[MemoryConfig] = None) -> bool:
    """True when no edit kind's acceptance rate moved by more than the configured drift (D82)."""
    config = config or MemoryConfig()
    decided = _decided(workdir, config.stability_window)
    if len(decided) < 2:
        return False
    middle = len(decided) // 2
    older, newer = _rates(decided[:middle]), _rates(decided[middle:])
    shared = set(older) & set(newer)
    if not shared:
        return False
    return all(abs(older[kind] - newer[kind]) < config.max_rate_drift for kind in shared)


def batch_allowed(workdir: str | Path, config: Optional[MemoryConfig] = None) -> bool:
    """Batches need the config flag, enough single rounds, and steady per-edit-kind rates (D82)."""
    config = config or MemoryConfig()
    if not config.batch_mode:
        return False
    if accepted_single_rounds(workdir) < config.single_rounds_before_batch:
        return False
    return rates_are_stable(workdir, config)


# --- the files of a Builder version (design section 4 item 21) ---


def snapshot_files(workdir: str | Path, source_dir: str | Path) -> tuple[str, str]:
    """Copy the Builder's files under builder_tree/files/<hash>/ and return (files_hash, files_dir)."""
    source = Path(source_dir)
    if not source.is_dir():
        raise TreeError(f"no Builder directory at {source}")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    digest = content_hash({str(p.relative_to(source)): content_hash(p.read_bytes().hex())
                           for p in files})[:16]
    relative = f"files/{digest}"
    destination = tree_dir(workdir) / relative
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return digest, relative


def restore(workdir: str | Path, node: Node | str, dest_dir: str | Path) -> list[Path]:
    """Put a node's files back into the Builder directory, so an edit can be reverted (D64).

    A file dest_dir holds that the snapshot does not is removed too: an edit that added a file and
    was then rejected would otherwise leave that file behind, which is not what "revert" promises.
    """
    current = _resolve(workdir, node)
    if not current.files_dir:
        raise TreeError(f"node {current.id} has no files snapshot to restore")
    source = tree_dir(workdir) / current.files_dir
    if not source.is_dir():
        raise TreeError(f"snapshot {current.files_dir} is missing under {tree_dir(workdir)}")
    dest = Path(dest_dir)
    relatives = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file())
    kept = set(relatives)
    if dest.is_dir():
        for path in sorted(p for p in dest.rglob("*") if p.is_file()):
            if path.relative_to(dest) not in kept:
                path.unlink()
    written = []
    for relative in relatives:
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        written.append(target)
    return written


# --- proposing, deciding, bisecting ---


def _new_node(workdir: str | Path, parent: Node, edits: list[str], prediction: str,
              files_hash: Optional[str], files_dir: Optional[str], edit_kind: Optional[str],
              edit_paths: Optional[list[str]], batch: bool, batch_id: Optional[str],
              seq: int) -> Node:
    named = read_only_hits(edits, edit_paths, edit_kind)
    if named:
        raise ReadOnlyEditError(
            "the improvement agent may not edit " + ", ".join(named)
            + " (D69 read-only list); propose an edit under the Builder's own directory instead"
        )
    node = Node(
        id=_node_id(parent.id, seq, files_hash or "", edits, prediction),
        parent_id=parent.id,
        seq=seq,
        files_hash=files_hash or "",
        files_dir=files_dir,
        edit_kind=edit_kind,
        edit_description="; ".join(edits),
        edits=list(edits),
        edit_paths=list(edit_paths or []),
        batch=batch,
        batch_id=batch_id,
        prediction=prediction,
    )
    return _write_node(workdir, node)


def _parent_for(workdir: str | Path, parent_id: Optional[str]) -> Node:
    parent = load_node(workdir, parent_id) if parent_id else head(workdir)
    if parent is None:
        raise TreeError("no accepted node to branch from")
    if parent.accepted is not True:
        state = "rejected" if parent.accepted is False else "still open"
        raise TreeError(
            f"node {parent.id} is {state}; only an edit the evaluator accepted can be built on (D64)"
        )
    return parent


def _refuse_open(workdir: str | Path) -> None:
    still_open = open_proposals(workdir)
    if still_open:
        ids = ", ".join(n.id for n in still_open)
        raise OpenProposalError(
            f"one change per round (D82): accept or reject {ids} before proposing another"
        )


def _files_for(workdir: str | Path, files_hash: Optional[str],
               files_dir: Optional[str | Path]) -> tuple[str, Optional[str]]:
    """A node never inherits its parent's hash: an edit changes the files, so it gets its own."""
    if files_dir is not None:
        digest, relative = snapshot_files(workdir, files_dir)
        return (files_hash or digest), relative
    return (files_hash or ""), None


def _prepare_proposal(workdir: str | Path, parent_id: Optional[str], files_hash: Optional[str],
                      files_dir: Optional[str | Path]) -> tuple[Node, str, Optional[str]]:
    """The setup a single edit and a batch both need: the tree exists, no proposal is open, and the
    files are snapshotted, before either builds its own Node (D82)."""
    init_tree(workdir)
    _refuse_open(workdir)
    parent = _parent_for(workdir, parent_id)
    digest, relative = _files_for(workdir, files_hash, files_dir)
    return parent, digest, relative


def propose(
    workdir: str | Path,
    edit_description: str,
    prediction: str,
    parent_id: Optional[str] = None,
    files_hash: Optional[str] = None,
    edit_kind: Optional[str] = None,
    config: Optional[MemoryConfig] = None,
    edit_paths: Optional[list[str]] = None,
    files_dir: Optional[str | Path] = None,
) -> Node:
    """Add a child node for one proposed edit; one open proposal at a time (D82)."""
    parent, digest, relative = _prepare_proposal(workdir, parent_id, files_hash, files_dir)
    return _new_node(workdir, parent, [edit_description], prediction, digest, relative,
                     edit_kind, edit_paths, False, None, _next_seq(_read_index(workdir)))


def propose_batch(
    workdir: str | Path,
    edits: list[str],
    prediction: str,
    parent_id: Optional[str] = None,
    files_hash: Optional[str] = None,
    edit_kind: Optional[str] = None,
    config: Optional[MemoryConfig] = None,
    edit_paths: Optional[list[str]] = None,
    files_dir: Optional[str | Path] = None,
) -> Node:
    """Several edits as one node, so the batch is accepted or rejected as a whole (D82)."""
    if len(edits or []) < 2:
        raise TreeError("a batch is two edits or more; use propose() for a single change")
    if not batch_allowed(workdir, config):
        raise TreeError(
            "batches are not allowed yet (D82): the config flag, enough accepted single rounds "
            "and steady per-edit-kind acceptance rates all have to hold"
        )
    parent, digest, relative = _prepare_proposal(workdir, parent_id, files_hash, files_dir)
    return _new_node(workdir, parent, list(edits), prediction, digest, relative,
                     edit_kind, edit_paths, True, None, _next_seq(_read_index(workdir)))


def bisect(workdir: str | Path, node: Node | str) -> list[Node]:
    """Split a rejected batch in two and re-propose the halves, so the culprit is found (D82)."""
    rejected = _resolve(workdir, node)
    if rejected.accepted is not False:
        raise TreeError(f"node {rejected.id} was not rejected; there is nothing to bisect")
    if len(rejected.edits) < 2:
        raise TreeError(f"node {rejected.id} carries one edit; the culprit is already known")
    _refuse_open(workdir)
    parent = _parent_for(workdir, rejected.parent_id)
    middle = len(rejected.edits) // 2
    halves = [rejected.edits[:middle], rejected.edits[middle:]]
    made = []
    for half in halves:
        made.append(_new_node(
            workdir, parent, half, rejected.prediction, rejected.files_hash, rejected.files_dir,
            rejected.edit_kind, rejected.edit_paths, len(half) > 1, rejected.id,
            _next_seq(_read_index(workdir)),
        ))
    return made


def _resolve(workdir: str | Path, node: Node | str) -> Node:
    return load_node(workdir, node.id if isinstance(node, Node) else node)


def evaluate(workdir: str | Path, node: Node | str, scorecard: dict,
             outcome: Optional[str] = None) -> Node:
    """Record what the evaluator measured on the anchor, and what actually happened (D64, D82)."""
    current = _resolve(workdir, node)
    if current.accepted is not None:
        raise TreeError(
            f"node {current.id} was already decided; the evaluator's record is not rewritten (D64)"
        )
    current.scorecard = dict(scorecard)
    if outcome is not None:
        current.outcome = outcome
    return _write_node(workdir, current)


def _decide(workdir: str | Path, node: Node | str, accepted: bool, note: Optional[str]) -> Node:
    current = _resolve(workdir, node)
    if current.accepted is not None:
        raise TreeError(f"node {current.id} was already decided")
    if accepted and not current.scorecard:
        raise TreeError(f"node {current.id} has no scorecard; evaluate it before accepting")
    if accepted:
        kept = [c for c in children(workdir, str(current.parent_id))
                if c.id != current.id and c.accepted is True]
        if kept:
            raise TreeError(
                f"node {kept[0].id} is already the accepted child of {current.parent_id}; "
                "the accepted lineage stays one chain, so propose a follow-up edit on it instead"
            )
    current.accepted = accepted
    if note is not None:
        current.note = note
    return _write_node(workdir, current)


def accept(workdir: str | Path, node: Node | str, note: Optional[str] = None) -> Node:
    """The evaluator outside the loop keeps the edit (D64)."""
    return _decide(workdir, node, True, note)


def reject(workdir: str | Path, node: Node | str, reason: Optional[str] = None) -> Node:
    """The evaluator outside the loop drops the edit; the node stays in the tree as the record."""
    return _decide(workdir, node, False, reason)


# --- lessons file (D87) ---

_HEADER = (
    "# Builder lessons\n\n"
    "Patterns carried between customers (D87). No customer tool, table, entity or value names:\n"
    "every entry passed the anonymization gate in memory.py before it was written here.\n"
)
_FIELDS = ("pattern", "fix", "confirming_result", "relevance_condition")
_MIN_VOCAB_LEN = 3
_VOCAB_FILE = "customer_vocabulary.json"


class _Missing:
    """Nothing was passed, which is not the same as an empty vocabulary passed on purpose."""


_MISSING = _Missing()


def lessons_path(target: str | Path) -> Path:
    """The lessons file: a path ending in .md is used as is, anything else is a workdir."""
    path = Path(target)
    return path if path.suffix == ".md" else path / "lessons.md"


def vocabulary_path(target: str | Path) -> Path:
    return lessons_path(target).parent / _VOCAB_FILE


def save_vocabulary(target: str | Path, vocabulary: Iterable[str]) -> set[str]:
    """Store this customer's names beside the lessons file, so the gate runs without being asked."""
    names = {str(n) for n in vocabulary if n}
    path = vocabulary_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(names), indent=2), encoding="utf-8")
    return names


def load_vocabulary(target: str | Path) -> Optional[set[str]]:
    path = vocabulary_path(target)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(n) for n in data}


def _vocabulary_for(target: str | Path, vocabulary: Any) -> Any:
    """The gate is on by default: an empty vocabulary has to be passed, or stored, on purpose."""
    if not isinstance(vocabulary, _Missing) and vocabulary is not None:
        return vocabulary
    stored = load_vocabulary(target)
    if stored is None:
        raise AnonymizationError(
            "no customer vocabulary was given and none is stored beside the lessons file; "
            "the anonymization gate cannot clear this lesson (D87). Pass vocabulary=[] only "
            "when there is deliberately nothing to check"
        )
    return stored


def _normalize(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9_]+", " ", str(text).lower()).strip() + " "


def _collapse(text: str) -> str:
    """Separators dropped, so search_products, search-products and searchProducts read alike."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """One field off a dict or an object, the shape a ToolSig or a policy span arrives in either way."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _lesson_text(lesson: Lesson) -> str:
    parts = [lesson.id or ""] + [getattr(lesson, name) or "" for name in _FIELDS]
    parts.append(lesson.retired_reason or "")
    for app in lesson.applications:
        parts += [app.build_id or "", app.outcome or ""]
    return " ".join(parts)


def _field_names(sig: Any) -> list[str]:
    names = []
    for group in ("args_fields", "result_schema"):
        for field in _get(sig, group, []) or []:
            name = _get(field, "name")
            if name:
                names.append(str(name))
    schema = _get(sig, "args_schema", {}) or {}
    properties = schema.get("properties") if isinstance(schema, dict) else None
    names += [str(k) for k in (properties or {})]
    return names


def customer_vocabulary(toolsigs: Any = None, tables: Any = None, columns: Any = None,
                        entity_ids: Any = None, extra: Any = None) -> set[str]:
    """The names a lesson may not contain: tool names, their argument and result field names,
    table and column names, entity ids."""
    names: set[str] = set()
    for sig in toolsigs or []:
        name = _get(sig, "name")
        if name:
            names.add(str(name))
        names.update(_field_names(sig))
    for group in (tables, columns, entity_ids, extra):
        for item in group or []:
            if item:
                names.add(str(item))
    return {n.lower() for n in names}


def _names_found(raw: str, vocabulary: Any) -> list[str]:
    """Every customer name this text contains; empty means the gate passes."""
    text, collapsed = _normalize(raw), _collapse(raw)
    found = set()
    for name in vocabulary or []:
        raw_name = str(name).strip()
        if len(raw_name) < _MIN_VOCAB_LEN:
            continue
        needle, squashed = _normalize(raw_name), _collapse(raw_name)
        if needle in text or (squashed and squashed in collapsed):
            found.add(str(name))
    return sorted(found)


def check_anonymized(lesson: Lesson, vocabulary: Any) -> list[str]:
    """Every customer name the lesson still contains; empty means the gate passes."""
    return _names_found(_lesson_text(lesson), vocabulary)


def _render(lesson: Lesson) -> str:
    lines = [f"## lesson {_one_line(lesson.id)}", ""]
    for name in _FIELDS:
        lines.append(f"- {name}: {_one_line(getattr(lesson, name) or '')}")
    lines.append(f"- retired: {'yes' if lesson.retired else 'no'}")
    if lesson.retired_reason:
        lines.append(f"- retired_reason: {_one_line(lesson.retired_reason)}")
    for app in lesson.applications:
        benefit = {True: "yes", False: "no"}.get(app.benefit, "unknown")
        lines.append(
            f"- application: build={_one_line(app.build_id)} | benefit={benefit} "
            f"| outcome={_one_line(app.outcome or '')}"
        )
    lines.append("")
    return "\n".join(lines)


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).replace("|", "/").strip()


def _write_lessons(target: str | Path, lessons: list[Lesson]) -> None:
    path = lessons_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + "\n" + "\n".join(_render(lesson) for lesson in lessons), encoding="utf-8")


def load_lessons(target: str | Path) -> list[Lesson]:
    """Read the whole lessons file: it is small and read at build start (D87)."""
    path = lessons_path(target)
    if not path.is_file():
        return []
    lessons: list[Lesson] = []
    current: Optional[dict] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## lesson "):
            if current is not None:
                lessons.append(Lesson(**current))
            current = {"id": line[len("## lesson "):].strip(), "pattern": "", "fix": "",
                       "applications": []}
            continue
        if current is None or not line.startswith("- "):
            continue
        key, _, value = line[2:].partition(":")
        key, value = key.strip(), value.strip()
        if key == "application":
            current["applications"].append(_parse_application(value))
        elif key == "retired":
            current["retired"] = value == "yes"
        elif key in _FIELDS or key == "retired_reason":
            current[key] = value
    if current is not None:
        lessons.append(Lesson(**current))
    return lessons


def active_lessons(target: Any) -> list[Lesson]:
    """The lessons that still have standing: retired ones are never judged or applied (D87)."""
    lessons = target if isinstance(target, list) else load_lessons(target)
    return [lesson for lesson in lessons if not lesson.retired]


def _parse_application(value: str) -> LessonApplication:
    fields = {}
    for part in value.split(" | "):
        key, _, item = part.partition("=")
        fields[key.strip()] = item.strip()
    benefit = {"yes": True, "no": False}.get(fields.get("benefit", "unknown"))
    return LessonApplication(build_id=fields.get("build", ""), benefit=benefit,
                             outcome=fields.get("outcome") or None)


def save_lesson(target: str | Path, lesson: Lesson, vocabulary: Any = _MISSING) -> Lesson:
    """Append a lesson, but only if it names nothing of the customer's (the D87 gate)."""
    for name in ("pattern", "fix", "relevance_condition"):
        if not (getattr(lesson, name) or "").strip():
            raise LessonError(f"a lesson needs a {name}; there would be nothing to judge (D87)")
    named = check_anonymized(lesson, _vocabulary_for(target, vocabulary))
    if named:
        raise AnonymizationError(
            "lesson names customer data and was not saved: " + ", ".join(named)
            + "; rewrite it as the pattern"
        )
    stored = lesson.model_copy(deep=True)
    if not stored.id:
        stored.id = content_hash(stored.pattern)[:12]
    lessons = load_lessons(target)
    for index, existing in enumerate(lessons):
        if existing.id == stored.id:
            # Retirement and the record of what it did belong to the evaluator, not the Builder.
            stored.applications = list(existing.applications)
            stored.retired = existing.retired
            stored.retired_reason = existing.retired_reason
            lessons[index] = stored
            _write_lessons(target, lessons)
            return stored
    lessons.append(stored)
    _write_lessons(target, lessons)
    return stored


def _update_lesson(target: str | Path, lesson_id: str, change) -> Lesson:
    lessons = load_lessons(target)
    for index, lesson in enumerate(lessons):
        if lesson.id == lesson_id:
            lessons[index] = change(lesson)
            _write_lessons(target, lessons)
            return lessons[index]
    raise KeyError(f"no lesson {lesson_id!r} in {lessons_path(target)}")


def record_application(target: str | Path, lesson_id: str, build_id: str,
                       benefit: Optional[bool] = None, outcome: Optional[str] = None,
                       vocabulary: Any = _MISSING) -> Lesson:
    """Append what this build's anchor said about the lesson, through the same gate (D87).

    Only the new application's own text is checked, not the whole reconstituted lesson: an
    already-saved application that cleared the gate once must not fail a later, unrelated call
    just because the customer's mined vocabulary grew in between (D87).
    """
    application = LessonApplication(build_id=build_id, benefit=benefit, outcome=outcome)

    def change(lesson: Lesson) -> Lesson:
        if lesson.retired:
            raise RetiredLessonError(
                f"lesson {lesson.id} is retired and takes no new applications (D87)"
            )
        named = _names_found(f"{application.build_id or ''} {application.outcome or ''}",
                             _vocabulary_for(target, vocabulary))
        if named:
            raise AnonymizationError(
                "application names customer data and was not saved: " + ", ".join(named)
            )
        updated = lesson.model_copy(deep=True)
        updated.applications = list(updated.applications) + [application]
        return updated

    return _update_lesson(target, lesson_id, change)


def retirement_candidates(target: Any, min_applications: int = 3) -> list[Lesson]:
    """Lessons with N applications and no confirmed benefit: what the evaluator retires (D87)."""
    return [lesson for lesson in active_lessons(target)
            if len(lesson.applications) >= min_applications
            and not any(a.benefit is True for a in lesson.applications)]


def retire_lesson(target: str | Path, lesson_id: str, reason: str = "",
                  vocabulary: Any = _MISSING, force: bool = False) -> Lesson:
    """The evaluator retires a lesson with applications and no confirmed benefit (D87), not the Builder.

    Only the new retirement reason is checked, not the whole reconstituted lesson: an old,
    already-approved application must not block a later, individually clean retirement just
    because the customer's mined vocabulary grew in between (D87).
    """

    def change(lesson: Lesson) -> Lesson:
        if not force and any(a.benefit is True for a in lesson.applications):
            raise LessonError(
                f"lesson {lesson.id} has a confirmed benefit; retiring it needs force=True (D87)"
            )
        named = _names_found(reason, _vocabulary_for(target, vocabulary))
        if named:
            raise AnonymizationError(
                "the retirement reason names customer data and was not saved: " + ", ".join(named)
            )
        updated = lesson.model_copy(deep=True)
        updated.retired = True
        updated.retired_reason = reason or updated.retired_reason
        return updated

    return _update_lesson(target, lesson_id, change)


# --- relevance judging (D87) ---

_JUDGE_SYSTEM = (
    "You judge whether a lesson from earlier customers applies to this customer. "
    "Answer with one JSON object: {\"relevant\": true or false, \"evidence\": \"...\"}. "
    "relevant must be the JSON literal true or false, never a string. "
    "Evidence must quote a tool name or a policy span from the material given. "
    "Say false when nothing in the material meets the relevance condition."
)
_NO_REASON = "set aside: the judge gave no reason"


def _tool_line(sig: Any) -> str:
    fields = _get(sig, "result_schema", []) or []
    names = [_get(f, "name", "") for f in fields]
    line = f"- {_get(sig, 'name', '')} (kind {_get(sig, 'kind', 'unknown')}): {_get(sig, 'description', '') or ''}"
    return line + (f" result fields: {', '.join(n for n in names if n)}" if names else "")


def _span_text(span: Any) -> str:
    if isinstance(span, str):
        return span
    return str(_get(span, "span_text", None) or _get(span, "text", "") or "")


def _span_line(span: Any) -> str:
    return f"- {_span_text(span)}"


def _relevance_messages(lesson: Lesson, toolsigs: Any, policy_spans: Any) -> list[dict]:
    tools = [_tool_line(s) for s in toolsigs or []] or ["- none"]
    spans = [_span_line(s) for s in policy_spans or []] or ["- none"]
    body = [
        f"Lesson pattern: {lesson.pattern}",
        f"Lesson fix: {lesson.fix}",
        f"Relevance condition: {lesson.relevance_condition}",
        "",
        "This customer's tools:",
        *tools,
        "",
        "This customer's policy spans:",
        *spans,
    ]
    return [{"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": "\n".join(body)}]


def _parse_json_object(text: str) -> Optional[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def evidence_in_material(evidence: str, toolsigs: Any = None, policy_spans: Any = None) -> bool:
    """The code gate on the judge's answer: the evidence has to quote what we sent it (D87)."""
    said = _collapse(evidence)
    if not said:
        return False
    for sig in toolsigs or []:
        name = _get(sig, "name")
        for candidate in [name] + _field_names(sig):
            squashed = _collapse(candidate or "")
            if len(squashed) >= _MIN_VOCAB_LEN and squashed in said:
                return True
    for span in policy_spans or []:
        squashed = _collapse(_span_text(span))
        if squashed and (squashed in said or said in squashed):
            return True
    return False


def judge_relevance(model, lesson: Lesson, toolsigs: Any = None, policy_spans: Any = None,
                    config=None) -> tuple[bool, str]:
    """Ask the given model whether this lesson's condition holds here, with evidence (D87)."""
    if lesson.retired:
        return False, "retired: " + (lesson.retired_reason or "the evaluator retired this lesson")
    if not (lesson.relevance_condition or "").strip():
        return False, "set aside: the lesson has no relevance condition to check"
    reply = model.query(_relevance_messages(lesson, toolsigs, policy_spans), None, config)
    text = (reply.content or "").strip()
    data = _parse_json_object(text)
    if data is None:
        return False, text or "set aside: the judge gave no answer"
    evidence = str(data.get("evidence") or "").strip()
    answer = data.get("relevant")
    if answer is not True:
        if answer is not False:
            return False, "set aside: the judge gave a non-boolean answer for relevant"
        return False, evidence or _NO_REASON
    if not evidence:
        return False, "set aside: the judge claimed relevance with no evidence"
    if not evidence_in_material(evidence, toolsigs, policy_spans):
        return False, "set aside: the evidence quotes nothing from this customer's material"
    return True, evidence


def judge_lessons(model, lessons: Any, toolsigs: Any = None, policy_spans: Any = None,
                  config=None) -> tuple[list[Lesson], list[SetAsideLesson]]:
    """Judge every lesson once and return the ones applied and the ones set aside with a reason,
    which is what the report lists (D87)."""
    applied: list[Lesson] = []
    set_aside: list[SetAsideLesson] = []
    for lesson in (lessons if isinstance(lessons, list) else load_lessons(lessons)):
        relevant, reason = judge_relevance(model, lesson, toolsigs, policy_spans, config)
        if relevant:
            applied.append(lesson)
        else:
            set_aside.append(SetAsideLesson(id=lesson.id, pattern=lesson.pattern,
                                            reason=reason or _NO_REASON))
    return applied, set_aside


# --- tool lessons (phase 6): gate-failure sequences per tool, workdir-scoped ---

_TOOL_LESSONS_FILE = "tool_lessons.json"


def tool_lessons_path(workdir: Any) -> Path:
    """The workdir-scoped tool-lesson file (distinct from the cross-customer lessons.md, D87)."""
    return Path(workdir) / _TOOL_LESSONS_FILE


def load_tool_lessons(workdir: Any) -> dict[str, list[list[str]]]:
    """Every recorded gate-failure sequence per tool, oldest first; {} when none recorded."""
    path = tool_lessons_path(workdir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def record_lesson(workdir: Any, tool: str, failures: list[str]) -> Path:
    """Append one gate-failure sequence for this tool; returns the file it was written to."""
    path = tool_lessons_path(workdir)
    data = load_tool_lessons(workdir)
    data.setdefault(tool, []).append(list(failures))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def lesson_for(workdir: Any, tool: str) -> str:
    """The failure sequences to inject into the next attempt's prompt; "" when the tool has none."""
    sequences = load_tool_lessons(workdir).get(tool, [])
    if not sequences:
        return ""
    lines = [f"Past gate failures for {tool} (do not repeat them):"]
    for i, failures in enumerate(sequences[-3:], start=1):
        lines.append(f"- attempt {i}: " + "; ".join(failures))
    return "\n".join(lines)
