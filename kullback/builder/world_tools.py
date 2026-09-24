"""The Builder's first pass as two tools: ingest and derive_world (stream 5a).

These run without the stage scheduler: `ingest` stores recordings as Traces,
and `derive_world` carries the mine, readers, cluster, canon_rules and
starting_state stages in that order, with the same calls and the same on-disk
outputs as the stage graph in build.py. Each stage body below is the stage's
own `run` logic carried over verbatim; the scheduler context is replaced by
explicit arguments (workdir, ledger, anchor).

The held-out anchor is drawn from the Tasks the moment clustering lands them
(D81), stored as anchor.json with the per Task seed counts beside it, and read
by the Starting state for provenance only. The four stages before the anchor
exists, and the canon rules after it, all read every Run by exemption
(build.ALL_RUNS_STAGES); nothing here imports the scheduler to do any of this.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback import sampling
from kullback.builder import cache_reach, cluster, compile_env, ingest, mine, readers, synth, templates
from kullback.builder.compile_env import DB_FILE
from kullback.gates import artifacts, tool_runs
from kullback.gates import ledger as ledger_mod
from kullback.gates import scorecard as scorecard_mod
from kullback.gates import stages as stage_gates
from kullback.runner import canon
from kullback.runner.records import (
    Task,
    Trace,
    as_dict,
)
from kullback.runner.records import read_json as _read_json
from kullback.runner.records import write_json as _write_json
from kullback.runner.world.loading import OVERLAY_DIR
from kullback.user import fidelity as user_fidelity
from kullback.user import rules as user_rules

CANON_RULES = "canon-rules.json"
TASK_SPLIT = "task_split.json"
WORLD_PROVENANCE_FILE = "world_provenance.json"
WORLD_REPORT_FILE = "world_report.json"
WORLD_LOCK_FILE = "world.lock"
READERS_DIR = "readers"

# The held-out draw, carried over from the scheduler this stream replaces so
# the bytes on disk stay what a build already under way stored (D81, D212).
# build.py and pipeline.py own the originals; this stream removes the scheduler,
# so the world owns its anchor from here on.
ANCHOR_FILE = "anchor.json"
EVIDENCE_COUNTS_FILE = "evidence_counts.json"
ANCHOR_SHARE = 0.20
ANCHOR_MIN_RUNS = 3
ANCHOR_SEED = 20260827
ANCHOR_KIND = "anchor"


class WorldError(RuntimeError):
    """The first pass cannot run: no traces, no model where one is owed, or a gate refused."""


@dataclass
class IngestReport:
    """What storing the customer files published: the counts and the ruling."""

    files: list[str] = field(default_factory=list)
    runs: int = 0
    tool_calls: int = 0
    gate: dict = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, body: dict) -> "IngestReport":
        return cls(**{k: body.get(k, getattr(cls, k, None)) for k in
                      ("files", "runs", "tool_calls", "gate", "paths")})


@dataclass
class WorldReport:
    """What the first pass derived: row counts, call counts, rulings, paths."""

    tables: dict[str, int] = field(default_factory=dict)
    tools: dict[str, int] = field(default_factory=dict)
    tasks: dict[str, int] = field(default_factory=dict)
    gates: dict[str, dict] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    from_cache: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, body: dict) -> "WorldReport":
        known = {f for f in ("tables", "tools", "tasks", "gates", "paths", "from_cache")}
        return cls(**{k: v for k, v in (body or {}).items() if k in known})


def _records(paths: Iterable[Path], model: type) -> list:
    return [model.model_validate(json.loads(p.read_text(encoding="utf-8"))) for p in paths]


def load_traces(workdir: Path) -> list[Trace]:
    folder = Path(workdir) / "traces"
    return _records(sorted(folder.glob("*.json")), Trace) if folder.is_dir() else []


def _ledger(workdir: Path) -> ledger_mod.GateLedger:
    return ledger_mod.GateLedger(workdir)


def ingest_files(files: list[Path], workdir: Path) -> IngestReport:
    """Store each customer file as Traces under workdir, then run the ingest gate.

    This is the ingest stage's own run: one `ingest_file` call per file, which
    writes the traces exactly where the stage wrote them, with the gate read
    off the traces on disk.
    """
    workdir = Path(workdir)
    for path in files:
        ingest.ingest_file(path, workdir)
    traces = load_traces(workdir)
    ruling = artifacts.ingest_gate(traces)
    _ledger(workdir).record("ingest", ruling)
    calls = sum(len(t.tool_calls) for t in traces)
    return IngestReport(
        files=[str(p) for p in files],
        runs=len(traces),
        tool_calls=calls,
        gate=as_dict(ruling),
        paths=[str(Path("traces") / f"{t.trace_id}.json") for t in traces],
    )


def _mine_step(workdir: Path, traces: list[Trace]) -> tuple[list, Any]:
    sigs = mine.mine_tools(traces)
    # The write tools are the sigs' own, so the composite-key rule reads the same kinds the rest
    # of the build does rather than classifying the tools a second time.
    schema = mine.mine_schema(traces, write_tools=sorted(s.name for s in sigs if s.kind == "write"))
    # D282: a tool the recording shows ending the Run with no write is recorded as a row, a write.
    compile_env.record_ending_actions(traces, sigs, schema)
    # D164: the names the recording refused on every call, and the ones a recorded agent
    # invented, are no tool of this customer. They are written beside the sigs so a build can
    # see them, and they are never a failure: a Run refuses them as the recording did.
    unknown = mine.unknown_tools(traces)
    _write_json(workdir / "tool_sigs.json", [as_dict(s) for s in sigs])
    _write_json(workdir / "unknown_tools.json", unknown)
    # Where each tool's result rows were homed and by which rule, with the rows no rule could
    # home. A lookup whose rows reach no table is the whole of its replay fidelity, and this is
    # where that is readable before a single body has been written.
    _write_json(workdir / "row_homes.json", mine.row_homes(traces))
    # The constants of the world: per tool the corpus called the same way every time and got
    # the same answer to, that answer. They are columns of one row of the schema's own
    # constants table; the file is what a reader of the build sees them by, since the mine gate
    # is frozen and cannot carry a count of them.
    _write_json(workdir / "world_constants.json",
                {"table": mine.constants_table_of(schema), "row": mine.CONSTANTS_ROW,
                 "columns": sorted(mine.constants_row(schema))})
    _write_json(workdir / "schema.json", as_dict(schema))
    calls = [c for t in traces for c in t.tool_calls]
    # "flag, do not synthesize": a tool the corpus barely shows stays in the build, named in
    # the gate, rather than being invented or dropped (design section 6).
    _ledger(workdir).record("mine", artifacts.mine_gate(sigs, calls, unknown=unknown))
    return sigs, schema


def _readers_step(workdir: Path, traces: list[Trace], schema: Any, sigs: list,
                  model: Any) -> tuple[Any, list, dict]:
    by_requestor = readers.prose_calls(traces)
    proposals, rows, nodes, values = {}, {}, [], {}
    fills, assumptions, unset = {}, [], {}
    if by_requestor and model is None:
        raise WorldError("this corpus has prose results from a requestor of its own and the "
                         "readers step has no model to propose them with")
    for requestor in sorted(by_requestor):
        proposal, attempts, parsed = readers.propose(
            model, requestor, by_requestor[requestor], traces, workdir / READERS_DIR,
            max_attempts=readers.MAX_ATTEMPTS)
        read_rows = readers.starting_rows(traces, proposal, parsed)
        filled, sentences, missing = readers.fills_for(read_rows, proposal)
        proposals[requestor] = proposal.to_dict()
        rows[requestor] = read_rows
        fills[requestor] = filled
        assumptions += sentences
        unset[requestor] = missing
        values[requestor] = readers.column_values(proposal, parsed)
        nodes += attempts
    artifact = {"proposals": proposals, "rows": rows, "fills": fills,
                "assumptions": assumptions, "unset": unset}
    kept = readers.proposals_from(artifact)
    readers.apply_to_schema(schema, kept, values)
    # Handed on as it came where no proposal moved it: the artifacts are compared by
    # identity downstream, and a rebuilt list of the same sigs is a new artifact every build.
    sigs = readers.apply_to_sigs(sigs, kept) if kept else sigs
    # D203: a homed prose result no reader answers pins nothing, and the proposal step is asked
    # only about another requestor's toolkit. Every such tool gets a reader derived from its own
    # recorded results, and one forced proposal where the corpus cannot settle the slots.
    pairs, gaps = templates.close_gaps(
        model, traces, schema, workdir,
        readers.result_reader({**artifact, "derived": []}, traces, workdir),
        max_forced=templates.MAX_FORCED)
    artifact = {**artifact, "derived": [pair.to_dict() for pair in pairs], "gaps": gaps}
    templates.apply_revealed(schema, pairs)
    # A corpus whose prose results are all read, or that has none, records the note and nothing
    # else, so a build over such a corpus writes the same artifact it always wrote.
    silent = not by_requestor and not pairs and not gaps.get("tools")
    if silent:
        artifact = {"note": readers.NO_PROSE, "proposals": {}, "rows": {}}
    _write_json(workdir / readers.READERS_FILE,
                {"note": readers.NO_PROSE} if silent else {**artifact, "attempts": nodes})
    _write_json(workdir / "schema.json", as_dict(schema))
    _write_json(workdir / "tool_sigs.json", [as_dict(s) for s in sigs])
    # Section 6: a proposal the gate could not satisfy is flagged and kept, never a failed build.
    _ledger(workdir).record(
        "readers", stage_gates.readers_gate(proposals.values(), len(by_requestor),
                                            assumptions=assumptions, unset=unset,
                                            kinds={p.requestor: readers.kinds_for(p) for p in kept},
                                            derived=pairs, totals=gaps.get("totals")))
    return schema, sigs, artifact


def _grouping(workdir: Path, live_tasks: Iterable[Any], traces: list[Trace],
              schema: Any) -> dict:
    """The grouping's fingerprint, the two inputs it may depend on, and which moved (D216)."""
    now = {"recordings": cluster.recordings_hash(traces),
           "homing": compile_env.homing_hash(schema)}
    fingerprint = cluster.grouping_fingerprint(live_tasks)
    path = Path(workdir) / cluster.GROUPING_FILE
    first = _read_json(path, None)
    moved = cluster.moved_input(fingerprint, now, first if isinstance(first, dict) else None)
    if not isinstance(first, dict) or not first.get("fingerprint") or moved:
        _write_json(path, {"format": cluster.GROUPING_FORMAT, "fingerprint": fingerprint, **now})
    return {"grouping": fingerprint, "grouping_inputs": now, "grouping_moved": moved}


def _cluster_step(workdir: Path, traces: list[Trace], schema: Any, sigs: list) -> tuple[list, list]:
    # D74: two Runs that saw one row in two versions before writing started in different
    # worlds, and a Task's overlay can pin only one, so they are different Tasks. D216: the
    # version is taken over the recording alone, so nothing proposed about the
    # corpus (the column classes, the readers, a cache format) can regroup it next round.
    worlds = compile_env.trace_worlds(traces, schema, cluster.write_tool_names(sigs))
    categories, tasks = cluster.cluster_runs(traces, sigs, worlds=worlds)
    # Taken over what the rebuild grouped, before the frozen list is laid back over it: the
    # question is whether this round's own split differs from the one that was frozen.
    grouping = _grouping(workdir, tasks, traces, schema)
    # D200: once a list is frozen it is the Task list. A rebuild may add Tasks for Runs nobody
    # froze, it may not drop, re-split or re-id a frozen one, because every number the build is
    # judged on is counted over that list and a Task that moves takes its ruling with it.
    tasks, split = cluster.resume_frozen(tasks, scorecard_mod.frozen_tasks(workdir),
                                         worlds=worlds)
    split.update(grouping)
    for task in tasks:
        _write_json(workdir / "tasks" / f"{task.id}.json", as_dict(task))
    _write_json(workdir / "tasks.json", {"tasks": [as_dict(t) for t in tasks]})
    _write_json(workdir / TASK_SPLIT, split)
    # D96: the coverage denominator is frozen once, here, before anything measures coverage.
    scorecard_mod.freeze_tasks(workdir, tasks)
    _ledger(workdir).record("cluster", stage_gates.cluster_gate(tasks, categories))
    return categories, tasks


def _rows_of(result: Any) -> list[dict]:
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    return []


def _canon_step(workdir: Path, traces: list[Trace], schema: Any) -> dict:
    rows = [row for trace in traces for call in trace.tool_calls
            for row in _rows_of(call.result)]
    rules = canon.learn_rules(schema, rows)
    canon.save_rules(rules, workdir / CANON_RULES)
    return rules.model_dump()


def _stored_anchor(workdir: Path) -> Optional[dict]:
    path = Path(workdir) / ANCHOR_FILE
    if not path.is_file():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    return body if isinstance(body, dict) else None


def draw_anchor(tasks: list, workdir: Path) -> dict[str, Any]:
    """Hold out each Task's Runs by their own keys and store what was held out (D81, D212).

    Every Task the stored anchor already knows keeps exactly the Runs it was
    given; a Task that appeared afterwards is drawn now and appended, because a
    Task with nothing held out is a Task the Builder can fit to. Membership is
    a threshold on each Run's own key under the workdir salt, so two builds of
    one workdir hold out the same Runs of the Runs they share.
    """
    workdir = Path(workdir)
    stored = _stored_anchor(workdir)
    held_out: dict[str, list[str]] = dict(stored.get("held_out") or {}) if stored else {}
    unguarded: list[str] = list(stored.get("unguarded") or []) if stored else []
    if stored is not None:
        share, min_runs, seed, salt = (stored.get("share", ANCHOR_SHARE),
                                       stored.get("min_runs", ANCHOR_MIN_RUNS),
                                       stored.get("seed", ANCHOR_SEED),
                                       stored.get("salt", ""))
    else:
        share, min_runs, seed, salt = ANCHOR_SHARE, ANCHOR_MIN_RUNS, ANCHOR_SEED, ""
    if stored is None:
        salt = sampling.build_salt(workdir)
    added = stored is None
    for task in tasks:
        task_id = task.id if isinstance(task, Task) else task["id"]
        run_ids = list(task.run_ids) if isinstance(task, Task) else list(task["run_ids"])
        if task_id in held_out:
            continue
        added = True
        run_ids = sorted(run_ids)
        if len(run_ids) < min_runs:
            held_out[task_id] = []
            unguarded.append(task_id)
            continue
        count = max(1, int(len(run_ids) * share))
        held_out[task_id] = sampling.keyed_share(ANCHOR_KIND, run_ids, salt, share,
                                                 minimum=count, maximum=count)
    anchor = {"held_out": held_out, "unguarded": sorted(set(unguarded)), "share": share,
              "min_runs": min_runs, "seed": seed, "salt": salt}
    if stored is None or added:
        path = workdir / ANCHOR_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        body = dict(anchor)
        if not body.get("salt"):
            body.pop("salt", None)
        path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    counts = {}
    for task in tasks:
        task_id = task.id if isinstance(task, Task) else task["id"]
        run_ids = list(task.run_ids) if isinstance(task, Task) else list(task["run_ids"])
        held = set(anchor["held_out"].get(task_id, []))
        counts[task_id] = {"evidence_traces": len([r for r in run_ids if r not in held]),
                           "anchor_traces": len([r for r in run_ids if r in held])}
    _write_json(workdir / EVIDENCE_COUNTS_FILE, counts)
    return anchor


def seed_runs(anchor: Optional[dict], task_id: str, run_ids: Iterable[str]) -> list[str]:
    """What may be built from: every Run of the Task except its anchor (D81)."""
    held = set((anchor or {}).get("held_out", {}).get(task_id, []))
    return [r for r in run_ids if r not in held]


def _state_step(workdir: Path, traces: list[Trace], schema: Any, tasks: list, sigs: list,
                readers_artifact: dict, canon_rules: dict, anchor: Optional[dict],
                grow: Optional[dict] = None) -> tuple[dict, list, list, list]:
    # Its own copy: build_starting_state tags the synthetic ids on the schema it is given, and
    # the artifact the mine step released is not this step's to write to.
    schema = schema.model_copy(deep=True)
    # D202: the bodies this workdir already holds, so a column a Task first touches with a write
    # is pinned from what that write recorded. They are read off disk and not an artifact,
    # because no body exists yet when this step first runs.
    bodies = dict(_read_json(workdir / "bodies.json", {}) or {})
    state = compile_env.build_starting_state(traces, schema, workdir, tasks, sigs, grow=dict(grow or {}),
                                             grow_seed=0,
                                             revealed_rows=readers.reader_rows(readers_artifact),
                                             revealed_assumptions=readers.reader_assumptions(
                                                 readers_artifact),
                                             read_result=readers.result_reader(
                                                 readers_artifact, traces, workdir),
                                             bodies=bodies, rules=canon.rules_of(
                                                 {"canon_rules": canon_rules}),
                                             readers=tool_runs.load_readers(readers_artifact),
                                             guessed_columns=readers.filled_columns(
                                                 readers_artifact))
    # The synthetic ids live on the schema; the world the bodies run on is the complete one.
    _write_json(workdir / "schema.json", as_dict(schema))
    # D220 rule 2: the world is complete, and what only a held-out Run witnessed is written
    # down beside it. This step may read the anchor; every step below reads the provenance
    # off this file and never the membership.
    held_out = sorted({run_id for runs in (anchor or {}).get("held_out", {}).values()
                       for run_id in runs})
    columns = compile_env.holdout_columns(state.witnesses, held_out)
    _write_json(workdir / WORLD_PROVENANCE_FILE,
                {"witnesses": state.witnesses, "holdout_columns": columns,
                 "holdout_rows": sum(len(rows) for rows in columns.values()),
                 "holdout_columns_total": sum(len(names) for rows in columns.values()
                                              for names in rows.values())})
    return state.db, list(state.overlays), list(state.assumptions), list(state.synthetic_rows)


def _module_hash(module: Any) -> str:
    return cache_reach.closure_hash(module)[:16]


def _world_key(traces: list[Trace], model: Any, grow: Optional[dict]) -> str:
    digest = hashlib.sha256()
    for trace in sorted(traces, key=lambda t: t.trace_id):
        digest.update(trace.trace_id.encode("utf-8"))
        digest.update(b"\0")
        for call in trace.tool_calls:
            digest.update(json.dumps({"tool": call.name, "args": call.args,
                                      "result": call.result},
                                     sort_keys=True, default=str).encode("utf-8"))
            digest.update(b"\0")
    digest.update(f"model={getattr(model, 'name', 'none')}".encode("utf-8"))
    digest.update(json.dumps({"grow": grow or {}}, sort_keys=True, default=str).encode("utf-8"))
    for module in (mine, readers, cluster, canon, compile_env, synth, user_rules):
        digest.update(_module_hash(module).encode("utf-8"))
    return digest.hexdigest()


def _user_rules_step(workdir: Path, traces: list[Trace], sigs: list) -> list[Path]:
    """One Simulated user rules file per stored Trace, read off its user turns by code (D44).

    The file is what a fresh Run's user answers from (`BuiltEnvironment.rules`), so a workdir
    without it plays every Run with nobody to open or to answer. The vocabulary is this build's
    own where it derived one, the generic core otherwise; rewriting on a rerun is idempotent.
    """
    writes = sorted(s.name for s in sigs if s.kind == "write")
    vocab = user_fidelity.vocabulary_of(workdir)
    out = workdir / user_fidelity.RULES_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for trace in traces:
        path = out / f"{trace.trace_id}.json"
        _write_json(path, as_dict(user_rules.derive_user_rules(trace, vocab, writes)))
        paths.append(path)
    return paths


def derive_world(workdir: Path, *, reader_model: Any = None,
                 grow: Optional[dict] = None) -> WorldReport:
    """Run the first pass over the traces on disk: mine, readers, cluster, canon, state.

    No model is asked except the readers step, through `reader_model`. Every Trace's
    Simulated user rules are written under user_rules/ by code (D44). Tables
    grow synthetic rows only in untrusted Tasks, never here: `grow` names a row
    count per table to reach with rows composed from the observed ones, and the
    added ids are reported as synthetic. Content cache: the inputs (trace
    bytes, the reader model name, grow, and the module hashes of mine,
    readers, cluster, canon, compile_env and synth) are hashed into
    workdir/world.lock; if unchanged, the report is returned from disk without
    recomputing, and says so in `from_cache`.
    """
    workdir = Path(workdir)
    traces = load_traces(workdir)
    if not traces:
        raise WorldError(f"no Traces under {workdir / 'traces'}; ingest files first")
    key = _world_key(traces, reader_model, grow)
    lock = _read_json(workdir / WORLD_LOCK_FILE, None)
    stored = _read_json(workdir / WORLD_REPORT_FILE, None)
    if (isinstance(lock, dict) and lock.get("key") == key
            and isinstance(stored, dict) and stored.get("paths")
            and all((workdir / p).exists() for p in stored["paths"])):
        return WorldReport.from_dict({**stored, "from_cache": True})

    written: list[str] = []

    def note(path: Path) -> None:
        written.append(str(path.relative_to(workdir)))

    sigs, schema = _mine_step(workdir, traces)
    schema, sigs, readers_artifact = _readers_step(workdir, traces, schema, sigs, reader_model)
    _, tasks = _cluster_step(workdir, traces, schema, sigs)
    anchor = draw_anchor(tasks, workdir)
    canon_rules = _canon_step(workdir, traces, schema)
    db, _, _, synthetic = _state_step(workdir, traces, schema, tasks, sigs,
                                      readers_artifact, canon_rules, anchor, grow)
    for path in _user_rules_step(workdir, traces, sigs):
        note(path)

    for name in ("tool_sigs.json", "unknown_tools.json", "row_homes.json",
                 "world_constants.json", "schema.json", readers.READERS_FILE,
                 "tasks.json", TASK_SPLIT, cluster.GROUPING_FILE, "tasks_frozen.json",
                 CANON_RULES, DB_FILE, "assumptions.json", WORLD_PROVENANCE_FILE,
                 ANCHOR_FILE, EVIDENCE_COUNTS_FILE):
        note(workdir / name)
    for task in tasks:
        note(workdir / "tasks" / f"{task.id}.json")
    for overlay in sorted((workdir / OVERLAY_DIR).glob("*.json")) \
            if (workdir / OVERLAY_DIR).is_dir() else []:
        note(overlay)
    if (workdir / "overlay_pins.json").is_file():
        note(workdir / "overlay_pins.json")
    if (workdir / "synthetic.json").is_file():
        note(workdir / "synthetic.json")
    if (workdir / READERS_DIR).is_dir():
        for path in sorted((workdir / READERS_DIR).rglob("*")):
            if path.is_file():
                note(path)

    calls_by_tool: dict[str, int] = {}
    for trace in traces:
        for call in trace.tool_calls:
            calls_by_tool[call.name] = calls_by_tool.get(call.name, 0) + 1
    report = WorldReport(
        tables={table: len(rows) for table, rows in (db or {}).items()},
        tools=dict(sorted(calls_by_tool.items())),
        tasks={t.id: len(t.run_ids) for t in tasks},
        gates={row.get("stage"): row for row in
               (_read_json(workdir / "gates.json", []) or [])
               if isinstance(row, dict) and row.get("stage") in ("mine", "readers", "cluster")},
        paths=sorted(set(written)),
        from_cache=False,
    )
    _write_json(workdir / WORLD_REPORT_FILE, report.to_dict())
    _write_json(workdir / WORLD_LOCK_FILE, {"key": key})
    return report
