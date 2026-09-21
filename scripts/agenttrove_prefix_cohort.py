from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

from kullback.builder import ingest, prefixes
from kullback.builder.sources import MapContext
from kullback.builder.sources.terminus_2 import Terminus2Adapter

ADAPTER = Terminus2Adapter()


def load_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heldout-ids", default=None)
    return parser.parse_args(argv)


class HeldoutError(Exception):
    pass


def strict_heldout(path_value):
    try:
        body = json.loads(Path(path_value).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HeldoutError("heldout list file is missing") from None
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        raise HeldoutError("heldout list file does not parse as JSON") from None
    if not isinstance(body, list):
        raise HeldoutError("heldout list must be a JSON array of nonempty strings")
    for item in body:
        if not isinstance(item, str) or not item:
            raise HeldoutError("heldout list must hold nonempty strings only")
    return frozenset(body)


def heldout_result(path_value):
    if path_value is None:
        return (frozenset(), None)
    try:
        return (strict_heldout(path_value), None)
    except HeldoutError as exc:
        return (frozenset(), str(exc))


def check_input(input_dir):
    inp = Path(input_dir)
    if not inp.exists():
        return "input directory is missing"
    if not inp.is_dir():
        return "input path is not a directory"
    if not sorted(inp.glob("*.json")):
        return "input directory holds no sample files"
    return None


def check_placement(input_dir, output):
    inp = Path(input_dir).resolve()
    out = Path(output).resolve()
    if out == inp:
        return "output must not be the input directory"
    for parent in out.parents:
        if parent == inp:
            return "output must not be inside the input directory"
    return None


def entry_error(path):
    target = Path(path)
    try:
        usable = target.is_file()
    except OSError:
        return "input entry cannot be inspected: " + target.name
    if not usable:
        return "input entry is not a readable regular file: " + target.name
    return None


def read_regular(path):
    name = Path(path).name
    try:
        handle = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return (None, "input file cannot be opened: " + name)
    try:
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            return (None, "input entry is not a readable regular file: " + name)
        parts = []
        while True:
            chunk = os.read(handle, 65536)
            if not chunk:
                break
            parts.append(chunk)
        return (b"".join(parts), None)
    except OSError:
        return (None, "input file cannot be read: " + name)
    finally:
        os.close(handle)


def take_snapshot(files):
    rows = []
    for path in files:
        payload, err = read_regular(path)
        if err is not None:
            return (None, "input files cannot be read for the opening snapshot")
        rows.append((path.name, len(payload), hashlib.sha256(payload).hexdigest()))
    return (rows, None)


def take_digest_map(files):
    digests = {}
    for path in files:
        payload, err = read_regular(path)
        if err is not None:
            return (None, "input files cannot be read for the closing snapshot")
        digests[path.name] = hashlib.sha256(payload).hexdigest()
    return (digests, None)


def inputs_error(before, after):
    names = {name for name, _size, _digest in before}
    for name, _size, digest in before:
        if after.get(name) != digest:
            return "input bytes changed during evaluation"
    for name in after:
        if name not in names:
            return "input bytes changed during evaluation"
    return None


class PublishError(Exception):
    pass


def stage_file(target, text):
    handle, tmppath = tempfile.mkstemp(prefix=target.name + ".staging-", dir=str(target.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmppath)
        raise
    return tmppath


def publish_staged(output, text):
    target = Path(output)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = stage_file(target, text)
    except OSError as exc:
        raise PublishError("cannot stage manifest: " + type(exc).__name__) from None
    try:
        os.link(staged, target)
    except FileExistsError:
        outcome = False
    except OSError as exc:
        raise PublishError("cannot publish manifest: " + type(exc).__name__) from None
    else:
        outcome = True
    finally:
        with contextlib.suppress(OSError):
            os.unlink(staged)
    return outcome


def derive_memory(path):
    payload, err = read_regular(path)
    if err is not None:
        raise OSError(err)
    raw_hash = hashlib.sha256(payload).hexdigest()
    document = json.loads(payload.decode("utf-8-sig"))
    confidence, _reasons = ADAPTER.detect(document)
    if confidence <= 0:
        raise ValueError("sample file is not terminus-2 behind the intake seam: " + Path(path).name)
    environment = ADAPTER.environment(document)
    recordings = list(ADAPTER.recordings(document))
    traces = []
    rejects = []
    for sim_index, simulation in enumerate(recordings):
        if not isinstance(simulation, dict):
            rejects.append(
                {
                    "trace_id": None,
                    "sim_index": sim_index,
                    "reason": "simulation is not an object",
                }
            )
            continue
        ctx = MapContext(
            raw_hash=raw_hash,
            index=sim_index,
            environment=environment,
            ingest_version=ingest.INGEST_VERSION,
        )
        try:
            trace = ADAPTER.to_trace(simulation, ctx)
        except (ValidationError, AttributeError, TypeError) as exc:
            rejects.append(
                {
                    "trace_id": str(simulation.get("id") or raw_hash[:12] + "-" + str(sim_index)),
                    "sim_index": sim_index,
                    "reason": "the recording is not parseable: " + type(exc).__name__,
                }
            )
            continue
        trace.hash = ingest.trace_hash(trace)
        traces.append(trace)
    ruling = ingest.rule_recordings(raw_hash, ADAPTER.name, recordings, traces, rejects)
    return (raw_hash, traces, ruling)


def original_tally(ruling):
    tally = {"task_eligible": 0, "evidence_only": 0, "rejected": 0}
    reasons = {}
    rows = ruling.get("recordings", []) if ruling else []
    for row in rows:
        standing = row.get("standing")
        if standing in tally:
            tally[standing] += 1
        reason = row.get("reason") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    return (tally, reasons, rows)


def rescuable(row):
    if row.get("standing") != ingest.EVIDENCE_ONLY:
        return False
    reasons = row.get("reasons") or []
    return "unresolved_call" in reasons


def evaluate_traces(traces, rows, heldout):
    by_digest = {}
    for row in rows:
        digest = row.get("trace_hash")
        if digest:
            by_digest[digest] = row
    selected = []
    withheld = {}
    retained_calls = 0
    dropped_calls = 0
    retained_turns = 0
    dropped_turns = 0
    candidates = 0
    for trace in traces:
        row = by_digest.get(trace.hash or "")
        if row is None or not rescuable(row):
            continue
        candidates += 1
        outcome = prefixes.select_prefix(trace, heldout_ids=heldout)
        if isinstance(outcome, prefixes.PrefixSelection):
            selected.append((trace, outcome))
            retained_calls += len(outcome.retained_calls)
            dropped_calls += len(trace.tool_calls) - len(outcome.retained_calls)
            retained_turns += len(outcome.retained_turns)
            dropped_turns += len(trace.turns) - len(outcome.retained_turns)
        else:
            withheld[outcome.reason] = withheld.get(outcome.reason, 0) + 1
    return (candidates, selected, withheld, retained_calls, dropped_calls, retained_turns, dropped_turns)


def range_row(trace, outcome):
    return {
        "cohort_id": outcome.cohort_id,
        "parent_trace_id": outcome.parent_trace_id,
        "parent_trace_hash": outcome.parent_trace_hash,
        "raw_hash": outcome.parent_raw_hash,
        "sim_index": outcome.sim_index,
        "turn_start": 0,
        "turn_end": outcome.turn_end,
        "retained_turns": len(outcome.retained_turns),
        "retained_calls": len(outcome.retained_calls),
        "dropped_turns": len(trace.turns) - len(outcome.retained_turns),
        "dropped_calls": len(trace.tool_calls) - len(outcome.retained_calls),
        "source": outcome.source,
        "ingest_version": outcome.ingest_version,
        "standing": outcome.standing,
    }


def evaluate_file(path, heldout):
    try:
        raw_hash, traces, ruling = derive_memory(path)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        return (None, "cannot derive " + path.name + ": " + type(exc).__name__)
    tally, reasons, rows = original_tally(ruling)
    found = evaluate_traces(traces, rows, heldout)
    candidates, selected, withheld, kept_calls, lost_calls, kept_turns, lost_turns = found
    return (
        {
            "file": path.name,
            "raw_hash": raw_hash,
            "recordings": len(rows),
            "original": dict(tally),
            "original_reasons": dict(reasons),
            "candidates": candidates,
            "selected": len(selected),
            "withheld_reasons": dict(withheld),
            "retained_calls": kept_calls,
            "dropped_calls": lost_calls,
            "retained_turns": kept_turns,
            "dropped_turns": lost_turns,
            "ranges": [range_row(trace, outcome) for trace, outcome in selected],
        },
        None,
    )


def prepare(args):
    bad_input = check_input(args.input_dir)
    if bad_input is not None:
        return (None, None, None, None, bad_input + ": " + str(args.input_dir))
    bad_place = check_placement(args.input_dir, args.output)
    if bad_place is not None:
        return (None, None, None, None, bad_place + ": " + str(args.output))
    heldout, heldout_err = heldout_result(args.heldout_ids)
    if heldout_err is not None:
        return (None, None, None, None, heldout_err)
    files = sorted(Path(args.input_dir).glob("*.json"))
    for path in files:
        flawed = entry_error(path)
        if flawed is not None:
            return (None, None, None, None, flawed)
    source = "none" if args.heldout_ids is None else "caller-supplied"
    return (Path(args.input_dir), files, heldout, source, None)


def finish(output, before, after, entries, totals, heldout_source, heldout_count):
    drift = inputs_error(before, after)
    if drift is not None:
        print(drift, file=sys.stderr)
        return 2
    historical = {"eligible": 92, "total": 200}
    manifest = {
        "selector": prefixes.SELECTOR_VERSION,
        "ingest_version": ingest.INGEST_VERSION,
        "floor": ingest.MIN_TASK_ELIGIBLE_SHARE,
        "writer_eligible": False,
        "heldout_anchor": heldout_source,
        "heldout_ids": heldout_count,
        "scratch_copies": "none; in-memory derivation only",
        "inputs_unchanged": True,
        "inputs": [
            {"file": name, "bytes": size, "sha256_before": digest, "sha256_after": after[name]}
            for name, size, digest in before
        ],
        "files": entries,
        "totals": dict(totals),
        "historical": dict(historical),
        "historical_match": totals["original_eligible"] == historical["eligible"]
        and totals["recordings"] == historical["total"],
        "standing_note": "every selected prefix evidence_only; original standings unchanged",
    }
    try:
        text = json.dumps(manifest, indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        print("cannot serialize manifest: " + type(exc).__name__, file=sys.stderr)
        return 2
    try:
        published = publish_staged(output, text)
    except PublishError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print("cannot encode manifest: " + type(exc).__name__, file=sys.stderr)
        return 2
    if not published:
        print("refusing to overwrite existing manifest: " + str(output), file=sys.stderr)
        return 2
    print("recordings: " + str(totals["recordings"]))
    print("original eligible: " + str(totals["original_eligible"]))
    print("selected prefixes: " + str(totals["selected_prefixes"]))
    print("inputs unchanged: True")
    print("historical match: " + str(manifest["historical_match"]))
    print("writer eligible: False")
    return 0


def run(argv):
    args = load_args(argv)
    input_dir, files, heldout, source, early = prepare(args)
    if early is not None:
        print(early, file=sys.stderr)
        return 2
    before, snap_err = take_snapshot(files)
    if snap_err is not None:
        print(snap_err, file=sys.stderr)
        return 2
    entries = []
    totals = {"recordings": 0, "original_eligible": 0, "selected_prefixes": 0}
    for path in files:
        entry, file_err = evaluate_file(path, heldout)
        if file_err is not None:
            print(file_err, file=sys.stderr)
            return 2
        entries.append(entry)
        totals["recordings"] += entry["recordings"]
        totals["original_eligible"] += entry["original"]["task_eligible"]
        totals["selected_prefixes"] += entry["selected"]
    after, final_err = take_digest_map(sorted(input_dir.glob("*.json")))
    if final_err is not None:
        print(final_err, file=sys.stderr)
        return 2
    return finish(args.output, before, after, entries, totals, source, len(heldout))


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
