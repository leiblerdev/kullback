"""Survey terminal agent recordings, admit them, and say what the pipeline would mine.

Read only over the sample: this script prints ids, column names, counts and shapes, never
trajectory text. It reuses the adapter behind the intake seam for every classification, so the
report and the pipeline count the same way.

Usage: uv run python scripts/terminal_survey.py <sample-dir> <workdir>
"""

import collections
import json
import sys
from pathlib import Path

from kullback.builder import ingest
from kullback.builder.sources import terminus_2 as adapter_mod

ADAPTER = adapter_mod.Terminus2Adapter()


def load_recordings(sample_dir: Path) -> tuple[list, list[tuple[str, dict]]]:
    """Every recording in the sample envelopes, with (file name, envelope) pairs."""
    envelopes = []
    recordings = []
    for path in sorted(sample_dir.glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelopes.append((path.name, envelope))
        for recording in ADAPTER.recordings(envelope):
            recordings.append((path.name, recording))
    return recordings, envelopes


def shape_survey(recordings: list) -> None:
    """Step 1: the mapping inputs, as counts and shapes."""
    turn_counts = []
    assistant_kinds: collections.Counter = collections.Counter()
    user_kinds: collections.Counter = collections.Counter()
    entry_shapes: collections.Counter = collections.Counter()
    batch_sizes: collections.Counter = collections.Counter()
    contradictions: collections.Counter = collections.Counter()
    command_entries = 0
    for _, recording in recordings:
        turns = recording.get(adapter_mod.COLUMN_TURNS) or []
        turn_counts.append(len(turns))
        for key, value in adapter_mod.contradictions(turns).items():
            contradictions[key] += value
        for position, turn in enumerate(turns):
            text = turn.get("content") or ""
            if turn.get("role") == "assistant":
                kind = adapter_mod.turn_kind(text)
                assistant_kinds[kind] += 1
                if kind in adapter_mod.COMMAND_KINDS:
                    entries = adapter_mod.command_entries(text) or []
                    command_entries += len(entries)
                    batch_sizes[len(entries)] += 1
                    for entry in entries:
                        entry_shapes[tuple(sorted(entry))] += 1
            else:
                headers = [h for h in adapter_mod.OUTPUT_HEADERS if h in text]
                if position == 0:
                    user_kinds["opening_instruction_with_state"] += 1
                elif headers:
                    if "WARNING" in text or "parsing errors" in text:
                        user_kinds["terminal_output_with_warning"] += 1
                    else:
                        user_kinds["terminal_output"] += 1
                else:
                    user_kinds["scaffold_note_" + adapter_mod.note_kind(text)] += 1
    print(f"recordings: {len(recordings)}")
    print(f"turns per recording: min {min(turn_counts)} max {max(turn_counts)} "
          f"mean {sum(turn_counts) / len(turn_counts):.2f}")
    print(f"turn count distribution: {sorted(collections.Counter(turn_counts).items())}")
    print(f"assistant turns by kind: {dict(assistant_kinds)}")
    print(f"user turns by kind: {dict(user_kinds)}")
    print(f"command entries: {command_entries} shapes: {dict(entry_shapes)}")
    print(f"entries per command turn: {sorted(batch_sizes.items())}")
    print(f"contradictions: {dict(contradictions)}")


def outcome_columns(recordings: list) -> None:
    """Which sidecar columns are filled, as counts."""
    columns: collections.Counter = collections.Counter()
    for _, recording in recordings:
        for key, value in recording.items():
            if key != adapter_mod.COLUMN_TURNS and value:
                columns[key] += 1
    print(f"filled sidecar columns (of {len(recordings)}): {dict(columns)}")


def admit(sample_dir: Path, workdir: Path) -> list:
    """Step 5: run the branch admission on each sample file, report standings with reasons."""
    admitted = []
    for path in sorted(sample_dir.glob("*.json")):
        raw_hash = ingest.store_raw(path, workdir).raw_hash
        try:
            summary = ingest.ingest_file(path, workdir)
        except ingest.IntakeGateError as exc:
            gate = exc.gate
            print(f"{path.name}: gate FAIL eligible {gate.metrics['task_eligible']} "
                  f"evidence {gate.metrics['evidence_only']} "
                  f"rejected {gate.metrics['rejected']} "
                  f"share {gate.metrics['eligible_share']:.2f}")
            _print_reasons(ingest.read_intake_ruling(workdir, raw_hash))
            continue
        gate = summary["gate"]
        print(f"{path.name}: gate pass eligible {gate['metrics']['task_eligible']} "
              f"evidence {gate['metrics']['evidence_only']} rejected {gate['metrics']['rejected']}")
        ruling = ingest.read_intake_ruling(workdir, summary["raw_hash"])
        _print_reasons(ruling)
        admitted.extend(summary["trace_hashes"])
    return admitted


def _print_reasons(ruling: dict) -> None:
    if ruling:
        print(f"  reasons: {ruling.get('reasons')} "
              f"set_aside outcomes: {ruling.get('set_aside', {}).get('outcomes')} "
              f"turns: {ruling.get('set_aside', {}).get('turns')} "
              f"calls: {ruling.get('set_aside', {}).get('tool_calls')}")


def mine_and_group(sample_dir: Path, workdir: Path) -> None:
    """Step 6, read only: what the miner and the grouping proposer do with eligible traces."""
    from kullback.builder.cluster import cluster_runs, write_tool_names
    from kullback.builder.mine import mine_tools

    eligible = []
    published = 0
    for path in sorted(sample_dir.glob("*.json")):
        raw_hash = ingest.store_raw(path, workdir).raw_hash
        traces = ingest.derive_traces(raw_hash, workdir)
        ruling = ingest.read_intake_ruling(workdir, raw_hash)
        wanted = {row["trace_hash"] for row in ruling.get("recordings", [])
                  if row["standing"] == ingest.TASK_ELIGIBLE and row.get("trace_hash")}
        eligible.extend(trace for trace in traces if trace.hash in wanted)
    published = len(list((workdir / "traces").glob("*.json")))
    print(f"eligible traces: {len(eligible)} (published to workdir/traces: {published}; "
          f"failed files publish nothing)")
    with_calls = sum(1 for trace in eligible if trace.tool_calls)
    print(f"eligible traces carrying shell calls: {with_calls}")
    print("eligible trace ids (first 20, for hand reading against their recordings):")
    for trace in eligible[:20]:
        print(f"  {trace.trace_id} turns {len(trace.turns)} calls {len(trace.tool_calls)}")
    sigs = mine_tools(eligible)
    for sig in sigs:
        fields = sorted((field.name, sorted(field.types)) for field in sig.args_fields)
        print(f"mined tool: {sig.name} kind {sig.kind} args {fields}")
    print(f"write tools: {sorted(write_tool_names(sigs))}")
    categories, tasks = cluster_runs(eligible, sigs)
    print(f"categories: {len(categories)} tasks: {len(tasks)} "
          f"singletons: {sum(1 for task in tasks if len(task.run_ids) < 2)}")


def main() -> None:
    sample_dir = Path(sys.argv[1])
    workdir = Path(sys.argv[2])
    recordings, _ = load_recordings(sample_dir)
    shape_survey(recordings)
    outcome_columns(recordings)
    admit(sample_dir, workdir)
    mine_and_group(sample_dir, workdir)


if __name__ == "__main__":
    main()
