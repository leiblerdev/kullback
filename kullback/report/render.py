"""The report as Markdown, one section at a time, in the order the design fixes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from kullback import claims, difficulty, round_snapshot, rounds
from kullback.report.data import (
    ENVIRONMENT,
    LESSONS,
    QUEUE,
    ROUNDS,
    SYNTHETIC,
    TASKS,
    USER_FIDELITY,
    ReportData,
    StageStatus,
)
from kullback.report.numbers import (
    _percent,
    _rate,
    assisted_tools_of,
    claims_for_task,
    false_rejection_by_task,
    flagged_tool_verdicts,
    suggestion,
    task_numbers,
)
from kullback.report.pipeline import environment_gate, pipeline_dag
from kullback.runner.records import RoundRecord, Run, Verdict


def _cell(text: str) -> str:
    """One Markdown table cell: a bar inside a gate failure or a scorecard note would end the cell
    and shift every column after it, and those strings come from the customer's own artifacts."""
    return str(text).replace("|", "\\|")


def _usd(value: Any) -> str:
    return "n/a" if value is None else f"${float(value):.2f}"


def _stop_lines(data: ReportData) -> list[str]:
    """Where the build stopped, what it spent and what finishing costs (D86)."""
    stop = data.stopped
    if not stop:
        return []
    done = [s.name for s in data.stages if s.status in ("ran", "cached", "rolled_back")]
    left = [s.name for s in data.stages if s.status in ("pending", "stopped", "failed")]
    lines = [
        f"Stopped in stage {stop.get('stage') or 'unknown'} on {stop.get('item') or 'no item named'}.",
        f"Completed stages: {', '.join(done) or 'none'}. Still to do: {', '.join(left) or 'none'}.",
        f"Spent {_usd(stop.get('spent'))} against a ceiling of {_usd(stop.get('ceiling_usd'))}; "
        f"the estimated cost to finish is {_usd(stop.get('estimate_to_finish'))}.",
    ]
    if stop.get("reason"):
        lines.append(f"Reason: {stop['reason']}.")
    lines.append("Continuing needs a person's permission, given as a new ceiling (D86).")
    return lines


def _bullets(items: list[str], empty: str) -> list[str]:
    """A bulleted subsection, or the one sentence that says there is nothing in it."""
    return items or [empty]


def _headline(data: ReportData) -> list[str]:
    """Built or not built, what decided that, and what stopped the build if anything did."""
    lines = [f"Environment built: {'yes' if data.built else 'no'}."]
    gate = environment_gate(data)
    if gate is not None:
        lines.append(
            f"The build Environment gate {'passed' if gate.passed else 'failed'}, which is what decides that"
            + (f": {'; '.join(gate.failures)}" if gate.failures else "")
            + "."
        )
    else:
        lines.append(
            "No build Environment gate was recorded, so this reads the Environment file and "
            "the pipeline status instead."
        )
    env = data.environment
    if env is not None:
        lines.append(
            f"Environment {env.env_id}, version {env.version} "
            f"(schema {env.schema_version}, tools {env.tools_version}, policy {env.policy_version})."
        )
    if data.stopped_reason:
        lines.append(f"Stopped: {data.stopped_reason}.")
    lines += _stop_lines(data)
    lines += _snapshot_lines(data)
    lines += _round_lines(data)
    if data.records_not_read:
        lines += [
            "",
            "### Records not read",
            "",
            "These files are on disk and did not load, so every number below is counted without them.",
        ]
        lines += [f"- {name}" for name in data.records_not_read]
    return lines


def _snapshot_lines(data: ReportData) -> list[str]:
    """Which round's Task table these numbers are read from, and how far the live files have moved
    from it (D218 rule 4).

    Every Task level number below is one round's answer, taken in one pass at that round's close.
    The live files go on being written afterwards, which is right, so the sentence that follows says
    how many Tasks now disagree with the table and at which stage: a reader who sees a number here
    that the workdir no longer agrees with is looking at movement and not at a regression.
    """
    if not data.drift:
        return []
    lines = [round_snapshot.drift_line(data.drift)]
    counts = dict((data.snapshot or {}).get("counts") or {})
    if counts:
        lines.append(
            f"That round ruled on {counts.get('tasks', 0)} Tasks: "
            f"{counts.get('fidelity', 0)} clearing fidelity, "
            f"{counts.get('reference', 0)} with a Reference, "
            f"{counts.get('verifier_passed', 0)} whose Verifier passed the suite, "
            f"{counts.get('trusted', 0)} trusted and {counts.get('refused', 0)} refused."
        )
    return lines


def _reverted_lines(data: ReportData) -> list[str]:
    """The repairs every round put back, by kind and by why (D201).

    One line per round that reverted anything, because a round that closed its red lights while
    three of its repairs were reverted for breaking Tasks elsewhere did less than its findings say,
    and the kind is what says which verb keeps buying nothing. The sentence is the driver's, off the
    round's own counts: this file reads records and works nothing out for itself.
    """
    lines = []
    for record in data.rounds:
        said = str((record.counts or {}).get("repairs_reverted") or "")
        if said:
            lines.append(f"round {record.round}: {said}.")
    return lines


def _round_lines(data: ReportData) -> list[str]:
    """What the rounds left: how many Tasks have a trusted Verifier, with the false-rejection number
    per Task beside it (D133), how many were refused and why, what the repairs of each round bought
    and what they were put back for (D201), and what a stalled exit hands a person."""
    if not data.rounds:
        return []
    last = data.rounds[-1].counts or {}
    fractions = false_rejection_by_task(data)
    per_task = ", ".join(f"{task_id} {_percent(value)}" for task_id, value in sorted(fractions.items()))
    lines = [
        f"{last.get('trusted', 0)} Tasks with a trusted Verifier; false rejection: per Task below"
        + (f" ({per_task})" if per_task else "")
        + "."
    ]
    refused = dict(last.get("refused") or {})
    lines.append(
        f"{len(refused)} Tasks refused"
        + (
            ": "
            + "; ".join(f"{task_id} ({reason or 'no reason recorded'})" for task_id, reason in sorted(refused.items()))
            if refused
            else ""
        )
        + "."
    )
    lines += _reverted_lines(data)
    if data.rounds[-1].exit == "stalled":
        unfinished = list(last.get("unfinished") or [])
        lines.append("stalled: these Tasks need a person: " + (", ".join(unfinished) or "none named") + ".")
    elif data.rounds[-1].exit == "max_rounds":
        lines.append(
            "round cap reached (D169): the loop stopped with "
            f"{len(list(last.get('unfinished') or []))} Tasks unfinished."
        )
    return lines


def _exit_cell(record: RoundRecord) -> str:
    """The exit a round ended on, with the beat that raised named beside it where one did (D231).

    The counts on the row are what the round measured before the raise, so without this a reader
    takes a round that stopped half way for a round that measured that much and stopped.
    """
    beat = str(((record.counts or {}).get(rounds.BEAT_ERROR) or {}).get("beat") or "")
    ended = record.exit or ""
    return f"{ended} (ended by {beat} error)".strip() if beat else ended


def _rounds_table(data: ReportData) -> list[str]:
    """One row per round: the counts the gates reported and the exit on the last (D126)."""
    lines = [
        "| round | fidelity | trusted | refused | assisted runs | probes passing | spend | exit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in data.rounds:
        counts = record.counts or {}
        spend = float((counts.get("spend") or {}).get("total") or 0.0)
        saved = (counts.get("spend") or {}).get("cache_saved")  # rounds before D152 carry none
        cache = f" (cache saved ${float(saved):.4f})" if saved is not None else ""
        lines.append(
            f"| {record.round} | {counts.get('fidelity', 0)}/{counts.get('tasks', 0)} | "
            f"{counts.get('trusted', 0)} | {counts.get('refused_count', 0)} | "
            f"{counts.get('assisted_runs', 0)} | {counts.get('probes_passing', 0)} | "
            f"${spend:.4f}{cache} | {_cell(_exit_cell(record))} |"
        )
    return lines


def _rounds(data: ReportData) -> list[str]:
    lines = [ROUNDS, ""]
    if not data.rounds:
        lines.append("No rounds recorded: this build ran no Builder and Examiner rounds, or rounds.json is missing.")
        return lines
    lines += _rounds_table(data)
    return lines


def _gates_table(data: ReportData) -> list[str]:
    lines = ["", "### Gates", "", "| stage | result | failures |", "| --- | --- | --- |"]
    for gate in data.gates or []:
        lines.append(
            f"| {_cell(gate.stage)} | {'pass' if gate.passed else 'fail'} | {_cell('; '.join(gate.failures))} |"
        )
    return lines if data.gates else lines + ["| none recorded |  |  |"]


def _scorecard_table(data: ReportData) -> list[str]:
    lines = [
        "",
        "### Scorecard, raw and explained side by side",
        "",
        "| number | raw | explained | note |",
        "| --- | --- | --- | --- |",
    ]
    for item in data.scorecard or []:
        lines.append(f"| {_cell(item.name)} | {_cell(item.raw)} | {_cell(item.explained)} | {_cell(item.note)} |")
    return lines if data.scorecard else lines + ["| none recorded |  |  |  |"]


def _difficulty_table(data: ReportData) -> list[str]:
    """The trusted count and the held-out solve rate per difficulty bucket (D209), off difficulty.json.

    One trusted count says nothing about what kind of Task it holds, so it is reported per bucket
    beside it and never instead of it. A build whose workdir has no difficulty.json says so rather
    than printing an empty table that reads as a build with no Tasks.
    """
    body = data.difficulty or {}
    rows = list(body.get("buckets") or [])
    if not rows and not body:
        return [
            "",
            "### Difficulty buckets",
            "",
            "No difficulty record was written for this build, so no bucket table is shown.",
        ]
    return ["", "### Difficulty buckets", ""] + difficulty.markdown_table(rows, len(body.get("no_record") or {}))


def _synthetic(data: ReportData) -> list[str]:
    """The Tasks walked over the mined graph, under their own heading and in nobody else's count (D224).

    A synthetic Task is a walk of the dependency graph the recordings showed, run in the rebuilt
    world, with a Verifier derived from where it landed. No recording gates it, so it is never
    trusted, never part of replay fidelity and never a Reference: what it buys is a pool wide enough
    to measure over-strictness on a thin corpus and Tasks at a difficulty a round asked for. The
    walks a body refused are printed beside the Tasks, because a corpus whose walks are mostly
    refused has a graph missing a precondition edge and the count is what says so.
    """
    lines = [SYNTHETIC, ""]
    body = data.synthetic or {}
    rows = list(body.get("tasks") or [])
    if not rows:
        return lines + ["No walk was generated into a bucket for this build."] + _domain_lines(data)
    counts = dict(body.get("counts") or {})
    graph_row = dict(counts.get("graph") or {})
    verified = sum(1 for row in rows if row.get("suite_passed"))
    lines += [
        f"{len(rows)} synthetic Tasks, {verified} of them verified by the D79 suite. None of them "
        "counts toward replay fidelity, a confirmed Reference or the trusted count.",
        "",
        f"Graph: {graph_row.get('nodes', 0)} tools, {graph_row.get('edges', 0)} edges "
        f"({graph_row.get('value_edges', 0)} carrying a value, {graph_row.get('row_edges', 0)} "
        f"joining a read to a write on one row), mined over {graph_row.get('runs', 0)} Runs.",
        "",
        f"Walks tried {counts.get('walks_tried', 0)}, refused by a body "
        f"{counts.get('walks_refused', 0)}, crashed {counts.get('walks_crashed', 0)}, "
        f"unbound {counts.get('walks_unbound', 0)}.",
        "",
    ]
    lines += ["| bucket asked | bucket reached | Tasks | suite passed | mean pool |", "| --- | --- | --- | --- | --- |"]
    grouped: dict = {}
    for row in rows:
        key = (str(row.get("bucket_requested") or ""), str(row.get("bucket") or ""))
        held = grouped.setdefault(key, {"tasks": 0, "passed": 0, "pool": 0})
        held["tasks"] += 1
        held["passed"] += 1 if row.get("suite_passed") else 0
        held["pool"] += int(row.get("pool") or 0)
    for (asked, reached), held in sorted(grouped.items()):
        mean = held["pool"] / held["tasks"] if held["tasks"] else 0.0
        lines.append(f"| {asked} | {reached} | {held['tasks']} | {held['passed']} | {mean:.1f} |")
    return lines + _domain_lines(data)


def _per_source_rows(rows: list) -> list[str]:
    """How many archetypes each page yielded and how many of those mapped, one row per URL."""
    held: dict = {}
    for row in rows or ():
        for url in row.get("sources") or [row.get("source")]:
            seen = held.setdefault(str(url), [0, 0])
            seen[0] += 1
            seen[1] += 1 if row.get("write_tools") else 0
    return [f"| {url} | {held[url][0]} | {held[url][1]} |" for url in sorted(held)]


def _domain_lines(data: ReportData) -> list[str]:
    """What the domain's own material attested, what could not be executed, and what was shaped (D225).

    Three numbers a reader wants beside the walks: how many task archetypes the public pages
    attested, how many of them this Environment can execute, and what fell at each rung of the
    realism bar. The gaps are the interesting half: they are what this domain does that this
    Environment does not, in the words of the page, and they are the first input to a specification
    environment rather than a defect of the reading.
    """
    body = data.domain or {}
    rows = list(body.get("archetypes") or [])
    if not rows and not data.domain_gaps:
        return []
    counts = dict(body.get("counts") or {})
    shaped = dict((data.shaped or {}).get("counts") or {})
    fell = dict(shaped.get("fell") or {})
    lines = [
        "",
        "### Task archetypes read off the domain's own material (D225)",
        "",
        f"{counts.get('sources_read', 0)} public pages read, "
        f"{counts.get('sources_refused', 0)} refused; "
        f"{counts.get('archetypes_extracted', 0)} archetypes extracted, "
        f"{counts.get('archetypes_contaminated', 0)} dropped as contaminated, "
        f"{counts.get('archetypes_copied', 0)} dropped for copying the page, "
        f"{counts.get('archetypes_folded', 0)} folded as one goal said twice.",
        "",
        f"{counts.get('archetypes_mapped', 0)} archetypes map onto tools this Environment can "
        f"reach, and {len(data.domain_gaps)} do not. Nothing here is trusted, part of replay "
        "fidelity or a confirmed Reference.",
        "",
    ]
    if shaped:
        lines += [
            f"Tasks shaped: {shaped.get('tasks_shaped', 0)} over {shaped.get('archetypes_read', 0)} archetypes.",
            "",
            "| realism rung | fell |",
            "| --- | --- |",
        ]
        lines += [f"| {rung} | {fell.get(rung, 0)} |" for rung in fell]
        lines += ["", "| source | Tasks shaped |", "| --- | --- |"]
        lines += [f"| {row.get('source')} | {row.get('tasks')} |" for row in shaped.get("per_source") or []] or [
            "| none |  |"
        ]
        lines.append("")
    lines += ["| source | archetypes | mapped |", "| --- | --- | --- |"]
    lines += _per_source_rows(rows)
    if data.domain_gaps:
        lines += ["", "Coverage gaps: what this domain does that this Environment cannot execute.", ""]
        lines += [f"- {row.get('goal')}" for row in data.domain_gaps]
    return lines


def _claims_table(data: ReportData) -> list[str]:
    """What the transcripts claimed against what the state received, per Task and over the corpus (D223).

    The Verdict already grades state alone (D46), so a false claim never earned a pass; what this
    adds is the class. A Candidate that answered "done" and wrote nothing and one that wrote the
    wrong row were one failure count, and they ask for two different repairs. The Tasks flagged
    below are the ones where every failing held-out Run claimed a write the state never received,
    which is where the Simulated user's end protocol accepted words for a state change.
    """
    body = data.claims or {}
    totals = body.get("totals") or {}
    if not body:
        return [
            "",
            "### Claims against state",
            "",
            "No claim record was written for this build, so no claim table is shown.",
        ]
    lines = [
        "",
        "### Claims against state",
        "",
        f"{totals.get('runs_with_claims', 0)} of {totals.get('runs', 0)} Runs claim a write in "
        f"words: {totals.get('claims', 0)} claims, {totals.get('claims_written', 0)} answered by "
        f"a write the state received and {totals.get('claims_unwritten', 0)} answered by none. "
        f"{totals.get('writes_unclaimed', 0)} writes happened that no transcript mentions.",
        f"Of {totals.get('failing_runs', 0)} failing Runs, "
        f"{totals.get('claimed_unwritten_failures', 0)} claimed a write nothing received "
        f"({_percent(totals.get('claimed_unwritten_share'))}). Mean partial completion "
        f"{_percent(totals.get('partial_completion_mean'))}.",
        "",
        claims.LEGEND,
        "",
    ]
    lines += claims.markdown_table(body.get("tasks") or {})
    flagged = list(body.get("flagged") or [])
    lines += [
        "",
        (
            "Flagged for the Simulated user's end protocol: "
            + ", ".join(flagged)
            + ". Every failing held-out Run of each claimed a write the state never received."
            if flagged
            else "No Task is flagged for the Simulated user's end protocol: no Task fails only on "
            "claims the state never received."
        ),
    ]
    return lines


def _claim_task_lines(data: ReportData, task_id: str) -> list[str]:
    """The two lines D223 puts beside a Task's numbers: what it claimed, and how far its Runs got."""
    row = claims_for_task(data, task_id)
    if not row:
        return []
    flagged = task_id in list((data.claims or {}).get("flagged") or [])
    lines = [
        f"- Claims: {row.get('claims', 0)}, of which {row.get('claims_unwritten', 0)} name a "
        f"write the state never received; {row.get('writes_unclaimed', 0)} writes no transcript "
        f"mentions",
        f"- Partial completion: {_percent(row.get('partial_completion_mean'))} of atoms confirmed "
        f"per Run (band {row.get('partial_completion_band', 'none')}), beside trusted and not "
        f"instead of it",
    ]
    if flagged:
        lines.append(
            "- Flagged: every failing held-out Run of this Task claimed a write the state "
            "never received, so the Simulated user's end protocol is what to read next"
        )
    return lines


def tool_fidelity_counts(data: ReportData, name: str) -> dict:
    """Both grains of one tool's replay fidelity, off tool_fidelity.json (D171).

    `calls` and `replayed` are the corpus number the compile_tools gate ruled on and the Builder
    repairs against. `tasks` is how many Tasks made a recorded call of the tool at all, and `blocked`
    how many of them have an own call the body answers differently, which is the only count that
    costs a Reference. The two part company: a tool can miss one call in the corpus and block one
    Task while forty others call it and are answered correctly.
    """
    tools = (data.tool_fidelity or {}).get("tools") or {}
    tasks = (data.tool_fidelity or {}).get("tasks") or {}
    per_tool = tools.get(name) or {}
    calling = [row for row in tasks.values() if name in (row or {})]
    return {
        "calls": int(per_tool.get("calls") or 0),
        "replayed": int(per_tool.get("replayed") or 0),
        "tasks": len(calling),
        "blocked": sum(1 for row in calling if row[name].get("differing")),
    }


def assisted_tool_note(data: ReportData, name: str) -> str:
    """The sentence beside one assisted tool: what it stood in for, and what it actually costs (D171)."""
    parts = []
    if name in data.assisted_share:
        parts.append(f"{_percent(data.assisted_share[name])} of its calls stood in")
    counts = tool_fidelity_counts(data, name)
    if counts["calls"]:
        parts.append(f"{counts['replayed']} of {counts['calls']} recorded calls replayed")
    if counts["tasks"]:
        parts.append(f"{counts['tasks']} Tasks call it, {counts['blocked']} blocked by their own differing calls")
    return f": {'; '.join(parts)}" if parts else ""


def _tool_notes(data: ReportData) -> list[str]:
    """What a person has to look at before trusting the numbers: tools that stood in, Tasks with
    no anchor, tools nobody classed read or write (D70), and the Environment's open flags."""
    env = data.environment
    assisted = list(env.assisted_tools) if env is not None else []
    lines = ["", "### Assisted tools", ""]
    lines += _bullets(
        [f"- {name}{assisted_tool_note(data, name)}" for name in assisted],
        "No assisted tools: every tool here is real code.",
    )

    lines += ["", "### Unguarded Tasks", ""]
    lines += _bullets(
        [f"- {t.id}: {t.name or t.intent or 'no name yet'}" for t in data.tasks if t.unguarded],
        "No unguarded Tasks: every Task held Runs back for the anchor.",
    )

    lines += ["", "### Flagged tools, read or write not confirmed (D70)", ""]
    lines += _bullets(
        [
            f"- {name}: {count} Verdicts rest on a Run that called it"
            for name, count in flagged_tool_verdicts(data).items()
        ],
        "No flagged tools: every tool here is confirmed read or write.",
    )

    lines += ["", "### Open flags on the Environment", ""]
    lines += _bullets(
        [f"- {flag}" for flag in (env.flags if env is not None else [])],
        "No open flags: nothing on the Environment is waiting on the setup review.",
    )
    return lines


def _finding_lines(data: ReportData) -> list[str]:
    """What the recorded system does that its tool descriptions do not say (D155).

    The Builder records one when several recorded calls agree with each other and part from the
    description at one leaf; the body reproduces the recording, because that is what a Candidate is
    graded against, and this is where the customer reads what was reproduced and on which calls.
    """
    if not data.findings:
        return []
    lines = ["", "### Findings, what the recorded system does that its descriptions do not say", ""]
    for row in data.findings:
        args = row.get("arguments") or {}
        evidence = [str(e) for e in (args.get("evidence") or [])]
        cited = f" (recorded calls: {', '.join(evidence)})" if evidence else ""
        lines.append(f"- {args.get('tool') or row.get('target')}: {args.get('finding') or ''}{cited}")
    return lines


def _overlay_lines(data: ReportData) -> list[str]:
    rows = sum(len(o.rows) for o in data.overlays)
    return ["", "### Overlays", "", f"- Tasks with an overlay: {len(data.overlays)}", f"- Overlay rows: {rows}"] + [
        f"  - {o.task_id}: {len(o.rows)} overlay rows" for o in data.overlays
    ]


def _pipeline_lines(data: ReportData) -> list[str]:
    lines = ["", "### Pipeline", "", "```mermaid", pipeline_dag(data.stages), "```"]
    cost = sum(stage.usd for stage in data.stages)
    if not cost:
        return lines + ["", "Cost per stage: nothing recorded, so this build's spend is not known."]
    lines += [
        "",
        "Cost per stage: " + ", ".join(_cost_cell(s) for s in data.stages if s.usd),
        f"Cost so far: ${cost:.2f}.",
    ]
    memo = sum(s.memo_hits for s in data.stages)
    if memo:
        lines += [f"Calls answered from the memo, at no cost: {memo}."]
    return lines


def _environment(data: ReportData) -> list[str]:
    """The Environment section, in the order a person reads it: what was built, what the gates and
    the scorecard said, what still needs a look, and what the pipeline did and cost."""
    return (
        [ENVIRONMENT, ""]
        + _headline(data)
        + _gates_table(data)
        + _scorecard_table(data)
        + _difficulty_table(data)
        + _claims_table(data)
        + _tool_notes(data)
        + _finding_lines(data)
        + _overlay_lines(data)
        + ["", "### Coverage", ""]
        + _coverage(data)
        + _pipeline_lines(data)
    )


def _cost_cell(stage: StageStatus) -> str:
    cell = f"{stage.name} ${stage.usd:.2f}"
    if stage.cache_share:
        cell += f" ({_percent(stage.cache_share)} of input from cache)"
    return cell


def _coverage(data: ReportData) -> list[str]:
    covered = [c for c in data.task_coverage if c.covered]
    total_runs = sum(c.run_count for c in data.task_coverage)
    covered_runs = sum(c.run_count for c in covered)
    share = _percent(_rate(covered_runs, total_runs))
    lines = [f"- Task coverage: {len(covered)} of {len(data.task_coverage)} Tasks covered, {share} of Runs."]
    for item in data.task_coverage:
        if not item.covered:
            lines.append(f"  - {item.task_id}: not covered, {item.reason or 'no reason recorded'}")
    exercised = [c for c in data.policy_items if c.id in set(data.policy_exercised)]
    untested = [c for c in data.policy_items if c.id not in set(data.policy_exercised)]
    lines.append(
        f"- Policy coverage: your traces exercise {len(exercised)} of {len(data.policy_items)} "
        "policy items; the rest are not tested."
    )
    for item in untested:
        lines.append(f"  - {item.id}: {item.text}")
    lines += _pending_review(data)
    return lines


def _pending_review(data: ReportData) -> list[str]:
    """Rewritten rules waiting for the setup review (D76, D48).

    A rule the Builder rewrote but nobody accepted is in no Verifier and in no residual list, so
    without this block nothing tells the reviewer it exists. The filter is policy.pending_review's,
    repeated over the records because the report package reads records off disk rather than importing builder/.
    """
    pending = [
        c
        for c in data.policy_items
        if c.rewritten_text and not c.compiled and not c.judge_atom and not c.residual_reason
    ]
    if not pending:
        return []
    lines = [
        f"- Awaiting setup review, not checked: {len(pending)} rewritten policy "
        f"{'rule' if len(pending) == 1 else 'rules'}. Until a person accepts the rewrite, each "
        "is in no Verifier and in no residual list, so nothing checks it."
    ]
    for item in pending:
        lines.append(f"  - {item.id}: {item.text}")
        lines.append(f"    rewritten as: {item.rewritten_text}")
    return lines


def _tasks(data: ReportData) -> list[str]:
    aside = {str(row.get("task_id")): str(row.get("reason", "disputed")) for row in data.tasks_aside}
    fractions = false_rejection_by_task(data)
    lines = [TASKS, ""]
    if data.kind == "batch":
        lines += ["These are the numbers for one Run batch against the Environment above.", ""]
    if not data.tasks:
        lines += ["No Tasks in this build.", ""]
    for task in data.tasks:
        numbers = task_numbers(data, task)
        lines.append(f"### Task {task.id}: {task.name or task.intent or 'no name yet'}")
        lines.append("")
        if task.intent:
            lines.append(f"Intent: {task.intent}")
            lines.append("")
        if task.id in aside:
            lines.append(f"Not gradeable, Reference disputed ({aside[task.id]}).")
            lines.append("")
        lines += _task_numbers_lines(data, numbers)
        lines += _claim_task_lines(data, task.id)
        if data.trusted is not None:
            lines += _trust_lines(data, task.id, fractions)
        lines += ["", suggestion(numbers, data.built, aside.get(task.id)), ""]
    return lines


def _trust_lines(data: ReportData, task_id: str, fractions: dict) -> list[str]:
    """Whether this Task's Verifier is trusted, and the false-rejection number that stands beside it (D133)."""
    metrics = data.trusted.metrics if data.trusted is not None else {}
    if task_id in (metrics.get("trusted") or []):
        standing = "trusted"
    elif task_id in (metrics.get("untrusted") or {}):
        standing = f"not trusted, {metrics['untrusted'][task_id]}"
    else:
        standing = "no Verifier ruled on"
    return [
        f"- Verifier: {standing}",
        f"- False rejection: {_percent(fractions.get(task_id))} of held-out frontier Runs rejected",
    ]


def _path_words(record: Verdict) -> str:
    """D46: a Verdict says whether the Run reached the End state on the Reference's path or another one."""
    if record.same_path is None:
        return "path not recorded"
    return "same path as the Reference" if record.same_path else "different path from the Reference"


def _verdict_line(record: Verdict, kinds: dict) -> str:
    """One counted Verdict as the design words it: pass or fail, the failing atom, the path, the cause."""
    atom = (
        f"failing atom {record.failing_atom} ({kinds.get(record.failing_atom, 'unknown')})"
        if record.failing_atom
        else "no failing atom"
    )
    cause = f"cause {record.cause}" if record.cause else "no cause recorded"
    return f"  - {record.run_id}: {'pass' if record.passed else 'fail'}, {atom}, {_path_words(record)}, {cause}"


def _uncounted_line(run: Optional[Run], record: Verdict) -> str:
    """One Verdict left out of the numbers, with the mark or the tool that excluded it (D49, D88)."""
    reasons = []
    if record.environment_suspected:
        reasons.append("environment suspected" + (f", cause {record.cause}" if record.cause else ""))
    if run is not None and run.assisted:
        stood_in = assisted_tools_of(run)
        reasons.append("assisted Run" + (f", {', '.join(stood_in)} stood in" if stood_in else ", a tool stood in"))
    return f"  - {record.run_id}: {'; '.join(reasons) or 'not counted'}"


def _task_numbers_lines(data: ReportData, numbers: dict) -> list[str]:
    judge_note = f"judge disagreement rate {_percent(numbers['judge_disagreement_rate'])}"
    judge_note += (
        f", audit rate {_percent(data.audit_rate)}"
        if data.audit_rate is not None
        else ", no human labels yet, so no error bound"
    )
    lines = [
        f"- Runs graded: {numbers['runs_graded']}",
        f"- Runs not counted (assisted or environment suspected): {numbers['assisted_not_counted']}",
        f"- Overlay rows in this Task's Starting state: {numbers['overlay_rows']}",
        f"- Judge atoms: {numbers['judge_atoms']} ({judge_note})",
        f"- Frontier pass rate: {_percent(numbers['frontier_pass_rate'])} ({numbers['frontier_runs']} Runs)",
        f"- Candidate pass rate: {_percent(numbers['candidate_pass_rate'])} ({numbers['candidate_runs']} Runs)",
        f"- Margin: {'n/a' if numbers['margin'] is None else format(numbers['margin'], '+.2f')}",
        "- Failing atoms by class: " + (_counts(numbers["failing_atoms"]) or "none"),
        "- Causes: " + (_counts(numbers["causes"]) or "none"),
    ]
    if numbers.get("superseded"):
        lines.append(
            f"- Superseded Verdicts not counted (an older Verifier or Environment version): "
            f"{numbers['superseded']}; each Run is counted once, under the versions on disk now"
        )
    kinds = numbers.get("atom_kinds") or {}
    counted = numbers.get("counted") or []
    lines.append("- Verdicts counted, one line each:" if counted else "- Verdicts counted, one line each: none")
    lines += [_verdict_line(record, kinds) for record in counted]
    uncounted = numbers.get("uncounted") or []
    if uncounted:
        lines.append("- Verdicts not counted, and what excluded them:")
        by_id = numbers.get("runs_by_id") or {}
        lines += [_uncounted_line(by_id.get(record.run_id), record) for record in uncounted]
    return lines


def _counts(counts: dict) -> str:
    return ", ".join(f"{name} {number}" for name, number in sorted(counts.items()))


def _judge_models_lines(data: ReportData) -> list[str]:
    """The judge models this build used, named when they are not the model that built it (D160).

    A judge on the build's own model is the default and says nothing new; a judge on another model,
    or two judges on two models, is what a reader has to know to read the rate below.
    """
    models = data.judge_models or {}
    build = models.get("build")
    named = [models[role] for role in ("judge", "second_judge") if models.get(role) and models[role] != build]
    if not named:
        return []
    against = f", where the build model is {build}" if build else ""
    return [f"Judge models: {', '.join(named)}{against}.", ""]


def _by_pair_lines(by_pair: dict) -> list[str]:
    """One line per pair of judges: how often those two parted (D160).

    Nothing is printed for a build whose pair rows predate the judge names, which is a silence about
    a number that was never recorded rather than a zero that was never measured.
    """
    if not by_pair:
        return []
    lines = ["", "Disagreement by judge pair:"]
    for name, row in sorted(by_pair.items()):
        lines.append(
            f"- {name}: {row.get('disagreements', 0)} of {row.get('pairs', 0)} pairs ({_percent(row.get('rate'))})"
        )
    return lines


def _queue(data: ReportData) -> list[str]:
    lines = [QUEUE, ""]
    pairs = data.judge_disagreement.get("pairs", 0)
    disagreements = data.judge_disagreement.get("disagreements", len(data.disagreement_queue))
    bound = (
        f" Audit rate {_percent(data.audit_rate)} from the queue items a person has resolved, "
        "which is the labelled set this number is bounded by (D92)."
        if data.audit_rate is not None
        else " No human labels yet, so this number has no error bound."
    )
    lines += _judge_models_lines(data)
    lines.append(
        f"Judge disagreement: {disagreements} of {pairs} pairs "
        f"({_percent(data.judge_disagreement.get('rate'))})." + bound
    )
    lines += _by_pair_lines(data.judge_disagreement.get("by_pair") or {})
    abstains = data.judge_disagreement.get("abstains")
    if abstains is not None:
        lines.append(
            f"Judge abstention: {abstains} of {pairs} pairs "
            f"({_percent(data.judge_disagreement.get('abstain_rate'))}). An abstain decides nothing, "
            "so D92 sends it to a person on the same terms as a split."
        )
    lines.append("")
    if not data.disagreement_queue and not data.tasks_aside:
        lines.append("The judges agreed everywhere they were asked: nothing in the queue, no Tasks set aside.")
        return lines
    split, undecided = _queue_split(data.disagreement_queue)
    for heading, rows in (
        ("### Items a person may resolve", split),
        ("### Items the judges did not decide", undecided),
    ):
        if not rows:
            continue
        lines += [heading, ""]
        for row in rows:
            third = f", a third sample said {row['verdict_c']}" if row.get("verdict_c") else ""
            lines.append(
                f"- {row.get('use', 'judge')} on {row.get('item_id', 'unknown')}: "
                f"one judge said {row.get('verdict_a', 'unknown')}, "
                f"the other said {row.get('verdict_b', 'unknown')}" + third + _queue_reason_words(row)
            )
            lines += _cited_spans(row)
        lines.append("")
    if data.tasks_aside:
        lines += ["### Tasks set aside, not gradeable", ""]
        for row in data.tasks_aside:
            lines.append(
                f"- {row.get('task_id', 'unknown')}: {row.get('reason', 'disputed')}, "
                "not gradeable until a person resolves it"
            )
        lines.append("")
    return lines


_QUEUE_REASONS = {
    "refused": ", and both refused for want of a tool check",
    "agreed_abstain": ", and both abstained",
    "abstain_majority": ", and the majority of the three abstained",
}


def _queue_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The queue in its two halves: judges that split, and judges that decided nothing (D92, D88).

    A row with no reason predates the reason field and is read as a split, which is what the two
    verdicts on it say.
    """
    split = [r for r in rows if (r.get("reason") or "split") == "split"]
    return split, [r for r in rows if (r.get("reason") or "split") != "split"]


def _queue_reason_words(row: dict) -> str:
    return _QUEUE_REASONS.get(str(row.get("reason") or ""), "")


def _cited_spans(row: dict) -> list[str]:
    """The spans each judge cited, which is what D92 puts in the queue beside the two verdicts."""
    lines = []
    for key, who in (("judge_a", "first judge"), ("judge_b", "second judge"), ("judge_c", "third sample")):
        side = row.get(key)
        spans = list((side or {}).get("cited_spans") or []) if isinstance(side, dict) else []
        for span in spans:
            lines.append(f"  - {who} cited: {span}")
    return lines


def _user_fidelity(data: ReportData) -> list[str]:
    """How close the Simulated user's turns are to the recorded ones (D214 rule 5).

    Half of an Environment is the person the Candidate is talking to, and a report that says nothing
    about it lets a corpus with a user that runs out of scenario read as a corpus with hard Tasks.
    A build written before D214 has no such file and says so in one line.
    """
    from kullback.user.fidelity import markdown_table

    lines = [USER_FIDELITY, ""]
    if not data.user_fidelity:
        lines.append(
            "This build recorded no user fidelity: nothing scored the Simulated user's turns against the recorded ones."
        )
        return lines
    lines.append(
        "Each driver's turns against the turns the recording holds, per Task, meaned over "
        "the corpus. The rule-driven user is the baseline and costs nothing."
    )
    lines.append("")
    lines += markdown_table(data.user_fidelity)
    return lines


def _lessons(data: ReportData) -> list[str]:
    lines = [LESSONS, ""]
    if not data.lessons_set_aside:
        lines.append("The Builder applied every lesson it carried: no lessons were set aside for this build.")
        return lines
    lines.append("Lessons the Builder judged not relevant here. A wrongly discarded lesson is visible below.")
    lines.append("")
    for lesson in data.lessons_set_aside:
        lines.append(f"- {lesson.id or 'lesson'}: {lesson.pattern} (set aside: {lesson.reason})")
    return lines


def render(data: ReportData) -> str:
    """The whole report as Markdown, in the one order D85 fixes."""
    lines = [f"# {data.title}", ""]
    lines += _environment(data) + [""]
    lines += _rounds(data) + [""]
    lines += _tasks(data) + [""]
    lines += _synthetic(data) + [""]
    lines += _queue(data) + [""]
    lines += _user_fidelity(data) + [""]
    lines += _lessons(data) + [""]
    return "\n".join(lines)


def write_report(data: ReportData, workdir: Any, name: str = "report.md") -> Path:
    """Write the report beside the records it read."""
    path = Path(workdir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(data), encoding="utf-8")
    return path
