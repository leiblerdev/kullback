# Scaling Environments

The plan for breadth: more Tasks and more Environments, beside the depth (replay fidelity) each build already
chases. It matters because Agent-World (arXiv 2604.18292) and Agent World Model (arXiv 2602.10090) both report
agent accuracy rising with the number of environments, steepest in the first five hundred.

Every mechanism here keys on classes the harness mines (tool dependency, row identity, column class, difficulty),
never on a corpus. Nothing synthetic counts as trusted: synthetic Tasks, specification Environments and forks each
report under their own heading, so a card says how much of an Environment is recorded and how much is generated.

## Four axes

1. More Tasks per Environment, without new traces. In code as `kullback synth`.
2. Environments from a specification instead of traces. Next.
3. Forks of a trusted Environment. After 2.
4. Many corpora at once. Operations.

### 1. Synthetic Tasks from scenarios

A scenario is one situation customers of the business meet, in general words, with the policy lines it touches
(D372). The generator writes scenarios from three sources: the recordings' opening asks, pages a cited web search
returns, and lines a customer typed. The Examiner reviews each one blind to its source (D373). A passed scenario
bound to rows of the world becomes a Task through the same Spec and review loop as a recorded Task, in its own
`synthetic` tier (D389).

```bash
uv run kullback synth scenarios work --model provider/model
uv run kullback synth review work --model provider/model
uv run kullback synth tasks work --store work
```

The rules a synthetic Task follows: it has no Reference and is never tested against a valid Run; its Verifier
comes from the scenario; realism is judged against the domain; difficulty rises only by more work in the world,
never by a vaguer ask or a stricter check.

Before scenarios, the source was walks over the tool-call dependency graph (`kullback synthesise`, D224).

### 1b. More rows in the Starting state

`builder/synth.py` grows the Starting state past the ids the traces named (D40, D107), through the Builder's
`grow` tool. Ids and closed vocabularies are drawn by code from the observed rows, related rows grow as one
bundle, and observed rows are never edited. A Run that reads a grown row is marked assisted, and growth never
lands in a trusted Task.

### 2. Environments from a specification

An API description, an MCP tool list or a schema is thinner evidence than a trace. Replay fidelity cannot be
measured, so consistency rulings take its place: the sensitivity gate (D195), round trips, effect checks (D215)
against the specification's own examples, plus generated examples per tool. Synthetic Tasks supply the Tasks. The
card says "specification, no recording".

### 3. Forks

`environment.json` carries `parent_env_id`. A fork changes one class of thing (a policy line, a column, the row
population, a refusal rule), keeps the rest, and re-derives the Verifiers the change touches. Forks give cheap
diversity for training and a controlled test of the Examiner: which Tasks flip when one policy line changes.

### 4. A fleet

One command runs every corpus under one budget, publishes each Environment with its card, and reports the
funnel per Environment side by side.

## How each step is measured

- Synthetic Tasks: Tasks per difficulty bucket and pool size per Task, before and after.
- Specification Environments: a candidate model's pass rate there against the same model on a traced
  Environment of the same domain.
- Forks: how many inherited Verifiers survive a one-line policy change.
