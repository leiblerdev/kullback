# AgentTrove (open-thoughts/AgentTrove)

## Link and owner

https://huggingface.co/datasets/open-thoughts/AgentTrove. Already the subject of substantial internal analysis: docs/papers.md (2026-09-18 row), docs/decision-log.md (around lines 1753 to 1807 and 1967), docs/learnings.md (section on "a corpus that builds on an empty world"), docs/founder-words.md line 806. This file adds fresh confirmation, not a first look.

Note: a separate, unrelated `gijl/AgentTrove` also exists on Hugging Face; do not confuse the two.

## What a row is

One row is one agent trajectory in ShareGPT-like format. The served schema's columns are `conversations` (the card's own documentation calls this field `messages`, which is a documented mismatch), `task_binary`, `path`, and `result` (confirmed live via the datasets-server API, 2026-09-24: `license:apache-2.0`, `downloads: 8243`, `lastModified: 2026-05-07`).

## Provenance

Fully model-made: "1,696,847 agent trajectories from 219 source datasets, terminus-2 harness format, generated with Harbor" (docs/papers.md, 2026-09-18 row). Roles are `user` and `assistant` only; there is no real human or real system on the other end of these conversations, only a generated one.

## Counts and size

1,696,847 rows total (docs/papers.md). Sampled 35 rows across the 1.7M via the datasets-server API (docs/decision-log.md, prior session): every row with a trajectory has `agentId` terminus-2; 24 of 27 sampled rows carry JSON keystroke commands inside the assistant's text; 3 of 30 rows carry no trajectory at all, only a `task_binary` and a `path` (a task definition, not a recording).

## Schema

`conversations` (called `messages` on the card), `task_binary`, `path`, `result`. `result` was filled on only 1 of 30 sampled rows; no reward column exists in the served schema (docs/decision-log.md, prior session, reconfirmed in spirit by this pass's re-check of the licence and row count).

## One excerpt

Not re-pulled this pass; see docs/decision-log.md for the prior session's direct sampling notes, which already quote the shape ("24 of 27 carry JSON keystroke commands inside the assistant text").

## Tool-call shape

Commands are not a separate structured tool-call field; they are JSON keystroke commands embedded inside the assistant's text turns, which the terminus-2 harness treats as one shell tool. Results are largely absent from the served schema (1 of 30 sampled rows had `result` filled).

## Environment and reference verdict

Partly reusable: the trajectories were generated with the open-source Harbor framework, which is itself installable and runnable, so the mechanism that produced these traces is not a black box. But AgentTrove is a concatenation of 219 separately-sourced datasets, so there is no single reference environment behind the whole corpus. Each source dataset would need its own starting-container and tool-body reconstruction. Per docs/decision-log.md's later ruling (~line 1967), the intake floor already fails every sampled AgentTrove file (no starting container on 3 of 30, no outcome on all but 1 of 30), yet the later pipeline stages (mine, readers, cluster, laws) succeed anyway on an effectively empty world, silently, with "one tool, zero tables, zero columns, 64 Tasks with no intent, and no gate stops the build" (docs/learnings.md).

## Licence (quoted)

`apache-2.0` (confirmed live via `https://huggingface.co/api/datasets/open-thoughts/AgentTrove`, 2026-09-24: `cardData.license = "apache-2.0"`, tag `license:apache-2.0`).

## How to download

```
hf download open-thoughts/AgentTrove --repo-type dataset
```
Or stream via the datasets-server rows API for sampling without a full download.

## Fit for Kullback

Large, cleanly-licensed, and already the founder's named direction ("we need to use this to build the environments", docs/decision-log.md, quoting the founder). The blocker is not access or licence, it is that our own intake pipeline currently accepts this corpus's near-empty world instead of refusing it, which produces a fidelity number that "means" nothing (docs/learnings.md). The work here is entirely on our side: an intake step that refuses an empty-world build, and a plan for re-deriving expected results by replaying recorded shell commands in a real container, since the served `result` field is almost always empty.

## Open questions

- Whether the founder has ruled on the open question at docs/decision-log.md ~line 1967: does the terminal-trace corpus become a fourth Environment, and is the coding domain in scope at all.
- Which of the 219 source datasets carry a real starting container and a real outcome, versus which are task-definition-only rows; a per-source breakdown was not done this pass.
- Whether re-deriving results by replaying shell commands in a real Harbor container is worth the cost ahead of the speed pass, per docs/decision-log.md's own listed tradeoffs (container time per build, honest-but-lower fidelity when hidden tests differ from what the recording showed).
