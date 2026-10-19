# Deferred work

The harness work that is known and not done, so nobody plans it twice. The company's list of 2026-09-10 is
in [archive/company-todo.md](archive/company-todo.md); what past builds taught is in [learnings.md](learnings.md).

## Next

- Read the failures of the latest builds (Examiner findings, refused tool calls, repeated failing calls, turns
  per repair, stop reasons) and propose tools, skills and prompts that raise the Verifier and trusted counts.
- Synthetic Tasks: grounded scenarios, augmented seeds and traps, then generated Tasks hardened the way
  TauForge augments, validates and hardens. Each needs concise instructions and a high quality Verifier.
  The scenario layer is in [scaling.md](scaling.md).
- Two patches wait in [frozen-patches/](frozen-patches/): `confinement-holes` (needs a rebase: its module-shape
  rule refuses today's generated skeleton) and `speed-3` (written against an older `budget.py`).
- Re-roll batches: one Builder run call plays at most `RUNS_PER_CALL = 20` fresh Runs. Measure whether a larger
  cap or parallel Runs helps.
- A cheaper runner model. Record the Run kind in the ledger first, since second-path Runs dominate runner spend.
- Move the airline body's hand-rolled id rule back to the context's `new_id`.
- Check the rebuilt world against evidence outside the recording; replay only confirms recorded calls.

## Before the first training run

Compare every built Task with the benchmark's test tasks: which test task each came from, and how close each
synthetic Task sits to one. The held-out anchor (`builder/world_tools.py draw_anchor`) holds out recordings
inside a Task, not Tasks, so today every benchmark task the traces cover is built and would be trained on.
Decide how to split by task before ingest, and write the held-out task ids into the package manifest.

## Later

- Kitaru (ZenML, https://www.zenml.io/product/kitaru): build evals on the built Environment. Kitaru records,
  replays and evaluates agents against frozen production sessions.
- The difficulty knob: `difficulty.py` measures the buckets; the second half is a rule that shifts Task difficulty
  as training proceeds.
- A real sandbox for model-written code. Today a tool body runs in a subprocess with imports blocked
  (`builder/sandbox.py`, `gates/confinement.py`), which is not a security boundary.
- Builder memory: deferred by the founder ("memory is totally different now, we will talk about it later").
- Feedback mining from traces: report in aggregate what customers ask for, hand over, are refused, give up on and
  want that the tools cannot do. Personal data stripped. Not before synthetic Tasks.

## Route checks (deferred)

A Verifier scores the end state and what the user was told, never the route, so a right result reached by a
wrong method passes. The founder deferred route checks: "how the tools should run is the route check, maybe
don't do that, add it in todo". Parked:

- the tools that should run for a Task, as a Hard atom over `called_before`;
- a no-detour check that no write outside the Task's fix touched a row the Task names;
- loosening an ending action from required to allowed when every re-roll fails only on it.

PAE (arXiv 2603.03116) finds 27 to 78 percent of reported tau-bench successes broke a required procedure.
Comparing Environments is fair only once every Environment scores the same thing.
