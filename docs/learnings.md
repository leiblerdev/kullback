# Learnings from the fidelity loop (D184 to D218, 2026-09-07 to 2026-09-09)

This page is the synthesis. The record of each decision, with its counts, sits in decision-log.md; the standing instructions sit in founder-words.md; the open items sit in todo.md. What follows is what the loop taught about building a harness that turns recorded traces into trusted Tasks, written so the next person (or the next model) does not relearn it by spending a build.

## 1. How to read a failure before fixing it

Check what code the build ran on before attributing a failure to an open class. Three builds read on 2026-09-09 showed twelve failure classes; six were already closed by decisions the builds never ran on (D198 to D206), because workdirs are frozen at a runner and gates hash and only the user re-freezes. A reader that does not check runner_version.json first writes a brief for a fix that exists. The reading brief now asks for the frozen hash and marks every pattern a shipped decision claims to close.

Readers overstate, builders measure. The retail reader counted 64 rows in disagreement between the round snapshot and the live status file; the builder joining the same files counted 38. The nested-scope reader named twelve Tasks as a homing fault; the builder found zero instances and four low-fidelity bodies underneath. A brief states the reader's count as a premise to be checked, and every decision-log entry has a paragraph that says where the measurement disagreed with the premise. A fix whose premise fails still ships as the more general rule with counters (D207), but the real class must then be found (D211).

Absolutes come from live builds only. A copy of a workdir carries a stale version history (verifiers ahead of examiner history), so trusted on a copy reads low (65 against 119). Measure deltas on copies, absolutes on the build, and say which.

One snapshot per round. Task-level rulings that keep moving after the round closes (loosening, re-derives, cache recomputes) make a regression indistinguishable from bookkeeping drift. D218 writes one Task-row table at round close, keeps per-tool compile rulings in their own file, stamps live rows with their round, and prints a drift count. Every search that can end without a ruling writes a reason; an empty record is a bug.

## 2. What makes a Task trusted, and what silently stops it

The funnel is fidelity, Reference confirmed, Verifier derived, the D79 suite, false rejection over a held-out pool, trusted. Each stage has been found to lose Tasks for a reason unrelated to the Task's own quality:

Fidelity. Replay compares each call's own result. It cannot see a write that changed a row the call did not name (a balance debited on another table, a history entry appended, a total recomputed from two catalogue rows) unless a later call rereads that row, and then it blames the read (D215 adds an effect check on the write and effect evidence for the writer). A cosmetic verdict reached through an exempt column or an unresolved semantic column confirmed Traces whose answers named different states (D217: a value present on one side only is never cosmetic; the judge is asked only when both sides carry the same domain tokens). A replay cursor that assumed alternating roles stalled on every telecom Trace (D204). A column only ever written was seeded to a corpus-wide constant (D202 runs the write's own body backwards to find the pre-state).

Reference. A fail-only judge asked to choose among surviving End states abstains every time (13 of 13, three separate measurements). Stop asking it to choose: derive a Verifier per survivor and let the suite and the pool decide (D198). A Verifier whose Reference was withdrawn stays on disk and is scored as strictness (34 of 206 retail, 23 of 85 airline) unless derived artefacts are retired with their source (D208).

Verifier and suite. A shape atom checked against the row the call writes rejects its own Reference when the value was read off another row (D206, retail oracle failures 15 to 1). A Task with one recorded path can never pass check 5, so every single-path Task on all three corpora was untrusted (54, 38, 9); D199 writes a second path from the Reference Run by reordering, repeating and dropping independent calls, and the difficulty record (D209) reports trusted per path count so this block is visible.

Pool. The held-out pool admitted Runs by termination reason alone; 146 of 569 retail and 198 of 250 airline pool Runs never finished the Task (D210 types the end kinds; whether the pool should admit goal_satisfied only is the next reading).

Repairs. A shared tool recompile that fixed nothing landed anyway and sank 24 airline Tasks between two rounds (28 trusted to 5). A repair lands only if it fixed what it targeted and cost nothing elsewhere (D201), and a kept body is replaced only by one that beats it (D184).

## 3. Design rules the loop kept confirming

Every rule keys on a class the harness mines (provenance class, atom kind, column class, call position, Run membership), never on a corpus, tool, table or column name. A fix that only moves the corpus it was read from is suspect and says so.

Anything derived from the harness's own proposals is not an identity. Task grouping keyed on the mined column view regrouped the telecom corpus whenever readers or a cache format changed (73 frozen Tasks stranded, 219 added). Keyed on the recorded bytes per tool and row (D216) the grouping is byte identical across those changes. The same principle produced D200 (the frozen list is authoritative) and D212 (probe membership keyed on Run identity, not Task order).

Strictness must hold at every layer independently. The telecom reboot tool showed the chain: a column the world never pinned (seeding gap), a body reading it (unwitnessed read), a verdict folding the wrong answer into cosmetic (lenient comparer). Fixing one layer without the others hides the fault at the next.

A stalled body needs a different lesson, not another round. Eleven kept bodies sat under 0.5 fidelity for 12 to 31 rounds because the lesson showed one leaf of one call. D211 gives the full diff, the relation catalogue across failing calls, the branches nothing witnessed, and a stall limit that forces a rewrite.

Evidence shown to a writer is filtered by held-out membership, never by value. Greptile's held-out concern on D211 was right in kind; the mechanism is that held-out Runs contribute nothing to what a writer sees, which holds for effect evidence too (D215).

Counters over claims. Every decision ships the counters a reader needs to see it work on the next build (auto_loosen_*, second_path_synthesised, cosmetic_by_judge, status_drift, tasks_frozen_only). A mechanism whose counter reads zero live is reported as shipped and unexercised, not as working.

## 4. What outside work contributed

Repo2RLEnv (huggingface): a sensitivity gate that a body must answer differently when the state moves (D195), and Intents written from symptoms only (D196). The harness design survey (pi.dev and others): transactional repairs (D201). TaskPilot (FrogNano): an objective difficulty record and buckets so the Task distribution can be shifted as training proceeds (D209; the knob's second half is in todo.md). tau-tau-Bench: typed facts the simulated user holds and an explicit end protocol with end kinds (D210). Mastra: keyed sampling so membership never depends on iteration order (D212). The agent-as-user question: an agent user grounded in the curated context as a fallback first (D214), primary where its fidelity beats the rules, then its own loop and gate.

## 5. Process learnings

Builds never run from a temporary directory (a scratchpad wipe killed a build mid round on 2026-09-08); worktrees, logs and briefs live under the repo's own ignored directory. Long test suites stall agents; run the suite in chunks covering the whole tree, in the background, and poll. One merge agent lands branches serially through PRs; Greptile findings are accepted, declined with a checked reason, or answered, never ignored. Customer data reaches reports easily (an address, passenger names, id-shaped tokens were scrubbed twice); a report carries task ids, tool names and column names only. A model change (the contributor model from 2026-09-09) makes the next round a new baseline, not a harness delta. After a re-freeze the first reading is of the counters the frozen decisions promised.

## 6. What is still open

Effect checks make fidelity stricter before they make it higher; the first contributor round will show the drop and the recompiles that recover it. The unresolved semantic route (no judge at replay) forgives 387 telecom calls and needs its own route and count. The pool rule on end kinds, the difficulty knob's second half, the held-out gate mirror, the schema check at the call boundary and the quality pass (complexity, CRAP, mutants, as CI checks) are listed in todo.md.
