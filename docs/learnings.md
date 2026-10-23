# Learnings

The rules the build loop taught that still apply, one line each. Read them before you diagnose a build or design
a fix, so you do not relearn one by spending a build. How each rule was found, with its counts, is in the
[decisions](decisions.md); terms are in the [glossary](glossary.md).

## Reading a failure

- Check the code hash a build ran on before you blame an open class; the fix may already exist
  ([architecture](architecture.md#one-code-hash)).
- A reader's count is a premise to check, not a result. Builders measure; readers overstate (D211).
- Take absolutes from live builds. A copy of a workdir carries stale history; measure only deltas on it.
- Read one snapshot, written once. Rulings that keep moving make a regression look like bookkeeping drift (D218).
- A cache key must cover all the code a stage reaches, or every number read off the cache is about old code.
- Measure a proposed failure class before you build a mechanism for it. Two mechanisms shipped on classes with
  zero instances.
- Every search that can end without a ruling writes a reason. An empty record is a bug.

## Trust and what silently stops it

- Replay compares each call's own result, so check the effect of a write on rows the call did not name (D215).
- A value present on one side only is never cosmetic (D217).
- Do not ask a fail-only judge to choose among end states; it abstains every time (D198).
- Retire derived artefacts with their source, or a withdrawn Reference keeps scoring (D208).
- A writer may probe the world, never the held-out split it is graded on.
- A body whose answers do not move when its inputs move is a constant; the sensitivity gate catches it (D195).
- A repair lands only if it fixed its target and cost nothing elsewhere (D201); a kept body is replaced only by
  one that beats it (D184).
- A scorer that fails every Run on an atom no code can evaluate zeroes trust everywhere.
- A thinner sighting of a row is not a new version of it.
- Every requestor's call through a listed tool is a sighting of the world, not only the assistant's.
- A corpus that builds on an empty world with no gate failing is worse than one that refuses to build.

## Design rules

- Every rule keys on a class the harness mines, never on a corpus, tool, table or column name.
- Anything derived from the harness's own proposals is not an identity; key on recorded bytes (D216).
- Strictness must hold at every layer on its own: seeding, reading and comparing.
- A stalled body needs a different lesson, not another round (D211).
- Filter what a writer sees by held-out membership, never by value (D215).
- Ship counters with every decision. A mechanism whose counter reads zero live is shipped and unexercised.
- The writer of a check and its judge never share a hand (D320).
- Feedback must carry the rows that differ; counts and gate words give a Builder nothing to act on.
- A repair verb that calls a fresh one-shot writer is not an edit; it never accumulates what failed.
- A result opens with the ruling of the target it names; a refused call hands back the corrected call once.
- Compact automatically at a share of the model's real window; model-first forgetting tools measured badly.

## Process

- Never run a build from a temporary directory; keep worktrees, logs and briefs under the repo.
- Run long test suites in chunks covering the whole tree.
- A report carries Task ids, tool names and column names only, never customer data.
- A model change makes the next round a new baseline, not a harness delta.
- Read counts on the way out of whatever ended the work; earn an exit against the number the loop is for.
- An agent repairs only what it was shown, and a number no script can print from the records is not a number.
- A view of a build is a subscriber: it never fails the build, and the decisive numbers come from files on disk.

## What outside work contributed

| Source | Taken as |
|---|---|
| Repo2RLEnv | Sensitivity gate (D195), Intents written from symptoms only (D196) |
| Harness design survey (pi.dev and others) | Transactional repairs (D201) |
| TaskPilot | Difficulty record and buckets (D209) |
| tau-tau-Bench | Typed user facts and an end protocol (D210) |
| Mastra | Keyed sampling (D212) |

## From the first Harbor benchmark run (2026-10-19)

What the attribution of the failed trials of the first Harbor run taught Spec writing and Task generation. Each
line names the decision that enforces it, or says none does yet.

- The instruction carries everything the Verifier gates on. A fact the user said later, on request or after a
  refusal, is still the user's fact: render it or drop the item, since a canned user cannot supply it (D422). A
  born Task's ask is complete at birth; an end fact is never the user's words (D414).
- Every Verifier cell path is readable on the start row, a list stepped by index and never by name. A plausible
  Verifier is not a readable one (D423; D431 resolves a one-member list from the start row, never the Reference).
- Weighted items are not free when a pass needs reward 1. Gate only what a right Run must do and weigh only what
  shapes training; the Harbor pass is gates only (D433), and a writer still never weighs an ask the user never makes.
- Learned effects are present and fresh when a Spec is written, or the world's own side effects fail sanity as
  collateral. An effects file with empty write lists counts as absent (D417, D432).
- The world is self-consistent: a row that lists ids holds those rows, or the agent reads "not found" off the
  user's own record. A world built on a slice binds every row the end reads (D425 counts, D429 prunes at export).
- The canned user says yes only to what it asked; a blanket yes reads as consent to the agent's own offer (D430).
- A gated value only the Reference supplied is a wrong gate. Where the ask and the policy leave a choice open,
  write one_of or no gate (D438, in build; the literal refusal D434 was held back); a born Task's scenario fixes
  the value.
- Duplicate items double an item's weight and give a judge item a second chance to fail; overlapping new-row
  specs starve each other (D428).
- A said number compiles to bare digits for every Task, so a formatted amount meets its digits (D427).
- A judge reply with one result is read whatever its key, and an unscored Run is an outage to count, never a
  silent fail (D424).
- Every benchmark ships with an empty-trial control and an offline code rescore. A rescore that reproduces the
  shipped reward says the scorer is faithful, not that the Verifier is right; a Verifier audit needs a blind
  second rater before its wrong rate is acted on (no decision row yet).
- Attribute every failed trial to the harness or the model in one read-only pass after each benchmark, before
  touching the agent or the prompts (no decision row yet).
