# Leibler

A lab that builds frontier-quality models for individual companies. The harness grades a customer's own LLM traces against the frontier model they already use, task by task, post-trains smaller frontier models until they match or beat that bar, deploys them optimized for the customer's traffic, and routes each task to the model that is best on it and cheapest to run. New traces feed the next round, so the models keep improving.

The core constraint: quality is fixed. The customer's frontier model sets "the bar"; nothing ships below it. The wedge is a one-line SDK wrapper that captures traces, followed within 48 hours by a report of which tasks a smaller model can already clear, then post-training for the rest.

## Language

**Category**: the group of Tasks whose Runs change the world through the same set of tools (all cancellations, all address changes). Tasks inside a Category differ by what the user wanted. Reports roll Tasks up to their Category.
_Avoid_: task type, task family, bucket

**Call**: a single LLM request (prompt + params) and its response (completion + usage + cost + latency). The transport primitive; carries no domain meaning on its own.
_Avoid_: trace, span, log line

**Step**: one decision point inside a Run: given everything the agent has seen so far (the prefix), the model emits reasoning and one action (a tool call or a final answer). The unit that is screened; never the unit that decides a verdict.
_Avoid_: turn, iteration, call

**Run**: one complete agent episode, from a user input to a final answer, made of ordered Steps and the observations their actions produced. The unit that is re-executed, verdicted on its end state, and clustered into Tasks.
_Avoid_: trace, session, trajectory, episode

**Environment**: an executable, resettable replica of the world a Task's Runs act on (tools, state, and the user's side of the conversation), built from the customer's traces so a Candidate can re-execute a Run. Central to grading: no Environment, no verdict.
_Avoid_: sandbox, mock, simulator, harness (as the product's frame)

**End state**: the effects of a Run: what the world looks like when it finishes (side effects on the Environment) plus everything the user was told or asked, including the final answer. The thing a verdict is computed on. Reads, the order of actions, and reasoning text are not effects and are not End state.
_Avoid_: output, result, trajectory

**Replica**: a real Run from the customer's traces, re-executed by a Candidate in the Environment. The only thing a routing-plan Verdict is computed on.
_Avoid_: replay, rerun, test case

**Scenario**: a synthesized task grounded in the customer's traces (seed intent x persona x trap) with its own generated Verifier. Used for coverage, hardening, and post-training the Student. Never counts toward the bar. Out of scope until Replicas are trusted.
_Avoid_: synthetic task, generated task, stress test (as a grading term)

**Verifier**: the generated End-state check for one Replica: required and allowed write effects, Hard constraints, and facts the final answer must state. Emitted as code wherever exactly checkable.
_Avoid_: grader, reward function, rubric, test

**Provenance**: where a value written by a Run came from, evidenced by a span of the trace: user-stated (in the opening request), system-derived (read from a tool), user-elicited (the agent asked and the user answered), or agent-chosen (no source). The first two make a Verifier atom required; the last two make it allowed.
_Avoid_: source, origin

**Intent**: a short statement of what the user wanted from a Run, written from the trace, with every clause tied to a span of the trace that evidences it. The spec a Verifier's atoms are derived from. Never graded directly.
_Avoid_: rubric, task description, goal, success criteria (as a free-text grading object)

**Simulated user**: the user's side of the conversation during re-execution. Exact on the facts the recorded user gave (which order, refund method, confirmations), representative of the customer's real users on everything else (wording, tone, patience). An answer to a question the recorded user was never asked is improvised and can only make an allowed atom, never a required one.
_Avoid_: persona, user model, fake user (in documents)

**Assisted**: the status of a re-executed Run in which at least one tool call was answered by a guess (an LLM standing in for a tool with no code and no recording) rather than by the Environment. An assisted Run gets a Verdict that is shown but never counted toward the bar, and never serves as evidence of Replay fidelity.
_Avoid_: simulated run, partial verdict

**Replay fidelity**: how faithfully the Environment reproduces a reference Run when the frontier's recorded actions are replayed in it. Measured in two parts: write effects, which must match exactly after canonicalization, and read observations, whose mismatches are classified cosmetic or semantic; semantic mismatches must be zero. The trust number for the Environment; a Task below the bar for it gets no Verdicts.
_Avoid_: environment accuracy, simulation quality, coverage

**Screen**: the cheap Step-level check (Match, then Appeal) that decides which Runs are worth re-executing. Produces no verdict.
_Avoid_: pre-eval, filter, step eval

**Verdict**: the pass/fail of one re-executed Run, decided only by its End state against the Reference's End state and by hard constraints. Never by the path taken, and never by the reasoning text.
_Avoid_: score, grade, judgment

**Trace**: the raw captured record of a Run (or, for non-agent traffic, of a single Call), as it arrives from the customer. Input data, not a unit of grading.
_Avoid_: log, dump

**Task**: a cluster of similar Runs sharing the same user intent, inside one Category; the unit that is graded, post-trained, and routed.
_Avoid_: request type, prompt category, intent, use case

**Starting state**: what the world looks like when a Task's Runs begin, as the Task's own Runs saw it. Two Tasks can start from the same entity in different states (an order pending in one, delivered in the other); a Task starts from a state, it does not create it.
_Avoid_: S0, initial DB, seed state

**The bar**: the customer's current frontier model's score on the customer's own inputs. Fixed; models climb to it.
_Avoid_: quality target, SLA, benchmark

**Reference**: the frontier model's recorded action at a Step, or its recorded Run once that Run's End state is confirmed as success. What a Candidate is compared against. A recorded Run whose success is unconfirmed is not a Reference.
_Avoid_: ground truth, gold, label

**Candidate**: a cheaper model being graded against the bar on a Task.
_Avoid_: challenger, alternative, small model (as a grading term)

**Match**: a Screen result reached without a judge: the Candidate's action structurally agrees with the Reference (same tool and required arguments, or the same decision to stop and answer).
_Avoid_: exact match, agreement score

**Appeal**: judge review of a Step where the Candidate did not Match. The judge sees the prefix and both actions and may rule only "equivalent" or "reference wins"; it can never rule that the Candidate is better.
_Avoid_: adequacy score, LLM-as-judge (as the product's frame)

**Hard constraint**: an action a Run must never take (a destructive or unauthorized side effect, a policy breach, a fabricated tool result), or must never take without a required prior action (a write without the confirmation the policy demands). Checked on the transcript; one violation fails the Run regardless of End state.
_Avoid_: minefield, guardrail, safety check

**Clears the bar**: a Candidate's computed status on a Task when its graded Runs are statistically non-inferior to the Reference. The report turns it into a suggestion; a person decides whether the Task enters the routing plan.
_Avoid_: good enough, passes, wins

**Harness**: Leibler's end-to-end system. It replays traces through candidate models, grades them against the bar, post-trains smaller frontier models, optimizes and deploys them, and routes traffic. Its first two parts are the Builder and the Runner.

**Kullback**: the open-source name of the Builder and the Runner together, the code that turns traces into an Environment and grades Runs in it. The other half of Kullback-Leibler. Repository and Python package name.

**Builder**: the part of the Harness that turns a customer's traces into an Environment (state, tools, Hard constraints, Simulated user) and the Verifiers for its Tasks. The environment creator.
_Avoid_: generator, synthesizer, environment creator (in documents)

**Runner**: the part of the Harness that re-executes Runs with a Candidate in an Environment and computes each Verdict.
_Avoid_: orchestrator, executor, agent loop (as the product term)
_Avoid_: platform, pipeline, gateway, proxy

**Student (smaller frontier model)**: a smaller model post-trained on the customer's traces until it clears the bar on a given task.
_Avoid_: distilled model, fine-tune, LoRA (as the product's frame)

**Routing plan**: per Task, the model a person chose after reading the report, the recommended traffic allocation, and the projected saving. Written from the person's decisions, not from the suggestions. The customer applies it in their own config.
_Avoid_: optimization plan, model strategy, report

**Deploy**: running the models in production, optimized for the customer's traffic (quantization, batching, caching, custom kernels, hardware matched to load) in Leibler's cloud, the customer's cloud, a VPC, or on-prem.

## Relationships

- A Trace is the captured record of a Run; a Run is an ordered list of Steps; a Step is executed by one Call (or several, if the agent retries).
- Runs cluster into Tasks by user intent.
- A Task is screened at Step level (replay the prefix, compare the next action) and verdicted at Run level (re-execute in the Environment, compare End states).
- A Screen result is a Match, or an Appeal ruling of "equivalent" or "reference wins". Match and "equivalent" both count as a pass of the Screen.
- A Verdict depends only on End state and Hard constraints. Effect-free actions (reads), their order, and the reasoning text never change a Verdict; reasoning is monitored for fabrication only.
- A Run without an Environment gets a Screen result but no Verdict.
- Verdicts come from Replicas only. Scenarios never enter pass rate or the routing plan.
- A recorded Run becomes a Reference only after its End state is confirmed as success (customer outcome signal, agreeing frontier re-rolls, or a human label). Runs the frontier itself failed are reported separately and never set the bar.
- A Verifier is derived from the reference Run plus k re-rolls of the frontier and must pass its validation gates before any Verdict uses it.
- Replay fidelity gates the Environment; Verifier validation gates the Verdict. Both must hold before a Task can clear the bar.
- A Verifier is anchored to the Intent, not to the Reference's End state. The Reference and its frontier re-rolls are samples of what satisfies the Intent; atoms present in every successful sample are required, atoms in some are allowed, write effects in none and unimplied by the Intent are forbidden.
- Hard constraints are compiled from the customer's policy and system prompt, and are checked on the Reference itself before it sets the bar.
- An Intent that cannot be grounded in the trace is rejected and its Run gets no Verdict.
- A Candidate that clears the bar on a Task enters the routing plan; nothing else does.
- Each Task is graded against the bar (the customer's frontier model).
- Tasks a smaller model already clears go straight into the routing plan.
- Tasks it can't clear are post-trained (the student) until they clear the bar.
- Models deploy optimized for the customer's traffic; new traces feed the next round.
- A customer's data trains only that customer's models and is deleted on request.

## Example dialogue

> **Dev:** "How do we know a smaller model is safe for a task?"
> **Domain expert:** "We grade it against the bar: your frontier model's score on your own inputs. If it clears, it ships. If not, we post-train a student on your traces until it does. Nothing ships below the bar."

## Flagged ambiguities

- "grade the reasoning" was the original intuition (2026-08-26). Resolved: reasoning text is never graded; it is monitored only (fabricated observation, plan/action mismatch). See ADR-0004.
- "how close the end state is" vs "pass rate". Resolved: a Verdict is binary; closeness (fraction of End-state checks met) is a diagnostic attached to failed Runs, never a Verdict.

- "trace" was used to mean a single LLM call (old glossary) and a whole multi-step agent episode (2026-08-26). Resolved: **Call** (one request), **Step** (one decision), **Run** (one episode); **Trace** is only the raw captured record. ADR-0001's "Run tree" refers to a Run.
- "routing" once meant both the offline plan and live, in-request routing. Resolved: the customer applies the routing plan in their own config; live routing is not part of the current offer.
- "distilled 8B" / "fine-tune" / "LoRA" were used to describe the student. Resolved: frame it as post-training a smaller frontier model against the bar.
- The bar vs. a benchmark. Resolved: the bar is the customer's own frontier model scored on their own inputs, never a public benchmark.

## Agent setup & conventions

### deepline (GTM CLI) is project-local, not global
`deepline` is installed **project-locally** here (this repo's `node_modules`, pinned in `package.json` as a devDependency). It was deliberately removed from the global npm install so it stays scoped to this project.
- Preferred invocation: `npx deepline ...` (resolves to `./node_modules/.bin/deepline`, works from any subdirectory and in non-interactive shells, offline).
- Bare `deepline` also works in interactive shells (Claude Code/Cursor) via a shell shim added to `~/.zshrc`/`~/.bashrc`.
- `node_modules/` is gitignored.

### Install scope convention
Before installing any skill, MCP server, CLI, or agent extension, **ask the user whether it should be global or project-local**, never assume a default. (Recorded globally in `~/.claude/CLAUDE.md` and Cursor global rules.)
