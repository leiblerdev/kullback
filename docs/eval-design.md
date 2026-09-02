# Evaluation design: from customer traces to routing-plan Verdicts

Status: working design, 2026-08-27. Terms are used as the decision log defines them; decisions with trade-offs are in `/docs/adr/`. Research behind each choice is in `../research/` (report numbers cited as R00 to R29).

## Principle

Simple by default, complexity where it pays. Build for the hardest Runs (long multi-call, multi-turn, off-path) so the common single-call Runs are the easy case of the same machinery, not a separate path.

Generalizability over overfitting (D51): every stage is tested on Runs it did not see; a perfect score on the seed Runs with a collapse on held-out Runs is reported as overfitting. Simplicity first. Every stage must be explainable in one sentence, and the first version of each stage is the simplest one that keeps the trust gates honest. Elaborations (trained simulators, Scenario generation, curricula) come only after a Replica has cleared Gate A on a real customer's traces.

## What we are doing

Ingest a customer's agent traces, rebuild the customer's world as an executable Environment, re-execute the real Runs (Replicas) with cheaper Candidates, and verdict each Candidate on End state against a Verifier anchored to the user's Intent. Report, per Task, which Candidates clear the bar and what that saves.

## What we are not doing (ADR-0003, ADR-0004, ADR-0005)

- Not grading reasoning text. Monitored for fabrication only (CoT faithfulness 20 to 40%, "Gaming the Judge" +90% false positives; R04).
- Not grading the path. A different tool sequence that reaches a satisfying End state passes. Path differences are reason codes.
- Not aggregating Step-level matches into a Run score. Static replay is unsound past the first divergent action (R06, the Replay Gap).
- Not letting a judge overrule an End-state check, and not letting a judge award a pass. A judge ruling can only remove a Run from the bar or widen a Verifier for every Candidate.
- Not counting synthesized Scenarios toward the bar. Replicas only (ADR-0005). Scenarios come later for hardening and post-training.
- Not issuing a Verdict without an Environment. No Environment, no Verdict.

## Pipeline

1. **Ingest** traces (any of OTel GenAI, OpenInference, Langfuse, LangSmith, OpenAI Agents SDK, Claude Code JSONL, MCP logs; R16 section 1). Normalize to ATIF-shaped steps (R14) plus three fields no format carries: read/write flag per call, pre/post state keyed on entity ids, determinism flag per tool. Handle the ten ingestion anomalies in R16 section 6 (orphan tool_use, retry duplicates, out-of-order spans, truncation, five error encodings, masking, sub-agent stitching, model drift, cache-token accounting, stateful handles).
2. **Filter** Runs (R18). Two gates and tags, nothing else. Gate 1, structural: a tool call without a response, unparseable arguments, or a Run ending mid-turn is tagged `unreplayable`, kept for diagnosis, never a fixture. Gate 2, harness failure: agent never started, tool server never connected, or every call errored is tagged `harness_failure` and only counted. Everything else is kept and tagged: `frontier_failed`, `retry_of` (SDK retries collapse to the last attempt; a model's own retry after a bad call stays in the Run), `duplicate_of` with a weight, `compacted` (replay from pre-compaction records, never from the summary), `truncated_observation`, `masked`, `unprofiled_tool` (held until the tool is profiled), `no_tool` (chit-chat stays; it is a Task with an answer-only Verifier). The bar uses Runs with no blocking tag; the report shows each tag's share of traffic. Tool profile (read/write, open or closed world, deterministic) is inferred from the tool definitions plus observed behavior and confirmed by the customer in a one-page table.
3. **Cluster** Runs into Tasks. Hard partition by write-tool signature first (Runs whose confirmed References write through different tool sets are never one Task; answer-only Runs skip this). Within a partition, cluster by grounded Intent (Clio-style: embed, cluster, frontier names the cluster in one line, cross-family check against sampled Intents); the name is the Task label. Merge two clusters only if one Verifier template (same atom types, same Hard constraints) fits every Run in both; split a cluster whose Runs need different atom types (D21 made mechanical). A Task is verdicted when its paired non-inferiority CI half-width is below the margin; the prior expectation is roughly 150 Runs for a 5-point margin and 30 as the floor for a provisional CI, both to be replaced by observed per-Task variance. Below the floor: "insufficient traffic", counted, not verdicted. Each Task carries a priority from traffic share, cost per Run and frontier failure rate; the report leads with the prominent Tasks.
4. **Build the Environment** (R09, R11, R12): schema reverse-engineered from tool outputs and write effects; tools as code over explicit state (SQLite), one file per tool; policy and system-prompt rules compiled into Hard constraints (SQL triggers or predicates, one positive and one negative test each); per-tool strategy from the R11 table (pure: cassette; closed-world: snapshot plus execution; open-world read: cassette then trained simulator; open-world idempotent write: stateful mock; open-world destructive write: simulate, never execute; time and random: frozen). Unprofiled tools default to destructive open-world. A call to a nonexistent tool or with invalid arguments is answered in the customer's own error encoding as seen in their traces; if their traces contain no errors, the encoding is a guess and is listed as an assumption on that Environment (D45). S0 by inverse replay: reconstruct from recorded reads, apply recorded writes.
5. **Confirm References.** A recorded Run becomes a Reference only after its success is confirmed: Hard constraints pass on the Reference itself, customer outcome signals, k=4 to 5 frontier re-rolls reach the same End state, an LLM judge that marks pass or fail and abstains to a human when unsure (D57; its agreement with humans published through D48 check 2), human label on a sample. Frontier-failed Runs go to a separate "frontier fails here too" bucket.
6. **Write the Intent** for each Reference: frontier model writes it from the trace; every clause must cite a trace span; a cross-family model checks grounding; ungrounded Intents are rejected. Shown to the customer as the Task name, trace spans one click away, so a domain expert can correct it (D47); a correction enters the Verifier only if grounded to a span, otherwise it is stored as a customer policy line and compiled to a Hard constraint; correction rate published per customer.
7. **Derive the Verifier** (R07, R12): diff S0 to each successful re-roll's End state over write-set tables (column classes: exempt ids and timestamps, hard exact, semantic fuzzy). Type atoms by agreement: in every successful re-roll and implied by Intent = required; in some = allowed; in none and unimplied = forbidden. Each written value carries a Provenance grounded to a trace span (D42): user-stated and system-derived values are required; user-elicited and agent-chosen values are allowed, so a different answer from the simulated user or a different free-text note never flips a Verdict. Required atoms whose value varies get a predicate proposed from the Intent (must pass the gates). Messages to the user are atoms under the same agreement rule (a question asked in every successful re-run is required; D43), and a write whose value came from a user reply must hold the reply given in the Candidate's own Run. Add communicate-info checks, Hard constraints (including "never write X without prior user confirmation", checked on the transcript), and entity-count checks for spurious side effects. Emit as code.
8. **Validate the Verifier** before it enters the pool: oracle (Reference) passes; null agent fails; alternative-path re-rolls pass; mutation of any hard atom flips the verdict; leakage grep clean; cross-family soft check; weak-model rollout. Then the setup review (D48 check 1): my reviewer, then the customer's domain expert, reviews the Task's atoms, Hard constraints and chosen References before any Verdict is issued on the Task; this covers the prominent Tasks by D36 priority, and the rest are marked "setup not reviewed". After Verdicts: blind human grading of a sample (10% prior), agreement rate published per Task, low agreement pulls the Task back to "under review" (D48 check 2). The reviewer decides before seeing the Verdict.
9. **Re-execute** each Replica with each Candidate, K=4 seeds. Same S0, same Simulated user: exact on the recorded user's facts, representative on everything else (D44); recorded turns replayed while they still apply, then simulated with the recorded facts as known_info. Fact consistency and style representativeness reported per Task.
10. **Verdict** each Candidate Run: binary, End state plus Hard constraints. Closeness is a diagnostic on failed Runs only.
11. **Dispute path.** A Candidate End state outside required and allowed atoms goes to a tool-equipped judge (reads replica state, re-runs the Verifier, replays writes on a fresh copy, reads policy). Outcomes: Verifier too narrow (proposes a widened atom, re-gated, applies to all Candidates), fail stands, or cannot tell (Run leaves the bar). Human audits 10% of rulings. Queue capped at about 5% of Runs per Task; above that the Task is reported as Verifier immature.
12. **Statistics** (R06): pre-registered non-inferiority margin 3 to 5 points, paired per-Run differences, clustered SEs, one-sided 95% CI, pass^k (k=4) as a conditional final gate, about 150 to 250 Runs per Task for a 5-point margin, 20% holdout.
13. **Report.**

## Sources of Tasks

1. **Replicas** from the customer's traces. The only source that counts toward the bar (ADR-0005).
2. **Scenarios from traces** (after Gate A): variations of a Replica (same Intent, different known_info or order state), one task per policy rule ("non-pending order cancel must be refused", from `policy.md` the way tau2 does it), one task per defined-but-never-called tool, one task per recorded tool error path. Seed intent x persona x trap.
3. **Scenarios from a snapshot** (ADR-0006 end rung): sample a real row and generate the goal from it (WorkArena and AppWorld do exactly this), cover every write tool and every precondition branch, generate on entities the traces never touched.
4. **Customer-supplied cases**: their own test cases, labeled Runs, escalations and complaints; the frontier-failed bucket is the highest-priority seed for these.

Sources 2 to 4 are for coverage, hardening and post-training; they are reported separately and never count toward the bar. Representativeness of the observed traces is a Hard constraint on every generated Task and every synthetic row (D41): a generated artifact that cannot be tied to observed data is rejected, the same rule as an ungrounded Intent.

## Trust gates (all pass/fail, no blends)

- Gate A, Environment, in two parts (D39). Write effects: replay fidelity 100% on recorded write Steps after a canonicalization layer (generated ids, timestamps, float formatting, key order, whitespace); the canonicalizer is part of the Environment and has its own tests. Read observations: every mismatch between the rebuilt answer and the recorded one is classified cosmetic (computed-at-read fields, unobserved fields, ordering, truncation, formatting, outside-world reads served from cassette) or semantic (a value the agent would act on differs); semantic mismatches must be zero; the cosmetic rate is reported per Task with a floor set from the first customer's data; unknown mismatch types default to semantic until a human reclassifies. A read fidelity of exactly 100% strict is flagged, not celebrated: it means the Environment is replaying recordings rather than computing a world. Below the gate: Task reported "Environment unverified", no Verdicts.
- Gate B, off-path: a Candidate Run in which any tool call was served by the LLM simulator gets no counted Verdict (reported as "assisted"). Strict until a first customer's tool mix is seen. A Task that cannot be replayed at all is reported "not gradeable" with the blocking tool or gap and the ladder rung that unblocks it; its assisted Verdicts and assisted share per tool are shown underneath, never counted, and never used as Gate A evidence (D49). Step agreement from the Screen is internal only.
- Gate C, Verifier validity: the gates in step 8, on a confirmed Reference.
- Gate D, frontier sanity: the frontier re-run in the Environment must score within the margin of its recorded success; otherwise the Task is held.

## Where an LLM judge is allowed

The Verdict is computed by code. An LLM judge appears in exactly three places, always as a rubric whose every item is grounded to a trace span (same rule as the Intent), and always reported with its audit agreement rate:

1. The `semantic` column class of the Verifier (free-text fields such as notes or email bodies): "says the same thing as the Reference value". Exact and hard columns never go to a judge.
2. Communicate facts that cannot be matched literally after normalization (a paraphrased explanation rather than a number or a name).
3. Reason codes on failed Runs and the dispute path (the tool-equipped judge of step 11), which can remove a Run or widen a Verifier but never award a pass.

tau2 for comparison: retail uses `NL_ASSERTION` in `reward_basis` on 112 of 114 tasks, airline uses `COMMUNICATE` on all 50, telecom uses `ENV_ASSERTION` on all 2,285 (32 also `ACTION`). So an LLM judge is in the reward for tau2's largest conversational domain; I keep it out of the Verdict and confine it to the three places above.

## Holes in a traces-only Environment, and what fills each

A traces-only Environment (Q10 option 1) knows only what the traces show. The holes, in order of how often Candidates hit them, and the fill for each:

| Hole | What the Candidate sees | Fill, in order of cost |
|---|---|---|
| Unseen entity (off-path read of a record no Run ever touched) | "not found" | (a) build S0 from the customer's whole trace corpus, not the single Run: every entity any Run read is a row (per-customer, allowed under ADR-0002); (b) synthetic rows generated by observing the real rows in the traces (same fields, value ranges, relationships; representativeness is a Hard constraint, D41), tagged synthetic, any Run touching one reported "assisted"; (c) DB snapshot, the end rung of ADR-0006 |
| Unseen field on a seen entity (frontier read `status`, never `items`) | null | (a) corpus-wide field union; (b) tool definitions and DDL give the type and default; (c) snapshot |
| Unseen tool or branch (defined, never called; or `cancel` only ever called on pending orders, so the non-pending error path was never observed) | no contract, or wrong error behavior | (a) tool definitions from the traces' `tools` parameter or MCP `tools/list` give the contract; (b) policy text gives the rules (tau2's `cancel_pending_order` rules are all in `policy.md`); (c) recorded tool errors (3.8% of calls in Claude Code traces) are the only observed edge logic, keep every one; (d) customer confirms the one-page tool profile; until then `unprofiled_tool`, calls to it are "assisted" |
| Write with no read-back (send email, post ticket, returns "ok") | no End state to verdict | model the sink as a table (AgentDojo's `Inbox.emails`, WorkArena's incident row); the Verifier checks the row, not the "ok" |
| State that other Runs changed between this Run's reads (cross-Run interference) | wrong S0 | order Runs by time; S0 per Run is the corpus state at that Run's start |
| Nondeterminism (ids, clocks, external processors) | replay drift | exempt columns, frozen clock, canonicalizer (Gate A) |
| User turns after the Candidate diverges | no recorded turn to replay | Simulated user exact on the recorded facts, representative on style (D44); conditioning on the customer's real user turns is part of the Environment build, not a later add-on; fact consistency and style representativeness published per Task |

The number that says which fills are needed is Gate B's "assisted" share per tool. Build traces-only, measure it on the first customer, ask for schema where the share is high.

## Customer-facing metrics (minimal set)

1. Run pass rate per Task with paired non-inferiority CI against the frontier.
2. Hard-constraint violations (any > 0 blocks).
3. Tool selection precision, recall, F1 against the Reference's necessary-action set (subset semantics), and hallucinated-tool calls per Run (nonexistent tool or invalid arguments; never a fail by default, promotable to a Hard constraint per customer; D45).
4. Cost per Run, then net savings at equal success (failures charged full cost plus penalty).
5. pass^k (k=4), conditional final gate for Tasks that already pass.
6. Environment fidelity: replay fidelity and share of Runs that were fully code- or cassette-served.
7. Per failed Run: the failing atom, computed by code, and whether the Candidate took the same path as the Reference (D46). "Transferred or gave up without acting" is its own failing-atom label, so inaction is never read as a pass (R22 item 6, 2607.02577). Judge-decided atoms carry their audit agreement rate. Free-text explanations are opt-in and labeled as opinion.

Internal only: argument-level error rates as reason codes, step accuracy, judge kappa, Verifier validity rates, reasoning monitors as flags.

## What the customer supplies, in order of value (ADR-0006: a trust ladder, asked one rung at a time, each rung justified by Gate B's assisted share)

1. Traces with tool definitions as sent, untruncated tool results, and session grouping (R16 section 6 minimal fields).
2. API docs or a staging endpoint for open-world tools (unblocks Gate B).
3. Policy documents and system prompts (compiled into Hard constraints).
4. 20 to 50 Runs labeled success or failure by their team (calibrates the audit).
5. Corrections to Intent labels in the report.
6. A database snapshot or staging copy, inside their deployment boundary: the end goal, for a complete Environment usable for evaluation and training.

## First build (D50)

Inputs: tau2 recorded trajectories (retail first, then airline and telecom) and real-world tool-using production traces (support, ticketing, CRM, banking, travel; source chosen from R23; not coding traces, D52). Proof: (1) Gate A from traces alone on both inputs; (2) on tau2, per-Run agreement between the computed Verdict and tau2's reward on the official result files, and aggregate pass rate within the margin of the official number; (3) on real traces, agreement with a human-labeled sample of 20 to 50 Runs. Harness: Tau-style layering, brain / environment / face, provider-neutral core, Environment in the tool-call hooks, JSONL per Run (D53). Build shape: TauForge stages, build Environment first, augment seeds and generate Tasks after Gate A, harden later (D54). Built as the smallest slice that validates the approach, then widened. Builder stop condition (D62): customer Verdict agreement; until then an Environment quality scorecard against tau2's real domains (tool fidelity on held-out calls, Task coverage (D96: covered Tasks over the Task list fixed at build start, plain and Run-weighted; policy coverage as its own report line), Simulated user fact consistency, per-Run Verdict agreement, tau3-taxonomy defects per Task). Smallest slice (D55): one tau2 retail Task, one cheaper model, rebuilt world vs real tau2, Verdict compared per Run; tau2's Environment quality is the bar before any improvement is attempted. Emitted files (D56): tau2's shape first (`data_model.py`, `tools.py`, `db.json`, `policy.md`, `tasks.json`) so tau2's harness loads the rebuild; Verifier detail that does not fit (Provenance, allowed atoms, questions asked, spans) in a sidecar next to `tasks.json`; my own shape molded from it once the customer's real traces arrive.

## Open questions (grill queue)

- Is the Intent shown to the customer or kept internal? (Recommendation: shown.)
- Task clustering: by Intent (Clio-style) with step-role sub-clusters; per-Task sample size.
- Evidence tiers for Tasks that cannot be re-executed.
- Architecture and the small first build.
