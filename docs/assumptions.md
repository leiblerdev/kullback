# Assumptions register

Things the design currently takes as true without proof. Each entry says what we assume, how strong the evidence is, what would show it wrong, and what breaks if it is. When an assumption is tested, move it to the bottom section with the result. New assumptions are added the moment they are made, not later.

Evidence grades: **measured** (we or a cited source measured it on comparable data), **cited** (a published number from a different setting), **inferred** (reasoned from adjacent evidence), **none** (a guess we chose to proceed on).

## About the traffic we will see

| # | Assumption | Evidence | How we find out | What breaks |
|---|---|---|---|---|
| A1 | Most API chat-agent traffic is single-call; the multi-call tail is where the Environment earns its cost. | cited, weak: one Datadog figure (59% of agentic requests make one service call), which counts service calls, not tool calls. No OpenAI or Google number exists. Anthropic cannot measure depth on its API. (R16, R20) | Per-customer histogram of tool calls per Run on ingest. | Nothing breaks; if wrong, the Environment machinery pays for itself sooner. Building for the hardest case (D24) already covers both. |
| A2 | Coding agents are not single-call: Claude Code shows 6.6 model calls per human message and 25 to 45 minute tail turns. | measured (R13) and cited (Anthropic autonomy report, R20) | Same histogram. | If a customer's coding traffic is shallow, the sub-agent stitching and compaction handling are over-built for them. |
| A3 | Our first customers run support, email, calendar and ticketing agents. | none: founder's read of the market | First three signed customers. | Report 19's reference environments (tau2, AgentDojo, WorkArena) stop being the right templates; the generator's emit list stays valid. |
| A4 | Tool definitions travel inside the traces (the `tools` parameter of each LLM call, MCP `tools/list`). | inferred: true for API-level and MCP traces (R16); false for Claude Code JSONL, which carries no tool schemas (R13). | Count Runs with and without tool schemas on ingest. | Hole 3 (unseen tools) is not free to fill; the schema ask moves up the trust ladder (ADR-0006). |

## About the Environment

| # | Assumption | Evidence | How we find out | What breaks |
|---|---|---|---|---|
| A5 | Canonicalization (ids, timestamps, float formatting, key order, whitespace) removes all cosmetic replay drift, so write-effect fidelity can reach 100%. | inferred: tau2 needed `strict=False` for `25` vs `25.0`; AppWorld needs `approx`, rounding to the day, order ignoring; BFCL uses plain equality with no normalization. No source evidences 100% (R20). | Write-effect fidelity per Task on the first customer, with the residual mismatches classified. | Gate A rejects good Environments; Tasks sit at "Environment unverified". Q12 addresses the read side. |
| A6 | Read-observation fidelity will be well below 100% on real traces and needs a measured floor rather than a fixed gate. | inferred from A5 and from tau2 skipping reads entirely. | Same measurement, reads separately from writes. | If reads replay at 100% too, Q12's amendment was unnecessary and can be reverted. |
| A7 | A traces-only Environment leaves a large "assisted" share under Gate B on the first customer. | inferred: Candidates diverge from the frontier's path 30 to 50% of the time (R06); every off-path read of an unseen entity is a hole. | Gate B assisted share per tool, first customer. | If small, the schema rung of ADR-0006 is rarely needed. If very large, the first report has few verdicted Runs and the schema ask comes early. |
| A8 | Reconstructing S0 by inverse replay (reads, then writes) from the whole corpus is sound when Runs are ordered by time. | inferred: tau2's `set_state` does the write half; nobody publishes the read half. | Replay fidelity itself is the test. | Cross-Run interference in S0 shows up as write-effect mismatches on Runs that share entities. |
| A9 | Recorded tool errors (3.8% of Claude Code calls) are enough to infer error-path logic for tools. | none | Share of tools with at least one recorded error, per customer. | Error paths in generated tools are guessed from policy text; a Candidate that hits one gets an invented answer. Tag as assisted. |
| A10 | Per-tool static read/write/idempotent/open-world flags (MCP annotations, tau2 `mutates_state`) are sufficient; no per-call flag is needed. | cited: OpenHands and tau2 both use static flags (R20). Bash-style tools are 9% ambiguous per call (R13). | Share of calls through ambiguous tools. | Ambiguous tools (shell, generic HTTP) need per-call classification, which is a model judgment and a new error source. |

## About the Verifier and the judge

| # | Assumption | Evidence | How we find out | What breaks |
|---|---|---|---|---|
| A11 | k = 4 to 5 frontier re-rolls are enough to separate required from allowed atoms. | none: chosen to match K = 4 seeds and pass^4. | Atom stability as k grows on the first customer (does the required set change from k=4 to k=8?). | Too few: allowed atoms mislabeled required, valid Candidates fail. Too many: cost. |
| A12 | A 10% human audit of Verifiers and judge rulings, with the agreement rate published, is enough to earn trust. | none for the rate; cited comparator: 18.5% evaluator-human misalignment across four benchmarks, 9.8% on tau2 retail (2607.02577, R20). | Observed disagreement on the first customer; tune the rate per Task from it (todo). | If disagreement is high, 10% is too small to bound it; if very low, 10% is wasted expert time. |
| A13 | Replaced by D48: setup review before any Verdict, blind audit after. Remaining assumption: the setup review is fast enough not to stall the first report. | none; decided 2026-08-27. | Time the review per Task on the first customer. | If review takes days per Task, reports are human-paced and coverage must be cut to prominent Tasks. |
| A14 | The customer will actually read and correct Intents shown as Task names (decided D47; the behavior is still assumed). | none: confirmed as a decision on 2026-08-27, not yet observed. | Founder decision; correction rate on the first customer. | If hidden, the customer cannot catch ungrounded Intents and the audit carries all of that load. |
| A15 | Two domain experts agree on pass/fail for Runs in a well-formed Task; agreement on diagnosis (reason codes) is materially lower. | cited: Anthropic rule; TRAIL 11% localization, AgentErrorTaxonomy kappa 0.55 (R18). | Audit agreement, verdicts and reason codes reported separately. | If verdict agreement is also low on some Task, that Task is malformed and must be split. |
| A16 | Substring communicate checks after normalization are reliable enough for numbers, names and ids. | cited, against us: "substring communication checks" is a named failure mode in 2607.02577. | Audit disagreements tagged to communicate atoms. | Move more communicate facts to the grounded-rubric judge (D25 place 2). |
| A17 | The dispute queue stays under about 5% of Runs per Task once Verifiers mature. | none | Queue share per Task. | Above 5% the Task is reported "Verifier immature"; if that is most Tasks, the atom derivation is too narrow. |

## About the statistics and the bar

| # | Assumption | Evidence | How we find out | What breaks |
|---|---|---|---|---|
| A18 | A 3 to 5 point non-inferiority margin is what customers accept. | none: chosen from R06's convention. | Customer conversations. | Margin too wide: routing plans ship Candidates that are noticeably worse. Too narrow: nothing ever clears. |
| A19 | Roughly 150 Runs per Task for a 5-point margin, 30 as the floor for a provisional CI. | cited: sample-size arithmetic at a 70% frontier pass rate (R06). | Observed per-Task variance; the rule is CI half-width below margin (D36). | Customers with many small Tasks see mostly "provisional" or "insufficient". |
| A20 | The frontier's recorded success is a meaningful bar. | cited, against us on hard Tasks: tau2 frontier pass^1 retail 74%, airline 56%, telecom 34%. | Absolute pass rates shown next to the delta. | "Non-inferior" can be true while both models are bad; the report must show the absolute number. |
| A21 | Duplicate Runs can be collapsed with a weight using embedding neighbor distance. | cited, weak: Nomic default 0.1 embedding cutoff and Lilac 0.85 Jaccard flag duplicates but do not weight them; Magpie's rule has no tool precedent (R20). | Share of Runs collapsed and the effect on CIs. | Over-collapsing shrinks sample sizes; under-collapsing inflates confidence. |

## About the simulated user

| # | Assumption | Evidence | How we find out | What breaks |
|---|---|---|---|---|
| A22 | Replaying recorded user turns until an applicability gate fails, then simulating from Intent, is sound. | none: our design, no published validation. | Simulator error rate on held-out multi-turn Runs. | Multi-turn Verdicts carry a noise floor larger than the margin (16 to 47% simulator error by domain, 9 to 14 point inflation; R08). |
| A23 | A simulator conditioned on the customer's real user turns is closer to real users than hand-written personas. | cited: 2606.20708 on 2,790 real sales conversations shows persona simulators halve expressed resistance; but even log-grounded simulators miss users who disengage (R20). | Paired-trajectory Turing test on held-out Runs; abandon rate. | Multi-turn Tasks stay noisy; add an abandon behavior and report it (todo). |
| A26 | When the frontier is re-run to derive atoms, the simulated user repeats the real user's answers (refund method, address) and varies only what the real user left open (a reason's wording). | none: this is what makes user-elicited values split into required and allowed (D42, D43). | On the first customer, compare re-run user replies against the recorded user turns per elicited field. | If the simulated user answers generically, user preferences become "allowed" and a Candidate that ignores them passes. Depends on D27. |
| A27 | The customer's traces contain at least one tool error per tool family, so the Environment can copy the real error encoding instead of guessing it. | none; R21 found five encodings across five benchmarks, so a wrong guess is likely. | Count error observations per tool at ingestion; list every guessed encoding on the Environment. | A Candidate that would recover from the real error text may loop or give up on the guessed one, and the Verdict reflects our guess, not the model. |
| A28 | An Environment built from one customer's traces generalizes to that customer's held-out Runs and alternative paths, not only to the seed Runs. | none; D51 makes this a required test. | Hold out Runs per Task before building; report seed vs held-out fidelity side by side. | Fidelity numbers are overfit and Verdicts on new traffic are wrong. |
| A29 | First-pass auto-derived Verifiers will carry a defect rate above the 10 to 20% seen in hand-written benchmark specs (tau2 75+ fixes, MCP-Atlas 13.5%, R22 section 1.5). | cited for hand-written specs; ours are generated, so inferred higher. | Defects found per Task by the setup review (D48 check 1) and the after-audit (check 2), tracked over time. | If the rate is much higher, prominent-Task-only review (D48) leaves too many bad checklists live; widen coverage or slow the pool. |

## About cost and effort

| # | Assumption | Evidence | How we find out | What breaks |
|---|---|---|---|---|
| A24 | Building a first Environment for a customer takes on the order of 20 to 30 expert hours. | inferred (R12), not measured. | Time the first two customers. | Pricing and onboarding promises. |
| A25 | Building the generator by rebuilding tau2's mock and retail domains from their own trajectories is a fair test of it. | inferred: tau2 has policy, tools, DB and trajectories, which is more than a customer gives. | Then run it on Claude Code JSONL, which has none of those. | The generator works only when handed a benchmark's scaffolding. |

## Tested (move entries here with the result)

None yet.

- **A30** (2026-08-27, D78): the required atom set (D43) stabilizes within 10 frontier re-runs on most Tasks. Test: tau2 retail slice, n = 10 re-runs per Task, required set as a function of k. If false on more than 10% of Tasks, revisit the D43 definition before raising k.

- **A31** (2026-08-27, D92): two agentic judges with tools plus the deterministic Verifier catch most Verifier and Reference defects without a human. Test: on the tau2 slice, judge disagreement rate and judge-vs-tau2-reward agreement per Task; on the first customer, judge verdicts against the customer's verdicts (D62). If agreement is below the D80 explained-100% bar, human review (D48) returns as required, not optional.
