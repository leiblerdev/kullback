# Synthesis: how to build the eval from agent traces (what the research says, 2026-08-26)

Eight reports in this folder. This file is the digest plus the collisions with decisions already recorded in `CONTEXT.md` and `docs/adr/`.

## The eight findings that decide the design

1. **Static replay of a logged Run with a swapped model is unsound past the first step.** The Replay Gap (CMU, Aug 2026): 74-77% of early swaps diverge at the first post-fork action; only 3-8% of the original later states stay valid; a constant "always fail" baseline beat log-stitching. Late forks are far safer (85-86% valid). Same-model FP8 replays diverged 90-96%, so serving stack matters too. Consequence: teacher-forced Step replay is a screen, never the verdict. (02, 03)

2. **Grade the end state, not the path.** Every benchmark that survived contact (tau2, Gaia2, MCPMark, SWE-bench, Terminal-Bench) compares world state or final artifact, and allows any path that gets there. Path-match graders (NESTFUL full-sequence ~25%) mostly measure imitation of one path. Anthropic: exact tool sequences are "too rigid". (01, 04)

3. **The frontier's own trajectory is a floor, not a ceiling, and it is a noisy label.** OpenAI found 59% of a SWE-bench Verified failure sample was label error; tau2 needed 75+ task fixes; expert trajectories contain failures. Judges collapse when the reference is wrong. Mitigation: re-roll the frontier on its own Runs (k=3-5) and keep only Runs whose outcome is stable. (01, 06)

4. **Teacher-forced step accuracy is neither necessary nor sufficient for Run success.** Agents recover from most local errors (61.9% of non-critical errors repaired; 58.8% of transition failures recovered); only ~16% of steps are "critical" (swapping the action flips the outcome). The Intervention Paradox: a 0.94-AUROC failure predictor still degraded agents when acted on. So "all-or-nothing over Steps" would be systematically too pessimistic. (06)

5. **Do not grade reasoning text for quality.** CoT faithfulness is 20-40% and falls with capability; "Gaming the Judge" inflated judge false positives by up to 90% by editing only the reasoning text. No lab recommends it. Use reasoning only as a monitor: fabricated observations, plan/action inconsistency, intent to bypass. Strip it before any pairwise judgment. (04, 06)

6. **Tool-call matching must be tiered per argument, and decomposed.** Hard fields (ids, enums, amounts, dates) exact after normalization; free-text fields (queries, messages) via cheap similarity then a binary LLM check, or by executing both calls and comparing results. Subset containment with a dependency DAG, not strict sequence. Always publish the decomposition (over-call, under-call, wrong tool, hallucinated tool, missing required param, hallucinated param, ungrounded value, repeated call). (05)

7. **Judges top out around 70-75% precision on trajectories and are lenient.** No judge beat ~70% on AgentRewardBench; ToolPRMBench frontier judges 73-75%; trained PRMs from math/web are at chance on tool use. What works: give the judge tools to verify (Agent-as-a-Judge 60% to 90%), observation-anchored binary checklists, ternary step labels (+1 / 0 exploratory / -1), cross-family judge, 3-5 samples majority vote, kappa against 100-200 human labels, false-positive rate on deliberately broken trajectories. (04)

8. **"Clears the bar" is a statistics claim and needs volume.** Pre-registered non-inferiority margin (3-5 points), K=4 seeds per Run, paired per-Run differences, clustered SEs, one-sided 95% CI, plus a pass^k reliability gate. A 5-point margin at 80% power needs ~150-250 Runs per Task; 20-40 Runs resolve only 10-15 point margins. Small Tasks get pooled with shrinkage or reported as "insufficient evidence". Hold out 20%. (06, 03)

## Supporting findings

- **Savings unit is often the Step, not the Task.** TwinRouterBench: 71% of steps verified low-tier-safe; downgrade late, escalate early. Cluster Runs by intent (Clio-style), then Steps by role (tool, position, output shape). (02)
- **Baselines before model swaps.** Anthropic's own data: effort reduction and prompt caching (2.5-3.7x) often beat switching models. Every recommendation must beat "frontier at low effort with caching". (02)
- **Charge failures at full cost plus penalty.** Report net savings at equal Run success, never token savings. Cheap models can take more steps (step appetite). (02)
- **Filtering signals for replayability**: full system prompt and tool schemas present, every tool result present, model id per step, MCP hints (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`), prefix re-execution reproduces recorded results (99.99% in hermetic envs). Non-deterministic tools (search, time) get cassettes. PII: keyed-hash pseudonymization at capture, frozen as the eval artifact. (03)
- **Label quality**: human error analysis on 100+ Runs first, binary labels, stop when ~20 more Runs add no new category; automated step-error localization is 11-34% accurate, so humans label the gold set. (03, 04)

## Collisions with decisions already made

| Recorded decision | What the research says | Resolution proposed |
|---|---|---|
| **Step** is "the unit that is replayed and graded" (CONTEXT.md) | Step replay is a screen; the verdict must come from re-executed Runs | Step is the unit that is *screened*; Run is the unit that is *verdicted* |
| Pending Q3: all-or-nothing Run verdict from Step verdicts | Neither necessary nor sufficient; too pessimistic; ~16% of steps are critical | Run verdict comes from branched re-execution (outcome check); Step screen only gates which Runs get re-executed |
| ADR-0003: Match first, Appeal can only rule "equivalent" or "reference wins" | Reference is a floor; path-match penalizes valid alternatives; but conservative direction is right | Keep ADR-0003 for the Step screen. Add: reference Runs must be stable under frontier re-roll, else excluded |
| User intuition: LLM judge evaluates the reasoning | Do not grade reasoning quality; use it only as a monitor | Reasoning text stripped from grading; monitored for fabrication and plan/action mismatch |
| "The bar" = frontier's score on the customer's inputs | Needs a margin, seeds, paired CIs, pass^k gate | "Clears the bar" = non-inferior within a pre-registered margin, with reliability gate |

## Metrics to track (the full list)

**Run level (the verdict)**
- Run pass rate (pass@1) on re-executed Runs, per Task, per model
- pass^k reliability (k=4): all seeds succeed
- Non-inferiority: paired difference vs frontier with one-sided 95% CI and the pre-registered margin
- Key-node completion: required milestones any valid path must hit
- Hard-constraint violations (minefields): destructive/unauthorized action, policy breach, fabricated tool result. Zero tolerance
- First divergence index (where the candidate leaves the frontier path) and first critical error (earliest step that flips the outcome)
- Steps per Run, tokens per Run, cost per Run, wall time per Run (step appetite)
- Net savings at equal success, failures charged full cost plus penalty
- Reference stability: frontier's own pass^k on the same Runs (label quality)

**Step level (the screen and the diagnostics)**
- Strict next-action accuracy (type AND tool AND args)
- Lenient accuracy after Appeal (equivalent counted as pass)
- Action-type confusion: over-call, under-call, abstention agreement, ask-vs-answer
- Tool selection precision / recall / F1; hallucinated-tool rate; repeated-call rate; extra-call rate; missing-call rate
- Argument schema validity: missing-required rate (PMR), hallucinated-param-name rate (PHR), type errors
- Typed-value accuracy on hard fields; free-text equivalence on soft fields (similarity, judge, or execution)
- Ungrounded-value rate (argument hallucination): value appears in neither user context nor prior observations
- Recovery rate after tool error; loop detection (identical call repeated N times)
- Grounding checklist (ternary): action consistent with last observation; conclusion follows from tool output; no claimed-but-unobserved result

**Judge quality (so the customer can trust the above)**
- Cohen's kappa vs 100-200 human-labelled items, per judge, per Task family
- False-positive rate on deliberately broken trajectories
- Position-swap consistency on pairwise calls
- State-based vs trajectory-based verdict agreement (estimates how often the reference was not the only valid path)
- Judge-human discordance rate published per Task

**Dataset and Task**
- Runs per Task, spend share, volume share
- Replay tier per Run (A: step-screen only; B: re-executable; C: needs simulated user)
- Sample size and power per Task (is a verdict even possible)
- Holdout status; refresh date; model versions and list prices used

**Reasoning (monitors only, never scores)**
- Fabricated-observation flag
- Plan/action mismatch flag
- Bypass-intent flag
