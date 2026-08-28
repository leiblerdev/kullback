# A Run's verdict comes from its end state after re-execution, never from path match or reasoning text

Status: accepted (2026-08-26)

To decide whether a cheaper model clears the bar on a Task, we re-execute the Task's Runs in an Environment built from the customer's traces and compare each Run's End state (side effects plus final answer) against the frontier Reference's End state, with hard constraints as an absolute gate. Step-level comparison against the frontier's recorded actions is used only as a Screen to choose which Runs to re-execute. The reasoning text a model emits is never graded; it is monitored for fabricated observations and plan/action mismatch only.

Why: static replay of a logged Run with a swapped model is unsound past the first divergent action (The Replay Gap, CMU 2026: 3-8% of later recorded states remain valid for early swaps). Path matching penalizes valid alternative routes and measures imitation, and every durable agent benchmark (tau2, Gaia2, SWE-bench, Terminal-Bench, MCPMark) grades the world, not the transcript. Grading reasoning is not possible from text alone: chain-of-thought faithfulness is 20-40% and falls with capability, and editing only the reasoning text inflates judge false positives by up to 90% (Gaming the Judge, 2026). No lab recommends it.

Considered: all-or-nothing aggregation of Step verdicts into a Run verdict (rejected: agents recover from most local errors, only ~16% of steps are outcome-critical, so it is systematically pessimistic); reasoning-quality rubrics (rejected: unfaithful, gameable, and any pressure on reasoning invites obfuscation).

Consequences: the Environment is central to the product; a Task whose Runs cannot be re-executed gets a Screen result but no Verdict and cannot be recommended for a move. Verdicts are binary; "closeness" is a diagnostic. The customer-facing "what we are not doing" list includes: we do not grade reasoning, we do not grade the path, we do not let a judge overrule an End-state check.

## Counter-evidence (added 2026-08-26)

This is a position, not the field consensus. Several systems add path information to reward and report gains: EnvFactory's ablation found 0.5 trajectory-match plus 0.5 final-state equivalence beat either alone (https://arxiv.org/abs/2605.18703); BFCL multi-turn requires the ground-truth call sequence to be a subset of the executed calls in addition to state checks (https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html); AgentScaler requires exact function-call match for read-only tasks (https://arxiv.org/abs/2509.13311); ClawTrack reports process-based trajectory filtering improves post-training (https://arxiv.org/abs/2607.28037). Those results concern training signal. For a pass/fail evaluation whose claim is "safe on your traffic", grading the path would fail valid alternative paths and would make the Verdict depend on the frontier's habits rather than the user's Intent, so we keep End state only and compute path agreement as a diagnostic (tool selection P/R/F1). If Scenarios are later used for post-training, revisit whether path signal belongs in that reward.

## Amendment (2026-08-27)

"End state" here means the effects of the Run, and effects fall on two things: the world (writes) and the user (what they were told or asked, not only the final message). A question the frontier asked in every successful re-run is therefore a required End-state atom, and a write whose value came from the user's answer must hold the answer the user gave in the Candidate's own Run. What remains path, and never changes a Verdict, is exactly what has no effect: reads, the order of actions, and reasoning text. This does not reopen trajectory matching; the counter-evidence above concerns tool-call sequences, which stay ungraded.
