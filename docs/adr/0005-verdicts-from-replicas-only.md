# Routing-plan verdicts come from Replicas only; synthesized Scenarios never count toward the bar

Status: accepted (2026-08-26)

The Environment generator can do two jobs: rebuild the customer's world so real Runs can be re-executed (Replicas), and synthesize new tasks grounded in the traces (Scenarios: seed intent x persona x hardening trap, with generated verifiers). We decided that only Replicas feed pass rate, "clears the bar", and the routing plan. Scenarios are a later phase for coverage, hardening, and post-training the Student, reported separately if at all.

Why: the bar is defined as the frontier model's score on the customer's own inputs. A synthesized, difficulty-tuned Scenario is not the customer's input; letting it into the verdict silently changes the claim from "safe on your traffic" to "safe on tasks we invented", in either direction (a Candidate can fail traps that never occur in production, or pass traps while failing real traffic). Trust in the report depends on the customer being able to say "these outputs are exactly what my system produces", which only a Replica with measured replay fidelity can support. Scenario generation is also where generated-verifier failure modes concentrate (leaky, narrow, hackable rubrics).

Considered: counting Scenarios whose verifier the frontier passes (rejected: the frontier passing a generated verifier proves the verifier is not impossible, not that the Scenario reflects production); weighting Scenarios lower (rejected: any weight makes the bar a blend the customer cannot audit).

Consequence: the first deliverable of the Environment generator is replay fidelity on Replicas, not task volume. Scenario generation is gated on that number and, when it ships, lives in a separate section of the report and in the post-training loop.
