# ADR-0005: Routing-plan Verdicts come from Replicas only; synthesized Scenarios never count toward the bar

Status: accepted 2026-08-26.

Context: the Environment generator can rebuild the customer's world so real Runs re-execute (Replicas) and can synthesize new Tasks grounded in the traces (Scenarios, with generated verifiers).

Decision: only Replicas feed the pass rate, "clears the bar" and the routing plan. Scenarios are a later phase for coverage, hardening and post-training, reported separately if at all.

Why: the bar is the frontier's score on the customer's own inputs. A synthesized, difficulty-tuned Scenario is not the customer's input, and letting it in changes the claim from "safe on your traffic" to "safe on tasks we invented". Generated verifiers are also where leaky and hackable rubrics concentrate.

Considered: counting Scenarios the frontier passes (proves the verifier is possible, not that the Scenario reflects production); weighting Scenarios lower (any weight makes the bar a blend the customer cannot audit).

Consequence: the generator's first deliverable is replay fidelity on Replicas, not Task volume. Scenario generation is gated on that number.
