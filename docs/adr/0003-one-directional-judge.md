# Step screening is agreement first, and the judge can only rule for the reference

Status: accepted. Scope narrowed 2026-08-26: this governs the Step-level Screen only. Verdicts come from re-executed Runs (ADR-0004).

When grading a Candidate model's Step against the frontier's Reference action from a production Run, we first check for a structural Match (same tool and required arguments, or the same decision to stop and answer). Only a mismatch goes to an LLM judge (the Appeal), and the judge may rule only "equivalent" or "reference wins"; it cannot rule that the Candidate did better. Final-answer steps always go through the Appeal under the same rule.

Why: the product promise is "nothing ships below the bar." A conservative error (keeping a Task on the frontier when the cheaper model was fine) costs the customer some savings; a liberal error (moving a Task that then regresses) costs us the customer. So the metric is built to make only the first kind of mistake. Letting the judge favor the Candidate would also make every verdict rest on judge quality, which is the weakest link, and would invite Goodharting the judge.

Considered: adequacy-only judging (fairer to small models, but verdicts become judge-quality-bound); agreement-only (deterministic, but penalizes every valid alternative path and makes the report needlessly pessimistic).

Consequence: reported savings are a floor, not an estimate. Any "Candidate is better than frontier" finding is out of scope for the routing plan and must come from a separate, explicitly labelled analysis if we ever want it.
