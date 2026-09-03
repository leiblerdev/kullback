# ADR-0003: Step screening is agreement first, and the judge can only rule for the Reference

Status: accepted. Scope narrowed 2026-08-26: this governs the Step-level Screen only; Verdicts come from re-executed Runs (ADR-0004).

Context: a Candidate's Step is compared to the frontier's recorded action. Some mismatches are valid alternatives, some are errors, and a judge that could favour the Candidate would make every result rest on judge quality.

Decision: check for a structural Match first (same tool and required arguments, or the same decision to stop and answer). Only a mismatch goes to a judge, and the judge may rule "equivalent" or "reference wins", never that the Candidate did better. Final-answer Steps always go to the judge under the same rule.

Why: the promise is "nothing ships below the bar". A conservative error costs the customer some savings; a liberal error costs the customer. The metric is built to make only the first kind.

Considered: adequacy-only judging (fair to small models, judge-bound); agreement-only (deterministic, penalizes every valid alternative).

Consequence: reported savings are a floor. A "Candidate beat the frontier" finding is out of scope for the routing plan.
