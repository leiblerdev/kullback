# ADR-0004: A Run's Verdict comes from its End state after re-execution, never from path match or reasoning text

Status: accepted 2026-08-26, amended 2026-08-27.

Context: to decide whether a cheaper model clears the bar on a Task, its Runs must be graded. Static replay with a swapped model is unsound past the first divergent action, path matching measures imitation, and reasoning text is unfaithful and gameable.

Decision: re-execute each Run in an Environment built from the customer's traces and grade its End state against the frontier Reference's, with Hard constraints as an absolute gate. Step comparison is only a Screen for choosing what to re-execute. Reasoning text is never graded, only monitored for fabricated observations.

Amendment: End state means the Run's effects, on the world (writes) and on the user (what they were told or asked). A question the frontier asked in every successful re-run is a required atom; a write whose value came from the user's answer must hold the answer given in the Candidate's own Run. Reads, action order and reasoning have no effect and never change a Verdict.

Why: every durable agent benchmark grades the world, not the transcript. Chain-of-thought faithfulness is low and falls with capability; editing reasoning text alone inflates judge false positives.

Considered: aggregating Step verdicts into a Run verdict (agents recover from most local errors, so it is systematically pessimistic); reasoning rubrics (unfaithful, gameable). Counter-evidence exists: several training pipelines gain from path signal in the reward. That concerns training; for a pass or fail claim of "safe on your traffic", path grading would fail valid alternatives, so path agreement stays a diagnostic.

Consequences: the Environment is the centre of the product. A Task whose Runs cannot be re-executed gets no Verdict and cannot be recommended for a move. Verdicts are binary.
