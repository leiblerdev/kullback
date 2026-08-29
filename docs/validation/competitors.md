# Competitors and adjacent products

One entry per company, kept short: what they sell, in their words where possible, and where it overlaps or does not with the Harness (an Environment plus Verifiers built from a customer's own traces, Verdicts computed by code). Add a line when a prospect names one.

## Arga Labs (added 2026-08-29, founder: "they build digital testing environments")

[argalabs.com](https://www.argalabs.com/): "Real world sandboxes for testing and training AI agents." Simulation environments made of "stateful twins of the APIs, CLIs, and MCPs your agents use" (Slack, GitHub, Gmail, Stripe and the like), "Scenarios" seeded with mock or production data, test runs that are graded, and evidence capture of "every provider call, response, latency, side effect, and service-state change." Use cases they name: RL training environments, enterprise agent sandboxes, code-change validation.

Overlap: the same claim that an agent has to be tested in an executable, stateful, repeatable world rather than judged on transcripts, and the same three outputs (a world, graded runs, evidence). Difference: their world is a twin of third-party services the agent calls, built by them per service; ours is a twin of the customer's own system, built from the customer's traces by the Builder, with the Tasks and the Verifiers derived from those traces. They answer "does the agent behave against Stripe", we answer "does the agent do what this company's agents were recorded doing, under this company's policy". A customer whose agent lives on public SaaS APIs is theirs; one whose agent lives on its own tools and policy is ours; many are both, and the twin of a public API could be one of our tool bodies.

## FinetuneDB (seen 2026-08-29)

[finetunedb.com](https://finetunedb.com/): a fine-tuning platform. Log production requests, curate them into datasets in a collaborative editor, evaluate outputs with human and AI feedback, fine-tune, repeat. The loop improves the model; ours improves the Environment the model is measured in. Adjacent, not competing: Runs that pass a code Verifier are labelled training data (todo: verified synthetic data), which is what their loop consumes.
