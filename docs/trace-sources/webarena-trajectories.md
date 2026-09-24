# WebArena trajectories

## Link and owner

Official environment and repo: https://github.com/web-arena-x/webarena, released 2023-12-21. Third-party trajectory collections on Hugging Face: https://huggingface.co/datasets/HaoranLiu/WebArena and https://huggingface.co/datasets/thuml/webarena-world-model-cot.

## What a row is

Not independently confirmed this pass for either the official or third-party releases. WebArena's own task suite includes human-authored gold trajectories for most of its tasks.

## Provenance

Human-authored gold trajectories for the official task suite (real people solving the tasks to produce a reference path); third-party sets are model-run attempts at the same tasks.

## Counts and size

812 tasks total, 794 with human-gold trajectories, per the official benchmark's own numbers.

## Schema

Not independently confirmed this pass.

## One excerpt

Not obtained this pass.

## Tool-call shape

Action plus observation pairs: a browser action (click, type, navigate) and the resulting page/DOM state, based on WebArena's general design; not independently re-verified field-by-field this pass.

## Environment and reference verdict

Reusable: WebArena's own environment is a set of self-hosted Docker web apps (shopping, forums, gitlab, wikipedia, maps), which is offline once installed. That is a genuine strength, similar to SWE-bench's per-instance images.

## Licence (quoted)

Not confirmed for the official release this pass. `HaoranLiu/WebArena` on Hugging Face lists licence "other" (unverified, not independently re-checked). `thuml/webarena-world-model-cot` is MIT.

## How to download

Official repo: `git clone https://github.com/web-arena-x/webarena.git`, then its own Docker setup for the self-hosted apps. Third-party sets: `hf download HaoranLiu/WebArena --repo-type dataset` or `hf download thuml/webarena-world-model-cot --repo-type dataset`.

## Fit for Kullback

Good environment fit (self-hosted, offline, real web apps), and a real human-gold-trajectory baseline exists for most tasks, which is a stronger provenance signal than most benchmarks in this survey. The open work is entirely about the trajectory data: confirming whether the official 794 gold trajectories are published in a form with full tool-call arguments and results (not just the final action sequence), and getting a clean licence read, since the two third-party sets disagree with each other (unverified "other" versus MIT) and neither has been checked against the official repo's own terms.

## Open questions

- Whether the official repo publishes the 794 human-gold trajectories with full step-by-step tool call and result detail, or only the final answer/action sequence.
- Exact licence of the official WebArena release itself, separate from either third-party HF collection.
- Whether either third-party set records multiple independent attempts per task (needed for the held-out check) or only one attempt each.
- Resource requirements to stand up the full self-hosted Docker stack (shopping, forums, gitlab, wikipedia, maps) locally.

## Related dataset found this pass

`webarena-x/webarena-infinity-trajectories` (MIT, README cached before the session restart) is a distinct, newer benchmark from the same GitHub org (github.com/web-arena-x/webarena-infinity): 2,329 successful trajectories across 3 models (Gemini 2.5 Flash, Kimi K2.5, Qwen 2.5 VL Plus) and 13 auto-generated web-application environments sourced from real product documentation (Gmail, GitLab, PayPal, Xero, Elation EHR, Superhuman, Handshake, Linear, Figma), with a `manifest.json` index, per-task `history.json` (reasoning plus actions) and `result.json` (pass/fail plus a verifier message) per trajectory. It is not the original WebArena and does not have the 794 human-gold trajectories the official benchmark does, but it is more clearly licensed (MIT, quoted directly on the card) and better documented than either third-party set above, and is worth treating as a separate, likely stronger candidate if the environment (auto-generated per-environment web apps rather than WebArena's five fixed sites) is judged close enough in kind.
