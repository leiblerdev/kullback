# AndroidWorld trajectories

## Link and owner

Environment: https://github.com/google-research/android_world, published by Google Research. No canonical first-party trajectory release was found. Third-party trajectory sets referencing AndroidWorld: MobileJudgeBench (931 trajectories across 6 benchmarks including AndroidWorld) and AndroidGen (116 expert demonstrations); exact HF/GitHub links for these were not pinned down before the session restart that ended this pass's deeper research, and are flagged as unverified below.

## What a row is

Not confirmed for either third-party set this pass. Expected shape, by analogy with similar mobile-agent trajectory sets: a sequence of UI actions (taps, swipes, text entry) paired with the resulting screen state.

## Provenance

AndroidWorld's own tasks are human-authored against a real Android emulator; any trajectory set built from it is model-run against those tasks.

## Counts and size

AndroidWorld itself: 116 base tasks, with randomized task-instantiation parameters (many parameter variants per task, which is not the same as multiple independent recordings of one fixed task). MobileJudgeBench: 931 trajectories across 6 benchmarks including AndroidWorld (subset specific to AndroidWorld not broken out). AndroidGen: 116 expert demonstrations.

## Schema

Not confirmed this pass.

## One excerpt

Not obtained this pass.

## Tool-call shape

Expected to be action-plus-observation pairs (UI action in, post-action screen state out), based on the general shape of GUI-agent trajectory sets, but not directly confirmed for AndroidWorld specifically.

## Environment and reference verdict

The strongest part of this candidate: AndroidWorld's own environment is a real Android emulator, installable and runnable offline, with reward computed via SQLite, filesystem, or settings state inspection, a state-diff-verifiable shape close to what Kullback already rebuilds for tau2. This is a materially better fit than most GUI-agent benchmarks, which score from pixels or an LLM judge instead.

## Licence (quoted)

Not confirmed this pass for either third-party trajectory set. AndroidWorld's own repo licence was not re-checked this pass either.

## How to download

Environment: `git clone https://github.com/google-research/android_world.git`, then follow its own emulator setup instructions (not detailed this pass). Trajectory sets: not confirmed, pending identification of the exact HF/GitHub location for MobileJudgeBench or AndroidGen's data.

## Fit for Kullback

The environment fit is genuinely good (real emulator, offline, state-diff-verifiable), which is unusual among GUI-agent benchmarks. The trajectory side is the open problem: no first-party published set exists, and the two third-party candidates named here were not verified deeply enough this pass to commit to either. This is a case where the environment alone might justify future work even if the current trajectory candidates turn out weak.

## Open questions

- Exact HF or GitHub location, row schema, and licence for MobileJudgeBench's AndroidWorld subset specifically (it spans 6 benchmarks, not only AndroidWorld).
- Same for AndroidGen's 116 expert demonstrations.
- Whether either set records both the UI action's parameters and the resulting screen state in a structured (not pixel-only) form.
- AndroidWorld's own repo licence and emulator installation resource requirements (Android Studio / AVD footprint).
