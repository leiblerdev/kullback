# Training plan

How we prove a Kullback Environment is good: train a model in it, then score that model on the real
environment. Everything else in the repo depends on this test, and if it fails the gap shows where the rebuild
is wrong.

In the founder's words: "we will pick a benchmark, evaluate a frontier model on it (or the cheaper ones) and then
we will train our model using OPD or RL ... on the environment this generates ... and then prove that this works.
If it doesn't we go back and improve our harness for synthetic environment generation."

## The benchmark

tau2 retail first: it is the one place with a real `tools.py` and `db.json` beside public traces, so a model
trained in the rebuilt environment can be scored on the real one. Airline second, then a domain outside tau2.

## Steps

1. Baselines. Run the frontier model that made the traces, two or three cheaper models and the untrained small
   model on the real environment, on held-out Tasks. Score with tau2's reward (pass^1 and pass^4) and with
   Kullback's Verdict on the same Runs, so the two graders are compared before either is used for training.
2. Build. Build from the seen traces only. Nothing from the held-out split, or from tau2's `tasks.json`, `db.json`
   or `tools.py`, touches the build. The Verifier is the reward.
3. Train, one plain script per method, each writing the same result record:
   - rejection-sampling fine-tuning on the frontier model's Runs the Verifier passes;
   - on-policy distillation, the frontier model as teacher on the student's own samples;
   - RL (GRPO or close) with the Verdict per Run as the return.
   Each method also runs on the real environment with the same budget; that run is the ceiling.
4. Score every trained model on the real environment's held-out Tasks. One row per method, three columns:
   untrained, trained in Kullback's environment, trained in the real one.

`kullback train` runs the training agent over a built Environment (`kullback/trainer/`).

## What counts as proof

Training in the Kullback environment recovers a large share of the gain from training in the real one, on
held-out real Tasks, for at least one method. If it gains nothing while the real run gains a lot, the Tasks the
trained model fails on the real environment but passes in ours are where the rebuild diverges; that list goes
back to the Builder. A model that scores well in ours and badly on the real one found a gap in the Verifier or
the tools: a Builder bug, not a training bug.

## Not decided

Which small model (Qwen3 1.7B, Gemma 3 1B or a 2B-class model with a tool-call format), which teacher, the
compute per run, and full fine-tuning or LoRA.

## Prior evidence

Distil Labs, "traces vs synthetic" (https://www.distillabs.ai/blog/traces-vs-synthetic-benchmark/): a Qwen3-1.7B
student trained on about 2,000 conversations a teacher generated from 327 traces ties direct training on clean
traces (0.866 against 0.864) and wins clearly when the traces are messy (schema drift 0.844 against 0.585). It is
one domain, scored by an LLM judge, with no environment in the loop; our version uses the Verifier and an
executable rebuild, which is the stronger test.
