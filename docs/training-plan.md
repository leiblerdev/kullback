# Training plan: prove the environment by training in it (2026-08-28)

My words: "plan being we will pick a benchmark, evaluate a frontier model on it (or the cheaper ones) and then we will train our model using OPD or RL (for that we will also create the harness for it) on the environment this generates (or for the time being we don't create the harness and just write the code ourselves for multiple methods) and then prove that this works. If it doesn't we go back and improve our harness for synthetic environment generation."

That is the loop. Everything else in this repo is downstream of whether a model trained inside a Kullback-built environment gets better on the real thing.

## The benchmark

tau2 retail first, because it is the one place with a real `tools.py` and `db.json` beside public traces, so a model trained in the rebuilt environment can be scored on the real environment, and the same training run can be repeated on the real environment as the ceiling. Airline second, telecom after the five retail-shaped assumptions in `cross-domain-check.md` are fixed. A domain that is not tau2 at all comes after that.

## Step 1: baselines on the real environment

Run the frontier model that produced the traces and two or three cheaper models, plus the untrained 2B model, on the real tau2 retail environment, on the held-out Task split (the 20% of Runs the build never sees, by task id). Score with tau2's own reward, pass^1 and pass^4, and beside it with Kullback's own Verdict on the same Runs, so the two graders can be compared before either one is used for training. This is the number every later row is measured against.

## Step 2: build the environment

Build from the seen 80% of traces with the harness as it is, with the gates in the README's measured section as the acceptance bar. The Verifier is the reward. Nothing from the held-out split, and nothing from tau2's `tasks.json`, `db.json` or `tools.py`, touches the build.

## Step 3: train in it, several methods, plain scripts

No training harness yet. One script per method, each reading the same Environment and writing the same result record:

- Rejection-sampling fine-tuning: sample the frontier model in the Kullback environment on the built Tasks, keep the trajectories the Verifier passes, fine-tune the 2B model on them.
- On-policy distillation: the 2B model samples in the Kullback environment, the frontier model scores each token or turn as the teacher, the student moves toward the teacher on its own samples.
- RL with the Verifier as the reward: GRPO or a close relative, the Verdict per Run as the return, the simulated user and the tools as the environment.

Each method also runs on the real tau2 environment with the same budget. That second run is the ceiling: the best the method can do when the environment is perfect.

## Step 4: score on the real environment

Every trained model goes back to the real tau2 retail environment on the held-out Tasks and is scored the way the baselines were. The table has one row per method and environment pair, and three columns: untrained 2B, trained in Kullback's environment, trained in the real environment.

## What counts as proof

The gain from training in the Kullback environment is a large share of the gain from training in the real one, on the held-out real Tasks, for at least one method. If the Kullback-trained model gains nothing while the real-environment run gains a lot, the environment is wrong somewhere and the gap is a debugging signal: which Tasks the trained model fails on the real environment but passes in ours are the Tasks where the rebuild diverges. That list goes back into the Builder.

Also tracked: reward hacking. A model that scores well in our environment and badly on the real one has found a gap in the Verifier or the tools, and that gap is a Builder bug to fix, not a training bug.

## Not yet decided

Which 2B model (Qwen3 1.7B, Gemma 3 1B or a 2B-class model with a chat and tool-call format). Which frontier teacher. The compute budget per run. Whether the first RL run uses full fine-tuning or LoRA. These get decided when step 1 is running.

## Prior evidence worth knowing

Distil Labs, "traces vs synthetic" (https://www.distillabs.ai/blog/traces-vs-synthetic-benchmark/, read 2026-08-28): a Qwen3-1.7B student (LoRA rank 64) fine-tuned on about 2,000 synthetic conversations that a teacher generated from 327 Schema-Guided Dialogue restaurant traces used as context, versus the same student fine-tuned directly on the traces. On clean traces the two tie (0.866 versus 0.864, LLM-judge score). When the traces are corrupted the synthetic route holds and the direct route falls: 50% corrupted tool calls 0.844 versus 0.721, schema drift 0.844 versus 0.585, only 5 traces 0.852 versus 0.649, 80% irrelevant traces 0.858 versus 0.694. The best teacher (GLM-5, 744B) scored 0.835, below the trained 1.7B student. Their explanation: the traces carry the distribution and the task description plus tool schema carry the norm, and the generation step keeps the first while the second corrects it.

What it says for this plan: the synthetic route from traces is the one with evidence, the student size (1.7B) is the size I named, and messy real traces are the case where generation from traces wins by the most, which is the case customers actually have. What it does not settle: one domain, one task type, scores from an LLM judge on turn pairs rather than on End state, and no environment in the loop, so the model was never run against tools that answer back. Our version replaces the judge with the Verifier and the context-only traces with an executable rebuild, which is a stronger test and also the one that could fail.

