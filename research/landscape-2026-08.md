# Competitive landscape: "traces to evals to replay to per-task cheaper-model recommendation"

Date of sweep: 2026-08-26. Sources: vendor docs, launch pages, blog posts and third-party comparisons fetched via web search. Where a doc did not state something, it is marked "not documented" rather than assumed.

## The exact loop being checked

1. Ingest a company's existing LLM/agent traces (API logs or agent trajectories), from whatever provider or observability tool they already use.
2. Automatically build an evaluation dataset from those traces (task clustering, sampling, expected outputs derived from the production model's own outputs or from human/judge labels).
3. Replay that dataset through cheaper or alternative models (off-the-shelf, not just fine-tunes).
4. Grade outputs on two axes: tool-calling correctness (right tool, right arguments, right sequence) plus LLM-as-judge on the final answer.
5. Report, per task or per workload, which cheaper model is "good enough", with cost delta, so the company can route those tasks and save money.

Column key used below: Traces to dataset (auto / manual / none), Replay across models, Grades tool calls, Grades with LLM judge, Per-task routing recommendation, Pricing / OSS.

---

## A. Routers and gateways

### Not Diamond
- URL: https://www.notdiamond.ai, docs https://docs.notdiamond.ai/docs/router-training-quickstart
- What it does: hosted model router (per-query `select_model`), custom router training, and "Prompt Adaptation" / Prompt Optimization (agentic prompt rewriting across models). Powers OpenRouter's Auto Router.
- Traces to dataset: None automatic. Custom router training requires the customer to supply three things themselves: representative inputs, candidate-model responses for each input, and evaluation scores (min 15 samples, max 10,000 or 5 MB per job). No documented ingestion of production logs.
- Replay across models: No. Not Diamond does not run the candidate models for you in the router-training flow; you bring the responses. Prompt Adaptation (enterprise early access, SAP anchor customer) does evaluate candidate models and parameter configurations in parallel against an eval dataset you provide, and outputs adapted prompts plus model recommendations.
- Grades tool calls: Not documented.
- LLM judge: Not built in; you bring the scores.
- Per-task recommendation: Output is a per-query router, not a per-task report. Prompt Adaptation gives model-prompt pairs per use case but requires your eval set.
- Pricing: pay-as-you-go around $0.05 per million tokens routed; enterprise custom.
- Verdict: closest router to the loop on the "learn from your data" axis, but the data preparation (responses and scores) is the customer's job, and it is not trace-driven.

### Martian
- URL: https://withmartian.com, docs https://docs.withmartian.com
- What it does: model router and gateway (200+ models) using proprietary "model mapping"; open-source ARES (RL framework for training and evaluating coding agents) and K-Steering. Claims 20 to 97 percent cost cuts.
- Traces to dataset: none documented. Replay across models: no. Tool-call grading: no (ARES is an RL environment framework, not a trace grader). Per-task report: no; per-request routing only.
- Pricing: free tier 2,500 requests, then about $20 per 5,000 requests.

### RouteLLM (LMSYS)
- URL: https://github.com/lm-sys/routellm
- What it does: open-source (Apache-2.0) framework for training and serving binary strong/weak routers (matrix factorization, SW ranking, BERT, causal LLM classifier) trained on Chatbot Arena preference data, optionally augmented with LLM-judge labels.
- Traces to dataset: no ingestion of your logs; you could retrain on your own preference pairs but nothing does it for you. Replay: no. Tool calls: no. Per-task report: no. It is a research router, single strong/weak pair.

### OpenRouter Auto Router
- URL: https://openrouter.ai/openrouter/auto
- What it does: task-aware router powered by Not Diamond and, since the July 2026 beta, by "wisdom of the market" (aggregate OpenRouter spend per task category). Picks a model per request from a curated pool; you pay the routed model's price.
- Traces to dataset: no. Replay: no. Tool calls: no. Per-task report: no. It is a black-box per-request router based on global usage, not your data.

### Unify.ai
- URL: https://unify.ai
- What it does: benchmark-driven router with quality/cost/latency sliders and a neural scoring function predicting per-prompt quality; live provider benchmarks refreshed every 10 minutes. Nothing documented about ingesting your production traces or grading tool calls. Per-request routing, not per-task reports.

### Requesty
- URL: https://www.requesty.ai
- What it does: low-latency gateway with smart routing (cost-based, constrained routing with cost and latency ceilings), caching, and a Request Feedback API for user ratings. No trace-to-dataset, no replay, no tool-call grading, no per-task recommendation. Feedback is for analytics, not routing training.

### Arch Router (Katanemo, now Plano)
- URL: https://github.com/katanemo/archgw, model https://huggingface.co/katanemo/Arch-Router-1.5B
- What it does: open-source 1.5B "preference-aligned" router; you write domain and action route policies in config and the router maps each request to a policy and thus a model. No training on your traces, no replay, no grading, no per-task cost report. Routing decisions are configurable and transparent but hand-authored.

### Portkey
- URL: https://portkey.ai
- What it does: AI gateway (acquired by Palo Alto Networks, May 2026), config-driven conditional routing, fallbacks, load balancing, guardrails, MCP gateway, logs and traces, runtime LLM-as-judge scoring available as a guardrail. No dataset-from-traces, no cross-model replay of logs, no tool-call correctness grading, no per-task cheapest-model report. Third-party comparisons note that teams wanting scored model comparisons to drive routing policy still need a separate eval workflow.

### LiteLLM
- URL: https://docs.litellm.ai/docs/proxy/auto_routing
- What it does: open-source proxy. Auto Router v2 (July 2026) merges semantic, complexity and adaptive (multi-armed bandit) routing; plus cost-based, latency, usage routing. The adaptive router learns which deployment "performs best" from online signals, but there is no offline eval loop: no dataset from traces, no replay, no tool-call grading, no per-task report. OSS (MIT) with paid enterprise.

### Helicone
- URL: https://www.helicone.ai, docs https://docs.helicone.ai/features/experiments
- What it does: OSS observability plus 2026 AI Gateway (cost-based routing to cheapest provider meeting thresholds). Experiments: spreadsheet UI where you pull production requests by prompt ID or random sample, or import datasets/CSV, run prompt and model variants, and run LLM-as-judge or custom evaluators side by side.
- Traces to dataset: manual selection from requests. Replay across models: yes, within Experiments, for single-prompt requests. Tool calls: not documented as a grader. Per-task recommendation: no; you eyeball the comparison. Cost dashboards help you "find opportunities to downgrade specific workflows" but the identification is manual.

### Cloudflare AI Gateway
- URL: https://developers.cloudflare.com/ai-gateway/evaluations/
- What it does: gateway with Dynamic Routing (percentage splits, budget fallbacks) and an Evaluations beta. Datasets are filtered log collections; the only evaluators are Cost, Speed and Human feedback (thumbs up rate). No replay, no LLM judge, no tool-call grading, no per-task recommendation. It compares metrics across log slices, not model candidates on the same inputs.

### Vercel AI Gateway
- URL: https://vercel.com/ai-gateway
- What it does: unified API, fallbacks, zero markup, observability. No evaluations, datasets, replay or routing recommendations documented.

### Arcee Conductor
- URL: https://docs.arcee.ai/arcee-conductor/introduction-to-arcee-conductor
- What it does: hosted router that classifies prompt complexity and sends simple prompts to Arcee's own SLMs. Per-request, opaque, no trace ingestion, no replay, no tool-call grading, no per-task report.

### Other routers seen: Orq.ai Smart Router (strong/economical pair, preference-trained), NVIDIA NeMo Switchyard (Aug 2026, judge-based escalation router for agents), Inworld Router, Morph Router, vLLM Semantic Router, NVIDIA llm-router blueprint, Riften (see section D). All are runtime routers; none build an eval set from your traces and report per task.

Notable one-off study: LangChain's Switchyard benchmark (Aug 2026) ran 145 multi-step agent tasks starting on Nemotron 3.5 Lightning with a judge model voting each turn and escalating to Claude Opus 4.8 after two negative verdicts. Result: 7 percent of calls needed the frontier model, 74 percent cost cut, accuracy 86 to 80 percent. This is exactly the kind of "how many of your calls actually need a frontier model" analysis the loop would automate, but it was a benchmark on public task suites, not a product on customer traces.

---

## B. Observability and evaluation platforms

### Langfuse
- URL: https://langfuse.com/docs/evaluation/overview
- Traces to dataset: manual (select traces or observations and add to a dataset; SDK can do it programmatically). Replay across models: yes for prompt-level experiments via UI (dataset x prompt version x model config) and via SDK experiments (your code). LLM judge: yes, managed evaluators run on dataset runs and on live traces. Tool calls: no built-in tool-call correctness evaluator; you write a code evaluator. Per-task recommendation: no; you compare experiment runs side by side. OSS (MIT core), cloud free to 50k units, Core $29/month.

### LangSmith
- URL: https://docs.langchain.com/langsmith/evaluation-concepts
- Traces to dataset: manual via annotation queues or "add to dataset" from runs; online evaluators score live traffic. Replay: playground runs a dataset against prompt and model variants; SDK evaluate() runs your target function. LLM judge: yes. Tool calls: trajectory evaluators exist in openevals / agentevals (trajectory match, LLM trajectory judge). Per-task recommendation: no. Paid seats plus usage.

### Braintrust
- URL: https://www.braintrust.dev/docs/evaluate/playgrounds
- Traces to dataset: one-click add from logs, manual selection; Loop (AI assistant) proposes dataset rows and scorers. Replay: yes, playgrounds run a dataset across multiple models and prompt variants side by side, diff mode shows timing and token variation; remote evals allow multi-step agent tasks. LLM judge: yes (autoevals). Tool calls: scorers are custom; Loop can write trajectory scorers from natural language but there is no packaged tool-call correctness scorer documented. Per-task recommendation: no automatic "cheapest passing model" output; their own articles describe it as a manual workflow of loading an expensive trace and trying cheaper models. Free tier, then usage-based.

### Arize Phoenix / Arize AX
- URL: https://github.com/Arize-ai/phoenix
- Traces to dataset: manual selection of spans into datasets. Replay: Span Replay re-executes a captured LLM span with a different prompt or model in the playground; datasets can be run through app versions in experiments. LLM judge: yes (evals library). Tool calls: OpenInference captures tool spans; tool-call evaluators exist in the evals library (tool calling eval template). Per-task recommendation: no. OSS (Elastic license) plus paid AX.

### Weights & Biases Weave
- URL: https://docs.wandb.ai/weave
- Traces to dataset: manual ("add selected rows to a dataset", including agent messages with tool calls). Replay: weave.Evaluation runs your model function across a dataset; comparison dashboards across evaluations. LLM judge: built-in model-graded scorers. Tool calls: you write the scorer. Per-task recommendation: no. Free tier, paid usage.

### Comet Opik
- URL: https://github.com/comet-ml/opik
- Traces to dataset: manual. Replay: experiments and prompt playground across models. LLM judge: 40+ metrics. Tool calls: agent trace trees captured; Agent Optimizer optimizes prompts, not model choice. Per-task recommendation: no. OSS (Apache-2.0) plus cloud.

### Laminar
- URL: https://laminar.sh, https://github.com/lmnr-ai/lmnr
- Traces to dataset: bulk-create datasets from traces, dataset rows or SQL results. Replay: code-first evals SDK (executor function plus evaluators). LLM judge: your code or "Signals". Tool calls: not packaged. Per-task recommendation: no. OSS (Apache-2.0), YC S24, raised $3M March 2026.

### Maxim AI
- URL: https://www.getmaxim.ai
- Traces to dataset: curate from production logs. Replay: run test suites across prompt and model configurations, plus agent simulation. LLM judge: yes. Tool calls: tool correctness and trajectory metrics at session and node level. Per-task recommendation: no cost-driven per-task report documented. Paid with free tier.

### AgentOps
- URL: https://agentops.ai
- Session replay (visual inspection), cost tracking per session and agent, benchmarks. No dataset from traces for evals, no cross-model replay, no tool-call grading, no per-task recommendation. Free to 5,000 events, Pro $40/month.

### Traceloop (OpenLLMetry)
- URL: https://www.traceloop.com
- OTel-based tracing with a library of quality, safety and structural metrics. No documented dataset-from-trace replay across models or per-task cost recommendation.

### Honeycomb for LLMs
- High-cardinality trace exploration; LLM-specific surface is thin. No datasets, replay, grading or routing recommendations.

### Datadog LLM Observability
- URL: https://docs.datadoghq.com/llm_observability/experiments/datasets/
- Traces to dataset: yes, build datasets from traces (manual selection), 3-year retention, versioned. Replay: Experiments run new prompts and models against datasets in the Playground and SDK. LLM judge: yes. Tool calls: no packaged tool-call grader documented. Per-task recommendation: no. Paid, per-ingested-span pricing.

### Galileo
- URL: https://galileo.ai
- Traces scored by Luna-2 small evaluator models (20+ metrics, cheap), guardrails at runtime. No cross-model replay of logs, no per-task cheapest-model report. Tool-call metrics exist among agent metrics.

### Patronus AI
- URL: https://www.patronus.ai
- Percival: agent that analyzes execution traces and classifies failures into reasoning, execution, planning and domain categories; Judge API and evaluators. No replay of traces through alternative models, no per-task cost report.

### Confident AI / DeepEval
- URL: https://www.confident-ai.com, https://deepeval.com
- Traces to dataset: traces can be curated into datasets on the platform, with human review workflow. Replay: DeepEval runs test cases against your app; model comparison is manual. LLM judge: G-Eval and others. Tool calls: ToolCorrectness metric and span-level agent metrics. Per-task recommendation: no. DeepEval OSS (Apache-2.0), platform paid.

### Judgment Labs (judgeval)
- URL: https://github.com/JudgmentLabs/judgeval
- OTel tracing, agent judges producing structured "behaviors", online scoring, CLI to run evals against production data. No cross-model replay or per-task cost report. OSS SDK.

### Humanloop
- Team acqui-hired by Anthropic Aug 2025; assets not acquired; product ceased independent operation late 2025. Not a live competitor.

### Openlayer
- URL: https://www.openlayer.com
- Governance-flavored evaluation and observability platform (Gartner 2026 market guide). Datasets, tests, monitoring. No trace replay across cheaper models with per-task recommendation documented.

### Vellum
- URL: https://www.vellum.ai
- Workflow builder plus evaluations (datasets, LLM judge, compare across providers and versions before release) and public leaderboards. Manual comparison, no per-task cost recommendation from traces.

### Promptfoo
- URL: https://github.com/promptfoo/promptfoo
- OSS CLI: declarative test cases run across 30+ providers, assertions and LLM-rubric graders, red teaming. Inputs are hand-written or CSV, not auto-derived from traces. Tool-call assertions are possible via custom JS/Python. No per-task cost report (it shows pass rates and token cost per provider in the matrix, which is a useful primitive).

### Inspect AI (UK AISI)
- URL: https://inspect.aisi.org.uk
- OSS eval framework: dataset, solver (including tool-use agents in sandboxes), scorer; provider switch is one line. Built for benchmark evals, not trace ingestion; no per-task routing report.

### Ragas
- OSS metrics including agent goal accuracy and tool call accuracy. A metrics library; no ingestion, replay or recommendation.

### MLflow 3 GenAI / Databricks Mosaic AI Agent Evaluation
- URL: https://mlflow.org/docs/latest/genai/datasets/end-to-end-workflow/
- Traces to dataset: select traces in UI or SDK to build an evaluation dataset (Unity Catalog on Databricks; also auto-generates synthetic sets from documents). Replay: mlflow.genai.evaluate runs your predict function over the dataset while capturing traces. LLM judge: built-in judges and custom scorers. Tool calls: trace-level scorers can inspect tool spans; no cost-per-model recommendation. OSS (Apache-2.0) with Databricks managed.

### OpenAI Evals / Datasets / Stored Completions / Distillation
- URLs: https://openai.com/index/api-model-distillation/, https://developers.openai.com/api/docs/guides/evaluation-getting-started
- Stored Completions capture input-output pairs in the dashboard (free). Evals can use stored completions as a source; graders: string check, text similarity, score model, label model, Python. Distill button fine-tunes a smaller OpenAI model on filtered stored completions. Datasets (newer) support graders and human annotation; "trace grading" for agents is mentioned in third-party coverage. Evals platform is being deprecated (read-only Oct 31 2026, shutdown Nov 30 2026). No cross-vendor replay, no per-task cheapest-model recommendation; single-vendor.

### Anthropic Console Evaluate tool
- URL: https://platform.claude.com/docs/en/test-and-evaluate/eval-tool
- Test cases (generated, manual, CSV), run prompt variants, side-by-side output comparison, 5-point human grading, compare Claude models. No trace ingestion, no tool-call grading, no per-task cost report. Free with API usage.

### Google Vertex AI Gen AI Evaluation Service
- URL: https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-agents
- Final-response and trajectory evaluation (trajectory_exact_match, in_order_match, precision, recall) against a reference trajectory, plus model-based metrics. Datasets are user-supplied. No trace ingestion or per-task cost recommendation. Pay per evaluation.

### Azure AI Foundry evaluations
- URL: https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/agent-evaluators
- Built-in agent evaluators: tool call accuracy, task adherence, intent resolution, task completion; pass/fail thresholds. Datasets user-supplied (can be exported from Foundry agent threads). No cross-model per-task cost recommendation.

---

## C. Cost optimization, model selection, distillation from logs

### OpenPipe (CoreWeave)
- URL: https://openpipe.ai (acquired Sept 2025)
- Captures prompt logs via SDK proxy, curates, fine-tunes smaller open models (and RL), evaluates fine-tune vs. original with LLM judge, deploys. Closest historical "data flywheel" product. It grades fine-tune vs. teacher on captured logs, not a set of off-the-shelf cheaper models per task; tool-call correctness not a packaged grader; output is a replacement model, not a per-task routing table.

### Fireworks AI
- URL: https://fireworks.ai/blog/Turning-Production-Logs-into-Evaluation-Datasets
- Training API (SFT, GRPO, distillation), Batch API. The linked post is a how-to: pull traces from Langfuse or Braintrust, embed, UMAP plus HDBSCAN clustering, stratified sample, run through Eval Protocol. Dataset curation, not an automated product loop; no cross-model per-task report.

### Together AI
- Fine-tuning and evaluations APIs. No trace-ingestion product loop documented.

### Predibase
- Reinforcement fine-tuning from as few as 10 examples, fine-tuning leaderboard comparing SLMs to GPT-4 across 30 tasks. Bring your own data; no trace ingestion or per-task off-the-shelf model recommendation.

### Inference.net Train
- URL: https://inference.net/products/train/
- Uses production traces captured by Inference.net Observe (their own gateway) to distill specialized models via NeMo Megatron Bridge. Same-vendor loop, trains rather than compares off-the-shelf models, no per-task routing report.

### Datawizz
- URL: https://datawizz.ai ($12.5M raise, Sept 2025)
- Gateway that logs every inference, tags and human feedback, trains "specialized language models" about 1,000x smaller, and routes by metadata tags (task=summarize to model X). Routing rules are hand-configured; savings claims 85 percent. No documented automated per-task "which existing cheaper model passes" report.

### Distil Labs
- URL: https://www.distillabs.ai
- Distills an SLM from a prompt plus a few dozen examples; can use docs, logs and tickets to guide synthetic data. Trains, does not evaluate off-the-shelf alternatives per task.

### Kiln AI
- URL: https://kiln.tech
- Free desktop app (OSS core). Evals V2 with a Compare screen: multiple run configs (model, provider, prompt, fine-tune) scored against the same evals with cost and latency side by side; tool-use evals verify right tool and parameters across the trace; synthetic data and fine-tuning. Inputs come from synthetic generation, manual entry or imports, not production trace ingestion. The compare view is the closest packaged "per-task model comparison with cost" table found, minus the trace-driven front end and minus an automatic recommendation.

### Pulze.ai
- Intent-tuned router, custom routers, KNN router (Go, OSS). Router experiments before integration; no trace-to-eval loop.

### Nomos
- No LLM routing or eval product found under this name in 2026 (nomos.sh is an agent marketplace). Treat as not a competitor.

### Parea AI (YC S23)
- Evals, observability, prompt playground. Still maintained; manual datasets; no per-task routing report.

### Lytix
- Observability plus evaluation; Optimodel OSS multi-provider client with guardrails; "guaranteed cheapest LLM calls" via provider switching. No trace-to-eval replay loop.

### Undrstnd
- Cheap fast inference API and cost monitoring for Llama-class models. Not a competitor to the loop.

### TensorZero
- URL: https://github.com/tensorzero/tensorzero (Apache-2.0)
- Self-hosted gateway that stores every inference and feedback in ClickHouse, builds datasets from them, runs inference and workflow evaluations with heuristics or LLM judges, runs adaptive A/B tests across variants (prompt plus model) per "function", and offers optimization recipes (SFT, RLHF, DICL, mixture-of-N). Tool calls are first-class in the gateway schema. It does not automatically generate a per-function "cheapest passing model" report; you define variants and read the A/B and eval results. Paid "Autopilot" is the managed automation layer. Closest open-source substrate for building the loop.

---

## D. Recent startups (2025 to 2026) explicitly near "eval from traces then route"

### Understudy Labs (YC S26)
- URL: https://understudylabs.com, https://www.ycombinator.com/companies/understudy-labs
- Positioning: "We watch your agent work, then train a smarter and cheaper successor." Captures production traces, turns them into evals (bring your suite or build one from traces, with human review), runs candidate models on the same workload (site mentions Claude Sonnet vs. Qwen3-8B on one route), post-trains open-weight specialists, and promotes a successor only when it beats the incumbent on the held-out eval. Dashboards show baseline vs. candidate cost, quality and latency. Explicit tool-call adapters and "tool contract" testing. Delivered via CLI, MCP server, skills and a local workbench; hosted infra optional. Private preview, pricing undisclosed, claims 80 percent Anthropic bill reduction.
- Relative to the loop: closest match found. Differences: emphasis is on training an owned model rather than reporting which off-the-shelf model is good enough per task; eval construction involves human review; not self-serve; the "per task" granularity is per workload or route rather than a table of tasks.

### Riften (YC S26)
- URL: https://riften.ai
- OpenAI and Anthropic compatible gateway that routes each request to the lowest-cost model that can do it, mostly hosted open-weight models; "real task results improve routing and train private models". Routing quality learning is online and opaque; no documented eval dataset from traces, no per-task report to the customer, no tool-call grading described.

### Laminar (YC S24): see section B. Observability with dataset-from-trace; no routing recommendation.

### Others glanced: Embedder (YC S25), Kosmoy, Adaline, Ambertrace, RelayPlane, ResultantAI Gateway, Orca Router. All are gateways, cost dashboards or observability; none document the full loop.

---

## Closest matches, ranked by proximity to the exact loop

1. Understudy Labs (YC S26, private preview). Traces in, evals built from traces, candidate models compared on the same workload with a quality gate, tool-call handling, cost vs. quality dashboard. Gaps vs. the loop: the product goal is to replace the incumbent with a trained open-weight specialist rather than to output a per-task "good enough" table of existing cheaper APIs; human review in eval building; not self-serve.
2. TensorZero (OSS). Has every primitive: inference and feedback store, datasets from inferences, LLM-judge evaluations, tool calls in schema, per-function variants and adaptive A/B tests, cost visible. Missing: automatic dataset construction from traces, packaged tool-call correctness grader, and an automatic per-function recommendation report. You assemble the loop yourself.
3. Braintrust. One-click logs to datasets, playground matrix across models with scorers, Loop proposes scorers and rows, remote evals for multi-step agents. Missing: automatic dataset synthesis, packaged tool-call grader, and any automatic cheapest-passing-model output; their own docs describe it as a manual iteration.
4. OpenPipe (CoreWeave) and Inference.net Train. Capture logs through their proxy, curate, distill a cheaper model, judge against the teacher. Single-direction (train, do not compare off-the-shelf models), single-vendor serving, no per-task table, tool-call grading not packaged.
5. Kiln AI (compare screen with cost and tool-use evals) tied with Langfuse / LangSmith / Datadog / Phoenix (datasets from traces plus experiments across models with LLM judge). All require manual dataset curation and manual reading of the comparison; none emit a per-task routing recommendation with savings.

Honorable mention: Not Diamond custom routers and Prompt Adaptation produce a router or model-prompt recommendation from evaluation data, but the customer must produce the candidate responses and scores; nothing ingests traces.

---

## Gaps: what nobody does today (as of Aug 2026)

- Automatic eval-set construction from raw traces. Every observability tool requires a human to pick traces or write a selection query. Fireworks published a clustering recipe as a blog post, not a product. Understudy describes it with human review in the loop.
- Replay of agent trajectories, not just single prompts. Playgrounds (Braintrust, Phoenix Span Replay, Langfuse, Helicone, Datadog) replay one LLM call. Replaying a multi-step trajectory step by step against a cheaper model, with the original tool results substituted in (teacher-forced replay), is not a packaged feature anywhere found.
- Tool-call correctness as a first-class grader on replayed logs. Tool-call metrics exist in DeepEval, Ragas, Vertex, Azure Foundry, Maxim, Kiln and Phoenix, but they grade against a reference trajectory the user supplies; none derive the reference from the production model's own tool calls in the trace automatically.
- A per-task "good enough" report with dollars. No product outputs "task cluster X: model Y passes at 96 percent of incumbent at 12 percent of cost; task cluster Z: keep frontier." Braintrust and Kiln show the matrix and leave the reading to you; routers pick per request and never tell you why or per task; LangChain's Switchyard benchmark did this once on public tasks.
- Vendor-neutral input. OpenAI Stored Completions, Inference.net Observe, Datawizz and Riften only see traffic through their own gateway. Nothing found ingests an export from Langfuse, LangSmith, Braintrust, Datadog or raw provider logs and produces routing advice without requiring a proxy switch first.
- Routing recommendation decoupled from a hosted router. Every "learn from your data" router (Not Diamond, Orq, Riften, LiteLLM adaptive) wants to sit in the request path. A one-off analysis deliverable ("here is your routing table and projected savings") that the customer implements in whatever gateway they already run does not exist as a product.
- Off-the-shelf comparison before distillation. Distillation vendors (OpenPipe, Inference.net, Datawizz, Distil Labs, Predibase) jump to training. None first check whether an existing cheaper API already passes the task, which is faster and has no serving obligation.
