# tau3-bench, 2026 agent environments, and what a good environment looks like

Research report R22, 2026-08-27. Source: web research agent (WebSearch plus direct fetches of GitHub raw files, arXiv abstracts and HTML, vendor blogs).

Builds on R01 (grading mechanics), R09 (environment synthesis), R12 (TauForge-style pipelines). Those are not repeated here; where a fact from them is needed for context it is cited in one line.

## Coverage caveats

What worked: Sierra tau3 blog, tau2-bench GitHub releases (v1.0.0, v1.0.1), CHANGELOG, evaluation docs, evaluator source, environment replay source, banking_knowledge environment source and task JSON, the taubench.com task-fix blog, tau-Knowledge (arXiv 2603.04370) and tau-Voice (arXiv 2603.13686) full text, Amazon tau2-bench-verified FIXES.md, AGI-Eval tau2-bench-revised airline task_revisions.md, Anthropic "Demystifying evals for AI agents", Halluminate Westworld blog, Surge CoreCraft paper and blog, Mercor APEX-Agents blog and HF dataset card, Scale MCP-Atlas paper and blog, Toolathlon README, AgentGym2, OlaBench, MCP-Persona, ToolOmni, RealUserSim, "Simulated Customers Never Walk Away", "When Simulation Lies", "Lost in Simulation", "Benchmarking the Benchmarks" (both the tool-calling audit 2607.02577 and the conversational-benchmark paper 2608.06329), Hack-Verifiable Environments, Hardening Agent Benchmarks, Fuzzing RLVR Verifiers, Grounded Scaling, BenchScope, Mechanize "Cheap RL tasks will waste compute", OpenEnv governance post.

What failed or returned nothing usable:
- `andonlabs.com/evals/vending-bench-2` returned HTTP 403 (used the Epoch AI mirror and X posts instead; no arXiv paper for v2 found).
- `openai.com/index/trustworthy-third-party-evaluations-foundations` returned HTTP 403 (used secondary coverage; quotes below are second-hand and marked as such).
- `unsloth.ai/blog/rl-environments` returned HTTP 403.
- `tau2-bench/docs/knowledge-retrieval.md` does not exist at that path (404); the read-allowlist mechanism was recovered from `environment.py` instead.
- The banking_knowledge `tasks.json` is 1.1 MB; the fetch tool only surfaced 28 task objects, while the release notes say 97. Treat the two sample tasks below as representative, not the count.
- No "BFCL v5" exists as of this date; the leaderboard is still V4. No "AppWorld 2" exists; AppWorld's last release is a dependency fix (2026-02-17) with "a larger release planned". No CRMArena release after 2025-10 was found. No 2026 update to TheAgentCompany or SWE-Lancer was found. Gaia2/ARE has no scenario or verifier changes documented since the 2025-10-02 leaderboard post.
- Fleet, Matrices, Kaizen: no primary-source statement on environment quality was found; only directory sites and press. Mechanize's blog index returned titles only; one post was fetched directly. Prime Intellect's Environments Hub post contains no quality criteria.
- Vendor "what makes a good environment" checklists in the strict sense exist only from Anthropic, Halluminate, Surge, Scale, and Mechanize. Everything else is inferred from what builders did (audits, fixes, filters).

---

## 1. tau3-bench (Sierra, March 2026)

### 1.1 What it is

tau3-bench is not a new repository. It is tau2-bench release 1.0.0 (2026-03-18) plus a grading-fix release 1.0.1 (2026-07-15), in `sierra-research/tau2-bench`. The blog framing:

```
Sierra is releasing τ³-Bench, expanding agent evaluation to two new frontiers: knowledge retrieval and voice — the real-world conditions where agents are most likely to break.
```
https://sierra.ai/blog/bench-advancing-agent-benchmarking-to-knowledge-and-voice

Three components, from the README:

```
1. Knowledge Domain – A customer service domain focused on knowledge retrieval with "configurable RAG pipelines, document search, embeddings, and agentic shell-based search."
2. Voice Full-Duplex – "End-to-end voice evaluation with realtime providers (OpenAI, Gemini, xAI)"
3. Task Quality Improvements – Over 75 task refinements across domains, addressing "incorrect expected actions, clarified ambiguous instructions, fixed impossible constraints, and added missing fallback behaviors."
```
https://github.com/sierra-research/tau2-bench

Domains now: mock, airline, retail, telecom, banking_knowledge. Papers: tau-Knowledge https://arxiv.org/abs/2603.04370, tau-Voice https://arxiv.org/abs/2603.13686. Leaderboard: http://taubench.com/#leaderboard.

### 1.2 tau-Banking (the knowledge domain)

Abstract, verbatim:

```
We introduce τ-Knowledge, an extension of τ-Bench for evaluating agents in environments where success depends on coordinating external, natural-language knowledge with tool outputs to produce verifiable, policy-compliant state changes. Our new domain, τ-Banking, models realistic fintech customer support workflows in which agents must navigate roughly 700 interconnected knowledge documents while executing tool-mediated account updates. Across embedding-based retrieval and terminal-based search, even frontier models with high reasoning budgets achieve only ∼25.5% pass rate, with reliability degrading sharply over repeated trials.
```
https://arxiv.org/abs/2603.04370

Size: 97 tasks, 698 documents (~195K tokens), 21 product categories, 12 retrieval configurations (BM25, grep, embedding retrievers, sandboxed shell). Repo layout: `data/tau2/domains/banking_knowledge/{db.json 270KB, tasks.json 1.1MB, tasks_voice.json 2.6MB}`; code in `src/tau2/domains/banking_knowledge/{tools.py 189KB, retrieval.py, data_model.py, db_query.py, environment.py}`.

How the knowledge base was built (structured first, then rendered to prose):

```
We first construct a structured knowledge base using LLMs. This process begins by generating a set of business categories (e.g., credit cards, savings accounts), followed by generating features within each category.
```
```
An LLM then allocates subsets of variables to document titles in which they would plausibly appear. Finally, each document title and its associated variables is passed to an LLM, which generates a natural-language article.
```
```
During generation, each feature is defined independently in the structured database, and interactions between features are introduced only when required by downstream tasks.
```
```
Some manual editing is also performed to improve clarity and realism.
```
https://arxiv.org/html/2603.04370

How tasks were built and validated:

```
The tasks and the database were co-constructed manually with LLM assistance to mirror common flows for fintech customer service, such as ordering replacement cards, disputing transactions, and recommending accounts.
```
```
Each task includes a minimal list of documents required to complete the task (gold documents).
```
```
All tasks and associated gold document sets were independently audited by two reviewers who were not involved in task creation.
```
```
Reviewers manually simulated at least one valid trajectory per task to confirm solvability under the gold condition. After large-scale experiments were conducted, all trajectories were re-audited to ensure that no unintended shortcuts, specification loopholes, or degenerate strategies enabled success without proper reasoning.
```
```
This process was designed to reduce benchmark artifacts and ensure that failures reflect model limitations rather than task design errors.
```
https://arxiv.org/html/2603.04370

Reward:

```
Each task specifies a target database state, and the task reward R:S→[0,1] is determined by whether the agent's sequence of retrieved knowledge, tool invocations, and interactions produces the correct final state in Sdb.
```
```
We evaluate agent performance using the passˆk metric, defined as the probability that a task is successfully completed in all k independent trials.
```
https://arxiv.org/html/2603.04370

User simulator: flow-based rather than free-form.

```
τ-Banking employs a flow-based user simulation. Each task defines a set of conditional rules that prescribe the user's next action ausera based on observable agent actions and/or environment outcomes.
```
```
Portions of the dialogue not governed by explicit flow rules are generated freely by the LLM-based user simulator, preserving linguistic diversity and conversational naturalness.
```
```
We randomly sampled two conversation traces per task (from different agent models) and had two annotators with domain expertise in customer service interactions to label each user utterance as error-free, task-benign, or task-critical. Across 194 annotated trajectories, only 4 contained task-critical user errors.
```
https://arxiv.org/html/2603.04370

Failure analysis categories (share of failures): search inefficiency and making assumptions (~23%), complex interdependencies between financial offerings (~14.5%), failure to respect implicit subtask ordering (~5%), overtrusting user assertions (~4%).

Results: best frontier model (GPT-5.2 with reasoning) ~25% pass; with the gold documents provided in context ("golden retriever" configuration) only ~40%. Sierra: "some models take nine times longer for similar accuracy."

### 1.3 Actual task JSON (banking_knowledge)

Two full tasks from `data/tau2/domains/banking_knowledge/tasks.json` (main, 2026-08). Note the fields that did not exist in tau2: `user_tools`, `required_documents`, and the fact that the reward-bearing write is performed by the *user* (`"requestor": "user"`), not the agent.

```json
{
  "id": "task_001",
  "description": {"purpose": "Task: task_001", "relevant_policies": null, "notes": null},
  "user_scenario": {
    "persona": null,
    "instructions": "You are playing the role of a customer contacting a customer service representative agent. Your character is a management consultant named Sarah Bosch who earns $100,000 annually. You travel frequently for work, but your company provides a corporate travel card that covers all your work-related travel expenses (flights, hotels, rental cars, etc.). \n\nYou're looking for a credit card to use for your everyday purchases. You want the card that gives you the highest cash back available in the company profile. You will not accept a credit card that has any annual fees unless it is the ONLY option available. \n\nIf you find out that there are no credit cards that fit your needs, you are happy to immediately take your business elsewhere and end the conversation.\n\nYou have access to a tool that allow you to apply for credit cards by specifying the card type. You're seeking advice on which personal credit card would be the best fit for your situation and spending patterns.\n\nYou receive a Rho-Bank+ subscription for free through your company. ONLY MENTION THIS if you are asked about this.\n\nAfter you receive enough information to make a decision, immediately apply for a credit card on your own, and there is no need to respond to the agent, or ask for instructions on how to apply.\n\nYou are currently on the line with a customer support agent. Only provide additional details about your situation when the agent asks for them. Don't dump all your information at once. Never respond as a customer service representative/assistant. You are playing the role of the customer."
  },
  "initial_state": null,
  "evaluation_criteria": {
    "actions": [
      {"name": "apply_for_credit_card",
       "arguments": {"card_type": "Gold Rewards Card", "customer_name": "Sarah Bosch", "annual_income": 100000, "rho_bank_subscription": true},
       "requestor": "user", "action_id": "001_0"}
    ],
    "communicate_info": [],
    "reward_basis": ["DB"]
  },
  "annotations": null,
  "user_tools": ["apply_for_credit_card"],
  "required_documents": ["doc_credit_cards_gold_rewards_card_001", "doc_credit_cards_silver_rewards_card_001", "doc_credit_cards_bronze_rewards_card_001", "doc_credit_cards_platinum_rewards_card_001"]
}
```

```json
{
  "id": "task_002",
  "user_scenario": {
    "persona": null,
    "instructions": "You are playing the role of a customer contacting a customer service representative agent. Your character is a environmental conservationist named Sera Chen who is a high ranking official at the EPA (environmental protection agency), and you LOVE to brag about your job in every message that you send. Your annual income is $250,000. You do NOT have a Rho-Bank+ subscription.\n\nYou're looking for a credit card to use for your everyday purchases. You want the card that gives you the highest cash back available in the Rho-Bank profile: make this extremely upfront. You are willing to pay up to an effective amount of 100 dollars a year in fees to fund the credit card. You plan to use the card on at least 50,000 dollars worth of purchases each month. DO NOT bring this up unless asked.\n\nIf you find out that there are no credit cards that fit your needs, you are happy to immediately take your business elsewhere and end the conversation.You take recommendations very strongly. If you are recommended a particular credit card, then you take that recommendation. If you are offered many options, ask for the customer support agent to provide only one option so that they may make the decision for you.\n\nYou have access to a tool that allow you to apply for credit cards by specifying the card type. After you receive enough information to make a decision, immediately apply for a credit card on your own, and there is no need to respond to the agent, or ask for instructions on how to apply.\n\nYou are currently on the line with a customer support agent. Don't dump all your information at once. Never respond as a customer service representative/assistant. You are playing the role of the customer."
  },
  "initial_state": null,
  "evaluation_criteria": {
    "actions": [
      {"name": "apply_for_credit_card",
       "arguments": {"card_type": "Platinum Rewards Card", "customer_name": "Sera Chen", "annual_income": 250000, "rho_bank_subscription": false},
       "requestor": "user", "action_id": "002_0"}
    ],
    "communicate_info": [],
    "reward_basis": ["DB"]
  },
  "user_tools": ["apply_for_credit_card"],
  "required_documents": ["doc_credit_cards_platinum_rewards_card_001", "doc_credit_cards_platinum_rewards_card_002", "doc_credit_cards_platinum_rewards_card_007", "doc_credit_cards_gold_rewards_card_001", "doc_credit_cards_silver_rewards_card_001", "doc_credit_cards_bronze_rewards_card_001"]
}
```
https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/banking_knowledge/tasks.json

Other fields seen across tasks: `compare_args` inside actions (which argument keys the ACTION check compares), `initial_state.initialization_data`, `initial_state.message_history`, `description.notes` for complexity markers.

Observations for us:
- The reward basis is `["DB"]` only, and the DB write is the user's. The agent is graded purely on whether it caused the customer to end up with the right card. That is "verdict on end state" in its purest form: the agent's words are graded only by their effect on the simulated user's action.
- Persona quirks ("LOVE to brag about your job in every message") are deliberate noise, not grading criteria.
- Hidden information with disclosure rules ("ONLY MENTION THIS if you are asked") is how a task forces the agent to ask the right question. This is the tau-bench way of testing "what the user was asked".

### 1.4 Reward code (tau2-bench main)

Composite reward is a product of components (`src/tau2/evaluator/evaluator.py`):

```python
reward = 1.0
# ... then for each applicable basis type:
reward *= env_reward_info.reward
reward *= action_reward_info.reward
reward *= communicate_reward_info.reward
reward *= nl_reward_info.reward
```

DB check builds the gold state by replaying the reference actions and compares hashes (`src/tau2/evaluator/evaluator_env.py`):

```python
golden_actions = task.evaluation_criteria.actions or []
for action in golden_actions:
    try:
        gold_environment.make_tool_call(
            tool_name=action.name,
            requestor=action.requestor,
            **action.arguments,
        )
    except Exception as e:
        logger.warning(
            f"Error in golden actions {action.name}({action.arguments}): {e}"
        )

agent_db_hash = gold_environment.get_db_hash()
user_db_hash = gold_environment.get_user_db_hash()
predicted_agent_db_hash = predicted_environment.get_db_hash()
predicted_user_db_hash = predicted_environment.get_user_db_hash()
agent_db_match = agent_db_hash == predicted_agent_db_hash
user_db_match = user_db_hash == predicted_user_db_hash
```

The predicted environment is reconstructed by replaying the full message history, with a `strict` flag added in 1.0.1:

```python
def calculate_reward(
    cls,
    environment_constructor: Callable[[], Environment],
    task: Task,
    full_trajectory: list[Message],
    solo_mode: bool = False,
    env_kwargs: dict = None,
    strict_replay: bool = True,
) -> RewardInfo:
```
```
Set False when re-grading historical trajectories whose recorded tool outputs may cosmetically differ from current tool code.
```

`Environment.set_state` (`src/tau2/environment/environment.py`), docstring verbatim:

```python
def set_state(
    self,
    initialization_data: Optional[InitializationData],
    initialization_actions: Optional[list[EnvFunctionCall]],
    message_history: list[Message],
    strict: bool = True,
):
    """
    Set the state of the environment given initialization data and a list of messages.

    Args:
        strict: When True (default), raise if a replayed mutating tool call
            returns different content than the recorded ToolMessage. When
            False, log a warning instead and continue the replay. Lenient
            mode is intended for re-grading historical trajectories whose
            recorded tool outputs contain cosmetic drift against current
            tool code (e.g. numeric argument echoes rendered as `25` by the
            code that produced them but `25.0` after numeric-argument
            normalization); the state mutation is applied identically
            either way.
    """
```

Replay behaviours in that method: unknown tool names are skipped ("Skipping unknown tool '{tool_call.name}' during replay (no-op, matching live env behavior on hallucinated tools)"); non-mutating tools are skipped during reconstruction; on output mismatch strict raises `ValueError`, lenient logs a warning.

Read logging in banking_knowledge (`src/tau2/domains/banking_knowledge/environment.py`): `get_environment()` takes `read_log_allowlist`, documented as:

```
Names of read-only discoverable tools whose calls should be logged to `agent_discoverable_tools` for eval.
```
```
typically derived from the task's golden trajectory so that required-read assertions still discriminate, while extra validation reads don't pollute the DB hash comparison.
```
https://raw.githubusercontent.com/sierra-research/tau2-bench/main/src/tau2/domains/banking_knowledge/environment.py

Evaluation doc (unchanged in substance from tau2): "any sequence of tool calls that produces an equivalent DB end state passes the DB check"; ACTION is only enforced when listed in `reward_basis`, otherwise actions are diagnostic; COMMUNICATE is substring match; NL_ASSERTION is an LLM judge marked experimental. https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md

### 1.5 The 75+ task fixes: what kinds of bugs

Release 1.0.0 notes:

```
Airline (27 tasks): Removed incorrect expected actions, clarified ambiguous instructions, fixed impossible constraints, closed policy loopholes, added missing fallback behaviors.
Retail (26 tasks): Removed invalid expected actions (e.g., unsupported PayPal refunds), clarified ambiguous instructions, fixed impossible same-item exchanges, added fallback behaviors.
```
https://github.com/sierra-research/tau2-bench/releases/tag/v1.0.0

CHANGELOG 1.0.0 says the total including banking was larger: "Over 100 tasks corrected across domains including incorrect expected actions, ambiguous user instructions, impossible constraints, policy loophole prevention, missing fallbacks, and data corrections."

The taubench.com post gives the five categories with examples (source of the fixes: "τ-Bench Verified (Cuadron et al., SABER)", Amazon's audit, plus community PRs including Anthropic's):

```
1. Incorrect Expected Actions
Airline tasks 2, 27, 38: Tasks expected the agent to offer compensation for delayed flights, but the policy only allows compensation for silver/gold members, insured passengers, or business flyers.
Airline tasks 9, 37: Expected cancellations violating policy (not within 24 hours, wrong ticket class)
Retail tasks 12, 13: Expected actions included refunding to PayPal, which isn't a supported refund method

2. Ambiguous User Instructions
Airline tasks 15, 16: Confusion between "economy" vs. "basic economy"
Retail tasks 0, 1: Changed "similar keyboard" to "the same one" for clarity
Retail task 76: Missing cancellation reasons
Retail tasks 2, 3, 4: Added precision to quantity requests

3. Impossible Constraints
Airline task 14: Required Mastercard when user profile had none
Airline task 42: Address contradicted scenario location
Retail tasks 18, 91: Tasks expected exchanging an item for the exact same item (same SKU), which the retail system doesn't support

4. Missing Fallback Behaviors
Retail task 62, 54, 20: No user guidance when search results or modifications failed

5. Policy Loophole Prevention
Airline tasks 45, 13, 29: Agents could exploit workarounds (cabin upgrades, cancel-and-rebook, destination changes)
```
Score effect: "pass^1 increased by +14.0 to +20.0 points depending on the model" (airline); retail "-0.4 to +5.5 points on pass^1"; "pass^4 improvements were often even larger than pass^1 improvements in airline."
https://taubench.com/blog/tau3-task-fixes.html

The Amazon audit (tau2-bench-verified FIXES.md) is the richer primary source. Categories and approximate counts: policy compliance ~18, database accuracy ~12, instruction clarity ~15, logical consistency ~8, evaluation determinism ~10; retail ~43, airline ~20. Twelve verbatim examples:

```
Task 2 (Airline - Noah Muller). Problem: Agent offered $50 certificate without user wanting to cancel/modify reservation. Fix: Added instruction that user will 'never change or cancel the reservation' and removed compensation action.
Task 17 (Retail - Mei Davis). Problem: Exchange target item 8069050545 was identical to existing item (blue leather chair). Fix: Changed new_item_ids to 3609437808 (red leather variant) to comply with requirement that "exchanges must be for different product option."
Task 26 (Retail - Isabella Johansson). Problem: Ambiguous what user expects when single-item cancellation fails. Fix: Added explicit instruction: "if removing one item is not possible, cancel the whole order" plus sequence specification for which orders to cancel first.
Task 9 (Airline - Aarav Ahmed). Problem: Attempted to cancel reservation NQNU5R with flights on 2024-05-13/14 when current time was 2024-05-15. Fix: Removed cancel action and updated assertion to "Agent does not cancel NQNU5R. Flights that have already departed cannot be cancelled."
Task 80 (Retail - Liam Thomas). Problem: Multiple payment methods available; unclear which to use for different operations. Fix: Added "you want to use your credit card ending in 3194 for the pending order and paypal for the return."
Task 45 (Retail - Earbuds). Problem: Original new_item_ids: ["1646531091"] ($232.49) ignored financial constraints. Fix: Changed to cheaper option 8555936349 ($226.49) given user's stated financial situation.
Task 59 (Retail - Yusuf Taylor). Problem: Agent couldn't identify older pending order without explicit help. Fix: Added hidden info: "one was placed much earlier in the year (W8268610, do not reveal this number to the agent)."
Task 84 (Retail - Yara Ito). Problem: Exchange target 1615379700 matched existing item (size 10, synthetic boots). Fix: Changed to 8106223139 (size 9, leather) and added fallback: "if agent doesn't allow exchange, get hiking boots with size 9 made of leather."
Task 31 (Airline - Ivan Rossi). Problem: Unclear that basic economy cannot be modified directly. Fix: Added instruction requiring separate steps: "first upgrade you to economy and confirm that change, then separately change the flights to nonstop."
Task 23 (Airline - Multiple Bookings). Problem: Passenger names didn't match database records. Fix: Changed first_name from "Aarav" to "Raj" (Sanchez) and "Evelyn" to "Liam" (Wilson).
Task 34 (Airline - Yara Garcia). Problem: Unclear whether budget applied to individual or combined changes. Fix: Added clarification: "you want all these changes as a complete package...if total cost exceeds budget, cancel entire request - do not accept partial changes."
Task 25 (Airline - Ivan Muller). Problem: User might waste certificate on low-cost booking. Fix: Added instruction: "use your certificate if price is higher than $400, otherwise MUST use gift card and credit card instead."
```
https://github.com/amazon-agi/tau2-bench-verified (FIXES.md)

A third independent audit (AGI-Eval `tau2-bench-revised`, adopting the fixes described in the Claude Opus 4.5 release report) lists 44 airline revisions. Recurring problem types there: eligibility not established before an expected action (compensation, cancellation), wrong dates/DOBs/names/fees in the gold, ambiguity between modification and new booking, missing bundling ("all changes as a package"), missing consent statement, health-claim honesty not specified, "Emphasize adherence to airline policy over tool capabilities" (task 39). It also changed the retail and telecom user simulator guidelines "to avoid failure modes from premature task termination, especially for transfer handling scenarios" (from the GLM-4.5 report). https://raw.githubusercontent.com/AGI-Eval-Official/tau2-bench-revised/main/data/tau2/domains/airline/task_revisions.md

Reading the three audits together, the bug taxonomy for a tau-style task is:

| Bug class | What it looks like | Which grader it breaks |
|---|---|---|
| Gold violates policy | expected compensation/cancel/refund the policy forbids | DB target is wrong; correct agents fail |
| Gold violates data | wrong item ID, passenger name, DOB, date, fee, same-SKU exchange | DB target unreachable |
| Impossible scenario | card the user does not have, already-departed flight, address contradiction | no valid trajectory |
| Underspecified user | which payment method, "similar" vs "same", quantity, bundle vs partial, budget scope | end state is nondeterministic across valid agents |
| Missing fallback | what the user does when the first request is refused or search fails | user sim improvises, end state diverges |
| Missing disclosure rule | which facts the user volunteers vs only on request | agent that asks the right question is not rewarded |
| Policy loophole | cancel-and-rebook, upgrade to unlock a change | DB reaches target via a path the policy forbids |
| Answer leakage | fee descriptions that state the refund amount (banking 1.0.1) | task tests reading, not reasoning |
| Grader artefact | extra reads logged into hashed table; 25 vs 25.0 hash mismatch; insertion vs documented ordering | false negatives unrelated to agent |

### 1.6 Release 1.0.1: grading bugs found after release (July 2026)

This release is the clearest public record of end-state grading failing for reasons unrelated to agent behaviour:

```
Every `call_discoverable_agent_tool` call was logged to a DB table that participates in the reward's DB-hash comparison
```
```
Grading change: `banking_knowledge` scores are not comparable across this release.
```
CHANGELOG bullets verbatim:
```
- Extra read calls no longer zero rewards; logging now uses per-task allowlists for required reads
- DB hashes no longer depend on numeric argument spelling (25 vs 25.0)
- Bank account transactions now returned in reverse chronological order per documentation
- Contradictory cash-back rate corrected in Platinum Rewards document
- Task 074: Light Blue ATM-fee refund now correctly honors two free out-of-network and two free foreign withdrawals monthly ($14.50 instead of $8.00)
- Answer-leaking fee descriptions replaced with neutral language across multiple tasks
- Re-scoring now applies per-task read allowlists and replays leniently for cosmetic tool output differences
- Hallucinated tool calls treated as no-ops during replay, preventing incorrect infrastructure errors
```
Also: "Tasks 077–086 (lost/stolen card scenarios) made agent-realizable by including account-listing reads in golden trajectories" and a new `tau2 evaluate-trajs --fresh-tasks` flag to re-grade stored trajectories against current task definitions. Re-grading moved scores "+0.47 to +9.02 pass@1 points across models, with zero downward score movement".
https://github.com/sierra-research/tau2-bench/releases/tag/v1.0.1 and CHANGELOG.md

### 1.7 tau-Voice: user simulator changes

```
A controllable and realistic voice user simulator provides diverse accents, realistic audio environments, and rich turn-taking dynamics; by decoupling simulation from wall-clock time, the user simulator can use the most capable LLM without real-time constraints.
```
```
Each tick, both parties exchange exactly τ ms of audio, enabling true full-duplex interaction where both can speak simultaneously.
```
```
Success is deterministically evaluated by comparing the end state of the environment (e.g., database records) against a gold standard.
```
```
Agent errors dominate: 79% of failures in the Voice-Fragile cohort and 90% in the Noise-Fragile cohort are attributed to the agent rather than the user simulator.
```
https://arxiv.org/html/2603.13686

278 tasks (retail 114, airline 50, telecom 114), tick τ=200 ms, user waits 1 s of silence before responding, LLM decides interruptions and backchannels, seven voice personas, G.711 μ-law telephony, frame drops, background and burst noise. Release notes add a "Hallucination reviewer and automatic retry system" for the simulator, and "LLM-based conversation review and quality checks" plus "per-task summary analysis and diagnostics" as evaluation tooling. Text GPT-5 (reasoning) 85%; voice agents 31 to 51% clean, 26 to 38% with noise and accents. The 79 to 90% failure attribution was done by human review of 91 failed simulations, a practice worth copying.

---

## 2. Other agent environments and benchmarks, released or updated after 2025-10

Only items with something new since October 2025 or not covered in R01/R09/R12. Sorted by relevance to support/enterprise tool use.

| Name (org, date) | State store | Tools | User sim | Task spec | Reward | Size | Env source public |
|---|---|---|---|---|---|---|---|
| tau3 banking_knowledge (Sierra, 2026-03, fixes 2026-07) | JSON DB + 698 docs | Python tools + pluggable retrieval + sandboxed shell | flow-rule LLM user with user tools | user instructions, gold actions (mostly user-side), gold docs | DB hash (product of components), pass^k | 97 tasks | yes (MIT) |
| tau-Voice (Sierra, 2026-03) | tau2 DBs | tau2 tools | tick-based full-duplex voice user, TTS, audio effects | tau2 tasks | same DB-state reward | 278 tasks | yes |
| EnterpriseBench CoreCraft (Surge, 2026-02) | 2,500+ entities, 14 types (customers, orders, tickets, SLAs, Slack) | 23 MCP tools (searchOrders, updateTicketStatus, processReturn, validateBuildCompatibility) | none reported | expert-written prompt | expert rubric, r = fraction of criteria; task pass requires all | 1,000 train + 150 held-out | leaderboard only; no source found |
| APEX-Agents (Mercor, 2026-01) | 33 "worlds", avg 166 files (docs, sheets, PDFs, mail, chat, calendar) | Archipelago apps, code execution | none (single-turn prompt) | prompt + 1 to 10 binary rubric criteria + gold output | judge model per criterion; Pass@1, Pass@8 | 480 tasks (banking, consulting, law) | yes (CC-BY, Archipelago on GitHub; training forbidden) |
| MCP-Atlas (Scale, 2026-02) | live MCP servers, some over real business datasets | 36 servers, 220 tools, 10+ visible per task, 3 to 7 needed | none | natural-language prompt, no server/tool named | claim-level rubric on final answer; 0.75 claim coverage threshold | 1,000 tasks; 500 public, 500 held out | tasks yes; servers are real |
| Toolathlon-Verified (HKUST, 2026-06-30) | per-task Docker container with local apps | 32 apps, 604 tools via MCP | none | manually sourced/crafted prompt | per-task evaluation script on final environment state | 108 tasks | yes |
| AgentGym2 (2026-07) | "de-idealized" real digital environments | composable toolbox: web, retrieval, files, multimodal, code | none | end-to-end scenarios with noisy, underspecified inputs; tools must be discovered | not stated in abstract | not stated | yes (AgentGym family) |
| Vending-Bench 2 (Andon Labs, 2025-11) | simulated business: inventory, orders, pricing, suppliers (LLMs), daily fees | email/ordering tools | LLM suppliers and customers | run a business for 365 simulated days | single metric: end balance in USD, avg of 5 runs; human reference ~$63k | one long-horizon task | leaderboard public; source not public |
| OlaBench / OlaMind (ByteDance, 2025-10 v3 2026) | real industrial customer-service data, de-identified | RAG, workflow, agent settings | none (static dialogues) | 3,000 core dialogues + 1,000 risk + 1,000 hallucination | six dimensions incl. policy compliance, tool calling, critical business risk, hallucination, latency; 5,000+ expert labels | ~3,600 instances | data: unclear |
| JourneyBench (2026-01) | graph representation of support scenarios | policy-driven flows | scenario generator | 703 conversations, 3 domains | User Journey Coverage Score (policy adherence) | 703 | paper only |
| MCP-Persona (2026-06) | code simulators of Reddit, Slack, Lark, Xiaohongshu built from real traces | LLM-generated Python simulators with context handlers | none | 173 human-verified tasks, fuzzified instructions | checkpoint scoring (0/0.5/1, LLM judge, 91.5% human agreement) plus execution checks on context state | 173 | yes |
| RealUserSim (2026-05) | n/a (user side) | n/a | 7,275 behavioural profiles mined from 14k WildChat conversations | plugs into existing benchmarks | agent success drops 3.2 to 3.5 points vs cooperative sims | 7,275 profiles | yes |
| OpenEnv (Meta-PyTorch + HF, multi-org since 2026-06-08) | any | any | any | Gymnasium `reset()/step()/state()` in Docker | deliberately not specified | hub | yes |

Notes per item.

MCP-Atlas: "Prompts do not specify servers, tools, or parameters, requiring agents to identify relevant tools among semantically plausible distractors and to compose multi-step, cross-server workflows." "Each task is scored with a claim-level rubric, where final answers are scored against atomic factual claims grounded in tool outputs. This answer-centric scoring permits valid alternative tool-call trajectories to receive credit". "63.3% of diagnosed failures are cognitive rather than tool-call related." Held-out half kept "to preserve leaderboard integrity". https://arxiv.org/abs/2602.00933. Scale blog: "Every task is human-written with real-world data", "Our Airtable tasks query real business datasets that Scale has acquired; search tasks pull live results." https://scale.com/blog/mcp-atlas. The 2607.02577 audit later found a 13.5% evaluator-human disagreement rate on MCP-Atlas (12 of 89 audited tasks).

Toolathlon-Verified: "This release marks the verified final version of Toolathlon, with task prompts, ground truths, and evaluators reviewed and aligned for the final benchmark release." "For each task we set up a separate container for execution." Max 5,400 s per task. https://github.com/hkust-nlp/Toolathlon. That every major 2025 benchmark now has a "Verified" edition (SWE-bench Verified, tau2-bench-verified, Toolathlon-Verified, tau3 fixes) is itself a finding: first-release task sets carry a 10 to 20% defect rate.

CoreCraft: "Every entity, tool, and data source exists to support diverse, challenging tasks rather than to maximize world complexity." Rubric axes: "Completeness: Did the agent address all required aspects? Correctness: Are factual claims accurate given the world state? Constraint Satisfaction: Are business rules and policies correctly applied? Format Compliance: Does the output follow required structure?" "Task-level pass rates require all rubric criteria to be satisfied, providing a strict measure of end-to-end task completion." Reward "r = (1/|C|) Σ 𝟙[criterion c satisfied]". Training GLM 4.6 for one epoch: 25.37% to 36.76% on held-out CoreCraft, and transfer +4.5 BFCL Parallel, +7.4 tau2 Retail, +6.8 Toolathlon. "direct comparison against degraded environment variants (synthetic tasks, simplified rubrics, reduced entity complexity) would quantify the contribution of each design principle" is listed as future work: no ablation. https://arxiv.org/html/2602.16179v2. Blog: "current RL environments are often too sterile"; failure modes: premature abandonment, hallucinated information, overly restrictive search, executing conditional branches unnecessarily. https://surgehq.ai/blog/enterprisebench-corecraft

APEX-Agents: worlds built by "Vice Presidents, Managing Directors, and Managers with five-to-ten years' experience at top-tier firms" who "worked in Google Workspace, simulating how coworkers would collaborate on a project" over 5 to 10 days per world; "Each task includes 1–10 pass/fail criteria" (mean 4.06); judge model grades each criterion; top Pass@1 24.0%, Pass@8 36.7%. https://www.mercor.com/blog/introducing-apex-agents/ and https://huggingface.co/datasets/mercor/apex-agents

AgentGym2: "most existing benchmarks evaluate agents in simplified, idealized settings. They typically rely on pre-packaged tool interfaces, overlook critical steps, and assume inputs are clean and fully specified. Consequently, they understate the difficulty of real deployments, where uncertainty and noise are ubiquitous and agents must proactively explore the environment to uncover new tools." https://arxiv.org/abs/2607.05174

Vending-Bench 2: "adds real-world messiness such as adversarial suppliers, delayed or failed deliveries, and customer refund demands, and streamlines scoring to a single headline metric"; "We've incorporated learnings from our real AI vending machines." Suppliers are LLMs "who can be jailbroken to give away stuff for free" and "daily sales simulated based on equations that can be gamed" (secondary summaries). https://epoch.ai/benchmarks/vending-bench-2, https://x.com/andonlabs/status/1990810936735641661

OpenEnv: "an interoperability layer for RL environments"; "Reward definition, scoring rubrics, and trainer-specific logic belong in the libraries that specialize in them." Planned "auto-validation" to "measure environment quality and contribution to model learning" but no criteria yet. Steering committee since 2026-06-08: Meta-PyTorch, Reflection, Unsloth, Modal, Prime Intellect, Nvidia, Mercor, Fleet AI, Microsoft, Hugging Face, RadixArk. https://huggingface.co/blog/openenv-agentic-rl. CUBE (arXiv 2603.15798) proposes a parallel "wrap once, use everywhere" standard on MCP plus Gym.

No change since 2025-10: Gaia2/ARE (leaderboard post 2025-10-02 has no scenario or verifier fixes), CRMArena-Pro (22 tasks, 2,140 instances; the "Agentic Benchmark for CRM" is a vendor league table on accuracy, cost, speed, trust, sustainability), AppWorld (dependency-fix release 2026-02-17), BFCL (still V4; BenchScope notes its "20 distinct function-calling capabilities" "collapse into seven effective groups"), Terminal-Bench 2.0 (covered in R01; the Hardening paper found 13 of 89 TB2 environments hackable from the description alone).

---

## 3. What does a good environment look like: explicit statements

### 3.1 Anthropic, "Demystifying evals for AI agents" (2026-01)

```
A good task is one where two domain experts would independently reach the same pass/fail verdict.
```
```
Each task should be passable by an agent that follows instructions correctly...agents shouldn't fail due to ambiguous specs.
```
```
There is a common instinct to check that agents followed very specific steps like a sequence of tool calls in the right order. We've found this approach too rigid...it's often better to grade what the agent produced, not the path it took.
```
```
Graders that check for very specific step sequences result in overly brittle tests, as agents regularly find valid approaches that eval designers didn't anticipate.
```
```
Each trial should be 'isolated' by starting from a clean environment. Unnecessary shared state between runs (leftover files, cached data, resource exhaustion) can cause correlated failures.
```
```
For example, in some internal evals we observed Claude gaining an unfair advantage on some tasks by examining the git history from previous trials.
```
```
Capability or 'quality' evals should start at a low pass rate, targeting tasks the agent struggles with.
```
```
A support agent that correctly identifies the problem and verifies the customer but fails to process a refund is meaningfully better than one that fails immediately.
```
```
Model grading often takes careful iteration to validate accuracy. LLM-as-judge graders should be closely calibrated with human experts.
```
```
20-50 simple tasks drawn from real failures is a great start...this large effect size means small sample sizes suffice.
```
```
You won't know if your graders are working well unless you read the transcripts and grades from many trials...reading transcripts is how you verify that your eval is measuring what actually matters.
```
```
An eval suite is a living artifact that needs ongoing attention and clear ownership to remain useful.
```
https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents

### 3.2 Sierra (tau-Knowledge, tau-Voice, tau3 fixes)

Properties Sierra enforces in practice: gold solvability by manual simulation ("Reviewers manually simulated at least one valid trajectory per task"), independent double audit, post-hoc re-audit of all trajectories for "unintended shortcuts, specification loopholes, or degenerate strategies", user-simulator error labelling with domain annotators (4 task-critical errors in 194 trajectories), flow-rule user simulation for reproducibility, answer-leak scrubbing, per-task read allowlists so validation reads do not fail the DB hash, numeric normalisation, lenient replay for cosmetic drift, human attribution of failures to agent vs simulator. The five fix categories in 1.5 are the negative image of Sierra's task-validity checklist.

### 3.3 Meta ARE / Gaia2 (from R09, restated as properties)

Deterministic given a fixed start; state = app states + simulated time + events; verifier checks every write action with hard checks on IDs and soft LLM checks on free text; scenarios at 0% or 100% pass flagged broken; judge hacking observed during RL, fixed with global sanity checks; 98% verifier agreement on 450 labelled trajectories. No 2026 update.

### 3.4 Halluminate, "Westworld"

```
Reproducibility: every agent sees the exact same state
Determinism: no CAPTCHAs, DOM drift, randomized search results, or data decay
Control: ability to scale difficulty, inject edge cases, log every step, and run RL with hosted sims without noisy/breaking site changes
```
Task templates derived from "real-world production queries"; "task-centric simulation" rather than app-centric; dates "compute dates relative to the actual runtime date"; grading by "RLVR-style unit tests" of three kinds (state-based DB diff, component-level selector checks, real-time ground truth queried from the simulator), all programmatic to avoid "label drift as real websites evolve."
https://www.halluminate.ai/blog/westworld

### 3.5 Surge (CoreCraft)

```
task-centric world building that optimizes for diverse, challenging tasks; expert-authored rubrics enabling reliable reward computation; and enterprise workflows that reflect realistic professional patterns.
```
```
Our results suggest that environment quality, diversity, and realism are key factors enabling generalizable agent capabilities.
```
https://arxiv.org/abs/2602.16179. Claimed, not ablated (see 3.13).

### 3.6 Scale (MCP-Atlas)

Principles named: "Realism," "Scale & Diversity," "Complexity"; distractors ("Each task includes 10+ available tools, but only 3-7 are needed"); answer-centric scoring so alternative trajectories pass; half the tasks held out. https://scale.com/blog/mcp-atlas

### 3.7 Mercor (APEX-Agents)

Domain experts build the world *and* "the exact grading criteria that define what 'client-ready' means"; binary criteria; external domain reviewer (Harvey) for realism. Training on the data is forbidden by licence, which is Mercor's anti-contamination mechanism.

### 3.8 Mechanize, "Cheap RL tasks will waste compute" (2025-08)

```
data and compute are complementary: to achieve high model performance, labs must spend a significant amount on both. Severely underinvesting in one relative to the other wastes most of your investment.
```
More tasks means "each task will on average be lower quality, less informative, and cover a narrower slice of the task distribution." Cost model: "$480 per RL task per run", "$2,400 for each RL task" over five reuses. Good tasks: "debugging complex distributed software systems under production-level load", "operating mock e-commerce businesses", built by "full-time domain experts who spend months developing sustained context and crafting individualized tasks."
https://www.mechanize.work/blog/cheap-rl-tasks-will-waste-compute/

### 3.9 OpenAI, "A shared playbook for trustworthy third party evaluations" (2026-05-29; secondary source, page returned 403)

Validity hazards as summarised by GIGAZINE: scores "artificially reduced if a model recognizes it is being evaluated and strategically underperforms, inflated if the model exploits a shortcut in the task, prompt, scorer, or harness, or distorted by contamination (where a model already knows or can find an answer without solving the task) or by 'broken' problems that are ambiguous, incorrectly scored, unsolvable, or vulnerable to unintended shortcuts." The harness "determines how a model uses tools, tracks information, and recovers from mistakes." Evaluators "should review samples for these behaviors every time an assessment is run." https://openai.com/index/trustworthy-third-party-evaluations-foundations/ via https://gigazine.net/gsc_news/en/20260601-openai-third-party-evaluations/

OpenAI grader docs: "Models being trained sometimes learn to exploit weaknesses in model graders, also known as 'grader hacking' or 'reward hacking.' A model that's hacked the grader will score highly on model grader evals but score poorly on expert human evaluations." https://developers.openai.com/api/docs/guides/graders

### 3.10 Academic audits of benchmark validity

"Benchmarking the Benchmarks: A Validity Audit of Tool-Calling Evaluation" (2607.02577, June 2026):

```
Tool-calling benchmarks are increasingly used to rank language-model agents, yet their scores are often treated as ground truth without validating the evaluators themselves.
```
Per benchmark: BFCL v4 40/200 (20.0%), tau2 Retail 11/112 (9.8%), MCP-Atlas 12/89 (13.5%), LiveMCPBench 29/95 (30.5%); total 92/496 (18.5%). LiveMCPBench: 23 repeated runs of the same setup ranged 57.9% to 76.8%. Two tau2 examples: a false negative where the agent computed remaining costs as $708 but the reference "embedded the canceled reservations' costs ($1,628)" behind a "brittle substring assertion"; a false positive where the agent "transferred to human without invoking the return tool" and passed because "final-state scoring can reward inaction." Recommendations: "Deploy deterministic state gates for factual verification before LLM judging", "Restrict LLM judges to qualitative criteria only", "Implement bounded repair windows with separate success/recovery tracking", "Preserve raw execution artifacts for trace-level inspection", "Achieve reproducibility through versioned, auditable evaluation components." Their Tool-Veritas reaches 95.5% human agreement by decomposing into (C_tool, C_task, C_outcome). https://arxiv.org/html/2607.02577

"Benchmarking the Benchmarks: Evaluating Benchmarks for Conversational Agents" (2608.06329, Aug 2026):

```
Poor benchmarks may contain inconsistent tasks, simplistic scenarios, or limited policy coverage, leading to unreliable evaluations. We introduce a reference-free framework that uses LLM judges to assess benchmark consistency, complexity, and policy coverage, while providing actionable diagnostics of weaknesses.
```
Metrics: "Description–Expected Behavior Alignment" (is expected behaviour "consistent with the task description and adequately addresses the user's intended goals"), "Policy–Expected Behavior Alignment" (does it "comply with the domain policy for the described scenario"), "Policy Violations per Task" (complexity), "Policy Violations Coverage" ("fraction of policy items violated by at least K tasks"). Human agreement Kendall τb 0.32 to 0.67. On tau3 airline (50 tasks): "broadly fall within the range observed for the lower-scoring IntellAgent benchmarks", with policy violations per task 0.38 to 0.60 vs 1.5 to 2.3 for the best generated sets. https://arxiv.org/html/2608.06329v1

"Grounded Scaling: Why Agentic AI Needs Deterministic Environments" (2606.22495):

```
Long-chain agent execution fails exponentially in environments designed for human tolerance: with per-step determinism δ<1, k-step chain success degrades as δ^k.
```
"At δ=0.9 per step, chain success is only 53%; at δ=0.8, it falls to 26%." Five properties: economic self-sufficiency, verifiability, interactivity, scale and multimodal richness, deterministic interface guarantee. Proposition 2: verifier mismatch bounds policy quality by C√(2ε). Evidence base is thin by the authors' own admission. https://arxiv.org/html/2606.22495

"Lost in Simulation" (2601.17087): agent success 45.2% with US humans vs 67 to 75.9% across simulated-user models; "evaluations with simulated users underestimate agent success on the hardest tasks (success with human users: 30.8%) while overestimating it on moderate tasks (success with human users: 39.0%)"; ECE 15.1 overall, 20.3 for AAVE speakers vs 11.7 SAE. Recommendation: "agentic benchmarks should assess robustness across multiple simulation models, validate simulated outcomes against demographically diverse human data if possible, and transparently acknowledge the limitations of user simulation." https://arxiv.org/html/2601.17087

BenchScope (2603.29357): Effective Dimensionality shows many suites report scores with almost no independent information (Open LLM Leaderboard six scores ≈ 1.7 axes). A property of a good eval suite: its sub-scores measure different things.

### 3.11 Reward hacking in environments (2026)

"Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops" (2606.08960): "We audit 1,968 tasks across five terminal-agent benchmarks and find 323 (16%) hackable by frontier models given only the task description." A hacker/fixer/solver loop drove KernelBench attack success "from 62% to 0%"; defences written by Gemini 3 Flash cut Gemini 3.1 Pro's attack success "from 76% to 0%". Seven attack patterns catalogued (input mutation, lazy evaluation, monkey-patched timing, global-state poisoning, result caching by input address, precision downgrade, reference fallback). https://arxiv.org/html/2606.08960

"Hack-Verifiable Environments" (2605.20744): wrap any environment with planted hidden solutions, logical bugs, readable and editable opponent prompts, detected deterministically. Hack rates: gpt-5.4 8.5%, claude-sonnet-4.6 9.5%, gemini-3.1-pro 20.2%, grok-4.1-fast 28.5%; "Hack rates increased monotonically as tasks became harder"; once a model hacks it "almost certainly" repeats within the same context. Design advice: keep realistic underspecification, embed plausible open-ended resources, grade difficulty of the exploit, verify deterministically, test across many environments. https://arxiv.org/html/2605.20744v1

"Before the Model Learns the Bug: Fuzzing RLVR Verifiers" (2606.01066): "if the verifier is wrong, optimization can learn the bug"; framework fuzzes verifiers with adversarial completions and reports false-positive, false-negative, disagreement, exploit, and uncertainty metrics. https://arxiv.org/abs/2606.01066

Earlier, from R09/R12 (not repeated): METR 30.4% hacking with visible scoring code; AgenticAI-Supervisor ~40% constraint misrepresentation under outcome-only rewards; RubricForge false-pass rates 11 to 17%.

### 3.12 Practitioner taxonomy (leehanchung, 2026-03)

E = {Tasks, Harness, Verifier, State, Configuration}. "Programmatic checks such as string match or code execution, are faster, cheaper, and more consistent than LLM-as-judge." "the lowest cost to collect tasks are single-turn with verifiable answers. The most valuable tasks for long-horizon behavior are expensive to construct." Notes Step-DeepResearch injecting 5 to 10% tool errors during training and co-evolving rubrics (RLER) resisting gaming better than static ones. https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/

### 3.13 Property matrix: who names what

| Property | Named explicitly by | Evidence beyond assertion |
|---|---|---|
| Deterministic, resettable, isolated | Anthropic, Halluminate, Grounded Scaling, ARE, Toolathlon (container per task) | Anthropic's git-history leak anecdote; δ^k model |
| Verifiable by code, LLM judge only for qualitative residue | Anthropic, Halluminate, 2607.02577, Sierra (DB hash), ScaleEnv (R12), leehanchung | 18.5% evaluator error where judges and substrings are used; Tool-Veritas 95.5% agreement |
| Grade outcome, not path; accept alternative paths | Anthropic, Sierra, MCP-Atlas, BFCL v3 state check, AppWorld | tau2 "trajectory lock-in" finding; tau3 ACTION only when in reward_basis |
| Two experts agree on pass/fail | Anthropic | tau-Banking double audit |
| Solvable by construction (oracle path exists) | Sierra, Envs-FORGE and CUA-Gym (R12), Anthropic | tau3 "impossible constraints" fixes |
| Unambiguous user with fallbacks and disclosure rules | Sierra, Amazon audit | 15+10 clarity/determinism fixes in tau2-verified |
| Policy closed (no loophole path to the target state) | Sierra | 3 airline loophole fixes; LOGIGEN triggers (R12) |
| No answer leakage into observations | Sierra 1.0.1, Envs-FORGE contract (R12) | fee-description scrub |
| Realism / grounding in real work | Surge, Mercor, Scale, Mechanize, AgentGym2, Halluminate | CoreCraft transfer (+4.5 to +7.4) but no ablation |
| Task-centric world (entities exist for tasks) | Surge, Halluminate | none quantitative |
| Diversity / coverage of policy items | 2608.06329, Surge, Mechanize | tau3 airline scores low on policy-violation coverage |
| Difficulty calibration (drop 0% and 100%) | Anthropic, ARE, Envs-FORGE/SPADE/RLAnything (R12) | pass-rate bands 0.2 to 0.8 standard in synthesis pipelines |
| Distractors (extra tools, similar entities) | Scale, ScaleEnv, LOGIGEN | MCP-Atlas 10+ visible, 3 to 7 needed |
| Noise and tool failures injected | AgentGym2, Vending-Bench 2, tau-Voice, When Simulation Lies, Step-DeepResearch | metadata and transition perturbations cost 30 to 40% |
| Held-out split, rotating set, licence bans on training | Scale, Mercor, Anthropic ("living artifact") | none |
| Robust to reward hacking, verifier fuzzed | Hardening loops, Hack-Verifiable, Fuzzing RLVR, OpenAI | 16% of terminal tasks hackable from description |
| User simulator validated against humans | Lost in Simulation, RealUserSim, Simulated Customers, Sierra (194 traces) | 9-point swing across user LLMs; 45% vs 67 to 76% |
| Read transcripts, attribute failures to agent vs env | Anthropic, Sierra tau-Voice (91 failures), OpenAI playbook | 79 to 90% agent-caused in tau-Voice |
| Sub-scores carry independent signal | BenchScope | BFCL 20 categories ≈ 7 |

---

## 4. Environments built from production logs or real traces (2026 evidence)

No paper found builds a full tau-style environment (DB + tools + policy + user) purely from production traces and reports generalisation from those seeds. What exists is partial: real data for one layer (tools, users, personas, tasks) and synthetic for the rest.

### 4.1 Tools from real traces: MCP-Persona (2606.02470)

The closest thing to our "tools as code from traces". Real MCP servers (Reddit, Slack, Lark, Xiaohongshu) were driven with sandbox accounts, traces were collected (seed calls from annotators, then "adversarial failure induction" perturbing valid inputs across type mismatches, schema violations, boundary conditions, semantic conflicts), and an LLM generated Python simulators conditioned on "(1) the tool's input schema; (2) the collected behavioral traces; and (3) the generated context handler APIs."

Fidelity, measured on 50 authentic Lark traces (25 successful, 25 failed) with reconstructed pre-condition state: traces-plus-schema ("Tool-Traverse") 93.8% F1 vs 53.3% for documentation-only; behavioural alignment 94% accuracy, 95.8% precision, 92% recall; response similarity METEOR 0.8703 vs 0.3214.

What failed or needed care: "a simulator might fail simply because it lacks a specific entity" (state must be reconstructed before comparison); simulators do best on "operation-dominated MCP servers" and may struggle with complex stateful interactions. Grading: checkpoint LLM scores 0/0.5/1 (91.5% human correlation) plus execution checks with dedicated Create/Update/Delete executors on the context files. 173 human-verified tasks. https://arxiv.org/html/2606.02470

Takeaway: traces roughly double tool-simulator fidelity over docs alone, and the failed-call half of the traces is what teaches error formats.

### 4.2 Hybrid real logs plus learned simulator: ToolOmni (2604.13787)

```
upon a tool invocation, the system first queries a comprehensive repository of historical real-world API call records. If a matching execution trace is found, the authentic response is returned directly.
```
Otherwise "the system seamlessly transitions to MirrorAPI, which serves as a high-fidelity proxy" that "faithfully replicate[s] the stochasticity and error distributions inherent in practical environments" and can model "missing parameters, authentication errors, and service timeouts." The paper does not report the share of calls served from logs vs simulator, nor fidelity discrepancies. https://arxiv.org/html/2604.13787. Same three-tier idea as CacheRL and the R09 recommendation (exact cache, generated code, LLM fallback), still without published hit-rate numbers.

### 4.3 Whole benchmark from real customer-service data: OlaBench / OlaMind (ByteDance, 2510.22143v3)

```
Existing benchmarks and training pipelines for industrial intelligent customer service (ICS) remain misaligned with real-world dialogue requirements, overemphasizing verifiable task success while under-measuring subjective service quality and realistic failure modes, leaving a gap between offline gains and deployable dialogue behavior.
```
Data "derived from real-world industrial customer-service data" across RAG, Workflow, and Agent settings, with "thorough manual inspection and human-in-the-loop de-identification". 3,000 core dialogues plus 1,000 risk plus 1,000 hallucination instances; 5,000+ expert labels; 95% inter-annotator agreement on 1,000 UX samples. Six dimensions: dialogue quality, policy compliance, tool calling, critical business risk ("admitting platform liability," "misidentifying the ICS role," "overcommitting"), hallucination, latency.

What worked: rubric-aware, instance-specific rewards; expert-strategy distillation from "top-performing experts (high resolution rates, five-star user satisfaction)"; staged RL; online A/B over 14,100 sessions: "+28.92% IRR and -6.08% HTR". What failed: direct LLM chain-of-thought distillation without human guidance produced 40.1% hallucination vs 19.7% with structured expert patterns; single-stage RL degraded safety (risk 10.9% vs 8.7%); response-only imitation 39.5% critical business risk. GPT-5.2 scored only 70.58 overall despite strong task completion. Not an executable environment (static dialogues), but the strongest 2026 evidence that real logs surface failure classes synthetic tasks miss (liability admission, overcommitting). https://arxiv.org/html/2510.22143v3

### 4.4 User side from real data

RealUserSim (2605.20204): "unconstrained LLM defaults produce a Formalism Ceiling (style match rates of 6-8% against real users), while hand-crafted behavioral directives trigger Directive Amplification, where models hyper-interpret instructions into unnatural behavioral extremes that vary dramatically across simulator models." 7,275 profiles from 14k WildChat conversations raised behavioural match from 24.2% to 45.3% and lowered agent success by 3.2 to 3.5 points, "surfacing three failure mechanisms invisible to cooperative simulators". https://arxiv.org/abs/2605.20204

"Simulated Customers Never Walk Away" (2606.20708): 2,790 production sales conversations (793 ending in verified payment). Simulators matched eventual buyers (+0.09 bias) but inflated engagement of eventual non-buyers (+0.40); non-buyer resistance halved (25.1% to 13.5%), deliberation nearly doubled (21.9% to 40.1%), d=0.38, replicated across model families, and "Explicit disengagement instructions barely reduced the outcome-conditioned contrast (d = 0.34)". Recommendation: "Ground simulators in real disengagement trajectories" and "Target conditional decision fidelity, not just communicative realism". https://arxiv.org/html/2606.20708v1

ECom-Bench (2507.05639): "dynamic user simulation based on persona information collected from real e-commerce customer interactions" and "a realistic task dataset derived from authentic e-commerce dialogues"; GPT-4o 10 to 20% pass. https://arxiv.org/abs/2507.05639

Sierra tau-Banking's flow-rule user is the opposite design (rules first, LLM fills gaps) and reports 4 task-critical user errors in 194 traces.

### 4.5 Tasks and data from real sources

Scale MCP-Atlas: Airtable tasks "query real business datasets that Scale has acquired". Halluminate: task templates "derived from real-world production queries" with synthetic data behind them. Mercor: experts create worlds by doing a fake project for a week. Surge: annotators populate entities "that approximate real-world e-commerce systems". Agent-World (R12): databases mined from the web by a research agent. None of these ingests a customer's traces.

### 4.6 Sim-to-real gap for tool agents: "When Simulation Lies" (2605.11928)

```
Tool-use language agents are evaluated on benchmarks that assume clean inputs, unambiguous tool registries, and reliable APIs. Real deployments violate all these assumptions...
```
22 perturbation types across observation, action, reward-relevant metadata, transition dynamics, each tied to a documented GitHub issue. Observation perturbations cost under 5%; reward-relevant metadata failures ~40%; transition dynamics ~30%; scale did not fix it. Domain-randomised training on a 3B model kept ~75% of clean accuracy and "closed roughly 27% of the transition gap despite never encountering transition perturbations during training". https://arxiv.org/abs/2605.11928v1

### 4.7 Generalisation vs overfitting to seeds

- Positive transfer from one realistic environment: CoreCraft (1,000 tasks, one world) gave +4.5 to +7.4 on three external benchmarks after one epoch. EnvFactory (R12): one grounded domain with ~200 audited tasks beat five times more environments.
- Overfitting evidence is about *users*, not tools: cooperative simulators inflate success (Lost in Simulation, RealUserSim, Simulated Customers); hand-written directives amplify into caricatures across simulator models (RealUserSim).
- Overfitting to seed *traces* specifically: no published measurement. AgentRefine (2501.01702) argues that imitation "merely memorize[s] the correct trajectory information, fundamentally leading to a lack of generalization capability"; ShopGym (R09) and MCP-Persona validate simulators on held-out real calls, which is the right test, but neither reports how far off-path the simulators stay faithful.
- Anti-overfitting mechanisms in use: held-out halves (MCP-Atlas), licence bans (APEX), fresh task synthesis per iteration (Agent-World), rotating tasks (Anthropic "living artifact"), read-allowlists derived per task rather than global (tau3).

---

## 5. What this changes for us

1. Adopt the tau3 bug taxonomy as the acceptance checklist for every rebuilt Run (section 1.5 table). Concretely, before a Run enters the pool: (a) the recorded end state is reachable under the compiled policy (no gold-violates-policy), (b) every ID/amount/date in the target exists in the reconstructed DB (no gold-violates-data), (c) the simulated user has a disclosure rule for every fact the agent had to ask for and a fallback for every refusal the trace shows, (d) a loophole probe (oracle told to reach the target while skipping the policy step) scores 0, (e) grep the environment observations for the target values (no answer leakage).

2. Reads must not participate in the end-state hash. tau3 1.0.1 lost real scores to exactly this bug ("Extra read calls zeroed rewards"). Our End state is writes plus what the user was told or asked; reads are diagnostic only. If a required read must be asserted (banking lost-card flows), use a per-Run allowlist derived from the source trace, as tau3 does.

3. Canonicalise before hashing: numeric spelling (25 vs 25.0), ordering conventions (newest-first vs insertion order), generated IDs and timestamps exempt. Ship lenient replay (warn on cosmetic output drift, fail on state divergence) from day one, plus a `--fresh-tasks` style regrade so we can re-score stored Runs after fixing an Environment.

4. Hallucinated tool names are no-ops, not errors. Both tau3 replay and the live env behave that way; a cheaper model inventing a tool should get a wrong end state, not an infrastructure exception.

5. Grade the user's action when the customer performs the write. The tau-Banking `requestor: "user"` pattern means the agent's advice is judged by what the simulated user then does. For our support/CRM/banking domains where the agent often only recommends, model user-side tools and include them in the End state.

6. Verdict from a product of components, with an explicit false-positive guard for inaction. 2607.02577 shows final-state scoring "can reward inaction" (agent transfers to human, DB unchanged, pass). Since our verdict is end state plus what the user was told, require that the communicated outcome matches too, and treat "transferred without acting" as a distinct verdict class, not a pass.

7. Expect a 10 to 20% defect rate in first-pass task specs and budget a Verified pass. Every serious benchmark shipped a Verified edition within a year (SWE-bench, tau2, Toolathlon, tau3, MCP-Atlas audited at 13.5%). Our rebuilt Runs are auto-generated from traces, so the rate will be higher, not lower. Sierra's method scales: two independent reviewers, at least one manual oracle trajectory per task, re-audit after the first large batch of results.

8. Validate tool simulators on held-out real calls, including failed calls. MCP-Persona: traces plus schema 93.8% F1 vs docs-only 53.3%, and the failed half of the trace set is what teaches error formats. Our replay-fidelity gate (R09) should report success and error fidelity separately and reconstruct pre-state before comparing.

9. User simulator: rules first, LLM second, and never trust cooperative defaults. tau-Banking's flow-rule design produced 4 critical user errors in 194 traces; free LLM users swing agent scores by 9 points and inflate success of eventual non-completers. Derive each Run's user rules from the trace (what was said when, what was refused), and add an explicit "walk away" branch when the trace shows one.

10. Difficulty and coverage checks, cheap versions: drop Runs where the cheap model passes 8/8 or the frontier model fails 0/8 (Anthropic, ARE, Envs-FORGE); compute the 2608.06329 "policy violations per task" number across the pool so we can tell a customer their traces exercise 6 of 40 policy items.

11. Treat verifier hardening as a pass over the pool, not per Run: a hacker agent that tries to reach a pass without doing the work, a fixer that patches the environment, a solver that confirms the oracle still passes (2606.08960). Cheap to run against our compiled policy triggers and DB-diff verdicts.

12. Attribute failures before reporting them. tau-Voice's human review of 91 failures (79 to 90% agent-caused) is what makes their numbers credible. Our dashboard should label each failed Run as agent, environment, or user-sim caused, with a small weekly human sample.

13. Do not claim generalisation beyond the seed traces. No published evidence measures overfitting to seed traces for tool environments. State what we do have: replay fidelity on held-out real calls, and (optionally) a small off-path probe against the customer's staging system.

14. Interface: OpenEnv `reset()/step()/state()` in Docker stays the right packaging (multi-org governance since June 2026), and OpenEnv explicitly leaves reward to us.

---

## 6. Where sources disagree

1. Realism vs verifiability. Surge, Mercor, Mechanize, AgentGym2 argue realism and expert-authored rubrics drive transfer; Halluminate, Anthropic, ScaleEnv, 2607.02577 argue for programmatic state checks and warn that rubric judges drift and get hacked. Surge's own reward is a rubric fraction graded by a model; 2607.02577 measured 30.5% evaluator error in the rubric-based LiveMCPBench. Nobody has ablated realism against verifiability; CoreCraft names the ablation as future work.

2. Outcome-only grading. Anthropic and Sierra say grade the product, not the path; 2607.02577 shows outcome-only passes inaction, and AgenticAI-Supervisor (R12) found ~40% constraint misrepresentation under outcome-only rewards. tau3 keeps ACTION checks available per task; Tool-Veritas adds a tool-invocation component. The middle position (end state plus required writes plus communicated facts) is what tau3 banking tasks implement and what we planned.

3. Determinism. Grounded Scaling and Halluminate want δ=1 environments; AgentGym2, Vending-Bench 2, When Simulation Lies, and Step-DeepResearch inject noise and adversarial counterparties on purpose. Reconcilable as: deterministic given a seed, with noise as a controlled, replayable input. tau-Voice does exactly this (audio effects are configured, simulation time is discrete).

4. User simulation. Sierra reports 2% task-critical simulator error with flow rules and attributes 79 to 90% of failures to agents; Lost in Simulation, RealUserSim, and Simulated Customers report large validity gaps for LLM users. The difference is design (rules vs free directives) and what is measured (task-critical error vs calibration against humans). Both can be true: a rules-based user rarely breaks a task, and still does not behave like a real customer.

5. Difficulty targets. Anthropic: capability evals "should start at a low pass rate". Synthesis pipelines (R12) keep tasks in a 0.2 to 0.8 band for RL signal. The 2608.06329 audit scores tau3 airline as relatively low complexity. For a regression tool on a customer's traces, the right target is neither: the population is fixed by what the customer's agent actually did, so we report the distribution rather than curate it.

6. Task quantity. Mechanize argues fewer, expensive, expert-built tasks; Agent-World, AgentOmnia, EnvScaler (R12) synthesise thousands of environments. EnvFactory and CoreCraft data favour one grounded environment with a few hundred audited tasks. Our design (one Environment per customer, Runs from traces) sits with the second camp.

7. Reward hacking prevalence. Hack-Verifiable finds 8.5 to 28.5% hack rates depending on model; Hardening finds 16% of tasks hackable from the description alone; METR found 0.7% on HCAST but 30.4% with visible scoring code. The variable is whether the verifier is visible or inferable. Keep verdict code out of the Environment the model sees.

8. Held-out sets. Scale and Mercor rely on held-out halves and licence bans; Anthropic and Agent-World rely on regenerating tasks. For a per-customer tool the contamination risk is smaller (the traces are private), but the rebuilt Environment is visible to the model under test, so hidden verdict code matters more than hidden tasks.
