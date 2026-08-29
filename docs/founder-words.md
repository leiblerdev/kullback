# Founder's words, verbatim

Every message the founder sent during the design sessions for the monitoring tool, in order, unedited (spelling and all). These are the primary source; `decision-log.md` is the index over them. Timestamps are local session time. Append new sessions at the bottom; never edit an entry.

### 2026-08-26 14:40:43

yo lets create a new folder named monitoring tool the tool will be about essentially it monitors the traffic on your api calls or you feed your traces to it and it tells you what kind of tasks can be done better and cheaper with other models so essentially it creates a eval dataset from the agent traces and then run the evals for different models to find out which task is suited for what model essentially. To create the evaluation would be very challenging my intuition tells me that it will be based on a combination of tool calling accuracy and and llm as a judge to evaluate the reasoning of the models essentially becasuse agent traces look like input -> observation -> reasoning -> tool call / action and repeat this until it arrives at and answer, lets do a /grill-with-docs session where we figure out how to build this evaluation dataset from the agent datasets ( search up the whole research universe and help me understand what is happening and how to do it ) and then proceed to build a tool, we can talk about how to build it and shape the architecture, but lets focus on how to do the evals first search everything do a /grill-with-docs session please.

---

### 2026-08-26 14:40:47

/grill-with-docs Design the evaluation dataset + eval method for a monitoring tool that ingests agent traces (input -> observation -> reasoning -> tool call, looped) and determines which tasks can be done cheaper/better by other models

---

### 2026-08-26 14:44:57

I would say we need to make it the tool as close to reality as possible so run and step but again tell me otherwise please - need to focus on giving value for customers think run and step would bring value

---

### 2026-08-26 14:47:05

I would say adequacy + agreement ( actually both why not right ) yes match + mismatch + a judge that seems resonable tbh

---

### 2026-08-26 14:47:58

we also need to establish filtering criteria for the traces as well, not also traces should be evaluated on and also should segregrate into tasks as well - wait for the research agents to come in with their recommendations and why.

---

### 2026-08-26 15:03:46

I agree with grading the end state not the path and also with that if the steps are wrong then it will diverge, damm didn't realize grading the reasoning steps would be illogical ( goal is to build the system which gives the right direction always so - we need to be very clear with what we are not doing ) also agreed on why the tool call failed ( that is a very good signal in itself ), we need to design the system to give the judge the tools to verify it output ( agreed with this as well ). one of the metrics 1. end state verification ( how close it is ) measured with pass rate, also hard constraint violattion, tool selection P/R/F1 hallucinated tool rate, ( we are only going to keep the metrics which are really really important - minimal and efficient please ). i agree on traces we need to build the right environment to execute those stuff and also that will server as the basis for evaluation as well right - we need to create a high fidelity mock environment as well to verify the execution as well ( this will be very central ) lets discuss on the metrics and automated environment generation as well ( research extensively on automated environment generation please. )

---

### 2026-08-26 15:11:25

that environment generation will definetly move the needle for a lot of stuff tbh. so there is the tauforge agent which as in input takes in prompt and artifacts ( in our cases the agent traces ) and then 1. build the initial environment ( generates the policy, the database, the tools, conditioned on the real usage of the data ) here the business information can be sampled from the logs or the agent traces or the business knowledge graphs and the state of the harness would be something like skills, past builds and utility scripts as well and then in phase 2 it will add new scenarios and generate the seeds the tasks as well ( these scenarios are again grounded in agent tarces or the logs the companies provides us ) and then it genrates the tasks for us with user instructions, rubrics, verifiable rewards by conditioning on the sampled ( scenario x user persona x hardening trap ) and then it is hardeded with tasks ( benchmarked with a varity of things ) the context of the environment is knowledge graphs, scenarios and personas, hardening gaps as well ( so here we get the human input as well ) and then it spits out the verifiable environment and tasks for the companies. Steps involving :* 
- Build the initial environment, generate the policy, the database and tools $\to$ Conditioned on the real usage of the data. 
- *Generate tasks now :* start with a database of seeds, these are abstract tasks (open, close the account and then tune them to the personas)
- Iterative procedure so that we can tune the tasks to the difficulty of the task hence making sure that the model learns effectively $\to$ giving it enough learning signal now.  so essentially generating a synthesized RL environment grounded in the agent traces. There is env rigger and env harness as as well other research suggests towards 1. scenario and task generation grounded 2. database design and synthesis and data synthesis and then interface and tool synthesis 4. execution based self correction as well 5. verification logic generation as well -> this environment creator will be the core differentiator. major questions are how do you create the verifiable things -> do a deep research on this we need to build a small version of this - very critical to our logic as well. yes all the minimal metric is good tbh, any thing else we can add ?

---

### 2026-08-26 15:18:56

lets keep our focus on the replica right now because success on the environment is the actual signal and then we extend it to harden to post train and etc - we need to be able to trust on the results and for the trust to come we need to able to say okay this outputs are exactly similar to my environment and then let the agent run with whatever the context is the nearby stuff are which are grounded in the customers trajectories. 1. diff is one thing we need to be able to have trust on the outcomes of the environment as well for the evaluation.

---

### 2026-08-26 15:27:36

as i can understand based on the environment creation based on your traces we get all the stuff and from that we can create synthetic functions state, database as well and then we will have the actual tool sequences ( judged whether it achieved the task or not ) and that gives us something to verify that are these similar to the actual tool sequences or the tool graph - just thinking out loud. 1. i agree with one that the environment which is created which should be 100 exact match from the tool names to everythign the inputs. 2. i didn't understand off path fidelity if we don't specify the tools arguments and if the candidate is calling them then it would be a hallucination tool call right ( rates ). 3. the recorded run should pass ( but first it should be judged whether it did achieve what the run - so essentially the frontier run was good or not ) 4. yes the frontier must clear its own bar -> first lets search on how the actual agents traces look like ( do a deep research on it ) so that we can understand how to create the environment and what to judge essentially.

---

### 2026-08-26 15:32:34

2. i agree on off fidelity tool calls, they are total valid as well and if we do impose that we need hard pass but there is always a chance that the fronteir is wrong as well so we need to build keeping this in mind as well that fronteir is wrong and models can take other paths as well which is totally valid as well lets discuss how to do this as well.

---

### 2026-08-26 15:39:16

i agree on verifier is anchored on the user's intent which is very important - help me understand case 3 please in depth. The verifier should check whether the model did a good job, good job meaning did it satisfy the user or not ( which needs to be extracted from the users intent as well ). we are building this for the companies their end goal is to satisfy the user, so when candidate coverge on a novel end state we need a rubric of what a good end state looks like which is going to be extracted from the traces as well ( or atleast approximated right ) but equiping a model with tools is really required ( judge model meaning ). ( yes this  so the Environment's inverse replay (reconstruct S0 from the reads, apply the writes) is not optional, it is the only way to get one. agreed ). print some of the agent traces ( real world please ) that will help us design this whole thing

---

### 2026-08-26 15:46:55

damm all the agentic interaction is around tool calls we defintely need to build the environments. yes agreed with the system prompt constraints, they become a real checker and system prompt is a good way to define the constraints as well. I agree on the approach that the rubric ( which we get from the intent statement needs to be really grounded ) the fronteir model should write it from the trace with a verification pass that whether the created rubric is there or not from the traces -> damm this is going to be a big and impactful project

---

### 2026-08-26 15:54:25

lets continue on our decisions please, we need to take a lot of decisions - everything should be grounded in simplicity to be honest, the whole project should be very simple very simple. also a good idea to understand how to build rl environments synthetically would be to see how the existing environments are made right now ( research on this, see a few examples and print the whole environment for me please ). did you search about the filters what are others doing ? it would also make sense to keep the runs which are failed as well to help them understand the diagnosis as well. maybe for retries we only take the end one of it ? the most recent one ? ( because retries is essentially a network error ) duplicate runs also don't add value to be honest, there will be compaction boundary inside the run we just need to deal with it. ( Orphaned or truncated will be helpful to understand why it was truncated or orphaned ). knowing the tool profile is important but that is injected in system prompt right usually right ? 4. i don't ggree on this but this is important as well but qusetion to ask here is pure chitchatting runs i mean if the users end goal is chitchatting ( but yes need to discuss on it ). lets do a research on agent trace filtering as well use parallel agents

---

### 2026-08-26 16:01:03

it is always good to see how the actual stuff and try to build this, for us it is always good to understand how the actual environments look right now and then build it right so that atleast we have a reference set right ? and once this synthetic environment generator works we can create 1000s of environments and do evaluations and post training very easily - quite rewarding.

---

### 2026-08-26 16:09:09

The gold one loads S0 but the problem here is that we don't have gold ( we might have an approximation to gold ). damm this is how the environmetn looks like ( see we have a good reference of how it works now ). we also need to have the rewards basis as well and evaluation criterias as well for synthetic environments, a good idea would be to see how the environments for real world tasks like sending an email or raising a ticket or call support interaction is actually done ( lookup and print it here please - because most of our customers will be the same right ). - Copy: set_state replay with strict output comparison. That is Gate A (replay fidelity) already implemented as a 60-line function: replay recorded writes, assert the Environment returns the recorded output.
- Copy: hash-the-whole-DB equality, with the refinement from ScaleEnv (exempt generated ids and timestamps, hard columns exact, semantic columns fuzzy) so a Candidate's task_7 versus the frontier's task_2 id does not fail a Run.
- Copy: assertion helpers callable by name from task JSON. Our Verifier atoms are exactly this: assert_task_status(task_2, completed). Generated Verifiers become a list of (func_name, arguments, assert_value) records plus a DB hash, no new machinery.
- Do not copy: nl_assertions as a reward component. tau2 itself marks it experimental and does not use it in retail/airline/telecom. Our communicate checks stay substring or normalized-number matches, and anything softer is a reason code. explain me these in simpler terms please.

---

### 2026-08-26 16:12:24

yes actually a good task is the one where two domains experts would independently reach teh same pass fail verdict. what did you find different in your research ? add this in the memory that you need to give me unbiased results. we neeed to keep it simple doesn't mean that where we need complexity we don't add that, if the complexity is worth it we add that pleas.e

---

### 2026-08-26 16:21:35

1. yes agreed but also later add that it should also add different outputs but keep it differnet ( slightly modify it ) but thats a case for later. 2. we should also see the actual db transactions because that is where the signal lies as well haha - this agrees with the assertion helpers right ? 4. help me understand 4 please more in detail. but for the path state we need process reward models ( for simplicity we can do the end state as well for now, add this in todo please. ) 2. i agree on this ( 100% exact will reject good environments as well ) 3. yes lovely 4. agreed 5. i agreeo n this as well ( after the environment generators are stable, we can create environments and post train models for this and see if they hill climb on the benchmarks that would be cool to observe add this in todo please ). 6. the simulated users must be as close to the actual users then it will work ( which we can do it by having a the real production traces right ) 7. build for the hardest case so that it can handle these requests well 8. what holes do they have ? and how can we fill them ? 9. we keep the LLM judges a rubrics then 10. add thsi in todo please 11. retry please, we also need to see the actual traces from these tasks please.

---

### 2026-08-26 16:26:04

you are recording each and every decision and the conversation right ? because this will be the philosophy of us and the tool which we are building and the choices which we are taking.

---

### 2026-08-26 16:34:13

keep everything of my words please, those are the real decisions as well. this is exactly what comes to my mind, given a task what are the actions we expect the agent should take ( and this is my friends we get this from actual production traces ). I really like tau2 reward basis we need to learn from all of these and make the best for us. this confirms that we need to monitor the actual db transactions as well. yes agreed we need to build the environment from the whole traces 2. what is question 10 again ?

---

### 2026-08-26 16:41:00

lets start with traces only and then as the trust of the customer increases we can ask for more information. ( traces plus a db snpashot is the end goal we get from the customer to build the environment for evaluation and training ). lovely.

---

### 2026-08-26 16:48:00

what is 30/150 thresholds ? we need to see the thresholds on teh real data as well but agreed that there will be tasks with more priority / prominece than the others yes make sense 1. 2. 3. 4. ( just need to figure out empirically ) what else can we use to create tasks ? ( once we have the snapshots then we can generate tasks ourselves or viewing the traces also think about other relevant tasks as well right )

---

### 2026-08-26 17:02:00

i know tau forge doesn't exist publically. in assumptions.md add the 9 and other assumptions. help me understand q12 please.

---

### 2026-08-26 17:05:00

this is will be our last question for the day then we continue this tommorrow make sure you create a worktree or whatever so that I can resume the same exact conversation.

---

### 2026-08-26 17:20:00

.Why writes can be 100% and reads mostly cannot. A write's effect is small and structural (one row's status, one refund line), so after canonicalizing generated ids and timestamps it either matches or the Environment is wrong. A read returns a whole record, and real records carry things our rebuilt world cannot know: last_viewed_at, a computed total that the real system rounds differently, search results in a different order, a field that no Run ever wrote so our S0 never had it, a 2 MB response the trace truncated. tau2 solved this by skipping reads entirely in set_state ("to avoid non-deterministic output comparison issues") and still needed strict=False for 25 vs 25.0. BFCL and AppWorld each carry their own tolerance machinery. No published system gets reads to 100%, which is why the retry report flagged D22. what are reads and writes ? and why is reads not 100% i need this information to make the decision .

---

### 2026-08-26 17:30:00

yes i agree on the read, we need the whole database in some sense ( but need to observer the reads and generate data synthetically to match it as closely as possible ), if we don't fake it we just make sure it is relevant to todays context essentially or generate synthetic datasset based on it ( which is very similar ). synthetic dataset gneeration for the reads is very necessary by observing the actual dataset from the traces. reproduced similary ( the tasks we generate should exactly / similarly represents the actual tasks but one things which should never fail is that they should be a representative ( have to be hard constraint ) of the actual tasks ). see all the other worlds are published system is somehow generating synthetic data right. one shouldnt get to 100% strict because then that would be a problem. - not this and we pick this up tomorrow.

---

### 2026-08-27 11:38:32

lets continue on the decisions please, and take each and every decisions every carefully. with /grill-with-docs

---

### 2026-08-27 12:00:09

dumb the question down please. and the answers as well.

---

### 2026-08-27 12:01:31

i would say 2. because the main goal was the cancel the order and get the reason.

---

### 2026-08-27 12:02:19

lets discuss everything in super detail the minutest part in super detail, we can go on with the grilling session for ever haha

---

### 2026-08-27 12:04:08

but here the end state ( cancelling the order is oen of the end state right ?, the final end state should be cancelling the order and asking why did you cancel ) and depending on the end state we get pass or fail.

---

### 2026-08-27 12:07:05

btw how many questions are left ? and it totally depends you need to understand that on our requests, if we want the model to ask why + cancel then we have that did it ask why and record the why because that is a tool call but if we don't need the model to ask why we don't ask why, totally depends on the design and from the traces because 1. asking why is also a tool call 2. cancelling the order is also a tool call.

---

### 2026-08-27 12:10:54

i mean it should be a representative of the real user right ? it doesn't need to be exactly that tbh.

---

### 2026-08-27 12:13:36

i agree on the segregration tbh, facts stays consistent and everything is a representative is a good setup.

---

### 2026-08-27 12:16:12

3. fabricated result are definetly not going to be accepted -> tools almost everyone has a database interaction in it. i agree on hallucinated too call, i agree with your recommendations.

---

### 2026-08-27 12:19:58

1. yeah the failing atom is the best option saying okay we failed thsi because of this ....

---

### 2026-08-27 12:20:43

dumb it down please.

---

### 2026-08-27 12:21:43

yeah show them and let them fix it, because the checklist comes from the tasks right.

---

### 2026-08-27 12:23:58

from next time onwards, don't have your recommendation please it biases my optinions. ideally someone should loook at the samples before the grading happens to make sure that we are going to run on the right sample right ? or the actual stuff is good.

---

### 2026-08-27 12:25:31

yes! that is correct, before the tasks are out we grade and after the tasks are out we also check if there is agreement with the human grader or not.

---

### 2026-08-27 12:27:32

our first and then the customer's expert. ( agreed on the second pair of eyes ) prominent tasks and ( essentially a sample of tasks which are important )

---

### 2026-08-27 12:30:15

also no recommendation before my answers, after I answer then give your recommendagtions pleazse. 1. i agree with the first one but I think we can still fake it right maybe till we get the actual stuff from them and till try to fake it as much as possible. help me understand what is assisted any which ways ?

---

### 2026-08-27 12:31:51

yes

---

### 2026-08-27 12:34:38

i would like it against tau2 + as close to reality as possible ( like the actual production traces ) 1. given only the traces it rebuilts the world in which ...... but here we also need to add about the customers verdict that it agrees with the world of our customer as well as close to real world as much as possible.

---

### 2026-08-27 12:40:53

also we need to care much more about generability than the overfitting please. https://twotimespi.dev/ and to build the whole harness we can learn from this. and tau forge uses 1. bulid environment 2. augmetn seeds 3. generate tasks 4. harden tasks ( which is for the later todo ) with it having hte logs, skills, past builds, utility scripts and knowledge graph, abstract scenarios, personas and hardening taps. so the environment should be grounded ( very much in the customers tasks a lot a lot and a lot ) we are not really focusing on coding traces we are focusing on real world tasks. also explore more benchs than tau 2 - tau3 has also come up now - we already have researched on the benchmarks as well and what doers a good environment looks like ( research more so that we can construct our intuition about what does a good environment looks like ). and lets build it iteratively please. smallest slice to validate the way we are building is correct or not - we need to discuss about how to build this in detail as well.

---

### 2026-08-27 12:42:08

we need to make the harness as efficient and simple as possible sir.

---

### 2026-08-27 12:45:48

i would say 2 but we are usign tau2 as our reference first is to give ( first we need to hit the environment created by tau2 quality level ) and then find ways to improve on it. what is your recommendatin ?

---

### 2026-08-27 12:52:04

i would really like the flexibility but first lets get it in tau2 shape and then we mold it to our shape ( i will start uploading the real traces very soon this should help a lot ).

---

### 2026-08-27 12:54:45

a vendor export, maybe in a week or two we have them. we will know ( we will have answers to all the question once it is here ) we just need to build the environment createor now, we have several gates including llms as a judge to mark pass fail and where judge is unsure we can have the humans. also add the questions for the customer traces sir. are we done with all the questions ?

---

### 2026-08-27 12:56:20

also lets talk about the design of the harness ( essentially the environment creator - my philosophy is that it should be simple enough and gets the job done in high quality. ). lets research on the principles of good harness design and get back to me with design recommendations. have a detailed discussion on the harness creations.

---

### 2026-08-27 12:57:25

yo also for the general purpose and stuff use the cheaper models not the expensive ones please.

---

### 2026-08-27 12:59:51

yes environment creatoar and the whole system.

---

### 2026-08-27 13:09:48

https://lilianweng.github.io/posts/2026-07-04-harness/ a good starting point is this tbh. in parallel with the tau2 slice ( just make sure that it is not biased please. ) We need to discuss harness engineering from first priciples.

---

### 2026-08-27 13:20:43

keep in mind we are building the harness ( which is kinda a workflow for RL environment generation, so all the decisions and principles of harness design should be coming from those ).  1. i agree on runner must be frozen ( that is teh essential context around it ) 2. builder should be improving till we hit that the harness is able to generate high quality environments. https://github.com/walkinglabs/learn-harness-engineering/blob/main/docs/en/harness-designs/deepseek/index.md this providers the repo in general is quite good.

---

### 2026-08-27 13:24:39

ideally it should be the customer agreenment but for the time being we need to compare our environments with the tau2 environments in terms of quality so that initial environmetn creator can be built and also I can evaluate the quality of the environments as well. search up the whole repository please for best practices of harness engineering we need to discuss how do we build this harness for our usecase.

---

### 2026-08-27 13:30:57

from my side the principels for developing our harness is 1. we don't let the context to increase beyond 40% hard constraint 2. it should be minimal in design and very compact 3. We need to get stuff when needed, get tool calls get system prompt and stuff 4. we need the model to analyze its mistake and improve the prompts, tools, middle ware and memory after stdying past failures -> we need sessions trees ( the model should decide that what it needs to improve on what it doesn't ) - we can debate on this. cool, quality numbers looks good tbh.

---

### 2026-08-27 13:31:04

like we need to grill ourselves to understand each and every component going into the environment builder and build this very carefully and from first principles.

---

### 2026-08-27 13:35:49

1. i agree on the one, tree for builder and flat files for the runner 2. agreed we need an evaluator who decides what improves or not. - builder gets the 40% cap ( because beyond that the it is not worth ti ). tools prompts are loaded when needed and delete when don't and the evaluator also gets the call here. we test candidate under production setting always.

---

### 2026-08-27 13:39:40

we should keep the raw but normalized should be derived from that ( we should always have the source of truth stored ) and derive as much information as much possibel from it.

---

### 2026-08-27 13:41:18

what do you recommend ? i am going towards small fixed taxonomy plus the raw text

---

### 2026-08-27 13:44:10

i mean we observe waht the tool is intended to do right ? and from that we already know if then then we use the LLM to classify ( they are very strong on classification with enough context ).

---

### 2026-08-27 13:44:28

how many questions remaining ?

---

### 2026-08-27 13:45:35

what are the other components we could have added but havent' ?

---


### 2026-08-27 16:09:04
add these to todos please. but we need the following, 1. clusters 2. intent writer 3. canonicalizer 4. report 5. cost and budget accounting 14. provider adapter ( see how open code does it ? ) 15. maybe later 19. we need this and also the ability for the builder to modify itself -> rest are todos which we can do right.

### 2026-08-27 16:18:07
lets finish up the rest of the questions please.

### 2026-08-27 16:19:11
i would say read and then unclassified.

### 2026-08-27 16:29:18
what do you recommend ?

### 2026-08-27 16:38:05
i would say this as well. we will need to discuss this more as well. talk tomorrow.

### 2026-08-28 12:09:34
lets continue with the intensive grill me session please.

### 2026-08-28 12:10:13
add in the todo that we need to observer the user behaviour as well.

### 2026-08-28 12:11:22
i would say union of everything which is fixed eventually.

### 2026-08-28 12:13:12
what do you suggest ?

### 2026-08-28 12:14:32
i would say llm per column + code by rule ( which is verified by the LLM )

### 2026-08-28 12:16:35
can you apply /humanizer to the questions you ask me ? i would say one s0 per task not per customer ( we should have per tasks classification not per customer )

### 2026-08-28 12:17:56
i mean afaiu we should be able to generate the world from the whole traces and then we come up with tasks in that world right ? and evaluate per task right ?

### 2026-08-28 12:18:53
so one task would be to make the order pending and the other task would be to make the order deliverd in june both are valid tasks right ?

### 2026-08-28 12:19:28
what do you recommend ?

### 2026-08-28 12:20:22
yes!

### 2026-08-28 12:23:15
i would say 1 + 2. but the rewriting should be check to be honest that is that the correct rewrite according to the traces we got.

### 2026-08-28 12:24:30
2 + 3

### 2026-08-28 12:26:03
the answers should always be grouned in the real world to be honest ( which are the traces ) -> should always be consistent with the world

### 2026-08-28 12:27:14
i mean we need to do experiments to do decide the number here right ?

### 2026-08-28 12:31:47
i mean you should check it and then if there are something then the human checks right ? what does the research suggests ? what other are doign ?

### 2026-08-28 12:37:08
CUA-Gym forces an empty run to score below the gold run. The hacker-fixer paper (2606.08960) found 16% of 1,968 benchmark tasks could be passed from the description alone, and a hacker/fixer loop drove that to zero. i like these approaches. bro you need to start talking in english in simplified terms using /humanizer i mean we cannot fix that, we ideally need 100% 3. what do you suggest ? the world needs be to as close as to the real world sir.

### 2026-08-28 12:38:17
i would say we do it per task and then a proportion per task ?

### 2026-08-28 12:39:38
several changes tested together what is the problem in that ? start with one change per round and then accept or reject and then several changes ?

### 2026-08-28 12:40:50
two differnet tasks but the category could be same, tasks also have hierarchy right ?

### 2026-08-28 12:41:59
the judge 3.

### 2026-08-28 12:42:18
also do a deep research on how to create good LLM judges for your tasks.

### 2026-08-28 12:44:04
report should also have that whether we were able to create the environment or not as well including the numbers and the verdict is decided by the the persons ( just show the numbers and suggest ) decision is the person.

### 2026-08-28 12:45:16
we need to be as close as to the reality sir, report as it is is. stop and seek permission to continue further.

### 2026-08-28 12:46:33
lessons that it learnt from the previous customers sir ( so that it doesn't repeat ) but also ask it to question the relevance of the lessons as well.

### 2026-08-28 12:47:30
3

### 2026-08-28 12:48:51
no! what do you recommend ?

### 2026-08-28 12:51:21
but our environment will envolve ( tau2 is the baseline right now, we will move further into more high fidelity better environment than tau2 ) so thinly wrap ? but lets first build the tau2 shape and then we build the openENV wrapper

### 2026-08-28 12:52:51
1. what do you suggest ?

### 2026-08-28 12:53:22
also do research on synthetic data genreation best practices and synthethic environment generation best practices usign parallel agents

### 2026-08-28 13:08:48
agent as a judge should be the go to thingy because model + verifier is when we see the biggest gains ( we need agentic judges as well ). we don't have human labelled set and our end goal is to just have a synthetic environmetn creator withotu the human support ( agentic judges and then we see the disagreement )

### 2026-08-28 13:10:07
human resolves the dispute then sir.

### 2026-08-28 13:11:22
i mean we did the research right and that said that we should only care about the end state also we are not grading the process right now so.

### 2026-08-28 13:12:53
ask for the rest and reconstruct also i don't think any schema would be 9000 character long as well.

### 2026-08-28 13:15:53
1. i won't trust touching the rows as the selection 2. lets define what does coverage means? i would love to hear it in the form of total tasks covered ( which were defined early ).

### 2026-08-28 13:15:59
also how many questions are left ? feel that these are too many questions.

### 2026-08-28 13:18:34
and then using sub agents ( parallel agents please ), start buildign the environment generation harness please ( use dynamic workflow, an army of parallel agents of opus 5 or sonnet 5 or other better model ) with dag graphs coming in.

### 2026-08-28 13:34:03
i can provide api keys ( just use gpt 5.6 luna please ) does that work ?

### 2026-08-28 13:34:03
use fable as the verifier please, and after everything is built just have it run using mutmut please to tighten the code and a qa agent as well please ( search up the internet for the best qa agent - skills for the same )

### 2026-08-28 15:29:18
https://github.com/leibler-dev after this we need to push the env-generator ( lets first decide a name ) to the leibler org repo and then i start contributing to it as kkkamur07.

### 2026-08-28 15:30:19
or what setup do you recommend, i want to make it open source and update the link on the website as well and once it start working need to publish a blog on it with doingb some experiments as well.

### 2026-08-28 15:30:32
a cool name like kullback - which is the harness for environment generation and stuff.

### 2026-08-28 15:31:04
and only the code and research related to the harness is pushed nothing else sir.

### 2026-08-28 15:31:10
yeah name it as kullback.

### 2026-08-28 15:31:22
what setup do you recommend should i create an organization from my own repo ?

### 2026-08-28 15:38:59
i have created the organization, we need to transfer leibler brain to this organization now and create kullback repository where we can push the harness.

### 2026-08-28 15:41:14
brain repo should be private and kullback is the open source harness for environment generation using your traces use /humanizer  to draft the readme.md please where you can explain the workflow using the /mermaid architecture diagrams 1. what is this and what pain does it sovle 2. how does it solve this ? 3. results ? 4. why should I care ? 5. future work.

### 2026-08-28 15:44:02
add the contributing rule and stuff please so that people can contribute as well.

### 2026-08-28 15:46:53
also add the readme for leiblerdev as well, saying our mission is to make post training accessible to everyone and we are starting with the core thing for synthetic data generation i.e. kullback harness which takes in your traces and gives you a verified synthetic environment which can be used for and also don't have claude committing the code any more please.

### 2026-08-28 15:49:03
transfer cdtm-job ( first rename it to cdtm community tool ) and then transfer it to cdtm organization please.

### 2026-08-28 15:50:14
also push the leiblerdev/kullback please whatever the current push is please.

### 2026-08-28 15:51:58
In 48 hours you know which tasks a smaller model already handles. here in the website please update that we create synthetic data and environment ( most importantly environment ) from your traces.

### 2026-08-28 16:02:40
make it public kullback please. also run /humanizer on the changed stuff from website please. ( as little words as possible please ) and push the chagnes after verifying.

### 2026-08-28 16:05:44
https://mermaid.ai/open-source/syntax/architecture.html use this to update the architecture of kullback please, make this like the architecture of other harness please ( that would be nice )

### 2026-08-28 16:07:45
add kullback to the leibler.dev page please.

### 2026-08-28 16:08:26
in kullback docs please have the design philosophy behind kullback in design-philosphy.md where you describe what we did and why and why we didn't do some stuff.

### 2026-08-28 16:09:48
change the rules of the repo, i need to review the pull request to accept contribution and stuff. and add precommit for ruff check please. just that for kullback

### 2026-08-28 16:11:28
The Builder, stage by stage explore the mermaid diagrams and see what is good for diagraming the builder stage by stage as well.

### 2026-08-28 16:12:33
in the website also add about kullback please, like a page or something. suggest that please so that it increases credibility as well and simplify all the diagrams and the loop arrows ( that is the arrow for the harness - every harness has it haha )

### 2026-08-28 16:12:48
transfer the website for leibler to leiberdev please.

### 2026-08-28 16:19:24
did you transfer the website repo to the organization ? transfer first then i point it.

### 2026-08-28 16:21:19
okay cool! we need to now segregrate stuff from the folders and organize to streamline our development as well, create seperate folders in leibler that point towards the repos.

## 2026-08-28 (later)

> lovely! thanks for doing it kullback webpage is shit, you need to make it minimalist and add some animations to it please.

> lovely! thanks for doing it kullback webpage is shit, you need to make it minimalist and add some animations to it please. and simplify it as well please so that people can understand and attach the github sign and logo as well on the top.

> no vercel is moved to the website

> do that bro remove the duplicate and publish the page.

> cool!

> yo! how is the harness coming along "

> do a pass over it to figure out the /simplification opportunities using /code-review and quality and also can you and mutmut to harden it and then we start testing the environment creation using the api key ( only use gpt 5.6 luna please ) i will add the key.

> you didn't do the organizations as well, we need to organize stuff as well and push them right, brain, kullback and website right ? organize that wasy please, it and add deepline and other stuff in gitignore i mean they can be a part of the brain right ? do this using a sub agent.

> also where do i add the .env key ?

> simplify the docs as well please with /humanizer ( but in the same time don't miss out the details please )

> Measured on Sierra's public retail traces (tau2), where the real tools and database exist to compare against. Seen: runs used for the build. Held out: runs the build never saw.
>
>  make sure we are not over fitting to this please. also we need to extend the support to langfuse and other observability formats as well.

> and other public benchmarks formats as well and make sure the judge design is judge as an agent.

> and add that in todo that we need to have synthetic data generation now as well.

> Inside the loop, Kullback is two programs over one set of data records, with every model behind one interface.
>
> this loop add the digaram from a differnet diagram ( the architecture is not going well here )

> or generate a good diagram please.

> the major win for this design would be to show that it can generate good environment and the models post trainined on this environmetn are quite performant.

> yo also remove the claims please, the major claim would be we post training a 2B parameter model on the environment created by this.

> we need to complete the tool to start testing sir; after the reorganization we should have brain, website and kullback right ? each pointing to a different repository. Add the wrapper or send a week of traces. We rebuild your environment from them, verified against the traces, plus synthetic data. In 48 hours you know which tasks a smaller model already handles. here add that the link to kullback pleas.e

> Three sentences from the founder set the direction and everything below is downstream of them: "it should be simple enough and gets the job done in high quality", "as efficient and simple as possible", and "care much more about generalizability than the overfitting".
>
> instead of this, write it from the perspective that I am writing it.

> use the /humanizer please to write stuff, write it from my perspective please in my tone

> also keep the readme.md sharp and crisp please and did you add the model providers so that we can literally choose any model like open code and pi ?

> please organize the files quickly.

> remove the slice results please.

> you also need to explore how to cache prompts as well please or else the cost would be too much.

> no for our environment generation sir. also did you explore various strategies to write prompts for the builder ? that is very important. nice, plan being we will pick a benchmark, evaluate a fronteir model on it ( or the cheaper ones ) and then we will trian our model using opd or RL ( for that we will also create the harness for it ) on the environment this generates ( or for the time being we don't create the harness  and just write the code ourselves for multiple methods ) and then prove that this works if it doesn't we go back and improve our harness for synthetic environment generation.

> we need good research on each and every component of harness creation including the tools, prompts and everything. literally everything ( each and every component  of harness )

> we also need to simplify the readme.md please as well.

> also do research on synthetic data generation with just traces please or generally what are the best practices to do synthetic data generation.

> One claim to earn.
> A 2B model, post-trained in this.
> Not claimed yet. Next: the same rebuild on domains it has never seen, then a 2B parameter model post-trained on trajectories the verifier passed, measured on the real held-out tasks. The numbers get published either way remove this on web page please.

> https://www.distillabs.ai/blog/traces-vs-synthetic-benchmark/ see this please.

2026-08-28: "did you apply the best strategies for prompt creation and the best strategies for harness design ? learnt by researching ?"

2026-08-28: "you still haven't organized the stuff inside the libler folder why ?"

2026-08-28: "the folder is still named as monitoring-tool instead of kullback and website and brain has no signs of yet"

2026-08-28: "Our mission is to make post-training accessible to everyone. [the org profile text, four paragraphs, pasted in full] simplify this please."

2026-08-28: "yo you didn't push a lot right ? why ? i don't see any stuff."

2026-08-28: "i want this folder to be organized like that sir. what the hell why can't you follow simple instruvctions"

2026-08-28: "there is leibler here right ? we need sub folders right now brain, website and kullback and github which are pointed to the repos."

2026-08-28: "there is leibler here right ? we need sub folders right now brain, website and kullback and github which are pointed to the repos. why can't you simply organize this repo repo meaining this folder which we are in"

2026-08-28: "lovely you did copy everything right ? website folder is not organized well, organize it well pelase."

2026-08-28: "everything pusehdc right ? delete the old folder which was inside website/leibler - we contienue development in dprogramming/leibler."

2026-08-29: "did you apply mut mut ? and qa for hardening of the existing harness ?"

2026-08-29: "https://trymaitai.com/ we need execution monitoring and production monitoring as well haha, we need to mine those from the traces right ? just have these in todo please."

2026-08-29: "continue improving the harness please, kill all the mutants ( which are relevant please ) and what are the next task you are doing to do ?"

> do we actually need to test stuff now ? i can drop in the env keys to see how well we can generate the environments ? by using the harness ? also can you create a cool tui ( which is essentially like open code and stuff ? ) would be nice core feature only ( inspired from pi ) where you have printed kullback ? https://www.feynman.is/ use this as a refernce please ( differnece over pi, this is built on top of pi ) as well so difference between them would exactly tell us how to build the harness.

> we are creating the harness we should be able have a tui and stuff please.

> we need to do comprehensive testing on whether it is able to create high quality environments or not that is the first step ( we need expand the blast radius to more such environments ( which are the benchmarks like tau2 ) ) so that we can be really sure of the high quality verifiable environments/ we should not over fit we them as reference to see what our harness generates.

> do the research and then we can start the grill me to improve the harness design as well

> i would say split the 64.3 numbers ( decompose more so that we can understand from where the error is coming from ) and we should generate synthetic rows ( to build this to augment the database and make it like the actual users ) can you see what are the best practices to do this ?

> and what is the next steep again should we start the /grill-with-docs again to improve the harness architecture. also we need the numbers of runners and the verifiers as well so that we can compare them to the original implementation.

> okay if any of the above problems can be fixed with better mining please do that. [...] So the split is: confinement, missing import, result shape, error prefix are ours to fix and all have signal in hand; schema shape is the one that needs the customer's schema, and it should be reported as "outside what traces can show" rather than as a Builder miss. ( note please don't over fit, we will test this on multiple environments from the benchmarks you mined to verify whether this harness can generalize or not )

> i would say we also need to focus on synthetic user generation and db generation as well, as described in tau forge where we just have a synthetic seed of user data - see the best practices to do that and then we need to do this as well.

> we need the verifier  The Verifier stage produced zero confirmed verifiers on the first build because nothing converts a Trace into the Run it consumes, so the pass condition that makes an environment usable for grading has never been derived for real. as well, the harness should build thsi as well what is the problem ?

> you don't need to build anything the harness needs to build everythign sir.

> environment is one major step the next major step is good and accurate verifiers -> i think mostly the rsearch has converged them with llm as judges / reward models with rubrics, here we don't have reward models so we use llm as judges

## 2026-08-29 (evening)

> add this in the todo that model should be able to change its own system prompt and stuff and also loop over its environment run it figure out where the problem is and then solve it and stuff ( that gives us teh best environments ) it should be a loop, we need this add this in todo please with that we need to /grill-with-docs there has to be this loop of 1. build the environment 2. augment the environment 3. generate the tasks 4. harden the tasks and the loop to do this so that environments are very very good. add this we need to discuss that.

> there has to be a loop which runs so that the harness can steer the creation of the environment..

> add in todo that we also need synthetic ( verified data ) when we move from beign just an evaluatoin platform, also why is the environment creation taking so much time ?

> kill the run and implement this and then restart the run pleas.e

> also a dag graph based execution ? would be good right ? what would solve the independence problem.

> it should be dag not on paper but quite good, also add in todo that we need to add that in todo about the workers and hwo it is launching the workers.

> [GLM 5.3's post-training environment design, quoted at length: expert-work tasks with real resources, research agents converting workflow patterns into long-horizon environments, a judge agent checking solvability, a verifier generated without the reference solution, solver trajectories closing reward shortcuts, the three verifier checks] we also need to use this deisgn for glm 5.3 for environment generation add this in todo.

> this judge agent which checks every environment is solvabale is important, also we need a judge as an agent as well. there are 3 checks 1. oracle checks ( must award reward ) 2. null run check ( agent did nothing -> must award none ) 4. incomplete check -> must aware none and then we should have a trusted verifier as well.
