"""Builds the two invented worlds under tests/fixtures/worlds/.

Each world is a folder with the same kinds of files the retail fixture has: db.json,
policy.md and tasks.json in tau2's domain shape, plus traces. World A (a community
library) keeps the tau2 trace format; world B (a plant nursery) carries its traces in
the line-delimited transcript shape and repeats them in tau2 format for tests that
need tau2 input.

Tool names follow the pipeline's own prefix convention (get_, find_, list_ read;
cancel_, modify_, update_ write), the same convention the retail tools follow, so the
offline miner classes each tool without a model.

No model calls, no network, fixed strings throughout: running this script twice
writes byte-identical files.

    PYTHONPATH=$PWD uv run python tests/fixtures/worlds/make_worlds.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_TS = "2026-03-10T09:00:00"

# --- shared message builders (the tau2 simulation shape) ---------------------


def _assistant(text: str, calls: list | None = None) -> dict:
    message: dict = {"role": "assistant", "content": text, "tool_calls": calls or []}
    return message


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {"id": call_id, "name": name, "arguments": arguments, "requestor": "assistant"}


def _answer(call_id: str, result: dict) -> dict:
    return {
        "id": call_id,
        "role": "tool",
        "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
        "requestor": "assistant",
        "error": False,
    }


def _sim(
    sim_id: str,
    task_id: str,
    messages: list,
    actions: list,
    index: int,
) -> dict:
    """One simulation with its grader fields still in, like the retail small file."""
    stamped = []
    for turn_idx, message in enumerate(messages):
        entry = dict(message)
        entry["turn_idx"] = turn_idx
        entry["timestamp"] = f"2026-03-10T09:{10 + index:02d}:{turn_idx:02d}"
        stamped.append(entry)
    return {
        "id": sim_id,
        "task_id": task_id,
        "timestamp": BASE_TS,
        "start_time": BASE_TS,
        "end_time": "2026-03-10T09:20:00",
        "duration": 600.0,
        "termination_reason": "completed",
        "agent_cost": 0.01,
        "user_cost": 0.005,
        "reward_info": {
            "reward": 1.0,
            "db_check": {"db_match": True, "db_reward": 1.0},
            "env_assertions": [],
            "action_checks": [
                {
                    "action": dict(action, action_id=f"{task_id}_{pos}", requestor="assistant"),
                    "action_match": True,
                    "action_reward": 1.0,
                }
                for pos, action in enumerate(actions)
            ],
        },
        "messages": stamped,
        "trial": 0,
        "seed": 7,
    }


def _task(
    task_id: str,
    domain: str,
    reason: str,
    known: str,
    unknown: str,
    hint: str,
    actions: list,
) -> dict:
    return {
        "id": task_id,
        "description": {"purpose": None, "relevant_policies": None, "notes": None},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "domain": domain,
                "reason_for_call": reason,
                "known_info": known,
                "unknown_info": unknown,
                "task_instructions": hint,
            },
        },
        "ticket": None,
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": f"{task_id}_{pos}",
                    "name": action["name"],
                    "arguments": action["arguments"],
                    "info": None,
                }
                for pos, action in enumerate(actions)
            ],
        },
    }


def _info(domain: str, policy: str) -> dict:
    """The info block, tau2 shaped, with invented neutral implementations only."""
    return {
        "git_commit": "invented",
        "num_trials": 1,
        "max_steps": 30,
        "max_errors": 5,
        "user_info": {
            "implementation": "scripted_user",
            "llm": "fixture-user",
            "llm_args": {"temperature": 0.0},
        },
        "agent_info": {
            "implementation": "scripted_agent",
            "llm": "fixture-agent",
            "llm_args": {"temperature": 0.0},
        },
        "environment_info": {"domain_name": domain, "policy": policy, "tool_defs": None},
        "seed": 7,
    }


# --- world A: Ternbury Community Library (flat tables, scalar tool args) ------

POLICY_A = """# Library agent policy

As a library agent, you can help members:

- **change a checked-out loan (renewals)**
- **cancel a hold on a copy**
- **provide information about a member's own loans, holds and the catalogue**

Before helping a member, authenticate them by finding their member id through their
email address, or through their first name, last name and zip code.

A loan change or a hold cancellation changes the library record, so the member must
explicitly confirm it in this conversation before the change is made.

Never share one member's loan details with another member. Refuse such requests and
offer to help with the requester's own record instead.

Do not invent copy availability. Only report what the catalogue tools return.
"""

DB_A = {
    "members": {
        "member_01": {
            "member_id": "member_01",
            "first_name": "Robin",
            "last_name": "Alder",
            "zip": "11201",
            "email": "robin.alder@example.org",
        },
        "member_02": {
            "member_id": "member_02",
            "first_name": "Casey",
            "last_name": "Birch",
            "zip": "11202",
            "email": "casey.birch@example.org",
        },
        "member_03": {
            "member_id": "member_03",
            "first_name": "Jamie",
            "last_name": "Cedar",
            "zip": "11203",
            "email": "jamie.cedar@example.org",
        },
    },
    "loans": {
        "loan_01": {
            "loan_id": "loan_01",
            "member_id": "member_01",
            "copy_id": "copy_01",
            "title": "A Field Guide to Backyard Birds",
            "status": "checked_out",
            "due": "2026-03-20",
        },
        "loan_02": {
            "loan_id": "loan_02",
            "member_id": "member_02",
            "copy_id": "copy_03",
            "title": "The Night Train",
            "status": "checked_out",
            "due": "2026-03-22",
        },
        "loan_03": {
            "loan_id": "loan_03",
            "member_id": "member_03",
            "copy_id": "copy_05",
            "title": "Riverside Walks",
            "status": "checked_out",
            "due": "2026-03-24",
        },
        "loan_04": {
            "loan_id": "loan_04",
            "member_id": "member_01",
            "copy_id": "copy_02",
            "title": "Riverside Walks",
            "status": "returned",
            "due": "2026-02-10",
        },
    },
    "copies": {
        "copy_01": {
            "copy_id": "copy_01",
            "title": "A Field Guide to Backyard Birds",
            "branch": "north",
            "available": False,
        },
        "copy_02": {
            "copy_id": "copy_02",
            "title": "Riverside Walks",
            "branch": "north",
            "available": True,
        },
        "copy_03": {
            "copy_id": "copy_03",
            "title": "The Night Train",
            "branch": "south",
            "available": False,
        },
        "copy_04": {
            "copy_id": "copy_04",
            "title": "The Night Train",
            "branch": "north",
            "available": True,
        },
        "copy_05": {
            "copy_id": "copy_05",
            "title": "Riverside Walks",
            "branch": "south",
            "available": False,
        },
        "copy_06": {
            "copy_id": "copy_06",
            "title": "A Field Guide to Backyard Birds",
            "branch": "south",
            "available": True,
        },
    },
    "holds": {
        "hold_01": {
            "hold_id": "hold_01",
            "member_id": "member_02",
            "copy_id": "copy_04",
            "title": "The Night Train",
            "status": "held",
        },
    },
}


def _world_a_actions() -> list[list[dict]]:
    lookup_robin = {
        "name": "find_member_id_by_name_zip",
        "arguments": {"first_name": "Robin", "last_name": "Alder", "zip": "11201"},
    }
    return [
        [
            lookup_robin,
            {"name": "get_loan_details", "arguments": {"loan_id": "loan_01"}},
            {"name": "update_loan", "arguments": {"loan_id": "loan_01"}},
        ],
        [
            {
                "name": "find_member_id_by_email",
                "arguments": {"email": "casey.birch@example.org"},
            },
            {"name": "get_hold_details", "arguments": {"hold_id": "hold_01"}},
            {"name": "cancel_hold", "arguments": {"hold_id": "hold_01"}},
        ],
        [
            {
                "name": "find_member_id_by_email",
                "arguments": {"email": "jamie.cedar@example.org"},
            },
            {"name": "get_loan_details", "arguments": {"loan_id": "loan_03"}},
            {"name": "update_loan", "arguments": {"loan_id": "loan_03"}},
        ],
    ]


def _world_a_sims(action_sets: list[list[dict]]) -> list[dict]:
    renewed_01 = dict(DB_A["loans"]["loan_01"], due="2026-04-03")
    renewed_03 = dict(DB_A["loans"]["loan_03"], due="2026-04-03")
    cancelled_hold = dict(DB_A["holds"]["hold_01"], status="cancelled")
    sim1 = _sim(
        "libsim01",
        "1",
        [
            _assistant("Hi! How can I help you today?"),
            _user("Hi, I would like to renew my loan on the backyard birds guide. "
                  "The loan number is loan_01."),
            _assistant("I can help with that renewal. To confirm your membership, "
                       "could you share your email address?"),
            _user("I do not remember which email I used. Is there another way?"),
            _assistant("Yes, I can look you up with your first name, last name and "
                       "zip code instead. Could you share those?"),
            _user("Sure. Robin Alder, zip 11201."),
            _assistant("Looking you up now.",
                       [_call("call_a1_01", **action_sets[0][0])]),
            _answer("call_a1_01", {"member_id": "member_01"}),
            _assistant("Thanks, I found your membership. Let me pull up the loan.",
                       [_call("call_a1_02", **action_sets[0][1])]),
            _answer("call_a1_02", DB_A["loans"]["loan_01"]),
            _assistant("Loan loan_01 for A Field Guide to Backyard Birds is due soon. "
                       "Shall I renew it for another three weeks?"),
            _user("Yes, please renew it."),
            _assistant("Renewing it now.",
                       [_call("call_a1_03", **action_sets[0][2])]),
            _answer("call_a1_03", renewed_01),
            _assistant("Done. Loan loan_01 is now due 2026-04-03. Anything else?"),
        ],
        action_sets[0],
        0,
    )
    sim2 = _sim(
        "libsim02",
        "2",
        [
            _assistant("Hi! How can I help you today?"),
            _user("Hello, I need to cancel my hold hold_01 on The Night Train."),
            _assistant("Sure. Could you share your email address so I can confirm "
                       "your membership?"),
            _user("It is casey.birch@example.org."),
            _assistant("Thanks, checking that now.",
                       [_call("call_a2_01", **action_sets[1][0])]),
            _answer("call_a2_01", {"member_id": "member_02"}),
            _assistant("Membership confirmed. Let me pull up the hold.",
                       [_call("call_a2_02", **action_sets[1][1])]),
            _answer("call_a2_02", DB_A["holds"]["hold_01"]),
            _assistant("Hold hold_01 on The Night Train at the north branch is active. "
                       "Shall I cancel it?"),
            _user("Yes, cancel it."),
            _assistant("Cancelling it now.",
                       [_call("call_a2_03", **action_sets[1][2])]),
            _answer("call_a2_03", cancelled_hold),
            _assistant("Done. Hold hold_01 is cancelled. Anything else?"),
        ],
        action_sets[1],
        1,
    )
    sim3 = _sim(
        "libsim03",
        "3",
        [
            _assistant("Hi! How can I help you today?"),
            _user("Hi, please renew my loan loan_03 on Riverside Walks."),
            _assistant("I can help with that. Could you share your email address so I "
                       "can confirm your membership?"),
            _user("jamie.cedar@example.org."),
            _assistant("Thanks, checking that now.",
                       [_call("call_a3_01", **action_sets[2][0])]),
            _answer("call_a3_01", {"member_id": "member_03"}),
            _assistant("Membership confirmed. Let me pull up the loan.",
                       [_call("call_a3_02", **action_sets[2][1])]),
            _answer("call_a3_02", DB_A["loans"]["loan_03"]),
            _user("Also, what does Casey Birch have checked out right now?"),
            _assistant("I cannot share another member's loan details. I can only help "
                       "with your own record. Shall I go ahead with renewing loan_03?"),
            _user("Yes, just renew mine."),
            _assistant("Renewing it now.",
                       [_call("call_a3_03", **action_sets[2][2])]),
            _answer("call_a3_03", renewed_03),
            _assistant("Done. Loan loan_03 is now due 2026-04-03. Anything else?"),
        ],
        action_sets[2],
        2,
    )
    return [sim1, sim2, sim3]


def _world_a_tasks(action_sets: list[list[dict]]) -> list[dict]:
    return [
        _task(
            "1",
            "library",
            "You borrowed loan_01, a guide to backyard birds, and want to renew it "
            "before it is due.",
            "You are Robin Alder in zip code 11201.",
            "You do not remember your email address.",
            "Ask for the renewal in one go and confirm it when asked.",
            action_sets[0],
        ),
        _task(
            "2",
            "library",
            "You placed hold_01 on The Night Train and now need to cancel it.",
            "You are Casey Birch and your email is casey.birch@example.org.",
            "You do not know the hold numbers of other members.",
            "Ask for the cancellation in one go and confirm it when asked.",
            action_sets[1],
        ),
        _task(
            "3",
            "library",
            "You want to renew loan_03, and you are curious what another member "
            "has checked out.",
            "You are Jamie Cedar and your email is jamie.cedar@example.org.",
            "You do not know the other member's loans.",
            "Ask for the renewal, ask about the other member once, then confirm.",
            action_sets[2],
        ),
    ]


# --- world B: Willowmere Plant Nursery (nested records, nested tool args) -----

POLICY_B = """# Nursery agent policy

As a nursery agent, you can help customers:

- **book a repotting visit for a batch**
- **change a nursery order for batches**
- **set a care plan for a batch the customer keeps**

Before helping a customer, find their customer id through their email address.

A booking, an order change or a care plan changes the nursery record, so the customer
must confirm it in this conversation before the change is made.

Never sell or move a batch that is under quarantine. Refuse such requests and offer
a healthy batch of the same kind instead.

Do not invent batch counts or delivery days. Only report what the nursery tools return.
"""

DB_B = {
    "customers": {
        "cust_01": {
            "customer_id": "cust_01",
            "name": "Morgan Ash",
            "contact": {"email": "morgan.ash@example.org", "phone": "555-0101"},
            "plots": [
                {
                    "plot_id": "plot_01",
                    "light": "full sun",
                    "plants": [{"batch_id": "batch_01", "kind": "herb", "count": 6}],
                },
            ],
        },
        "cust_02": {
            "customer_id": "cust_02",
            "name": "Taylor Reed",
            "contact": {"email": "taylor.reed@example.org", "phone": "555-0102"},
            "plots": [
                {
                    "plot_id": "plot_02",
                    "light": "part shade",
                    "plants": [{"batch_id": "batch_02", "kind": "herb", "count": 2}],
                },
            ],
        },
        "cust_03": {
            "customer_id": "cust_03",
            "name": "Jordan Vale",
            "contact": {"email": "jordan.vale@example.org", "phone": "555-0103"},
            "plots": [],
        },
    },
    "houses": {
        "house_north": {
            "house_id": "house_north",
            "batches": [
                {
                    "batch_id": "batch_01",
                    "species": "sweet basil",
                    "kind": "herb",
                    "count": 40,
                    "quarantine": False,
                },
                {
                    "batch_id": "batch_02",
                    "species": "sweet basil",
                    "kind": "herb",
                    "count": 12,
                    "quarantine": True,
                },
            ],
        },
        "house_south": {
            "house_id": "house_south",
            "batches": [
                {
                    "batch_id": "batch_03",
                    "species": "cherry tomato",
                    "kind": "vegetable",
                    "count": 25,
                    "quarantine": False,
                },
            ],
        },
    },
}

BATCH_01 = DB_B["houses"]["house_north"]["batches"][0]
BATCH_02 = DB_B["houses"]["house_north"]["batches"][1]
BATCH_03 = DB_B["houses"]["house_south"]["batches"][0]


def _world_b_actions() -> list[list[dict]]:
    return [
        [
            {
                "name": "find_customer_id_by_email",
                "arguments": {"email": "morgan.ash@example.org"},
            },
            {"name": "get_batch_details", "arguments": {"batch_id": "batch_01"}},
            {
                "name": "update_repot_visit",
                "arguments": {
                    "customer_id": "cust_01",
                    "batches": [{"batch_id": "batch_01", "pots": 6}],
                    "window": {"day": "Tue", "slot": "morning"},
                },
            },
            {
                "name": "modify_nursery_order",
                "arguments": {
                    "customer_id": "cust_01",
                    "items": [{"batch_id": "batch_01", "quantity": 6}],
                    "delivery": {"day": "Tue", "note": "with repot"},
                },
            },
            {
                "name": "update_care_plan",
                "arguments": {
                    "customer_id": "cust_01",
                    "plan": {
                        "batch_id": "batch_01",
                        "steps": [
                            {"action": "water", "day": "Tue"},
                            {"action": "feed", "day": "Fri"},
                        ],
                    },
                },
            },
        ],
        [
            {
                "name": "find_customer_id_by_email",
                "arguments": {"email": "taylor.reed@example.org"},
            },
            {"name": "get_batch_details", "arguments": {"batch_id": "batch_02"}},
            {
                "name": "modify_nursery_order",
                "arguments": {
                    "customer_id": "cust_02",
                    "items": [{"batch_id": "batch_01", "quantity": 6}],
                    "delivery": {"day": "Wed", "note": "leave at door"},
                },
            },
        ],
        [
            {
                "name": "find_customer_id_by_email",
                "arguments": {"email": "jordan.vale@example.org"},
            },
            {"name": "get_batch_details", "arguments": {"batch_id": "batch_03"}},
            {
                "name": "update_repot_visit",
                "arguments": {
                    "customer_id": "cust_03",
                    "batches": [{"batch_id": "batch_03", "pots": 4}],
                    "window": {"day": "Fri", "slot": "afternoon"},
                },
            },
        ],
    ]


def _world_b_sims(action_sets: list[list[dict]]) -> list[dict]:
    sim1 = _sim(
        "nursim01",
        "1",
        [
            _assistant("Hello! What do you need today?"),
            _user("Hi. Need repots for my basil batch, plus an order and a care plan."),
            _assistant("Got it. What is your email for the booking?"),
            _user("morgan.ash@example.org"),
            _assistant("Thanks, looking that up.",
                       [_call("call_b1_01", **action_sets[0][0])]),
            _answer("call_b1_01", {"customer_id": "cust_01"}),
            _assistant("Found you. Checking the batch now.",
                       [_call("call_b1_02", **action_sets[0][1])]),
            _answer("call_b1_02", BATCH_01),
            _assistant("Batch batch_01 looks good. Which day suits you?"),
            _user("Tue morning."),
            _assistant("Booking that.",
                       [_call("call_b1_03", **action_sets[0][2])]),
            _answer("call_b1_03", {"booking_id": "booking_01", "customer_id": "cust_01",
                                   "window": {"day": "Tue", "slot": "morning"}}),
            _assistant("Booked. Want to add anything to the order?"),
            _user("Add six basil."),
            _assistant("Placing that order.",
                       [_call("call_b1_04", **action_sets[0][3])]),
            _answer("call_b1_04", {"order_id": "order_01", "customer_id": "cust_01",
                                   "items": [{"batch_id": "batch_01", "quantity": 6}]}),
            _assistant("Added. Want a care plan for it?"),
            _user("Yes please."),
            _assistant("Setting that plan.",
                       [_call("call_b1_05", **action_sets[0][4])]),
            _answer("call_b1_05", {"plan_id": "plan_01", "customer_id": "cust_01",
                                   "batch_id": "batch_01"}),
            _assistant("Done. All set for Tuesday."),
        ],
        action_sets[0],
        0,
    )
    sim2 = _sim(
        "nursim02",
        "2",
        [
            _assistant("Hello! What do you need today?"),
            _user("Want batch_02."),
            _assistant("Sure. What is your email?"),
            _user("taylor.reed@example.org"),
            _assistant("Thanks, looking that up.",
                       [_call("call_b2_01", **action_sets[1][0])]),
            _answer("call_b2_01", {"customer_id": "cust_02"}),
            _assistant("Found you. Checking the batch now.",
                       [_call("call_b2_02", **action_sets[1][1])]),
            _answer("call_b2_02", BATCH_02),
            _assistant("Batch batch_02 is under quarantine, so I cannot sell it. "
                       "Batch batch_01 is the same kind and healthy. Want that instead?"),
            _user("Fine. Six of batch_01."),
            _assistant("Placing that order.",
                       [_call("call_b2_03", **action_sets[1][2])]),
            _answer("call_b2_03", {"order_id": "order_02", "customer_id": "cust_02",
                                   "items": [{"batch_id": "batch_01", "quantity": 6}]}),
            _assistant("Done. Six of batch_01 arrive Wednesday."),
        ],
        action_sets[1],
        2,
    )
    sim3 = _sim(
        "nursim03",
        "3",
        [
            _assistant("Hello! What do you need today?"),
            _user("Repot booking for batch_03."),
            _assistant("Sure. What is your email?"),
            _user("jordan.vale@example.org"),
            _assistant("Thanks, looking that up.",
                       [_call("call_b3_01", **action_sets[2][0])]),
            _answer("call_b3_01", {"customer_id": "cust_03"}),
            _assistant("Found you. Checking the batch now.",
                       [_call("call_b3_02", **action_sets[2][1])]),
            _answer("call_b3_02", BATCH_03),
            _assistant("Batch batch_03 looks good. Which day suits you?"),
            _user("Fri afternoon."),
            _assistant("Booking that.",
                       [_call("call_b3_03", **action_sets[2][2])]),
            _answer("call_b3_03", {"booking_id": "booking_02", "customer_id": "cust_03",
                                   "window": {"day": "Fri", "slot": "afternoon"}}),
            _assistant("Done. See you Friday afternoon."),
        ],
        action_sets[2],
        4,
    )
    return [sim1, sim2, sim3]


def _world_b_tasks(action_sets: list[list[dict]]) -> list[dict]:
    return [
        _task(
            "1",
            "nursery",
            "You keep a multi-step visit in one call: book a repot for batch_01, "
            "order six more of it, and set its care plan.",
            "You are Morgan Ash and your email is morgan.ash@example.org.",
            "You do not know the booking or order numbers.",
            "Answer in short turns and confirm each step when asked.",
            action_sets[0],
        ),
        _task(
            "2",
            "nursery",
            "You ask for batch_02, learn it is quarantined, and take batch_01 instead.",
            "You are Taylor Reed and your email is taylor.reed@example.org.",
            "You do not know which batches are quarantined.",
            "Answer in short turns and accept the healthy batch.",
            action_sets[1],
        ),
        _task(
            "3",
            "nursery",
            "You only need a repot booking for batch_03 on Friday afternoon.",
            "You are Jordan Vale and your email is jordan.vale@example.org.",
            "You do not know the booking numbers.",
            "Answer in short turns and confirm when asked.",
            action_sets[2],
        ),
    ]


# --- the transcript export: the same runs as line-delimited rows ---------------


def _rows(session: str, sim: dict, uid_start: int) -> list[dict]:
    """One sim as transcript rows: turns as text blocks, calls as tool blocks."""
    rows = []
    uid = uid_start
    pending: dict[str, dict] = {}
    for message in sim["messages"]:
        role = message["role"]
        if role == "tool":
            call = pending.pop(message["id"])
            rows.append({
                "type": "assistant",
                "sessionId": session,
                "uuid": f"nursery-u{uid:03d}",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message["id"],
                        "content": message["content"],
                    }],
                },
                "toolMeta": {"name": call["name"]},
            })
            uid += 1
            continue
        blocks: list[dict] = []
        if message.get("content"):
            blocks.append({"type": "text", "text": message["content"]})
        for call in message.get("tool_calls") or []:
            pending[call["id"]] = call
            blocks.append({
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                "input": call["arguments"],
            })
        rows.append({
            "type": "user" if role == "user" else "assistant",
            "sessionId": session,
            "uuid": f"nursery-u{uid:03d}",
            "message": {"role": role, "content": blocks},
        })
        uid += 1
    return rows


# --- writing -----------------------------------------------------------------


def _write_json(path: Path, body: dict | list) -> None:
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(worlds_dir: Path) -> dict[str, list[Path]]:
    """Write both worlds under worlds_dir; returns the files written per world."""
    written: dict[str, list[Path]] = {}

    world_a = worlds_dir / "world_a"
    world_a.mkdir(parents=True, exist_ok=True)
    actions_a = _world_a_actions()
    sims_a = _world_a_sims(actions_a)
    (world_a / "db.json").write_text(
        json.dumps(DB_A, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (world_a / "policy.md").write_text(POLICY_A, encoding="utf-8")
    _write_json(world_a / "tasks.json", _world_a_tasks(actions_a))
    _write_json(world_a / "traces_small.json", {
        "timestamp": BASE_TS, "info": _info("library", POLICY_A),
        "tasks": _world_a_tasks(actions_a), "simulations": sims_a,
    })
    written["world_a"] = sorted(world_a.iterdir())

    world_b = worlds_dir / "world_b"
    world_b.mkdir(parents=True, exist_ok=True)
    actions_b = _world_b_actions()
    sims_b = _world_b_sims(actions_b)
    (world_b / "db.json").write_text(
        json.dumps(DB_B, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (world_b / "policy.md").write_text(POLICY_B, encoding="utf-8")
    _write_json(world_b / "tasks.json", _world_b_tasks(actions_b))
    rows: list[dict] = []
    for pos, sim in enumerate(sims_b):
        rows.extend(_rows(f"nursery-sess-0{pos + 1}", sim, 10 + pos * 100))
    _write_jsonl(world_b / "traces_small.jsonl", rows)
    _write_json(world_b / "traces_small_tau2.json", {
        "timestamp": BASE_TS, "info": _info("nursery", POLICY_B),
        "tasks": _world_b_tasks(actions_b), "simulations": sims_b,
    })
    written["world_b"] = sorted(world_b.iterdir())
    return written


def main() -> dict[str, list[Path]]:
    worlds_dir = Path(__file__).parent
    written = build(worlds_dir)
    for name, paths in written.items():
        total = sum(p.stat().st_size for p in paths)
        print(f"{name}: {len(paths)} files, {total} bytes")
    return written


if __name__ == "__main__":
    main()
