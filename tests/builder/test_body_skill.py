"""The body skill: naming only gates that exist and nothing of any customer (D168)."""

import re

from kullback.builder.body_skill import BODY_SKILL


def test_the_body_skill_names_only_gates_that_exist():
    gates = {"parses", "confined", "executes_on_s0", "deterministic", "non_trivial", "replay_fidelity",
             "refuses_unknown", "memorised_values"}
    named = set(re.findall(r"\b([a-z]+_[a-z_]+)\b", BODY_SKILL))
    assert named and named <= gates, named - gates


def test_the_body_skill_names_nothing_of_any_customer():
    for fragment in ("order", "booking", "flight", "airline", "retail", "telecom", "phone", "tau2", "task_"):
        assert fragment not in BODY_SKILL.lower(), fragment
    assert "\u2014" not in BODY_SKILL and "\u2013" not in BODY_SKILL
