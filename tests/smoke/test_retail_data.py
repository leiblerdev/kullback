"""The real retail data as a smoke tier: each builder stage over the real thing.

Unit tests elsewhere prove the pipeline on the invented worlds. This tier keeps the
real-data checks they used to carry: ingest, mining, policy, one tool compile and one
played user task. Fast and offline: no keys, no network, no raw corpus needed.
"""

import json

import pytest

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.builder import ingest, mine
from kullback.builder import policy as pol
from kullback.builder.user_sim import SimulatedUser, derive_user_rules
from kullback.builder.vocabulary import GENERIC
from kullback.runner.records import Trace, Turn


def _derived_traces(tau2_small_path, workdir):
    """Traces off the small retail file, stored and derived in a scratch workdir."""
    summary = ingest.ingest_file(tau2_small_path, workdir)
    assert summary["gate"]["pass"] is True
    return summary, ingest.derive_traces(summary["raw_hash"], workdir)


@pytest.mark.smoke
def test_retail_ingest_passes_with_grader_fields_stripped(
        tau2_small_path, tau2_small, workdir):
    """The small file ingests, and the answer key lands beside the trace, not in it."""
    summary, traces = _derived_traces(tau2_small_path, workdir)
    assert summary["runs"] == len(tau2_small["simulations"]) == 3
    assert summary["tool_calls"] > 0
    for trace in traces:
        blob = json.dumps(trace.model_dump(mode="json", by_alias=True))
        assert "reward_info" not in blob and "task_id" not in blob
        sidecar = json.loads(ingest.grader_file(trace, workdir).read_text(encoding="utf-8"))
        assert "reward_info" in sidecar["fields"]


@pytest.mark.smoke
def test_retail_mining_finds_tools(tau2_small_path, workdir):
    """The miner sees a non-empty tool set on the real traces."""
    _, traces = _derived_traces(tau2_small_path, workdir)
    assert mine.mine_tools(traces)


@pytest.mark.smoke
def test_retail_policy_extracts_sentences(tau2_retail_dir):
    """The extractor finds rules in the real policy text."""
    sentences = pol.split_policy((tau2_retail_dir / "policy.md").read_text(encoding="utf-8"))
    assert len(sentences) > 0


@pytest.mark.smoke
def test_retail_compiles_one_tool(tau2_small_path, tau2_retail_dir, workdir):
    """One mined tool builds into an environment with a hand-written body standing in
    for what a model would write."""
    _, traces = _derived_traces(tau2_small_path, workdir)
    policy_text = (tau2_retail_dir / "policy.md").read_text(encoding="utf-8")
    schema = mine.mine_schema(traces, db_json_path=tau2_retail_dir / "db.json")
    sig = mine.mine_tools(traces)[0]
    env = ce.build_environment(schema, [sig], {sig.name: "pass"}, policy_text)
    assert env.tools_version


@pytest.mark.smoke
def test_retail_plays_one_recorded_user_task(tau2_small):
    """The simulator derives rules off the first recording and speaks its goal first."""
    sim = tau2_small["simulations"][0]
    pairs = [(message["role"], message.get("content")) for message in sim["messages"]
             if message["role"] in ("user", "assistant")]
    trace = Trace(
        trace_id=sim["id"],
        raw_hash="rawhash",
        ingest_version="0",
        source="tau2",
        turns=[Turn(idx=idx, role=role, content=content, raw_ptr=PTR)
               for idx, (role, content) in enumerate(pairs)],
        raw_ptr=PTR,
    )
    rules = derive_user_rules(trace, vocab=GENERIC)
    assert rules.facts
    first = SimulatedUser(rules).reply(
        [{"role": "assistant", "content": "Hi! How can I help you today?"}])
    assert first
