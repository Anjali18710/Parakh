"""Regression tests built from failures seen in real runs with gpt-oss-120b, on the real demo data.

Each test drives the full agent graph with a model that repeats a mistake the real model made,
and checks that the code-enforced rules still produce a correct, fully verified case file.
"""
import pytest
from fakes import MisbehavingLLM

from parakh.agents.graph import Deps, build_case_graph
from parakh.agents.service import load_or_screen
from parakh.config import DEFAULT_DATA_PATH, Settings
from parakh.data.store import TransactionStore
from parakh.patterns.library import PatternLibrary
from parakh.tools.investigation import InvestigationTools

HUB, DISTRIBUTOR, LOOP = "AC387042", "AC153168", "AC511856"


@pytest.fixture(scope="module")
def demo():
    store = TransactionStore.from_csv(str(DEFAULT_DATA_PATH))
    alerts = {a["account"]: a for a in load_or_screen(store, str(DEFAULT_DATA_PATH))}
    return store, InvestigationTools(store), PatternLibrary(), alerts


def run(demo, account, llm):
    store, itools, library, alerts = demo
    graph = build_case_graph(Deps(llm, store, itools, library, Settings()), "multi", human_review=False)
    return graph.invoke({"case_id": account, "account": account, "alert": alerts[account], "mode": "multi",
                         "analyst_notes": [], "evidence": [], "timeline": [], "tool_calls": 0})


def test_hub_is_gather_scatter_even_when_the_agent_says_fan_out_then_none(demo):
    state = run(demo, HUB, MisbehavingLLM(pattern_answers=("fan_out", "none"), bad_arithmetic=False))
    assert state["pattern"]["pattern_id"] == "gather_scatter"
    assert state["pattern"]["source"] == "rules"
    assert "fan_in" in state["pattern"]["excluded_by_shape"]
    assert sum(e["action"].startswith("rejected pattern") for e in state["timeline"]) == 2
    assert state["report"]["pattern_id"] == "gather_scatter"


def test_totals_are_exact_because_code_writes_them(demo):
    state = run(demo, HUB, MisbehavingLLM(pattern_answers=("gather_scatter",), bad_arithmetic=False))
    v = state["verification"]
    assert v["passed"] and v["claims_verified"] == v["claims_total"] >= 2
    assert state["report_attempts"] == 1  # right first time, no rewrites needed
    assert "{" not in state["report"]["summary"] and "{" not in state["report"]["claims"][0]["statement"]
    inflow = next(c for c in state["report"]["claims"] if "received" in c["statement"])
    assert inflow["statement"].startswith(f"{HUB} received ₹")


def test_wrong_arithmetic_never_reaches_the_analyst(demo):
    state = run(demo, HUB, MisbehavingLLM(pattern_answers=("gather_scatter",), bad_arithmetic=True))
    v = state["verification"]
    # The model never fixes its sums. After the retries, code corrects the totals from the cited
    # transactions, drops the uncited "nothing found" claim, and the case still has its evidence.
    assert v["passed"], v["issues"]
    assert len(v["corrected_claims"]) == 2 and len(v["removed_claims"]) == 1
    assert v["claims_total"] == 2 and v["claims_verified"] == 2
    store = demo[0]
    for claim in state["report"]["claims"]:
        true_total = float(store.get_txns(claim["txn_ids"]).amount.sum())
        assert abs(claim["amount"] - true_total) < 0.01


def test_distributor_is_wholesale_and_not_suspicious(demo):
    state = run(demo, DISTRIBUTOR, MisbehavingLLM(pattern_answers=("fan_in", "fan_in"), bad_arithmetic=False))
    assert state["pattern"]["pattern_id"] == "wholesale"
    assert state["pattern"]["kind"] == "innocent"
    assert state["report"]["verdict"] == "not_suspicious"
    assert state["verification"]["passed"] and state["verification"]["claims_verified"] >= 2


def test_loop_is_a_round_trip_not_a_chain(demo):
    state = run(demo, LOOP, MisbehavingLLM(pattern_answers=("layering_chain", "cycle"), bad_arithmetic=False))
    assert state["pattern"]["pattern_id"] == "cycle"
    assert state["pattern"]["source"] == "agent"
    assert state["verification"]["passed"] and state["verification"]["claims_verified"] >= 1


def test_miscounted_senders_and_duplicate_citation_from_real_run(demo):
    """Real claim from a gpt-oss run: 7 payments cited (one twice), but the text said 'eight distinct senders'."""
    from parakh.agents.rules import correct_claim
    from parakh.agents.verifier import verify_report
    store = demo[0]
    ids = ["T0049760", "T0049854", "T0049870", "T0049875", "T0049880", "T0049884", "T0049998", "T0049998"]
    claim = {"statement": f"{HUB} received ₹64,07,207 in 7 payments from eight distinct senders within 24 hours.",
             "txn_ids": ids, "accounts": [HUB]}
    result = verify_report({"verdict": "not_suspicious", "summary": "", "claims": [claim]}, store, HUB)
    problems = result["claim_results"][0]["problems"]
    assert problems and "eight senders" in problems[0]
    fixed = correct_claim(claim, problems, store, HUB)
    assert "seven distinct senders" in fixed["statement"] and len(fixed["txn_ids"]) == 7
    again = verify_report({"verdict": "not_suspicious", "summary": "", "claims": [fixed]}, store, HUB)
    assert again["passed"]
