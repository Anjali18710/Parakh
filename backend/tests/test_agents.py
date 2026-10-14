from fakes import FakeLLM

from parakh.agents.graph import Deps, build_case_graph
from parakh.agents.service import CaseService
from parakh.agents.triage import brief, triage_by_score, triage_with_llm
from parakh.config import Settings
from parakh.patterns.library import PatternLibrary
from parakh.storage.cases import MemoryRepository
from parakh.tools.investigation import InvestigationTools
from parakh.tools.screening import screen


def service(store, llm=None):
    return CaseService(cfg=Settings(), store=store, repo=MemoryRepository(), llm=llm or FakeLLM(),
                       library=PatternLibrary())


def test_pattern_library_ranks_relevant_pattern_first():
    lib = PatternLibrary()
    assert lib.search("same employees paid every month on the same day", 1)[0]["id"] == "payroll"


def test_verifier_failure_triggers_rewrite(store):
    llm = FakeLLM(bad_first_report=True)
    deps = Deps(llm, store, InvestigationTools(store), PatternLibrary(), Settings())
    graph = build_case_graph(deps, "multi", human_review=False)
    acc = screen(store)[0].account
    state = graph.invoke({"case_id": "t1", "account": acc, "alert": {"signals": []}, "mode": "multi",
                          "analyst_notes": [], "evidence": [], "timeline": [], "tool_calls": 0})
    assert state["report_attempts"] == 2
    assert state["verification"]["passed"]
    assert state["status"] in ("auto_suspicious", "auto_not_suspicious")
    assert state["pattern"]["pattern_id"] in state["pattern"]["candidates"] + ["none"]


def test_verifier_not_enforced_keeps_bad_report(store):
    deps = Deps(FakeLLM(bad_first_report=True), store, InvestigationTools(store), PatternLibrary(), Settings())
    graph = build_case_graph(deps, "multi", human_review=False, enforce_verifier=False)
    acc = screen(store)[0].account
    state = graph.invoke({"case_id": "t2", "account": acc, "alert": {"signals": []}, "mode": "multi",
                          "analyst_notes": [], "evidence": [], "timeline": [], "tool_calls": 0})
    assert state["report_attempts"] == 1
    assert not state["verification"]["passed"]


def test_tool_budget_is_respected(store):
    cfg = Settings()
    cfg.max_tool_calls = 2
    llm = FakeLLM(script=["account_profile"] * 6)
    deps = Deps(llm, store, InvestigationTools(store), PatternLibrary(), cfg)
    graph = build_case_graph(deps, "single", human_review=False)
    acc = screen(store)[0].account
    state = graph.invoke({"case_id": "t3", "account": acc, "alert": {"signals": []}, "mode": "single",
                          "analyst_notes": [], "evidence": [], "timeline": [], "tool_calls": 0})
    assert state["tool_calls"] == 2
    assert state.get("pattern") is None  # single agent has no separate pattern agent


def test_full_run_with_human_review_and_reinvestigation(store):
    svc = service(store)
    run = svc.start_run(2, background=False)
    run = svc.repo.get("runs", run["_id"])
    assert run["status"] == "done" and len(run["case_ids"]) == 2
    case_id = run["case_ids"][0]
    assert svc.repo.get("cases", case_id)["status"] == "awaiting_review"

    svc.decide(case_id, "reinvestigate", "check the source of funds", background=False)
    case = svc.repo.get("cases", case_id)
    assert case["status"] == "awaiting_review"
    assert case["analyst_notes"] == ["check the source of funds"]
    assert any(e["agent"] == "reviewer" for e in case["timeline"])

    svc.decide(case_id, "reject", "this is payroll", background=False)
    assert svc.repo.get("cases", case_id)["status"] == "closed_not_suspicious"


def test_decision_after_restart_still_works(store):
    svc = service(store)
    case_id = svc.open_case(screen(store)[0].account, background=False)
    fresh = service(store)  # new process: paused LangGraph run is gone, case is still stored
    fresh.repo = svc.repo
    fresh.decide(case_id, "approve", background=False)
    assert svc.repo.get("cases", case_id)["status"] == "confirmed_suspicious"


def test_triage_selection_respects_budget(store):
    alerts = [brief(a.to_dict()) for a in screen(store)]
    chosen = [d for d in triage_with_llm(FakeLLM(), alerts, 3) if d["selected"]]
    assert 0 < len(chosen) <= 3
    assert sum(d["selected"] for d in triage_by_score(alerts, 3)) == 3


def test_structural_candidates_follow_the_shape_of_the_flow():
    from parakh.agents.rules import structural_candidates
    hub = {"senders_in_window": 8, "recipients_in_window": 6}
    assert structural_candidates(hub, {})[0] == "gather_scatter" and "fan_in" not in structural_candidates(hub, {})
    trader = {"senders_in_window": 13, "recipients_in_window": 1, "inflow_from_repeat_senders": 1.0}
    assert structural_candidates(trader, {}) == ["wholesale"]
    loop = {"senders_in_window": 1, "recipients_in_window": 1, "in_loop_within_7_days": True}
    assert structural_candidates(loop, {})[0] == "cycle" and "layering_chain" not in structural_candidates(loop, {})


def test_relationships_separate_business_from_laundering(store):
    tools = InvestigationTools(store)
    rel = tools.account_profile(store.df.to_account.value_counts().index[0])["relationships"]
    assert 0 <= rel["inflow_from_repeat_senders"] <= 1 and "outflow_schedule" in rel


def test_pattern_conflicts_with_measured_shape():
    from parakh.agents.rules import pattern_conflict
    hub = {"senders_in_window": 8, "recipients_in_window": 6}
    assert pattern_conflict("fan_in", hub) is not None
    assert pattern_conflict("gather_scatter", hub) is None
    trader = {"senders_in_window": 13, "recipients_in_window": 1, "inflow_from_repeat_senders": 1.0}
    assert pattern_conflict("wholesale", trader) is None
    assert pattern_conflict("fan_in", trader) is not None
    assert pattern_conflict("wholesale", {**trader, "inflow_from_repeat_senders": 0.0}) is not None
