"""The agent team, wired together with LangGraph.

Per case (multi-agent mode):

    investigate ─► classify_pattern ─► write_report ─► verify ─┬─► human_review ─┬─► finalize
         ▲                                  ▲                   │                 │
         │                                  └── fails, retry ───┘                 │
         └──────────────────────── analyst asks to reinvestigate ─────────────────┘

Single-agent mode (ablation): one agent with every tool does the whole job,
then writes the report. Verification is still *measured*, but not enforced,
so we can compare how often each setup makes unsupported claims.
"""
from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from parakh.agents import prompts
from parakh.agents.schemas import CaseReport, PatternFinding
from parakh.agents.verifier import verify_report
from parakh.config import Settings
from parakh.data.store import TransactionStore
from parakh.patterns.library import PatternLibrary
from parakh.tools.investigation import InvestigationTools

Mode = Literal["multi", "single"]
TOOL_RESULT_CHARS = 3500
EVIDENCE_CHARS = 14000


class CaseState(TypedDict, total=False):
    case_id: str
    account: str
    alert: dict
    mode: str
    analyst_notes: Annotated[list[str], operator.add]
    evidence: Annotated[list[dict], operator.add]
    timeline: Annotated[list[dict], operator.add]
    investigation_summary: str
    tool_calls: int
    pattern: dict | None
    report: dict | None
    report_attempts: int
    verification: dict | None
    decision: dict | None
    status: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _event(agent: str, action: str, detail: Any = None) -> dict:
    return {"time": _now(), "agent": agent, "action": action, "detail": detail}


def _text(msg) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def _signals_text(alert: dict) -> str:
    sigs = alert.get("signals", [])
    lines = [f"- {s['name']}: {s['detail']}" if isinstance(s, dict) else f"- {s}" for s in sigs]
    return "\n".join(lines) or "- (no screening signals; opened manually)"


def make_tools(itools: InvestigationTools, library: PatternLibrary | None = None) -> list[StructuredTool]:
    fns: list[Callable] = [
        itools.account_profile,
        itools.list_transactions,
        itools.trace_money,
        itools.find_cycles,
        itools.common_counterparties,
    ]
    tools = [StructuredTool.from_function(f, name=f.__name__, description=(f.__doc__ or f.__name__).strip())
             for f in fns]
    if library is not None:
        def search_patterns(description: str) -> list[dict]:
            """Find known laundering patterns and innocent look-alikes that match a plain-English description."""
            return library.search(description, k=4)

        tools.append(StructuredTool.from_function(search_patterns))
    return tools


def run_tool_loop(llm, tools: list[StructuredTool], system: str, user: str, budget: int,
                  config: dict | None, agent_name: str):
    """A small ReAct loop with a hard tool-call budget. Returns summary, evidence, events, calls."""
    by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools)
    msgs = [SystemMessage(system), HumanMessage(user)]
    evidence, events, calls = [], [], 0
    while True:
        ai: AIMessage = bound.invoke(msgs, config=config)
        msgs.append(ai)
        if not ai.tool_calls:
            break
        for call in ai.tool_calls:
            if calls >= budget:
                msgs.append(ToolMessage("Tool budget used up. Do not call more tools.", tool_call_id=call["id"]))
                continue
            calls += 1
            tool = by_name.get(call["name"])
            try:
                result = tool.invoke(call["args"]) if tool else {"error": f"unknown tool {call['name']}"}
            except Exception as exc:  # bad arguments from the model should not crash the case
                result = {"error": f"{type(exc).__name__}: {exc}"}
            text = json.dumps(result, default=str)[:TOOL_RESULT_CHARS]
            msgs.append(ToolMessage(text, tool_call_id=call["id"]))
            evidence.append({"tool": call["name"], "args": call["args"], "result": text})
            events.append(_event(agent_name, f"called {call['name']}", call["args"]))
        if calls >= budget:
            msgs.append(HumanMessage("Your tool budget is used up. Write your evidence summary now."))
            ai = llm.invoke(msgs, config=config)
            msgs.append(ai)
            break
    return _text(msgs[-1]), evidence, events, calls


def _evidence_text(state: CaseState) -> str:
    parts = [f"Investigator summary:\n{state.get('investigation_summary', '')}"]
    for e in state.get("evidence", []):
        parts.append(f"[{e['tool']} {json.dumps(e['args'])}]\n{e['result']}")
    text = "\n\n".join(parts)
    return text[:EVIDENCE_CHARS]


@dataclass
class Deps:
    llm: Any
    store: TransactionStore
    itools: InvestigationTools
    library: PatternLibrary
    cfg: Settings


def build_case_graph(deps: Deps, mode: Mode = "multi", human_review: bool = True,
                     enforce_verifier: bool = True, checkpointer=None):
    inv_tools = make_tools(deps.itools)
    all_tools = make_tools(deps.itools, deps.library)

    def investigate(state: CaseState, config) -> dict:
        notes = state.get("analyst_notes") or []
        notes_txt = ("\nAnalyst instructions for this round:\n" + "\n".join(f"- {n}" for n in notes) + "\n") if notes else ""
        if mode == "single":
            system = prompts.SINGLE_AGENT.format(account=state["account"], signals=_signals_text(state["alert"]),
                                                 budget=deps.cfg.max_tool_calls) + notes_txt
            tools, name = all_tools, "analyst"
        else:
            system = prompts.INVESTIGATOR.format(account=state["account"], signals=_signals_text(state["alert"]),
                                                 notes=notes_txt, budget=deps.cfg.max_tool_calls)
            tools, name = inv_tools, "investigator"
        summary, evidence, events, calls = run_tool_loop(
            deps.llm, tools, system, f"Investigate account {state['account']}.", deps.cfg.max_tool_calls, config, name)
        return {
            "investigation_summary": summary,
            "evidence": evidence,
            "timeline": [_event(name, "started investigation")] + events + [_event(name, "finished investigation", summary[:600])],
            "tool_calls": state.get("tool_calls", 0) + calls,
            "report_attempts": 0,
            "verification": None,
        }

    def classify_pattern(state: CaseState, config) -> dict:
        candidates = deps.library.search(state.get("investigation_summary", ""), k=5)
        cand_txt = "\n".join(f"- id={c['id']} ({c['kind']}): {c['description']} Tell-tale checks: {c['tell_tale_checks']}"
                             for c in candidates)
        model = deps.llm.with_structured_output(PatternFinding)
        finding: PatternFinding = model.invoke(
            [SystemMessage(prompts.PATTERN.format(candidates=cand_txt)),
             HumanMessage(state.get("investigation_summary", ""))], config=config)
        data = finding.model_dump()
        if data["pattern_id"] not in deps.library.ids():
            data["pattern_id"] = "none"
        data["candidates"] = [c["id"] for c in candidates]
        return {"pattern": data,
                "timeline": [_event("pattern", f"matched pattern '{data['pattern_id']}'",
                                    {"confidence": data["confidence"], "ruled_out": data["ruled_out"]})]}

    def write_report(state: CaseState, config) -> dict:
        attempts = state.get("report_attempts", 0)
        pattern = state.get("pattern")
        system = prompts.REPORT.format(account=state["account"])
        if pattern:
            context = f"Pattern finding:\n{json.dumps(pattern)}"
        else:
            context = "Allowed pattern ids: " + ", ".join(deps.library.ids() + ["none"])
        msgs = [SystemMessage(system),
                HumanMessage(f"Screening signals:\n{_signals_text(state['alert'])}\n\n{context}\n\nEvidence:\n{_evidence_text(state)}")]
        if attempts > 0 and state.get("verification"):
            msgs.append(HumanMessage(prompts.REPORT_RETRY.format(issues="\n".join(state["verification"]["issues"]))))
        report: CaseReport = deps.llm.with_structured_output(CaseReport).invoke(msgs, config=config)
        data = report.model_dump()
        label = "rewrote report after fact-check" if attempts else "wrote report"
        return {"report": data, "report_attempts": attempts + 1,
                "timeline": [_event("report", label, {"verdict": data["verdict"], "claims": len(data["claims"])})]}

    def verify(state: CaseState) -> dict:
        result = verify_report(state["report"], deps.store, state["account"])
        result["enforced"] = enforce_verifier
        msg = "all claims verified" if result["passed"] else f"{len(result['issues'])} problem(s) found"
        return {"verification": result,
                "timeline": [_event("verifier", msg, result["issues"][:5])]}

    def after_verify(state: CaseState) -> str:
        v = state["verification"]
        if enforce_verifier and not v["passed"] and state.get("report_attempts", 0) <= deps.cfg.max_report_retries:
            return "write_report"
        return "human_review" if human_review else "finalize"

    def human_review_node(state: CaseState) -> dict:
        decision = interrupt({"case_id": state.get("case_id"), "report": state.get("report"),
                              "verification": state.get("verification")})
        update: dict = {"decision": decision,
                        "timeline": [_event("reviewer", f"decided: {decision.get('action')}", decision.get("note"))]}
        if decision.get("action") == "reinvestigate" and decision.get("note"):
            update["analyst_notes"] = [decision["note"]]
        return update

    def after_review(state: CaseState) -> str:
        return "investigate" if (state.get("decision") or {}).get("action") == "reinvestigate" else "finalize"

    def finalize(state: CaseState) -> dict:
        decision = (state.get("decision") or {}).get("action")
        verdict = (state.get("report") or {}).get("verdict")
        if decision == "approve":
            status = "confirmed_suspicious" if verdict == "suspicious" else "closed_not_suspicious"
        elif decision == "reject":
            status = "closed_not_suspicious" if verdict == "suspicious" else "confirmed_suspicious"
        else:
            status = f"auto_{verdict}"
        return {"status": status, "timeline": [_event("system", f"case closed: {status}")]}

    g = StateGraph(CaseState)
    g.add_node("investigate", investigate)
    g.add_node("write_report", write_report)
    g.add_node("verify", verify)
    g.add_node("human_review", human_review_node)
    g.add_node("finalize", finalize)
    g.add_edge(START, "investigate")
    if mode == "multi":
        g.add_node("classify_pattern", classify_pattern)
        g.add_edge("investigate", "classify_pattern")
        g.add_edge("classify_pattern", "write_report")
    else:
        g.add_edge("investigate", "write_report")
    g.add_edge("write_report", "verify")
    g.add_conditional_edges("verify", after_verify, ["write_report", "human_review", "finalize"])
    g.add_conditional_edges("human_review", after_review, ["investigate", "finalize"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
