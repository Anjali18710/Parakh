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
import re
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
from parakh.fmt import inr
from parakh.patterns.library import PatternLibrary
from parakh.tools.flow import case_flow
from parakh.tools.investigation import InvestigationTools

Mode = Literal["multi", "single"]
TOOL_RESULT_CHARS = 2000   # every tool result is re-sent on each later step, so it is kept small
DIGEST_CHARS = 500
LEDGER_ROWS = 45
_TXN_ID = re.compile(r"\bT\d{7,8}\b")


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


def _emit(config: dict | None, event: dict) -> dict:
    """Send an event to the live trail (if the caller provided a sink) and return it for the state."""
    sink = ((config or {}).get("configurable") or {}).get("event_sink")
    if sink:
        try:
            sink(event)
        except Exception:  # the live trail is a convenience; never let it break an investigation
            pass
    return event


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
            text = json.dumps(result, default=str, separators=(",", ":"), ensure_ascii=False)[:TOOL_RESULT_CHARS]
            msgs.append(ToolMessage(text, tool_call_id=call["id"]))
            evidence.append({"tool": call["name"], "args": call["args"], "result": text})
            events.append(_emit(config, _event(agent_name, f"called {call['name']}", call["args"])))
        if calls >= budget:
            msgs.append(HumanMessage("Your tool budget is used up. Write your evidence summary now."))
            ai = llm.invoke(msgs, config=config)
            msgs.append(ai)
            break
    return _text(msgs[-1]), evidence, events, calls


def _ledger(state: CaseState, store: TransactionStore) -> str:
    """Every transaction any tool surfaced, looked up again in the store so the report cites exact figures."""
    ids: list[str] = []
    for e in state.get("evidence", []):
        ids += _TXN_ID.findall(e["result"])
    ids += _TXN_ID.findall(state.get("investigation_summary", ""))
    rows = store.get_txns(list(dict.fromkeys(ids)))
    if rows.empty:
        return "(no transactions surfaced)"
    acc = state["account"]
    # Keep the case account's own transactions first, then the largest of the rest.
    rows = rows.assign(own=(rows.from_account == acc) | (rows.to_account == acc))
    rows = rows.sort_values(["own", "amount"], ascending=[False, False]).head(LEDGER_ROWS).sort_values("timestamp")
    return "\n".join(f"{r.txn_id} | {r.timestamp:%Y-%m-%d %H:%M} | {r.from_account} -> {r.to_account} | {inr(r.amount)}"
                     f" ({r.amount:.2f})" for r in rows.itertuples())


def _evidence_text(state: CaseState, store: TransactionStore) -> str:
    digests = []
    for e in state.get("evidence", []):
        if e["tool"] in ("account_profile", "find_cycles", "common_counterparties"):
            body = _TXN_ID.sub("", e["result"]) if e["tool"] != "account_profile" else e["result"]
            digests.append(f"[{e['tool']}] {body[:DIGEST_CHARS]}")
    return (f"Investigator summary:\n{state.get('investigation_summary', '')}\n\n"
            f"Ledger (txn_id | time | from -> to | amount):\n{_ledger(state, store)}\n\n"
            f"Tool results:\n" + "\n".join(digests))


def shape_facts(store: TransactionStore, itools: InvestigationTools, account: str, cited: list[str]) -> dict:
    """Facts about the shape of the money flow, computed in code so the pattern agent cannot misread them."""
    flow = case_flow(store, account, cited)
    rel = (itools.account_profile(account) or {}).get("relationships", {})
    return {
        "senders_in_window": flow.get("senders", 0),
        "recipients_in_window": flow.get("recipients", 0),
        "share_passed_through": flow.get("passed_through", 0),
        "activity_span_hours": flow.get("span_hours", 0),
        **{k: v for k, v in rel.items() if k != "note"},
    }


def structural_candidates(shape: dict, alert: dict) -> list[str]:
    """Patterns the flow's shape makes plausible, laundering and innocent alike."""
    s, r = shape.get("senders_in_window", 0), shape.get("recipients_in_window", 0)
    ids: list[str] = []
    if s >= 3 and r >= 3:
        ids += ["gather_scatter", "bipartite"]
    if s >= 3 and r <= 2:
        ids += ["fan_in", "wholesale"]
    if r >= 3 and s <= 2:
        ids += ["fan_out", "scatter_gather", "payroll"]
    if s <= 2 and r <= 2:
        ids += ["layering_chain", "investment", "property"]
    if any((sig.get("name") if isinstance(sig, dict) else "") == "in_cycle" for sig in alert.get("signals", [])):
        ids += ["cycle", "family"]
    return ids


def pattern_conflict(pattern_id: str, shape: dict, alert: dict) -> str | None:
    """Does the chosen pattern contradict the measured shape of the money flow? Returns the reason, or None."""
    s, r = shape.get("senders_in_window", 0), shape.get("recipients_in_window", 0)
    repeat_in = shape.get("inflow_from_repeat_senders", 0)
    repeat_out = shape.get("outflow_to_repeat_recipients", 0)
    rules = {
        "fan_in": (r <= 2, f"fan_in forwards to one or two accounts, but money went to {r} recipients"),
        "fan_out": (s <= 2, f"fan_out starts from one or two sources, but money came from {s} senders"),
        "gather_scatter": (s >= 3 and r >= 3, f"gather_scatter needs 3+ senders and 3+ recipients, measured {s} and {r}"),
        "layering_chain": (s <= 2 and r <= 2, f"a layering chain passes money one-to-one, measured {s} senders and {r} recipients"),
        "wholesale": (repeat_in >= 0.5, f"wholesale trade has regular buyers, but only {repeat_in:.0%} of money came from repeat senders"),
        "payroll": (repeat_out >= 0.5, f"payroll pays the same staff each month, but only {repeat_out:.0%} went to repeat recipients"),
    }
    ok, reason = rules.get(pattern_id, (True, ""))
    return None if ok else reason


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
        started = _emit(config, _event(name, "started investigation"))
        summary, evidence, events, calls = run_tool_loop(
            deps.llm, tools, system, f"Investigate account {state['account']}.", deps.cfg.max_tool_calls, config, name)
        finished = _emit(config, _event(name, "finished investigation", summary[:700]))
        return {
            "investigation_summary": summary,
            "evidence": evidence,
            "timeline": [started] + events + [finished],
            "tool_calls": state.get("tool_calls", 0) + calls,
            "report_attempts": 0,
            "verification": None,
        }

    def classify_pattern(state: CaseState, config) -> dict:
        # Search with both the investigator's story and the raw screening signals, so a pattern the
        # summary describes loosely (e.g. a hub that both collects and pays out) is still offered.
        # Hybrid retrieval: patterns the flow's measured shape makes plausible, plus FAISS matches
        # for the investigator's story and the screening signals.
        cited = [t for e in state.get("evidence", []) for t in _TXN_ID.findall(e["result"])]
        shape = shape_facts(deps.store, deps.itools, state["account"], cited)
        candidates: list[dict] = []
        for pid in structural_candidates(shape, state["alert"]):
            c = deps.library.get(pid)
            if c:
                candidates.append(c)
        for query, k in ((state.get("investigation_summary", ""), 4), (_signals_text(state["alert"]), 3)):
            for c in deps.library.search(query, k=k):
                if c["id"] not in {x["id"] for x in candidates}:
                    candidates.append(c)
        candidates = candidates[:8]
        cand_txt = "\n".join(f"- id={c['id']} ({c['kind']}): {c['description']} Tell-tale checks: {c['tell_tale_checks']}"
                             for c in candidates)
        model = deps.llm.with_structured_output(PatternFinding)
        msgs = [SystemMessage(prompts.PATTERN.format(candidates=cand_txt)),
                HumanMessage(f"Measured facts (computed from the ledger, trust these over the narrative):\n"
                             f"{json.dumps(shape)}\n\nInvestigator summary:\n{state.get('investigation_summary', '')}")]
        finding: PatternFinding = model.invoke(msgs, config=config)
        events = []
        # Pattern fact-check: the choice must fit the measured shape. One retry with the reason.
        conflict = pattern_conflict(finding.pattern_id, shape, state["alert"])
        if conflict:
            events.append(_emit(config, _event("verifier", f"rejected pattern '{finding.pattern_id}'", conflict)))
            msgs.append(HumanMessage(f"Your choice '{finding.pattern_id}' contradicts the measured facts: {conflict}. "
                                     f"Choose again from the candidates."))
            finding = model.invoke(msgs, config=config)
        data = finding.model_dump()
        data["rechecked"] = bool(conflict)
        if data["pattern_id"] not in deps.library.ids():
            data["pattern_id"] = "none"
        data["candidates"] = [c["id"] for c in candidates]
        data["shape"] = shape
        return {"pattern": data,
                "timeline": events + [_emit(config, _event("pattern", f"matched pattern '{data['pattern_id']}'",
                                                  {"confidence": data["confidence"], "ruled_out": data["ruled_out"]}))]}

    def write_report(state: CaseState, config) -> dict:
        attempts = state.get("report_attempts", 0)
        pattern = state.get("pattern")
        system = prompts.REPORT.format(account=state["account"])
        if pattern:
            context = f"Pattern finding:\n{json.dumps(pattern)}"
        else:
            context = "Allowed pattern ids: " + ", ".join(deps.library.ids() + ["none"])
        msgs = [SystemMessage(system),
                HumanMessage(f"Screening signals:\n{_signals_text(state['alert'])}\n\n{context}\n\n"
                             f"Evidence:\n{_evidence_text(state, deps.store)}")]
        if attempts > 0 and state.get("verification"):
            msgs.append(HumanMessage(prompts.REPORT_RETRY.format(issues="\n".join(state["verification"]["issues"]))))
        report: CaseReport = deps.llm.with_structured_output(CaseReport).invoke(msgs, config=config)
        data = report.model_dump()
        label = "rewrote report after fact-check" if attempts else "wrote report"
        return {"report": data, "report_attempts": attempts + 1,
                "timeline": [_emit(config, _event("report", label, {"verdict": data["verdict"], "claims": len(data["claims"]),
                                                                    "checks": len(data.get("checks", []))}))]}

    def verify(state: CaseState, config) -> dict:
        tools_used = {e["tool"] for e in state.get("evidence", [])}
        result = verify_report(state["report"], deps.store, state["account"], tools_used)
        result["enforced"] = enforce_verifier
        msg = "all claims verified" if result["passed"] else f"{len(result['issues'])} problem(s) found"
        return {"verification": result,
                "timeline": [_emit(config, _event("verifier", msg, result["issues"][:5]))]}

    def after_verify(state: CaseState) -> str:
        v = state["verification"]
        if enforce_verifier and not v["passed"] and state.get("report_attempts", 0) <= deps.cfg.max_report_retries:
            return "write_report"
        return "human_review" if human_review else "finalize"

    def human_review_node(state: CaseState, config) -> dict:
        decision = interrupt({"case_id": state.get("case_id"), "report": state.get("report"),
                              "verification": state.get("verification")})
        update: dict = {"decision": decision,
                        "timeline": [_emit(config, _event("reviewer", f"decided: {decision.get('action')}",
                                                          decision.get("note")))]}
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
