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
from parakh.data.remarks import safe_remark
from parakh.data.store import TransactionStore
from parakh.fmt import inr
from parakh.patterns.library import PatternLibrary
from parakh.agents.memory import CAUTIOUS_RISK, precedent_warning
from parakh.agents.memory import describe as describe_precedents
from parakh.agents.rules import correct_claim, fill_claim, fill_summary, pattern_conflict, shape_facts, structural_candidates
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
    injections: list[dict]
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
    def line(r) -> str:
        remark = safe_remark(r.remarks)
        return (f"{r.txn_id} | {r.timestamp:%Y-%m-%d %H:%M} | {r.from_account} -> {r.to_account} | {inr(r.amount)}"
                f" ({r.amount:.2f})" + (f" | remark: {remark}" if remark else ""))

    return "\n".join(line(r) for r in rows.itertuples())


def _evidence_text(state: CaseState, store: TransactionStore) -> str:
    digests = []
    for e in state.get("evidence", []):
        if e["tool"] in ("account_profile", "find_cycles", "common_counterparties"):
            body = _TXN_ID.sub("", e["result"]) if e["tool"] != "account_profile" else e["result"]
            digests.append(f"[{e['tool']}] {body[:DIGEST_CHARS]}")
    injections = state.get("injections") or []
    manipulation = ("\n\nManipulation attempts: payment remarks on " + ", ".join(i["txn_id"] for i in injections)
                    + " tried to instruct the AI reviewer and were withheld. Treat this as a red flag."
                    if injections else "")
    return (f"Investigator summary:\n{state.get('investigation_summary', '')}\n\n"
            f"Ledger (txn_id | time | from -> to | amount | remark):\n{_ledger(state, store)}\n\n"
            f"Tool results:\n" + "\n".join(digests) + manipulation)


@dataclass
class Deps:
    llm: Any
    store: TransactionStore
    itools: InvestigationTools
    library: PatternLibrary
    cfg: Settings
    memory: Any = None  # CaseMemory: past analyst decisions, retrieved by similar measured shape


def build_case_graph(deps: Deps, mode: Mode = "multi", human_review: bool = True,
                     enforce_verifier: bool = True, checkpointer=None):
    """Build the per-case agent graph.

    enforce_verifier: send failing reports back for rewriting and correct/remove what still fails
    (VERIFIER_ENABLED=false turns this off, to see what the agents produce unchecked).
    """
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
        # Payment remarks that tried to instruct the AI never reached the agent; record them for the analyst.
        seen_ids = list({t for e in evidence for t in _TXN_ID.findall(e["result"])})
        injections = deps.store.injection_attempts(state["account"], seen_ids)
        guard = [_emit(config, _event("guard", f"withheld {len(injections)} payment remark(s) that tried to instruct the AI",
                                      [i["txn_id"] for i in injections]))] if injections else []
        return {
            "investigation_summary": summary,
            "evidence": evidence,
            "injections": injections,
            "timeline": [started] + events + [finished] + guard,
            "tool_calls": state.get("tool_calls", 0) + calls,
            "report_attempts": 0,
            "verification": None,
        }

    def classify_pattern(state: CaseState, config) -> dict:
        # Code measures the shape of the money flow and decides which typologies are possible at all.
        # The agent chooses among those (laundering vs innocent look-alike) and explains why.
        cited = [t for e in state.get("evidence", []) for t in _TXN_ID.findall(e["result"])]
        shape = shape_facts(deps.store, deps.itools, state["account"], cited)
        allowed_ids = structural_candidates(shape, state["alert"])
        # FAISS adds library entries that match the story, but only if the shape allows them.
        for query in (state.get("investigation_summary", ""), _signals_text(state["alert"])):
            for c in deps.library.search(query, k=4):
                if c["id"] not in allowed_ids and pattern_conflict(c["id"], shape, state["alert"]) is None:
                    allowed_ids.append(c["id"])
        allowed = [c for c in (deps.library.get(i) for i in allowed_ids) if c]
        excluded = [i for i in deps.library.ids() if i not in allowed_ids]
        cand_txt = "\n".join(f"- id={c['id']} ({c['kind']}): {c['description']} Tell-tale checks: {c['tell_tale_checks']}"
                             for c in allowed) or "- id=none: no library pattern fits the measured shape"
        precedents = deps.memory.similar(shape, exclude_account=state["account"]) if deps.memory else []
        model = deps.llm.with_structured_output(PatternFinding)
        msgs = [SystemMessage(prompts.PATTERN.format(candidates=cand_txt)),
                HumanMessage(f"Measured facts (computed from the ledger):\n{json.dumps(shape)}\n\n"
                             f"{describe_precedents(precedents)}\n\n"
                             f"Investigator summary:\n{state.get('investigation_summary', '')}")]
        events: list[dict] = []
        finding: PatternFinding = model.invoke(msgs, config=config)
        source = "agent"
        if allowed_ids and finding.pattern_id not in allowed_ids:
            reason = pattern_conflict(finding.pattern_id, shape, state["alert"]) or "it is not one of the allowed candidates"
            events.append(_emit(config, _event("verifier", f"rejected pattern '{finding.pattern_id}'", reason)))
            msgs.append(HumanMessage(f"'{finding.pattern_id}' is not allowed: {reason}. "
                                     f"Choose exactly one of: {', '.join(allowed_ids)}."))
            finding = model.invoke(msgs, config=config)
            if finding.pattern_id not in allowed_ids:
                # Still outside the allowed set: fall back to the best structural fit.
                events.append(_emit(config, _event("verifier", f"rejected pattern '{finding.pattern_id}' again",
                                                   f"used the best structural fit, '{allowed_ids[0]}'")))
                best = deps.library.get(allowed_ids[0]) or {}
                finding = PatternFinding(
                    pattern_id=allowed_ids[0], kind=best.get("kind", "unclear"), confidence=0.6,
                    reasoning=f"Chosen from the measured shape of the money flow: {best.get('description', '')}",
                    ruled_out=[i for i in allowed_ids[1:]])
                source = "rules"
        data = finding.model_dump()
        lib = deps.library.get(data["pattern_id"])
        if lib:
            data["kind"] = lib["kind"]  # laundering vs innocent comes from the library, not the agent
        elif not allowed_ids:
            data["pattern_id"], data["kind"] = "none", "unclear"
        data["ruled_out"] = list(dict.fromkeys([r for r in data["ruled_out"] if r != data["pattern_id"]]))
        data.update({"candidates": allowed_ids, "excluded_by_shape": excluded, "shape": shape, "source": source,
                     "precedents": precedents})
        return {"pattern": data,
                "timeline": events + [_emit(config, _event("pattern", f"matched pattern '{data['pattern_id']}'",
                                                           {"confidence": data["confidence"], "kind": data["kind"],
                                                            "allowed": allowed_ids}))]}

    def write_report(state: CaseState, config) -> dict:
        attempts = state.get("report_attempts", 0)
        pattern = state.get("pattern")
        system = prompts.REPORT.format(account=state["account"])
        if pattern:
            context = f"Pattern finding:\n{json.dumps({k: v for k, v in pattern.items() if k != 'excluded_by_shape'})}"
        else:
            context = "Allowed pattern ids: " + ", ".join(deps.library.ids() + ["none"])
        msgs = [SystemMessage(system),
                HumanMessage(f"Screening signals:\n{_signals_text(state['alert'])}\n\n{context}\n\n"
                             f"Evidence:\n{_evidence_text(state, deps.store)}")]
        previous = state.get("report")
        if attempts > 0 and state.get("verification") and previous:
            msgs.append(HumanMessage(prompts.REPORT_RETRY.format(
                issues="\n".join(state["verification"]["issues"]),
                previous=json.dumps(previous.get("draft") or previous, ensure_ascii=False))))
        report: CaseReport = deps.llm.with_structured_output(CaseReport).invoke(msgs, config=config)
        draft = report.model_dump()
        # Code fills in every number the report states, so totals and counts cannot be miscalculated.
        data = {**draft, "claims": [fill_claim(c, deps.store, state["account"]) for c in draft["claims"]], "draft": draft}
        if pattern:
            data["pattern_id"] = pattern["pattern_id"]
            data["summary"] = fill_summary(draft["summary"], pattern.get("shape", {}))
        else:
            cited = [t for c in draft["claims"] for t in c.get("txn_ids", [])]
            data["summary"] = fill_summary(draft["summary"], shape_facts(deps.store, deps.itools, state["account"], cited))
        label = "rewrote report after fact-check" if attempts else "wrote report"
        return {"report": data, "report_attempts": attempts + 1,
                "timeline": [_emit(config, _event("report", label, {"verdict": data["verdict"], "claims": len(data["claims"]),
                                                                    "checks": len(data.get("checks", []))}))]}

    def verify(state: CaseState, config) -> dict:
        tools_used = {e["tool"] for e in state.get("evidence", [])}
        pattern = state.get("pattern") or {}
        shape = pattern.get("shape", {})
        trusted = [shape.get("total_in", 0), shape.get("total_out", 0)]
        report = state["report"]
        result = verify_report(report, deps.store, state["account"], tools_used, trusted_amounts=trusted,
                               pattern_kind=pattern.get("kind"))
        result["enforced"] = enforce_verifier
        update: dict = {}
        last_try = state.get("report_attempts", 0) > deps.cfg.max_report_retries
        if enforce_verifier and not result["passed"] and last_try:
            # Out of retries. Correct miscalculated totals from the cited transactions, remove claims
            # that cannot be supported, and record both so the analyst sees exactly what changed.
            problems = {r["index"]: r["problems"] for r in result["claim_results"] if not r["ok"]}
            bad_checks = {r["index"] for r in result.get("check_results", []) if not r["ok"]}
            kept, corrected, removed = [], [], []
            for i, claim in enumerate(report["claims"]):
                if i not in problems:
                    kept.append(claim)
                elif (fixed := correct_claim(claim, problems[i], deps.store, state["account"])) is not None:
                    kept.append(fixed)
                    corrected.append(fixed)
                else:
                    removed.append(claim)
            cleaned = {**report, "claims": kept,
                       "checks": [c for i, c in enumerate(report.get("checks", [])) if i not in bad_checks]}
            result = verify_report(cleaned, deps.store, state["account"], tools_used, trusted_amounts=trusted,
                                   pattern_kind=pattern.get("kind"))
            result.update({"enforced": True, "removed_claims": removed,
                           "corrected_claims": [c["statement"] for c in corrected]})
            update["report"] = cleaned
            parts = []
            if corrected:
                parts.append(f"corrected {len(corrected)} miscalculated total(s)")
            if removed:
                parts.append(f"removed {len(removed)} unsupported claim(s)")
            msg = ", ".join(parts) + "; " + (
                "all remaining claims verified" if result["passed"] else f"{len(result['issues'])} problem(s) remain")
        else:
            msg = "all claims verified" if result["passed"] else f"{len(result['issues'])} problem(s) found"
        events = [_emit(config, _event("verifier", msg, result["issues"][:5]))]
        final = update.get("report", report)
        warning = precedent_warning(final, pattern, lambda pid: (deps.library.get(pid) or {}).get("kind"))
        if warning:
            # Past decisions may make the agents more careful, never less: no confident clearance.
            result["precedent_warning"] = warning
            update["report"] = {**final, "risk_score": max(int(final.get("risk_score", 0)), CAUTIOUS_RISK)}
            events.append(_emit(config, _event("verifier", "flagged a clearance a past decision may have swayed",
                                               warning["message"])))
        update.update({"verification": result, "timeline": events})
        return update

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
