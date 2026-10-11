"""A scripted stand-in for a chat model, so the agent graph can be tested without API keys."""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, ToolMessage

from parakh.agents.schemas import CaseReport, Claim, PatternFinding, TriageDecision, TriageResult


def _all_text(msgs) -> str:
    return "\n".join(str(getattr(m, "content", m)) for m in msgs)


class _Structured:
    def __init__(self, parent: "FakeLLM", schema):
        self.parent, self.schema = parent, schema

    def invoke(self, msgs, config=None):
        p = self.parent
        if self.schema is TriageResult:
            alerts = json.loads(msgs[-1].content)
            return TriageResult(decisions=[
                TriageDecision(account=a["account"], open_case=a["score"] >= 2, priority=1 if a["score"] >= 3 else 2,
                               reason="several signals") for a in alerts])
        if self.schema is PatternFinding:
            return PatternFinding(pattern_id=p.pattern_id, kind="laundering", confidence=0.8,
                                  reasoning="Matches the tell-tale checks.", ruled_out=["payroll"])
        if self.schema is CaseReport:
            p.report_calls += 1
            text = _all_text(msgs)
            txns = re.findall(r'"txn_id": "(T\d+)", "time": "[^"]+", "from": "(\w+)", "to": "(\w+)", "amount": ([\d.]+)', text)
            claims = [Claim(statement=f"{f} sent ₹{float(a):,.2f} to {t}.", txn_ids=[tid], accounts=[f, t], amount=float(a))
                      for tid, f, t, a in txns[:3]]
            if p.bad_first_report and p.report_calls == 1:
                claims.append(Claim(statement="An invented transfer of ₹12,34,567.", txn_ids=["T9999999"], amount=1234567))
            return CaseReport(verdict="suspicious" if len(claims) >= 2 else "not_suspicious", risk_score=80,
                              pattern_id=p.pattern_id, summary="Money moved quickly through the account.",
                              claims=claims, recommended_action="File a suspicious transaction report.")
        raise NotImplementedError(self.schema)


class FakeLLM:
    def __init__(self, pattern_id: str = "fan_out", bad_first_report: bool = False, script=None):
        self.pattern_id = pattern_id
        self.bad_first_report = bad_first_report
        self.report_calls = 0
        self.script = script or ["account_profile", "list_transactions"]

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return _Structured(self, schema)

    def invoke(self, msgs, config=None):
        done = sum(isinstance(m, ToolMessage) for m in msgs)
        acc = re.search(r"account (AC\d+)", _all_text(msgs[:2]))
        account = acc.group(1) if acc else "AC000000"
        if done < len(self.script):
            name = self.script[done]
            return AIMessage("", tool_calls=[{"name": name, "args": {"account": account}, "id": f"call{done}"}])
        return AIMessage(f"Evidence summary for {account}: large inflow forwarded within hours.")
