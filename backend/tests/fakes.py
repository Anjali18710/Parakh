"""A scripted stand-in for a chat model, so the agent graph can be tested without API keys."""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, ToolMessage

from parakh.agents.schemas import CaseReport, Check, Claim, PatternFinding, TriageDecision, TriageResult


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
            txns = re.findall(r"(T\d{7,8}) \| [^|]+ \| (\w+) -> (\w+) \| [^(]+\(([\d.]+)\)", text)
            claims = [Claim(statement=f"{f} sent ₹{float(a):,.2f} to {t}.", txn_ids=[tid], accounts=[f, t], amount=float(a))
                      for tid, f, t, a in sorted(txns, key=lambda x: -float(x[3]))[:3]]
            checks = [Check(description="No loops of large transfers involve this account.", tool="account_profile")]
            if p.bad_first_report and p.report_calls == 1:
                claims.append(Claim(statement="An invented transfer of ₹12,34,567.", txn_ids=["T9999999"], amount=1234567))
            return CaseReport(verdict="suspicious" if len(claims) >= 2 else "not_suspicious", risk_score=80,
                              pattern_id=p.pattern_id, summary="Money moved quickly through the account.",
                              claims=claims, checks=checks, recommended_action="File a suspicious transaction report.")
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


_LEDGER = re.compile(r"(T\d{7,8}) \| [^|]+ \| (\w+) -> (\w+) \| [^(]+\(([\d.]+)\)")


class MisbehavingLLM:
    """Reproduces the failure modes seen with the real model:

    * pattern: answers each pattern question with the next id in `pattern_answers`
      (e.g. "fan_out" then "none", as gpt-oss did for a gather-scatter hub);
    * report: if `bad_arithmetic`, writes totals it computed itself, off by a few hundred rupees,
      and puts a "nothing found" statement in claims with no citations, on every attempt;
      otherwise uses the {total}/{count} placeholders as instructed;
    * verdict: says 'suspicious' first, but corrects it when told it contradicts the pattern.
    """

    def __init__(self, pattern_answers=("fan_out", "none"), bad_arithmetic=True):
        self.pattern_answers = list(pattern_answers)
        self.bad_arithmetic = bad_arithmetic
        self.pattern_calls = 0
        self.report_calls = 0

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return _Misbehaving(self, schema)

    def invoke(self, msgs, config=None):
        script = ["account_profile", "trace_money", "trace_money", "common_counterparties", "find_cycles"]
        done = sum(isinstance(m, ToolMessage) for m in msgs)
        account = re.search(r"account (AC\d+)", _all_text(msgs[:2])).group(1)
        if done < len(script):
            args = {"account": account}
            if script[done] == "trace_money":
                args["direction"] = "in" if done == 1 else "out"
            return AIMessage("", tool_calls=[{"name": script[done], "args": args, "id": f"c{done}"}])
        return AIMessage(f"{account} received many large payments and paid them out within a day.")


class _Misbehaving:
    def __init__(self, parent: MisbehavingLLM, schema):
        self.p, self.schema = parent, schema

    def invoke(self, msgs, config=None):
        p = self.p
        if self.schema is PatternFinding:
            answer = p.pattern_answers[min(p.pattern_calls, len(p.pattern_answers) - 1)]
            p.pattern_calls += 1
            return PatternFinding(pattern_id=answer, kind="unclear", confidence=0.85,
                                  reasoning="Looks like it.", ruled_out=["cycle"])
        if self.schema is CaseReport:
            p.report_calls += 1
            text = _all_text(msgs)
            account = re.search(r"case file for account (AC\d+)", text).group(1)
            rows = [(t, f, to, float(a)) for t, f, to, a in _LEDGER.findall(text)]
            inflow = [r for r in rows if r[2] == account]
            outflow = [r for r in rows if r[1] == account]
            claims = []
            for label, side in (("received", inflow), ("sent", outflow)):
                if not side:
                    continue
                ids = [r[0] for r in side]
                if p.bad_arithmetic:
                    wrong = sum(r[3] for r in side) + 357  # the kind of slip the real model made
                    stmt = f"{account} {label} ₹{wrong:,.0f} in {len(side)} payments."
                    claims.append(Claim(statement=stmt, txn_ids=ids, accounts=[account], amount=wrong))
                else:
                    claims.append(Claim(statement=f"{account} {label} {{total}} in {{count}} payments.",
                                        txn_ids=ids, accounts=[account]))
            if p.bad_arithmetic:
                claims.append(Claim(statement="No loops involving the account were found.", txn_ids=[]))
            told_off = "contradicts the pattern finding" in text
            pattern = re.search(r'"kind": "(\w+)"', text)
            innocent = bool(pattern and pattern.group(1) == "innocent")
            verdict = "not_suspicious" if (told_off or (innocent and p.report_calls > 1)) else "suspicious"
            return CaseReport(verdict=verdict, risk_score=40 if verdict == "not_suspicious" else 85,
                              pattern_id="fan_in", summary="Money came in from {senders} accounts totalling {total_in}.",
                              claims=claims, checks=[Check(description="No loops found.", tool="find_cycles")],
                              recommended_action="Review.")
        raise NotImplementedError(self.schema)
