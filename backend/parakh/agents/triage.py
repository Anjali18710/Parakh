"""Triage agent and the rules-only baseline."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from parakh.agents import prompts
from parakh.agents.schemas import TriageResult

BATCH = 30


def triage_with_llm(llm, alerts: list[dict], budget: int, config: dict | None = None) -> list[dict]:
    """Ask the triage agent which alerts to open. Alerts are sent in batches to keep prompts small."""
    model = llm.with_structured_output(TriageResult)
    decisions: list[dict] = []
    for i in range(0, len(alerts), BATCH):
        batch = alerts[i:i + BATCH]
        result: TriageResult = model.invoke(
            [SystemMessage(prompts.TRIAGE.format(budget=budget)),
             HumanMessage(json.dumps(batch, ensure_ascii=False))], config=config)
        known = {a["account"] for a in batch}
        decisions += [d.model_dump() for d in result.decisions if d.account in known]
    opened = [d for d in decisions if d["open_case"]]
    opened.sort(key=lambda d: d["priority"])
    chosen = {d["account"] for d in opened[:budget]}
    for d in decisions:
        d["selected"] = d["account"] in chosen
    return decisions


def triage_by_score(alerts: list[dict], budget: int) -> list[dict]:
    """Baseline: open the highest-scoring alerts."""
    ranked = sorted(alerts, key=lambda a: -a["score"])
    return [{"account": a["account"], "open_case": True, "priority": 1, "selected": i < budget,
             "reason": f"rule score {a['score']}"} for i, a in enumerate(ranked)]


# --------------------------------------------------------------- rules-only case verdict
def rules_report(alert: dict, threshold: float = 2.0) -> dict:
    """What a purely rule-based system would conclude about an alert, without any agent."""
    names = {s["name"] for s in alert["signals"]}
    if "in_cycle" in names:
        pattern = "cycle"
    elif {"fan_out_burst", "fan_in_burst"} <= names:
        pattern = "gather_scatter"
    elif "fan_out_burst" in names:
        pattern = "fan_out"
    elif "fan_in_burst" in names:
        pattern = "fan_in"
    elif "rapid_pass_through" in names:
        pattern = "layering_chain"
    else:
        pattern = "none"
    suspicious = alert["score"] >= threshold
    claims = [{"statement": s["detail"], "txn_ids": s.get("txn_ids", []), "accounts": [alert["account"]],
               "amount": None, "count": None} for s in alert["signals"] if s.get("txn_ids")]
    return {
        "verdict": "suspicious" if suspicious else "not_suspicious",
        "risk_score": min(100, int(alert["score"] * 20)),
        "pattern_id": pattern if suspicious else "none",
        "summary": f"Rule score {alert['score']} from signals: {', '.join(sorted(names))}.",
        "claims": claims,
        "recommended_action": "Escalate" if suspicious else "Close",
    }
