"""Triage agent and the rules-only baseline."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from parakh.agents import prompts
from parakh.agents.schemas import TriageResult

BATCH = 30


def brief(alert: dict) -> dict:
    """Compact view of an alert for the triage agent (no transaction lists)."""
    return {"account": alert["account"], "score": alert["score"],
            "signals": [f"{s['name']}: {s['detail']}" for s in alert["signals"]]}


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
