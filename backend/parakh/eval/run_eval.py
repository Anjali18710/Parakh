"""Experiments: how much do the agents add over rules, and what does each part contribute?

Modes compared on the *same* random sample of rule alerts:
  rules           – screening rules only (score >= 2 means suspicious)
  single          – one agent with every tool, no separate pattern agent, verifier not enforced
  multi_noverify  – full agent team, verifier measured but not enforced
  multi           – full agent team with the fact-checking loop

Example:
    python -m parakh.eval.run_eval --sample 40 --modes rules,single,multi_noverify,multi --sleep 2
    python -m parakh.eval.run_eval --triage --budget 20

Results go to results/ as JSON (every case, for failure analysis) and Markdown (summary table).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from parakh.agents.graph import Deps, build_case_graph
from parakh.agents.llm import UsageTracker, get_llm
from parakh.agents.triage import rules_report, triage_by_score, triage_with_llm
from parakh.config import settings
from parakh.data.store import TransactionStore
from parakh.patterns.library import PatternLibrary
from parakh.tools.investigation import InvestigationTools
from parakh.agents.service import load_or_screen

ALL_MODES = ["rules", "single", "multi_noverify", "multi"]


def load_roles(data_path: str) -> dict[str, str]:
    path = Path(data_path.replace("transactions", "account_roles").replace(".csv.gz", ".csv"))
    if path.exists() and path != Path(data_path):
        return pd.read_csv(path, index_col=0).role.to_dict()
    return {}


def run_case(mode: str, alert: dict, deps: Deps | None, graphs: dict) -> dict:
    if mode == "rules":
        report = rules_report(alert)
        # Rule alerts are not written claims, so fact-checking them is not meaningful.
        return {"report": report, "verification": None,
                "usage": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "est_cost_usd": 0.0},
                "seconds": 0.0, "tool_calls": 0}
    tracker = UsageTracker()
    start = time.time()
    graph = graphs[mode]
    state = graph.invoke(
        {"case_id": f"eval-{mode}-{alert['account']}", "account": alert["account"], "alert": alert, "mode": mode,
         "analyst_notes": [], "evidence": [], "timeline": [], "tool_calls": 0},
        {"callbacks": [tracker], "recursion_limit": 60},
    )
    return {"report": state.get("report"), "verification": state.get("verification"),
            "pattern": state.get("pattern"), "usage": tracker.summary(settings),
            "seconds": round(time.time() - start, 2), "tool_calls": state.get("tool_calls", 0),
            "report_attempts": state.get("report_attempts", 0)}


def score(rows: list[dict], store: TransactionStore, roles: dict[str, str]) -> dict:
    truth = store.laundering_accounts
    patterns = store.instance_patterns
    tp = fp = fn = tn = 0
    pattern_hits = pattern_total = 0
    claims_ok = claims_total = 0
    false_alarm_roles: Counter = Counter()
    errors = 0
    for r in rows:
        if r.get("error"):
            errors += 1
            continue
        acc = r["account"]
        is_bad = acc in truth
        said_bad = (r["report"] or {}).get("verdict") == "suspicious"
        if is_bad and said_bad:
            tp += 1
            true_pats = {patterns.get(i) for i in truth[acc]}
            pattern_total += 1
            pattern_hits += (r["report"].get("pattern_id") in true_pats)
        elif said_bad:
            fp += 1
            false_alarm_roles[roles.get(acc, "unknown")] += 1
        elif is_bad:
            fn += 1
        else:
            tn += 1
        v = r.get("verification") or {}
        claims_ok += v.get("claims_verified", 0)
        claims_total += v.get("claims_total", 0)
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    usage = [r["usage"] for r in rows if r.get("usage")]
    secs = [r["seconds"] for r in rows if "seconds" in r]
    return {
        "cases": n, "errors": errors, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "accuracy": round((tp + tn) / n, 3) if n else 0.0,
        "pattern_accuracy": round(pattern_hits / pattern_total, 3) if pattern_total else None,
        "claim_support_rate": round(claims_ok / claims_total, 3) if claims_total else None,
        "unsupported_claims": claims_total - claims_ok,
        "false_alarms_by_role": dict(false_alarm_roles),
        "avg_tokens": round(sum(u["input_tokens"] + u["output_tokens"] for u in usage) / len(usage)) if usage else 0,
        "avg_cost_usd": round(sum(u["est_cost_usd"] for u in usage) / len(usage), 5) if usage else 0.0,
        "avg_seconds": round(sum(secs) / len(secs), 1) if secs else 0.0,
    }


def markdown(results: dict, meta: dict) -> str:
    lines = [f"# Parakh evaluation ({meta['timestamp']})", "",
             f"Data: `{meta['data']}`. Sample: {meta['sample']} rule alerts "
             f"({meta['laundering_in_sample']} truly laundering). Model: {meta['model']}.", "",
             "| Mode | Precision | Recall | F1 | Pattern accuracy | Claims supported | Avg tokens | Avg cost (USD) | Avg time (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for mode, m in results.items():
        pa = "–" if m["pattern_accuracy"] is None else f"{m['pattern_accuracy']:.0%}"
        cs = "–" if m["claim_support_rate"] is None else f"{m['claim_support_rate']:.0%}"
        lines.append(f"| {mode} | {m['precision']:.0%} | {m['recall']:.0%} | {m['f1']:.2f} | {pa} | {cs} | "
                     f"{m['avg_tokens']:,} | {m['avg_cost_usd']:.4f} | {m['avg_seconds']} |")
    lines += ["", "False alarms by account type (who got wrongly flagged):", ""]
    for mode, m in results.items():
        lines.append(f"- **{mode}**: {m['false_alarms_by_role'] or 'none'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=settings.data_path)
    p.add_argument("--modes", default=",".join(ALL_MODES))
    p.add_argument("--sample", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep", type=float, default=0.0, help="pause between cases to respect API rate limits")
    p.add_argument("--triage", action="store_true", help="evaluate the triage agent instead")
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    store = TransactionStore.from_csv(a.data)
    alerts = load_or_screen(store, a.data)
    roles = load_roles(a.data)
    truth = store.laundering_accounts
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if a.triage:
        briefs = [{"account": x["account"], "score": x["score"],
                   "signals": [f"{s['name']}: {s['detail']}" for s in x["signals"]]} for x in alerts]
        pool = briefs[: max(60, a.budget * 4)]
        res = {}
        for name, fn in [("rules_top_score", lambda: triage_by_score(pool, a.budget)),
                         ("triage_agent", lambda: triage_with_llm(get_llm(), pool, a.budget))]:
            chosen = [d["account"] for d in fn() if d["selected"]]
            hits = [c for c in chosen if c in truth]
            inst = set().union(*(truth[c] for c in hits)) if hits else set()
            res[name] = {"opened": len(chosen), "laundering": len(hits),
                         "precision": round(len(hits) / len(chosen), 3) if chosen else 0, "instances_found": len(inst)}
        print(json.dumps(res, indent=2))
        (out / f"triage_{stamp}.json").write_text(json.dumps(res, indent=2))
        return

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    rng = random.Random(a.seed)
    sample = rng.sample(alerts, min(a.sample, len(alerts)))
    needs_llm = any(m != "rules" for m in modes)
    llm = get_llm() if needs_llm else None
    deps = Deps(llm, store, InvestigationTools(store), PatternLibrary(), settings)
    graphs = {}
    if needs_llm:
        graphs = {
            "single": build_case_graph(deps, "single", human_review=False, enforce_verifier=False),
            "multi_noverify": build_case_graph(deps, "multi", human_review=False, enforce_verifier=False),
            "multi": build_case_graph(deps, "multi", human_review=False, enforce_verifier=True),
        }

    results, all_rows = {}, {}
    for mode in modes:
        rows = []
        for i, alert in enumerate(sample, 1):
            try:
                r = run_case(mode, alert, deps, graphs)
            except Exception as exc:  # keep going; failures are counted and reported
                r = {"error": f"{type(exc).__name__}: {exc}"}
            r.update({"account": alert["account"], "truly_laundering": alert["account"] in truth,
                      "role": roles.get(alert["account"]), "alert_score": alert["score"]})
            rows.append(r)
            verdict = (r.get("report") or {}).get("verdict", r.get("error", "?"))
            print(f"[{mode}] {i}/{len(sample)} {alert['account']} -> {verdict}", flush=True)
            if mode != "rules" and a.sleep:
                time.sleep(a.sleep)
        results[mode] = score(rows, store, roles)
        all_rows[mode] = rows

    meta = {"timestamp": stamp, "data": a.data, "sample": len(sample), "seed": a.seed,
            "laundering_in_sample": sum(x["account"] in truth for x in sample),
            "model": settings.gemini_model if settings.llm_provider == "gemini" else settings.groq_model}
    (out / f"eval_{stamp}.json").write_text(json.dumps({"meta": meta, "results": results, "cases": all_rows},
                                                       indent=2, default=str))
    md = markdown(results, meta)
    (out / f"eval_{stamp}.md").write_text(md)
    (out / "latest.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
