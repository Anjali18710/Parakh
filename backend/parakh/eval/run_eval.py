"""Evaluation: how much do the agents add over rules, and what does each guardrail contribute?

The same sample of rule alerts goes through four setups:

  rules          screening rules alone (rule score >= 2 means suspicious)
  single         one agent with every tool; no separate pattern agent; no fact-check loop
  no_guardrails  the agent team, but no shape rules and no fact-check loop (agents decide everything)
  full           the agent team with shape rules and the fact-check / correction loop (what the app runs)

Every case result is saved as soon as it finishes, so a run that hits the LLM provider's daily
limit can be resumed with the same command the next day. When all cases are done (or on demand
with --summarise) the summary is written to evaluation/<name>.json, which the dashboard shows.

Examples:
    python -m parakh.eval.run_eval --name demo --sample 30
    python -m parakh.eval.run_eval --name demo --summarise
    DATA_PATH=data/ibm_sample.csv.gz python -m parakh.eval.run_eval --name ibm --sample 30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from parakh.agents.graph import Deps, build_case_graph
from parakh.agents.llm import UsageTracker, get_llm
from parakh.agents.service import load_or_screen
from parakh.agents.triage import rules_report
from parakh.config import BACKEND_DIR, settings
from parakh.data.store import TransactionStore
from parakh.patterns.library import PatternLibrary
from parakh.tools.investigation import InvestigationTools

ALL_MODES = ["rules", "single", "no_guardrails", "full"]
MODE_INFO = {
    "rules": "Screening rules alone",
    "single": "One agent with every tool",
    "no_guardrails": "Agent team, agents decide everything",
    "full": "Agent team with shape rules and fact-check loop",
}
RUNS_DIR = BACKEND_DIR / "results"
PUBLISHED_DIR = BACKEND_DIR / "evaluation"


class DailyLimitReached(Exception):
    pass


def _is_daily_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("per day" in text or "tpd" in text or "rpd" in text) and ("rate" in text or "limit" in text)


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


def load_roles(data_path: str) -> dict[str, str]:
    path = Path(str(data_path).replace("transactions", "account_roles").replace(".csv.gz", ".csv"))
    if path.exists() and path != Path(data_path):
        return pd.read_csv(path, index_col=0).role.to_dict()
    return {}


def screening_coverage(alerts: dict, truth: dict, roles: dict) -> dict:
    """How many laundering accounts the rules alert on at all. External source accounts (where criminal money
    enters from outside the bank) are not customers the bank could alert on, so they are left out."""
    customers = [acc for acc in truth if roles.get(acc) != "external source"]
    return {"alerts": len(alerts), "laundering_accounts": len(customers),
            "laundering_accounts_alerted": sum(acc in alerts for acc in customers)}


def pick_sample(alerts: list[dict], truth: dict, n: int, seed: int) -> list[str]:
    """Half truly-laundering alerts, half not, so precision and recall are both meaningful at small n."""
    rng = random.Random(seed)
    bad = [a["account"] for a in alerts if a["account"] in truth]
    good = [a["account"] for a in alerts if a["account"] not in truth]
    k_bad = min(len(bad), n // 2)
    k_good = min(len(good), n - k_bad)
    sample = rng.sample(bad, k_bad) + rng.sample(good, k_good)
    rng.shuffle(sample)
    return sample


def run_one(mode: str, alert: dict, deps: Deps | None, graphs: dict) -> dict:
    if mode == "rules":
        report = rules_report(alert)
        return {"verdict": report["verdict"], "pattern_id": report["pattern_id"], "risk_score": report["risk_score"],
                "tokens": 0, "cost_usd": 0.0, "seconds": 0.0}
    tracker = UsageTracker()
    start = time.time()
    state = graphs[mode].invoke(
        {"case_id": f"eval-{mode}-{alert['account']}", "account": alert["account"], "alert": alert, "mode": mode,
         "analyst_notes": [], "evidence": [], "timeline": [], "tool_calls": 0},
        {"callbacks": [tracker], "recursion_limit": 60})
    report = state.get("report") or {}
    v = state.get("verification") or {}
    usage = tracker.summary(settings)
    return {
        "verdict": report.get("verdict"), "pattern_id": report.get("pattern_id"), "risk_score": report.get("risk_score"),
        "pattern_source": (state.get("pattern") or {}).get("source"),
        "claims_total": v.get("claims_total", 0), "claims_verified": v.get("claims_verified", 0),
        "claims_corrected": len(v.get("corrected_claims", [])), "claims_removed": len(v.get("removed_claims", [])),
        "report_attempts": state.get("report_attempts", 0), "tool_calls": state.get("tool_calls", 0),
        "tokens": usage["input_tokens"] + usage["output_tokens"], "cost_usd": usage["est_cost_usd"],
        "seconds": round(time.time() - start, 1),
    }


def summarise(rows: list[dict], meta: dict) -> dict:
    by_mode: dict[str, list[dict]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    results = {}
    for mode in [m for m in ALL_MODES if m in by_mode]:
        rs = [r for r in by_mode[mode] if not r.get("error")]
        tp = sum(r["truly_laundering"] and r["verdict"] == "suspicious" for r in rs)
        fp = sum(not r["truly_laundering"] and r["verdict"] == "suspicious" for r in rs)
        fn = sum(r["truly_laundering"] and r["verdict"] != "suspicious" for r in rs)
        tn = sum(not r["truly_laundering"] and r["verdict"] != "suspicious" for r in rs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        caught = [r for r in rs if r["truly_laundering"] and r["verdict"] == "suspicious"]
        pattern_hits = sum(r["pattern_id"] in r["true_patterns"] for r in caught)
        claims_total = sum(r.get("claims_total", 0) for r in rs)
        llm_rows = [r for r in rs if r.get("tokens")]
        results[mode] = {
            "label": MODE_INFO[mode], "cases": len(rs), "errors": len(by_mode[mode]) - len(rs),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
            "innocent_cleared": round(tn / (tn + fp), 3) if tn + fp else None,
            "pattern_accuracy": round(pattern_hits / len(caught), 3) if caught else None,
            "claims_total": claims_total,
            "claims_verified": sum(r.get("claims_verified", 0) for r in rs),
            "claims_corrected": sum(r.get("claims_corrected", 0) for r in rs),
            "claims_removed": sum(r.get("claims_removed", 0) for r in rs),
            "avg_tokens": round(sum(r["tokens"] for r in llm_rows) / len(llm_rows)) if llm_rows else 0,
            "avg_cost_usd": round(sum(r["cost_usd"] for r in llm_rows) / len(llm_rows), 5) if llm_rows else 0.0,
            "avg_seconds": round(sum(r["seconds"] for r in llm_rows) / len(llm_rows), 1) if llm_rows else 0.0,
            "false_alarms_by_role": dict(Counter(r["role"] or "unknown" for r in rs
                                                 if not r["truly_laundering"] and r["verdict"] == "suspicious")),
            "missed_by_pattern": dict(Counter(p for r in rs if r["truly_laundering"] and r["verdict"] != "suspicious"
                                              for p in r["true_patterns"])),
        }
    cases = [{k: r.get(k) for k in ("mode", "account", "truly_laundering", "true_patterns", "role", "verdict",
                                     "pattern_id", "risk_score", "claims_total", "claims_verified", "error")}
             for r in rows]
    return {**meta, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "complete": all(len(by_mode.get(m, [])) == meta["sample_size"] for m in meta["modes"]),
            "results": results, "cases": cases}


def markdown(summary: dict) -> str:
    lines = [f"# Parakh evaluation: {summary['name']}", "",
             f"{summary['sample_size']} rule alerts ({summary['laundering_in_sample']} truly laundering), "
             f"model `{summary['model']}`, data `{summary['data']}`.", "",
             f"Screening coverage: the rules alert on {summary['screening']['laundering_accounts_alerted']} of "
             f"{summary['screening']['laundering_accounts']} laundering accounts.", "",
             "| Setup | Precision | Recall | F1 | Innocent cleared | Pattern accuracy | Claims verified | Avg tokens | Avg cost (USD) |",
             "|---|---|---|---|---|---|---|---|---|"]
    pct = lambda v: "–" if v is None else f"{v:.0%}"  # noqa: E731
    for mode, m in summary["results"].items():
        claims = f"{m['claims_verified']}/{m['claims_total']}" if m["claims_total"] else "–"
        lines.append(f"| {m['label']} | {pct(m['precision'])} | {pct(m['recall'])} | {m['f1']:.2f} | "
                     f"{pct(m['innocent_cleared'])} | {pct(m['pattern_accuracy'])} | {claims} | {m['avg_tokens']:,} | "
                     f"{m['avg_cost_usd']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="demo", help="name of this evaluation run (results are resumed by name)")
    p.add_argument("--data", default=settings.data_path)
    p.add_argument("--modes", default=",".join(ALL_MODES))
    p.add_argument("--sample", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep", type=float, default=2.0, help="pause between LLM cases (seconds)")
    p.add_argument("--only", help="run only these setups now (comma-separated); the rest can run later")
    p.add_argument("--summarise", action="store_true", help="only rebuild the summary from saved results")
    p.add_argument("--out", default=str(RUNS_DIR))
    p.add_argument("--publish", default=str(PUBLISHED_DIR), help="where the dashboard reads summaries from")
    a = p.parse_args()

    store = TransactionStore.from_csv(a.data)
    alerts = {x["account"]: x for x in load_or_screen(store, a.data)}
    truth, patterns = store.laundering_accounts, store.instance_patterns
    roles = load_roles(a.data)
    run_dir = Path(a.out) / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path, rows_path = run_dir / "meta.json", run_dir / "cases.jsonl"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"Resuming '{a.name}': {meta['sample_size']} alerts, modes {', '.join(meta['modes'])}")
    else:
        sample = pick_sample(list(alerts.values()), truth, a.sample, a.seed)
        meta = {"name": a.name, "data": Path(a.data).name, "seed": a.seed, "sample": sample,
                "sample_size": len(sample), "laundering_in_sample": sum(x in truth for x in sample),
                "modes": [m.strip() for m in a.modes.split(",") if m.strip()],
                "model": settings.gemini_model if settings.llm_provider == "gemini" else settings.groq_model,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "screening": screening_coverage(alerts, truth, roles)}
        meta_path.write_text(json.dumps(meta, indent=2))

    done = []
    if rows_path.exists():
        done = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    finished = {(r["mode"], r["account"]) for r in done if not r.get("error")}

    if not a.summarise:
        run_modes = [m.strip() for m in a.only.split(",")] if a.only else meta["modes"]
        needs_llm = any(m != "rules" for m in run_modes)
        deps = graphs = None
        if needs_llm:
            store_tools = InvestigationTools(store)
            deps = Deps(get_llm(), store, store_tools, PatternLibrary(), settings)
            graphs = {
                "single": build_case_graph(deps, "single", human_review=False, enforce_verifier=False, shape_rules=False),
                "no_guardrails": build_case_graph(deps, "multi", human_review=False, enforce_verifier=False,
                                                  shape_rules=False),
                "full": build_case_graph(deps, "multi", human_review=False, enforce_verifier=True, shape_rules=True),
            }
        todo = [(m, acc) for m in run_modes for acc in meta["sample"] if (m, acc) not in finished]
        print(f"{len(todo)} case(s) to run")
        try:
            with rows_path.open("a") as out:
                for i, (mode, acc) in enumerate(todo, 1):
                    row = {"mode": mode, "account": acc, "truly_laundering": acc in truth,
                           "true_patterns": sorted({patterns.get(x, "") for x in truth.get(acc, set())}),
                           "role": roles.get(acc)}
                    for attempt in (1, 2):
                        try:
                            row.update(run_one(mode, alerts[acc], deps, graphs))
                            row.pop("error", None)
                            break
                        except Exception as exc:  # noqa: BLE001 - every failure is recorded, never hidden
                            if _is_daily_limit(exc):
                                raise DailyLimitReached(str(exc)) from exc
                            if _is_rate_limit(exc) and attempt == 1:
                                print("  rate limited, waiting 60s")
                                time.sleep(60)
                                continue
                            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    done.append(row)
                    print(f"[{i}/{len(todo)}] {mode:<14} {acc} -> {row.get('verdict', row.get('error'))}", flush=True)
                    if mode != "rules" and a.sleep:
                        time.sleep(a.sleep)
        except DailyLimitReached:
            print("\nThe LLM provider's daily limit was reached. Progress is saved; run the same command "
                  "tomorrow to continue.")
        except KeyboardInterrupt:
            print("\nStopped. Progress is saved; run the same command to continue.")

    # Keep only the latest result per (mode, account).
    latest: dict[tuple, dict] = {}
    for r in done:
        latest[(r["mode"], r["account"])] = r
    summary = summarise(list(latest.values()), meta)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "summary.md").write_text(markdown(summary))
    publish = Path(a.publish)
    publish.mkdir(parents=True, exist_ok=True)
    (publish / f"{a.name}.json").write_text(json.dumps(summary, indent=2))
    print("\n" + markdown(summary))
    if not summary["complete"]:
        print("Not all cases are done yet; the summary covers what has finished so far.")
    sys.exit(0)


if __name__ == "__main__":
    main()
