"""Rules alone vs Parakh, scored against IBM's answer key.

Takes a fixed sample of screening alerts (half on truly laundering accounts, half on innocent ones)
and asks two questions of each:

  rules    the screening rules alone: a rule score of 2 or more means suspicious
  parakh   the full agent team with the fact-checker, no analyst in the loop

The answer key is IBM's is_laundering label, which neither side ever sees.

Each finished case is saved straight away. On Groq's free tier the daily token limit runs out
after about a dozen cases; the script then stops cleanly and the same command carries on the
next day. The summary is rewritten after every run:

    python -m parakh.eval.compare            # run (or continue) the comparison
    python -m parakh.eval.compare --summary  # only print the summary of what has finished
    python -m parakh.eval.compare --dev      # try out a change on 10 other alerts, never the 30 above
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from parakh.agents.graph import Deps, build_case_graph
from parakh.agents.llm import UsageTracker, friendly_error, get_llm
from parakh.agents.service import load_or_screen
from parakh.config import BACKEND_DIR, settings
from parakh.data.store import TransactionStore
from parakh.patterns.library import PatternLibrary
from parakh.tools.investigation import InvestigationTools

RESULTS_DIR = BACKEND_DIR / "results"
RULE_THRESHOLD = 2.0


class DailyLimit(Exception):
    """Groq's daily token limit is used up; progress is saved and the run can continue tomorrow."""


def rules_verdict(alert: dict, threshold: float = RULE_THRESHOLD) -> str:
    return "suspicious" if alert["score"] >= threshold else "not_suspicious"


def true_patterns(store: TransactionStore) -> dict[str, list[str]]:
    """account -> IBM's laundering types it took part in (empty for innocent accounts)."""
    kind = store.labels.dropna(subset=["instance_id"]).set_index("instance_id")["pattern"].to_dict()
    return {acc: sorted({kind.get(i, "") for i in ids}) for acc, ids in store.laundering_accounts.items()}


def pick_sample(alerts: list[dict], laundering: set[str], n: int, seed: int,
                exclude: set[str] | None = None) -> list[str]:
    """Half alerts on laundering accounts, half on innocent ones, in a fixed random order."""
    rng = random.Random(seed)
    pool = [a["account"] for a in alerts if a["account"] not in (exclude or set())]
    bad = sorted(acc for acc in pool if acc in laundering)
    good = sorted(acc for acc in pool if acc not in laundering)
    k = min(n // 2, len(bad), len(good))
    sample = rng.sample(bad, k) + rng.sample(good, k)
    rng.shuffle(sample)
    return sample


def investigate(graph, alert: dict) -> dict:
    tracker = UsageTracker()
    started = time.time()
    state = graph.invoke(
        {"case_id": f"compare-{alert['account']}", "account": alert["account"], "alert": alert, "mode": "multi",
         "analyst_notes": [], "evidence": [], "timeline": [], "tool_calls": 0},
        {"callbacks": [tracker], "recursion_limit": 60})
    report = state.get("report") or {}
    check = state.get("verification") or {}
    pattern = state.get("pattern") or {}
    return {
        "verdict": report.get("verdict"), "risk_score": report.get("risk_score"),
        "pattern_id": report.get("pattern_id"),
        "claims": check.get("claims_total", 0), "claims_verified": check.get("claims_verified", 0),
        "rewrites": max(0, state.get("report_attempts", 1) - 1), "tool_calls": state.get("tool_calls", 0),
        "tokens": tracker.input_tokens + tracker.output_tokens, "seconds": round(time.time() - started, 1),
        # What the agents wrote, so a wrong verdict can be traced to the step that went wrong.
        "summary": report.get("summary"), "claim_texts": [c.get("statement") for c in report.get("claims", [])],
        "pattern_candidates": pattern.get("candidates"), "pattern_reasoning": pattern.get("reasoning"),
        "investigation": (state.get("investigation_summary") or "")[:2000],
    }


def run(store: TransactionStore, alerts: list[dict], llm, out: Path, n: int = 30, seed: int = 7,
        max_cases: int | None = None, log=print, exclude: set[str] | None = None) -> list[dict]:
    """Run every unfinished case of the sample. Raises DailyLimit when Groq's daily limit is reached."""
    out.mkdir(parents=True, exist_ok=True)
    by_account = {a["account"]: a for a in alerts}
    truth = true_patterns(store)
    sample_path, rows_path = out / "sample.json", out / "cases.jsonl"
    if sample_path.exists():
        sample = json.loads(sample_path.read_text())["accounts"]
    else:
        sample = pick_sample(alerts, set(truth), n, seed, exclude)
        sample_path.write_text(json.dumps({"seed": seed, "accounts": sample}, indent=1))

    rows = load_rows(rows_path)
    todo = [acc for acc in sample if acc not in rows]
    if max_cases is not None:
        todo = todo[:max_cases]
    log(f"{len(rows)} of {len(sample)} cases done, {len(todo)} to run now")
    graph = build_case_graph(Deps(llm, store, InvestigationTools(store), PatternLibrary(), settings),
                             mode="multi", human_review=False, enforce_verifier=True) if todo else None
    with rows_path.open("a", encoding="utf-8") as fh:
        for i, acc in enumerate(todo, 1):
            alert = by_account[acc]
            row = {"account": acc, "truly_laundering": acc in truth, "true_patterns": truth.get(acc, []),
                   "rule_score": alert["score"], "rules": rules_verdict(alert)}
            try:
                row["parakh"] = investigate(graph, alert)
            except Exception as exc:  # recorded, and run again next time
                kind, message = friendly_error(exc)
                if kind == "rate_limit":
                    raise DailyLimit(message) from exc
                row["error"] = message[:300]
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows[acc] = row
            got = row.get("parakh", {}).get("verdict") or "error"
            log(f"[{i}/{len(todo)}] {acc}  truth: {'laundering' if row['truly_laundering'] else 'innocent':<10}"
                f" rules: {row['rules']:<15} parakh: {got}")
    return [rows[a] for a in sample if a in rows]


def load_rows(path: Path) -> dict[str, dict]:
    """Latest finished result per account. Cases that ended in an error are left out, so they run again."""
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if "parakh" in row:
                    rows[row["account"]] = row
    return rows


def score(rows: list[dict], verdict) -> dict:
    tp = sum(r["truly_laundering"] and verdict(r) == "suspicious" for r in rows)
    fp = sum(not r["truly_laundering"] and verdict(r) == "suspicious" for r in rows)
    bad = sum(r["truly_laundering"] for r in rows)
    good = len(rows) - bad
    return {"caught": tp, "laundering": bad, "cleared": good - fp, "innocent": good, "false_alarms": fp,
            "right": tp + good - fp, "cases": len(rows),
            "precision": round(tp / (tp + fp), 2) if tp + fp else None}


def summarise(rows: list[dict], total: int) -> dict:
    rules = score(rows, lambda r: r["rules"])
    parakh = score(rows, lambda r: r["parakh"]["verdict"])
    caught = [r for r in rows if r["truly_laundering"] and r["parakh"]["verdict"] == "suspicious"
              and r["true_patterns"] and r["true_patterns"] != ["other_scheme"]]
    named = sum(r["parakh"]["pattern_id"] in r["true_patterns"] for r in caught)
    p = [r["parakh"] for r in rows]
    return {
        "cases_done": len(rows), "cases_total": total, "rules": rules, "parakh": parakh,
        "pattern_named": named, "pattern_checked": len(caught),
        "claims": sum(x["claims"] for x in p), "claims_verified": sum(x["claims_verified"] for x in p),
        "rewrites": sum(x["rewrites"] > 0 for x in p),
        "avg_tokens": round(sum(x["tokens"] for x in p) / len(p)) if p else 0,
        "avg_seconds": round(sum(x["seconds"] for x in p) / len(p), 1) if p else 0,
    }


def markdown(s: dict, model: str) -> str:
    def line(name: str, m: dict) -> str:
        precision = "-" if m["precision"] is None else f"{m['precision']:.0%}"
        return (f"| {name} | {m['caught']} of {m['laundering']} | {m['cleared']} of {m['innocent']} | "
                f"{m['false_alarms']} | {precision} | {m['right']} of {m['cases']} |")

    done = "" if s["cases_done"] == s["cases_total"] else f" ({s['cases_done']} of {s['cases_total']} done so far)"
    return "\n".join([
        f"# Rules alone vs Parakh on IBM's AML data{done}", "",
        (f"{s['cases_total']} screening alerts from the IBM slice, half on accounts IBM labels as laundering and "
         f"half on innocent ones. Model: `{model}`. Answer key: IBM's `is_laundering` labels."), "",
        "| | Laundering caught | Innocent cleared | False alarms | Precision | Right overall |",
        "|---|---|---|---|---|---|",
        line("Rules alone (score >= 2)", s["rules"]),
        line("Parakh", s["parakh"]), "",
        (f"- Pattern named correctly: {s['pattern_named']} of {s['pattern_checked']} caught laundering accounts "
         f"(accounts whose IBM type is in Parakh's library)."),
        (f"- Claims that passed the fact-checker: {s['claims_verified']} of {s['claims']}; "
         f"{s['rewrites']} report(s) were sent back for a rewrite."),
        f"- Average per case: {s['avg_tokens']:,} tokens, {s['avg_seconds']} s.", "",
    ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, default=30, help="alerts in the sample (half laundering, half innocent)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max", type=int, help="run at most this many cases now (leaves quota for the website)")
    p.add_argument("--summary", action="store_true", help="only print the summary of finished cases")
    p.add_argument("--dev", action="store_true",
                   help="use a separate set of 10 alerts for trying out changes (never the 30 used for the result)")
    p.add_argument("--out", default=str(RESULTS_DIR))
    a = p.parse_args()

    out = Path(a.out)
    store = TransactionStore.from_csv(settings.data_path)
    alerts = load_or_screen(store, settings.data_path)
    exclude = None
    if a.dev:
        # Changes are tried out on these alerts only; the 30 test alerts stay untouched until the final run.
        out, a.sample, a.seed = out / "dev", 10, 99
        exclude = set(pick_sample(alerts, set(true_patterns(store)), 30, 7))
        if (Path(a.out) / "sample.json").exists():
            exclude |= set(json.loads((Path(a.out) / "sample.json").read_text())["accounts"])
    if a.summary:
        sample = json.loads((out / "sample.json").read_text())["accounts"]
        rows = [r for acc, r in load_rows(out / "cases.jsonl").items() if acc in sample]
    else:
        try:
            run(store, alerts, get_llm(), out, a.sample, a.seed, a.max, exclude=exclude)
        except DailyLimit as stop:
            print(f"\nStopped: {stop}\nGroq's daily limit refills a little at a time over 24 hours, so a full case "
                  "(about 15,000 tokens) usually has to wait until about this time tomorrow.\n"
                  "Everything finished so far is saved. Run the same command then to continue.")
        except KeyboardInterrupt:
            print("\nStopped. Everything finished so far is saved; run the same command to continue.")
        sample = json.loads((out / "sample.json").read_text())["accounts"]
        rows = [r for acc, r in load_rows(out / "cases.jsonl").items() if acc in sample]
    if not rows:
        print("No cases finished yet.")
        return
    model = settings.gemini_model if settings.llm_provider == "gemini" else settings.groq_model
    text = markdown(summarise(rows, len(sample)), model)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
