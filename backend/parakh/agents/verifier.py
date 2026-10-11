"""Code-based fact checking of case reports.

The verifier is deliberately *not* an LLM. Every claim in a report must cite
transaction IDs, and we check the claim against those transactions in Python:
do they exist, do they involve the accounts named, do the amounts and counts
match. One LLM grading another can be fooled; arithmetic cannot.
"""
from __future__ import annotations

import re

from parakh.data.store import TransactionStore

_NUM = re.compile(r"(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(crore|cr|lakh|lakhs|lac|l\b|k\b)?", re.I)
_UNITS = {"crore": 1e7, "cr": 1e7, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "l": 1e5, "k": 1e3}


def _amounts_in_text(text: str) -> list[float]:
    """Rupee amounts mentioned in a sentence (ignores small numbers like counts, hours and percentages)."""
    # Drop things that look like numbers but are not amounts: ids, dates, times.
    text = re.sub(r"\b[A-Za-z]+[0-9]+\b", " ", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    found = []
    for m in _NUM.finditer(text):
        raw, unit = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        try:
            value = float(raw) * _UNITS.get(unit, 1)
        except ValueError:
            continue
        if text[m.end():m.end() + 1] == "%" or text[m.end():m.end() + 2] == " %":
            continue
        has_currency = bool(re.match(r"(₹|rs|inr)", m.group(0).strip(), re.I))
        if not has_currency and not unit and value.is_integer() and 1900 <= value <= 2100:
            continue  # a year
        if value >= 1000:
            found.append(value)
    return found


def _close(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1.0)


def _matches_amount(value: float, amounts: list[float]) -> bool:
    if not amounts:
        return False
    total = sum(amounts)
    candidates = [total, max(amounts), min(amounts), total / len(amounts)] + amounts
    # Allow rounding in text such as "₹48 lakh" for 47.9 lakh.
    return any(_close(value, c, 0.02) for c in candidates)


def verify_report(report: dict, store: TransactionStore, case_account: str) -> dict:
    claim_results, issues = [], []
    cited_accounts: set[str] = set()

    for i, claim in enumerate(report.get("claims", [])):
        problems = []
        ids = list(dict.fromkeys(claim.get("txn_ids") or []))
        if not ids:
            problems.append("cites no transactions")
        txns = store.get_txns(ids)
        missing = [t for t in ids if t not in set(txns.txn_id)]
        if missing:
            problems.append(f"transaction ids do not exist: {', '.join(missing[:5])}")
        involved = set(txns.from_account) | set(txns.to_account)
        cited_accounts |= involved
        for acc in claim.get("accounts") or []:
            if acc not in involved:
                problems.append(f"account {acc} is not part of the cited transactions")
        amounts = [float(a) for a in txns.amount]
        if claim.get("amount") is not None and not _matches_amount(float(claim["amount"]), amounts):
            problems.append(f"amount ₹{float(claim['amount']):,.0f} does not match the cited transactions")
        for value in _amounts_in_text(claim.get("statement", "")):
            if not _matches_amount(value, amounts):
                problems.append(f"amount ₹{value:,.0f} in the statement does not match the cited transactions")
                break
        if claim.get("count") is not None:
            n = int(claim["count"])
            allowed = {len(ids), len(involved), max(len(involved) - 1, 0),
                       txns.to_account.nunique(), txns.from_account.nunique()}
            if n not in allowed:
                problems.append(f"count {n} does not match the cited transactions")
        claim_results.append({"index": i, "ok": not problems, "problems": problems})

    if report.get("claims") and case_account not in cited_accounts:
        issues.append(f"no claim cites a transaction involving the case account {case_account}")
    verified = sum(r["ok"] for r in claim_results)
    if report.get("verdict") == "suspicious" and verified < 2:
        issues.append("a 'suspicious' verdict needs at least two verified claims")
    for r in claim_results:
        if not r["ok"]:
            issues.append(f"claim {r['index'] + 1}: " + "; ".join(r["problems"]))

    return {
        "passed": not issues,
        "claims_total": len(claim_results),
        "claims_verified": verified,
        "claim_results": claim_results,
        "issues": issues,
    }
