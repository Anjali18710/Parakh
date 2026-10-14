"""Code-based fact checking of case reports.

The verifier is deliberately *not* an LLM. Every claim in a report must cite
transaction IDs, and we check the claim against those transactions in Python:
do they exist, do they involve the accounts named, do the amounts and counts
match. One LLM grading another can be fooled; arithmetic cannot.
"""
from __future__ import annotations

import re

from parakh.data.store import TransactionStore
from parakh.fmt import inr


def inr_plain(amount: float) -> str:
    return inr(amount).lstrip("₹")

_NUM = re.compile(
    r"(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(crores?|cr\b|lakhs?|lacs?|l\b|k\b|million|mn\b|m\b)?", re.I)
_UNITS = {"crore": 1e7, "crores": 1e7, "cr": 1e7, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5,
          "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6}
# "above ₹5 lakh", "≥ ₹5,00,000", "under ₹10 lakh" describe a threshold, not a fact about the money.
_THRESHOLD = re.compile(r"(≥|>=|>|≤|<=|<|\babove|\bover|\bbelow|\bunder|\bat least|\bat most|\bmore than|"
                        r"\bless than|\bexceeding|\bthreshold of|\bminimum of|\bup to)\s*$", re.I)


def _amounts_with_tolerance(text: str) -> list[tuple[float, float]]:
    """Rupee amounts in a sentence, each with how far it may be from the truth given how it is written.

    "₹66,41,114" is an exact figure (allowed off by ₹1); "₹66.4 lakh" is rounded to 0.1 lakh
    (allowed off by ₹5,000). Counts, hours, percentages, ids, dates and thresholds are ignored.
    """
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
        if _THRESHOLD.search(text[max(0, m.start() - 20):m.start()]):
            continue
        has_currency = bool(re.match(r"(₹|rs|inr)", m.group(0).strip(), re.I))
        if not has_currency and not unit and value.is_integer() and 1900 <= value <= 2100:
            continue  # a year
        if value >= 1000:
            decimals = len(raw.split(".")[1]) if "." in raw else 0
            unit_size = _UNITS.get(unit, 1)
            tol = max(1.0, 0.5 * unit_size / (10 ** decimals)) if unit else 1.0
            found.append((value, tol))
    return found


_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                 "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
_NOUNS = {
    "senders": r"senders?|sources?|payers?|depositors?|remitters?",
    "recipients": r"recipients?|beneficiar(?:y|ies)|payees?|destinations?",
    "accounts": r"accounts?|counterpart(?:y|ies)|parties|party",
    "transactions": r"payments?|transactions?|transfers?|credits?|debits?|inflows?|outflows?|deposits?|remittances?",
}
_STOP = r"(?:" + "|".join(_NUMBER_WORDS) + r"|in|to|from|of|on|at|by|for|with|within|over|and|or|than|per)\b"
_COUNT = re.compile(
    r"(?<![\d.,₹])\b(\d{1,3}|" + "|".join(_NUMBER_WORDS[1:]) + r")\s+((?:(?!" + _STOP + r")[A-Za-z/-]+\s+){0,3}?)("
    + "|".join(f"(?P<{k}>{v})" for k, v in _NOUNS.items()) + r")\b", re.I)


def count_mentions(statement: str, txns, case_account: str | None) -> list[dict]:
    """Counts stated in a sentence ("eight distinct senders", "6 payments") that the cited rows contradict.

    Returns one entry per wrong count with the text to replace and the correct number.
    """
    if txns is None or len(txns) == 0:
        return []
    others = lambda col: {a for a in txns[col] if a != case_account}  # noqa: E731
    truth = {"senders": len(others("from_account")), "recipients": len(others("to_account")),
             "transactions": len(txns)}
    truth["accounts"] = len(others("from_account") | others("to_account"))
    wrong = []
    for m in _COUNT.finditer(statement):
        raw = m.group(1).lower()
        n = int(raw) if raw.isdigit() else _NUMBER_WORDS.index(raw)
        kind = next(k for k in _NOUNS if m.group(k))
        allowed = {truth[kind]} if kind != "accounts" else {truth["accounts"], truth["senders"], truth["recipients"]}
        if n not in allowed:
            expected = truth[kind]
            fixed = (_NUMBER_WORDS[expected] if not raw.isdigit() and expected < len(_NUMBER_WORDS) else str(expected))
            if raw[0].isupper() if not raw.isdigit() else False:
                fixed = fixed.capitalize()
            wrong.append({"text": m.group(1), "start": m.start(1), "end": m.end(1), "stated": n,
                          "expected": expected, "replacement": fixed, "noun": m.group(3)})
    return wrong


def _candidates(amounts: list[float]) -> list[float]:
    if not amounts:
        return []
    total = sum(amounts)
    return [total, max(amounts), min(amounts), total / len(amounts)] + amounts


def _matches_amount(value: float, amounts: list[float], tol: float = 1.0, extra: list[float] = ()) -> bool:
    """True if `value` equals a cited amount, their total, max, min or average (or an `extra` subtotal)."""
    return any(abs(value - c) <= tol for c in _candidates(amounts) + list(extra))


def verify_report(report: dict, store: TransactionStore, case_account: str,
                  tools_used: set[str] | None = None, trusted_amounts: list[float] | None = None,
                  pattern_kind: str | None = None) -> dict:
    """Check every cited claim against the ledger, and every 'nothing found' check against the tools actually run."""
    claim_results, issues = [], []
    cited_accounts: set[str] = set()
    supported_amounts: list[float] = [float(a) for a in (trusted_amounts or []) if a]

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
        # A claim about money in and out may quote either side: what reached the case account, or what left it.
        sides = [float(txns[txns.to_account == case_account].amount.sum()),
                 float(txns[txns.from_account == case_account].amount.sum())]
        sides = [x for x in sides if x]
        supported_amounts.extend(_candidates(amounts) + sides)
        if claim.get("amount") is not None and not _matches_amount(float(claim["amount"]), amounts, 1.0, sides):
            problems.append(f"amount ₹{inr_plain(float(claim['amount']))} does not match the cited transactions "
                            f"(they total ₹{inr_plain(sum(amounts))})")
        for value, tol in _amounts_with_tolerance(claim.get("statement", "")):
            if not _matches_amount(value, amounts, tol, sides):
                problems.append(f"amount ₹{inr_plain(value)} in the statement does not match the cited transactions "
                                f"(they total ₹{inr_plain(sum(amounts))})")
                break
        for w in count_mentions(claim.get("statement", ""), txns, case_account):
            problems.append(f"count '{w['text']} {w['noun']}' does not match the cited transactions "
                            f"({w['expected']} {w['noun']})")
        if claim.get("count") is not None:
            n = int(claim["count"])
            allowed = {len(ids), len(involved), max(len(involved) - 1, 0),
                       txns.to_account.nunique(), txns.from_account.nunique()}
            if n not in allowed:
                problems.append(f"count {n} does not match the cited transactions")
        unfilled = re.findall(r"\{\w+\}", claim.get("statement", ""))
        if unfilled:
            problems.append(f"placeholder {unfilled[0]} could not be filled in from the cited transactions")
        claim_results.append({"index": i, "ok": not problems, "problems": problems})

    if report.get("claims") and case_account not in cited_accounts:
        issues.append(f"no claim cites a transaction involving the case account {case_account}")
    verified = sum(r["ok"] for r in claim_results)
    if report.get("verdict") == "suspicious" and verified < 2:
        issues.append("a 'suspicious' verdict needs at least two verified claims")
    for r in claim_results:
        if not r["ok"]:
            issues.append(f"claim {r['index'] + 1}: " + "; ".join(r["problems"]))

    if pattern_kind == "innocent" and report.get("verdict") == "suspicious":
        issues.append("verdict 'suspicious' contradicts the pattern finding, which is an innocent explanation")

    # The summary may only quote amounts that some cited evidence supports.
    for value, tol in _amounts_with_tolerance(report.get("summary", "")):
        if not any(abs(value - c) <= tol for c in supported_amounts):
            issues.append(f"summary: amount ₹{inr_plain(value)} is not supported by any cited transactions")

    # Negative findings ("no loops found") cannot cite transactions. They are valid only if the
    # investigator really ran the tool that would have found something.
    check_results = []
    for i, check in enumerate(report.get("checks", [])):
        ok = tools_used is None or check.get("tool") in tools_used
        check_results.append({"index": i, "ok": ok,
                              "problems": [] if ok else [f"tool '{check.get('tool')}' was never run in this case"]})
        if not ok:
            issues.append(f"check {i + 1}: tool '{check.get('tool')}' was never run in this case")

    return {
        "passed": not issues,
        "checks_total": len(check_results),
        "checks_verified": sum(r["ok"] for r in check_results),
        "check_results": check_results,
        "claims_total": len(claim_results),
        "claims_verified": verified,
        "claim_results": claim_results,
        "issues": issues,
    }
