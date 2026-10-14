"""Deterministic rules the agents cannot override.

Division of labour:
  * LLM agents decide what to look at, which evidence matters, and how to explain it.
  * Code does arithmetic and enforces which typologies are even possible for the
    measured shape of the money flow.
"""
from __future__ import annotations

import re

from parakh.data.store import TransactionStore
from parakh.fmt import inr
from parakh.tools.flow import case_flow
from parakh.tools.investigation import InvestigationTools

# ---------------------------------------------------------------- measured shape


def shape_facts(store: TransactionStore, itools: InvestigationTools, account: str, cited: list[str]) -> dict:
    """Facts about the shape of the money flow, computed in code so no agent can misread them."""
    flow = case_flow(store, account, cited)
    rel = (itools.account_profile(account) or {}).get("relationships", {})
    loops = itools.find_cycles(account, max_length=6, window_days=7).get("cycles", [])
    return {
        "senders_in_window": flow.get("senders", 0),
        "recipients_in_window": flow.get("recipients", 0),
        "total_in": flow.get("total_in", 0.0),
        "total_out": flow.get("total_out", 0.0),
        "share_passed_through": flow.get("passed_through", 0),
        "activity_span_hours": flow.get("span_hours", 0),
        "in_loop_within_7_days": any(c.get("within_window") for c in loops),
        **{k: v for k, v in rel.items() if k != "note"},
    }


def _signal_names(alert: dict) -> set[str]:
    return {s.get("name") for s in alert.get("signals", []) if isinstance(s, dict)}


def pattern_conflict(pattern_id: str, shape: dict, alert: dict | None = None) -> str | None:
    """Why `pattern_id` is impossible for this measured shape, or None if it fits."""
    s, r = shape.get("senders_in_window", 0), shape.get("recipients_in_window", 0)
    rep_in = shape.get("inflow_from_repeat_senders", 0.0)
    rep_out = shape.get("outflow_to_repeat_recipients", 0.0)
    loop = shape.get("in_loop_within_7_days", False) or "in_cycle" in _signal_names(alert or {})
    rules: dict[str, list[tuple[bool, str]]] = {
        "fan_in": [(s >= 3, f"fan-in needs 3+ senders, measured {s}"),
                   (r <= 2, f"fan-in forwards to one or two accounts, measured {r} recipients"),
                   (rep_in < 0.5, f"fan-in uses fresh senders, but {rep_in:.0%} of money came from repeat senders")],
        "fan_out": [(r >= 3, f"fan-out needs 3+ recipients, measured {r}"),
                    (s <= 2, f"fan-out starts from one or two sources, measured {s} senders"),
                    (rep_out < 0.5, f"fan-out uses fresh recipients, but {rep_out:.0%} went to repeat recipients")],
        "gather_scatter": [(s >= 3 and r >= 3, f"gather-scatter needs 3+ senders and 3+ recipients, measured {s} and {r}")],
        "bipartite": [(s >= 3 and r >= 3, f"bipartite needs 3+ senders and 3+ recipients, measured {s} and {r}")],
        "stack": [(s >= 3 and r >= 3, f"stacked layers need 3+ senders and 3+ recipients, measured {s} and {r}")],
        "scatter_gather": [(r >= 3 and s <= 2, f"scatter-gather splits one source to many, measured {s} senders and {r} recipients")],
        "layering_chain": [(s <= 2 and r <= 2, f"a layering chain passes money one-to-one, measured {s} senders and {r} recipients"),
                           (not loop, "money returns to this account in a loop, which is a round trip, not a chain")],
        "cycle": [(loop, "no loop of transfers back to this account was found")],
        "wholesale": [(s >= 3, f"wholesale trade has many buyers, measured {s} senders"),
                      (rep_in >= 0.5, f"wholesale buyers repeat, but only {rep_in:.0%} of money came from repeat senders")],
        "payroll": [(r >= 3, f"payroll pays many staff, measured {r} recipients"),
                    (rep_out >= 0.5, f"payroll pays the same staff, but only {rep_out:.0%} went to repeat recipients")],
        "investment": [(r <= 2, f"an investment transfer goes to one account, measured {r} recipients"),
                       (s <= 2, f"an investment transfer comes from one income source, measured {s} senders"),
                       (not loop, "money returns to this account in a loop")],
        "property": [(r <= 2, f"a property purchase pays one seller, measured {r} recipients"),
                     (s <= 2, f"a property purchase is funded by one loan or source, measured {s} senders"),
                     (not loop, "money returns to this account in a loop")],
        "family": [(s <= 4 and r <= 4, f"family transfers stay in a small group, measured {s} senders and {r} recipients"),
                   (rep_in >= 0.3 or rep_out >= 0.3 or loop, "family members transact repeatedly, but counterparties are all new")],
    }
    for ok, reason in rules.get(pattern_id, []):
        if not ok:
            return reason
    return None


def structural_candidates(shape: dict, alert: dict) -> list[str]:
    """Typologies the measured shape makes plausible, most specific first."""
    order = ["cycle", "gather_scatter", "fan_in", "fan_out", "scatter_gather", "layering_chain",
             "wholesale", "payroll", "investment", "property", "family", "bipartite", "stack"]
    return [p for p in order if pattern_conflict(p, shape, alert) is None]


# ---------------------------------------------------------------- numbers by code

_PLACEHOLDER = re.compile(r"\{(total|count|accounts|total_in|total_out|senders|recipients|passed_through)\}")


_ANY_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def fill_claim(claim: dict, store: TransactionStore, account: str | None = None) -> dict:
    """Replace placeholders such as {total} and {count} with values computed from the cited rows.

    Duplicate citations are dropped first, so a transaction is never counted twice. Models
    sometimes invent their own names ({total_out}, {share_passed_through}); the common ones are
    understood too, and any that are not stay visible so the fact-checker rejects the claim.
    """
    ids = list(dict.fromkeys(claim.get("txn_ids") or []))
    rows = store.get_txns(ids)
    total = float(rows.amount.sum()) if not rows.empty else 0.0
    senders = {a for a in rows.from_account if a != account} if not rows.empty else set()
    recipients = {a for a in rows.to_account if a != account} if not rows.empty else set()
    if account and not rows.empty:
        t_in = float(rows[rows.to_account == account].amount.sum())
        t_out = float(rows[rows.from_account == account].amount.sum())
    else:
        t_in = t_out = total
    share = f"{min(t_out / t_in, 1.0):.0%}" if t_in and t_out else None
    values = {
        "total": inr(total), "amount": inr(total), "count": str(len(rows)), "payments": str(len(rows)),
        "senders": str(len(senders)), "recipients": str(len(recipients)), "accounts": str(len(senders | recipients)),
        "total_in": inr(t_in), "inflow": inr(t_in), "total_inflow": inr(t_in),
        "total_out": inr(t_out), "outflow": inr(t_out), "total_outflow": inr(t_out),
    }
    if share:
        values.update({"share_passed_through": share, "passed_through": share, "share": share, "pass_through": share})
    text = claim.get("statement", "")
    used_total = "{total}" in text or "{amount}" in text
    text = _ANY_PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), text)
    out = {**claim, "statement": text, "txn_ids": ids}
    if used_total:
        out["amount"] = round(total, 2)
    return out


_WESTERN_RUPEES = re.compile(r"₹\s?(\d{1,3}(?:,\d{3}){2,})(\.\d+)?")
_PATTERN_IDS = re.compile(r"\b(fan_out|fan_in|scatter_gather|gather_scatter|layering_chain)\b")


def fill_summary(summary: str, shape: dict) -> str:
    share = f"{shape.get('share_passed_through', 0):.0%}"
    values = {
        "total_in": inr(shape.get("total_in", 0)), "total_out": inr(shape.get("total_out", 0)),
        "senders": str(shape.get("senders_in_window", 0)), "recipients": str(shape.get("recipients_in_window", 0)),
        "passed_through": share, "share_passed_through": share,
    }
    text = _ANY_PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), summary)
    # Indian digit grouping (₹68,69,564, not ₹6,869,564), and readable pattern names.
    text = _WESTERN_RUPEES.sub(lambda m: inr(float(m.group(1).replace(",", ""))) + (m.group(2) or ""), text)
    return _PATTERN_IDS.sub(lambda m: m.group(1).replace("_", "-"), text)


_RUPEES = re.compile(r"₹\s?[0-9][0-9,]*(?:\.[0-9]+)?(?:\s?(?:lakh|lakhs|crore|crores|L|Cr|M|million))?", re.I)


def correct_claim(claim: dict, problems: list[str], store: TransactionStore,
                  account: str | None = None) -> dict | None:
    """Repair a claim whose only faults are a miscalculated amount or a miscounted number.

    Returns None if it cannot be repaired.

    The cited transactions are the evidence; if the agent merely added them up wrong, the claim is
    rewritten with the true total rather than thrown away.
    """
    if not claim.get("txn_ids") or not problems or not all(p.startswith(("amount", "count")) for p in problems):
        return None
    ids = list(dict.fromkeys(claim["txn_ids"]))
    rows = store.get_txns(ids)
    if rows.empty or len(rows) != len(ids):
        return None
    statement, notes, out = claim.get("statement", ""), [], {**claim, "txn_ids": ids}
    if any(p.startswith("amount") for p in problems):
        amounts = _RUPEES.findall(statement)
        if len(amounts) != 1:
            return None
        total = float(rows.amount.sum())
        statement = statement.replace(amounts[0], inr(total), 1)
        out["amount"] = round(total, 2)
        notes.append(amounts[0].strip())
    if any(p.startswith("count") for p in problems):
        from parakh.agents.verifier import count_mentions  # local import avoids a cycle

        for w in sorted(count_mentions(statement, rows, account), key=lambda w: -w["start"]):
            statement = statement[:w["start"]] + w["replacement"] + statement[w["end"]:]
            notes.append(f"{w['text']} {w['noun']}")
        out.pop("count", None)
    out["statement"] = statement
    out["corrected_from"] = ", ".join(notes)
    return out
