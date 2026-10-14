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


def fill_claim(claim: dict, store: TransactionStore) -> dict:
    """Replace {total}/{count}/{accounts} in a claim with values computed from its cited transactions."""
    rows = store.get_txns(list(dict.fromkeys(claim.get("txn_ids") or [])))
    total = float(rows.amount.sum()) if not rows.empty else 0.0
    parties = set(rows.from_account) | set(rows.to_account) if not rows.empty else set()
    text = claim.get("statement", "")
    used_total = "{total}" in text
    text = text.replace("{total}", inr(total)).replace("{count}", str(len(rows)))
    text = text.replace("{accounts}", str(max(len(parties) - 1, 0)))
    out = {**claim, "statement": text}
    if used_total:
        out["amount"] = round(total, 2)
    return out


def fill_summary(summary: str, shape: dict) -> str:
    values = {
        "total_in": inr(shape.get("total_in", 0)), "total_out": inr(shape.get("total_out", 0)),
        "senders": str(shape.get("senders_in_window", 0)), "recipients": str(shape.get("recipients_in_window", 0)),
        "passed_through": f"{shape.get('share_passed_through', 0):.0%}",
    }
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), summary)


_RUPEES = re.compile(r"₹\s?[0-9][0-9,]*(?:\.[0-9]+)?(?:\s?(?:lakh|lakhs|crore|crores|L|Cr|M|million))?", re.I)


def correct_claim(claim: dict, problems: list[str], store: TransactionStore) -> dict | None:
    """Repair a claim whose only fault is a miscalculated amount. Returns None if it cannot be repaired.

    The cited transactions are the evidence; if the agent merely added them up wrong, the claim is
    rewritten with the true total rather than thrown away.
    """
    if not claim.get("txn_ids") or not problems or not all(p.startswith("amount") for p in problems):
        return None
    rows = store.get_txns(list(dict.fromkeys(claim["txn_ids"])))
    if rows.empty or len(rows) != len(set(claim["txn_ids"])):
        return None
    amounts = _RUPEES.findall(claim.get("statement", ""))
    if len(amounts) != 1:
        return None
    total = float(rows.amount.sum())
    fixed = claim["statement"].replace(amounts[0], inr(total), 1)
    return {**claim, "statement": fixed, "amount": round(total, 2),
            "corrected_from": amounts[0].strip()}
