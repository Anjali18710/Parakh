"""Money-flow summary of one case, for the dashboard's flow graph and summary strip."""
from __future__ import annotations

import pandas as pd

from parakh.data.store import TransactionStore

MAX_PARTIES = 7


def _side(df: pd.DataFrame, party_col: str, cited: set[str]) -> list[dict]:
    if df.empty:
        return []
    groups = []
    for party, g in df.groupby(party_col):
        groups.append({"account": party, "total": round(float(g.amount.sum()), 2), "count": len(g),
                       "cited": bool(cited & set(g.txn_id)), "txn_ids": list(g.sort_values("timestamp").txn_id)[:10],
                       "first": str(g.timestamp.min()), "last": str(g.timestamp.max())})
    groups.sort(key=lambda x: (-x["cited"], -x["total"]))
    top, rest = groups[:MAX_PARTIES], groups[MAX_PARTIES:]
    if rest:
        top.append({"account": f"{len(rest)} other" + ("s" if len(rest) > 1 else ""), "total": round(sum(r["total"] for r in rest), 2),
                    "count": sum(r["count"] for r in rest), "cited": any(r["cited"] for r in rest),
                    "txn_ids": [], "first": min(r["first"] for r in rest), "last": max(r["last"] for r in rest),
                    "other": True})
    return top


def case_flow(store: TransactionStore, account: str, cited_ids: list[str]) -> dict:
    """Money in and out of `account` around the activity the case is about.

    The window is the span of the cited transactions (padded by a day). Without a
    report yet, it is the account's busiest three days.
    """
    txns = store.account_txns(account)
    if txns.empty:
        return {"account": account, "inflows": [], "outflows": []}
    cited = set(cited_ids)
    own_cited = txns[txns.txn_id.isin(cited)]
    if not own_cited.empty:
        # Centre on the heaviest 72-hour burst of cited activity, so a stray citation weeks away
        # does not stretch the picture.
        best_t, best_sum = None, -1.0
        for t in own_cited.timestamp:
            in_win = own_cited[(own_cited.timestamp >= t) & (own_cited.timestamp <= t + pd.Timedelta(hours=72))]
            if in_win.amount.sum() > best_sum:
                best_t, best_sum = t, float(in_win.amount.sum())
        burst = own_cited[(own_cited.timestamp >= best_t) & (own_cited.timestamp <= best_t + pd.Timedelta(hours=72))]
        start = burst.timestamp.min() - pd.Timedelta(days=1)
        end = burst.timestamp.max() + pd.Timedelta(days=1)
    else:
        daily = txns.groupby(txns.timestamp.dt.floor("D")).amount.sum()
        peak = daily.idxmax()
        start, end = peak - pd.Timedelta(days=1), peak + pd.Timedelta(days=2)
    win = txns[(txns.timestamp >= start) & (txns.timestamp <= end)]
    inc, out = win[win.to_account == account], win[win.from_account == account]
    total_in, total_out = float(inc.amount.sum()), float(out.amount.sum())
    active = win[win.txn_id.isin(cited)] if not own_cited.empty else win
    if active.empty:
        active = win
    span_h = (active.timestamp.max() - active.timestamp.min()).total_seconds() / 3600 if len(active) > 1 else 0
    return {
        "account": account,
        "window": {"start": str(start), "end": str(end)},
        "total_in": round(total_in, 2),
        "total_out": round(total_out, 2),
        "passed_through": round(min(total_in, total_out) / total_in, 3) if total_in else 0,
        "senders": int(inc.from_account.nunique()),
        "recipients": int(out.to_account.nunique()),
        "span_hours": round(span_h, 1),
        "inflows": _side(inc, "from_account", cited),
        "outflows": _side(out, "to_account", cited),
    }
