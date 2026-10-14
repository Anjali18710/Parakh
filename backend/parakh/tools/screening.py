"""Screening: cheap, deterministic checks that raise alerts on accounts.

These are the kind of rules banks already run. They catch real laundering but
also flag many innocent accounts (employers on salary day, busy shops). The
triage agent decides which alerts are worth a full investigation; the rules
on their own are also our "rules-only" baseline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from parakh.data.store import TransactionStore
from parakh.fmt import inr

DAY = np.timedelta64(24, "h")


@dataclass
class Signal:
    name: str
    detail: str
    txn_ids: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class Alert:
    account: str
    score: float
    signals: list[Signal]

    def to_dict(self) -> dict:
        return {"account": self.account, "score": round(self.score, 2), "signals": [asdict(s) for s in self.signals]}


def _burst(times: np.ndarray, parties: np.ndarray, amounts: np.ndarray, ids: np.ndarray,
           min_parties: int, min_total: float, window=DAY):
    """Find the 24h window with the most distinct counterparties."""
    best = None
    j = 0
    for i in range(len(times)):
        while times[i] - times[j] > window:
            j += 1
        uniq = len(set(parties[j:i + 1]))
        total = float(amounts[j:i + 1].sum())
        if uniq >= min_parties and total >= min_total and (best is None or uniq > best[0]):
            best = (uniq, total, list(ids[j:i + 1]))
    return best


def _pass_through(inc: pd.DataFrame, out: pd.DataFrame, min_in: float = 50_000, hours: int = 48):
    """Share of large inflows that leave the account again within `hours`."""
    big = inc[inc.amount >= min_in]
    if big.empty or out.empty:
        return None
    total_in, moved, ids = 0.0, 0.0, []
    out_t = out.timestamp.values
    for row in big.itertuples():
        mask = (out_t >= np.datetime64(row.timestamp)) & (out_t <= np.datetime64(row.timestamp) + np.timedelta64(hours, "h"))
        leaving = out[mask]
        total_in += row.amount
        moved += min(float(leaving.amount.sum()), row.amount)
        if not leaving.empty:
            ids += [row.txn_id] + list(leaving.txn_id)[:5]
    ratio = moved / total_in if total_in else 0
    return ratio, total_in, ids


def screen(store: TransactionStore, min_score: float = 1.0) -> list[Alert]:
    alerts: dict[str, list[Signal]] = {}

    def add(acc, sig):
        alerts.setdefault(acc, []).append(sig)

    accounts = set(store.outgoing) | set(store.incoming)
    for acc in accounts:
        out, inc = store.out_txns(acc), store.in_txns(acc)

        if len(out) >= 5:
            b = _burst(out.timestamp.values, out.to_account.values, out.amount.values, out.txn_id.values, 5, 500_000)
            if b:
                add(acc, Signal("fan_out_burst", f"paid {b[0]} different accounts {inr(b[1])} within 24h", b[2][:15], 1.0))
        if len(inc) >= 6:
            b = _burst(inc.timestamp.values, inc.from_account.values, inc.amount.values, inc.txn_id.values, 6, 200_000)
            if b:
                add(acc, Signal("fan_in_burst", f"received from {b[0]} different accounts {inr(b[1])} within 24h", b[2][:15], 1.0))

        pt = _pass_through(inc, out)
        if pt and pt[0] >= 0.8:
            add(acc, Signal("rapid_pass_through", f"{pt[0]:.0%} of {inr(pt[1])} large inflows left within 48h", pt[2][:15], 1.2))

        near = pd.concat([out, inc])
        near = near[(near.amount >= 900_000) & (near.amount < 1_000_000)]
        if len(near) >= 2:
            add(acc, Signal("near_threshold", f"{len(near)} payments just under ₹10 lakh", list(near.txn_id)[:15], 0.8))

    # Cycles among large, recent transfers
    big = store.df[store.df.amount >= 50_000]
    g = nx.DiGraph()
    for row in big.itertuples(index=False):
        g.add_edge(row.from_account, row.to_account)
    for cyc in nx.simple_cycles(g, length_bound=6):
        if len(cyc) < 3:
            continue
        for acc in cyc:
            if not any(s.name == "in_cycle" for s in alerts.get(acc, [])):
                add(acc, Signal("in_cycle", f"part of a {len(cyc)}-account loop of large transfers: {' → '.join(cyc)}", [], 1.5))

    result = [Alert(a, sum(s.weight for s in sigs), sigs) for a, sigs in alerts.items()]
    result = [a for a in result if a.score >= min_score]
    result.sort(key=lambda a: (-a.score, a.account))
    return result


if __name__ == "__main__":
    # Pre-compute the alert cache: python -m parakh.tools.screening [data_path]
    import sys

    from parakh.agents.service import load_or_screen
    from parakh.config import settings

    path = sys.argv[1] if len(sys.argv) > 1 else settings.data_path
    print(f"{len(load_or_screen(TransactionStore.from_csv(path), path))} alerts cached")
