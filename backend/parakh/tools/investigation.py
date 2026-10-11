"""Investigation tools the agents call.

The design rule: an LLM can never read millions of rows, so every tool
returns a *compact summary* (a few numbers and at most ~20 transactions),
and every fact it returns carries the transaction IDs it came from. Those IDs
are what the report cites and what the verifier checks.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import networkx as nx
import pandas as pd

from parakh.data.store import TransactionStore

MAX_ROWS = 20


def _fmt_txn(row) -> dict:
    return {
        "txn_id": row.txn_id,
        "time": pd.Timestamp(row.timestamp).strftime("%Y-%m-%d %H:%M"),
        "from": row.from_account,
        "to": row.to_account,
        "amount": round(float(row.amount), 2),
        "format": row.payment_format,
    }


class InvestigationTools:
    def __init__(self, store: TransactionStore):
        self.store = store
        self._graphs: dict[float, tuple[pd.DataFrame, nx.DiGraph]] = {}

    def _large_graph(self, min_amount: float):
        if min_amount not in self._graphs:
            df = self.store.df[self.store.df.amount >= min_amount]
            g = nx.DiGraph()
            g.add_edges_from(zip(df.from_account, df.to_account))
            self._graphs[min_amount] = (df, g)
        return self._graphs[min_amount]

    # ------------------------------------------------------------------ 1
    def account_profile(self, account: str) -> dict:
        """Overall behaviour of an account and its busiest 24 hours compared with a normal day."""
        s = self.store
        if not s.has_account(account):
            return {"error": f"Unknown account {account}"}
        out, inc = s.out_txns(account), s.in_txns(account)
        allt = s.account_txns(account)
        days_active = max(1, allt.timestamp.dt.normalize().nunique())
        daily = allt.groupby(allt.timestamp.dt.floor("D")).amount.sum()
        busiest_day = daily.idxmax()
        return {
            "account": account,
            "first_seen": str(allt.timestamp.min())[:16],
            "last_seen": str(allt.timestamp.max())[:16],
            "active_days": int(days_active),
            "incoming": {"count": len(inc), "total": round(float(inc.amount.sum()), 2),
                         "distinct_senders": int(inc.from_account.nunique())},
            "outgoing": {"count": len(out), "total": round(float(out.amount.sum()), 2),
                         "distinct_recipients": int(out.to_account.nunique())},
            "typical_payment": round(float(allt.amount.median()), 2),
            "largest_payment": round(float(allt.amount.max()), 2),
            "busiest_day": {"date": str(busiest_day)[:10], "total": round(float(daily.max()), 2),
                            "vs_median_day": round(float(daily.max() / max(daily.median(), 1)), 1)},
            "payment_formats": allt.payment_format.value_counts().to_dict(),
        }

    # ------------------------------------------------------------------ 2
    def list_transactions(self, account: str, direction: str = "both", start: str | None = None,
                          end: str | None = None, min_amount: float = 0, limit: int = MAX_ROWS) -> dict:
        """Transactions of an account, largest first, optionally filtered by date range and minimum amount."""
        s = self.store
        df = {"in": s.in_txns(account), "out": s.out_txns(account)}.get(direction, s.account_txns(account))
        if start:
            df = df[df.timestamp >= pd.Timestamp(start)]
        if end:
            df = df[df.timestamp <= pd.Timestamp(end)]
        df = df[df.amount >= min_amount]
        total = len(df)
        df = df.sort_values("amount", ascending=False).head(min(int(limit), MAX_ROWS)).sort_values("timestamp")
        return {"account": account, "matching": total, "shown": len(df),
                "transactions": [_fmt_txn(r) for r in df.itertuples()]}

    # ------------------------------------------------------------------ 3
    def trace_money(self, account: str, direction: str = "out", hops: int = 2,
                    window_hours: int = 72, min_amount: float = 10_000) -> dict:
        """Follow money forward ('out') or backward ('in') hop by hop, respecting time order.

        At each hop we only follow transfers that happen after (forward) or
        before (backward) the previous one, within `window_hours`.
        """
        s = self.store
        hops = max(1, min(int(hops), 4))
        window = timedelta(hours=int(window_hours))
        frontier = [(account, None)]  # (account, reference time)
        edges, seen = [], set()
        for hop in range(1, hops + 1):
            nxt = []
            for acc, t_ref in frontier:
                df = s.out_txns(acc) if direction == "out" else s.in_txns(acc)
                df = df[df.amount >= min_amount]
                if t_ref is not None:
                    if direction == "out":
                        df = df[(df.timestamp >= t_ref) & (df.timestamp <= t_ref + window)]
                    else:
                        df = df[(df.timestamp <= t_ref) & (df.timestamp >= t_ref - window)]
                for r in df.sort_values("amount", ascending=False).head(8).itertuples():
                    if r.txn_id in seen:
                        continue
                    seen.add(r.txn_id)
                    e = _fmt_txn(r)
                    e["hop"] = hop
                    edges.append(e)
                    nxt.append((r.to_account if direction == "out" else r.from_account, r.timestamp))
            frontier = nxt
            if not frontier:
                break
        # Where did the money end up, and how much of it?
        last_hop = max((e["hop"] for e in edges), default=0)
        ends = defaultdict(float)
        for e in edges:
            if e["hop"] == last_hop:
                ends[e["to"] if direction == "out" else e["from"]] += e["amount"]
        first_total = sum(e["amount"] for e in edges if e["hop"] == 1)
        last_total = sum(ends.values())
        return {
            "account": account,
            "direction": direction,
            "hops_reached": last_hop,
            "transfers": edges[:30],
            "hop1_total": round(first_total, 2),
            "final_hop_total": round(last_total, 2),
            "share_reaching_final_hop": round(last_total / first_total, 3) if first_total else 0,
            "final_hop_accounts": dict(sorted(ends.items(), key=lambda kv: -kv[1])[:10]),
        }

    # ------------------------------------------------------------------ 4
    def find_cycles(self, account: str, max_length: int = 6, window_days: int = 7,
                    min_amount: float = 50_000) -> dict:
        """Look for loops of large transfers that start and end at this account within `window_days`."""
        df, g = self._large_graph(float(min_amount))
        if account not in g:
            return {"account": account, "cycles": []}
        depth = int(max_length) - 1
        fwd = nx.single_source_shortest_path_length(g, account, cutoff=depth)
        bwd = nx.single_source_shortest_path_length(g.reverse(copy=False), account, cutoff=depth)
        local = g.subgraph(set(fwd) & set(bwd))
        found = []
        for cyc in nx.simple_cycles(local, length_bound=int(max_length)):
            if account not in cyc or len(cyc) < 2:
                continue
            i = cyc.index(account)
            cyc = cyc[i:] + cyc[:i]
            legs = []
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                pair = df[(df.from_account == a) & (df.to_account == b)].sort_values("timestamp")
                legs.append(pair.iloc[-1] if not pair.empty else None)
            if any(leg is None for leg in legs):
                continue
            times = [leg.timestamp for leg in legs]
            span = (max(times) - min(times)).total_seconds() / 86400
            found.append({
                "accounts": cyc,
                "length": len(cyc),
                "span_days": round(span, 2),
                "within_window": span <= window_days,
                "transfers": [_fmt_txn(leg) for leg in legs],
                "amount_kept_round_trip": round(float(legs[-1].amount / legs[0].amount), 3),
            })
            if len(found) >= 5:
                break
        return {"account": account, "cycles": found}

    # ------------------------------------------------------------------ 5
    def common_counterparties(self, account: str, direction: str = "out", window_hours: int = 72,
                              min_amount: float = 10_000) -> dict:
        """Did the accounts this one paid (or was paid by) all pass money on to the same place?

        'out': recipients of this account -> where they sent money next (scatter-gather).
        'in' : senders to this account -> where their money came from (shared source).
        """
        s = self.store
        window = timedelta(hours=int(window_hours))
        first = s.out_txns(account) if direction == "out" else s.in_txns(account)
        first = first[first.amount >= min_amount]
        links = defaultdict(list)
        for r in first.itertuples():
            mid = r.to_account if direction == "out" else r.from_account
            nxt = s.out_txns(mid) if direction == "out" else s.in_txns(mid)
            nxt = nxt[nxt.amount >= min_amount]
            if direction == "out":
                nxt = nxt[(nxt.timestamp >= r.timestamp) & (nxt.timestamp <= r.timestamp + window)]
            else:
                nxt = nxt[(nxt.timestamp <= r.timestamp) & (nxt.timestamp >= r.timestamp - window)]
            for n in nxt.itertuples():
                far = n.to_account if direction == "out" else n.from_account
                if far != account:
                    links[far].append({"via": mid, "first_txn": r.txn_id, "second_txn": n.txn_id,
                                       "amount": round(float(n.amount), 2)})
        shared = {k: v for k, v in links.items() if len({x["via"] for x in v}) >= 2}
        shared = dict(sorted(shared.items(), key=lambda kv: -len(kv[1]))[:5])
        return {
            "account": account,
            "direction": direction,
            "first_hop_accounts": int(first.to_account.nunique() if direction == "out" else first.from_account.nunique()),
            "shared_accounts": {k: {"reached_via": len({x["via"] for x in v}),
                                    "total": round(sum(x["amount"] for x in v), 2),
                                    "links": v[:8]} for k, v in shared.items()},
        }
