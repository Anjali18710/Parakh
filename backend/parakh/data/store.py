"""In-memory transaction store.

Loads the transaction file once and keeps two things strictly apart:
  * `df`     – what the agents may see (no labels).
  * `labels` – the answer key of the synthetic data (which transfers are laundering), never shown to the agents.
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import networkx as nx
import pandas as pd

AGENT_COLUMNS = ["txn_id", "timestamp", "from_account", "to_account", "amount", "payment_format", "remarks"]
LABEL_COLUMNS = ["txn_id", "is_laundering", "pattern", "instance_id"]


class TransactionStore:
    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col in ("pattern", "instance_id"):
            if col in df:
                df[col] = df[col].fillna("")
        if "is_laundering" not in df:
            df["is_laundering"] = 0
        # Remarks are raw customer text. Tools never pass them to an agent unfiltered (see data/remarks.py).
        df["remarks"] = df["remarks"].fillna("").astype(str) if "remarks" in df else ""
        df = df.sort_values("timestamp").reset_index(drop=True)

        self.labels = df[[c for c in LABEL_COLUMNS if c in df]].set_index("txn_id")
        self.df = df[AGENT_COLUMNS].copy()
        self.by_id = self.df.set_index("txn_id", drop=False)

    @classmethod
    def from_csv(cls, path: str) -> "TransactionStore":
        df = pd.read_csv(path)
        from parakh.data.remarks import remarks_path  # local import keeps the store free of heavier deps

        rp = remarks_path(path)
        if "remarks" not in df and rp.exists() and rp != Path(path):
            df = df.merge(pd.read_csv(rp, keep_default_na=False), on="txn_id", how="left")
        return cls(df)

    def injection_attempts(self, account: str, extra_txn_ids: list[str] | None = None) -> list[dict]:
        """Remarks on this account's payments (and any other cited payments) that try to instruct an AI."""
        from parakh.data.remarks import injection_reasons

        rows = pd.concat([self.account_txns(account), self.get_txns(list(extra_txn_ids or []))])
        rows = rows.drop_duplicates("txn_id")
        out = []
        for r in rows[rows.remarks != ""].itertuples():
            reasons = injection_reasons(r.remarks)
            if reasons:
                out.append({"txn_id": r.txn_id, "from": r.from_account, "to": r.to_account,
                            "amount": float(r.amount), "remark": r.remarks, "reasons": reasons})
        return out

    # --- lookups used by tools ---------------------------------------------
    @cached_property
    def outgoing(self) -> dict[str, pd.DataFrame]:
        return {k: g for k, g in self.df.groupby("from_account")}

    @cached_property
    def incoming(self) -> dict[str, pd.DataFrame]:
        return {k: g for k, g in self.df.groupby("to_account")}

    @cached_property
    def graph(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for row in self.df.itertuples(index=False):
            g.add_edge(row.from_account, row.to_account, txn_id=row.txn_id, amount=row.amount, ts=row.timestamp)
        return g

    def out_txns(self, account: str) -> pd.DataFrame:
        return self.outgoing.get(account, self.df.iloc[0:0])

    def in_txns(self, account: str) -> pd.DataFrame:
        return self.incoming.get(account, self.df.iloc[0:0])

    def account_txns(self, account: str) -> pd.DataFrame:
        return pd.concat([self.out_txns(account), self.in_txns(account)]).sort_values("timestamp")

    def has_account(self, account: str) -> bool:
        return account in self.outgoing or account in self.incoming

    def get_txns(self, txn_ids: list[str]) -> pd.DataFrame:
        ids = [t for t in txn_ids if t in self.by_id.index]
        return self.by_id.loc[ids]

    # --- answer key (tests only) -------------------------------------------
    @cached_property
    def laundering_accounts(self) -> dict[str, set[str]]:
        """account -> set of laundering instance ids it took part in."""
        bad = self.labels[self.labels.is_laundering == 1]
        merged = self.df.set_index("txn_id").join(bad, how="inner")
        out: dict[str, set[str]] = {}
        for row in merged.itertuples():
            for acc in (row.from_account, row.to_account):
                out.setdefault(acc, set()).add(row.instance_id)
        return out
