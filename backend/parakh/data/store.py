"""In-memory transaction store.

Loads the transaction file once and keeps two things strictly apart:
  * `df`     – what the agents may see (no labels).
  * `labels` – ground truth, used only by the evaluation code.
"""
from __future__ import annotations

from functools import cached_property

import networkx as nx
import pandas as pd

AGENT_COLUMNS = ["txn_id", "timestamp", "from_account", "to_account", "amount", "payment_format"]
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
        df = df.sort_values("timestamp").reset_index(drop=True)

        self.labels = df[[c for c in LABEL_COLUMNS if c in df]].set_index("txn_id")
        self.df = df[AGENT_COLUMNS].copy()
        self.by_id = self.df.set_index("txn_id", drop=False)

    @classmethod
    def from_csv(cls, path: str) -> "TransactionStore":
        return cls(pd.read_csv(path))

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

    # --- ground truth (evaluation only) -------------------------------------
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

    @cached_property
    def instance_patterns(self) -> dict[str, str]:
        bad = self.labels[self.labels.is_laundering == 1]
        return dict(zip(bad.instance_id, bad.pattern))
