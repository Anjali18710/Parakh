"""Convert IBM's AML dataset into Parakh's transaction format.

Dataset: "IBM Transactions for Anti Money Laundering (AML)" on Kaggle (IBM's AMLworld simulator).
Download `HI-Small_Trans.csv` and `HI-Small_Patterns.txt`, then run:

    python -m parakh.data.ibm_loader --trans HI-Small_Trans.csv --patterns HI-Small_Patterns.txt

The full file has about 5 million transactions between 515,000 accounts, far more than a small
server can hold, so we keep a fair slice:

  * complete laundering schemes, the same number of each type, chosen at random;
  * ordinary accounts that are quiet (a few payments) and busy ones (dozens to hundreds), so the
    screening rules also flag innocent accounts, as they do in a real bank;
  * every transaction that touches a chosen account.

On the way: amounts are converted to rupees at fixed rates, accounts get neutral ids
(AC123456, so ids cannot hint at the answer) and self-transfers are dropped.
"""
from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN_MAP = {
    "FAN-OUT": "fan_out", "FAN-IN": "fan_in", "CYCLE": "cycle", "SCATTER-GATHER": "scatter_gather",
    "GATHER-SCATTER": "gather_scatter", "STACK": "stack", "BIPARTITE": "bipartite", "RANDOM": "random",
}

# Approximate rupees per unit in September 2022, the month the data is dated. Fixed on purpose so
# the conversion is reproducible; exactness does not matter for detection.
INR_PER_UNIT = {
    "US Dollar": 80.0, "Euro": 80.0, "Swiss Franc": 82.0, "UK Pound": 92.0, "Yuan": 11.5, "Yen": 0.56,
    "Rupee": 1.0, "Ruble": 1.35, "Shekel": 23.5, "Canadian Dollar": 61.0, "Australian Dollar": 54.0,
    "Mexican Peso": 4.0, "Brazil Real": 15.4, "Saudi Riyal": 21.3, "Bitcoin": 1_600_000.0,
}

_BEGIN = re.compile(r"BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Z\-]+)")


@dataclass
class Scheme:
    kind: str
    instance_id: str
    keys: list[tuple] = field(default_factory=list)
    accounts: set[str] = field(default_factory=set)


def _account(bank, acc) -> str:
    return f"{str(bank).strip()}-{str(acc).strip()}"


def _key(ts, src, dst, amount) -> tuple:
    return str(ts).strip(), src, dst, round(float(amount), 2)


def parse_schemes(path: str | Path) -> list[Scheme]:
    """Every laundering attempt in the patterns file, with its transactions and accounts."""
    schemes: list[Scheme] = []
    current: Scheme | None = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            m = _BEGIN.search(line)
            if m:
                kind = PATTERN_MAP.get(m.group(1).rstrip("-"), m.group(1).lower().replace("-", "_"))
                current = Scheme(kind, f"{kind}_{len(schemes) + 1:05d}")
                schemes.append(current)
            elif line.startswith("END LAUNDERING ATTEMPT"):
                current = None
            elif current and line:
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 8:
                    src, dst = _account(p[1], p[2]), _account(p[3], p[4])
                    current.keys.append(_key(p[0], src, dst, p[7]))
                    current.accounts |= {src, dst}
    return schemes


def load_transactions(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    raw.columns = ["ts", "from_bank", "from_acc", "to_bank", "to_acc", "amount_received", "receiving_currency",
                   "amount_paid", "payment_currency", "payment_format", "is_laundering"]
    df = pd.DataFrame({
        "ts": raw.ts,
        "src": raw.from_bank.str.strip() + "-" + raw.from_acc.str.strip(),
        "dst": raw.to_bank.str.strip() + "-" + raw.to_acc.str.strip(),
        "amount_paid": raw.amount_paid.astype(float),
        "currency": raw.payment_currency.str.strip(),
        "payment_format": raw.payment_format.str.strip(),
        "is_laundering": raw.is_laundering.astype(int),
    })
    return df[df.src != df.dst].reset_index(drop=True)  # self-transfers ("Reinvestment") carry no flow


def choose(df: pd.DataFrame, schemes: list[Scheme], per_type: int, quiet: int, busy: int,
           max_degree: int, seed: int) -> tuple[list[Scheme], set[str]]:
    """Pick whole schemes (per_type of each kind) and samples of innocent quiet and busy accounts."""
    rng = random.Random(seed)
    degree = pd.concat([df.src, df.dst]).value_counts()
    chosen: list[Scheme] = []
    for kind in sorted({s.kind for s in schemes}):
        pool = [s for s in schemes if s.kind == kind and max(degree.get(a, 0) for a in s.accounts) <= max_degree]
        chosen += rng.sample(pool, min(per_type, len(pool)))
    laundering_accounts = set(df.src[df.is_laundering == 1]) | set(df.dst[df.is_laundering == 1])
    innocent = degree[~degree.index.isin(laundering_accounts)]
    quiet_pool = innocent[(innocent >= 3) & (innocent < 40)].index.tolist()
    busy_pool = innocent[(innocent >= 40) & (innocent <= 600)].index.tolist()
    selected = set().union(*(s.accounts for s in chosen))
    selected |= set(rng.sample(quiet_pool, min(quiet, len(quiet_pool))))
    selected |= set(rng.sample(busy_pool, min(busy, len(busy_pool))))
    return chosen, selected


def convert(trans_path: str | Path, patterns_path: str | Path, per_type: int = 10, quiet: int = 2000,
            busy: int = 250, max_degree: int = 2000, seed: int = 7) -> pd.DataFrame:
    df = load_transactions(trans_path)
    schemes = parse_schemes(patterns_path)
    chosen, selected = choose(df, schemes, per_type, quiet, busy, max_degree, seed)
    df = df[df.src.isin(selected) | df.dst.isin(selected)].copy()

    # Label each laundering row with its scheme; laundering rows from schemes we did not pick
    # (an account can take part in several) keep their label under "other_scheme".
    by_key = {k: s for s in schemes for k in s.keys}
    chosen_ids = {s.instance_id for s in chosen}
    keys = [_key(t, a, b, x) for t, a, b, x in zip(df.ts, df.src, df.dst, df.amount_paid)]
    found = [by_key.get(k) for k in keys]
    df["pattern"] = [s.kind if s and s.instance_id in chosen_ids else "" for s in found]
    df["instance_id"] = [s.instance_id if s and s.instance_id in chosen_ids else "" for s in found]
    lone = (df.is_laundering == 1) & (df.instance_id == "")
    df.loc[lone, "pattern"] = "other_scheme"
    df.loc[lone, "instance_id"] = [f"other_{i:05d}" for i in range(int(lone.sum()))]

    unknown = set(df.currency) - set(INR_PER_UNIT)
    if unknown:
        raise ValueError(f"no exchange rate for: {sorted(unknown)}")
    df["amount"] = (df.amount_paid * df.currency.map(INR_PER_UNIT)).round(2)

    accounts = sorted(set(df.src) | set(df.dst))
    codes = np.random.default_rng(seed).choice(np.arange(100_000, 1_000_000), len(accounts), replace=False)
    ids = {acc: f"AC{code}" for acc, code in zip(accounts, codes)}
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df.ts, format="%Y/%m/%d %H:%M").dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "from_account": df.src.map(ids), "to_account": df.dst.map(ids), "amount": df.amount,
        "payment_format": df.payment_format, "is_laundering": df.is_laundering,
        "pattern": df.pattern, "instance_id": df.instance_id,
    }).sort_values("timestamp", kind="stable").reset_index(drop=True)
    out.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(out))])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Cut a fair slice of IBM's AML dataset into Parakh's format.")
    p.add_argument("--trans", required=True)
    p.add_argument("--patterns", required=True)
    p.add_argument("--out", default="data/ibm_transactions.csv.gz")
    p.add_argument("--per-type", type=int, default=10, help="complete laundering schemes of each type")
    p.add_argument("--quiet", type=int, default=2000, help="innocent accounts with 3-39 transactions")
    p.add_argument("--busy", type=int, default=250, help="innocent accounts with 40-600 transactions")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    df = convert(a.trans, a.patterns, a.per_type, a.quiet, a.busy, seed=a.seed)
    df.to_csv(a.out, index=False)
    bad = df[df.is_laundering == 1]
    print(f"Wrote {len(df):,} transactions between {len(set(df.from_account) | set(df.to_account)):,} accounts "
          f"({len(bad):,} laundering, {bad.instance_id.nunique()} schemes) to {a.out}")
    print(bad.drop_duplicates("instance_id").pattern.value_counts().to_string())


if __name__ == "__main__":
    main()
