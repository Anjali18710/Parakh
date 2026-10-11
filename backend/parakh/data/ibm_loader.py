"""Convert IBM's synthetic AML dataset into Parakh's transaction format.

Dataset: "IBM Transactions for Anti Money Laundering (AML)" on Kaggle.
Download `HI-Small_Trans.csv` and `HI-Small_Patterns.txt`, then run:

    python -m parakh.data.ibm_loader \
        --trans HI-Small_Trans.csv --patterns HI-Small_Patterns.txt \
        --out data/ibm_sample.csv.gz --days 10

The full file has millions of rows, so we keep a time window (default 10 days)
which still contains many labelled laundering attempts.
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

PATTERN_MAP = {
    "FAN-OUT": "fan_out",
    "FAN-IN": "fan_in",
    "CYCLE": "cycle",
    "SCATTER-GATHER": "scatter_gather",
    "GATHER-SCATTER": "gather_scatter",
    "STACK": "stack",
    "BIPARTITE": "bipartite",
    "RANDOM": "random",
}

_BEGIN = re.compile(r"BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Z\-]+)")


def _key(ts, from_bank, from_acc, to_bank, to_acc, amount) -> tuple:
    return (
        str(ts).strip(),
        f"{str(from_bank).strip()}-{str(from_acc).strip()}",
        f"{str(to_bank).strip()}-{str(to_acc).strip()}",
        round(float(amount), 2),
    )


def parse_patterns(path: str) -> dict[tuple, tuple[str, str]]:
    """Map each laundering transaction to (pattern, instance_id)."""
    mapping: dict[tuple, tuple[str, str]] = {}
    current, count = None, 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            m = _BEGIN.search(line)
            if m:
                name = m.group(1).rstrip("-").strip()
                current = PATTERN_MAP.get(name, name.lower().replace("-", "_"))
                count += 1
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                current = None
                continue
            if current and line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 8:
                    k = _key(parts[0], parts[1], parts[2], parts[3], parts[4], parts[7])
                    mapping[k] = (current, f"{current}_{count:05d}")
    return mapping


def convert(trans_path: str, patterns_path: str | None, days: int, max_rows: int | None) -> pd.DataFrame:
    raw = pd.read_csv(trans_path, nrows=max_rows)
    raw.columns = [c.strip() for c in raw.columns]
    raw["ts"] = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M")
    start = raw["ts"].min()
    raw = raw[raw["ts"] < start + pd.Timedelta(days=days)].copy()

    df = pd.DataFrame(
        {
            "timestamp": raw["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "from_account": raw["From Bank"].astype(str) + "-" + raw["Account"].astype(str),
            "to_account": raw["To Bank"].astype(str) + "-" + raw["Account.1"].astype(str),
            "amount": raw["Amount Paid"].round(2),
            "payment_format": raw["Payment Format"],
            "is_laundering": raw["Is Laundering"].astype(int),
        }
    )
    df["pattern"] = ""
    df["instance_id"] = ""
    if patterns_path:
        mapping = parse_patterns(patterns_path)
        keys = [
            _key(t, fb, fa, tb, ta, a)
            for t, fb, fa, tb, ta, a in zip(
                raw["Timestamp"], raw["From Bank"], raw["Account"], raw["To Bank"], raw["Account.1"], raw["Amount Paid"]
            )
        ]
        found = [mapping.get(k) for k in keys]
        df["pattern"] = [f[0] if f else "" for f in found]
        df["instance_id"] = [f[1] if f else "" for f in found]
    # Laundering rows that are not part of a documented pattern get their own instance.
    lone = (df.is_laundering == 1) & (df.instance_id == "")
    df.loc[lone, "pattern"] = "unlabelled"
    df.loc[lone, "instance_id"] = [f"unlabelled_{i:05d}" for i in range(int(lone.sum()))]

    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:08d}" for i in range(len(df))])
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trans", required=True)
    p.add_argument("--patterns")
    p.add_argument("--out", default="data/ibm_sample.csv.gz")
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--max-rows", type=int)
    a = p.parse_args()
    df = convert(a.trans, a.patterns, a.days, a.max_rows)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df):,} transactions ({int(df.is_laundering.sum()):,} laundering) to {a.out}")
    print(df[df.is_laundering == 1].pattern.value_counts().to_string())


if __name__ == "__main__":
    main()
