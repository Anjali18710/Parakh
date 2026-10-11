"""Synthetic bank-transaction generator with planted money-laundering patterns.

The generator produces ordinary activity (person-to-person payments, shop
payments, salaries, bills) plus a known set of laundering "instances", each
following one of six classic typologies. It also produces innocent accounts
that *look* suspicious to simple rules (employers paying salaries look like
fan-out, busy shops look like fan-in). Telling those apart is the hard part.

Every row carries ground-truth labels (`is_laundering`, `pattern`,
`instance_id`) so we can measure how well the agents do.

Usage:
    python -m parakh.data.generator --out data/demo_transactions.csv.gz
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PATTERNS = ["fan_out", "fan_in", "cycle", "scatter_gather", "gather_scatter", "layering_chain"]
FORMATS_NORMAL = ["UPI", "UPI", "UPI", "IMPS", "NEFT"]
FORMATS_LAUNDER = ["IMPS", "NEFT", "RTGS", "IMPS"]
ROLE_NAMES = {
    "AC": "retail customer", "MR": "shop", "EM": "employer", "UT": "utility", "SP": "supplier",
    "CL": "business client", "MU": "mule account", "EX": "external source", "DS": "distributor",
    "MF": "manufacturer", "BR": "broker", "BD": "builder", "BK": "bank loan desk",
}


@dataclass
class GeneratorConfig:
    seed: int = 7
    start: datetime = datetime(2026, 7, 1)
    days: int = 60
    n_retail: int = 1500
    n_merchants: int = 30
    n_employers: int = 12
    n_utilities: int = 5
    p2p_per_day: float = 0.25          # average P2P payments per retail account per day
    shop_per_day: float = 0.35         # average shop payments per retail account per day
    instances_per_pattern: int = 6
    reuse_retail_as_mule: float = 0.3  # share of mules recruited from ordinary customers
    n_distributors: int = 15
    n_investors: int = 70
    n_home_buyers: int = 25
    n_family_loops: int = 15


@dataclass
class _Builder:
    cfg: GeneratorConfig
    rng: np.random.Generator
    rows: list[dict] = field(default_factory=list)

    def add(self, ts, src, dst, amount, fmt, pattern="", instance=""):
        self.rows.append(
            {
                "timestamp": ts,
                "from_account": src,
                "to_account": dst,
                "amount": round(float(amount), 2),
                "payment_format": fmt,
                "is_laundering": int(bool(pattern)),
                "pattern": pattern,
                "instance_id": instance,
            }
        )

    def rand_time(self, day_offset: float, hour_lo=8, hour_hi=22) -> datetime:
        hour = self.rng.uniform(hour_lo, hour_hi)
        return self.cfg.start + timedelta(days=float(day_offset), hours=float(hour))


def _acct(prefix: str, i: int) -> str:
    return f"{prefix}{i:05d}"


def generate(cfg: GeneratorConfig | None = None) -> pd.DataFrame:
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    b = _Builder(cfg, rng)

    retail = [_acct("AC", i) for i in range(cfg.n_retail)]
    merchants = [_acct("MR", i) for i in range(cfg.n_merchants)]
    employers = [_acct("EM", i) for i in range(cfg.n_employers)]
    utilities = [_acct("UT", i) for i in range(cfg.n_utilities)]
    suppliers = [_acct("SP", i) for i in range(10)]
    clients = [_acct("CL", i) for i in range(20)]

    # --- Ordinary activity -------------------------------------------------
    n_p2p = int(cfg.n_retail * cfg.p2p_per_day * cfg.days)
    src = rng.choice(retail, n_p2p)
    dst = rng.choice(retail, n_p2p)
    amounts = np.clip(rng.lognormal(mean=7.6, sigma=1.0, size=n_p2p), 50, 150_000)
    days = rng.uniform(0, cfg.days, n_p2p)
    for s, d, a, dy in zip(src, dst, amounts, days):
        if s != d:
            b.add(b.rand_time(int(dy)), s, d, a, rng.choice(FORMATS_NORMAL))

    # Shops: many small payments in (looks like fan-in), weekly supplier payments out.
    n_shop = int(cfg.n_retail * cfg.shop_per_day * cfg.days)
    shop_weights = rng.pareto(1.2, cfg.n_merchants) + 1
    shop_weights /= shop_weights.sum()
    src = rng.choice(retail, n_shop)
    dst = rng.choice(merchants, n_shop, p=shop_weights)
    amounts = np.clip(rng.lognormal(mean=6.2, sigma=0.9, size=n_shop), 20, 25_000)
    days = rng.uniform(0, cfg.days, n_shop)
    for s, d, a, dy in zip(src, dst, amounts, days):
        b.add(b.rand_time(int(dy), 9, 23), s, d, a, "UPI")
    for m, w in zip(merchants, shop_weights):
        weekly_revenue = w * n_shop / cfg.days * 7 * 600
        for week in range(cfg.days // 7):
            b.add(b.rand_time(week * 7 + 6), m, rng.choice(suppliers), weekly_revenue * rng.uniform(0.5, 0.7), "NEFT")

    # Employers: client payments in, monthly salaries out to a stable set of staff (looks like fan-out).
    for e in employers:
        staff = list(rng.choice(retail, rng.integers(10, 40), replace=False))
        salary = {p: round(rng.uniform(25_000, 120_000), -2) for p in staff}
        payroll = sum(salary.values())
        for month in range(cfg.days // 30 + 1):
            day0 = month * 30
            for _ in range(rng.integers(1, 3)):
                if day0 + 25 < cfg.days:
                    b.add(b.rand_time(day0 + rng.integers(0, 25)), rng.choice(clients), e,
                          payroll * rng.uniform(0.35, 0.6), "RTGS")
            # Main client usually pays just before payday, which is then paid straight out as salaries.
            if 0 < day0 < cfg.days and rng.random() < 0.7:
                b.add(b.rand_time(day0 - 1, 12, 18), rng.choice(clients), e, payroll * rng.uniform(0.9, 1.2), "RTGS")
            if day0 < cfg.days:
                for p in staff:
                    b.add(b.rand_time(day0, 9, 11), e, p, salary[p], "NEFT")

    # Bills
    for acc in retail:
        for month in range(cfg.days // 30):
            b.add(b.rand_time(month * 30 + rng.integers(3, 15)), acc, rng.choice(utilities),
                  rng.uniform(300, 4000), "UPI")

    # --- Innocent look-alikes (these trip the screening rules) -------------
    # Wholesale distributors: many shops pay in, most of it goes to the manufacturer within a day or two.
    manufacturers = [_acct("MF", i) for i in range(6)]
    for d in [_acct("DS", i) for i in range(cfg.n_distributors)]:
        mf = rng.choice(manufacturers)
        buyers = list(rng.choice(merchants + retail[:200], 25, replace=False))
        for day in range(0, cfg.days, int(rng.integers(3, 6))):
            total = 0.0
            for buyer in rng.choice(buyers, rng.integers(6, 15), replace=False):
                a = rng.uniform(15_000, 80_000)
                total += a
                b.add(b.rand_time(day, 9, 18), buyer, d, a, str(rng.choice(["NEFT", "IMPS", "UPI"])))
            b.add(b.rand_time(day + rng.uniform(0.5, 1.8), 10, 17), d, mf, total * rng.uniform(0.82, 0.95), "RTGS")

    # Investors: a bonus or property-sale receipt moved straight to a broker or FD account.
    brokers = [_acct("BR", i) for i in range(5)]
    for acc in rng.choice(retail, cfg.n_investors, replace=False):
        for _ in range(rng.integers(1, 3)):
            day = rng.uniform(1, cfg.days - 3)
            a = rng.uniform(9.2e5, 9.99e5) if rng.random() < 0.4 else rng.uniform(1.2e5, 8e5)
            b.add(b.rand_time(day), rng.choice(clients + employers), acc, a, "NEFT")
            b.add(b.rand_time(day + rng.uniform(0.3, 1.8)), acc, rng.choice(brokers), a * rng.uniform(0.85, 1.0), "NEFT")

    # Home buyers: instalments to a builder just under ₹10 lakh, funded by a loan disbursal.
    builders = [_acct("BD", i) for i in range(4)]
    for acc in rng.choice(retail, cfg.n_home_buyers, replace=False):
        b.add(b.rand_time(rng.uniform(0, 5)), _acct("BK", 0), acc, rng.uniform(2.5e6, 4e6), "RTGS")
        for k in range(rng.integers(2, 4)):
            b.add(b.rand_time(10 + k * 15 + rng.uniform(0, 5)), acc, rng.choice(builders), rng.uniform(9.0e5, 9.9e5), "RTGS")

    # Families moving money between their own accounts over several weeks (a slow, innocent loop).
    for _ in range(cfg.n_family_loops):
        fam = list(rng.choice(retail, rng.integers(3, 5), replace=False))
        day = rng.uniform(0, 15)
        gap = (0.5, 2) if rng.random() < 0.5 else (7, 14)  # some families move money quickly for a purchase
        for i in range(len(fam)):
            day += rng.uniform(*gap)
            if day < cfg.days:
                b.add(b.rand_time(day), fam[i], fam[(i + 1) % len(fam)], rng.uniform(5e4, 3e5), "IMPS")

    # --- Laundering instances ---------------------------------------------
    mule_counter = iter(range(100_000))
    ext_counter = iter(range(100_000))

    def mule() -> str:
        if rng.random() < cfg.reuse_retail_as_mule:
            return str(rng.choice(retail))
        acc = _acct("MU", next(mule_counter))
        # A little cover activity so mules are not trivially empty accounts.
        for _ in range(rng.integers(2, 8)):
            b.add(b.rand_time(rng.uniform(0, cfg.days)), acc, str(rng.choice(retail)),
                  rng.lognormal(7, 0.8), "UPI")
        return acc

    def ext() -> str:
        return _acct("EX", next(ext_counter))

    def structured_amount() -> float:
        # Often kept just under the ₹10 lakh reporting threshold.
        if rng.random() < 0.5:
            return rng.uniform(9.0e5, 9.95e5)
        return rng.uniform(1.5e5, 8.5e5)

    def fmt() -> str:
        return str(rng.choice(FORMATS_LAUNDER))

    for pattern in PATTERNS:
        for k in range(cfg.instances_per_pattern):
            inst = f"{pattern}_{k:02d}"
            t0 = cfg.start + timedelta(days=float(rng.uniform(5, cfg.days - 6)), hours=float(rng.uniform(0, 24)))

            def step(hours_lo=0.5, hours_hi=8.0):
                return timedelta(hours=float(rng.uniform(hours_lo, hours_hi)))

            if pattern == "fan_out":
                s, outs = mule(), [mule() for _ in range(rng.integers(6, 12))]
                total = sum(structured_amount() for _ in outs)
                b.add(t0, ext(), s, total * rng.uniform(1.01, 1.03), "RTGS", pattern, inst)
                t = t0
                for o in outs:
                    t += step(0.1, 3)
                    b.add(t, s, o, total / len(outs) * rng.uniform(0.95, 1.0), fmt(), pattern, inst)

            elif pattern == "fan_in":
                c, ins = mule(), [mule() for _ in range(rng.integers(6, 12))]
                t, total = t0, 0.0
                for i in ins:
                    t += step(0.2, 4)
                    a = structured_amount()
                    total += a
                    b.add(t - step(1, 6), ext(), i, a * rng.uniform(1.0, 1.02), "IMPS", pattern, inst)
                    b.add(t, i, c, a, fmt(), pattern, inst)
                b.add(t + step(2, 12), c, ext(), total * rng.uniform(0.96, 0.99), "RTGS", pattern, inst)

            elif pattern == "cycle":
                ring = [mule() for _ in range(rng.integers(3, 7))]
                a, t = structured_amount(), t0
                b.add(t - step(1, 5), ext(), ring[0], a * 1.01, "RTGS", pattern, inst)
                for i in range(len(ring)):
                    t += step(1, 10)
                    a *= rng.uniform(0.97, 0.995)
                    b.add(t, ring[i], ring[(i + 1) % len(ring)], a, fmt(), pattern, inst)

            elif pattern == "scatter_gather":
                s, d = mule(), mule()
                mids = [mule() for _ in range(rng.integers(4, 9))]
                total = sum(structured_amount() for _ in mids)
                b.add(t0, ext(), s, total * 1.01, "RTGS", pattern, inst)
                t = t0
                for m in mids:
                    t += step(0.1, 2)
                    a = total / len(mids) * rng.uniform(0.97, 1.0)
                    b.add(t, s, m, a, fmt(), pattern, inst)
                    b.add(t + step(1, 12), m, d, a * rng.uniform(0.97, 0.995), fmt(), pattern, inst)

            elif pattern == "gather_scatter":
                h = mule()
                ins = [mule() for _ in range(rng.integers(4, 9))]
                outs = [mule() for _ in range(rng.integers(4, 9))]
                t, total = t0, 0.0
                for i in ins:
                    t += step(0.2, 3)
                    a = structured_amount()
                    total += a
                    b.add(t, i, h, a, fmt(), pattern, inst)
                for o in outs:
                    t += step(0.2, 3)
                    b.add(t, h, o, total / len(outs) * rng.uniform(0.95, 0.99), fmt(), pattern, inst)

            elif pattern == "layering_chain":
                chain = [mule() for _ in range(rng.integers(4, 8))]
                a, t = structured_amount() * rng.uniform(1.5, 4), t0
                b.add(t, ext(), chain[0], a, "RTGS", pattern, inst)
                for i in range(len(chain) - 1):
                    t += step(0.5, 6)
                    a *= rng.uniform(0.98, 0.995)
                    b.add(t, chain[i], chain[i + 1], a, fmt(), pattern, inst)
                b.add(t + step(1, 6), chain[-1], ext(), a * 0.99, "RTGS", pattern, inst)

    df = pd.DataFrame(b.rows).sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(df))])

    # Replace role-revealing prefixes (EM, MU, DS...) with neutral IDs so nothing leaks to the agents.
    accounts = sorted(set(df.from_account) | set(df.to_account))
    codes = rng.choice(np.arange(100_000, 1_000_000), len(accounts), replace=False)
    mapping = {acc: f"AC{code}" for acc, code in zip(accounts, codes)}
    df["from_account"] = df.from_account.map(mapping)
    df["to_account"] = df.to_account.map(mapping)
    # Kept outside the transaction file; used only to explain false alarms during evaluation.
    df.attrs["roles"] = {mapping[a]: ROLE_NAMES.get(a[:2], "other") for a in accounts}
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic transactions with planted laundering.")
    parser.add_argument("--out", default="data/demo_transactions.csv.gz")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    df = generate(GeneratorConfig(seed=args.seed))
    df.to_csv(args.out, index=False)
    roles_path = args.out.replace("transactions", "account_roles").replace(".csv.gz", ".csv")
    pd.Series(df.attrs["roles"], name="role").rename_axis("account").to_csv(roles_path)
    laundering = df[df.is_laundering == 1]
    print(f"Wrote {len(df):,} transactions to {args.out}")
    print(f"Laundering transactions: {len(laundering):,} across {laundering.instance_id.nunique()} instances")


if __name__ == "__main__":
    main()
