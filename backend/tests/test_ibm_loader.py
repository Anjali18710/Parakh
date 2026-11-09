"""The IBM converter keeps whole schemes, converts money to rupees and hides the original ids."""
import re

import pandas as pd

from parakh.data.ibm_loader import INR_PER_UNIT, convert

COLS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received", "Receiving Currency",
        "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]


def row(ts, fb, fa, tb, ta, amount, currency="US Dollar", fmt="ACH", bad=0):
    return [ts, fb, fa, tb, ta, amount, currency, amount, currency, fmt, bad]


def test_converter(tmp_path):
    rows = [
        # a fan-out scheme: M pays three mules
        row("2022/09/01 00:10", "10", "M", "20", "X1", 1000.0, bad=1),
        row("2022/09/01 00:20", "10", "M", "21", "X2", 900.0, "Euro", bad=1),
        row("2022/09/01 00:30", "10", "M", "22", "X3", 800.0, "Yuan", bad=1),
        # a laundering payment from a scheme we will not pick, by a mule of the chosen one
        row("2022/09/02 09:00", "20", "X1", "40", "Z", 500.0, bad=1),
        # a self-transfer, which carries no flow
        row("2022/09/01 01:00", "10", "M", "10", "M", 50.0, fmt="Reinvestment"),
        # an innocent quiet account with three payments
        row("2022/09/01 02:00", "30", "Q", "31", "Shop", 10.0),
        row("2022/09/02 02:00", "30", "Q", "31", "Shop", 12.0),
        row("2022/09/03 02:00", "30", "Q", "31", "Shop", 11.0),
    ]
    trans = tmp_path / "t.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(trans, index=False)
    patterns = tmp_path / "p.txt"
    patterns.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 3-degree Fan-Out\n"
        "2022/09/01 00:10,10,M,20,X1,1000.00,US Dollar,1000.00,US Dollar,ACH,1\n"
        "2022/09/01 00:20,10,M,21,X2,900.00,Euro,900.00,Euro,ACH,1\n"
        "2022/09/01 00:30,10,M,22,X3,800.00,Yuan,800.00,Yuan,ACH,1\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n"
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops\n"
        "2022/09/05 00:10,50,P,51,R,7.00,US Dollar,7.00,US Dollar,ACH,1\n"
        "END LAUNDERING ATTEMPT - CYCLE\n")

    df = convert(trans, patterns, per_type=1, quiet=1, busy=0, seed=1)

    assert list(df.columns[:3]) == ["txn_id", "timestamp", "from_account"]
    accounts = set(df.from_account) | set(df.to_account)
    assert all(re.fullmatch(r"AC\d{6}", a) for a in accounts)  # original ids never shown
    assert "Reinvestment" not in set(df.payment_format)  # self-transfer dropped
    scheme = df[df.pattern == "fan_out"]
    assert len(scheme) == 3 and scheme.instance_id.nunique() == 1  # the whole scheme, under one id
    assert sorted(scheme.amount) == sorted([1000 * INR_PER_UNIT["US Dollar"], 900 * INR_PER_UNIT["Euro"],
                                            800 * INR_PER_UNIT["Yuan"]])  # in rupees
    stray = df[(df.is_laundering == 1) & (df.pattern != "fan_out")]
    assert list(stray.pattern) == ["other_scheme"]  # still counted as laundering
    assert (df.is_laundering == 0).sum() == 3  # the quiet innocent account came along
