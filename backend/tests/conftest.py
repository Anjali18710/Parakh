import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from parakh.data.generator import GeneratorConfig, generate  # noqa: E402
from parakh.data.store import TransactionStore  # noqa: E402


@pytest.fixture(scope="session")
def small_df():
    cfg = GeneratorConfig(seed=11, days=30, n_retail=300, n_merchants=8, n_employers=4, instances_per_pattern=2,
                          n_distributors=4, n_investors=15, n_home_buyers=5, n_family_loops=4)
    return generate(cfg)


@pytest.fixture(scope="session")
def store(small_df):
    return TransactionStore(small_df)


@pytest.fixture(scope="session")
def data_file(small_df, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "test_transactions.csv.gz"
    small_df.to_csv(path, index=False)
    return str(path)


def first_txn(store, pattern):
    """First laundering transaction of a pattern (uses ground truth, test-only)."""
    lab = store.labels
    tid = lab[lab.pattern == pattern].index[0]
    return store.by_id.loc[tid]
