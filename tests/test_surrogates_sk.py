import numpy as np
from pyGPGO.surrogates.RandomForest import RandomForest, ExtraForest
from pyGPGO.surrogates.BoostedTrees import BoostedTrees


def f(x):
    return -((6 * x - 2) ** 2 * np.sin(12 * x - 4))


def test_rf():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 5, 20)[:, np.newaxis]
    y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

    rf = RandomForest()
    rf.fit(X, y)


def test_ef():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 5, 20)[:, np.newaxis]
    y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

    ef = ExtraForest()
    ef.fit(X, y)


def test_bt():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 5, 20)[:, np.newaxis]
    y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

    bt = BoostedTrees()
    bt.fit(X, y)


if __name__ == '__main__':
    test_rf()
    test_ef()
    test_bt()
