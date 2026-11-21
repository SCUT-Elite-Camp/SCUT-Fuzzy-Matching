import numpy as np

from clustering.kmeans_cosine import (
    run_cosine_kmeans,
    run_legacy_cosine_kmeans,
)


def test_emeans_contract_and_determinism():
    rng = np.random.RandomState(5)
    x = np.vstack((rng.normal(loc=-2, size=(40, 4)), rng.normal(loc=2, size=(40, 4))))
    c1, a1 = run_cosine_kmeans(x, 4, iterations=3, random_state=7)
    c2, a2 = run_cosine_kmeans(x, 4, iterations=3, random_state=7)
    assert c1.shape == (4, 4) and a1.shape == (80,)
    assert np.array_equal(a1, a2) and np.array_equal(c1, c2)
    assert np.allclose(np.linalg.norm(c1, axis=1), 1)
    assert a1.min() >= 0 and a1.max() < 4
    assert run_legacy_cosine_kmeans(x, 4, iterations=3)[0].shape == c1.shape


def test_small_problem_hyperparameters_and_empty_input():
    try:
        run_cosine_kmeans(np.empty((0, 2)), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("empty input must fail")


def test_control_switch(monkeypatch):
    x = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    monkeypatch.setenv("SCUT_CLUSTER_ALGORITHM", "kmeans")
    actual = run_cosine_kmeans(x, 2, iterations=2, random_state=1)
    expected = run_legacy_cosine_kmeans(x, 2, iterations=2, random_state=1)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
