"""Shared test fixtures."""
from __future__ import annotations

import pytest
import numpy as np

from qap_solver.problem import QAPInstance


@pytest.fixture
def tiny_instance() -> QAPInstance:
    """A trivial 4x4 QAP instance for fast testing."""
    flow = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ],
        dtype=np.int64,
    )
    distance = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ],
        dtype=np.int64,
    )
    return QAPInstance(n=4, flow=flow, distance=distance, name="tiny4")


@pytest.fixture
def tiny_optimum(tiny_instance: QAPInstance) -> int:
    """Brute-force optimal cost for the tiny instance."""
    import itertools
    from qap_solver.evaluate import evaluate

    best = float("inf")
    for perm in itertools.permutations(range(tiny_instance.n)):
        cost = evaluate(tiny_instance, perm)
        if cost < best:
            best = cost
    return int(best)


@pytest.fixture
def nug12_instance() -> QAPInstance:
    """Load the bundled nug12 benchmark."""
    from qap_solver.benchmarks import load_benchmark
    return load_benchmark("nug12")
