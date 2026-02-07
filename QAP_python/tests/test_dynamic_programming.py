"""Tests for the dynamic programming solver."""
from __future__ import annotations

import numpy as np
import pytest

from qap_solver.problem import QAPInstance
from qap_solver.solvers.brute_force import BruteForceSolver
from qap_solver.solvers.dynamic_programming import DPSolver


def test_finds_optimum(tiny_instance, tiny_optimum):
    solver = DPSolver()
    solution = solver.solve(tiny_instance)
    assert solution.cost == tiny_optimum
    assert solution.solver_name == "dynamic_programming"


def test_agrees_with_brute_force(tiny_instance):
    """DP and brute-force must agree on the optimal cost."""
    bf = BruteForceSolver().solve(tiny_instance)
    dp = DPSolver().solve(tiny_instance)
    assert dp.cost == bf.cost


def test_agrees_with_brute_force_random_6x6():
    """Cross-validate DP vs brute-force on a random 6x6 instance."""
    rng = np.random.default_rng(42)
    n = 6
    flow = rng.integers(0, 50, size=(n, n), dtype=np.int64)
    dist = rng.integers(0, 50, size=(n, n), dtype=np.int64)
    np.fill_diagonal(flow, 0)
    np.fill_diagonal(dist, 0)
    inst = QAPInstance(n=n, flow=flow, distance=dist, name="random6")

    bf = BruteForceSolver().solve(inst)
    dp = DPSolver().solve(inst)
    assert dp.cost == bf.cost


def test_agrees_with_brute_force_random_8x8():
    """Cross-validate DP vs brute-force on a random 8x8 instance."""
    rng = np.random.default_rng(123)
    n = 8
    flow = rng.integers(0, 30, size=(n, n), dtype=np.int64)
    dist = rng.integers(0, 30, size=(n, n), dtype=np.int64)
    np.fill_diagonal(flow, 0)
    np.fill_diagonal(dist, 0)
    inst = QAPInstance(n=n, flow=flow, distance=dist, name="random8")

    bf = BruteForceSolver(max_n=8).solve(inst)
    dp = DPSolver().solve(inst)
    assert dp.cost == bf.cost


def test_rejects_large_instance():
    n = 20
    inst = QAPInstance(
        n=n,
        flow=np.zeros((n, n), dtype=np.int64),
        distance=np.zeros((n, n), dtype=np.int64),
    )
    solver = DPSolver(max_n=15)
    with pytest.raises(ValueError, match="exceeds DP limit"):
        solver.solve(inst)
