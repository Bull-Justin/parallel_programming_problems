"""Tests for the brute-force solver."""
from __future__ import annotations

import pytest

from qap_solver.solvers.brute_force import BruteForceSolver


def test_finds_optimum(tiny_instance, tiny_optimum):
    solver = BruteForceSolver()
    solution = solver.solve(tiny_instance)
    assert solution.cost == tiny_optimum
    assert solution.solver_name == "brute_force"
    assert "elapsed_seconds" in solution.metadata


def test_rejects_large_instance():
    import numpy as np
    from qap_solver.problem import QAPInstance

    n = 15
    inst = QAPInstance(
        n=n,
        flow=np.zeros((n, n), dtype=np.int64),
        distance=np.zeros((n, n), dtype=np.int64),
    )
    solver = BruteForceSolver(max_n=12)
    with pytest.raises(ValueError, match="exceeds brute-force limit"):
        solver.solve(inst)


def test_metadata(tiny_instance):
    solver = BruteForceSolver()
    solution = solver.solve(tiny_instance)
    assert solution.metadata["permutations_evaluated"] == 24  # 4!
