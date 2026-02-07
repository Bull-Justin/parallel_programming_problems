"""Tests for QAPLIB I/O."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from qap_solver.io import load_qaplib, load_qaplib_solution, save_qaplib
from qap_solver.problem import QAPInstance, QAPSolution


def test_load_nug12():
    from qap_solver.benchmarks import get_benchmark_path

    path = get_benchmark_path("nug12")
    assert path is not None
    inst = load_qaplib(path)
    assert inst.n == 12
    assert inst.flow.shape == (12, 12)
    assert inst.distance.shape == (12, 12)
    assert inst.known_optimum == 578


def test_load_sln():
    from qap_solver.benchmarks import get_benchmark_path

    path = get_benchmark_path("nug12")
    assert path is not None
    sln_path = path.with_suffix(".sln")
    sol = load_qaplib_solution(sln_path)
    assert sol is not None
    assert sol.cost == 578
    assert len(sol.permutation) == 12
    # Should be 0-indexed
    assert min(sol.permutation) == 0
    assert max(sol.permutation) == 11


def test_round_trip():
    flow = np.array([[0, 5], [5, 0]], dtype=np.int64)
    dist = np.array([[0, 3], [3, 0]], dtype=np.int64)
    inst = QAPInstance(n=2, flow=flow, distance=dist, name="test2")
    sol = QAPSolution(permutation=(1, 0), cost=30)

    with tempfile.TemporaryDirectory() as tmpdir:
        dat_path = Path(tmpdir) / "test.dat"
        sln_path = Path(tmpdir) / "test.sln"
        save_qaplib(inst, dat_path, solution=sol, sln_path=sln_path)

        loaded = load_qaplib(dat_path, sln_path)
        assert loaded.n == 2
        np.testing.assert_array_equal(loaded.flow, flow)
        np.testing.assert_array_equal(loaded.distance, dist)
        assert loaded.known_optimum == 30

        loaded_sol = load_qaplib_solution(sln_path)
        assert loaded_sol is not None
        assert loaded_sol.permutation == (1, 0)
        assert loaded_sol.cost == 30
