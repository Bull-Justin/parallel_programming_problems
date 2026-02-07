"""Tests for the simulated annealing solver."""
from __future__ import annotations

from qap_solver.solvers.simulated_annealing import SimulatedAnnealingSolver


def test_finds_good_solution(tiny_instance, tiny_optimum):
    solver = SimulatedAnnealingSolver(seed=42)
    solution = solver.solve(tiny_instance)
    # For a tiny instance, SA should easily find the optimum
    assert solution.cost == tiny_optimum


def test_nug12_quality(nug12_instance):
    """SA should get within 10% of optimum on nug12."""
    solver = SimulatedAnnealingSolver(
        initial_temp=200.0,
        cooling_rate=0.9999,
        max_iterations=500_000,
        seed=42,
    )
    solution = solver.solve(nug12_instance)
    gap = (solution.cost - 578) / 578 * 100
    assert gap < 10.0, f"SA gap {gap:.1f}% is too large"


def test_reproducibility(tiny_instance):
    """Same seed should produce same result."""
    s1 = SimulatedAnnealingSolver(seed=123).solve(tiny_instance)
    s2 = SimulatedAnnealingSolver(seed=123).solve(tiny_instance)
    assert s1.cost == s2.cost
    assert s1.permutation == s2.permutation


def test_metadata(tiny_instance):
    solver = SimulatedAnnealingSolver(seed=42, max_iterations=1000)
    solution = solver.solve(tiny_instance)
    assert "iterations" in solution.metadata
    assert "accepted_moves" in solution.metadata
    assert "elapsed_seconds" in solution.metadata
