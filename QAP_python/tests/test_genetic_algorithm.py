"""Tests for the genetic algorithm solver."""
from __future__ import annotations

from qap_solver.solvers.genetic_algorithm import GeneticAlgorithmSolver


def test_finds_good_solution(tiny_instance, tiny_optimum):
    solver = GeneticAlgorithmSolver(
        population_size=50,
        generations=100,
        seed=42,
    )
    solution = solver.solve(tiny_instance)
    assert solution.cost == tiny_optimum


def test_nug12_quality(nug12_instance):
    """GA should get within 15% of optimum on nug12."""
    solver = GeneticAlgorithmSolver(
        population_size=100,
        generations=300,
        seed=42,
    )
    solution = solver.solve(nug12_instance)
    gap = (solution.cost - 578) / 578 * 100
    assert gap < 15.0, f"GA gap {gap:.1f}% is too large"


def test_memetic_variant(tiny_instance, tiny_optimum):
    """GA with local search should also find the optimum on tiny."""
    solver = GeneticAlgorithmSolver(
        population_size=30,
        generations=50,
        local_search_iters=5,
        seed=42,
    )
    solution = solver.solve(tiny_instance)
    assert solution.cost == tiny_optimum


def test_reproducibility(tiny_instance):
    s1 = GeneticAlgorithmSolver(seed=99).solve(tiny_instance)
    s2 = GeneticAlgorithmSolver(seed=99).solve(tiny_instance)
    assert s1.cost == s2.cost
    assert s1.permutation == s2.permutation
