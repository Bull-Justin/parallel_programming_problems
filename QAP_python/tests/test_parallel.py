"""Tests for the parallel solvers."""
from __future__ import annotations

from qap_solver.solvers.parallel import ParallelBruteForceSolver, ParallelMultiStartSolver
from qap_solver.solvers.simulated_annealing import SimulatedAnnealingSolver


def test_parallel_bf_finds_optimum(tiny_instance, tiny_optimum):
    solver = ParallelBruteForceSolver(num_workers=2)
    solution = solver.solve(tiny_instance)
    assert solution.cost == tiny_optimum
    assert solution.solver_name == "parallel_brute_force"


def test_parallel_multistart_sa(tiny_instance, tiny_optimum):
    solver = ParallelMultiStartSolver(
        base_solver_class=SimulatedAnnealingSolver,
        base_solver_kwargs={"max_iterations": 10_000},
        num_starts=4,
        num_workers=2,
    )
    solution = solver.solve(tiny_instance)
    assert solution.cost == tiny_optimum
