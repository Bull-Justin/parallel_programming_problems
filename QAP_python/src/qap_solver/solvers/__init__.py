"""Solver registry and factory function."""
from __future__ import annotations

from qap_solver.solvers.base import QAPSolver
from qap_solver.solvers.brute_force import BruteForceSolver
from qap_solver.solvers.dynamic_programming import DPSolver
from qap_solver.solvers.genetic_algorithm import GeneticAlgorithmSolver
from qap_solver.solvers.parallel import ParallelBruteForceSolver, ParallelMultiStartSolver
from qap_solver.solvers.simulated_annealing import SimulatedAnnealingSolver

SOLVER_REGISTRY: dict[str, type[QAPSolver]] = {
    "brute_force": BruteForceSolver,
    "dp": DPSolver,
    "sa": SimulatedAnnealingSolver,
    "ga": GeneticAlgorithmSolver,
    "parallel_bf": ParallelBruteForceSolver,
}


def get_solver(name: str, **kwargs) -> QAPSolver:
    """Instantiate a solver by its short name."""
    if name not in SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver '{name}'. Available: {list(SOLVER_REGISTRY.keys())}"
        )
    return SOLVER_REGISTRY[name](**kwargs)


__all__ = [
    "QAPSolver",
    "BruteForceSolver",
    "DPSolver",
    "SimulatedAnnealingSolver",
    "GeneticAlgorithmSolver",
    "ParallelBruteForceSolver",
    "ParallelMultiStartSolver",
    "get_solver",
    "SOLVER_REGISTRY",
]
