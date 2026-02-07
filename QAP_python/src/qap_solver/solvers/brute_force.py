"""Brute-force exact solver — enumerates all n! permutations."""
from __future__ import annotations

import itertools

from qap_solver.evaluate import evaluate
from qap_solver.problem import QAPInstance, QAPSolution
from qap_solver.solvers.base import QAPSolver


class BruteForceSolver(QAPSolver):
    """Exhaustive enumeration of all n! permutations. Feasible for n <= ~12."""

    @property
    def name(self) -> str:
        return "brute_force"

    def __init__(self, max_n: int = 12) -> None:
        self.max_n = max_n

    def _solve_impl(self, instance: QAPInstance) -> QAPSolution:
        if instance.n > self.max_n:
            raise ValueError(
                f"Instance size {instance.n} exceeds brute-force limit {self.max_n}. "
                f"Use a heuristic or the parallel brute-force solver."
            )

        best_cost = float("inf")
        best_perm: tuple[int, ...] = ()
        count = 0

        for perm in itertools.permutations(range(instance.n)):
            cost = evaluate(instance, perm)
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
            count += 1

        return QAPSolution(
            permutation=best_perm,
            cost=int(best_cost),
            metadata={"permutations_evaluated": count},
        )
