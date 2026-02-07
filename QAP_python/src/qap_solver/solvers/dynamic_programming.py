"""Dynamic Programming solver for QAP.

Unlike TSP, QAP does not have optimal substructure over bitmask-only states
because every facility interacts with every other — the full assignment
order matters. This solver builds assignments level-by-level using partial
permutation tuples as state keys, with greedy upper-bound pruning.

O(n! * n) worst case, feasible for n <= ~15 with pruning.
"""
from __future__ import annotations

import numpy as np

from qap_solver.evaluate import evaluate
from qap_solver.problem import QAPInstance, QAPSolution
from qap_solver.solvers.base import QAPSolver


class DPSolver(QAPSolver):
    """Exact solver using level-by-level dynamic programming with pruning."""

    @property
    def name(self) -> str:
        return "dynamic_programming"

    def __init__(self, max_n: int = 15) -> None:
        self.max_n = max_n

    def _solve_impl(self, instance: QAPInstance) -> QAPSolution:
        n = instance.n
        if n > self.max_n:
            raise ValueError(f"n={n} exceeds DP limit {self.max_n}")

        flow = instance.flow
        dist = instance.distance

        upper_bound = self._greedy_upper_bound(instance)
        states_explored = 0

        current_level: dict[tuple[int, ...], int] = {(): 0}

        for k in range(n):
            next_level: dict[tuple[int, ...], int] = {}

            for assignment, cost in current_level.items():
                if cost >= upper_bound:
                    continue

                assigned = set(assignment)
                for f in range(n):
                    if f in assigned:
                        continue

                    inc = 0
                    for loc in range(k):
                        g = assignment[loc]
                        inc += (
                            flow[f][g] * dist[k][loc]
                            + flow[g][f] * dist[loc][k]
                        )

                    new_cost = cost + inc
                    if new_cost >= upper_bound:
                        continue

                    new_assignment = assignment + (f,)
                    if (
                        new_assignment not in next_level
                        or new_cost < next_level[new_assignment]
                    ):
                        next_level[new_assignment] = new_cost

                    if k == n - 1 and new_cost < upper_bound:
                        upper_bound = new_cost

                states_explored += 1

            current_level = next_level

        best_assignment = min(current_level, key=lambda a: current_level[a])
        best_cost = current_level[best_assignment]

        # assignment[loc] = facility -> perm[facility] = location
        result_perm = [0] * n
        for loc, fac in enumerate(best_assignment):
            result_perm[fac] = loc

        return QAPSolution(
            permutation=tuple(result_perm),
            cost=int(best_cost),
            metadata={"states_explored": states_explored},
        )

    @staticmethod
    def _greedy_upper_bound(instance: QAPInstance) -> int:
        n = instance.n
        flow = instance.flow
        dist = instance.distance

        assigned: list[int] = []
        remaining = set(range(n))

        for k in range(n):
            best_f = -1
            best_inc = float("inf")

            for f in remaining:
                inc = 0
                for loc in range(k):
                    g = assigned[loc]
                    inc += (
                        flow[f][g] * dist[k][loc]
                        + flow[g][f] * dist[loc][k]
                    )
                if inc < best_inc:
                    best_inc = inc
                    best_f = f

            assigned.append(best_f)
            remaining.remove(best_f)

        perm = [0] * n
        for loc, fac in enumerate(assigned):
            perm[fac] = loc

        return int(evaluate(instance, tuple(perm)))
