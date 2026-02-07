"""Parallel solvers using ProcessPoolExecutor."""
from __future__ import annotations

import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from qap_solver.evaluate import evaluate
from qap_solver.problem import QAPInstance, QAPSolution
from qap_solver.solvers.base import QAPSolver


class ParallelBruteForceSolver(QAPSolver):
    """Partitions permutation space by first element across CPU cores."""

    @property
    def name(self) -> str:
        return "parallel_brute_force"

    def __init__(self, max_n: int = 13, num_workers: int | None = None) -> None:
        self.max_n = max_n
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)

    def _solve_impl(self, instance: QAPInstance) -> QAPSolution:
        n = instance.n
        if n > self.max_n:
            raise ValueError(f"n={n} exceeds parallel brute-force limit {self.max_n}")

        flow = instance.flow
        distance = instance.distance

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_partition, flow, distance, n, first_elem
                ): first_elem
                for first_elem in range(n)
            }

            best_cost = float("inf")
            best_perm: tuple[int, ...] = ()
            for future in as_completed(futures):
                perm, cost = future.result()
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm

        return QAPSolution(
            permutation=best_perm,
            cost=int(best_cost),
            metadata={"num_workers": self.num_workers},
        )


def _evaluate_partition(
    flow: np.ndarray,
    distance: np.ndarray,
    n: int,
    first_elem: int,
) -> tuple[tuple[int, ...], int]:
    """Worker: evaluate all permutations with a fixed first element."""
    remaining = [x for x in range(n) if x != first_elem]
    best_cost = float("inf")
    best_perm: tuple[int, ...] = ()

    for rest in itertools.permutations(remaining):
        perm = (first_elem,) + rest
        cost = int(np.sum(flow * distance[np.ix_(perm, perm)]))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    return best_perm, int(best_cost)


class ParallelMultiStartSolver(QAPSolver):
    """Run multiple solver instances with different seeds in parallel."""

    @property
    def name(self) -> str:
        return f"parallel_multistart_{self._base_name}"

    def __init__(
        self,
        base_solver_class: type[QAPSolver],
        base_solver_kwargs: dict | None = None,
        num_starts: int = 8,
        num_workers: int | None = None,
    ) -> None:
        self.base_solver_class = base_solver_class
        self.base_solver_kwargs = base_solver_kwargs or {}
        self.num_starts = num_starts
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self._base_name = base_solver_class.__name__

    def _solve_impl(self, instance: QAPInstance) -> QAPSolution:
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(self.num_starts):
                kwargs = {**self.base_solver_kwargs, "seed": i * 12345 + 42}
                futures.append(
                    executor.submit(
                        _run_solver, self.base_solver_class, kwargs, instance
                    )
                )
            results = [f.result() for f in futures]

        best = min(results, key=lambda s: s.cost)
        return QAPSolution(
            permutation=best.permutation,
            cost=best.cost,
            metadata={**best.metadata, "num_starts": self.num_starts},
        )


def _run_solver(
    solver_class: type[QAPSolver],
    solver_kwargs: dict,
    instance: QAPInstance,
) -> QAPSolution:
    """Worker for multiprocessing — must be module-level for pickling."""
    solver = solver_class(**solver_kwargs)
    return solver._solve_impl(instance)
