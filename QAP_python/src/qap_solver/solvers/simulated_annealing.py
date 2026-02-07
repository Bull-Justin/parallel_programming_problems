"""Simulated Annealing solver for QAP."""
from __future__ import annotations

import math
import random

import numpy as np

from qap_solver.evaluate import evaluate, evaluate_swap_delta
from qap_solver.problem import QAPInstance, QAPSolution
from qap_solver.solvers.base import QAPSolver


class SimulatedAnnealingSolver(QAPSolver):
    """Simulated annealing with pairwise swap perturbation."""

    @property
    def name(self) -> str:
        return "simulated_annealing"

    def __init__(
        self,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.9995,
        min_temp: float = 1e-6,
        max_iterations: int = 1_000_000,
        seed: int | None = None,
    ) -> None:
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.seed = seed

    def _solve_impl(self, instance: QAPInstance) -> QAPSolution:
        rng = random.Random(self.seed)
        n = instance.n

        current_perm = np.array(rng.sample(range(n), n), dtype=np.intp)
        current_cost = evaluate(instance, current_perm)

        best_perm = current_perm.copy()
        best_cost = current_cost

        temp = self.initial_temp
        iterations = 0
        accepted = 0

        while temp > self.min_temp and iterations < self.max_iterations:
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1

            delta = evaluate_swap_delta(instance, current_perm, i, j)

            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                current_cost += delta
                accepted += 1

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_perm = current_perm.copy()

            temp *= self.cooling_rate
            iterations += 1

        return QAPSolution(
            permutation=tuple(int(x) for x in best_perm),
            cost=int(best_cost),
            metadata={
                "iterations": iterations,
                "accepted_moves": accepted,
                "final_temperature": temp,
            },
        )
