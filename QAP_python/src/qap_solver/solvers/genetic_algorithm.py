"""Genetic Algorithm solver for QAP."""
from __future__ import annotations

import random

import numpy as np

from qap_solver.evaluate import evaluate, evaluate_swap_delta
from qap_solver.problem import QAPInstance, QAPSolution
from qap_solver.solvers.base import QAPSolver


class GeneticAlgorithmSolver(QAPSolver):
    """Genetic algorithm with OX1 crossover and swap mutation."""

    @property
    def name(self) -> str:
        return "genetic_algorithm"

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 500,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_size: int = 5,
        elite_count: int = 2,
        local_search_iters: int = 0,
        seed: int | None = None,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.local_search_iters = local_search_iters
        self.seed = seed

    def _solve_impl(self, instance: QAPInstance) -> QAPSolution:
        rng = random.Random(self.seed)
        n = instance.n

        population: list[tuple[np.ndarray, int]] = []
        for _ in range(self.population_size):
            perm = np.array(rng.sample(range(n), n), dtype=np.intp)
            cost = evaluate(instance, perm)
            population.append((perm, cost))

        best_perm, best_cost = min(population, key=lambda x: x[1])
        best_perm = best_perm.copy()

        for _gen in range(self.generations):
            population.sort(key=lambda x: x[1])

            new_population: list[tuple[np.ndarray, int]] = []

            for i in range(min(self.elite_count, len(population))):
                new_population.append((population[i][0].copy(), population[i][1]))

            while len(new_population) < self.population_size:
                p1 = self._tournament_select(population, rng)
                p2 = self._tournament_select(population, rng)

                if rng.random() < self.crossover_rate:
                    child = self._order_crossover(p1[0], p2[0], n, rng)
                else:
                    child = p1[0].copy()

                if rng.random() < self.mutation_rate:
                    self._swap_mutation(child, n, rng)

                if self.local_search_iters > 0:
                    child = self._local_search(instance, child, n)

                cost = evaluate(instance, child)
                new_population.append((child, cost))

                if cost < best_cost:
                    best_cost = cost
                    best_perm = child.copy()

            population = new_population

        return QAPSolution(
            permutation=tuple(int(x) for x in best_perm),
            cost=int(best_cost),
            metadata={
                "generations": self.generations,
                "population_size": self.population_size,
            },
        )

    def _tournament_select(
        self,
        population: list[tuple[np.ndarray, int]],
        rng: random.Random,
    ) -> tuple[np.ndarray, int]:
        candidates = rng.sample(
            population, min(self.tournament_size, len(population))
        )
        return min(candidates, key=lambda x: x[1])

    @staticmethod
    def _order_crossover(
        parent1: np.ndarray,
        parent2: np.ndarray,
        n: int,
        rng: random.Random,
    ) -> np.ndarray:
        """Order Crossover (OX1): copy a random segment from parent1,
        fill remaining positions from parent2 in order."""
        start = rng.randrange(n)
        end = rng.randrange(start + 1, n + 1)

        child = np.full(n, -1, dtype=np.intp)
        child[start:end] = parent1[start:end]

        placed = set(child[start:end].tolist())
        fill_order = [x for x in parent2 if int(x) not in placed]

        idx = 0
        for pos in range(n):
            if child[pos] == -1:
                child[pos] = fill_order[idx]
                idx += 1

        return child

    @staticmethod
    def _swap_mutation(perm: np.ndarray, n: int, rng: random.Random) -> None:
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        perm[i], perm[j] = perm[j], perm[i]

    def _local_search(
        self,
        instance: QAPInstance,
        perm: np.ndarray,
        n: int,
    ) -> np.ndarray:
        perm = perm.copy()
        improved = True
        iters = 0

        while improved and iters < self.local_search_iters:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1

            for i in range(n - 1):
                for j in range(i + 1, n):
                    delta = evaluate_swap_delta(instance, perm, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j

            if best_delta < 0:
                perm[best_i], perm[best_j] = perm[best_j], perm[best_i]
                improved = True
            iters += 1

        return perm
