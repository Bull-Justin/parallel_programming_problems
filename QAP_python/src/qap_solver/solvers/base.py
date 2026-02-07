"""Abstract base class defining the solver interface."""
from __future__ import annotations

import abc
import time

from qap_solver.problem import QAPInstance, QAPSolution


class QAPSolver(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def _solve_impl(self, instance: QAPInstance) -> QAPSolution: ...

    def solve(self, instance: QAPInstance) -> QAPSolution:
        start = time.perf_counter()
        solution = self._solve_impl(instance)
        elapsed = time.perf_counter() - start
        metadata = {**solution.metadata, "elapsed_seconds": elapsed}
        return QAPSolution(
            permutation=solution.permutation,
            cost=solution.cost,
            solver_name=self.name,
            metadata=metadata,
        )
