"""QAP instance and solution data structures."""
from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True, slots=True)
class QAPInstance:

    n: int
    flow: npt.NDArray[np.int64]
    distance: npt.NDArray[np.int64]
    name: str = ""
    known_optimum: int | None = None

    def __post_init__(self) -> None:
        if self.flow.shape != (self.n, self.n):
            raise ValueError(
                f"Flow matrix shape {self.flow.shape} does not match n={self.n}"
            )
        if self.distance.shape != (self.n, self.n):
            raise ValueError(
                f"Distance matrix shape {self.distance.shape} does not match n={self.n}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class QAPSolution:
    """permutation[i] = j means facility i is assigned to location j (0-indexed)."""

    permutation: tuple[int, ...]
    cost: int
    solver_name: str = ""
    metadata: dict = dataclasses.field(default_factory=dict)
