"""Objective function evaluation for QAP solutions."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qap_solver.problem import QAPInstance


def evaluate(
    instance: QAPInstance,
    permutation: tuple[int, ...] | list[int] | npt.NDArray,
) -> int:
    """cost = sum flow[i][j] * distance[perm[i]][perm[j]]"""
    perm = np.asarray(permutation, dtype=np.intp)
    return int(np.sum(instance.flow * instance.distance[np.ix_(perm, perm)]))


def evaluate_swap_delta(
    instance: QAPInstance,
    permutation: npt.NDArray[np.intp],
    i: int,
    j: int,
) -> int:
    """O(n) Burkard-Rendl swap delta: cost_after - cost_before."""
    n = instance.n
    flow = instance.flow
    dist = instance.distance
    pi = permutation

    delta = (
        (flow[i][i] - flow[j][j]) * (dist[pi[j]][pi[j]] - dist[pi[i]][pi[i]])
        + (flow[i][j] - flow[j][i]) * (dist[pi[j]][pi[i]] - dist[pi[i]][pi[j]])
    )

    for k in range(n):
        if k == i or k == j:
            continue
        delta += (
            (flow[i][k] - flow[j][k]) * (dist[pi[j]][pi[k]] - dist[pi[i]][pi[k]])
            + (flow[k][i] - flow[k][j]) * (dist[pi[k]][pi[j]] - dist[pi[k]][pi[i]])
        )

    return int(delta)
