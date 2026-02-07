"""Tests for objective function evaluation."""
from __future__ import annotations

import numpy as np

from qap_solver.evaluate import evaluate, evaluate_swap_delta
from qap_solver.problem import QAPInstance


def test_evaluate_identity(tiny_instance):
    """Identity permutation should give a valid cost."""
    perm = tuple(range(tiny_instance.n))
    cost = evaluate(tiny_instance, perm)
    assert isinstance(cost, int)
    assert cost >= 0


def test_evaluate_known_optimum(nug12_instance):
    """Known optimal permutation for nug12 should give cost 578."""
    from qap_solver.io import load_qaplib_solution
    from qap_solver.benchmarks import get_benchmark_path

    path = get_benchmark_path("nug12")
    sln_path = path.with_suffix(".sln")
    sol = load_qaplib_solution(sln_path)

    cost = evaluate(nug12_instance, sol.permutation)
    assert cost == 578


def test_swap_delta_consistency(tiny_instance):
    """Swap delta should equal the difference of full evaluations."""
    n = tiny_instance.n
    perm = np.array([2, 0, 3, 1], dtype=np.intp)

    original_cost = evaluate(tiny_instance, perm)

    for i in range(n):
        for j in range(i + 1, n):
            delta = evaluate_swap_delta(tiny_instance, perm, i, j)

            swapped = perm.copy()
            swapped[i], swapped[j] = swapped[j], swapped[i]
            swapped_cost = evaluate(tiny_instance, swapped)

            assert delta == swapped_cost - original_cost, (
                f"Swap ({i},{j}): delta={delta}, "
                f"actual={swapped_cost - original_cost}"
            )
