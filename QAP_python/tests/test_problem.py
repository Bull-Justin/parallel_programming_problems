"""Tests for QAPInstance and QAPSolution."""
from __future__ import annotations

import numpy as np
import pytest

from qap_solver.problem import QAPInstance, QAPSolution


def test_instance_creation():
    flow = np.zeros((3, 3), dtype=np.int64)
    dist = np.zeros((3, 3), dtype=np.int64)
    inst = QAPInstance(n=3, flow=flow, distance=dist)
    assert inst.n == 3
    assert inst.flow.shape == (3, 3)
    assert inst.distance.shape == (3, 3)


def test_instance_shape_mismatch():
    flow = np.zeros((3, 3), dtype=np.int64)
    dist = np.zeros((4, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="does not match"):
        QAPInstance(n=3, flow=flow, distance=dist)


def test_instance_immutable():
    flow = np.zeros((2, 2), dtype=np.int64)
    dist = np.zeros((2, 2), dtype=np.int64)
    inst = QAPInstance(n=2, flow=flow, distance=dist)
    with pytest.raises(AttributeError):
        inst.n = 5  # type: ignore[misc]


def test_solution_creation():
    sol = QAPSolution(permutation=(2, 0, 1), cost=42, solver_name="test")
    assert sol.permutation == (2, 0, 1)
    assert sol.cost == 42
    assert sol.solver_name == "test"
    assert sol.metadata == {}


def test_solution_with_metadata():
    sol = QAPSolution(
        permutation=(0, 1),
        cost=10,
        metadata={"iterations": 100},
    )
    assert sol.metadata["iterations"] == 100
