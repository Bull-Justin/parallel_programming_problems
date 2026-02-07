"""Tests for the CLI entry point."""
from __future__ import annotations

import subprocess
import sys


def test_help():
    result = subprocess.run(
        [sys.executable, "-m", "qap_solver.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "qap-solve" in result.stdout or "usage" in result.stdout.lower()


def test_solve_nug12_sa():
    result = subprocess.run(
        [sys.executable, "-m", "qap_solver.cli", "nug12", "-s", "sa", "-v"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "Cost:" in result.stdout


def test_invalid_solver():
    result = subprocess.run(
        [sys.executable, "-m", "qap_solver.cli", "nug12", "-s", "nonexistent"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
