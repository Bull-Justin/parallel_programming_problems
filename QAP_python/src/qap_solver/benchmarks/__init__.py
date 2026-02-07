"""Bundled QAPLIB benchmark instances."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from qap_solver.problem import QAPInstance

_DATA_DIR = Path(__file__).parent / "data"

AVAILABLE_BENCHMARKS: dict[str, dict] = {
    "nug12": {"file": "nug12.dat", "n": 12, "optimum": 578},
    "had12": {"file": "had12.dat", "n": 12, "optimum": 1652},
    "chr12a": {"file": "chr12a.dat", "n": 12, "optimum": 9552},
    "tai12a": {"file": "tai12a.dat", "n": 12, "optimum": 224416},
}


def get_benchmark_path(name: str) -> Path | None:
    """Return the path to a bundled benchmark .dat file, or None."""
    info = AVAILABLE_BENCHMARKS.get(name)
    if info is None:
        return None
    path = _DATA_DIR / info["file"]
    return path if path.exists() else None


def load_benchmark(name: str) -> QAPInstance:
    """Load a bundled benchmark instance by name."""
    from qap_solver.io import load_qaplib

    path = get_benchmark_path(name)
    if path is None:
        raise ValueError(
            f"Unknown benchmark '{name}'. "
            f"Available: {list(AVAILABLE_BENCHMARKS.keys())}"
        )
    info = AVAILABLE_BENCHMARKS[name]
    sln_path = path.with_suffix(".sln")
    instance = load_qaplib(path, sln_path if sln_path.exists() else None)

    if instance.known_optimum is None:
        instance = replace(instance, known_optimum=info["optimum"], name=name)
    elif not instance.name:
        instance = replace(instance, name=name)

    return instance


def list_benchmarks() -> list[str]:
    """Return names of all available bundled benchmarks."""
    return list(AVAILABLE_BENCHMARKS.keys())
