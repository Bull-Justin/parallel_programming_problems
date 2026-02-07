"""QAPLIB file format reader and writer."""
from __future__ import annotations

import pathlib

import numpy as np

from qap_solver.problem import QAPInstance, QAPSolution


def load_qaplib(
    dat_path: str | pathlib.Path,
    sln_path: str | pathlib.Path | None = None,
) -> QAPInstance:
    """Load a QAPLIB .dat file, auto-detecting .sln if present."""
    dat_path = pathlib.Path(dat_path)
    text = dat_path.read_text()
    numbers = _parse_integers(text)

    n = numbers[0]
    expected = 1 + 2 * n * n
    if len(numbers) < expected:
        raise ValueError(
            f"Expected at least {expected} integers in {dat_path}, got {len(numbers)}"
        )

    flow_flat = numbers[1 : 1 + n * n]
    dist_flat = numbers[1 + n * n : 1 + 2 * n * n]

    flow = np.array(flow_flat, dtype=np.int64).reshape(n, n)
    distance = np.array(dist_flat, dtype=np.int64).reshape(n, n)

    name = dat_path.stem
    known_optimum = None

    if sln_path is None:
        auto_sln = dat_path.with_suffix(".sln")
        if auto_sln.exists():
            sln_path = auto_sln

    if sln_path is not None:
        sln_path = pathlib.Path(sln_path)
        if sln_path.exists():
            sln_numbers = _parse_integers(sln_path.read_text())
            if len(sln_numbers) >= 2:
                known_optimum = sln_numbers[1]

    return QAPInstance(
        n=n,
        flow=flow,
        distance=distance,
        name=name,
        known_optimum=known_optimum,
    )


def load_qaplib_solution(
    sln_path: str | pathlib.Path,
) -> QAPSolution | None:
    """Load a QAPLIB .sln file and return the solution (0-indexed)."""
    sln_path = pathlib.Path(sln_path)
    if not sln_path.exists():
        return None

    numbers = _parse_integers(sln_path.read_text())
    if len(numbers) < 2:
        return None

    n = numbers[0]
    cost = numbers[1]
    perm_1indexed = numbers[2 : 2 + n]

    if len(perm_1indexed) != n:
        return None

    # Convert from 1-indexed to 0-indexed
    permutation = tuple(p - 1 for p in perm_1indexed)

    return QAPSolution(permutation=permutation, cost=cost, solver_name="qaplib_known")


def save_qaplib(
    instance: QAPInstance,
    dat_path: str | pathlib.Path,
    solution: QAPSolution | None = None,
    sln_path: str | pathlib.Path | None = None,
) -> None:
    """Write a QAPInstance to QAPLIB .dat format."""
    dat_path = pathlib.Path(dat_path)
    n = instance.n

    lines = [str(n), ""]
    for row in instance.flow:
        lines.append("  ".join(str(int(x)) for x in row))
    lines.append("")
    for row in instance.distance:
        lines.append("  ".join(str(int(x)) for x in row))
    lines.append("")

    dat_path.write_text("\n".join(lines))

    if solution is not None:
        if sln_path is None:
            sln_path = dat_path.with_suffix(".sln")
        sln_path = pathlib.Path(sln_path)
        # Convert to 1-indexed for QAPLIB format
        perm_1indexed = [p + 1 for p in solution.permutation]
        sln_lines = [
            f"{n} {solution.cost}",
            "  ".join(str(p) for p in perm_1indexed),
            "",
        ]
        sln_path.write_text("\n".join(sln_lines))


def _parse_integers(text: str) -> list[int]:
    """Extract all integers from a text string."""
    return [int(x) for x in text.split()]
