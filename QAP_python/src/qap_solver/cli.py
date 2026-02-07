"""CLI entry point for the qap-solve command."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qap_solver.io import load_qaplib
from qap_solver.solvers import SOLVER_REGISTRY, get_solver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qap-solve",
        description="Solve Quadratic Assignment Problems using various methods.",
    )
    parser.add_argument(
        "problem",
        type=Path,
        help=(
            "Path to a QAPLIB .dat file, or name of a bundled benchmark "
            "(e.g. 'nug12')."
        ),
    )
    parser.add_argument(
        "-s",
        "--solver",
        choices=list(SOLVER_REGISTRY.keys()),
        default="sa",
        help="Solver method to use (default: sa).",
    )
    parser.add_argument(
        "--params",
        type=json.loads,
        default={},
        help='Solver parameters as JSON, e.g. \'{"initial_temp": 200}\'.',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write solution in QAPLIB .sln format.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed solver output.",
    )
    return parser


def _resolve_problem(problem_arg: Path) -> Path:
    """Resolve: either a file path or a bundled benchmark name."""
    if problem_arg.exists():
        return problem_arg
    from qap_solver.benchmarks import get_benchmark_path

    path = get_benchmark_path(problem_arg.stem)
    if path is not None:
        return path
    print(f"Error: Cannot find problem '{problem_arg}'", file=sys.stderr)
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    problem_path = _resolve_problem(args.problem)
    instance = load_qaplib(problem_path)

    if args.verbose:
        print(f"Loaded: {instance.name or problem_path.stem} (n={instance.n})")
        if instance.known_optimum is not None:
            print(f"Known optimum: {instance.known_optimum}")

    solver = get_solver(args.solver, **args.params)

    if args.verbose:
        print(f"Solving with {solver.name}...")

    solution = solver.solve(instance)

    print(f"Cost: {solution.cost}")
    print(f"Permutation: {list(solution.permutation)}")
    elapsed = solution.metadata.get("elapsed_seconds")
    if elapsed is not None:
        print(f"Time: {elapsed:.4f}s")

    if instance.known_optimum is not None:
        gap = (solution.cost - instance.known_optimum) / instance.known_optimum * 100
        print(f"Gap from optimum: {gap:.2f}%")

    if args.verbose:
        for k, v in solution.metadata.items():
            if k != "elapsed_seconds":
                print(f"  {k}: {v}")

    if args.output:
        from qap_solver.io import save_qaplib

        save_qaplib(
            instance, dat_path=problem_path, solution=solution, sln_path=args.output
        )
        print(f"Solution written to {args.output}")


if __name__ == "__main__":
    main()
