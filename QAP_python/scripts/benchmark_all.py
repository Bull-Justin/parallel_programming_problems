"""Run all QAP solvers and produce a Markdown comparison summary with charts.

Usage:  python scripts/benchmark_all.py [output_dir]
"""
from __future__ import annotations

import base64
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qap_solver.benchmarks import list_benchmarks, load_benchmark
from qap_solver.problem import QAPInstance
from qap_solver.solvers import get_solver


@dataclass
class RunResult:
    benchmark: str
    solver: str
    cost: int
    elapsed: float
    optimum: int | None = None

    @property
    def gap_pct(self) -> float | None:
        if self.optimum is None or self.optimum == 0:
            return None
        return (self.cost - self.optimum) / self.optimum * 100

    @property
    def gap_str(self) -> str:
        gap = self.gap_pct
        if gap is None:
            return "—"
        if gap == 0.0:
            return "0.00%"
        return f"+{gap:.2f}%"


def make_small_instance(n: int = 8, seed: int = 42) -> QAPInstance:
    rng = np.random.default_rng(seed)
    flow = rng.integers(0, 30, size=(n, n), dtype=np.int64)
    dist = rng.integers(0, 30, size=(n, n), dtype=np.int64)
    np.fill_diagonal(flow, 0)
    np.fill_diagonal(dist, 0)
    return QAPInstance(n=n, flow=flow, distance=dist, name=f"random_{n}")


EXACT_SOLVERS = ["brute_force", "dp", "parallel_bf"]
META_SOLVERS = ["sa", "ga"]
ALL_SOLVERS = EXACT_SOLVERS + META_SOLVERS

SOLVER_LABELS = {
    "brute_force": "Brute Force",
    "dp": "DP",
    "parallel_bf": "Parallel BF",
    "sa": "Sim. Annealing",
    "ga": "Genetic Alg.",
}

SOLVER_COLORS = {
    "brute_force": "#4C72B0",
    "dp": "#55A868",
    "parallel_bf": "#C44E52",
    "sa": "#8172B3",
    "ga": "#CCB974",
}


def run_solver(solver_name: str, instance: QAPInstance, **kwargs) -> RunResult:
    solver = get_solver(solver_name, **kwargs)
    solution = solver.solve(instance)
    return RunResult(
        benchmark=instance.name or "unknown",
        solver=solver_name,
        cost=solution.cost,
        elapsed=solution.metadata.get("elapsed_seconds", 0.0),
        optimum=instance.known_optimum,
    )


def run_all() -> tuple[list[RunResult], list[RunResult]]:
    """Return (small_results, benchmark_results)."""
    small = make_small_instance()
    small_results: list[RunResult] = []

    bf_result = run_solver("brute_force", small, max_n=8)
    true_optimum = bf_result.cost
    bf_result.optimum = true_optimum
    small_results.append(bf_result)

    small_with_opt = QAPInstance(
        n=small.n,
        flow=small.flow,
        distance=small.distance,
        name=small.name,
        known_optimum=true_optimum,
    )

    for solver_name in ALL_SOLVERS:
        if solver_name == "brute_force":
            continue
        kwargs = {}
        if solver_name == "parallel_bf":
            kwargs["max_n"] = 8
        result = run_solver(solver_name, small_with_opt, **kwargs)
        small_results.append(result)

    benchmark_results: list[RunResult] = []
    for bm_name in list_benchmarks():
        instance = load_benchmark(bm_name)
        for solver_name in META_SOLVERS:
            result = run_solver(solver_name, instance)
            benchmark_results.append(result)

    return small_results, benchmark_results


def _img_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def generate_cost_chart(results: list[RunResult], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))

    solvers = [r.solver for r in results]
    costs = [r.cost for r in results]
    colors = [SOLVER_COLORS[s] for s in solvers]
    labels = [SOLVER_LABELS[s] for s in solvers]
    optimum = results[0].optimum

    bars = ax.bar(labels, costs, color=colors, edgecolor="white", linewidth=0.8)

    if optimum is not None:
        ax.axhline(y=optimum, color="#E74C3C", linestyle="--", linewidth=1.5,
                    label=f"Optimum ({optimum})")
        ax.legend(fontsize=10)

    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{cost:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Solution Cost by Solver — Small Instance (n=8)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylabel("Cost", fontsize=11)
    ax.set_ylim(bottom=optimum * 0.95 if optimum else 0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = out_dir / "cost_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_time_chart(results: list[RunResult], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))

    solvers = [r.solver for r in results]
    times = [r.elapsed for r in results]
    colors = [SOLVER_COLORS[s] for s in solvers]
    labels = [SOLVER_LABELS[s] for s in solvers]

    bars = ax.bar(labels, times, color=colors, edgecolor="white", linewidth=0.8)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.3f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Execution Time by Solver — Small Instance (n=8)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = out_dir / "time_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_gap_chart(results: list[RunResult], out_dir: Path) -> Path:
    benchmarks: list[str] = []
    for r in results:
        if r.benchmark not in benchmarks:
            benchmarks.append(r.benchmark)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(benchmarks))
    width = 0.35

    for i, solver_name in enumerate(META_SOLVERS):
        gaps = []
        for bm in benchmarks:
            match = [r for r in results if r.benchmark == bm and r.solver == solver_name]
            gaps.append(match[0].gap_pct if match and match[0].gap_pct is not None else 0.0)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, gaps, width,
                      label=SOLVER_LABELS[solver_name],
                      color=SOLVER_COLORS[solver_name],
                      edgecolor="white", linewidth=0.8)

        for bar, gap in zip(bars, gaps):
            label = f"{gap:.2f}%" if gap > 0 else "optimal"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Gap from Known Optimum — QAPLIB Benchmarks (n=12)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylabel("Gap from Optimum (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = out_dir / "qaplib_gap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def format_summary(
    small_results: list[RunResult],
    benchmark_results: list[RunResult],
    chart_paths: dict[str, Path] | None = None,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []

    lines.append("# QAP Solver Benchmark Summary")
    lines.append("")
    lines.append(f"Run: {timestamp}")
    lines.append("")

    lines.append("## All Solvers — Small Instance (n=8, seed=42)")
    lines.append("")
    lines.append("| Solver | Cost | Time (s) | Gap |")
    lines.append("|--------|------|----------|-----|")
    for r in small_results:
        lines.append(
            f"| {r.solver} | {r.cost} | {r.elapsed:.4f} | {r.gap_str} |"
        )
    lines.append("")

    if chart_paths:
        for key, title in [
            ("cost", "Solution Cost Comparison"),
            ("time", "Execution Time Comparison"),
        ]:
            if key in chart_paths:
                b64 = _img_to_base64(chart_paths[key])
                lines.append(f"### {title}")
                lines.append("")
                lines.append(
                    f'<img src="data:image/png;base64,{b64}" '
                    f'alt="{title}" width="700">'
                )
                lines.append("")

    benchmarks_seen: list[str] = []
    for r in benchmark_results:
        if r.benchmark not in benchmarks_seen:
            benchmarks_seen.append(r.benchmark)

    lines.append("## Metaheuristics — QAPLIB Benchmarks (n=12)")
    lines.append("")

    header = "| Benchmark | Optimum"
    sep = "|-----------|--------"
    for s in META_SOLVERS:
        header += f" | {s.upper()} Cost | {s.upper()} Time | {s.upper()} Gap"
        sep += " | --- | --- | ---"
    header += " |"
    sep += " |"
    lines.append(header)
    lines.append(sep)

    for bm in benchmarks_seen:
        bm_rows = [r for r in benchmark_results if r.benchmark == bm]
        opt = bm_rows[0].optimum if bm_rows else None
        opt_str = str(opt) if opt is not None else "—"
        row = f"| {bm} | {opt_str}"
        for s in META_SOLVERS:
            match = [r for r in bm_rows if r.solver == s]
            if match:
                r = match[0]
                row += f" | {r.cost} | {r.elapsed:.2f}s | {r.gap_str}"
            else:
                row += " | — | — | —"
        row += " |"
        lines.append(row)

    lines.append("")

    if chart_paths and "gap" in chart_paths:
        lines.append("### Gap from Optimum")
        lines.append("")
        b64 = _img_to_base64(chart_paths["gap"])
        lines.append(
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="Gap from Optimum" width="750">'
        )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running benchmarks...", flush=True)
    t0 = time.perf_counter()
    small_results, benchmark_results = run_all()
    elapsed = time.perf_counter() - t0
    print(f"Completed in {elapsed:.1f}s", flush=True)

    print("Generating charts...", flush=True)
    chart_paths = {
        "cost": generate_cost_chart(small_results, out_dir),
        "time": generate_time_chart(small_results, out_dir),
        "gap": generate_gap_chart(benchmark_results, out_dir),
    }

    summary = format_summary(small_results, benchmark_results, chart_paths)

    summary_path = out_dir / "benchmark_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Summary written to {summary_path}")
    print(f"Charts: {', '.join(str(p) for p in chart_paths.values())}")

    # Write a plain summary for GitHub step summary
    plain_summary = format_summary(small_results, benchmark_results, chart_paths=None)
    plain_path = out_dir / "step_summary.md"
    with open(plain_path, "w", encoding="utf-8") as f:
        f.write(plain_summary)
    print(f"Step summary written to {plain_path}")

    print()
    print(plain_summary)


if __name__ == "__main__":
    main()
