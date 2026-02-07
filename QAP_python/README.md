# QAP Solver

A Python package for solving the **Quadratic Assignment Problem** (QAP) using multiple methods.

## The Problem

QAP assigns *n* facilities to *n* locations to minimize:

```
cost = Σᵢ Σⱼ flow[i][j] × distance[perm[i]][perm[j]]
```

where `perm[i] = j` means facility *i* is assigned to location *j*.

## Solvers

| Solver | Type | Key | CLI name |
|--------|------|-----|----------|
| Brute Force | Exact | Enumerates all n! permutations | `brute_force` |
| Dynamic Programming | Exact | Bitmask DP, O(n² · 2ⁿ) | `dp` |
| Simulated Annealing | Metaheuristic | Geometric cooling, O(n) swap delta | `sa` |
| Genetic Algorithm | Metaheuristic | OX1 crossover, tournament selection | `ga` |
| Parallel Brute Force | Exact | Distributes across CPU cores | `parallel_bf` |
| Parallel Multi-Start | Meta-wrapper | Runs multiple SA/GA in parallel | (programmatic) |

## Installation

```bash
pip install -e ".[dev]"
```

## CLI Usage

```bash
# Solve bundled benchmark with simulated annealing
qap-solve nug12 -s sa -v

# Solve with brute force (small instances only)
qap-solve tai10a -s brute_force -v

# Solve with DP
qap-solve nug12 -s dp -v

# Custom solver parameters
qap-solve nug12 -s sa --params '{"initial_temp": 200, "max_iterations": 500000}'

# Solve a custom .dat file
qap-solve path/to/problem.dat -s ga
```

## Python API

```python
from qap_solver import load_qaplib, evaluate_solution, get_solver
from qap_solver.benchmarks import load_benchmark

# Load a bundled benchmark
instance = load_benchmark("nug12")

# Solve with simulated annealing
solver = get_solver("sa", max_iterations=500_000)
solution = solver.solve(instance)

print(f"Cost: {solution.cost}")
print(f"Permutation: {solution.permutation}")
print(f"Time: {solution.metadata['elapsed_seconds']:.2f}s")
```

## Running Tests

```bash
pytest
```
