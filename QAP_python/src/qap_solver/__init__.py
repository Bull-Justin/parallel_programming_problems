"""QAP Solver — Multiple methods for the Quadratic Assignment Problem."""

__version__ = "0.1.0"

from qap_solver.evaluate import evaluate as evaluate_solution
from qap_solver.io import load_qaplib, save_qaplib
from qap_solver.problem import QAPInstance, QAPSolution
from qap_solver.solvers import get_solver

__all__ = [
    "QAPInstance",
    "QAPSolution",
    "load_qaplib",
    "save_qaplib",
    "evaluate_solution",
    "get_solver",
]
