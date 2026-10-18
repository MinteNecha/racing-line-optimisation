"""Shared result type and helpers for the optimisers."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    x: np.ndarray
    fitness: float
    history: np.ndarray  # rows of (evaluations used, best lap time so far)


def finish(X, f, history):
    i = f.argmin()
    return Result(X[i].copy(), float(f[i]), np.array(history))


def notify(observer, evals, X, f):
    """Report the current population to an optional observer after each generation."""
    if observer is not None:
        observer(evals, X, f)


def uniform_population(prob, rng, pop):
    return rng.uniform(-1, 1, (pop, prob.dim))
