"""Random search control model."""
import numpy as np

from racing_line.optimisers.base import Result, notify, uniform_population


def random_search(prob, budget, rng, pop=60, observer=None):
    best_x, best_f, evals, hist = None, np.inf, 0, []
    while evals + pop <= budget:
        X = uniform_population(prob, rng, pop)
        f = prob.evaluate(X)
        evals += pop
        if f.min() < best_f:
            best_x, best_f = X[f.argmin()].copy(), float(f.min())
        hist.append((evals, best_f))
        notify(observer, evals, X, f)
    return Result(best_x, best_f, np.array(hist))
