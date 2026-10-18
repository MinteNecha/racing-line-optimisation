"""Differential evolution with three mutation strategies."""
import numpy as np

from racing_line.optimisers.base import finish, notify, uniform_population


def mutant(X, best, a, b, c, F, strategy):
    if strategy == "rand1":
        return a + F * (b - c)
    if strategy == "best1":
        return best + F * (a - b)
    if strategy == "ctb1":
        return X + F * (best - X) + F * (a - b)
    raise ValueError(f"unknown strategy '{strategy}'")


def de(prob, budget, rng, pop=60, F=0.6, CR=0.9, strategy="rand1", observer=None):
    X = uniform_population(prob, rng, pop)
    f = prob.evaluate(X)
    evals, hist = pop, [(pop, f.min())]
    rows = np.arange(pop)
    notify(observer, evals, X, f)
    while evals + pop <= budget:
        scores = rng.random((pop, pop))
        scores[rows, rows] = 2.0
        picks = np.argsort(scores, axis=1)[:, :3]
        a, b, c = X[picks[:, 0]], X[picks[:, 1]], X[picks[:, 2]]
        V = mutant(X, X[f.argmin()], a, b, c, F, strategy)
        mask = rng.random(X.shape) < CR
        mask[rows, rng.integers(0, prob.dim, pop)] = True
        U = np.clip(np.where(mask, V, X), -1, 1)
        fu = prob.evaluate(U)
        evals += pop
        win = fu <= f
        X[win], f[win] = U[win], fu[win]
        hist.append((evals, f.min()))
        notify(observer, evals, X, f)
    return finish(X, f, hist)
