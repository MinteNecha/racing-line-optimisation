"""Real coded genetic algorithm."""
import numpy as np

from racing_line.optimisers.base import finish, notify, uniform_population


def crossover(p1, p2, kind, pc, rng):
    if kind == "blx":
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        span = hi - lo
        child = rng.uniform(lo - 0.5 * span, hi + 0.5 * span)
    elif kind == "arith":
        w = rng.random((len(p1), 1))
        child = w * p1 + (1 - w) * p2
    elif kind == "uniform":
        child = np.where(rng.random(p1.shape) < 0.5, p1, p2)
    else:
        raise ValueError(f"unknown crossover '{kind}'")
    return np.where(rng.random((len(p1), 1)) < pc, child, p1)


def ga(prob, budget, rng, pop=60, pc=0.9, pm=0.05, sigma=0.1, elite=2, k=3, cross="blx", observer=None):
    """Tournament selection, crossover, Gaussian mutation and elitism."""
    X = uniform_population(prob, rng, pop)
    f = prob.evaluate(X)
    evals, hist = pop, [(pop, f.min())]
    notify(observer, evals, X, f)
    m = pop - elite
    while evals + m <= budget:
        order = np.argsort(f)
        idx = rng.integers(0, pop, (2 * m, k))
        win = idx[np.arange(2 * m), f[idx].argmin(1)]
        C = crossover(X[win[:m]], X[win[m:]], cross, pc, rng)
        C += (rng.random(C.shape) < pm) * rng.normal(0, sigma, C.shape)
        C = np.clip(C, -1, 1)
        X = np.vstack([X[order[:elite]], C])
        f = np.concatenate([f[order[:elite]], prob.evaluate(C)])
        evals += m
        hist.append((evals, f.min()))
        notify(observer, evals, X, f)
    return finish(X, f, hist)
