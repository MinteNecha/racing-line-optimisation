"""Particle swarm optimisation with global and ring neighbourhoods."""
import numpy as np

from racing_line.optimisers.base import finish, notify, uniform_population


def pso(prob, budget, rng, pop=60, w=0.72, c1=1.49, c2=1.49, topology="gbest", vmax=0.5, observer=None):
    """Inertia weight PSO. 'gbest' uses the whole swarm, 'lbest' a ring of three neighbours."""
    if topology not in ("gbest", "lbest"):
        raise ValueError(f"unknown topology '{topology}'")
    X = uniform_population(prob, rng, pop)
    V = np.zeros_like(X)
    f = prob.evaluate(X)
    P, pf = X.copy(), f.copy()
    evals, hist = pop, [(pop, f.min())]
    rows = np.arange(pop)
    ring = (rows[:, None] + np.array([-1, 0, 1])) % pop
    notify(observer, evals, X, f)
    while evals + pop <= budget:
        if topology == "gbest":
            G = np.broadcast_to(P[pf.argmin()], P.shape)
        else:
            G = P[ring[rows, pf[ring].argmin(1)]]
        r1, r2 = rng.random((2, *X.shape))
        V = np.clip(w * V + c1 * r1 * (P - X) + c2 * r2 * (G - X), -vmax, vmax)
        X = X + V
        V[(X < -1) | (X > 1)] = 0.0
        X = np.clip(X, -1, 1)
        f = prob.evaluate(X)
        evals += pop
        better = f < pf
        P[better], pf[better] = X[better], f[better]
        hist.append((evals, pf.min()))
        notify(observer, evals, X, f)
    return finish(P, pf, hist)
