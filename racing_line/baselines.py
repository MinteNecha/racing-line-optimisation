"""Reference lines that the evolutionary methods are compared against."""
import numpy as np
from scipy.optimize import minimize

from racing_line.vehicle import geometry


def baselines(prob):
    """Centreline, shortest path and minimum curvature lines in the problem's parametrisation."""

    def length(a):
        return geometry(*prob.line(a))[0].sum() / prob.track.length

    def curvature(a):
        seg, kap = geometry(*prob.line(a))
        return (kap ** 2 * seg).sum()

    zero = np.zeros(prob.dim)
    bounds = [(-1, 1)] * prob.dim
    options = {"ftol": 1e-13, "gtol": 1e-10, "maxiter": 5_000, "maxfun": 1_000_000}

    def solve(fn):
        return minimize(fn, zero, method="L-BFGS-B", bounds=bounds, options=options).x

    return {"centreline": zero, "shortest": solve(length), "min_curvature": solve(curvature)}


def reference_times(prob):
    """Reference lines and their lap times: ({name: vector}, {name: seconds})."""
    lines = baselines(prob)
    return lines, {name: float(prob.evaluate(x)[0]) for name, x in lines.items()}
