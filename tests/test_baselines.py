import numpy as np
import pytest
from scipy.optimize import minimize

from racing_line.baselines import reference_times
from racing_line.problem import LineProblem
from racing_line.track import CIRCUITS, build_track
from racing_line.vehicle import geometry


def test_baselines_are_valid_solutions(reference_lines, problem):
    assert set(reference_lines) == {"centreline", "shortest", "min_curvature"}
    for x in reference_lines.values():
        assert x.shape == (problem.dim,)
        assert np.abs(x).max() <= 1.0


def test_shortest_line_is_shorter_than_the_centreline(reference_lines, problem):
    length = lambda name: geometry(*problem.line(reference_lines[name]))[0].sum()
    assert length("shortest") < length("centreline")


def test_min_curvature_line_is_smoother_than_the_centreline(reference_lines, problem):
    def bending(name):
        seg, kap = geometry(*problem.line(reference_lines[name]))
        return (kap ** 2 * seg).sum()
    assert bending("min_curvature") < bending("centreline")


def test_lap_time_ordering_of_the_baselines(reference_lines, problem):
    t = {name: problem.evaluate(x)[0] for name, x in reference_lines.items()}
    assert t["min_curvature"] < t["shortest"] < t["centreline"]


@pytest.mark.parametrize("circuit", list(CIRCUITS))
def test_reference_times_keep_their_ordering_on_every_circuit(circuit):
    lines, times = reference_times(LineProblem(build_track(circuit)))
    assert set(lines) == set(times)
    assert times["min_curvature"] < times["shortest"] < times["centreline"]


@pytest.mark.parametrize("circuit", list(CIRCUITS))
def test_min_curvature_line_is_converged(circuit):
    problem = LineProblem(build_track(circuit))
    lines, _ = reference_times(problem)

    def bending(a):
        seg, kap = geometry(*problem.line(a))
        return (kap ** 2 * seg).sum()

    start = lines["min_curvature"]
    again = minimize(bending, start, method="L-BFGS-B", bounds=[(-1, 1)] * problem.dim,
                     options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5_000, "maxfun": 1_000_000})
    assert bending(start) - again.fun < 1e-4 * bending(start)
