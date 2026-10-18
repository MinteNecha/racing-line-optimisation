import numpy as np
import pytest

from racing_line.ui.parameters import defaults
from racing_line.ui.simulation import simulate, single_view_limits, train_live


@pytest.fixture(scope="module")
def base_t(reference_lines, problem):
    return {k: float(problem.evaluate(x)[0]) for k, x in reference_lines.items()}


def train(problem, reference_lines, base_t, algo="pso", budget=3_000, frames=8, **params):
    drawn = []
    trace = train_live(problem, reference_lines, base_t, algo, {**defaults(algo), **params}, budget, 0,
                       lambda png, evals: drawn.append((evals, png)), frames=frames)
    return trace, drawn


def test_training_draws_a_growing_series_of_png_frames(problem, reference_lines, base_t):
    trace, drawn = train(problem, reference_lines, base_t)
    evals = [e for e, _ in drawn]
    assert 2 <= len(drawn) <= 8 + 1
    assert evals == sorted(evals)
    assert evals[-1] == trace.evals[-1]
    assert all(png[:8] == b"\x89PNG\r\n\x1a\n" for _, png in drawn)


@pytest.mark.parametrize("algo", ["ga", "de", "pso", "random"])
def test_live_training_matches_a_plain_run(problem, reference_lines, base_t, algo):
    trace, _ = train(problem, reference_lines, base_t, algo=algo)
    plain = simulate(problem, algo, defaults(algo), 3_000, seed=0)
    assert np.array_equal(trace.best_x, plain.best_x)
    assert np.array_equal(trace.evals, plain.evals)


def test_a_change_in_settings_changes_the_run(problem, reference_lines, base_t):
    slow, _ = train(problem, reference_lines, base_t, w=0.4)
    fast, _ = train(problem, reference_lines, base_t, w=0.9)
    assert slow.best_f[-1] != fast.best_f[-1]


def test_view_limits_are_anchored_to_the_reference_line(base_t):
    limits = single_view_limits(base_t, 30_000)
    assert limits["budget"] == 30_000
    assert limits["lowest"] < base_t["min_curvature"]
