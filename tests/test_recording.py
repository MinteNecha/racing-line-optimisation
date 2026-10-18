import numpy as np
import pytest

from racing_line.optimisers import OPTIMISERS
from racing_line.recording import Recorder, Trace
from racing_line.ui.simulation import evals_to_beat, simulate, timeline

BUDGET = 1_800


def run(problem, algo, observer=None, seed=3):
    return OPTIMISERS[algo](problem, BUDGET, np.random.default_rng(seed), observer=observer)


@pytest.mark.parametrize("algo", list(OPTIMISERS))
def test_watching_a_run_does_not_change_it(problem, algo):
    plain, watched = run(problem, algo), run(problem, algo, Recorder())
    assert np.array_equal(plain.x, watched.x)
    assert np.array_equal(plain.history, watched.history)


@pytest.mark.parametrize("algo", list(OPTIMISERS))
def test_one_frame_is_recorded_per_generation(problem, algo):
    recorder = Recorder()
    result = run(problem, algo, recorder)
    trace = recorder.trace()
    assert trace.frames == len(result.history)
    assert np.array_equal(trace.evals, result.history[:, 0])
    assert np.allclose(trace.best_f, result.history[:, 1])


def test_trace_has_consistent_shapes_and_a_truthful_best(problem):
    trace = simulate(problem, "ga", {}, BUDGET, seed=0, sample=12)
    assert trace.best_x.shape == (trace.frames, problem.dim)
    assert trace.sample_x.shape == (trace.frames, 12, problem.dim)
    assert (np.diff(trace.best_f) <= 1e-12).all()
    assert np.isclose(problem.evaluate(trace.best_x[-1])[0], trace.best_f[-1])
    assert np.abs(trace.sample_x).max() <= 1.0
    assert trace.diversity.shape == (trace.frames,)
    assert (trace.diversity > 0).all()


@pytest.mark.parametrize("algo,ratio", [("ga", 0.5), ("pso", 0.5), ("de", 0.9)])
def test_diversity_shrinks_as_a_search_converges(problem, algo, ratio):
    trace = simulate(problem, algo, {}, 6_000, seed=0)
    assert trace.diversity[-1] < ratio * trace.diversity[0]


def test_random_search_keeps_its_diversity(problem):
    trace = simulate(problem, "random", {}, 3_000, seed=0)
    assert trace.diversity.min() > 0.9 * trace.diversity.max()


def test_frame_at_returns_the_last_frame_not_after_the_request():
    trace = Trace(np.array([60, 120, 180]), np.zeros(3), np.zeros((3, 2)), np.zeros((3, 1, 2)), np.zeros(3))
    assert trace.frame_at(10) == 0
    assert trace.frame_at(120) == 1
    assert trace.frame_at(179) == 1
    assert trace.frame_at(10_000) == 2


def test_timeline_is_increasing_dense_early_and_inside_every_run(problem):
    traces = {a: simulate(problem, a, {}, 6_000, 0) for a in ("ga", "de")}
    grid = timeline(traces, count=40)
    assert grid == sorted(set(grid))
    assert grid[0] == max(t.evals[0] for t in traces.values())
    assert grid[-1] == min(t.evals[-1] for t in traces.values())
    assert grid[1] - grid[0] < grid[-1] - grid[-2]


def test_evals_to_beat_reports_the_first_crossing_or_none():
    trace = Trace(np.array([60, 120, 180]), np.array([64.0, 62.0, 61.0]),
                  np.zeros((3, 2)), np.zeros((3, 1, 2)), np.zeros(3))
    assert evals_to_beat(trace, 62.5) == 120
    assert evals_to_beat(trace, 60.0) is None
