import numpy as np


def test_zero_vector_is_the_centreline(problem, track):
    px, py = problem.line(np.zeros(problem.dim))
    assert np.allclose(px, track.cx)
    assert np.allclose(py, track.cy)


def test_lines_never_leave_the_usable_track(problem, track):
    rng = np.random.default_rng(1)
    A = np.vstack([rng.uniform(-3, 3, (20, problem.dim)), np.ones(problem.dim), -np.ones(problem.dim)])
    PX, PY = problem.points(A)
    distance = np.hypot(PX - track.cx, PY - track.cy)
    assert distance.max() <= track.limit + 1e-9


def test_evaluate_returns_one_positive_lap_time_per_solution(problem):
    A = np.random.default_rng(2).uniform(-1, 1, (7, problem.dim))
    times = problem.evaluate(A)
    assert times.shape == (7,)
    assert (times > 0).all()


def test_evaluate_is_deterministic_and_accepts_a_single_vector(problem):
    a = np.random.default_rng(3).uniform(-1, 1, problem.dim)
    assert problem.evaluate(a)[0] == problem.evaluate(a)[0]
    assert problem.evaluate(a).shape == (1,)
