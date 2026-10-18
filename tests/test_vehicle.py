import numpy as np

from racing_line.config import A_LAT, V_MAX
from racing_line.vehicle import geometry, lap_time, lap_times, profile, speed_profile


def test_curvature_of_a_circle_is_one_over_radius(circle):
    px, py = circle(200.0)
    _, kap = geometry(px, py)
    assert np.allclose(kap, 1 / 200.0, rtol=1e-3)


def test_wide_circle_is_driven_at_top_speed(circle):
    px, py = circle(1000.0)
    expected = 2 * np.pi * 1000.0 / V_MAX
    assert np.isclose(lap_time(px, py), expected, rtol=0.02)


def test_tight_circle_is_limited_by_lateral_grip(circle):
    px, py = circle(100.0)
    grip_limited = 2 * np.pi * 100.0 / np.sqrt(A_LAT * 100.0)
    t = lap_time(px, py)
    assert grip_limited * 0.99 <= t <= grip_limited * 1.05


def test_bigger_circles_are_faster_per_metre(circle):
    small = lap_time(*circle(80.0)) / (2 * np.pi * 80.0)
    large = lap_time(*circle(300.0)) / (2 * np.pi * 300.0)
    assert large < small


def test_speed_respects_top_speed_and_grip_limits(problem, reference_lines):
    px, py = problem.line(reference_lines["min_curvature"])
    v, _ = profile(px, py)
    _, kap = geometry(px, py)
    assert v.max() <= V_MAX + 1e-9
    assert (v ** 2 * kap).max() <= A_LAT * 1.001
    assert v.min() > 0


def test_lap_time_does_not_depend_on_the_start_point(problem, reference_lines):
    px, py = problem.line(reference_lines["min_curvature"])
    assert np.isclose(lap_time(px, py), lap_time(np.roll(px, 37), np.roll(py, 37)), rtol=1e-3)


def test_batch_evaluation_matches_single_evaluation(problem):
    A = np.random.default_rng(0).uniform(-1, 1, (5, problem.dim))
    PX, PY = problem.points(A)
    expected = [lap_time(PX[i], PY[i]) for i in range(5)]
    assert np.allclose(lap_times(PX, PY), expected)


def test_speed_profile_returns_one_speed_per_point(problem):
    px, py = problem.line(np.zeros(problem.dim))
    assert speed_profile(px, py).shape == px.shape
