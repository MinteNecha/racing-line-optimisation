import numpy as np
import pytest

import downforce_run as dr
from racing_line.problem import LineProblem


def lines(n=4, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (n, 50))


def test_with_constant_grip_it_reproduces_the_standard_model_exactly(track):
    constant = dr.DownforceProblem(track, low=38.0, high=38.0)
    x = lines()
    assert np.array_equal(constant.evaluate(x), LineProblem(track).evaluate(x))


def test_downforce_changes_lap_times_and_keeps_them_sensible(track):
    x = lines()
    mine, standard = dr.DownforceProblem(track).evaluate(x), LineProblem(track).evaluate(x)
    assert np.all(mine > 30) and np.all(mine < 200) and not np.allclose(mine, standard)


def test_more_top_speed_grip_never_makes_a_lap_slower(track):
    x = lines()
    weak, strong = dr.DownforceProblem(track, 20, 30).evaluate(x), dr.DownforceProblem(track, 20, 60).evaluate(x)
    assert np.all(strong <= weak + 1e-9)


def test_speed_profile_matches_the_lap_time_and_respects_the_speed_limit(track):
    problem = dr.DownforceProblem(track)
    v = problem.speed(lines(1)[0])
    assert v.shape == (track.n,) and v.max() <= 95.0 + 1e-9 and v.min() > 0


def test_ranking_orders_by_mean_lap_time():
    rows = [{"model": "m", "method": name, "mean_s": t} for name, t in
            [("GA (BLX)", 61.5), ("DE/rand/1", 61.4), ("PSO (gbest)", 61.6), ("Random search", 63.0)]]
    assert dr.ranking(rows, "m") == ["DE/rand/1", "GA (BLX)", "PSO (gbest)"]


def test_small_study_writes_tables_and_a_figure(tmp_path):
    rows, tests = dr.main(seeds=2, budget=1_200, workers=1, out=tmp_path)
    assert len(rows) == 8 and len(tests) == 6
    for name in ("downforce_run.csv", "downforce_tests.csv", "downforce_run.png"):
        assert (tmp_path / name).stat().st_size > 100