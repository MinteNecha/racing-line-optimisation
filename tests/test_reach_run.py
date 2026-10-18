import numpy as np
import pytest

import reach_run
from racing_line.experiments.configs import HEADLINE
from racing_line.problem import LineProblem


def test_a_reach_of_one_is_the_standard_problem(track):
    x = np.random.default_rng(0).uniform(-1, 1, (4, 50))
    assert np.allclose(reach_run.WideProblem(track, 1.0).evaluate(x), LineProblem(track).evaluate(x))


@pytest.mark.parametrize("reach", [1.0, 2.0, 3.0])
def test_lines_stay_on_the_track_whatever_the_reach(track, reach):
    problem = reach_run.WideProblem(track, reach)
    x = np.random.default_rng(1).uniform(-1, 1, 50)
    px, py = problem.line(x)
    assert np.hypot(px - track.cx, py - track.cy).max() <= track.limit + 1e-9


def test_a_larger_reach_lets_the_line_stay_on_the_track_edge_longer(track):
    x = np.full(50, 1.0)
    distance = lambda reach: np.hypot(*(np.array(reach_run.WideProblem(track, reach).line(x)) - [track.cx, track.cy]))
    assert (distance(1.0) < track.limit - 1e-6).any() or np.allclose(distance(1.0), track.limit)
    assert np.isclose(distance(3.0), track.limit).all()


def test_polish_never_returns_a_slower_line(problem):
    x = np.random.default_rng(2).uniform(-1, 1, 50)
    start = float(problem.evaluate(x)[0])
    polished, lap, used = reach_run.polish(problem, x, evaluations=600)
    assert lap <= start
    assert np.isclose(float(problem.evaluate(polished)[0]), lap)
    assert np.abs(polished).max() <= 1.0
    assert used > 0


def test_reach_study_writes_tables_and_a_figure(tmp_path):
    rows, polish_rows = reach_run.main(reaches=(1.0, 2.0), seeds=1, budget=1_200, workers=1, out=tmp_path,
                                       evaluations=300)
    assert len(rows) == 2 * len(HEADLINE)
    assert [r["reach"] for r in polish_rows] == [1.0, 2.0]
    assert all(r["polish_gain_s"] >= 0 for r in polish_rows)
    for name in ("reach_run.csv", "reach_polish.csv", "reach_run.png"):
        assert (tmp_path / name).stat().st_size > 100