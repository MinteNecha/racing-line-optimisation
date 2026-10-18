import numpy as np
import pytest

import knots_study as ks
from racing_line.problem import LineProblem


def test_fifty_pins_is_the_standard_problem(track):
    x = np.random.default_rng(0).uniform(-1, 1, (4, 50))
    assert np.array_equal(LineProblem(track, 50).evaluate(x), LineProblem(track).evaluate(x))


@pytest.mark.parametrize("pins", [10, 40, 60, 120])
def test_the_search_space_matches_the_pin_count_and_lines_stay_on_the_track(track, pins):
    problem = LineProblem(track, pins)
    assert problem.dim == pins
    px, py = problem.line(np.random.default_rng(1).uniform(-1, 1, pins))
    assert np.hypot(px - track.cx, py - track.cy).max() <= track.limit + 1e-9


def test_a_constant_offset_is_the_same_line_whatever_the_pin_count(track):
    times = [LineProblem(track, pins).evaluate(np.full(pins, 0.4))[0] for pins in (20, 40, 50, 60, 100)]
    assert np.allclose(times, times[0])


def test_small_study_writes_tables_and_a_figure_and_always_includes_the_reference(tmp_path):
    rows = ks.main(knots=(30, 60), seeds=2, budget=1_200, workers=1, out=tmp_path)
    assert sorted({r["knots"] for r in rows}) == [30, 50, 60] and len(rows) == 9
    assert all(r["p_vs_ref"] == 1.0 and r["change_vs_ref_s"] == 0.0 for r in rows if r["knots"] == 50)
    for name in ("knots_study_30_50_60_b1200.csv", "knots_study_30_50_60_b1200.png"):
        assert (tmp_path / name).stat().st_size > 100


def test_another_reference_replaces_fifty_and_is_compared_with_itself(tmp_path):
    rows = ks.main(knots=(40, 60), seeds=2, budget=1_200, workers=1, out=tmp_path, reference=60)
    assert sorted({r["knots"] for r in rows}) == [40, 60] and len(rows) == 6
    assert all(r["p_vs_ref"] == 1.0 and r["change_vs_ref_s"] == 0.0 for r in rows if r["knots"] == 60)
    assert (tmp_path / "knots_study_40_60_b1200.csv").exists()


def test_the_pin_spacing_column_is_lap_length_over_pins(tmp_path):
    rows = ks.main(knots=(40,), seeds=1, budget=1_200, workers=1, out=tmp_path)
    assert all(np.isclose(r["spacing_m"] * r["knots"], 4609, atol=2) for r in rows)