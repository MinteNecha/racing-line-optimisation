import numpy as np
import pytest

from racing_line.baselines import reference_times
from racing_line.problem import LineProblem
from racing_line.track import CIRCUITS, build_track
from racing_line.ui.rendering import COLOURS, TITLES, Dashboard
from racing_line.ui.simulation import simulate

ALGOS = ["ga", "de", "pso", "random"]


@pytest.fixture(scope="module")
def traces(problem):
    return {a: simulate(problem, a, {}, 3_000, 0) for a in ALGOS}


@pytest.fixture(scope="module")
def base_t(reference_lines, problem):
    return {k: float(problem.evaluate(x)[0]) for k, x in reference_lines.items()}


def build(problem, reference_lines, base_t, traces, algos, focus=None):
    return Dashboard(problem, reference_lines, base_t, {a: traces[a] for a in algos}, focus)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_dashboard_builds_for_one_to_four_algorithms(problem, reference_lines, base_t, traces, count):
    dash = build(problem, reference_lines, base_t, traces, ALGOS[:count])
    assert len(dash.fig.axes) == (5 if count == 1 else 2 * count + 2)


def test_every_algorithm_has_a_colour_and_a_title():
    assert set(COLOURS) == set(TITLES) == set(ALGOS)


def test_update_moves_the_numbers_and_curves_forward(problem, reference_lines, base_t, traces):
    dash = build(problem, reference_lines, base_t, traces, ["ga"])
    trace = traces["ga"]
    dash.update(trace.evals[0])
    first_text, first_points = dash._big["ga"].get_text(), len(dash._curve["ga"].get_xdata())
    dash.update(trace.evals[-1])
    assert dash._big["ga"].get_text() == f"{trace.best_f[-1]:.3f} s"
    assert dash._big["ga"].get_text() != first_text
    assert len(dash._curve["ga"].get_xdata()) > first_points


def test_status_text_reports_the_gain_over_the_minimum_curvature_line(problem, reference_lines, base_t, traces):
    dash = build(problem, reference_lines, base_t, traces, ["pso"])
    dash.update(traces["pso"].evals[-1])
    gain = base_t["min_curvature"] - traces["pso"].best_f[-1]
    assert f"{gain:+.3f} s" in dash._small["pso"].get_text()


def test_drawn_best_line_stays_on_the_track(problem, reference_lines, base_t, traces):
    dash = build(problem, reference_lines, base_t, traces, ["de"])
    dash.update(traces["de"].evals[-1])
    t = problem.track
    for _, best in dash._views["de"]:
        x, y = best.get_data()
        assert np.hypot(x - t.cx, y - t.cy).max() <= t.limit + 1e-6


def test_single_layout_follows_the_population_diversity(problem, reference_lines, base_t, traces):
    dash = build(problem, reference_lines, base_t, traces, ["pso"])
    trace = traces["pso"]
    dash.update(trace.evals[-1])
    assert np.array_equal(dash._diversity.get_ydata(), trace.diversity)
    assert dash.fig.axes[3].get_yscale() == "log"


def test_fixed_axes_let_a_partial_run_be_drawn(problem, reference_lines, base_t, traces):
    trace = traces["ga"]
    partial = type(trace)(*(field[:5] for field in (trace.evals, trace.best_f, trace.best_x,
                                                    trace.sample_x, trace.diversity)))
    dash = Dashboard(problem, reference_lines, base_t, {"ga": partial}, budget=30_000, lowest=61.0)
    assert dash._curve["ga"].axes.get_xlim() == (0, 30_000)
    assert dash._curve["ga"].axes.get_ylim()[0] == 61.0


def test_unknown_focus_falls_back_to_the_first_algorithm(problem, reference_lines, base_t, traces):
    dash = build(problem, reference_lines, base_t, traces, ["de", "pso"], focus="ga")
    assert dash.focus == "de"


def test_png_bytes_are_a_valid_image(problem, reference_lines, base_t, traces):
    dash = build(problem, reference_lines, base_t, traces, ["ga", "pso"])
    data = dash.png(traces["ga"].evals[-1], dpi=60)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_frame_can_be_saved_as_an_image(problem, reference_lines, base_t, traces, tmp_path):
    dash = build(problem, reference_lines, base_t, traces, ALGOS[:3])
    dash.update(traces["ga"].evals[-1])
    path = tmp_path / "frame.png"
    dash.fig.savefig(path, dpi=60)
    assert path.stat().st_size > 10_000


@pytest.mark.parametrize("circuit", list(CIRCUITS))
def test_dashboard_draws_every_circuit_inside_the_fixed_speed_axis(circuit):
    problem = LineProblem(build_track(circuit))
    lines, base_t = reference_times(problem)
    trace = simulate(problem, "ga", {}, 1_500, 0)
    dash = Dashboard(problem, lines, base_t, {"ga": trace})
    assert dash.png(trace.evals[-1], dpi=50)[:8] == b"\x89PNG\r\n\x1a\n"
    x0, x1, y0, y1 = dash._window
    t = problem.track
    assert x0 < t.cx.min() + t.length and x1 > x0 and y1 > y0
    assert dash._big["ga"].get_text() == f"{trace.best_f[-1]:.3f} s"
