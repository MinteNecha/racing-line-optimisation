import numpy as np

import long_run
from racing_line.experiments.configs import HEADLINE, NAMES


def test_best_at_follows_the_history_of_each_seed():
    class Run:
        history = np.array([[100, 70.0], [200, 65.0], [400, 64.0]])

    assert np.allclose(long_run.best_at([Run(), Run()], 200), [65.0, 65.0])
    assert np.allclose(long_run.best_at([Run()], 300), [64.5])


def test_extended_run_writes_a_table_and_a_figure(tmp_path):
    rows = long_run.main(budget=3_200, seeds=2, workers=1, out=tmp_path)
    assert len(rows) == len(HEADLINE) * len(long_run.CHECKPOINTS)
    assert (tmp_path / "long_run.png").stat().st_size > 10_000
    assert len((tmp_path / "long_run.csv").read_text().splitlines()) == 1 + len(rows)
    for key in HEADLINE:
        mine = [r for r in rows if r["method"] == NAMES[key]]
        assert mine[-1]["evaluations"] == 3_200
        means = [r["mean_s"] for r in mine]
        assert means == sorted(means, reverse=True)


def test_gain_is_measured_against_the_minimum_curvature_line(tmp_path):
    rows = long_run.main(budget=1_600, seeds=1, workers=1, out=tmp_path)
    reference = {r["mean_s"] + r["gain_vs_min_curvature_s"] for r in rows}
    assert max(reference) - min(reference) < 1e-9