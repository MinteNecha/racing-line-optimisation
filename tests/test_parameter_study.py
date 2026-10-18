import numpy as np
import pytest

import parameter_study as ps
from racing_line.optimisers import OPTIMISERS
from racing_line.ui.parameters import PARAMS, defaults

SMALL = {"ga": {"pc": [0.5, 0.9], "elite": [0, 2]}, "de": {"pop": [20, 60]}, "pso": {"vmax": [0.25, 0.5]}}


def test_every_swept_setting_is_a_real_setting_with_a_default_in_its_grid():
    for algo, grid in ps.GRIDS.items():
        names = {p.name for p in PARAMS[algo]}
        for param, values in grid.items():
            assert param in names
            assert defaults(algo)[param] in values


def test_every_configuration_runs_and_gives_a_valid_line(problem):
    for algo, params in ps.build_configs().values():
        result = OPTIMISERS[algo](problem, 600, np.random.default_rng(0), **params)
        assert result.x.shape == (problem.dim,) and abs(result.x).max() <= 1.0


def test_the_default_run_is_shared_so_it_is_only_run_once_per_algorithm():
    configs = ps.build_configs()
    assert [k for k in configs if k[1] == "default"] == [("ga", "default"), ("de", "default"), ("pso", "default")]
    assert ("ga", "pc", 0.9) not in configs and ("ga", "pc", 0.5) in configs


def test_small_study_gives_one_row_per_value_and_correct_comparisons(tmp_path):
    rows = ps.main(grids=SMALL, seeds=2, budget=1_200, workers=1, out=tmp_path)
    assert len(rows) == sum(len(v) for g in SMALL.values() for v in g.values())
    for r in rows:
        assert 0.0 <= r["p_vs_default"] <= 1.0
        if r["is_default"]:
            assert r["change_vs_default_s"] == 0.0 and r["p_vs_default"] == 1.0
    for name in ("parameter_study.csv", "parameter_study.png"):
        assert (tmp_path / name).stat().st_size > 100


def test_unknown_setting_is_rejected(problem):
    with pytest.raises(TypeError):
        OPTIMISERS["ga"](problem, 600, np.random.default_rng(0), mutation=0.1)