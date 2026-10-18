import pytest

import sweep_study as ss
from racing_line.ui.parameters import PARAMS, defaults


@pytest.mark.parametrize("algo", ss.ALGORITHMS)
def test_every_setting_of_every_algorithm_is_swept_inside_its_range(algo):
    grid = ss.make_grids()[algo]
    assert set(grid) == {p.name for p in PARAMS[algo]}
    for p in PARAMS[algo]:
        values = grid[p.name]
        assert defaults(algo)[p.name] in values and len(values) >= 2
        assert values == sorted(values) or p.options
        if p.options:
            assert set(values) == set(p.options)
        elif p.name != "pop":
            assert p.low <= min(values) and max(values) <= p.high


def test_population_size_uses_the_same_grid_for_all_algorithms():
    grids = ss.make_grids()
    assert grids["ga"]["pop"] == grids["de"]["pop"] == grids["pso"]["pop"] == ss.POPULATIONS


def test_integer_settings_stay_integers_and_near_duplicates_are_dropped():
    grids = ss.make_grids()
    assert all(isinstance(v, int) for v in grids["ga"]["elite"] + grids["ga"]["k"])
    assert 1.5 not in grids["pso"]["c1"] and 1.49 in grids["pso"]["c1"]


def test_sensitivity_ranks_settings_by_spread():
    rows = [{"algorithm": "GA", "setting": s, "value": v, "mean_s": m, "p_vs_default": p}
            for s, v, m, p in [("a", 1, 61.0, 1.0), ("a", 2, 61.5, 0.01), ("b", 1, 61.0, 1.0), ("b", 2, 61.1, 0.5)]]
    summary = ss.sensitivity(rows)
    assert [s["setting"] for s in summary] == ["a", "b"]
    assert summary[0]["spread_s"] == pytest.approx(0.5) and summary[0]["significant_values"] == 1


def test_small_sweep_runs_and_writes_everything(tmp_path):
    grids = {"ga": {"pc": [0.5, 0.9]}, "de": {"strategy": ["rand1", "best1"]}, "pso": {"topology": ["gbest", "lbest"]}}
    rows = ss.main(grids=grids, seeds=2, budget=1_200, workers=1, out=tmp_path)
    assert len(rows) == 6 and all(r["p_vs_default"] == 1.0 for r in rows if r["is_default"])
    for name in ("sweep_study.csv", "sweep_sensitivity.csv", "sweep_study.png"):
        assert (tmp_path / name).stat().st_size > 100