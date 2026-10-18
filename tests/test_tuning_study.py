import numpy as np
import pytest

import tuning_study as ts
from racing_line.optimisers import OPTIMISERS
from racing_line.ui.parameters import PARAMS, defaults


def test_the_three_seed_sets_never_overlap():
    sets = [set(ts.SCREEN_SEEDS), set(ts.REFINE_SEEDS), set(ts.VALIDATION_SEEDS)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


@pytest.mark.parametrize("algo", list(ts.SPACES))
def test_search_space_stays_inside_the_dashboard_ranges(algo):
    names = {p.name: p for p in PARAMS[algo]}
    assert set(ts.SPACES[algo]) <= set(names)
    for name, (kind, *args) in ts.SPACES[algo].items():
        if kind == "choice":
            assert set(args[0]) <= set(names[name].options)
        else:
            assert names[name].low <= args[0] and args[1] <= names[name].high


@pytest.mark.parametrize("algo", list(ts.SPACES))
def test_sampling_is_reproducible_starts_with_the_defaults_and_runs(algo, problem):
    first = ts.sample_settings(algo, 4, np.random.default_rng(1))
    again = ts.sample_settings(algo, 4, np.random.default_rng(1))
    assert first == again and first[0] == defaults(algo) and len(first) == 5
    for settings in first:
        result = OPTIMISERS[algo](problem, 600, np.random.default_rng(0), **settings)
        assert result.x.shape == (problem.dim,)


def test_log_and_int_draws_respect_their_limits():
    rng = np.random.default_rng(0)
    logs = [ts.draw(("log", 0.01, 0.2), rng) for _ in range(300)]
    ints = {ts.draw(("int", 2, 4), rng) for _ in range(300)}
    assert 0.01 <= min(logs) and max(logs) <= 0.2
    assert ints == {2, 3, 4}


def test_small_tuning_run_finds_a_winner_per_algorithm_and_writes_everything(tmp_path):
    rows = ts.main(configs=2, finalists=2, screen_seeds=(0,), refine_seeds=(1,), validation_seeds=(2, 3),
                   budget=1_200, workers=1, out=tmp_path)
    assert [(r["algorithm"], r["version"]) for r in rows] == [
        ("GA", "default"), ("GA", "tuned"), ("DE", "default"), ("DE", "tuned"), ("PSO", "default"), ("PSO", "tuned")]
    assert all(r["p_vs_default"] == 1.0 for r in rows if r["version"] == "default")
    for name in ("tuning_screen.csv", "tuning_finalists.csv", "tuning_validation.csv", "tuning_validation.png"):
        assert (tmp_path / name).stat().st_size > 100
    finalists = (tmp_path / "tuning_finalists.csv").read_text().splitlines()[1:]
    assert sum(line.endswith("True") for line in finalists) == 3


def test_the_search_is_deterministic(tmp_path):
    kwargs = dict(configs=2, finalists=1, screen_seeds=(0,), refine_seeds=(1,), validation_seeds=(2,),
                  budget=1_200, workers=1)
    first = ts.main(out=tmp_path / "a", **kwargs)
    second = ts.main(out=tmp_path / "b", **kwargs)
    assert [r["mean_s"] for r in first] == [r["mean_s"] for r in second]