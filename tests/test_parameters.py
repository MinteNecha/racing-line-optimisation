import inspect

import numpy as np
import pytest

from racing_line.optimisers import OPTIMISERS
from racing_line.ui.parameters import BLURBS, PARAMS, defaults, describe
from racing_line.ui.rendering import TITLES

EXTREMES = [(algo, p, bound) for algo, params in PARAMS.items() for p in params if not p.options
            for bound in (p.low, p.high)]
CHOICES = [(algo, p, option) for algo, params in PARAMS.items() for p in params if p.options for option in p.options]


def run(problem, algo, **params):
    return OPTIMISERS[algo](problem, 1_200, np.random.default_rng(0), **params)


def test_every_algorithm_is_described():
    assert set(PARAMS) == set(BLURBS) == set(TITLES) == set(OPTIMISERS)


@pytest.mark.parametrize("algo", list(PARAMS))
def test_every_setting_is_an_argument_of_its_optimiser(algo):
    accepted = set(inspect.signature(OPTIMISERS[algo]).parameters)
    assert {p.name for p in PARAMS[algo]} <= accepted


@pytest.mark.parametrize("algo", list(PARAMS))
def test_defaults_sit_inside_their_ranges_and_match_the_optimiser_defaults(algo):
    signature = inspect.signature(OPTIMISERS[algo]).parameters
    for p in PARAMS[algo]:
        assert signature[p.name].default == p.default
        if p.options:
            assert p.default in p.options
        else:
            assert p.low <= p.default <= p.high


@pytest.mark.parametrize("algo,param,bound", EXTREMES, ids=lambda v: getattr(v, "name", v))
def test_both_ends_of_every_slider_give_a_valid_run(problem, algo, param, bound):
    result = run(problem, algo, **{**defaults(algo), param.name: bound})
    assert np.abs(result.x).max() <= 1.0
    assert np.isfinite(result.fitness)


@pytest.mark.parametrize("algo,param,option", CHOICES, ids=lambda v: getattr(v, "name", v))
def test_every_choice_gives_a_valid_run(problem, algo, param, option):
    result = run(problem, algo, **{**defaults(algo), param.name: option})
    assert np.isfinite(result.fitness)


def test_describe_shows_readable_labels_and_explanations():
    rows = describe("pso", {**defaults("pso"), "topology": "lbest"})
    by_setting = {r["Setting"]: r for r in rows}
    assert by_setting["Neighbourhood"]["Value"] == "Ring (two neighbours each)"
    assert by_setting["Inertia weight w"]["Value"] == "0.72"
    assert all(r["What it does"] for r in rows)
