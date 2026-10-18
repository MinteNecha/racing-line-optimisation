import numpy as np
import pytest

from racing_line.optimisers import OPTIMISERS, de, ga, pso, random_search

BUDGET = 1_800
VARIANTS = [("ga", {"cross": c}) for c in ("blx", "arith", "uniform")] \
    + [("de", {"strategy": s}) for s in ("rand1", "best1", "ctb1")] \
    + [("pso", {"topology": t}) for t in ("gbest", "lbest")] \
    + [("random", {})]


def run(problem, algo, seed=0, budget=BUDGET, **params):
    return OPTIMISERS[algo](problem, budget, np.random.default_rng(seed), **params)


@pytest.mark.parametrize("algo,params", VARIANTS)
def test_result_is_a_valid_solution_with_a_truthful_fitness(problem, algo, params):
    result = run(problem, algo, **params)
    assert result.x.shape == (problem.dim,)
    assert np.abs(result.x).max() <= 1.0
    assert np.isclose(result.fitness, problem.evaluate(result.x)[0])


@pytest.mark.parametrize("algo,params", VARIANTS)
def test_history_never_gets_worse_and_respects_the_budget(problem, algo, params):
    history = run(problem, algo, **params).history
    assert (np.diff(history[:, 1]) <= 1e-12).all()
    assert (np.diff(history[:, 0]) > 0).all()
    assert history[-1, 0] <= BUDGET
    assert np.isclose(history[-1, 1], run(problem, algo, **params).fitness)


@pytest.mark.parametrize("algo", OPTIMISERS)
def test_same_seed_reproduces_and_different_seeds_differ(problem, algo):
    a, b, c = run(problem, algo, seed=5), run(problem, algo, seed=5), run(problem, algo, seed=6)
    assert np.array_equal(a.x, b.x)
    assert not np.array_equal(a.x, c.x)


@pytest.mark.parametrize("algo", ["ga", "de", "pso"])
def test_search_improves_on_its_starting_population(problem, algo):
    history = run(problem, algo, budget=6_000).history
    assert history[-1, 1] < history[0, 1] - 0.3


def test_evolutionary_methods_beat_random_search(problem):
    random_best = run(problem, "random", budget=6_000).fitness
    for algo in ("ga", "de", "pso"):
        assert run(problem, algo, budget=6_000).fitness < random_best


def test_unknown_options_are_rejected(problem):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ga(problem, 600, rng, cross="nope")
    with pytest.raises(ValueError):
        de(problem, 600, rng, strategy="nope")
    with pytest.raises(ValueError):
        pso(problem, 600, rng, topology="nope")


def test_random_search_uses_whole_batches(problem):
    result = random_search(problem, 600, np.random.default_rng(0), pop=60)
    assert result.history[-1, 0] == 600
