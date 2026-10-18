"""Run optimiser configurations over many seeds, in parallel when requested."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from racing_line.optimisers import OPTIMISERS

_PROBLEM = None


def _init_worker(problem):
    global _PROBLEM
    _PROBLEM = problem


def _work(job):
    label, algo, params, seed, budget = job
    result = OPTIMISERS[algo](_PROBLEM, budget, np.random.default_rng(seed), **params)
    return label, result


def run_all(problem, configs, seeds, budget, workers=None):
    """Return {label: [Result per seed]}. workers=1 runs in the calling process."""
    jobs = [(label, algo, params, seed, budget)
            for label, (algo, params) in configs.items() for seed in range(seeds)]
    if workers == 1:
        _init_worker(problem)
        done = [_work(job) for job in jobs]
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(problem,)) as pool:
            done = list(pool.map(_work, jobs, chunksize=1))
    runs = {label: [] for label in configs}
    for label, result in done:
        runs[label].append(result)
    return runs


def finals(results):
    return np.array([r.fitness for r in results])
