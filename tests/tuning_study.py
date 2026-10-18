"""Search for good settings for each algorithm, then check them on seeds the search never saw.

1. Screen: random settings drawn from a wide range (plus the defaults), 3 seeds each.
2. Refine: the best few from the screen, re-run on 5 new seeds.
3. Validate: the winner and the defaults, 10 more new seeds, compared with a Mann-Whitney U test.

Picking the best of many tries on the same seeds flatters it, so only step 3 is an honest comparison.
Kept in tests/ on purpose, so the main system is untouched. Run it from the project folder:
    python -m tests.tuning_study
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from racing_line.experiments.runner import _init_worker, _work
from racing_line.problem import LineProblem
from racing_line.track import DEFAULT_CIRCUIT, build_track
from racing_line.ui.parameters import defaults

BUDGET = 50_000
CONFIGS = 30
FINALISTS = 5
SCREEN_SEEDS = (0, 1, 2)
REFINE_SEEDS = (3, 4, 5, 6, 7)
VALIDATION_SEEDS = tuple(range(100, 110))
SAMPLING_SEED = 12345
NAMES = {"ga": "GA", "de": "DE", "pso": "PSO"}
SPACES = {
    "ga": {"pop": ("int", 20, 160), "pc": ("float", 0.6, 1.0), "pm": ("log", 0.01, 0.2),
           "sigma": ("log", 0.02, 0.4), "elite": ("int", 0, 10), "k": ("int", 2, 6),
           "cross": ("choice", ["blx", "arith", "uniform"])},
    "de": {"pop": ("int", 20, 80), "F": ("float", 0.2, 1.0), "CR": ("float", 0.1, 1.0),
           "strategy": ("choice", ["rand1", "best1", "ctb1"])},
    "pso": {"pop": ("int", 20, 120), "w": ("float", 0.4, 0.95), "c1": ("float", 0.5, 2.5),
            "c2": ("float", 1.0, 3.0), "vmax": ("log", 0.1, 1.0), "topology": ("choice", ["gbest", "lbest"])},
}


def draw(spec, rng):
    kind, *args = spec
    if kind == "int":
        return int(rng.integers(args[0], args[1] + 1))
    if kind == "float":
        return round(float(rng.uniform(*args)), 3)
    if kind == "log":
        return round(float(np.exp(rng.uniform(*np.log(args)))), 3)
    return args[0][int(rng.integers(len(args[0])))]


def sample_settings(algo, count, rng):
    """The defaults first, then `count` random settings from the search space."""
    random = [{name: draw(spec, rng) for name, spec in SPACES[algo].items()} for _ in range(count)]
    return [defaults(algo)] + random


def run_jobs(problem, jobs, workers):
    if workers == 1:
        _init_worker(problem)
        return [_work(job) for job in jobs]
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(problem,)) as pool:
        return list(pool.map(_work, jobs, chunksize=1))


def evaluate(problem, candidates, seeds, budget, workers):
    """{label: (algorithm, settings)} -> {label: final lap time for each seed}."""
    jobs = [(label, algo, settings, seed, budget)
            for label, (algo, settings) in candidates.items() for seed in seeds]
    outcome = {label: [] for label in candidates}
    for label, result in run_jobs(problem, jobs, workers):
        outcome[label].append(result.fitness)
    return {label: np.array(times) for label, times in outcome.items()}


def describe(settings):
    return " ".join(f"{k}={v}" for k, v in settings.items())


def tune(problem, algo, count, finalists, screen_seeds, refine_seeds, budget, workers, rng):
    """Returns (screen rows, refine rows, winning settings)."""
    settings = sample_settings(algo, count, rng)
    screen = evaluate(problem, {(algo, i): (algo, s) for i, s in enumerate(settings)}, screen_seeds, budget, workers)
    order = sorted(screen, key=lambda label: screen[label].mean())
    screen_rows = [{"algorithm": NAMES[algo], "id": label[1], "is_default": label[1] == 0,
                    "settings": describe(settings[label[1]]), "screen_mean_s": screen[label].mean(), "rank": rank}
                   for rank, label in enumerate(order, 1)]
    top = order[:finalists]
    refined = evaluate(problem, {label: (algo, settings[label[1]]) for label in top}, refine_seeds, budget, workers)
    winner = min(top, key=lambda label: refined[label].mean())
    refine_rows = [{"algorithm": NAMES[algo], "id": label[1], "settings": describe(settings[label[1]]),
                    "screen_mean_s": screen[label].mean(), "refine_mean_s": refined[label].mean(),
                    "chosen": label == winner} for label in top]
    return screen_rows, refine_rows, settings[winner[1]]


def validate(problem, winners, seeds, budget, workers):
    candidates = {}
    for algo, settings in winners.items():
        candidates[(algo, "default")] = (algo, defaults(algo))
        candidates[(algo, "tuned")] = (algo, settings)
    times = evaluate(problem, candidates, seeds, budget, workers)
    rows = []
    for (algo, version), f in times.items():
        reference = times[(algo, "default")]
        rows.append({"algorithm": NAMES[algo], "version": version, "settings": describe(candidates[(algo, version)][1]),
                     "mean_s": f.mean(), "std_s": f.std(), "best_s": f.min(),
                     "change_vs_default_s": f.mean() - reference.mean(),
                     "p_vs_default": 1.0 if version == "default" else mannwhitneyu(f, reference).pvalue})
    return rows, times


def write_rows(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})


def print_validation(rows, times):
    print("\nValidation on seeds the search never used (negative change = faster than the default)")
    for r in rows:
        print(f"{r['algorithm']:4s}{r['version']:8s}{r['mean_s']:9.3f}{r['std_s']:8.3f}{r['change_vs_default_s']:9.3f}"
              f"  p={r['p_vs_default']:.3f}  {r['settings']}")
    for version in ("default", "tuned"):
        print(f"\nPairwise Mann-Whitney between the {version} algorithms")
        for a, b in combinations(sorted({algo for algo, _ in times}), 2):
            print(f"  {NAMES[a]} vs {NAMES[b]}: p = {mannwhitneyu(times[(a, version)], times[(b, version)]).pvalue:.4f}")


def plot(times, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = sorted(times, key=lambda key: (key[0], key[1] != "default"))
    box = ax.boxplot([times[key] for key in labels], tick_labels=[f"{NAMES[a]}\n{v}" for a, v in labels], patch_artist=True)
    for patch, (_, version) in zip(box["boxes"], labels):
        patch.set_facecolor("#cccccc" if version == "default" else "#6baed6")
    ax.set_ylabel("Final lap time (s), held-out seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(configs=CONFIGS, finalists=FINALISTS, screen_seeds=SCREEN_SEEDS, refine_seeds=REFINE_SEEDS,
         validation_seeds=VALIDATION_SEEDS, budget=BUDGET, workers=None, out=Path("results"),
         circuit=DEFAULT_CIRCUIT):
    out.mkdir(parents=True, exist_ok=True)
    problem = LineProblem(build_track(circuit))
    rng = np.random.default_rng(SAMPLING_SEED)
    screen_rows, refine_rows, winners = [], [], {}
    for algo in SPACES:
        screen, refine, winners[algo] = tune(problem, algo, configs, finalists, screen_seeds, refine_seeds,
                                             budget, workers, rng)
        screen_rows += screen
        refine_rows += refine
        default_rank = next(r["rank"] for r in screen if r["is_default"])
        print(f"{NAMES[algo]}: default ranked {default_rank} of {len(screen)} in the screen; "
              f"chosen settings {describe(winners[algo])}")
    rows, times = validate(problem, winners, validation_seeds, budget, workers)
    print_validation(rows, times)
    write_rows(screen_rows, out / "tuning_screen.csv")
    write_rows(refine_rows, out / "tuning_finalists.csv")
    write_rows(rows, out / "tuning_validation.csv")
    plot(times, out / "tuning_validation.png")
    return rows


if __name__ == "__main__":
    main()