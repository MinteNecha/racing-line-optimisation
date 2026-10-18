"""Extended run: do the headline algorithms keep improving beyond the standard 50,000 evaluations?

This lives in tests/ on purpose, so the main system has no setting for the budget.
Run it from the project folder:  python -m tests.long_run
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from racing_line.baselines import reference_times
from racing_line.experiments import report
from racing_line.experiments.configs import HEADLINE, NAMES, main_configs
from racing_line.experiments.runner import run_all
from racing_line.problem import LineProblem
from racing_line.track import DEFAULT_CIRCUIT, build_track

BUDGET = 200_000
SEEDS = 5
CHECKPOINTS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1)   # fractions of the budget
STANDARD_BUDGET = 50_000


def best_at(results, evaluations):
    """Best lap time each seed had reached after the given number of evaluations."""
    return np.array([np.interp(evaluations, r.history[:, 0], r.history[:, 1]) for r in results])


def table(runs, base_t, budget):
    rows = []
    for key, results in runs.items():
        for fraction in CHECKPOINTS:
            evaluations = int(budget * fraction)
            f = best_at(results, evaluations)
            rows.append({"method": NAMES[key], "evaluations": evaluations, "mean_s": f.mean(), "std_s": f.std(),
                         "gain_vs_min_curvature_s": base_t["min_curvature"] - f.mean()})
    return rows


def write_table(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v if isinstance(v, str) else f"{v:.4f}" for k, v in row.items()})


def print_table(rows):
    print(f"{'method':14s}{'evaluations':>13s}{'mean':>9s}{'std':>8s}{'gain':>8s}")
    for r in rows:
        print(f"{r['method']:14s}{r['evaluations']:13,d}{r['mean_s']:9.3f}{r['std_s']:8.3f}"
              f"{r['gain_vs_min_curvature_s']:8.3f}")


def plot(runs, base_t, budget, path):
    grid = np.geomspace(budget / 100, budget, 300)
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, results in runs.items():
        curves = best_at(results, grid)
        mean, std = curves.mean(0), curves.std(0)
        ax.plot(grid, mean, label=NAMES[key])
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
    ax.axhline(base_t["min_curvature"], color="k", ls="--", lw=1, label="Minimum curvature baseline")
    ax.axvline(STANDARD_BUDGET, color="gray", ls=":", lw=1, label="Standard benchmark budget")
    ax.set_xscale("log")
    ax.set_xlabel("Fitness evaluations (log scale)")
    ax.set_ylabel("Best lap time (s)")
    lowest = min(best_at(results, budget).min() for results in runs.values())
    ax.set_ylim(lowest - 0.05, base_t["min_curvature"] + 0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(budget=BUDGET, seeds=SEEDS, workers=None, out=Path("results"), circuit=DEFAULT_CIRCUIT):
    out.mkdir(parents=True, exist_ok=True)
    problem = LineProblem(build_track(circuit))
    _, base_t = reference_times(problem)
    configs = {key: main_configs()[key] for key in HEADLINE}

    runs = run_all(problem, configs, seeds, budget, workers)
    rows = table(runs, base_t, budget)
    print(f"Circuit: {circuit}, {seeds} seeds, {budget:,} evaluations, minimum curvature {base_t['min_curvature']:.3f} s")
    print_table(rows)
    report.print_significance(runs)
    write_table(rows, out / "long_run.csv")
    plot(runs, base_t, budget, out / "long_run.png")
    return rows


if __name__ == "__main__":
    main()