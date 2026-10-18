"""Does the sideways reach of the search box limit the lap time?

The standard optimisers search each of the 50 spline knots in [-1, 1]. A line can still sit on the track edge
when a knot is larger than 1, because the finished line is clipped to the track. This study lets the knots
reach further (a reach of 2 means [-2, 2]) and polishes the best line with a classical local search.

Kept in tests/ on purpose, so the main system is untouched. Run it from the project folder:
    python -m tests.reach_run
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
from scipy.optimize import minimize

from racing_line.baselines import reference_times
from racing_line.experiments.configs import HEADLINE, NAMES, main_configs
from racing_line.experiments.runner import finals, run_all
from racing_line.problem import LineProblem
from racing_line.track import DEFAULT_CIRCUIT, build_track

REACHES = (1.0, 1.5, 2.0, 3.0)
SEEDS = 5
BUDGET = 50_000
POLISH_EVALUATIONS = 20_000


class WideProblem(LineProblem):
    """Same problem, but a decision value of 1 means a knot at `reach` instead of 1."""

    def __init__(self, track, reach):
        super().__init__(track)
        self.reach = reach

    def points(self, A):
        return super().points(np.atleast_2d(A) * self.reach)


def polish(problem, x, evaluations=POLISH_EVALUATIONS):
    """Powell search from x inside the same box. Returns (decision vector, lap time, evaluations used)."""
    start = float(problem.evaluate(x)[0])
    result = minimize(lambda v: float(problem.evaluate(np.clip(v, -1, 1))[0]), x, method="Powell",
                      options={"maxfev": evaluations, "xtol": 1e-6, "ftol": 1e-10})
    polished = np.clip(result.x, -1, 1)
    lap = float(problem.evaluate(polished)[0])
    return (polished, lap, result.nfev) if lap < start else (x, start, result.nfev)


def study(track, base_t, reaches, seeds, budget, workers, evaluations):
    configs = {key: main_configs()[key] for key in HEADLINE}
    rows, polish_rows = [], []
    for reach in reaches:
        problem = WideProblem(track, reach)
        runs = run_all(problem, configs, seeds, budget, workers)
        for key, results in runs.items():
            f = finals(results)
            rows.append({"reach": reach, "method": NAMES[key], "mean_s": f.mean(), "std_s": f.std(),
                         "best_s": f.min(), "gain_vs_min_curvature_s": base_t["min_curvature"] - f.mean()})
        best = min((r for results in runs.values() for r in results), key=lambda r: r.fitness)
        _, lap, used = polish(problem, best.x, evaluations)
        polish_rows.append({"reach": reach, "best_evolved_s": best.fitness, "after_polish_s": lap,
                            "polish_gain_s": best.fitness - lap, "polish_evaluations": used})
    return rows, polish_rows


def write_rows(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})


def print_rows(rows, title):
    print(title)
    print("  ".join(f"{k:>{max(len(k), 9)}s}" for k in rows[0]))
    for row in rows:
        print("  ".join(f"{(f'{v:.3f}' if isinstance(v, float) else str(v)):>{max(len(k), 9)}s}" for k, v in row.items()))


def plot(rows, base_t, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in dict.fromkeys(r["method"] for r in rows):
        mine = [r for r in rows if r["method"] == method]
        ax.errorbar([r["reach"] for r in mine], [r["mean_s"] for r in mine], [r["std_s"] for r in mine],
                    marker="o", capsize=3, label=method)
    ax.axhline(base_t["min_curvature"], color="k", ls="--", lw=1, label="Minimum curvature baseline")
    ax.set_xlabel("Reach of the spline knots (1 = standard search box)")
    ax.set_ylabel("Mean final lap time (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(reaches=REACHES, seeds=SEEDS, budget=BUDGET, workers=None, out=Path("results"),
         circuit=DEFAULT_CIRCUIT, evaluations=POLISH_EVALUATIONS):
    out.mkdir(parents=True, exist_ok=True)
    track = build_track(circuit)
    _, base_t = reference_times(LineProblem(track))
    rows, polish_rows = study(track, base_t, reaches, seeds, budget, workers, evaluations)
    print(f"Circuit: {circuit}, {seeds} seeds, {budget:,} evaluations, minimum curvature {base_t['min_curvature']:.3f} s")
    print_rows(rows, "Mean final lap time by reach")
    print_rows(polish_rows, "Powell polish of the best evolved line")
    write_rows(rows, out / "reach_run.csv")
    write_rows(polish_rows, out / "reach_polish.csv")
    plot(rows, base_t, out / "reach_run.png")
    return rows, polish_rows


if __name__ == "__main__":
    main()