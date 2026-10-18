"""Does the algorithm ranking survive a car whose grip grows with speed?

The standard model has constant cornering grip. Here a copy of the lap model gives the car grip that rises with
the square of its speed, as aerodynamic downforce does:  grip(v) = LOW + (HIGH - LOW) * (v / V_MAX)^2.
Slow corners get less grip than before and fast corners get more. Only cornering grip changes, and downforce
drag is ignored. LOW and HIGH are illustrative values, not calibrated to a real car.

The main system is untouched. Lap times of the two models are not comparable with each other, so the study
compares rankings, p-values and the shape of the best line. Run from the project folder:
    python -m tests.downforce_run
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.stats import mannwhitneyu

from racing_line.baselines import reference_times
from racing_line.config import A_BRAKE, A_TRACTION, DRAG, MASS, POWER, V_MAX
from racing_line.experiments.configs import HEADLINE, NAMES, study_configs
from racing_line.experiments.plots import _draw_edges
from racing_line.experiments.runner import finals, run_all
from racing_line.problem import LineProblem
from racing_line.track import DEFAULT_CIRCUIT, build_track
from racing_line.vehicle import geometry

LOW_GRIP = 20.0     # m/s^2 at standstill
HIGH_GRIP = 50.0    # m/s^2 at V_MAX
SEEDS = 10
BUDGET = 50_000


@njit
def _profile(px, py, low, slope):
    """Same passes as racing_line.vehicle.profile, with grip(v) = low + slope * v^2."""
    n = len(px)
    seg, kap = geometry(px, py)
    v = np.empty(n)
    for i in range(n):
        spare = kap[i] - slope
        v[i] = V_MAX if spare <= 0 else min(V_MAX, np.sqrt(low / (spare + 1e-9)))
    for k in range(2 * n):
        i = k % n
        j = (i + 1) % n
        r = v[i] * v[i] * kap[i] / (low + slope * v[i] * v[i])
        acc = min(A_TRACTION, POWER / (MASS * max(v[i], 1.0))) * np.sqrt(max(0.0, 1.0 - r * r))
        acc -= DRAG * v[i] * v[i]
        v[j] = min(v[j], np.sqrt(max(v[i] * v[i] + 2 * acc * seg[i], 0.0)))
    for k in range(2 * n):
        j = (n - k) % n
        i = (j - 1) % n
        r = v[j] * v[j] * kap[j] / (low + slope * v[j] * v[j])
        dec = A_BRAKE * np.sqrt(max(0.0, 1.0 - r * r)) + DRAG * v[j] * v[j]
        v[i] = min(v[i], np.sqrt(v[j] * v[j] + 2 * dec * seg[i]))
    return v, seg


@njit
def _lap_times(PX, PY, low, slope):
    out = np.empty(PX.shape[0])
    for row in range(PX.shape[0]):
        v, seg = _profile(PX[row], PY[row], low, slope)
        n = len(v)
        t = 0.0
        for i in range(n):
            t += 2 * seg[i] / (v[i] + v[(i + 1) % n])
        out[row] = t
    return out


class DownforceProblem(LineProblem):
    """Same decision vector and track, lap time from the speed dependent grip model."""

    def __init__(self, track, low=LOW_GRIP, high=HIGH_GRIP):
        super().__init__(track)
        self.low = low
        self.slope = (high - low) / V_MAX ** 2

    def evaluate(self, A):
        return _lap_times(*self.points(A), self.low, self.slope)

    def speed(self, a):
        return _profile(*self.line(a), self.low, self.slope)[0]


def study(track, seeds, budget, workers):
    models = {"constant grip": LineProblem(track), "downforce": DownforceProblem(track)}
    configs = study_configs()
    rows, tests, lines = [], [], {}
    for model, problem in models.items():
        _, base_t = reference_times(problem)
        runs = run_all(problem, configs, seeds, budget, workers)
        f = {key: finals(results) for key, results in runs.items()}
        for key in configs:
            rows.append({"model": model, "method": NAMES[key], "mean_s": f[key].mean(), "std_s": f[key].std(),
                         "best_s": f[key].min(), "gain_vs_min_curvature_s": base_t["min_curvature"] - f[key].mean()})
        for a, b in combinations(HEADLINE, 2):
            tests.append({"model": model, "pair": f"{NAMES[a]} vs {NAMES[b]}",
                          "difference_s": f[a].mean() - f[b].mean(), "p": mannwhitneyu(f[a], f[b]).pvalue})
        lines[model] = min((r for key in HEADLINE for r in runs[key]), key=lambda r: r.fitness).x
    return rows, tests, lines, models


def ranking(rows, model):
    mine = [r for r in rows if r["model"] == model and r["method"] in {NAMES[k] for k in HEADLINE}]
    return [r["method"] for r in sorted(mine, key=lambda r: r["mean_s"])]


def write_rows(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})


def print_rows(rows, title):
    print(f"\n{title}")
    print("  ".join(f"{k:>{max(len(k), 9)}s}" for k in rows[0]))
    for row in rows:
        print("  ".join(f"{(f'{v:.3f}' if isinstance(v, float) else str(v)):>{max(len(k), 9)}s}" for k, v in row.items()))


def plot(rows, lines, models, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, model in zip(axes[:2], models):
        mine = [r for r in rows if r["model"] == model]
        ax.bar([r["method"] for r in mine], [r["mean_s"] for r in mine], yerr=[r["std_s"] for r in mine], capsize=3)
        low = min(r["mean_s"] for r in mine if r["method"] != NAMES[("random", "")])
        ax.set_ylim(low - 0.4, max(r["mean_s"] for r in mine) + 0.2)
        ax.set_title(f"Mean final lap time, {model} model")
        ax.set_ylabel("Lap time (s)")
        ax.tick_params(axis="x", labelrotation=20)
    _draw_edges(axes[2], models["constant grip"].track)
    for (model, problem), colour in zip(models.items(), ("tab:blue", "tab:orange")):
        axes[2].plot(*problem.line(lines[model]), color=colour, lw=1.6, label=f"best line, {model}")
    axes[2].legend(loc="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(seeds=SEEDS, budget=BUDGET, workers=None, out=Path("results"), circuit=DEFAULT_CIRCUIT):
    out.mkdir(parents=True, exist_ok=True)
    rows, tests, lines, models = study(build_track(circuit), seeds, budget, workers)
    print(f"Circuit: {circuit}, {seeds} seeds, {budget:,} evaluations. Grip {LOW_GRIP:g} to {HIGH_GRIP:g} m/s^2 with speed.")
    print_rows(rows, "Mean final lap time by model (lap times are not comparable between models)")
    print_rows(tests, "Pairwise tests between the headline algorithms (Mann-Whitney U)")
    for model in models:
        print(f"Ranking, {model}: " + " < ".join(ranking(rows, model)))
    write_rows(rows, out / "downforce_run.csv")
    write_rows(tests, out / "downforce_tests.csv")
    plot(rows, lines, models, out / "downforce_run.png")
    return rows, tests


if __name__ == "__main__":
    main()