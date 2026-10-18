"""One-at-a-time study of the algorithm settings that the standard benchmark does not sweep.

Each setting is varied while everything else keeps its default, 5 seeds per value, and every value is compared
with the default run using a Mann-Whitney U test. Kept in tests/ on purpose, so the main system is untouched.
Run it from the project folder:  python -m tests.parameter_study
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from racing_line.experiments.runner import finals, run_all
from racing_line.problem import LineProblem
from racing_line.track import DEFAULT_CIRCUIT, build_track
from racing_line.ui.parameters import PARAMS, defaults

SEEDS = 5
BUDGET = 50_000
GRIDS = {
    "ga": {"pc": [0.5, 0.7, 0.9, 1.0], "sigma": [0.02, 0.05, 0.1, 0.2, 0.4],
           "elite": [0, 2, 5, 10], "k": [2, 3, 5, 7]},
    "de": {"pop": [20, 40, 60, 120, 250, 500]},
    "pso": {"c1": [0.5, 1.0, 1.49, 2.0], "c2": [0.5, 1.0, 1.49, 2.0], "vmax": [0.1, 0.25, 0.5, 1.0]},
}
NAMES = {"ga": "GA", "de": "DE", "pso": "PSO"}


def label(algo, param):
    return next(p.label for p in PARAMS[algo] if p.name == param)


def build_configs(grids=GRIDS):
    """{key: (algorithm, settings)}. The default run is shared by every setting of that algorithm."""
    configs = {}
    for algo, grid in grids.items():
        configs[(algo, "default")] = (algo, {})
        for param, values in grid.items():
            for value in values:
                if value != defaults(algo)[param]:
                    configs[(algo, param, value)] = (algo, {param: value})
    return configs


def study(configs, grids, seeds, budget, workers, circuit=DEFAULT_CIRCUIT):
    runs = run_all(LineProblem(build_track(circuit)), configs, seeds, budget, workers)
    rows = []
    for algo, grid in grids.items():
        reference = finals(runs[(algo, "default")])
        for param, values in grid.items():
            for value in values:
                is_default = value == defaults(algo)[param]
                f = finals(runs[(algo, "default") if is_default else (algo, param, value)])
                p = 1.0 if is_default else mannwhitneyu(f, reference).pvalue
                rows.append({"algorithm": NAMES[algo], "setting": label(algo, param), "value": value,
                             "is_default": is_default, "mean_s": f.mean(), "std_s": f.std(), "best_s": f.min(),
                             "change_vs_default_s": f.mean() - reference.mean(), "p_vs_default": p})
    return rows


def write_rows(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})


def print_rows(rows):
    current = None
    for r in rows:
        if (r["algorithm"], r["setting"]) != current:
            current = (r["algorithm"], r["setting"])
            print(f"\n{r['algorithm']}: {r['setting']}")
            print(f"{'value':>8s}{'mean':>9s}{'std':>8s}{'change':>9s}{'p':>8s}")
        mark = "  <- default" if r["is_default"] else ("  *" if r["p_vs_default"] < 0.05 else "")
        print(f"{r['value']:>8}{r['mean_s']:9.3f}{r['std_s']:8.3f}{r['change_vs_default_s']:9.3f}"
              f"{r['p_vs_default']:8.3f}{mark}")
    print("\n* = significantly different from the default at the 5% level (negative change = faster)")


def plot(rows, path):
    panels = list(dict.fromkeys((r["algorithm"], r["setting"]) for r in rows))
    columns = 4
    rows_needed = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows_needed, columns, figsize=(16, 3.6 * rows_needed), sharey=True, squeeze=False)
    for ax, (algorithm, setting) in zip(axes.flat, panels):
        mine = [r for r in rows if (r["algorithm"], r["setting"]) == (algorithm, setting)]
        x = range(len(mine))
        ax.errorbar(x, [r["mean_s"] for r in mine], [r["std_s"] for r in mine], marker="o", capsize=3)
        default = next(i for i, r in enumerate(mine) if r["is_default"])
        ax.axvline(default, color="gray", ls=":", lw=1)
        ax.set_xticks(list(x), [str(r["value"]) for r in mine])
        ax.set_title(f"{algorithm}: {setting}", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean final lap time (s)")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(grids=GRIDS, seeds=SEEDS, budget=BUDGET, workers=None, out=Path("results"), circuit=DEFAULT_CIRCUIT):
    out.mkdir(parents=True, exist_ok=True)
    rows = study(build_configs(grids), grids, seeds, budget, workers, circuit)
    print(f"Circuit: {circuit}, {seeds} seeds, {budget:,} evaluations per run")
    print_rows(rows)
    write_rows(rows, out / "parameter_study.csv")
    plot(rows, out / "parameter_study.png")
    return rows


if __name__ == "__main__":
    main()