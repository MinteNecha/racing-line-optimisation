"""Identical one-at-a-time sweep of every setting of GA, DE and PSO.

The grid is generated from the dashboard ranges, so no algorithm is treated differently:
  numeric setting -> 5 evenly spaced values across its full allowed range, plus the default
                     (values closer to the default than 1/20 of the range are dropped as near duplicates)
  choice setting  -> every option
  population size -> one shared grid for all three algorithms
Everything else stays at its default, 5 seeds per value, each value tested against that algorithm's default run
with a Mann-Whitney U test. Run from the project folder:  python -m tests.sweep_study
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
from pathlib import Path

import numpy as np

try:
    from tests import parameter_study as ps
except ImportError:
    import parameter_study as ps

from racing_line.track import DEFAULT_CIRCUIT
from racing_line.ui.parameters import PARAMS, defaults

LEVELS = 5
POPULATIONS = [20, 40, 60, 120, 250, 500]
ALGORITHMS = ("ga", "de", "pso")


def values_for(param, default):
    if param.options:
        return list(param.options)
    if param.name == "pop":
        return sorted(set(POPULATIONS) | {default})
    spread = (param.high - param.low) / 20
    grid = [v for v in np.linspace(param.low, param.high, LEVELS) if abs(v - default) >= spread]
    if isinstance(default, int):
        return sorted({int(round(v)) for v in grid} | {default})
    return sorted({round(float(v), 4) for v in grid} | {default})


def make_grids(algorithms=ALGORITHMS):
    return {algo: {p.name: values_for(p, defaults(algo)[p.name]) for p in PARAMS[algo]} for algo in algorithms}


def sensitivity(rows):
    """One line per setting: how much the mean lap time moves across its tested values."""
    summary = []
    for key in dict.fromkeys((r["algorithm"], r["setting"]) for r in rows):
        mine = [r for r in rows if (r["algorithm"], r["setting"]) == key]
        best = min(mine, key=lambda r: r["mean_s"])
        worst = max(mine, key=lambda r: r["mean_s"])
        summary.append({"algorithm": key[0], "setting": key[1], "values_tested": len(mine),
                        "spread_s": worst["mean_s"] - best["mean_s"], "best_value": best["value"],
                        "best_mean_s": best["mean_s"], "worst_value": worst["value"],
                        "significant_values": sum(r["p_vs_default"] < 0.05 for r in mine)})
    return sorted(summary, key=lambda s: (s["algorithm"], -s["spread_s"]))


def print_sensitivity(summary):
    print("\nSensitivity ranking inside each algorithm (spread = slowest minus fastest mean lap time)")
    print(f"{'algo':5s}{'setting':26s}{'spread':>8s}{'best':>9s}{'sig.':>6s}")
    for s in summary:
        print(f"{s['algorithm']:5s}{s['setting']:26s}{s['spread_s']:8.3f}{str(s['best_value']):>9s}"
              f"{s['significant_values']:6d}")


def main(grids=None, seeds=ps.SEEDS, budget=ps.BUDGET, workers=None, out=Path("results"), circuit=DEFAULT_CIRCUIT):
    out.mkdir(parents=True, exist_ok=True)
    grids = grids or make_grids()
    rows = ps.study(ps.build_configs(grids), grids, seeds, budget, workers, circuit)
    summary = sensitivity(rows)
    print(f"Circuit: {circuit}, {seeds} seeds, {budget:,} evaluations per run")
    ps.print_rows(rows)
    print_sensitivity(summary)
    ps.write_rows(rows, out / "sweep_study.csv")
    ps.write_rows(summary, out / "sweep_sensitivity.csv")
    ps.plot(rows, out / "sweep_study.png")
    return rows


if __name__ == "__main__":
    main()