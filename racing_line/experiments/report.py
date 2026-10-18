"""Tables and statistics for the report."""
import csv
from itertools import combinations

from scipy.stats import mannwhitneyu

from racing_line.experiments.configs import HEADLINE, NAMES
from racing_line.experiments.runner import finals


def summary_rows(runs, base_t):
    rows = []
    for key, results in runs.items():
        f = finals(results)
        rows.append((NAMES[key], f.mean(), f.std(), f.min(), f.max(), base_t["min_curvature"] - f.mean()))
    return rows


def write_summary(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "mean_s", "std_s", "best_s", "worst_s", "gain_vs_min_curvature_s"])
        for name, *values in rows:
            writer.writerow([name] + [f"{v:.4f}" for v in values])


def print_summary(rows, base_t):
    print("Baselines (s):", {k: round(v, 3) for k, v in base_t.items()})
    print(f"{'method':24s}{'mean':>9s}{'std':>8s}{'best':>9s}{'gain':>8s}")
    for name, mean, std, best, _, gain in rows:
        print(f"{name:24s}{mean:9.3f}{std:8.3f}{best:9.3f}{gain:8.3f}")


def print_significance(runs):
    for a, b in combinations(HEADLINE, 2):
        p = mannwhitneyu(finals(runs[a]), finals(runs[b])).pvalue
        print(f"Mann-Whitney {NAMES[a]} vs {NAMES[b]}: p = {p:.4f}")


def circuit_rows(results):
    """One row per circuit and method from the cross circuit study."""
    rows = []
    for circuit, (base_t, runs) in results.items():
        for key, seed_results in runs.items():
            f = finals(seed_results)
            gain = base_t["min_curvature"] - f.mean()
            rows.append({"circuit": circuit, "method": NAMES[key], "mean_s": f.mean(), "std_s": f.std(),
                         "best_s": f.min(), "min_curvature_s": base_t["min_curvature"],
                         "gain_s": gain, "gain_pct": 100 * gain / base_t["min_curvature"]})
    return rows


def write_circuits(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v if isinstance(v, str) else f"{v:.4f}" for k, v in row.items()})


def print_circuits(rows):
    print(f"{'circuit':14s}{'method':16s}{'mean':>9s}{'std':>8s}{'min curv.':>11s}{'gain':>8s}{'gain %':>8s}")
    for r in rows:
        print(f"{r['circuit']:14s}{r['method']:16s}{r['mean_s']:9.3f}{r['std_s']:8.3f}"
              f"{r['min_curvature_s']:11.3f}{r['gain_s']:8.3f}{r['gain_pct']:8.2f}")


def best_evolved(runs):
    """Best result among the evolutionary methods (random search excluded)."""
    candidates = [r for key, results in runs.items() if key[0] != "random" for r in results]
    return min(candidates, key=lambda r: r.fitness)
