"""Fine search of the genetic algorithm mutation rate.

The earlier sweep tried 0, 0.05, 0.075, 0.15, 0.225 and 0.3, so nothing between 0 and 0.05 was tested, while the
lecture notes suggest about 2% or less. This study fills that gap: the same GA (BLX crossover, default settings),
budget and seeds, with only the mutation rate changed, every rate compared with a reference rate (0.05, the project
default) using a Mann-Whitney U test. The default pin count is 80, the best formulation found in the pin study.

Kept in tests/ on purpose, so the main system is untouched. Run it from the project folder:
    python -m tests.mutation_study
    python -m tests.mutation_study 0.01 0.02 0.05 --knots 50 --seeds 10
The reference rate is always included in the run.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from racing_line.experiments.runner import finals, run_all
from racing_line.problem import LineProblem
from racing_line.track import CIRCUITS, DEFAULT_CIRCUIT, build_track

RATES = (0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075)
REFERENCE = 0.05
KNOTS = 80
SEEDS = 20
BUDGET = 50_000


def study(track, rates, reference, knots, seeds, budget, workers):
    configs = {rate: ("ga", {"pm": rate}) for rate in rates}
    done = {rate: finals(runs) for rate, runs in run_all(LineProblem(track, knots), configs, seeds, budget, workers).items()}
    base = done[reference]
    return [{"pm": rate, "genes_per_child": rate * knots, "mean_s": f.mean(), "std_s": f.std(), "best_s": f.min(),
             "change_vs_ref_s": f.mean() - base.mean(), "p_vs_ref": 1.0 if rate == reference else mannwhitneyu(f, base).pvalue}
            for rate, f in done.items()]


def write_rows(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})


def print_rows(rows, reference):
    print(f"{'pm':>7s}{'genes/child':>13s}{'mean':>9s}{'std':>8s}{'change':>9s}{'p':>8s}")
    for r in rows:
        mark = "  <- reference" if r["pm"] == reference else ("  *" if r["p_vs_ref"] < 0.05 else "")
        print(f"{r['pm']:7.3f}{r['genes_per_child']:13.2f}{r['mean_s']:9.3f}{r['std_s']:8.3f}{r['change_vs_ref_s']:9.3f}{r['p_vs_ref']:8.3f}{mark}")
    print(f"\ngenes/child = mutation rate x number of pins, change and p are against pm = {reference}, "
          "* = significant at the 5% level (negative change = faster)")


def plot(rows, reference, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar([r["pm"] for r in rows], [r["mean_s"] for r in rows], [r["std_s"] for r in rows], marker="o", capsize=3)
    ax.axvline(reference, color="gray", ls=":", lw=1)
    ax.set_xlabel("GA mutation rate per gene")
    ax.set_ylabel("Mean final lap time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(rates=RATES, seeds=SEEDS, budget=BUDGET, workers=None, out=Path("results"), circuit=DEFAULT_CIRCUIT,
         reference=REFERENCE, knots=KNOTS):
    out.mkdir(parents=True, exist_ok=True)
    rates = sorted(set(rates) | {reference})
    rows = study(build_track(circuit), rates, reference, knots, seeds, budget, workers)
    print(f"Circuit: {circuit}, {knots} pins, {seeds} seeds, {budget:,} evaluations per run, reference pm {reference}")
    print_rows(rows, reference)
    tag = f"{knots}pins_b{budget}"
    write_rows(rows, out / f"mutation_study_{tag}.csv")
    plot(rows, reference, out / f"mutation_study_{tag}.png")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rates", nargs="*", type=float, default=list(RATES), help="mutation rates to test")
    parser.add_argument("--reference", type=float, default=REFERENCE, help="rate every other rate is compared with")
    parser.add_argument("--knots", type=int, default=KNOTS)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--circuit", choices=sorted(CIRCUITS), default=DEFAULT_CIRCUIT)
    args = parser.parse_args()
    main(tuple(args.rates), args.seeds, args.budget, circuit=args.circuit, reference=args.reference, knots=args.knots)