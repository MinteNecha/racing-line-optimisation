"""How many spline pins (knots) should a racing line have?

The standard problem uses 50 pins. Here the same optimisers, settings, budget and seeds run with other pin counts,
and every count is compared with a reference count (50 unless told otherwise) using a Mann-Whitney U test.
Lap times from different counts are comparable, because the judge and the track are identical.
Only the number of searched values changes.

Kept in tests/ on purpose, so the main system is untouched. Run it from the project folder:
    python -m tests.knots_study                                  (40, 50 and 60 pins against 50)
    python -m tests.knots_study 70 80 90 --reference 80 --seeds 20
    python -m tests.knots_study 80 100 120 --reference 80 --budget 200000
The reference count is always included in the run.
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

from racing_line.config import KNOTS
from racing_line.experiments.configs import HEADLINE, NAMES, main_configs
from racing_line.experiments.runner import finals, run_all
from racing_line.problem import LineProblem
from racing_line.track import CIRCUITS, DEFAULT_CIRCUIT, build_track

GRID = (40, 50, 60)
SEEDS = 10
BUDGET = 50_000


def study(track, knots, reference, seeds, budget, workers):
    configs = {key: main_configs()[key] for key in HEADLINE}
    finals_by_knots = {k: {key: finals(runs) for key, runs in run_all(LineProblem(track, k), configs, seeds, budget, workers).items()}
                       for k in knots}
    rows = []
    for k in knots:
        for key in HEADLINE:
            f, base = finals_by_knots[k][key], finals_by_knots[reference][key]
            rows.append({"knots": k, "spacing_m": track.length / k, "method": NAMES[key], "mean_s": f.mean(), "std_s": f.std(),
                         "best_s": f.min(), "change_vs_ref_s": f.mean() - base.mean(),
                         "p_vs_ref": 1.0 if k == reference else mannwhitneyu(f, base).pvalue})
    return rows


def write_rows(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})


def print_rows(rows, reference):
    for method in dict.fromkeys(r["method"] for r in rows):
        print(f"\n{method}")
        print(f"{'pins':>6s}{'spacing':>9s}{'mean':>9s}{'std':>8s}{'change':>9s}{'p':>8s}")
        for r in (r for r in rows if r["method"] == method):
            mark = "  <- reference" if r["knots"] == reference else ("  *" if r["p_vs_ref"] < 0.05 else "")
            print(f"{r['knots']:6d}{r['spacing_m']:9.1f}{r['mean_s']:9.3f}{r['std_s']:8.3f}{r['change_vs_ref_s']:9.3f}{r['p_vs_ref']:8.3f}{mark}")
    print(f"\nspacing = metres between pins, change and p are against {reference} pins, * = significant at the 5% level (negative change = faster)")


def plot(rows, reference, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in dict.fromkeys(r["method"] for r in rows):
        mine = [r for r in rows if r["method"] == method]
        ax.errorbar([r["knots"] for r in mine], [r["mean_s"] for r in mine], [r["std_s"] for r in mine], marker="o", capsize=3, label=method)
    ax.axvline(reference, color="gray", ls=":", lw=1)
    ax.set_xlabel("Number of pins (knots) in the racing line")
    ax.set_ylabel("Mean final lap time (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(knots=GRID, seeds=SEEDS, budget=BUDGET, workers=None, out=Path("results"), circuit=DEFAULT_CIRCUIT, reference=KNOTS):
    out.mkdir(parents=True, exist_ok=True)
    knots = sorted(set(knots) | {reference})
    tag = "_".join(map(str, knots)) + f"_b{budget}"
    rows = study(build_track(circuit), knots, reference, seeds, budget, workers)
    print(f"Circuit: {circuit}, {seeds} seeds, {budget:,} evaluations per run, pins {knots}, reference {reference}")
    print_rows(rows, reference)
    write_rows(rows, out / f"knots_study_{tag}.csv")
    plot(rows, reference, out / f"knots_study_{tag}.png")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("knots", nargs="*", type=int, default=list(GRID), help="pin counts to test")
    parser.add_argument("--reference", type=int, default=KNOTS, help="pin count every other count is compared with")
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--circuit", choices=sorted(CIRCUITS), default=DEFAULT_CIRCUIT)
    args = parser.parse_args()
    main(tuple(args.knots), args.seeds, args.budget, circuit=args.circuit, reference=args.reference)