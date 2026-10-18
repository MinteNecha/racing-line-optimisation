"""Command line entry point: runs every experiment and writes results/."""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from pathlib import Path

import numpy as np

from racing_line.baselines import reference_times
from racing_line.config import FULL, QUICK
from racing_line.experiments import plots, report
from racing_line.experiments.circuits import study
from racing_line.experiments.configs import main_configs, sweep_configs
from racing_line.experiments.runner import run_all
from racing_line.problem import LineProblem
from racing_line.track import CIRCUITS, DEFAULT_CIRCUIT, build_track


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small budget smoke test")
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: all cores)")
    parser.add_argument("--circuit", choices=list(CIRCUITS), default=DEFAULT_CIRCUIT,
                        help="circuit for the full benchmark (default: %(default)s)")
    parser.add_argument("--circuits", action="store_true",
                        help="instead of the full benchmark, compare the headline algorithms on every circuit")
    parser.add_argument("--out", type=Path, default=Path("results"),
                        help="output folder (a non default circuit writes to a subfolder named after it)")
    return parser.parse_args()


def benchmark(args, settings):
    out = args.out if args.circuit == DEFAULT_CIRCUIT else args.out / args.circuit
    out.mkdir(parents=True, exist_ok=True)

    prob = LineProblem(build_track(args.circuit))
    lines, base_t = reference_times(prob)

    runs = run_all(prob, main_configs(), settings.seeds, settings.budget, args.workers)
    rows = report.summary_rows(runs, base_t)
    print("Circuit:", args.circuit)
    report.print_summary(rows, base_t)
    report.print_significance(runs)
    report.write_summary(rows, out / "summary.csv")

    best = report.best_evolved(runs)
    print(f"Best evolved lap: {best.fitness:.3f} s")
    np.save(out / "best_line.npy", best.x)

    plots.convergence(runs, settings.budget, base_t, out / "convergence.png")
    plots.variations(runs, base_t, out / "variations.png")
    plots.track_lines(prob, lines, best.x, out / "track_lines.png")
    plots.speed_comparison(prob, lines, best.x, out / "speed_profile.png")

    sweeps = run_all(prob, sweep_configs(), settings.sweep_seeds, settings.budget, args.workers)
    plots.sweeps(sweeps, out / "sweeps.png")
    return out


def circuit_study(args, settings):
    args.out.mkdir(parents=True, exist_ok=True)
    results = study(list(CIRCUITS), settings.seeds, settings.budget, args.workers)
    rows = report.circuit_rows(results)
    report.print_circuits(rows)
    for name, (_, runs) in results.items():
        print("Significance on", name)
        report.print_significance(runs)
    report.write_circuits(rows, args.out / "circuits.csv")
    plots.circuits(rows, results, args.out / "circuits.png")
    return args.out


def main():
    args = parse_args()
    settings = QUICK if args.quick else FULL
    out = circuit_study(args, settings) if args.circuits else benchmark(args, settings)
    print("Done. Outputs in", out.resolve())


if __name__ == "__main__":
    main()
