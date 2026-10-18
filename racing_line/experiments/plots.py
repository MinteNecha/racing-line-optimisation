"""Figures for the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from racing_line.experiments.configs import NAMES, SWEEP_GRIDS
from racing_line.experiments.report import best_evolved
from racing_line.experiments.runner import finals
from racing_line.problem import LineProblem
from racing_line.track import CIRCUIT_TITLES, build_track
from racing_line.vehicle import speed_profile

DPI = 200


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _draw_edges(ax, track):
    for side in (-1, 1):
        ax.plot(track.cx + side * track.nx * track.width / 2,
                track.cy + side * track.ny * track.width / 2, "k", lw=0.8)
    ax.set_aspect("equal")
    ax.axis("off")


def convergence(runs, budget, base_t, path):
    grid = np.linspace(budget / 50, budget, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in [("ga", "blx"), ("de", "rand1"), ("pso", "gbest"), ("random", "")]:
        curves = np.array([np.interp(grid, r.history[:, 0], r.history[:, 1]) for r in runs[key]])
        mean, std = curves.mean(0), curves.std(0)
        ax.plot(grid, mean, label=NAMES[key])
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
    ax.axhline(base_t["min_curvature"], color="k", ls="--", lw=1, label="Minimum curvature baseline")
    ax.axhline(base_t["centreline"], color="gray", ls=":", lw=1, label="Centreline baseline")
    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("Best lap time (s)")
    lowest = min(r.fitness for results in runs.values() for r in results)
    ax.set_ylim(lowest - 0.3, base_t["centreline"] + 0.5)
    ax.legend()
    _save(fig, path)


def variations(runs, base_t, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, algo in zip(axes, ["ga", "de", "pso"]):
        keys = [k for k in runs if k[0] == algo]
        ax.boxplot([finals(runs[k]) for k in keys], tick_labels=[NAMES[k] for k in keys])
        ax.axhline(base_t["min_curvature"], color="k", ls="--", lw=1)
        ax.set_ylabel("Final lap time (s)")
        ax.tick_params(axis="x", labelrotation=15)
    _save(fig, path)


def sweeps(runs, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (algo, (name_a, values_a, name_b, values_b)) in zip(axes, SWEEP_GRIDS.items()):
        grid = np.array([[finals(runs[(algo, a, b)]).mean() for b in values_b] for a in values_a])
        image = ax.imshow(grid, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(values_b)), values_b)
        ax.set_yticks(range(len(values_a)), values_a)
        ax.set_xlabel(name_b)
        ax.set_ylabel(name_a)
        ax.set_title(f"{algo.upper()}: mean final lap time (s)")
        for i in range(len(values_a)):
            for j in range(len(values_b)):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(image, ax=ax)
    _save(fig, path)


def track_lines(prob, lines, best_x, path):
    px, py = prob.line(best_x)
    speed = speed_profile(px, py) * 3.6
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax in axes:
        _draw_edges(ax, prob.track)
    styles = {"centreline": ("gray", ":"), "shortest": ("green", "-"), "min_curvature": ("tab:blue", "-")}
    for name, (colour, style) in styles.items():
        axes[0].plot(*prob.line(lines[name]), color=colour, ls=style, lw=1.2, label=name.replace("_", " "))
    axes[0].plot(px, py, color="tab:red", lw=1.5, label="best evolved line")
    axes[0].legend(loc="center")
    pts = np.vstack([np.column_stack([px, py]), [px[0], py[0]]])
    collection = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap="plasma", linewidth=3)
    collection.set_array(speed)
    axes[1].add_collection(collection)
    fig.colorbar(collection, ax=axes[1], label="Speed (km/h)", shrink=0.8)
    _save(fig, path)


def speed_comparison(prob, lines, best_x, path):
    n, length = prob.track.n, prob.track.length
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, x in [("Minimum curvature", lines["min_curvature"]), ("Best evolved line", best_x)]:
        ax.plot(np.arange(n) * length / n, speed_profile(*prob.line(x)) * 3.6, label=label)
    ax.set_xlabel("Distance along track (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend()
    _save(fig, path)


def circuits(rows, results, path):
    """Top: each circuit with its best evolved line. Bottom: gain over the minimum curvature line."""
    names = list(results)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, len(names), height_ratios=[1, 1.3])
    for i, name in enumerate(names):
        ax = fig.add_subplot(grid[0, i])
        problem = LineProblem(build_track(name))
        _draw_edges(ax, problem.track)
        ax.plot(*problem.line(best_evolved(results[name][1]).x), color="tab:red", lw=1.5)
        ax.set_title(CIRCUIT_TITLES[name], fontsize=10)
    ax = fig.add_subplot(grid[1, :])
    width = 0.8 / len(methods)
    for j, method in enumerate(methods):
        chosen = [r for r in rows if r["method"] == method]
        ax.bar(np.arange(len(names)) + j * width, [r["gain_pct"] for r in chosen], width, label=method,
               yerr=[100 * r["std_s"] / r["min_curvature_s"] for r in chosen], capsize=3)
    ax.set_xticks(np.arange(len(names)) + width * (len(methods) - 1) / 2, [CIRCUIT_TITLES[n] for n in names])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Gain over minimum curvature line (% of lap time)")
    ax.legend()
    _save(fig, path)
