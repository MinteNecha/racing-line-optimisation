"""Matplotlib dashboard that shows how the optimisers learn, built once and updated per frame."""
import io

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

from racing_line.vehicle import geometry, speed_profile

COLOURS = {"ga": "#1f77b4", "de": "#d95f02", "pso": "#2ca02c", "random": "#666666"}
TITLES = {"ga": "Genetic algorithm", "de": "Differential evolution",
          "pso": "Particle swarm", "random": "Random search"}
SPEED_RANGE = (80, 360)
KILO = FuncFormatter(lambda x, _: f"{x / 1000:g}k")
DIVERSITY_RANGE = (0.004, 1.0)
ZOOM_REACH = 90.0   # metres of track shown either side of the tightest corner
ZOOM_PAD = 6.0      # metres of margin around the track edges in the close-up


class Dashboard:
    """One algorithm: track, close-up, learning curve, diversity and speed. Several: side by side.

    budget and lowest fix the axes of the learning curve, which a run still in progress cannot know.
    Replace an entry of self.traces with a longer trace to follow a run as it unfolds.
    """

    def __init__(self, problem, lines, base_t, traces, focus=None, budget=None, lowest=None):
        self.problem, self.lines, self.base_t, self.traces = problem, lines, base_t, dict(traces)
        self.focus = focus if focus in traces else next(iter(traces))
        self.budget = budget or max(t.evals[-1] for t in traces.values())
        self.lowest = lowest if lowest is not None else min(t.best_f[-1] for t in traces.values()) - 0.15
        self._views = {algo: [] for algo in traces}   # (population lines, best line) per panel
        self._big, self._small, self._curve, self._dot = {}, {}, {}, {}
        self._diversity = None
        self._window = self._corner_window()
        if len(traces) == 1:
            self._layout_single()
        else:
            self._layout_compare()
        self.update(max(t.evals[0] for t in traces.values()))

    def _layout_single(self):
        self.fig = Figure(figsize=(14, 9.4), layout="constrained")
        grid = self.fig.add_gridspec(2, 12, height_ratios=[1.25, 1.0])
        self._build_track_panel(self.fig.add_subplot(grid[0, :7]), self.focus, scale=1.5)
        self._build_zoom_panel(self.fig.add_subplot(grid[0, 7:]), self.focus)
        self._build_convergence(self.fig.add_subplot(grid[1, :5]))
        self._build_diversity(self.fig.add_subplot(grid[1, 5:8]))
        self._build_speed(self.fig.add_subplot(grid[1, 8:]))

    def _layout_compare(self):
        n = len(self.traces)
        x0, x1, y0, y1 = self._window
        zoom_height = (14 / n) * 0.85 * (y1 - y0) / (x1 - x0) + 0.6
        heights = [3.2, float(np.clip(zoom_height, 1.6, 4.2)), 4.2]
        self.fig = Figure(figsize=(14, sum(heights) + 0.6), layout="constrained")
        grid = self.fig.add_gridspec(3, 12, height_ratios=heights)
        width = 12 // n
        scale = 1.0 if n <= 3 else 0.8
        for i, algo in enumerate(self.traces):
            cols = slice(i * width, (i + 1) * width)
            self._build_track_panel(self.fig.add_subplot(grid[0, cols]), algo, scale)
            self._build_zoom_panel(self.fig.add_subplot(grid[1, cols]), algo)
        self._build_convergence(self.fig.add_subplot(grid[2, :7]))
        self._build_speed(self.fig.add_subplot(grid[2, 7:]))

    def _corner_window(self):
        """Bounding box (x0, x1, y0, y1) around the stretch of track containing the tightest corner."""
        track = self.problem.track
        tightest = int(geometry(track.cx, track.cy)[1].argmax())
        reach = int(ZOOM_REACH / (track.length / track.n))
        idx = np.arange(tightest - reach, tightest + reach + 1) % track.n
        pad = track.width / 2 + ZOOM_PAD
        return (track.cx[idx].min() - pad, track.cx[idx].max() + pad,
                track.cy[idx].min() - pad, track.cy[idx].max() + pad)

    def _draw_track(self, ax, algo):
        track, colour = self.problem.track, COLOURS[algo]
        for side in (-1, 1):
            ax.plot(track.cx + side * track.nx * track.width / 2,
                    track.cy + side * track.ny * track.width / 2, color="black", lw=0.8)
        ax.plot(*self.problem.line(self.lines["min_curvature"]), "--", color="black", lw=1, alpha=0.6)
        population = [ax.plot([], [], color=colour, lw=1, alpha=0.3)[0]
                      for _ in range(self.traces[algo].sample_x.shape[1])]
        best, = ax.plot([], [], color=colour, lw=2.4)
        self._views[algo].append((population, best))
        ax.set_aspect("equal")
        ax.axis("off")

    def _build_track_panel(self, ax, algo, scale):
        colour = COLOURS[algo]
        self._draw_track(ax, algo)
        x0, x1, y0, y1 = self._window
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="gray", lw=1.2))
        big = 17 * scale
        self._big[algo] = self._caption(ax, -4, big, fontweight="bold", color=colour)
        self._small[algo] = self._caption(ax, -4 - 1.5 * big, 8.5 * scale, color="#333333")
        ax.set_title(TITLES[algo], color=colour, fontweight="bold")

    @staticmethod
    def _caption(ax, offset, size, **style):
        """Text centred under the axes, offset in points so it never covers the track."""
        return ax.annotate("", (0.5, 0), xytext=(0, offset), xycoords="axes fraction", textcoords="offset points",
                           ha="center", va="top", fontsize=size, **style)

    def _build_zoom_panel(self, ax, algo):
        self._draw_track(ax, algo)
        x0, x1, y0, y1 = self._window
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title("Close-up of the tightest corner", fontsize=9, color="#555555")

    def _build_convergence(self, ax):
        for algo in self.traces:
            self._curve[algo], = ax.plot([], [], color=COLOURS[algo], lw=2, label=TITLES[algo])
            self._dot[algo], = ax.plot([], [], "o", color=COLOURS[algo], ms=6)
        ax.axhline(self.base_t["min_curvature"], color="black", ls="--", lw=1, label="Minimum curvature line")
        ax.axhline(self.base_t["centreline"], color="gray", ls=":", lw=1, label="Centreline")
        highest = max(self.base_t["centreline"], *(t.best_f[0] for t in self.traces.values()))
        ax.set_xlim(0, self.budget)
        ax.set_ylim(self.lowest, highest + 0.15)
        ax.xaxis.set_major_formatter(KILO)
        ax.set_xlabel("Fitness evaluations")
        ax.set_ylabel("Best lap time so far (s)")
        ax.set_title("Learning curve")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    def _build_diversity(self, ax):
        colour = COLOURS[self.focus]
        self._diversity, = ax.plot([], [], color=colour, lw=2)
        self._diversity_dot, = ax.plot([], [], "o", color=colour, ms=6)
        ax.set_yscale("log")
        ax.set_xlim(0, self.budget)
        ax.set_ylim(*DIVERSITY_RANGE)
        ax.xaxis.set_major_formatter(KILO)
        ax.set_xlabel("Fitness evaluations")
        ax.set_ylabel("Spread of the population")
        ax.set_title("Diversity: explore, then settle")
        ax.grid(alpha=0.25, which="both")

    def _build_speed(self, ax):
        n, length = self.problem.track.n, self.problem.track.length
        self._distance = np.arange(n) * length / n
        reference = speed_profile(*self.problem.line(self.lines["min_curvature"])) * 3.6
        ax.plot(self._distance, reference, color="black", ls="--", lw=1, label="Minimum curvature line")
        self._speed, = ax.plot([], [], color=COLOURS[self.focus], lw=2, label=TITLES[self.focus])
        ax.set_ylim(*SPEED_RANGE)
        ax.set_xlim(0, length)
        ax.set_xlabel("Distance along the lap (m)")
        ax.set_ylabel("Speed (km/h)")
        ax.set_title("Speed of the best line" if len(self.traces) == 1
                     else f"Speed profile of the best line: {TITLES[self.focus]}")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)

    def update(self, evals):
        """Move every panel to the state each run had reached after the given evaluations."""
        for algo, trace in self.traces.items():
            k = trace.frame_at(evals)
            xs, ys = self.problem.points(trace.sample_x[k])
            best_xy = self.problem.line(trace.best_x[k])
            for population, best in self._views[algo]:
                for line, x, y in zip(population, xs, ys):
                    line.set_data(x, y)
                best.set_data(*best_xy)
            best_f = trace.best_f[k]
            self._big[algo].set_text(f"{best_f:.3f} s")
            self._small[algo].set_text(f"{self.base_t['min_curvature'] - best_f:+.3f} s vs min curvature\n"
                                       f"{int(trace.evals[k]):,} evaluations")
            self._curve[algo].set_data(trace.evals[:k + 1], trace.best_f[:k + 1])
            self._dot[algo].set_data([trace.evals[k]], [best_f])
        focus = self.traces[self.focus]
        k = focus.frame_at(evals)
        if self._diversity is not None:
            self._diversity.set_data(focus.evals[:k + 1], focus.diversity[:k + 1])
            self._diversity_dot.set_data([focus.evals[k]], [focus.diversity[k]])
        self._speed.set_data(self._distance, speed_profile(*self.problem.line(focus.best_x[k])) * 3.6)
        return self.fig

    def png(self, evals, dpi=100):
        """The dashboard at the given evaluation count as PNG bytes."""
        buffer = io.BytesIO()
        self.update(evals).savefig(buffer, format="png", dpi=dpi)
        return buffer.getvalue()
