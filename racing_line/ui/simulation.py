"""Run optimisers with a recorder attached and build the replay timeline."""
import numpy as np

from racing_line.optimisers import OPTIMISERS
from racing_line.recording import Recorder
from racing_line.ui.rendering import Dashboard


def simulate(problem, algo, params, budget, seed, sample=12):
    """Run one optimiser and return its Trace. params is a dict of optimiser keyword arguments."""
    recorder = Recorder(sample)
    OPTIMISERS[algo](problem, budget, np.random.default_rng(seed), observer=recorder, **params)
    return recorder.trace()


def timeline(traces, count=80):
    """Evaluation counts to replay, dense early on where most of the learning happens."""
    first = max(t.evals[0] for t in traces.values())
    last = min(t.evals[-1] for t in traces.values())
    grid = np.unique(np.round(np.geomspace(first, last, count)).astype(int))
    return [int(e) for e in grid]


def evals_to_beat(trace, target):
    """Evaluations needed to get a lap time below the target, or None if never reached."""
    hit = np.nonzero(trace.best_f < target)[0]
    return int(trace.evals[hit[0]]) if len(hit) else None


def single_view_limits(base_t, budget):
    """Fixed learning curve axes for one algorithm, so a run in progress and its replay match."""
    return {"budget": budget, "lowest": base_t["min_curvature"] - 1.3}


def train_live(problem, lines, base_t, algo, params, budget, seed, draw, frames=40, dpi=90, sample=12):
    """Train one optimiser and call draw(png_bytes, evals) at a geometric series of points.

    Returns the full Trace. Early frames are close together because most learning happens early.
    """
    recorder = Recorder(sample)
    marks = np.geomspace(params.get("pop", 60), budget, frames)
    state = {"dashboard": None, "next": 0}

    def render(evals):
        trace = recorder.trace()
        if state["dashboard"] is None:
            state["dashboard"] = Dashboard(problem, lines, base_t, {algo: trace},
                                           **single_view_limits(base_t, budget))
        else:
            state["dashboard"].traces[algo] = trace
        draw(state["dashboard"].png(evals, dpi), evals)

    def observer(evals, X, f):
        recorder(evals, X, f)
        if state["next"] < len(marks) and evals >= marks[state["next"]]:
            while state["next"] < len(marks) and marks[state["next"]] <= evals:
                state["next"] += 1
            render(evals)

    OPTIMISERS[algo](problem, budget, np.random.default_rng(seed), observer=observer, **params)
    trace = recorder.trace()
    render(int(trace.evals[-1]))
    return trace
