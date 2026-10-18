"""Run the headline algorithms on every circuit to see how far the results carry over."""
from racing_line.baselines import reference_times
from racing_line.experiments.configs import study_configs
from racing_line.experiments.runner import run_all
from racing_line.problem import LineProblem
from racing_line.track import build_track


def study(circuits, seeds, budget, workers=None):
    """Return {circuit: (reference lap times, {label: [Result per seed]})}."""
    results = {}
    for name in circuits:
        problem = LineProblem(build_track(name))
        _, base_t = reference_times(problem)
        results[name] = (base_t, run_all(problem, study_configs(), seeds, budget, workers))
    return results
