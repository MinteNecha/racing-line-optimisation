"""Capture how a run unfolds, generation by generation, for replay in the dashboard."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Trace:
    evals: np.ndarray      # (frames,) evaluations used at each frame
    best_f: np.ndarray     # (frames,) best lap time so far
    best_x: np.ndarray     # (frames, dim) best solution so far
    sample_x: np.ndarray   # (frames, sample, dim) evenly spread slice of the population
    diversity: np.ndarray  # (frames,) average spread of the population along each decision variable

    @property
    def frames(self):
        return len(self.evals)

    def frame_at(self, evals):
        """Index of the last frame recorded at or before the given evaluation count."""
        return max(int(np.searchsorted(self.evals, evals, side="right")) - 1, 0)


class Recorder:
    """Observer for the optimisers: stores the population and the best solution each generation."""

    def __init__(self, sample=12):
        self.sample = sample
        self._evals, self._best_f, self._best_x, self._sample_x, self._diversity = [], [], [], [], []
        self._best = (np.inf, None)

    def __call__(self, evals, X, f):
        i = int(f.argmin())
        if f[i] < self._best[0]:
            self._best = (float(f[i]), X[i].copy())
        picks = np.linspace(0, len(X) - 1, self.sample).astype(int)
        self._evals.append(evals)
        self._best_f.append(self._best[0])
        self._best_x.append(self._best[1])
        self._sample_x.append(X[picks].copy())
        self._diversity.append(float(X.std(axis=0).mean()))

    def trace(self):
        return Trace(np.array(self._evals), np.array(self._best_f),
                     np.array(self._best_x), np.array(self._sample_x), np.array(self._diversity))
