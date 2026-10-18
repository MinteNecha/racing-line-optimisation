"""Racing line optimisation problem: decision vector to geometry to lap time."""
import numpy as np
from scipy.interpolate import CubicSpline

from racing_line.config import KNOTS
from racing_line.vehicle import lap_times


class LineProblem:
    """A solution is a vector in [-1, 1]^dim of sideways offsets at evenly spaced knots.

    The offsets are joined by a periodic cubic spline and scaled to the usable track half width.
    """

    def __init__(self, track, knots=KNOTS):
        self.track = track
        self.dim = knots
        knot_t = np.linspace(0, 1, knots + 1)
        values = np.vstack([np.eye(knots), np.eye(knots)[:1]])
        s = np.arange(track.n) / track.n
        self.basis = CubicSpline(knot_t, values, bc_type="periodic")(s)

    def points(self, A):
        t = self.track
        off = np.clip(np.atleast_2d(A) @ self.basis.T, -1, 1) * t.limit
        return t.cx + t.nx * off, t.cy + t.ny * off

    def line(self, a):
        px, py = self.points(a)
        return px[0], py[0]

    def evaluate(self, A):
        return lap_times(*self.points(A))
