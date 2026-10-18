"""Closed circuit geometry sampled at equal arc length."""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from racing_line.config import CAR_HALF_WIDTH, TRACK_POINTS, TRACK_WIDTH

CIRCUITS = {
    "grand_prix": [
        (0, 0), (450, -10), (900, 0), (1080, 10), (1190, 70), (1200, 150),
        (1120, 200), (980, 190), (900, 235), (830, 185), (760, 245), (770, 340),
        (880, 420), (1050, 450), (1190, 510), (1210, 640), (1110, 730),
        (900, 750), (700, 700), (590, 625), (450, 690), (250, 760), (60, 700),
        (-60, 570), (-110, 400), (-40, 250), (-70, 100),
    ],
    "technical": [
        (0, 0), (160, -5), (320, 0), (430, 30), (470, 100), (430, 160), (340, 175),
        (300, 225), (350, 275), (440, 290), (500, 350), (470, 430), (380, 460),
        (290, 440), (240, 380), (160, 365), (90, 410), (10, 395), (-45, 330),
        (-30, 260), (40, 225), (35, 160), (-40, 120), (-50, 50),
    ],
    "flowing": [
        (0, 0), (500, 0), (1000, -20), (1450, 0), (1720, 110), (1800, 320),
        (1680, 500), (1430, 540), (1180, 450), (980, 500), (840, 650), (580, 700),
        (320, 640), (170, 490), (-40, 440), (-170, 290), (-130, 110),
    ],
}
CIRCUIT_TITLES = {
    "grand_prix": "Grand Prix (4.6 km, mixed)",
    "technical": "Technical (2.2 km, tight hairpins)",
    "flowing": "Flowing (4.7 km, fast sweepers)",
}
DEFAULT_CIRCUIT = "grand_prix"


@dataclass
class Track:
    cx: np.ndarray
    cy: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    length: float
    width: float
    name: str = DEFAULT_CIRCUIT

    @property
    def n(self):
        return len(self.cx)

    @property
    def limit(self):
        """Largest sideways offset from the centreline that keeps the car on the track."""
        return self.width / 2 - CAR_HALF_WIDTH


def build_track(name=DEFAULT_CIRCUIT, n=TRACK_POINTS, width=TRACK_WIDTH):
    pts = np.array(CIRCUITS[name], dtype=float)
    closed = np.vstack([pts, pts[:1]])
    chord = np.r_[0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
    spline = CubicSpline(chord, closed, bc_type="periodic")
    dense = np.linspace(0, chord[-1], 20000, endpoint=False)
    xy = spline(dense)
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T))]
    u = np.interp(np.linspace(0, arc[-1], n, endpoint=False), arc[:-1], dense)
    cx, cy = spline(u).T
    tx, ty = spline(u, 1).T
    norm = np.hypot(tx, ty)
    return Track(cx, cy, -ty / norm, tx / norm, arc[-1], width, name)
