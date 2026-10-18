"""Point mass lap time model with a friction circle and forward/backward speed passes."""
import numpy as np
from numba import njit

from racing_line.config import A_BRAKE, A_LAT, A_TRACTION, DRAG, MASS, POWER, V_MAX


@njit(cache=True)
def geometry(px, py):
    """Segment lengths and discrete curvature (Menger) of a closed polyline."""
    n = len(px)
    seg = np.empty(n)
    kap = np.empty(n)
    for i in range(n):
        j = (i + 1) % n
        seg[i] = np.hypot(px[j] - px[i], py[j] - py[i])
    for i in range(n):
        h = (i - 1) % n
        j = (i + 1) % n
        ax, ay = px[i] - px[h], py[i] - py[h]
        bx, by = px[j] - px[i], py[j] - py[i]
        cx, cy = px[j] - px[h], py[j] - py[h]
        area2 = abs(ax * by - ay * bx)
        kap[i] = 2 * area2 / (np.hypot(ax, ay) * np.hypot(bx, by) * np.hypot(cx, cy) + 1e-12)
    return seg, kap


@njit(cache=True)
def _grip(v, k):
    """Fraction of grip left for braking or accelerating after cornering."""
    r = v * v * k / A_LAT
    return np.sqrt(max(0.0, 1.0 - r * r))


@njit(cache=True)
def profile(px, py):
    """Speed at every point and the segment lengths of a closed line."""
    n = len(px)
    seg, kap = geometry(px, py)
    v = np.empty(n)
    for i in range(n):
        v[i] = min(V_MAX, np.sqrt(A_LAT / (kap[i] + 1e-9)))
    for k in range(2 * n):
        i = k % n
        j = (i + 1) % n
        acc = min(A_TRACTION, POWER / (MASS * max(v[i], 1.0))) * _grip(v[i], kap[i])
        acc -= DRAG * v[i] * v[i]
        v[j] = min(v[j], np.sqrt(max(v[i] * v[i] + 2 * acc * seg[i], 0.0)))
    for k in range(2 * n):
        j = (n - k) % n
        i = (j - 1) % n
        dec = A_BRAKE * _grip(v[j], kap[j]) + DRAG * v[j] * v[j]
        v[i] = min(v[i], np.sqrt(v[j] * v[j] + 2 * dec * seg[i]))
    return v, seg


@njit(cache=True)
def lap_time(px, py):
    v, seg = profile(px, py)
    n = len(px)
    t = 0.0
    for i in range(n):
        t += 2 * seg[i] / (v[i] + v[(i + 1) % n])
    return t


@njit(cache=True)
def lap_times(PX, PY):
    """Lap time for each row of PX, PY."""
    out = np.empty(PX.shape[0])
    for r in range(PX.shape[0]):
        out[r] = lap_time(PX[r], PY[r])
    return out


def speed_profile(px, py):
    return profile(px, py)[0]
