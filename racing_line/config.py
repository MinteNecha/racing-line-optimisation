"""Car model constants and experiment settings."""
from dataclasses import dataclass

MASS = 800.0            # kg
POWER = 750e3           # W
A_TRACTION = 12.0       # m/s^2, traction limited acceleration
A_BRAKE = 45.0          # m/s^2, braking deceleration
A_LAT = 38.0            # m/s^2, lateral grip limit
V_MAX = 95.0            # m/s
DRAG = 0.5 * 1.2 * 1.0 / MASS   # 0.5 * air density * CdA / mass
CAR_HALF_WIDTH = 0.9    # m

TRACK_WIDTH = 14.0      # m
TRACK_POINTS = 400      # samples along the centreline
KNOTS = 50              # decision variables per solution


@dataclass(frozen=True)
class Settings:
    budget: int         # fitness evaluations per run
    seeds: int          # independent runs per configuration
    sweep_seeds: int    # independent runs per parameter sweep cell


FULL = Settings(budget=50_000, seeds=10, sweep_seeds=5)
QUICK = Settings(budget=6_000, seeds=3, sweep_seeds=2)
