import numpy as np
import pytest

from racing_line.baselines import baselines
from racing_line.problem import LineProblem
from racing_line.track import build_track


@pytest.fixture(scope="session")
def track():
    return build_track()


@pytest.fixture(scope="session")
def problem(track):
    return LineProblem(track)


@pytest.fixture(scope="session")
def reference_lines(problem):
    return baselines(problem)


@pytest.fixture
def circle():
    def make(radius, n=400):
        angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return radius * np.cos(angle), radius * np.sin(angle)
    return make
