import numpy as np
import pytest

from racing_line.config import CAR_HALF_WIDTH
from racing_line.track import CIRCUIT_TITLES, CIRCUITS, DEFAULT_CIRCUIT, build_track
from racing_line.vehicle import geometry


@pytest.fixture(scope="module", params=list(CIRCUITS))
def track(request):
    return build_track(request.param)


def test_centreline_points_are_evenly_spaced(track):
    seg, _ = geometry(track.cx, track.cy)
    assert seg.std() / seg.mean() < 0.01
    assert np.isclose(seg.sum(), track.length, rtol=1e-3)


def test_normals_are_unit_and_perpendicular_to_the_track(track):
    assert np.allclose(np.hypot(track.nx, track.ny), 1.0)
    tx = np.roll(track.cx, -1) - np.roll(track.cx, 1)
    ty = np.roll(track.cy, -1) - np.roll(track.cy, 1)
    dot = (tx * track.nx + ty * track.ny) / np.hypot(tx, ty)
    assert np.abs(dot).max() < 0.05


def test_tightest_corner_is_wider_than_half_the_track(track):
    _, kap = geometry(track.cx, track.cy)
    assert 1 / kap.max() > track.width / 2


def test_usable_half_width_accounts_for_the_car(track):
    assert np.isclose(track.limit, track.width / 2 - CAR_HALF_WIDTH)
    assert track.limit > 0


def test_no_part_of_the_circuit_comes_within_a_track_width_of_another_part(track):
    gap = int(2 * track.width / (track.length / track.n))
    distance = np.hypot(track.cx[:, None] - track.cx[None], track.cy[:, None] - track.cy[None])
    index = np.arange(track.n)
    apart = np.abs(index[:, None] - index[None])
    apart = np.minimum(apart, track.n - apart)
    assert distance[apart > gap].min() > track.width


def test_every_circuit_has_a_title_and_a_name():
    assert set(CIRCUITS) == set(CIRCUIT_TITLES)
    assert DEFAULT_CIRCUIT in CIRCUITS
    assert all(build_track(name).name == name for name in CIRCUITS)


def test_the_circuits_are_genuinely_different():
    lengths = [build_track(name).length for name in CIRCUITS]
    assert max(lengths) > 2 * min(lengths)
    tightest = [1 / geometry(t.cx, t.cy)[1].max() for t in map(build_track, CIRCUITS)]
    assert max(tightest) > 3 * min(tightest)


def test_unknown_circuit_is_rejected():
    with pytest.raises(KeyError):
        build_track("monza")
