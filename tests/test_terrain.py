import numpy as np
import pytest
from ats.config import Config
from ats.terrain import Terrain


def test_target_has_flat_bottom_and_end_slopes():
    t = Terrain(Config())
    assert np.all(t.target[t.bottom] == -0.25)
    assert t.initial_excavation > 2*1*0.25
    assert np.all(np.abs(np.diff(t.target, axis=0)) <= t.cell*t.slope+1e-12)
    assert np.all(np.abs(np.diff(t.target, axis=1)) <= t.cell*t.slope+1e-12)
    assert t.target[63, 69] > -0.25  # slope beyond the end of the bottom


def test_scoop_never_exceeds_design_or_payload():
    t = Terrain(Config())
    mass = t.scoop(16, 16, 60)
    assert 0 < mass <= 60
    assert np.all(t.height >= t.target-1e-14)
    assert np.count_nonzero(t.height < 0) > 1
    assert t.mass+mass == pytest.approx(t.initial_volume*1600, abs=1e-7)


def test_repeated_transfers_and_avalanches_conserve_mass():
    t = Terrain(Config())
    initial = t.mass
    for _ in range(100):
        excess = np.where(t.excavation, t.height-t.target, -np.inf)
        y, x = np.unravel_index(np.argmax(excess), excess.shape)
        mass = t.scoop(t.axis[x], t.axis[y], 60)
        if mass:
            point = t.dump_location(mass/1600)
            t.deposit(*point, mass)
        t.relax(5)
        assert t.mass == pytest.approx(initial, abs=1e-7)


def test_closed_edges_and_stable_avalanche():
    t = Terrain(Config())
    t.height[0, 0] += 2
    before = t.mass
    initial_excess = t.max_slope_excess()
    t.relax(250)
    assert t.mass == pytest.approx(before, abs=1e-7)
    assert t.max_slope_excess() < initial_excess*1e-5
    assert np.min(t.height) >= 0


def test_dump_failure_does_not_destroy_cargo_or_edit_terrain():
    t = Terrain(Config())
    before = t.height.copy()
    with pytest.raises(ValueError, match="Отвал заполнен"):
        t.deposit(26, 24, (t.dump_space+1)*1600)
    np.testing.assert_array_equal(t.height, before)
