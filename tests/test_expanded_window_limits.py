"""Joined active windows retain native-grid data past the old 1024-cell cap."""
from types import SimpleNamespace
import numpy as np
import pytest

from ats.config import Config
from ats.landscape import Landscape, workspace_config
from ats.maps import MAX_ACTIVE_CELLS, MAX_ACTIVE_SIDE_M
from ats.soil import SoilField, SoilProfile
from tests.test_landscape import raster


def test_above_old_grid_limit_keeps_sparse_mixed_soil_and_region_roundtrip(tmp_path):
    n = 1056
    origin = [16., 24.]
    profile = SoilProfile.qattara()
    land = Landscape(raster(0., spacing=1000.), soil_profile=profile, cache_bytes=32768)
    baseline, height = land.read_window(origin, n)
    assert height.shape == (n, n) and not land.tiles and not land.soil_tiles
    field = SoilField(profile, n, land.cell_size, land.soil_window(origin, n, land.cell_size))
    source = 10*n+10
    dump = (n-15)*n+n-15
    change, load = field.extract([source], [12.], commit=True)
    assert np.all(load.masses > 0.)  # Cross both native contacts, not uniform sand.
    height.ravel()[source] -= change[0]
    fill = load.volume / land.cell_size**2
    field.add([dump], [fill], load)
    height.ravel()[dump] += fill
    np.testing.assert_allclose(field.volume_delta_by_material, 0., atol=1e-12)
    land.commit_window(origin, height, soil=field)
    assert len(land.tiles) == 2 and len(land.soil_tiles) == 2
    # Write-through does not need to retain either block. Explicitly read both
    # into a one-height-tile LRU, forcing a real resident-block eviction.
    for key in land.tiles:
        assert np.all(np.isfinite(land.tiles[key]))
    assert land.cache_report()['resident_bytes'] <= 32768
    assert land.cache_report()['evictions'] > 0
    assert land.report()['soil_delta_m3'] == pytest.approx(0., abs=1e-12)

    path = tmp_path/'joined-region.atsregion'
    land.save(path)
    restored = Landscape.load(path)
    reference, actual = restored.read_window(origin, n)
    np.testing.assert_array_equal(reference, baseline)
    np.testing.assert_array_equal(actual, height)
    again = SoilField(profile, n, restored.cell_size,
                      restored.soil_window(origin, n, restored.cell_size))
    assert set(again.tiles) == set(field.tiles)
    for key in field.tiles:
        np.testing.assert_array_equal(again.tiles[key], field.tiles[key])
    np.testing.assert_allclose(again.volume_delta_by_material, 0., atol=1e-12)
    np.testing.assert_allclose(again.height_delta([source, dump]), [-change[0], fill], atol=1e-12)


def test_grid_cap_rejects_before_sampling_or_dense_allocation(monkeypatch):
    land = Landscape(raster(spacing=1000.))
    def unexpected(*args, **kwargs):
        pytest.fail('Oversized grid reached DEM interpolation')
    monkeypatch.setattr(land, '_window', unexpected)
    with pytest.raises(ValueError, match='2048'):
        land.read_window([0., 0.], MAX_ACTIVE_CELLS+1)
    with pytest.raises(ValueError, match='сетка грунта'):
        SoilField(SoilProfile.qattara(), MAX_ACTIVE_CELLS+1, .25)
    c = Config()
    c.terrain.n = MAX_ACTIVE_CELLS+1
    with pytest.raises(ValueError, match='сетка'):
        c.validate()
    assert not land.tiles and not land.soil_tiles


def test_large_workspace_uses_extended_cell_budget_without_allocating_mocked_data(monkeypatch):
    # Test resolution/coordinate dispatch at the largest admitted extent. The
    # physical >1024 grid and soil persistence are exercised above; allocating
    # four million Python-list entries twice here adds no additional coverage.
    land = Landscape(raster(spacing=1000.))
    calls = []
    marker = [[0.]]
    grid = SimpleNamespace(min=lambda: 0., tolist=lambda: marker)
    def read(origin, n, *, cell_size, prepared):
        calls.append((origin, n, cell_size))
        return grid, grid
    monkeypatch.setattr(land, 'read_window', read)
    monkeypatch.setattr(Config, 'validate', lambda self: None)
    c = Config()
    c.terrain.cell_size = 1.
    result = workspace_config(land, c, [16., 24.], MAX_ACTIVE_SIDE_M)
    assert calls == [([16., 24.], MAX_ACTIVE_CELLS, 1.)]
    assert result.terrain.n == MAX_ACTIVE_CELLS
    assert result.terrain.cell_size == c.terrain.cell_size
    assert result.initial_height is marker and result.reference_height is marker
    assert result.georeference['origin_m'] == [16., 24.]


def test_workspace_extent_limit_rejects_before_reading(monkeypatch):
    land = Landscape(raster(spacing=1000.))
    def unexpected(*args, **kwargs):
        pytest.fail('Oversized extent reached the region reader')
    monkeypatch.setattr(land, 'read_window', unexpected)
    with pytest.raises(ValueError, match='2048'):
        workspace_config(land, Config(), [0., 0.], MAX_ACTIVE_SIDE_M+1.)


def test_expanded_native_grid_survives_preparation_of_next_regional_window(monkeypatch):
    land = Landscape(raster(spacing=1000.))
    calls = []
    grid = SimpleNamespace(min=lambda: 0., tolist=lambda: [[0.]])
    def read(origin, n, *, cell_size, prepared):
        calls.append((origin, n, cell_size))
        return grid, grid
    monkeypatch.setattr(land, 'read_window', read)
    monkeypatch.setattr(Config, 'validate', lambda self: None)
    c = Config()
    c.terrain.n = 2048
    c.terrain.cell_size = .5
    c.georeference = dict(origin_m=[0., 0.], vertical_datum='MSL')
    result = workspace_config(land, c, [64., 64.], 1024.)
    assert calls == [([64., 64.], 2048, .5)]
    assert result.terrain.n == c.terrain.n
    assert result.terrain.cell_size == c.terrain.cell_size


def test_neighbours_make_real_wide_channel_patch_fit_without_reset_or_purchase():
    from ats.complexes import ComplexProject
    from ats.conventional import ConventionalConfig
    from ats.conventional_siting import prepare
    from ats.plan import PlanConfig, PlanSegment
    from ats.simulation import Simulation
    land = Landscape(raster(0., spacing=1000.), cell_size=.5)
    c = workspace_config(land, Config(), [256., 256.], 256.)
    c.plan = PlanConfig([PlanSegment([[20., 128.], [238., 128.]], width=240., height=-10.)],
        channel=dict(start=[20., 128.], tangent=[1., 0.], cell_size=.5))
    c.terrain.dump_rect = (2., 2., 5., 5.)
    c.terrain.floor = -100.
    sim = Simulation.conventional_only(c)
    project = ComplexProject(sim, land)
    site = project.enable_shared()
    before = sim.report()
    old_height = site.core.height.copy()
    old_target = site.core.target.copy()
    old_position = sim.base.position.copy()
    with pytest.raises(ValueError, match='окн'):
        prepare(sim, ConventionalConfig.for_simulation(sim))
    assert sim.conventional is None and sim.report() == before
    land.prefetch_window([192., 192.], 768, cell_size=.5)
    assert land._prefetch_runtime.wait(10.)['state'] == 'READY'
    site.enlarge(384., origin_m=[192., 192.], preserve_resolution=True)
    assert site.core.height.shape == (768, 768) and site.core.cell == .5
    np.testing.assert_array_equal(site.core.height[128:640, 128:640], old_height)
    np.testing.assert_array_equal(site.core.target[128:640, 128:640], old_target)
    np.testing.assert_array_equal(sim.base.position[:2]+[192., 192.], old_position[:2]+[256., 256.])
    resolved = prepare(sim, ConventionalConfig.for_simulation(sim))
    x, y, width, height = resolved.work_rect
    assert height > 256.  # The same real cross section could not fit before.
    assert 0. <= x < x+width <= 384. and 0. <= y < y+height <= 384.
    assert sim.conventional is None and project.time == before['time_seconds']
    assert abs(project.report()['mass_error_kg']) < 1e-7
