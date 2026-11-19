import numpy as np
import pytest

from tests.test_conventional_shared import mixed
from tests.test_conventional_routes import loaded_truck


def finished_shared_patch():
    project, _, fleet = mixed()
    t = fleet.terrain
    # A revised shallow work order has just 2 cm of cut left in one cell.
    cell = np.flatnonzero(fleet.work_mask & t.dig_mask & ~fleet.ramp_mask)[0]
    t.target[fleet.work_mask] = t.height[fleet.work_mask]
    t.target.flat[cell] -= .02
    t.revision += 1
    assert not fleet._choose_excavator_site(fleet.excavator)
    assert fleet.ground_stage_drained
    return project.shared_site, fleet, cell


def test_tighter_live_grade_reopens_owned_patch_before_handoff():
    site, fleet, _ = finished_shared_patch()
    fleet.set_cut_tolerance(0.)
    assert not fleet._work_complete
    assert not site.release_conventional_stage()
    assert np.any(site.core._conventional_mask)


def test_recheck_before_handoff_even_without_a_fleet_tick():
    site, fleet, cell = finished_shared_patch()
    site.core.target.flat[cell] -= .5
    site.core.revision += 1
    assert not site.release_conventional_stage()
    assert not fleet._work_complete
    assert np.any(site.core._conventional_mask)


@pytest.mark.parametrize('old_save', [False, True])
def test_released_work_never_reclaimed_or_rescanned(old_save, monkeypatch):
    site, fleet, cell = finished_shared_patch()
    assert site.release_conventional_stage()
    if old_save:
        del fleet._ground_stage_released
        del fleet._completed_revision
    site.core.target.flat[cell] -= .5
    site.core.revision += 1
    monkeypatch.setattr(fleet, '_work_remaining', lambda: pytest.fail('released site rescanned'))
    assert fleet.ground_stage_drained
    assert not site.release_conventional_stage()
    with pytest.raises(ValueError, match='передан'):
        fleet.set_cut_tolerance(0.)
    assert fleet.config.cut_tolerance_m == .25
    assert not np.any(site.core._conventional_mask)
    site.enlarge(192)
    assert fleet.ground_stage_drained
    assert not np.any(site.core._conventional_mask)


def test_live_setting_preserves_a_committed_haul():
    _, fleet, truck = loaded_truck(soil=True)
    truck.state = 'DRIVING'
    truck.after_drive = 'UNLOADING'
    truck.path = [(94., 100.), (95., 100.)]
    truck.destination = truck.path[-1]
    before = (truck.cargo_kg, truck.state, truck.target, list(truck.path), truck.destination)
    materials = truck.soil.masses.copy()
    with pytest.raises(ValueError):
        fleet.set_cut_tolerance(.31)
    assert fleet.config.cut_tolerance_m == .25
    fleet.set_cut_tolerance(.20)
    assert fleet.config.cut_tolerance_m == .20
    assert (truck.cargo_kg, truck.state, truck.target, truck.path, truck.destination) == before
    np.testing.assert_array_equal(truck.soil.masses, materials)
