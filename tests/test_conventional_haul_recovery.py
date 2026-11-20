"""Continue a loaded haul after its initially safe tipping stand is lost."""
import numpy as np
import pytest

from ats.checkpoint import load_state, save_state
from tests.test_conventional_loading_progress import waiting_load
from tests.test_conventional_routes import balanced, loaded_truck


def assigned_haul():
    sim, fleet, truck = loaded_truck(soil=True)
    truck.position[:] = (74., 100., 0.)
    assert fleet._assign_route(truck, (94., 100.), 'UNLOADING')
    truck.target = (100., 100.)
    truck.timer = 30.
    return sim, fleet, truck


def changed_stand():
    sim, fleet, truck = assigned_haul()
    terrain = sim.terrain
    # Another real load reaches the intended wheel position after dispatch.
    # It is conserved separately from the tested truck's full, untouched body.
    spoil = terrain.scoop_load(72., 50., 8000., source='conventional', radius=6.)
    terrain.deposit(*truck.destination, spoil.mass, radius=1., load=spoil)
    sim._terrain_active = True
    assert truck.route_token != terrain.route_token(truck.route_bounds)
    assert not fleet.passable(truck.destination, truck.radius, truck)
    balanced(sim, fleet)
    return sim, fleet, truck


def occupied_stand():
    sim, fleet, truck = assigned_haul()
    # The excavator arrives after the truck receives its haul order, then
    # remains stationary for the test. Its body does not edit the terrain.
    fleet.excavator.position[:] = (*truck.destination, 0.)
    fleet.excavator.retry_route_at = sim.time + 400.
    assert truck.route_token == sim.terrain.route_token(truck.route_bounds)
    assert not fleet.passable(truck.destination, truck.radius, truck)
    return sim, fleet, truck


def run_to_first_tip(sim):
    fleet = sim.conventional
    truck = fleet.machines[1]
    initial_mass = truck.cargo_kg
    initial_materials = truck.soil.masses.copy()
    fuel_before = truck.fuel_l
    for _ in range(180):
        before = truck.position[:2].copy()
        sim.step()
        if not np.array_equal(before, truck.position[:2]):
            assert np.linalg.norm(truck.position[:2] - before) <= fleet.config.truck_speed_m_s + 1e-9
            assert fleet.segment_safe(before, truck.position, truck.radius, truck)
        if fleet.delivered_mass_kg:
            break
        assert truck.cargo_kg == initial_mass
        np.testing.assert_array_equal(truck.soil.masses, initial_materials)
    assert fleet.delivered_mass_kg > 0, truck.reason
    assert not np.allclose(truck.position[:2], (94., 100.))
    assert truck.fuel_l < fuel_before
    assert truck.cargo_kg + fleet.delivered_mass_kg == pytest.approx(initial_mass)
    np.testing.assert_allclose(fleet.delivered_materials + fleet.carried_soil.masses, initial_materials)
    balanced(sim, fleet)


def test_stale_loaded_haul_reselects_safe_stand_even_from_checkpoint(tmp_path):
    sim, _, truck = changed_stand()
    # A checkpoint from the old dispatcher can already have a pending retry.
    truck.reason = 'Маршрут изменён грунтом или препятствием; груз сохранён'
    truck.retry_route_at = sim.time + 30.
    snapshot = tmp_path / 'blocked-haul.atsstate'
    save_state(snapshot, sim)
    restored = load_state(snapshot).simulation
    for world in (sim, restored):
        run_to_first_tip(world)
    assert sim.report() == restored.report()


def test_partial_truck_reaches_a_different_loading_bay_without_tipping():
    sim, fleet, excavator, truck = waiting_load(trucks=2)
    bay = truck.position.copy()
    cargo = truck.soil.masses.copy()
    load_started = truck.load_started
    truck.position[:] = (65., 35., sim.terrain.sample(65., 35.))
    assert fleet._assign_route(truck, bay[:2], 'WAIT_LOAD')
    # A second truck now occupies the assigned bay. Hold the work front
    # for this short approach so only the loading stand needs to change.
    other = fleet.machines[2]
    assert other.state == 'PARKED' and not other.path
    other.position[:] = bay
    other.retry_route_at = excavator.next_advance_at = sim.time + 300.
    for _ in range(110):
        sim.step()
        assert truck.after_drive == 'WAIT_LOAD'
        assert fleet.delivered_mass_kg == 0
        np.testing.assert_array_equal(truck.soil.masses, cargo)
        if truck.state == 'WAIT_LOAD':
            break
    assert truck.state == 'WAIT_LOAD', truck.reason
    assert not np.allclose(truck.position, bay)
    assert np.linalg.norm(truck.position[:2] - excavator.position[:2]) <= fleet._loading_distance(truck) + .5
    assert truck.load_started == load_started
    balanced(sim, fleet)


@pytest.mark.parametrize('blocker', [changed_stand, occupied_stand])
def test_unavailable_dump_waits_with_cargo_then_resumes(monkeypatch, blocker):
    sim, fleet, truck = blocker()
    t = sim.terrain
    original_bounds = truck.route_bounds
    terrain_token = t.route_token(original_bounds)
    cap = t.dump_cap.copy()
    t.dump_cap[t.dump_mask] = 0.
    original = fleet._dump_point
    searches = []

    def counted(vehicle):
        searches.append(sim.time)
        return original(vehicle)

    monkeypatch.setattr(fleet, '_dump_point', counted)
    mass = truck.cargo_kg
    material = truck.soil.masses.copy()
    stopped_at = None
    for _ in range(100):
        sim.step()
        if searches:
            if stopped_at is None:
                stopped_at = truck.position.copy()
            np.testing.assert_array_equal(truck.position, stopped_at)
    assert 2 <= len(searches) <= 5, searches
    assert fleet.delivered_mass_kg == 0 and truck.cargo_kg == mass
    assert not np.allclose(truck.position[:2], (94., 100.))
    if blocker is occupied_stand:
        assert terrain_token == t.route_token(original_bounds)
    np.testing.assert_array_equal(truck.soil.masses, material)
    balanced(sim, fleet)
    t.dump_cap[:] = cap
    run_to_first_tip(sim)
