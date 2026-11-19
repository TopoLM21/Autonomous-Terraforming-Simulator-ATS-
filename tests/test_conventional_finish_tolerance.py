"""Vertical acceptance at the design surface, using real bucket and avalanche physics."""
import numpy as np
import pytest

from ats.checkpoint import load_state, save_state
from ats.conventional import ConventionalConfig
from ats.plan import PlanConfig, PlanSegment
from ats.simulation import Simulation
from ats.soil import SoilProfile
from tests.test_conventional import balanced, fuel_tanks
from tests.test_conventional_workspace import workspace_config


def small_cut(*, depth=.6, width=.5, tolerance=None, initial=None, road=False, soil=False):
    config = workspace_config()
    config.terrain.floor = -10.
    end_x = 42.25 if road else 35.3
    config.plan = PlanConfig([PlanSegment([[35.25, 50.25], [end_x, 50.25]],
                                         width=width, height=-depth)])
    if initial is not None:
        config.initial_height = initial
    if soil:
        config.soil_profile = SoilProfile.qattara()
    sim = Simulation.conventional_only(config)
    ramp_y = 50.25 if road else 58.
    fleet_config = ConventionalConfig(base_x=12., base_y=12., work_rect=(35., 40., 24., 20.),
                                      ramp_entry_m=(35., ramp_y), ramp_end_m=(58., ramp_y))
    if tolerance is not None:
        fleet_config.cut_tolerance_m = tolerance
    fleet = sim.install_conventional(fleet_config)
    excavator, truck = fleet.machines
    excavator.position[:] = (42., 50.25, 0.)
    truck.position[:] = (47., 58., 0.)
    return sim, fleet


def test_repeated_physical_slumping_stops_with_measurable_unexcavated_soil():
    def run(tolerance):
        sim, fleet = small_cut(tolerance=tolerance)
        excavator, truck = fleet.machines
        bucket_masses, avalanches = [], []
        for _ in range(20):
            before = truck.cargo_kg
            if not fleet._cut(excavator, truck):
                break
            bucket_masses.append(truck.cargo_kg - before)
            avalanches.append(sim.terrain.relax())
        balanced(sim, fleet)
        return fleet, bucket_masses, avalanches

    exact, precise_masses, _ = run(0.)
    assert len(precise_masses) == 20
    assert 0 < precise_masses[-1] < 1.
    assert any(10. < mass < 15. for mass in precise_masses)
    assert not exact._work_complete

    fleet, masses, avalanches = run(None)
    assert fleet.config.cut_tolerance_m == .25
    assert 0 < len(masses) < 5
    assert any(volume > 0 for volume in avalanches)
    assert 0 < fleet._remaining().max() <= .25
    excavator, truck = fleet.machines
    previous = truck.cargo_kg
    assert not fleet._choose_excavator_site(excavator)
    assert fleet._work_complete
    assert not fleet._cut(excavator, truck)
    assert truck.cargo_kg == previous
    assert fleet.report()['remaining_work_m3'] > 0
    assert fleet.report()['remaining_actionable_m3'] == 0


def test_dispatch_delivers_final_partial_truckload_and_parks_both_machines():
    sim, fleet = small_cut()
    for machine in fleet.machines:
        machine.position[:] = machine.home
    fuel_tanks(fleet)
    for _ in range(450):
        sim.step()
        if fleet.ground_stage_drained:
            break
    assert fleet.ground_stage_drained
    assert 0 < fleet.delivered_m3 < fleet.config.truck_body_m3
    assert fleet.delivered_m3 == pytest.approx(fleet.excavated_m3)
    assert fleet.report()['remaining_work_m3'] > 0
    assert fleet.report()['remaining_actionable_m3'] == 0
    balanced(sim, fleet)


@pytest.mark.parametrize('soil', [False, True])
def test_straddling_bucket_preserves_acceptable_and_out_of_allocation_cells(soil):
    # Bed cells start 25 cm above the design elevation, except the 26 cm
    # central cell. A higher neighbour lies outside the assigned worksite.
    initial = np.zeros((256, 256))
    row, col = 100, 70
    initial[row, col] = .01
    initial[row, col - 1] = .12
    initial[row, col + 1] = -.02
    # The stock came from an equal volume of the same surface sand outside
    # this worksite, so the initial material ledger also sums to zero.
    initial[row, col - 3] = -.11
    sim, fleet = small_cut(depth=.25, width=2., initial=initial, soil=soil)
    t = sim.terrain
    assert t.target[row, col] == pytest.approx(-.25)
    before = t.height.copy()
    excavator, truck = fleet.machines
    assert fleet._cut(excavator, truck)
    changed = np.argwhere(t.height != before)
    assert changed.tolist() == [[row, col]]
    assert t.height[row, col] == pytest.approx(t.target[row, col])
    assert fleet.excavated_m3 == pytest.approx(.26 * t.area)
    assert fleet.target_yield(excavator) == 0
    assert not fleet._cut(excavator, truck)
    balanced(sim, fleet)


def test_traversable_ramp_keeps_its_stricter_finish_and_support_surface():
    sim, fleet = small_cut(depth=1., width=4., road=True)
    excavator, truck = fleet.machines
    t = sim.terrain
    row, col = 100, 72
    assert fleet.access_mask[row, col]
    allowance = t.height[row, col] - fleet.ramp_surface[row, col]
    assert .10 < allowance < .25
    patch = t.patch(*excavator.position[:2], fleet.config.excavator_reach_m)
    _, allowed = fleet._bucket_geometry(excavator, patch)
    assert allowed[row - patch[0].start, col - patch[1].start]
    for _ in range(12):
        if not fleet._cut(excavator, truck):
            break
        if t.height[row, col] < 0:
            break
    assert t.height[row, col] == pytest.approx(fleet.ramp_surface[row, col])
    assert np.all(t.height[fleet.ramp_mask] >= fleet.ramp_surface[fleet.ramp_mask] - 1e-12)
    balanced(sim, fleet)


def test_real_avalanche_crossing_the_limit_makes_this_stance_productive_again():
    initial = np.zeros((256, 256))
    initial[99:102, 69] = .8  # A bank immediately outside the excavation.
    sim, fleet = small_cut(depth=.20, width=2., initial=initial)
    excavator, truck = fleet.machines
    fuel_tanks(fleet)
    assert fleet.target_yield(excavator) == 0
    assert not fleet._choose_excavator_site(excavator)
    assert fleet._work_complete
    assert sim.terrain.relax() > 0
    assert fleet._remaining().max() > fleet.config.cut_tolerance_m
    sim.step()
    assert not fleet._work_complete
    volume = fleet.target_yield(excavator)
    assert volume > 0
    assert fleet._cut(excavator, truck)
    assert fleet.excavated_m3 == pytest.approx(volume)
    balanced(sim, fleet)


@pytest.mark.parametrize(('tolerance', 'expected'), [(None, .25), (0., 0.), (.30, .30)])
def test_saved_tolerance_and_partly_loaded_truck_survive_checkpoint(tmp_path, tolerance, expected):
    sim, fleet = small_cut(tolerance=tolerance)
    excavator, truck = fleet.machines
    assert fleet._cut(excavator, truck)
    sim.terrain.relax()
    if tolerance is None:
        # Older manifests serialized no tolerance field. Omit it while preserving
        # the real checkpoint graph, soil, bucket load and equipment positions.
        del vars(fleet.config)['cut_tolerance_m']
    path = tmp_path / 'finish.atsstate'
    save_state(path, sim)
    restored = load_state(path).simulation
    loaded = restored.conventional
    assert loaded.config.cut_tolerance_m == expected
    assert loaded.carried_mass_kg == fleet.carried_mass_kg
    np.testing.assert_array_equal(restored.terrain.height, sim.terrain.height)
    assert loaded._cut(*loaded.machines) == fleet._cut(*fleet.machines)
    np.testing.assert_array_equal(restored.terrain.height, sim.terrain.height)
    balanced(restored, loaded)
