"""A retained access target must not strand a crew with reachable work."""
import numpy as np

from ats.checkpoint import load_state, save_state
from ats.conventional import ConventionalConfig
from ats.plan import PlanConfig, PlanSegment
from ats.simulation import Simulation
from tests.test_conventional import balanced, fuel_tanks
from tests.test_conventional_workspace import workspace_config


def test_saved_crew_resumes_nearby_cut_after_reaching_depth_limit(tmp_path):
    config = workspace_config()
    config.terrain.floor = -10.
    config.terrain.dump_rect = (100., 100., 20., 20.)
    config.plan = PlanConfig([PlanSegment([[35., 60.], [85., 60.]], width=20., height=-5.)])
    axis = (np.arange(config.terrain.n) + .5) * config.terrain.cell_size
    x, y = np.meshgrid(axis, axis)
    pilot = (abs(x - 45.25) < 1.) & (abs(y - 60.25) < 1.)
    config.initial_height = np.where(pilot, -.6, 0.)
    config.reference_height = np.zeros_like(x)
    config.local_reference = True

    original = Simulation.conventional_only(config)
    fleet = original.install_conventional(ConventionalConfig(
        base_x=12., base_y=12., work_rect=(30., 43., 65., 34.),
        truck_count=1, excavator_dig_depth_m=.5))
    fuel_tanks(fleet)
    excavator, truck = fleet.machines
    excavator.position[:] = (40., 58., 0.)
    excavator.state = 'WAIT_TRUCK'
    # The previously requested road cell is still above its design surface,
    # but a 0.5 m digging depth cannot reach the bottom of this 0.6 m cut.
    excavator.expose_target = (45.25, 60.25)
    assert fleet.passable(excavator.position, excavator.radius, excavator)
    assert fleet._home_route(excavator, validate=True) is not None
    assert np.linalg.norm(excavator.position[:2] - truck.position[:2]) > 30.
    path = tmp_path / 'unfinished-ramp.atsstate'
    save_state(path, original)

    resumed = load_state(path).simulation
    fleet = resumed.conventional
    excavator, truck = fleet.machines
    assert excavator.expose_target == (45.25, 60.25)
    for _ in range(210):
        resumed.step()
        if truck.cargo_kg > 0.:
            break
    assert truck.distance_m > 20.
    assert fleet.excavated_m3 > 0. and truck.cargo_kg > 0., excavator.reason
    assert excavator.distance_m == 0.  # Work was already in reach from here.
    assert excavator.target[:2] != (45.25, 60.25)
    balanced(resumed, fleet)
