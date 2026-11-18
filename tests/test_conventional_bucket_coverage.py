"""Bucket selection on an intermediate shelf with a nearly finished lift."""
import numpy as np
import pytest
from ats.conventional import ConventionalConfig
from ats.plan import PlanConfig, PlanSegment
from ats.simulation import Simulation
from tests.test_conventional_workspace import workspace_config


@pytest.mark.parametrize('remaining_depth', [.2, .0001])
def test_reachable_bite_after_the_first_128_depth_limited_centres(remaining_depth):
    config = workspace_config()
    config.terrain.floor = -10.
    config.local_reference = True
    config.plan = PlanConfig([PlanSegment([[25., 60.], [95., 60.]], width=20., height=-5.)])
    axis = (np.arange(config.terrain.n) + .5) * config.terrain.cell_size
    x, y = np.meshgrid(axis, axis)
    tracks = np.hypot(x - 56., y - 60.) <= 4.35
    channel = (x > 39.) & (x < 71.) & (np.abs(y - 60.) < 10.)
    config.reference_height = np.zeros_like(x)
    config.initial_height = np.where(channel & ~tracks, -.2, 0.)
    bite = (x == 61.25) & (y == 60.25)
    config.initial_height[bite] += remaining_depth
    sim = Simulation.conventional_only(config)
    fleet = sim.install_conventional(ConventionalConfig(
        base_x=12., base_y=12., work_rect=(22., 42., 76., 36.), excavator_dig_depth_m=.2))
    excavator, truck = fleet.machines
    excavator.position[:] = (56., 60., 0.)
    assert fleet.passable(excavator.position, excavator.radius, excavator)
    ids, _ = fleet._target_indices(excavator, focus_access=False)
    # All of these unfinished western centres sort ahead of the only reachable
    # soil; their entire bucket footprints are already at the depth limit.
    assert np.count_nonzero((x.ravel()[ids] < 59.) & (np.abs(y.ravel()[ids]-60.) <= 4.5)) > 128
    target = fleet._available_target(excavator, focus_access=False)
    assert target is not None
    assert fleet.target_yield(excavator, target) == pytest.approx(remaining_depth * sim.terrain.area)
    # The selected centre can be on the finished level while its bucket rim
    # actually takes the remaining soil. Follow the queued order physically.
    excavator.expose_target = target[:2]
    excavator.side_bucket = True
    volume_before = sim.terrain.volume
    assert fleet._cut(excavator, truck)
    assert fleet.excavated_m3 == pytest.approx(remaining_depth * sim.terrain.area)
    assert volume_before - sim.terrain.volume == pytest.approx(fleet.excavated_m3,abs=1e-9)
    assert truck.cargo_kg == pytest.approx(fleet.excavated_m3 * config.terrain.density)


def test_trial_pose_does_not_erase_the_actual_exposure_order():
    from tests.test_conventional_siting import channel_site
    s, config = channel_site(depth=5.)
    fleet = s.install_conventional(config)
    e = fleet.excavator
    e.expose_target = (23.25, 64.25)
    intended = e.expose_target
    row, col = int(intended[1]/s.terrain.cell), int(intended[0]/s.terrain.cell)
    s.terrain.height[row,col] = fleet.ramp_surface[row,col]
    remaining = fleet._work_remaining()
    fleet._available_target(e, remaining)
    assert e.expose_target == intended
    fleet._available_target(e)
    assert e.expose_target is None
