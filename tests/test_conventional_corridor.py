"""The fast ground clearance proof must agree with detailed physical samples."""
import numpy as np

from tests.test_conventional import world


def test_clear_ramp_proof_rechecks_live_support_water_and_vehicles(monkeypatch):
    sim, fleet = world(trucks=1)
    excavator, truck = fleet.machines
    terrain = sim.terrain
    terrain.height[:] = .12 * terrain.x
    start, goal = np.array([40., 70.]), np.array([72., 70.])
    radius = excavator.radius

    def check(expected):
        fast = fleet.segment_safe(start, goal, radius, excavator)
        with monkeypatch.context() as override:
            override.setattr(fleet, '_clear_ground_corridor', lambda *args: False)
            assert fleet.segment_safe(start, goal, radius, excavator) == fast
        assert fast == expected

    # The total rise is several metres, but the gradient and grid sampling
    # bounds prove that the entire swept support is safe on this smooth ramp.
    assert fleet._clear_ground_corridor(start, goal, radius, excavator)
    check(True)
    truck.position[:2] = (56., 70.)
    check(False)
    truck.position[:2] = truck.home[:2]
    row, col = int(72.5 / terrain.cell), int(56. / terrain.cell)
    terrain.height[row, col] += 1.
    check(False)
    terrain.height[row, col] = .12 * terrain.x[row,col]
    fleet._marine[row,col] = True
    check(False)
    fleet._marine[row,col] = False
    check(True)
