import numpy as np
import pytest
from ats.drone import DroneState as DS
from ats.simulation import Simulation
from ats.config import Config


def test_finite_speed_states_and_collision_clearance():
    s = Simulation()
    states = set()
    for _ in range(400):
        old = s.drone.position.copy()
        old_load = s.drone.cargo
        s.step()
        states.add(s.drone.state)
        pos = s.drone.position
        speed = s.config.drone.loaded_speed if old_load else s.config.drone.empty_speed
        assert np.linalg.norm(pos[:2]-old[:2]) <= speed+1e-9
        assert abs(pos[2]-old[2]) <= s.config.drone.vertical_speed+1e-9
        assert pos[2] >= s.terrain.sample(*pos[:2])
        if not s.drone.docked:
            assert s.base.destination is None
    assert states == set(DS)-{DS.CHARGING}


def test_base_relocation_is_finite_speed_and_safe():
    s = Simulation()
    moved = False
    for _ in range(3000):
        before = s.base.position.copy()
        s.step()
        distance = np.linalg.norm(s.base.position[:2]-before[:2])
        assert distance <= s.config.base.speed+1e-9
        if distance > 0:
            moved = True
            assert s.drone.docked
            assert np.allclose(s.drone.position[:2], s.base.position[:2])
            assert s.base.solar_power == 0
    assert moved


def test_base_refuses_trench_crossing():
    s = Simulation()
    assert not s.base.safe_path((25, 16), s.terrain)
    assert s.base.safe_path((10, 16), s.terrain)


def test_bad_dt_and_invalid_geometry_are_rejected():
    s = Simulation()
    with pytest.raises(ValueError, match="фиксированный"):
        s.step(200)
    c = Config()
    c.trench.depth = -1
    with pytest.raises(ValueError):
        Simulation(c)


def test_obstructed_flight_is_blocked_without_teleporting():
    s = Simulation()
    s.step()
    s.step()
    before = s.drone.position.copy()
    s.terrain.height[int(before[1]/s.terrain.cell), int(before[0]/s.terrain.cell)] = 100
    s.step()
    assert s.state.value == "BLOCKED"
    assert "пересекает" in s.reason
    np.testing.assert_array_equal(s.drone.position, before)
