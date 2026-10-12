import numpy as np
import pytest
from ats.config import Config
from ats.scenarios import scenario
from ats.simulation import Simulation, MissionState


def test_demo_completes_with_payload_and_energy_reserve():
    s = Simulation()
    for _ in range(6000):
        s.step()
        assert -1e-9 <= s.drone.battery <= s.config.drone.battery+1e-9
        assert -1e-9 <= s.base.battery <= s.config.base.battery+1e-9
        assert 0 <= s.drone.cargo <= s.config.drone.payload+1e-9
        assert abs(s.report()["mass_error_kg"]) < 1e-7
        if s.state != MissionState.RUNNING:
            break
    assert s.state == MissionState.COMPLETED
    assert s.terrain.remaining < 0.05
    assert s.drone.cargo == 0
    assert s.drone.docked
    assert s.drone.battery >= s.config.drone.battery*s.config.drone.reserve
    assert s.base.charging_input > 0
    assert abs(s.report()["energy_error_j"]) < 1e-4
    assert np.all(s.terrain.height[~s.terrain.dump_mask & ~s.terrain.excavation] < 1e-8)


def test_reproducible_headless_result():
    a, b = Simulation(scenario(seed=42)), Simulation(scenario(seed=42))
    assert a.run(3000) == b.run(3000)
    np.testing.assert_array_equal(a.terrain.height, b.terrain.height)


def test_insufficient_dump_is_explicitly_blocked():
    c = Config()
    c.terrain.dump_rect = (24, 24, 0.5, 0.5)
    s = Simulation(c)
    assert s.state == MissionState.BLOCKED
    assert "отвале" in s.reason
    assert s.time == 0


def test_no_energy_source_blocks_instead_of_waiting_forever():
    c = Config()
    c.drone.initial_charge = 0.2
    c.base.initial_charge = 0
    c.base.panel_area = 0
    s = Simulation(c)
    s.run(10)
    assert s.state == MissionState.BLOCKED
    assert "разряжена" in s.reason


def test_battery_too_small_for_any_trip():
    c = Config()
    c.drone.battery = 1000
    s = Simulation(c)
    s.run(5)
    assert s.state == MissionState.BLOCKED
    assert "рейса" in s.reason


@pytest.mark.parametrize("preset,max_steps", [("medium", 200000), ("large", 2000000)])
def test_larger_presets_complete(preset, max_steps):
    s = Simulation(scenario(preset))
    report = s.run(max_steps)
    assert s.state == MissionState.COMPLETED, report
    assert report["remaining_m3"] < 0.05
    assert abs(report["mass_error_kg"]) < 1e-7
    assert abs(report["energy_error_j"]) < 0.002
    assert report["minimum_drone_kwh"] >= s.config.drone.battery*s.config.drone.reserve/3_600_000
    assert report["cargo_kg"] == 0
    # Optional evidence export, leaving ordinary test runs read-only.
    import os
    if os.environ.get("ATS_REPORT_DIR"):
        import json
        from pathlib import Path
        directory = Path(os.environ["ATS_REPORT_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory/f"{preset}-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
