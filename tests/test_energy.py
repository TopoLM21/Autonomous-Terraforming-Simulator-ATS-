import numpy as np
import pytest
from ats.config import Config, KWH
from ats.drone import Drone, DroneState
from ats.solar_base import SolarBase
from ats.simulation import Simulation


def objects():
    c = Config()
    c.base.latitude_deg = 0.0
    c.base.start_day_of_year = 80  # equinox: the zenith noon reference
    c.drone.initial_charge = 0.0
    c.base.initial_charge = 0.5
    return c, SolarBase(c.base, np.zeros(3)), Drone(c.drone, np.zeros(3))


def test_solar_bound_night_and_noon():
    c, b, d = objects()
    for h in [0, 5.99, 18, 23.9]:
        assert b.irradiance(h) == 0
    assert b.irradiance(12) == c.base.peak_irradiance
    b.tick(1, 12, d, False)
    assert b.solar_power == 4000


def test_charge_with_losses_and_no_double_counting():
    c, b, d = objects()
    initial = b.battery+d.battery
    for _ in range(1000):
        b.tick(1, 12, d, True)
    assert b.battery+d.battery+b.charging_losses+b.curtailed+b.consumed == pytest.approx(initial+b.generated, abs=1e-5)
    assert b.charging_losses == pytest.approx(b.charging_input*0.1)
    assert d.battery == pytest.approx(b.charging_input*0.9)


def test_no_charging_in_flight_and_no_solar_while_moving():
    c, b, d = objects()
    d.state = DroneState.TO_DIG
    b.tick(1, 12, d, True)
    assert d.battery == 0
    d.state = DroneState.DOCKED
    b.destination = np.array([1, 0])
    before = b.battery
    b.tick(1, 12, d, True)
    assert b.solar_power == 0
    assert b.battery == before-c.base.move_power
    assert b.position[0] == c.base.speed
    assert d.battery == 0


def test_empty_base_and_night_waits_for_sun():
    c = Config(start_hour=5.9)
    c.base.latitude_deg = 0.0
    c.base.start_day_of_year = 80
    c.drone.initial_charge = 0.2
    c.base.initial_charge = 0.0
    s = Simulation(c)
    s.run(300)
    assert s.drone.docked and s.drone.battery == c.drone.battery*0.2
    assert s.base.battery == 0
    s.run(20000)  # now a complete 5 kWh charge is required before departure
    assert s.excavated_mass > 0  # dawn eventually enables autonomous work
    assert abs(s.report()["energy_error_j"]) < 1e-4


def test_hover_power_scales_with_load():
    c, b, d = objects()
    assert d.power(cargo=60) > d.power(cargo=0)
    assert d.power(cargo=60) == pytest.approx(18000*(165/105)**1.5)


def test_winch_accounts_for_gravitational_work():
    c, b, d = objects()
    added_power = d.power(winching=True, cargo=60, lift_height=10)-d.power(cargo=60)-c.drone.winch_power
    assert added_power*c.drone.winch_seconds*c.drone.winch_efficiency == pytest.approx(60*c.drone.gravity*10)


def test_full_base_can_use_solar_while_charging():
    c, b, d = objects()
    b.battery = b.config.battery
    b.tick(1, 12, d, True)
    assert b.curtailed == 0
    assert b.battery == b.config.battery+4000-8000
    assert d.battery == 7200


def test_base_full_drone_full_and_empty_never_overflow():
    c, b, d = objects()
    b.battery = b.config.battery
    d.battery = d.config.battery
    b.tick(1, 12, d, True)
    assert b.battery == b.config.battery and d.battery == d.config.battery
    assert b.curtailed == 4000
    b.battery = 0
    d.battery = 0
    b.tick(1, 0, d, True)
    assert b.battery == 0 and d.battery == 0
