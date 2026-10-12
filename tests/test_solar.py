import math
import numpy as np
import pytest
from ats.config import Config
from ats.solar import sunlight
from ats.simulation import Simulation
from ats.solar_base import SolarBase
from ats.drone import Drone


def test_equinox_equator_geometry_and_incident_flux():
    noon = sunlight(0, 80, 12)
    assert noon.irradiance == 1000
    assert noon.elevation_deg == 90
    assert noon.daylight_hours == pytest.approx(12)
    assert noon.sunrise_hour == pytest.approx(6)
    assert noon.sunset_hour == pytest.approx(18)
    assert sunlight(0, 80, 9).irradiance == pytest.approx(1000/math.sqrt(2))
    for hour in [0, 5, 6, 18, 23]:
        assert sunlight(0, 80, hour).irradiance == 0


def test_latitude_season_and_southern_hemisphere():
    north_summer = sunlight(55, 172, 12)
    north_winter = sunlight(55, 355, 12)
    assert north_summer.irradiance > north_winter.irradiance > 0
    assert north_summer.daylight_hours > 16
    assert north_winter.daylight_hours < 8
    assert sunlight(-55, 172, 12).daylight_hours < 8
    assert sunlight(-55, 355, 12).daylight_hours > 16
    # A horizontal panel receives the cosine of the solar zenith angle.
    assert north_summer.irradiance == pytest.approx(1000*math.sin(math.radians(north_summer.elevation_deg)))


@pytest.mark.parametrize("latitude", [80, 90])
def test_polar_day_and_night(latitude):
    summer = sunlight(latitude, 172, 0)
    winter = sunlight(latitude, 355, 12)
    assert summer.daylight_hours == 24 and summer.irradiance > 0
    assert summer.sunrise_hour is None and summer.sunset_hour is None
    assert winter.daylight_hours == 0 and winter.irradiance == 0
    assert winter.sunrise_hour is None and winter.sunset_hour is None


def test_clock_elapsed_and_calendar_midnight_rollover():
    s = Simulation()
    s.time = 2572
    r = s.report()
    assert r["elapsed_display"] == "0 д 00:42:52"
    assert r["solar_clock"] == "09:42:52"
    assert r["date"] == "21.06"
    s.config.start_hour = 23.75
    s.time = 3600
    assert s.clock_text == "00:45:00"
    assert s.day_of_year == 173
    assert s.date_text == "22.06"
    s.config.base.start_day_of_year = 365
    assert s.day_of_year == 1 and s.date_text == "01.01"


def test_mass_affects_track_power_and_partial_move_energy():
    c = Config()
    b = SolarBase(c.base, np.zeros(3))
    d = Drone(c.drone, np.zeros(3))
    assert b.motion_power(d) == 1000
    c.base.mass = 2000
    assert b.motion_power(d) == pytest.approx(1000*2105/1105)
    b.destination = np.array([0.1, 0.0])
    b.battery = b.motion_power(d)*0.5
    initial = b.battery
    b.tick(1, 12, d, False)
    assert b.destination is None
    assert b.position[0] == 0.1
    assert b.battery == pytest.approx(0, abs=1e-9)
    assert b.consumed == pytest.approx(initial)
    assert b.solar_power == 0


def test_panel_area_scales_power_with_unchanged_sun():
    c = Config()
    d = Drone(c.drone, np.zeros(3))
    b = SolarBase(c.base, np.zeros(3))
    b.tick(1, 12, d, False)
    first = b.solar_power
    c.base.panel_area *= 2
    b.tick(1, 12, d, False)
    assert b.solar_power == pytest.approx(2*first)


def test_dump_lowers_grabber_and_releases_at_surface_without_landing():
    s = Simulation()
    for _ in range(100):
        s.step()
        if s.drone.state.value == "DUMP":
            break
    assert s.drone.state.value == "DUMP"
    z = s.drone.position[2]
    cargo = s.drone.cargo
    before = s.terrain.mass
    for _ in range(4):
        s.step()
        assert s.drone.cargo == cargo
        assert s.drone.position[2] == z
    s.step()
    assert s.drone.dump_released and s.drone.cargo == 0
    assert s.drone.grabber_extension == 1
    assert s.terrain.mass == pytest.approx(before+cargo, abs=1e-7)
    assert s.drone.position[2] == z
    for _ in range(5):
        s.step()
    assert s.drone.state.value == "RETURN"
    assert s.drone.grabber_extension == 0
    assert abs(s.report()["mass_error_kg"]) < 1e-7


@pytest.mark.parametrize("field,value", [("latitude_deg", 91), ("start_day_of_year", 0), ("mass", -1)])
def test_invalid_solar_and_base_settings(field, value):
    c = Config()
    setattr(c.base, field, value)
    with pytest.raises(ValueError):
        Simulation(c)
