"""Clear-sky geometric estimate for horizontal panels, in local solar time."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Sunlight:
    irradiance: float
    elevation_deg: float
    declination_deg: float
    daylight_hours: float
    sunrise_hour: float | None
    sunset_hour: float | None


def sunlight(latitude_deg, day_of_year, hour, peak_irradiance=1000.0):
    """365-day seasonal cycle; noon is 12:00, without weather or air-mass model.

    Irradiance is peak_irradiance * max(sin(elevation), 0). Panels are
    horizontal; the peak is the estimate at a zenith sun, not today's noon.
    """
    latitude = math.radians(latitude_deg)
    declination = math.radians(23.44)*math.sin(2*math.pi*(day_of_year-80)/365)
    hour_angle = math.pi*((hour % 24)-12)/12
    a = math.sin(latitude)*math.sin(declination)
    b = math.cos(latitude)*math.cos(declination)
    sine = max(-1.0, min(1.0, a+b*math.cos(hour_angle)))
    elevation = math.degrees(math.asin(sine))
    irradiance = peak_irradiance*sine if sine > 1e-12 else 0.0
    if abs(b) < 1e-12:
        daylight = 24.0 if a > 1e-12 else 0.0
        sunrise = sunset = None
    elif -a/b <= -1:
        daylight, sunrise, sunset = 24.0, None, None
    elif -a/b >= 1:
        daylight, sunrise, sunset = 0.0, None, None
    else:
        half_day = 12*math.acos(-a/b)/math.pi
        daylight = 2*half_day
        sunrise, sunset = 12-half_day, 12+half_day
    return Sunlight(irradiance, elevation, math.degrees(declination), daylight, sunrise, sunset)
