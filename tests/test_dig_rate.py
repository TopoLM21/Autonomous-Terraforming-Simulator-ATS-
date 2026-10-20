import pytest
from ats.config import Config
from ats.drone import DroneState as DS
from ats.simulation import Simulation


@pytest.mark.parametrize('payload,rate,minimum,duration',[(60.,6.,10.,10),(120.,6.,10.,20),(60.,7.,.1,9)])
def test_soil_is_taken_gradually_at_finite_rate_with_energy_and_mass_accounting(payload,rate,minimum,duration):
    c=Config();c.drone.payload=payload;c.drone.dig_rate_kg_s=rate;c.drone.scoop_seconds=minimum
    s=Simulation(c)
    for _ in range(100):
        s.step()
        if s.drone.state==DS.SCOOP:break
    d=s.drone;assert d.state==DS.SCOOP and d.cargo==0
    assert d.trip.mass==pytest.approx(payload)
    steps=0
    while d.state==DS.SCOOP:
        before=d.cargo;energy=d.battery;expected=s.flight_power(d)
        s.step();steps+=1
        assert 0<d.cargo-before<=rate+1e-8
        assert energy-d.battery==pytest.approx(expected,abs=1e-7)
        report=s.report()
        assert abs(report['mass_error_kg'])<1e-7 and abs(report['energy_error_j'])<1e-4
        assert steps<=duration
    assert steps==duration and d.cargo==pytest.approx(payload)
    assert d.state==DS.LIFT_GRABBER


def test_invalid_rate_or_specific_work_rejected():
    c=Config();c.drone.dig_rate_kg_s=0
    with pytest.raises(ValueError,match='копания'):c.validate()
    c.drone.dig_rate_kg_s=6;c.drone.digging_j_m3=-1
    with pytest.raises(ValueError,match='копания'):c.validate()
