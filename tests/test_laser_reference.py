from ats.laser_profiles import with_laser_profile
from tests.test_laser import hover,balanced
from ats.config import Config
import pytest


def reference(count=1):
    c=Config();c.fleet.count=count;c.drone.initial_charge=.5
    return with_laser_profile(c,'realistic_reference')


def test_laser_reference_power():
    s=hover(reference());s.step();link=s.laser_links[0]
    assert link.input_w==pytest.approx(5000)
    assert link.received_w==pytest.approx(600,rel=.005)
    assert link.receiver_heat_w==pytest.approx(900,rel=.005)
    assert s.drone.config.equipment.receiver.mass_kg==2.8
    assert s.base.config.equipment.emitters[0].mass_kg==300
    balanced(s)


def test_laser_blocked_or_overheated():
    s=hover(reference());s.drone.position[2]=0;s.step()
    assert s.drone.laser_received==0;balanced(s)
    s=hover(reference());s.drone.receiver_temperature_c=70
    s.drone.config.equipment.receiver.cooling_w_k=0;s.step()
    assert s.drone.laser_received==0
    assert s.laser_links[0].code=='thermal_throttling';balanced(s)


def test_reference_shared_station_and_local_dust():
    s=hover(reference(2));s.step()
    assert s.base.laser_input==pytest.approx(5000)
    assert sum(d.laser_power_w for d in s.drones)<601;balanced(s)
    clear=hover(reference());dusty=hover(reference());dusty.dust.field[:]=100
    clear.step();dusty.step()
    assert dusty.drone.laser_power_w<clear.drone.laser_power_w*.5
    balanced(dusty)


def test_legacy_profile_restores_user_experiment():
    from dataclasses import asdict
    from ats.equipment import resolve_config,LaserEmitter
    c=resolve_config(Config());c.base.equipment.emitters=[LaserEmitter(input_power_kw=197,efficiency=.7)]
    c.fleet.models[0].equipment.receiver.installed=True
    c.fleet.models[0].equipment.receiver.max_power_kw=23
    old=asdict(c.base.equipment.emitters[0]);receiver=asdict(c.fleet.models[0].equipment.receiver)
    new=with_laser_profile(with_laser_profile(c,'realistic_reference'),'existing_legacy')
    assert asdict(new.base.equipment.emitters[0])==old
    assert asdict(new.fleet.models[0].equipment.receiver)==receiver
