from copy import deepcopy
import pytest
from ats.config import Config
from ats.work_schedule import WorkSchedule
from ats.experiments import experiment_config
from ats.simulation import Simulation
from ats.checkpoint import save_state,load_state


def world(*,hour=9.,day=172,mode='hours',hours=6.,count=4):
    c=experiment_config(Config(),mode='hybrid',drones=count,containers=0,drone_spare_sets=0)
    c.logistics.excavator_enabled=False;c.start_hour=hour
    c.base.latitude_deg=31.078;c.base.start_day_of_year=day
    c.operating_schedule=WorkSchedule(mode,60,273,hours).to_data()
    return Simulation(c)


def balanced(s):
    r=s.report(full=False)
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002


def test_calendar_wraps_year_without_assuming_the_northern_hemisphere():
    schedule=WorkSchedule('continuous',300,60)
    assert len(schedule.days())==126
    assert schedule.contains(365) and schedule.contains(1) and not schedule.contains(200)
    assert schedule.allows(-31.,355,0.)


def test_daylight_minimum_and_fixed_hours_are_distinct():
    light=WorkSchedule('daylight',1,365).summary(31.078)
    assert 9<light['min_work_hours']<11 and light['max_work_hours']>13
    fixed=WorkSchedule('hours',1,365,6.).summary(31.078)
    assert fixed['min_work_hours']==fixed['max_work_hours']==6
    assert not fixed['insufficient_daylight_days']
    assert WorkSchedule('hours',1,365,15.).summary(31.078)['insufficient_daylight_days']


def test_polar_night_has_no_day_shift_but_continuous_work_remains_an_energy_question():
    assert WorkSchedule('daylight',1,365).interval(80.,355) is None
    assert WorkSchedule('continuous',1,365).allows(80.,355,0.)


def test_outside_shift_stops_work_but_not_time_or_contact_charging():
    c=deepcopy(world(hour=23.).config)
    for model in c.fleet.models:model.initial_charge=.2
    s=Simulation(c);before=s.terrain.height.copy();energy=s.drone.battery
    s.run(120)
    assert s.time==120 and not s.work_allowed and s.state.value=='RUNNING'
    assert s.drone.battery>energy and all(d.grounded for d in s.drones)
    assert (s.terrain.height==before).all()
    assert 'WORK_WINDOW' in s.logistics.wait_reason
    balanced(s)


def test_shift_opens_without_restarting_the_mission():
    s=world(hour=9-30/3600)
    s.run(20);assert all(d.grounded for d in s.drones)
    s.run(30)
    assert s.work_allowed and any(not d.grounded for d in s.drones)
    balanced(s)


def test_work_season_is_inclusive_and_can_close_without_blocking_the_mission():
    s=world(day=274,mode='continuous');s.run(20)
    assert not s.work_allowed and s.state.value=='RUNNING'
    assert s.trips==0 and 'рабочего периода' in s.work_wait_reason
    assert WorkSchedule().allows(31.,273,12.)
    balanced(s)


def test_snapshot_continuation_preserves_calendar_and_closed_shift(tmp_path):
    s=world(hour=9-30/3600);s.run(20)
    path=tmp_path/'calendar.atsstate';save_state(path,s)
    other=load_state(path).simulation
    for _ in range(120):s.step();other.step()
    assert s.report()==other.report() and s.work_allowed==other.work_allowed
    assert s.config.operating_schedule==other.config.operating_schedule
    balanced(s);balanced(other)


def test_closing_a_shift_finishes_cargo_without_new_dispatches():
    s=world(hour=12.)
    for _ in range(500):
        s.step()
        if any(d.cargo>0 for d in s.drones):break
    assert any(d.cargo>0 for d in s.drones)
    # Moving to a closed calendar is also how a new date/latitude takes effect:
    # no stale interval may survive the configuration change.
    s.config.operating_schedule=WorkSchedule('hours',172,172,.25).to_data()
    # At 12:08 this 15-minute noon window is closed.
    s.config.start_hour=12.2
    before=s.excavated_mass
    for _ in range(600):
        s.step()
        if all(d.grounded for d in s.drones):break
    assert not s.work_allowed and all(d.grounded and abs(d.cargo)<1e-8 for d in s.drones)
    assert s.state.value in ('RUNNING','COMPLETED'),s.reason
    for _ in range(30):s.step()
    assert all(d.grounded for d in s.drones) and s.excavated_mass>=before
    balanced(s)


def test_latitude_change_recomputes_daylight_interval():
    s=world(hour=7.,day=355,mode='daylight')
    s.config.operating_schedule=WorkSchedule('daylight',1,365).to_data()
    assert not s.work_allowed
    s.base.config.latitude_deg=-31.
    assert s.work_allowed


@pytest.mark.parametrize('data',[{'mode':'night'},{'hours':0},{'first_day':0},{'last_day':366}])
def test_invalid_calendar_is_rejected(data):
    with pytest.raises(ValueError):WorkSchedule.from_data(data)
