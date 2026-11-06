"""Conservative cruise fast path, real swept separation and bounded lane count."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from ats.airspace import above_ground,ground_ceiling
from ats.config import Config
from ats.simulation import Simulation
from ats.drone import DroneState as DS
from ats.traffic import conflicts
from ats.checkpoint import save_state,load_state


def aircraft():
    c=Config();c.fleet.airspace_mode='shared_lanes'
    return Simulation(c)


def test_whole_high_segment_skips_ground_sampling_without_skipping_motion(monkeypatch):
    s=aircraft();d=s.drone;z=s.cruise(d)
    d.position[:]=[10.,8.,z];d.state=DS.TO_DIG;s._starts={0:d.position.copy()}
    def fail(*a,**kw):raise AssertionError('No terrain sampling above the proven scene ceiling')
    monkeypatch.setattr(s.terrain,'sample',fail)
    s._move(d,[11.,8.,z],DS.SCOOP,1.)
    assert d.state==DS.SCOOP and s.state.value=='RUNNING'
    np.testing.assert_array_equal(d.position,[11.,8.,z])


def test_low_segment_still_samples_and_rejects_an_obstacle(monkeypatch):
    s=aircraft();d=s.drone;d.position[:]=[10.,8.,1.];d.state=DS.TO_DIG;s._starts={0:d.position.copy()}
    calls=[]
    def ground(x,y):calls.append((x,y));return 2.
    monkeypatch.setattr(s.terrain,'sample',ground)
    before=d.position.copy();s._move(d,[11.,8.,1.],DS.SCOOP,1.)
    assert calls and s.state.value=='BLOCKED'
    np.testing.assert_array_equal(d.position,before)


def test_an_untracked_revision_edit_raises_bound_and_cannot_hide_under_fast_path():
    s=aircraft();d=s.drone;old=s.cruise(d)
    assert above_ground(s,np.array([10.,8.,old]),np.array([11.,8.,old]))
    s.terrain.height[32,42]=50.;s.terrain.revision+=1
    assert ground_ceiling(s)>=50.+s.config.base.roof_height
    assert not above_ground(s,np.array([10.,8.,old]),np.array([11.,8.,old]))


def test_controlled_dump_and_avalanche_preserve_the_declared_ground_bound():
    s=aircraft();t=s.terrain;bound=t.clearance_surface
    t.deposit(25.,25.,200.);t.relax(20)
    assert float(t.height.max())<=t.clearance_surface
    assert t.clearance_surface==bound


def test_replacing_hardware_updates_envelope_even_without_advancing_clock():
    s=aircraft();old=ground_ceiling(s)
    s.base.layout_spec={'platforms':[dict(dimensions_m=[6.,3.,30.])]}
    assert ground_ceiling(s)>=30.+s.terrain.clearance_surface>old
    s.base.layout_spec={'platforms':[dict(dimensions_m=[6.,3.,40.])]}
    assert ground_ceiling(s)>=40.+s.terrain.clearance_surface


def test_high_altitude_does_not_bypass_other_aircraft():
    s=aircraft();c=s.config;c.fleet.count=2;c.fleet.models=[];s=Simulation(c);a,b=s.drones;z=s.cruise(a)
    a.position[:]=[10.,8.,z];b.position[:]=[11.,8.,z]
    s._starts={d.identifier:d.position.copy() for d in s.drones};before=a.position.copy()
    s._move(a,[12.,8.,z],DS.SCOOP,1.)
    np.testing.assert_array_equal(a.position,before)
    assert a.traffic_yields==1 and s.state.value=='RUNNING'


def test_vertical_broad_phase_keeps_a_crossing_ascent_and_ignores_separated_cruise():
    a=SimpleNamespace(identifier=0,position=np.array([1.,0.,10.]))
    b=SimpleNamespace(identifier=1,position=np.array([1.,0.,20.]))
    old=np.array([0.,0.,10.]);new=np.array([2.,0.,10.])
    starts={0:old,1:np.array([1.,0.,0.])}
    assert conflicts(a,old,new,[a,b],starts,2.) is b
    starts[1]=np.array([1.,0.,20.])
    assert conflicts(a,old,new,[a,b],starts,2.) is None


def test_global_palette_reuses_only_four_levels_with_common_ground_floor():
    from tests.test_regional import region_group
    p=region_group(count=3)
    for u in p.units:u.simulation.config.fleet.airspace_mode='shared_lanes'
    heights=[u.simulation.cruise(d) for u in p.units for d in u.simulation.drones]
    assert len(set(heights))==4 and max(heights)-min(heights)==9.
    assert all(h>=ground_ceiling(p.active.simulation)+6. for h in heights)


def test_planner_budgets_the_actual_common_lane_height():
    s=aircraft();d=s.drone
    trip=s._choose_trip(d)
    assert trip is not None and trip.altitude==s.cruise(d)
    assert trip.energy<=d.battery-d.config.battery*d.config.reserve


def test_committed_sortie_keeps_its_height_but_a_new_trip_is_budgeted_for_new_hardware():
    from ats.airspace import cruise
    s=aircraft();d=s.drone;trip=s._choose_trip(d)
    d.trip=trip;d.state=DS.TO_DIG;d.flight_altitude_m=trip.altitude
    s.base.layout_spec={'platforms':[dict(dimensions_m=[6.,3.,30.])]}
    assert s.cruise(d)==trip.altitude and cruise(s,d)>trip.altitude
    next_trip=s._choose_trip(d,airborne=True)
    assert next_trip is not None and next_trip.altitude==cruise(s,d)


def test_outside_window_segment_cannot_use_a_ground_bound():
    s=aircraft();z=s.cruise(s.drone)
    assert not above_ground(s,np.array([1.,1.,z]),np.array([-1.,1.,z]))


def test_common_lane_snapshot_continuation_keeps_inventory_energy_and_clock(tmp_path):
    from ats.experiments import experiment_config
    from ats.maps import with_map_size
    c=experiment_config(with_map_size(Config(),64),drones=4,panel_area=20.);c.fleet.airspace_mode='shared_lanes'
    a=Simulation(c);a.run(75);path=tmp_path/'airborne.atsstate';save_state(path,a)
    b=load_state(path).simulation;a.run(125);b.run(125)
    assert a.report()==b.report()
    for x,y in zip(a.drones,b.drones):np.testing.assert_array_equal(x.position,y.position)
    assert abs(a.report()['mass_error_kg'])<1e-7 and abs(a.report()['energy_error_j'])<.002


def test_self_solar_highest_pose_is_bounded_during_all_opening_fractions():
    from tests.test_self_solar import world
    s=world(288);w=s.logistics;a=w.solar_array;bound=ground_ceiling(s)
    for fraction in (0.,.25,.5,1.):
        for q in a.sections:q.open_fraction=fraction
        a._obstacle_key=None
        for q in a.sections:
            point=a.centre(w,q)
            assert a.obstacle_height(w,point)<=bound
