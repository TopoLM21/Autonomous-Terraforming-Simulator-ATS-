from copy import deepcopy
import numpy as np
import pytest
from ats.config import Config,KWH
from ats.simulation import Simulation
from ats.plan import PlanConfig,PlanSegment,save_plan,load_plan,line_distance
from ats.profiles import profiles,segments_for
from ats.drone import DroneState as DS


def assert_balances(s):
    r=s.report()
    assert abs(r['mass_error_kg'])<1e-7
    assert abs(r['energy_error_j'])<1e-4
    assert all(d.config.battery*d.config.reserve-1e-6<=d.battery<=d.config.battery+1e-6 for d in s.drones)


def test_above_threshold_launches_without_charging_and_continues_between_transfers():
    c=Config();c.drone.initial_charge=.8
    s=Simulation(c);s.step()
    assert s.drone.state==DS.TAKEOFF
    assert s.drone.battery==c.drone.battery*.8
    for _ in range(500):
        s.step()
        if s.trips>=3:break
    assert s.trips>=3
    assert s.base.charging_input==0
    assert s.drone.charge_cycles==0
    assert_balances(s)


def test_low_charge_latches_until_full_even_after_crossing_threshold():
    c=Config();c.drone.initial_charge=.25
    s=Simulation(c);crossed=False;departed=False
    for _ in range(3000):
        previous=s.drone.state
        s.step()
        if s.drone.state==DS.CHARGING and s.drone.battery>c.drone.battery*c.drone.charge_threshold:
            crossed=True
            assert s.drone.charge_requested
        if previous==DS.CHARGING and s.drone.state==DS.TAKEOFF:
            departed=True;assert s.drone.battery==pytest.approx(c.drone.battery,abs=1e-6);break
    assert crossed and departed
    assert s.drone.charge_cycles==1
    assert_balances(s)


def test_threshold_changes_when_charging_starts():
    c=Config();c.drone.initial_charge=.5;c.drone.charge_threshold=.6
    s=Simulation(c);s.step();assert s.drone.state==DS.CHARGING
    c=Config();c.drone.initial_charge=.5;c.drone.charge_threshold=.3
    s=Simulation(c);s.step();assert s.drone.state==DS.TAKEOFF
    c.drone.charge_threshold=.1
    with pytest.raises(ValueError,match='Порог'):c.validate()


@pytest.mark.parametrize('count,slots',[(1,1),(3,1),(6,2)])
def test_dig_then_brush_fill_uses_existing_stockpile_and_preserves_both_ledgers(count,slots):
    c=Config();c.fleet.count=count;c.fleet.charge_slots=slots
    s=Simulation(c);assert s.run(20000)['state']=='COMPLETED'
    before=s.terrain.height.copy();mass=s.terrain.mass;energy=s.report()['energy_error_j'];stock=s.terrain.stockpile_volume
    s.apply_plan(PlanConfig([PlanSegment([[15,16],[17,16]],'fill',1.5,0,'Засыпка канавы')]))
    np.testing.assert_array_equal(s.terrain.height,before)
    assert s.terrain.mass==mass
    assert s.report()['energy_error_j']==energy
    assert s.terrain.remaining_fill>0
    assert s.run(80000)['state']=='COMPLETED',s.report()
    assert s.filled_mass>1000
    assert sum(d.reclaimed_mass for d in s.drones)>1000
    assert s.terrain.stockpile_volume<stock
    assert s.terrain.remaining_fill<.05
    assert all(d.grounded and d.cargo==0 for d in s.drones)
    assert_balances(s)


def test_simultaneous_cut_and_fill_reuses_excavation_directly():
    c=Config();c.fleet.count=3;c.fleet.charge_slots=2
    c.plan=PlanConfig([PlanSegment([[12,12],[14,12]],'dig',1,-.3,'Выемка'),PlanSegment([[16,12],[18,12]],'fill',1,.2,'Насыпь')])
    s=Simulation(c)
    assert s.terrain.remaining_cut>0 and s.terrain.remaining_fill>0
    assert s.run(80000)['state']=='COMPLETED',s.report()
    assert s.filled_mass>1000
    assert s.terrain.height.max()>=.19
    assert_balances(s)


def test_fill_without_a_finite_source_is_blocked_and_does_not_create_sand():
    c=Config();c.plan=PlanConfig([PlanSegment([[12,12],[14,12]],'fill',1,.2)])
    s=Simulation(c);before=s.terrain.height.copy()
    assert s.run(10)['state']=='BLOCKED'
    assert 'Недостаточно грунта' in s.reason
    np.testing.assert_array_equal(s.terrain.height,before)
    assert_balances(s)


def test_adjacent_incompatible_cut_and_fill_targets_are_rejected_before_an_endless_soil_cycle():
    s=Simulation();target=s.terrain.target.copy();height=s.terrain.height.copy()
    plan=PlanConfig([PlanSegment([[12,12],[14,12]],'dig',1,-.5),PlanSegment([[14,12],[16,12]],'fill',1,.5)])
    with pytest.raises(ValueError,match='круче угла откоса'):s.apply_plan(plan)
    np.testing.assert_array_equal(s.terrain.target,target)
    np.testing.assert_array_equal(s.terrain.height,height)
    assert_balances(s)


def test_initially_met_fill_target_remains_active_after_conservative_soil_redistribution():
    c=Config();c.plan=PlanConfig([PlanSegment([[12,12]],'fill',2.5,0)])
    s=Simulation(c);assert s.terrain.remaining_fill==0
    # Equal and opposite height changes represent redistribution after planning.
    s.terrain.height[48,48]-=1
    s.terrain.height[100,104]+=1
    s.terrain.revision+=1;s._terrain_active=True
    assert s.terrain.remaining_fill==pytest.approx(c.terrain.cell_size**2)
    assert s.run(10000)['state']=='COMPLETED',s.report()
    assert s.filled_mass>0
    assert s.terrain.remaining_fill<.05
    assert_balances(s)


def test_plan_changes_require_grounded_empty_drones_and_protect_pads():
    s=Simulation();before=s.terrain.target.copy();mass=s.terrain.mass
    with pytest.raises(ValueError,match='посадочн'):
        s.apply_plan(PlanConfig([PlanSegment([[5,16]],'dig',1,-.3)]))
    np.testing.assert_array_equal(s.terrain.target,before)
    assert s.terrain.mass==mass
    s.run(10)
    with pytest.raises(ValueError,match='посадк'):
        s.apply_plan(PlanConfig([PlanSegment([[12,12]],'dig',1,-.3)]))
    s.request_landing()
    for _ in range(1000):
        s.step()
        if all(d.grounded for d in s.drones):break
    assert all(d.grounded and d.cargo==0 for d in s.drones)
    s.apply_plan(PlanConfig([PlanSegment([[12,12]],'dig',1,-.3)]))
    assert not s.landing_requested
    assert_balances(s)


def test_partial_fill_keeps_excess_cargo_without_overfilling():
    c=Config();c.plan=PlanConfig([PlanSegment([[12,12]],'fill',.25,.1)])
    s=Simulation(c);needed=s.terrain.remaining_fill*c.terrain.density
    received=s.terrain.deposit_fill(12,12,needed+60)
    assert received==pytest.approx(needed)
    assert np.all(s.terrain.height<=s.terrain.target+1e-12)
    assert s.terrain.mass-s.terrain.initial_volume*c.terrain.density==pytest.approx(received,abs=1e-7)


def test_landing_request_prevents_waiting_drone_from_starting_transfer_to_a_free_slot():
    c=Config();c.fleet.count=2;c.fleet.charge_slots=1
    waiting=deepcopy(c.drone);waiting.initial_charge=.26
    c.fleet.models=[deepcopy(c.drone),waiting]
    s=Simulation(c)
    while s.slots[0] is not None:s.step()
    assert s.drones[1].state==DS.WAIT_GROUND
    s.request_landing();s.step()
    assert s.drones[1].state==DS.WAIT_GROUND
    for _ in range(1000):
        s.step()
        if all(d.grounded for d in s.drones):break
    assert all(d.grounded and d.cargo==0 for d in s.drones)
    assert s.state.value=='RUNNING'
    assert_balances(s)


def test_saved_plan_reproduces_starting_terrain_and_mission(tmp_path):
    s=Simulation();s.run(10000)
    s.apply_plan(PlanConfig([PlanSegment([[15,16],[17,16]],'fill',1.5,0)]))
    path=tmp_path/'plan.json';save_plan(path,s.config)
    config=load_plan(path,Config())
    np.testing.assert_allclose([config.base.x,config.base.y],s.base.position[:2])
    a,b=Simulation(deepcopy(s.config)),Simulation(config)
    np.testing.assert_array_equal(a.terrain.height,b.terrain.height)
    assert a.terrain.stockpile_volume>0
    assert a.run(20000)==b.run(20000)
    assert a.state.value=='COMPLETED'
    assert_balances(a)


def test_segment_and_joined_profiles_sample_actual_surface_without_connecting_disjoint_segments():
    c=Config();c.plan=PlanConfig([PlanSegment([[10,10],[12,10]],'dig',1,-.3,'А'),PlanSegment([[16,12],[16,15]],'dig',1,-.5,'Б')])
    s=Simulation(c);segments=segments_for(c)
    sections=profiles(s.terrain,segments,None,'cross',.5)
    assert len(sections)==2
    assert min(sections[0].target)==pytest.approx(-.3)
    assert min(sections[1].target)==pytest.approx(-.5)
    assert np.all(sections[0].points[:,0]==11)
    assert np.all(sections[1].points[:,1]==13.5)
    joined=profiles(s.terrain,segments,None,'along')[0]
    assert np.nanmax(joined.axis)==pytest.approx(5)
    assert np.count_nonzero(np.isnan(joined.current))==1
    assert joined.boundaries==(2.,)
    assert np.nanmin(joined.target)==pytest.approx(-.5)
    assert len(profiles(s.terrain,segments,1,'along'))==1
    np.testing.assert_allclose(line_distance(np.array([10,11,12]),10,[[10,10],[12,10]]),0)
