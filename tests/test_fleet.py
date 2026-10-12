from copy import deepcopy
import numpy as np
import pytest
from ats.config import Config,KWH
from ats.simulation import Simulation
from ats.drone import DroneState as DS,Drone
from ats.solar_base import SolarBase
from ats.traffic import closest_separation
from ats.comparison import compare_fleets


def fleet(count=3,slots=1,mode='ground'):
    c=Config();c.fleet.count=count;c.fleet.charge_slots=slots;c.fleet.wait_mode=mode
    return c


@pytest.mark.parametrize('count,slots',[(1,1),(3,1),(3,3),(6,2)])
def test_fleet_completion_mass_energy_and_full_charge_departures(count,slots):
    s=Simulation(fleet(count,slots))
    departures=0
    for _ in range(12000):
        before=[d.state for d in s.drones]
        s.step()
        for previous,d in zip(before,s.drones):
            if previous in (DS.DOCKED,DS.CHARGING,DS.WAIT_GROUND) and d.state==DS.TAKEOFF:
                assert d.battery==pytest.approx(d.config.battery,abs=1e-6)
                departures+=1
            assert d.config.battery*d.config.reserve-1e-6<=d.battery<=d.config.battery+1e-6
            assert 0<=d.cargo<=d.config.payload+1e-9
        assert sum(v is not None for v in s.slots)<=slots
        assert len({v for v in s.slots if v is not None})==sum(v is not None for v in s.slots)
        assert abs(s.report()['mass_error_kg'])<1e-7
        for a in s.drones:
            for b in s.drones[a.identifier+1:]:
                assert closest_separation(s._starts[a.identifier],a.position,s._starts[b.identifier],b.position)>=s.config.fleet.separation-1e-9
        if s.state.value!='RUNNING':break
    r=s.report()
    assert r['state']=='COMPLETED',r
    assert departures>=20
    assert sum(d.trips for d in s.drones)==s.trips
    assert all(d.grounded and d.cargo==0 for d in s.drones)
    assert abs(r['energy_error_j'])<1e-4


def test_shared_charger_power_not_duplicated_between_slots():
    c=fleet(3,3)
    drones=[Drone(deepcopy(c.drone),np.zeros(3)) for _ in range(3)]
    for d in drones:d.battery=0
    b=SolarBase(c.base,np.zeros(3)); initial=b.battery
    b.tick(1,0,drones,[True]*3)
    assert sum(d.battery for d in drones)==pytest.approx(8000*.9)
    assert b.battery==pytest.approx(initial-8000)
    assert b.charging_losses==pytest.approx(800)
    assert all(d.battery==pytest.approx(2400) for d in drones)


def test_no_charging_in_air_or_on_waiting_ground_pad():
    s=Simulation(fleet(3,1))
    for d in s.drones:d.battery=4*KWH
    before=[d.battery for d in s.drones]
    s.step()
    assert s.drones[0].battery>before[0]
    assert s.drones[1].battery==before[1]
    assert s.drones[2].battery==before[2]
    assert s.drones[1].state==DS.WAIT_GROUND


def test_air_queue_spends_energy_and_emergency_lands_preserving_reserve():
    s=Simulation(fleet(3,1,'air'))
    seen=False; fallback=False
    for _ in range(200000):
        old=[d.state for d in s.drones]
        charges=[d.battery for d in s.drones]
        s.step()
        for previous,charge,d in zip(old,charges,s.drones):
            if previous==DS.WAIT_HOVER and d.state==DS.WAIT_HOVER:
                seen=True;assert d.battery<charge
            if previous==DS.WAIT_HOVER and d.state==DS.WAIT_LANDING:fallback=True
            assert d.battery>=d.config.battery*d.config.reserve-1e-6
        if s.state.value!='RUNNING':break
    assert seen and fallback
    assert s.state.value=='COMPLETED',s.report()
    assert abs(s.report()['energy_error_j'])<.002
    assert abs(s.report()['mass_error_kg'])<1e-7


def test_swept_crossing_is_detected_and_detour_changes_actual_route():
    assert closest_separation(np.array([0,0,0]),np.array([4,0,0]),np.array([2,-2,0]),np.array([2,2,0]))==0
    s=Simulation(fleet(2,1));a,b=s.drones
    a.position=np.array([10.,10.,10.]);b.position=np.array([12.,10.,10.]);a.state=DS.TO_DIG
    visited=[]
    for _ in range(30):
        s._starts={d.identifier:d.position.copy() for d in s.drones}
        old=a.position.copy();s._move(a,[14,10,10],DS.TO_DIG,1);visited.append(a.position.copy())
        assert closest_separation(old,a.position,b.position,b.position)>=2-1e-9
        if np.linalg.norm(a.position-np.array([14,10,10]))<1e-8:break
    assert a.traffic_yields>0
    assert any(abs(p[1]-10)>1 for p in visited)
    np.testing.assert_allclose(a.position,[14,10,10])


def test_mixed_models_and_deterministic_comparison():
    c=fleet(2,2)
    light=deepcopy(c.drone);light.name='Лёгкий';light.empty_mass=75;light.payload=30;light.battery=3*KWH;light.hover_power=12000
    c.fleet.models=[deepcopy(c.drone),light]
    a=Simulation(c);r=a.run(20000)
    assert r['state']=='COMPLETED'
    assert r['drones'][1]['payload_kg']==30
    assert all(row['trips']>0 for row in r['drones'])
    results=compare_fleets(c,[1,2],20000)
    assert [row['drone_count'] for row in results]==[1,2]
    assert all(row['state']=='COMPLETED' for row in results)
    assert results==compare_fleets(c,[1,2],20000)


def test_base_does_not_teleport_waiting_drones_when_slots_are_insufficient():
    s=Simulation(fleet(3,1));s.run(12000)
    assert s.state.value=='COMPLETED'
    assert s.base.relocations==0


def test_multiple_dust_sources_add_instead_of_averaging():
    from ats.dust import Dust
    c=Config();dust=Dust(c)
    dust.emit(1,(16,16,6),6,1);amount=dust.field.sum()
    dust.emit(1,(16,16,6),6,1)
    assert dust.field.sum()==pytest.approx(2*amount)


def test_disabled_charger_blocks_after_using_initial_full_battery():
    c=Config();c.base.charge_power=0
    s=Simulation(c);s.run(10000)
    assert s.state.value=='BLOCKED'
    assert 'отключено' in s.reason
    assert s.trips==1
