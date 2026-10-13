from copy import deepcopy
import numpy as np
import pytest
from ats.config import Config,KWH
from ats.drone import DroneState as DS
from ats.equipment import LaserEmitter,resolve_config,equipment_data,load_equipment_data
from ats.laser import visible
from ats.scenarios import scenario
from ats.simulation import Simulation


def laser_config(count=1,power=20,receiver_limit=8,charge=.5):
    c=Config();c.fleet.count=count;c.drone.initial_charge=charge
    resolve_config(c)
    c.base.equipment.emitters=[LaserEmitter(drone_number=i+1,input_power_kw=power,range_m=60) for i in range(count)]
    for model in c.fleet.models:
        model.equipment.receiver.installed=True;model.equipment.receiver.max_power_kw=receiver_limit
    return c


def hover(c):
    s=Simulation(c)
    # Isolate a real one-second hovering load from route selection.
    s.slots=[None]*len(s.slots)
    for i,d in enumerate(s.drones):
        d.slot=None;d.state=DS.WAIT_HOVER;d.position=np.array([10.+i*3,16.+i*4,10.])
    return s


def balanced(s):
    r=s.report()
    assert abs(r['mass_error_kg'])<1e-7
    assert abs(r['energy_error_j'])<1e-4
    assert all(d.laser_step_energy==0 for d in s.drones)
    assert s.base.laser_input==pytest.approx(sum(d.laser_received for d in s.drones+s.retired_drones)+s.base.laser_losses,abs=1e-5)


def test_optical_conversion_offsets_hover_draw_and_debits_base():
    s=hover(laser_config());d=s.drone;before=d.battery;base=s.base.battery;power=d.power()
    s.step()
    assert d.laser_power_w==4500  # 20 kW electric * 50% emitter * 45% receiver
    assert d.battery==pytest.approx(before+4500-power,abs=1e-7)
    assert d.consumed==pytest.approx(power)
    assert s.base.battery==pytest.approx(base+s.base.generated-20000,abs=1e-7)
    assert s.base.laser_losses==15500
    balanced(s)


def test_full_battery_can_power_motors_directly_and_does_not_overflow():
    s=hover(laser_config(power=1000,receiver_limit=100,charge=1))
    for _ in range(50):
        s.step();assert s.drone.battery==pytest.approx(s.drone.config.battery,abs=1e-7);balanced(s)
    assert s.drone.laser_received==pytest.approx(s.drone.consumed,abs=1e-5)
    assert s.base.laser_input==pytest.approx(s.drone.consumed/.225,abs=1e-5)
    assert 'battery' in s.report()['lasers'][0]['limits']


@pytest.mark.parametrize('cause,code',[('receiver','no_receiver'),('emitter','no_emitter'),('disabled','disabled'),('power','disabled'),('range','out_of_range'),('moving','base_moving')])
def test_missing_or_unavailable_link_has_no_energy_transfer(cause,code):
    c=laser_config()
    if cause=='receiver':c.fleet.models[0].equipment.receiver.installed=False
    if cause=='emitter':c.base.equipment.emitters=[]
    if cause=='disabled':c.base.equipment.emitters[0].enabled=False
    if cause=='power':c.base.equipment.emitters[0].input_power_kw=0
    if cause=='range':c.base.equipment.emitters[0].range_m=1
    s=hover(c)
    if cause=='moving':s.base.destination=s.base.position[:2]+[.2,0]
    before=s.drone.battery;s.step()
    assert s.report()['lasers'][0]['code']==code
    assert s.drone.battery<before and s.base.laser_input==s.drone.laser_received==0
    balanced(s)


def test_receiver_limit_saturates_instead_of_creating_unlimited_power():
    readings=[]
    for power in [20,40,100000]:
        s=hover(laser_config(power=power));s.step();readings.append((s.drone.laser_power_w,s.base.laser_input,s.drone.battery))
        balanced(s)
        if power>20:assert 'receiver' in s.report()['lasers'][0]['limits']
    assert readings[0][0]==4500
    assert readings[1][0]==pytest.approx(8000)
    assert readings[1]==pytest.approx(readings[2])
    assert readings[1][1]==pytest.approx(8000/.225)


def test_receiver_area_and_beam_divergence_change_captured_power():
    readings=[]
    for area in [.25,10]:
        c=laser_config();c.base.equipment.emitters[0].beam_radius_m=1
        c.base.equipment.emitters[0].divergence_mrad=10
        c.fleet.models[0].equipment.receiver.area_m2=area
        s=hover(c);s.step();balanced(s);readings.append(s.report()['lasers'][0])
    small,large=readings
    assert small['capture_fraction']<1 and large['capture_fraction']==1
    assert small['received_power_w']==pytest.approx(4500*small['capture_fraction'])
    assert large['received_power_w']==4500
    assert 'beam_area' in small['limits']


def test_sand_ridge_and_base_body_block_visibility():
    s=Simulation();a=np.array([5.,16.,2.]);b=np.array([15.,16.,10.])
    assert visible(a,b,s.terrain,base=s.base)
    s.terrain.height[64,40]=20  # ridge at x=10 m
    assert not visible(a,b,s.terrain,base=s.base)
    s.terrain.height[64,40]=0
    assert not visible(a,np.array([4.,16.,.2]),s.terrain,base=s.base)
    assert visible(a,np.array([2.,16.,2.]),s.terrain,base=s.base)


def test_other_drone_can_occlude_the_real_link():
    s=hover(laser_config(count=2));s.drones[1].position=np.array([7.5,16.,6.])
    s.step()
    assert s.report()['lasers'][0]['code']=='occluded'
    assert s.drone.laser_received==0
    balanced(s)


def test_shared_laser_bus_has_no_duplicated_energy_when_base_is_low():
    c=laser_config(count=2);c.start_hour=0;c.base.initial_charge=10000/c.base.battery
    s=hover(c);s.step()
    assert s.base.laser_input==pytest.approx(10000)
    assert [d.laser_power_w for d in s.drones]==pytest.approx([1125,1125])
    assert s.base.battery==pytest.approx(0)
    assert all('base_energy' in link['limits'] for link in s.report()['lasers'])
    balanced(s)


def test_socket_charging_has_priority_over_lasers_on_the_same_base():
    c=laser_config(count=2);c.start_hour=0;c.base.initial_charge=15000/c.base.battery
    c.fleet.models[0].initial_charge=.2
    s=hover(c);d=s.drones[0];d.position=s.slot_position(0);d.slot=0;d.state=DS.CHARGING;s.slots[0]=0
    before=d.battery;s.step()
    assert d.battery==pytest.approx(before+7200)
    assert s.base.charging_input==8000 and s.base.laser_input==pytest.approx(7000)
    assert s.drones[1].laser_power_w==pytest.approx(1575)
    balanced(s)


def test_no_storage_uses_current_sunlight_and_night_has_no_free_power():
    for hour in [12,0]:
        c=laser_config();c.base.equipment.batteries=[];c.start_hour=hour
        s=hover(c);s.step();balanced(s)
        assert s.base.battery==0 and s.base.curtailed==0
        assert s.base.laser_input==pytest.approx(s.base.generated)
        assert s.drone.laser_received==pytest.approx(s.base.generated*.225)
        if hour==0:assert s.report()['lasers'][0]['code']=='no_base_energy'


def test_wireless_full_charge_releases_air_queue_and_resumes_work():
    c=laser_config(power=1000,receiver_limit=100,charge=1-10000/(5*KWH))
    s=hover(c);s.drone.charge_requested=True;s.queue=[0]
    s.step()
    assert s.drone.battery==pytest.approx(s.drone.config.battery,abs=1e-7)
    assert not s.drone.charge_requested and s.drone.charge_cycles==1
    assert s.queue==[] and s.drone.slot is None and s.slots==[None]
    assert s.drone.state in (DS.TO_DIG,DS.LOWER_GRABBER)
    balanced(s)


def test_live_disable_preserves_state_and_safe_autonomous_return():
    c=laser_config(power=1000,receiver_limit=100,charge=1)
    s=Simulation(c);s.run(20);assert s.drone.laser_received>0
    c=deepcopy(s.config);c.base.equipment.emitters[0].enabled=False
    time=s.time;soil=s.terrain.height.copy();received=s.drone.laser_received
    assert s.request_equipment(c)=='applied' and s.time==time
    np.testing.assert_array_equal(soil,s.terrain.height)
    before=s.drone.battery;s.step()
    assert s.drone.laser_received==received and s.drone.battery<before
    assert s.run(10000)['state']=='COMPLETED'
    assert s.drone.minimum_charge>=s.drone.config.battery*s.drone.config.reserve-1e-6
    balanced(s)


def test_laser_history_survives_drone_retirement_and_equipment_roundtrip():
    c=laser_config(count=2,charge=1)
    c=load_equipment_data(equipment_data(c),c)
    assert c.base.equipment.emitters[0].enabled and c.base.equipment.emitters[0].divergence_mrad==2
    s=Simulation(c);s.run(80);s.request_landing()
    for _ in range(500):
        s.step()
        if all(d.grounded for d in s.drones):break
    assert all(d.grounded for d in s.drones)
    total=s.report()['laser_received_kwh'];assert s.drones[1].laser_received>0
    c=deepcopy(s.config);c.fleet.count=1;c.fleet.models=c.fleet.models[:1]
    s.request_equipment(c)
    assert s.report()['laser_received_kwh']==total
    assert s.report()['retired_drones'][0]['laser_received_kwh']>0
    balanced(s)


def test_stopped_step_settles_all_delivered_energy_even_for_unprocessed_drone():
    c=laser_config(count=2,charge=1);c.fleet.models[0].initial_charge=0
    s=hover(c);s.step()
    assert s.state.value=='BLOCKED'
    assert s.drones[0].battery>0 and s.drones[1].battery<=s.drones[1].config.battery
    balanced(s)


def test_first_autonomous_flight_lasts_longer_with_identical_laser_hardware():
    records=[]
    for enabled in [False,True]:
        c=resolve_config(scenario('medium',42));c.base.equipment.emitters=[LaserEmitter(enabled=enabled,range_m=60)]
        c.fleet.models[0].equipment.receiver.installed=True;s=Simulation(c)
        airborne=False
        for _ in range(2000):
            s.step();airborne|=not s.drone.grounded
            if airborne and s.drone.grounded:break
        assert airborne and s.drone.grounded
        assert s.drone.battery>=s.drone.config.battery*s.drone.config.reserve-1e-6
        balanced(s);records.append(s.report())
    off,on=records
    assert off['drones'][0]['empty_mass_kg']==on['drones'][0]['empty_mass_kg']
    assert on['drones'][0]['last_flight_seconds']>off['drones'][0]['last_flight_seconds']*1.15
    assert on['drones'][0]['trips']>off['drones'][0]['trips']
    assert off['laser_received_kwh']==0 and on['laser_received_kwh']>0
