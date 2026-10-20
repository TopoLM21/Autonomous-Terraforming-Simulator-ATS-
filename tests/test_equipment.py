from copy import deepcopy
import json
import numpy as np
import pytest
from ats.config import Config,KWH
from ats.simulation import Simulation,MissionState
from ats.equipment import (BatteryModule,PanelModule,LaserEmitter,BaseEquipment,DroneEquipment,
                           equipment_data,load_equipment_data,resolve_config,retained_energy)
from ats.productivity import Productivity


def balanced(sim):
    r=sim.report()
    assert abs(r['mass_error_kg'])<1e-7
    assert abs(r['energy_error_j'])<1e-4


def land(sim):
    sim.request_landing()
    for _ in range(2000):
        if all(d.grounded and d.cargo==0 for d in sim.drones):return
        sim.step()
        assert sim.state==MissionState.RUNNING,sim.reason
    pytest.fail('Fleet did not land')


def test_equipment_mass_capacity_and_solar_weighted_efficiency():
    e=BaseEquipment()
    assert e.mass==1000
    assert sum(v for k,v in e.breakdown().items() if k!='total_kg')==1000
    assert e.capacity==30*KWH and e.panel_area==20
    e.panels.append(PanelModule(quantity=5,area_m2=2,mass_kg=10,efficiency=.3))
    assert e.panel_efficiency==pytest.approx((20*.2+10*.3)/30)
    assert e.mass==1050
    d=DroneEquipment();assert d.mass==105 and d.capacity==5*KWH
    c=resolve_config(Config());m=c.fleet.models[0];old_power=m.hover_power
    m.equipment.batteries[0].quantity+=1;m.equipment.bucket.volume_m3=.10
    m.equipment.resolve(m,1600)
    assert m.empty_mass==pytest.approx(120) and m.payload==120
    assert m.hover_power==pytest.approx(old_power*(120/105)**1.5)


def test_live_base_inventory_preserves_simulation_while_drone_flies():
    s=Simulation();s.run(20);assert not s.drone.grounded
    terrain=s.terrain;height=terrain.height.copy();target=terrain.target.copy()
    time=s.time;position=s.drone.position.copy();battery=s.base.battery;drone_battery=s.drone.battery
    c=deepcopy(s.config);c.base.equipment.batteries[0].quantity+=1
    c.base.equipment.panels[0].quantity+=5;c.base.equipment.landing_slots=2
    pad=s.slot_position(0).copy();generated=s.base.generated
    assert s.request_equipment(c)=='applied'
    assert s.terrain is terrain and s.time==time
    np.testing.assert_array_equal(height,s.terrain.height);np.testing.assert_array_equal(target,s.terrain.target)
    np.testing.assert_array_equal(position,s.drone.position);np.testing.assert_array_equal(pad,s.slot_position(0))
    assert s.base.battery==battery and s.drone.battery==drone_battery
    assert s.config.base.battery==35*KWH and s.config.base.panel_area==25
    assert s.config.base.mass==1090  # 50 battery + five 4 kg panels + 20 pad
    s.step();assert s.base.generated>generated
    balanced(s)


def test_added_cells_empty_removed_cells_export_charge_proportionally():
    s=Simulation();s.base.battery=18*KWH;s._initial_energy=23*KWH
    c=deepcopy(s.config);c.base.equipment.batteries[0].quantity=8
    s.request_equipment(c);assert s.base.battery==18*KWH
    c=deepcopy(s.config);c.base.equipment.batteries[0].quantity=4
    s.request_equipment(c);assert s.base.battery==9*KWH
    assert s.equipment_exported_energy==9*KWH
    balanced(s)
    c=deepcopy(s.config);c.base.equipment.batteries[0].capacity_kwh=6
    s.request_equipment(c);assert s.base.battery==0
    assert s.equipment_exported_energy==18*KWH
    balanced(s)
    assert retained_energy([BatteryModule(quantity=2)],[BatteryModule(quantity=1),BatteryModule(quantity=1)],100)==100


@pytest.mark.parametrize('during_scoop',[False,True])
def test_drone_changes_wait_for_cargo_delivery_and_keep_history(during_scoop):
    s=Simulation()
    for _ in range(100):
        s.step()
        if s.drone.cargo>0 and (during_scoop or s.drone.cargo==60):break
    initial_cargo=s.drone.cargo
    assert 0<initial_cargo<=60
    c=deepcopy(s.config);c.fleet.models[0].equipment.bucket.volume_m3=.01875
    c.fleet.models[0].equipment.batteries[0].quantity=3
    assert s.request_equipment(c)=='pending'
    assert s.drone.config.payload==60 and s.drone.cargo==initial_cargo
    for _ in range(1000):
        before=s.drone.battery
        s.step();balanced(s)
        if s.pending_equipment is None:break
    assert s.pending_equipment is None and s.drone.grounded and s.drone.cargo==0
    assert s.drone.config.payload==30 and s.drone.config.battery==7.5*KWH
    assert s.drone.battery<=before+1e-6  # empty newly installed pack
    assert s.drone.productivity.total_mass==60
    assert s.time>0 and s.trips==1
    assert s.run(10000)['state']=='COMPLETED'
    balanced(s)


def test_retiring_and_delivering_drones_preserves_energy_history_and_ids():
    c=Config();c.fleet.count=2;c.fleet.charge_slots=2
    s=Simulation(c);s.run(80);land(s)
    old=s.drones[1];assert old.consumed>0
    old_energy=old.battery;old_consumed=s.report()['consumed_kwh']
    c=deepcopy(s.config);c.fleet.count=1;c.fleet.models=c.fleet.models[:1];c.base.equipment.landing_slots=1
    assert s.request_equipment(c)=='applied'
    assert s.retired_drones==[old] and old.retired_at==s.time
    assert s.equipment_exported_energy==old_energy
    assert s.report()['consumed_kwh']==old_consumed
    c=deepcopy(s.config);c.fleet.count=2;c.fleet.models.append(deepcopy(c.drone))
    c.base.equipment.landing_slots=2
    assert s.request_equipment(c)=='applied'
    assert s.drones[1].serial==3 and s.drones[1].created_at==s.time
    assert s.equipment_imported_energy==s.drones[1].battery
    assert s.report()['retired_drones'][0]['serial']==2
    balanced(s)


def test_pad_removal_rejects_occupied_slot_atomically():
    c=Config();c.fleet.count=2;c.fleet.charge_slots=2;s=Simulation(c)
    proposed=deepcopy(s.config);proposed.base.equipment.landing_slots=1
    before=s.report()
    with pytest.raises(ValueError,match='свободное'):s.request_equipment(proposed)
    assert s.report()==before
    c=Config();c.base.y=5;s=Simulation(c)
    proposed=deepcopy(s.config);proposed.base.equipment.landing_slots=6
    before=s.report()
    with pytest.raises(ValueError,match='посадочное'):s.request_equipment(proposed)
    assert s.report()==before


def test_base_without_storage_charges_directly_from_sun():
    s=Simulation();c=deepcopy(s.config);c.base.equipment.batteries=[]
    s.request_equipment(c);assert s.base.battery==0
    # Replace empty drone batteries on the dock; removed cell energy is exported.
    c=deepcopy(s.config);c.fleet.models[0].equipment.batteries[0].capacity_kwh=3
    s.request_equipment(c);assert s.drone.battery==0
    for _ in range(10):s.step();balanced(s)
    assert s.drone.battery>0 and s.base.battery==0
    assert s.base.charging_losses>0


def test_equipment_roundtrip_and_laser_hardware():
    s=Simulation();c=deepcopy(s.config)
    c.base.equipment.emitters=[LaserEmitter(drone_number=1)]
    c.fleet.models[0].equipment.receiver.installed=True
    c=load_equipment_data(json.loads(json.dumps(equipment_data(c))),s.config)
    assert equipment_data(c)==equipment_data(load_equipment_data(equipment_data(c),s.config))
    s.request_equipment(c);assert s.config.base.mass==1030 and s.drone.config.empty_mass==110
    assert s.report()['laser_transfer_enabled'] is True
    assert s.equipment_imported_energy==0
    s.run(500);balanced(s)
    assert s.drone.laser_received>0 and s.base.laser_input>0


def test_pending_installation_can_be_cancelled_without_config_change():
    s=Simulation();s.run(20);c=deepcopy(s.config);c.fleet.models[0].equipment.bucket.volume_m3*=2
    assert s.request_equipment(c)=='pending'
    s.cancel_equipment()
    assert s.pending_equipment is None and not s.landing_requested
    assert s.drone.config.payload==60 and s.time==20
    balanced(s)


def test_delivery_without_charge_or_socket_rejected_atomically():
    s=Simulation();c=deepcopy(s.config);c.fleet.count=2
    model=deepcopy(c.drone);model.initial_charge=0;c.fleet.models.append(model)
    before=s.report()
    with pytest.raises(ValueError,match='заряд'):s.request_equipment(c)
    assert s.report()==before


def test_larger_bucket_and_radius_finish_conservatively():
    s=Simulation();c=deepcopy(s.config);e=c.fleet.models[0].equipment
    e.bucket.volume_m3=.075;e.bucket.radius_m=.75;e.bucket.mass_kg=25
    s.request_equipment(c);r=s.run(10000)
    assert r['state']=='COMPLETED'
    assert r['drones'][0]['payload_kg']==120 and r['drones'][0]['empty_mass_kg']==115
    assert r['trips']<20
    assert r['drones'][0]['transported_mass_kg']==pytest.approx(r['excavated_mass_kg'])
    balanced(s)


def test_daily_productivity_midnight_and_rolling_24h():
    p=Productivity();p.record(1,0,60,1600,'dump');p.record(2,1,20,1600,'fill')
    r=p.report(2,1)
    assert r['today_mass_kg']==20 and r['last24_mass_kg']==80
    assert r['transported_volume_m3']==.05
    assert r['daily'][0]['dump_mass_kg']==60 and r['daily'][1]['fill_mass_kg']==20
    assert p.report(86401,2)['last24_mass_kg']==20
    assert p.report(86402,2)['last24_mass_kg']==0
    assert p.report(86402,2)['transported_mass_kg']==80


def test_productivity_counts_unload_not_pickup_and_uses_solar_midnight():
    c=Config();c.start_hour=23.99;s=Simulation(c)
    for _ in range(100):
        s.step()
        if s.drone.cargo:break
    assert s.drone.cargo>0 and s.report()['drones'][0]['transported_mass_kg']==0
    for _ in range(200):
        s.step()
        if s.drone.productivity.total_mass:break
    r=s.report()['drones'][0]
    assert r['transported_mass_kg']==60 and r['today_mass_kg']==60
    assert r['daily'][0]['mission_day']==1 and r['daily'][0]['deliveries']==1
    balanced(s)


def test_invalid_inventory_rejected_before_state_changes():
    s=Simulation();c=deepcopy(s.config);c.fleet.models[0].equipment.batteries=[]
    before=s.report()
    with pytest.raises(ValueError,match='батарея'):s.request_equipment(c)
    assert s.report()==before
    c=deepcopy(s.config);c.base.equipment.emitters=[LaserEmitter(),LaserEmitter()]
    with pytest.raises(ValueError,match='один'):s.request_equipment(c)
    assert s.report()==before
