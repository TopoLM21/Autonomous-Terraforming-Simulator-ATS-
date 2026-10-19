"""Assigned module slots are independent; energy remains a shared finite resource."""
from copy import deepcopy
import pytest
from ats.config import Config
from ats.experiments import experiment_config,save_configuration,load_configuration
from ats.inventory import BatteryModule,ChargingSlot,Location as L
from ats.simulation import Simulation


def reserve_world(**values):
    c=experiment_config(Config(),drones=2,drone_spare_sets=3)
    c.logistics.excavator_enabled=False;c.logistics.container_count=0
    for name,value in values.items():setattr(c.logistics,name,value)
    return Simulation(c)


def test_every_spare_charges_simultaneously_even_when_old_port_count_is_one():
    s=reserve_world(charger_ports=1,drone_spare_initial_soc=.2)
    w=s.logistics;service=w.battery_service
    packs=[b for b in w.inventory.select(BatteryModule) if '_spare_' in b.id]
    slots=w.inventory.select(ChargingSlot)
    assert len(packs)==12 and len(slots)==16
    assert len({slot.module_id for slot in slots})==16
    assert all(service.slots[b.id].state=='FULL' for b in w.inventory.select(BatteryModule,L.BASE_STORAGE) if b.soc==1)
    before={b.id:b.energy_j for b in packs};identities=set(w.inventory.items)
    s.step();r=s.report();ports=r['battery_service']
    assert set(w.inventory.items)==identities
    assert len(ports['ports'])==12 and ports['capacity']==16 and ports['queue']==[]
    assert all(b.energy_j>before[b.id] for b in packs)
    assert ports['module_input_power_w']==pytest.approx(12*1500.,abs=1e-7)
    assert sum(b.energy_j-before[b.id] for b in packs)==pytest.approx(12*1500*.9)
    assert len([p for p in ports['slots'] if p['state']=='CHARGING'])==12
    assert len([p for p in ports['slots'] if p['state']=='EMPTY'])==4
    assert all(0<=p['input_power_w']<=p['rated_w']+1e-7 for p in ports['slots'])
    assert abs(r['energy_error_j'])<1e-4 and abs(r['mass_error_kg'])<1e-7


def test_slot_hardware_mass_and_fixed_pairing_survive_a_real_swap():
    c=experiment_config(Config(),drones=1)
    c.logistics.excavator_enabled=False;c.logistics.container_count=0
    c.logistics.drone_swap_seconds=3.
    c.fleet.models[0].initial_charge=.25
    s=Simulation(c);w=s.logistics;old=list(s.drone.battery_ids)
    mapping={id:slot.id for id,slot in w.battery_service.slots.items()}
    identities=set(w.inventory.items)
    control=deepcopy(c);control.logistics.individual_chargers=False
    shared=Simulation(control)
    assert s.base.config.mass-shared.base.config.mass==pytest.approx(len(mapping)*c.logistics.charger_slot_mass_kg)
    assert s.report()['initial_energy_kwh']==shared.report()['initial_energy_kwh']
    s.run(4)
    assert w.battery_service.drone_swap_count==1
    assert not set(old).intersection(s.drone.battery_ids)
    assert mapping=={id:slot.id for id,slot in w.battery_service.slots.items()}
    assert set(w.inventory.items)==identities
    assert all(w.battery_service.slots[id].input_power_w>0 for id in old)
    assert all(w.battery_service.slots[id].state=='EMPTY' for id in s.drone.battery_ids)
    assert abs(s.report()['energy_error_j'])<1e-4


def test_empty_base_does_not_charge_an_individual_slot():
    c=experiment_config(Config(),drones=1)
    c.logistics.excavator_enabled=False;c.logistics.container_count=0
    c.logistics.drone_spare_initial_soc=.2
    c.base.initial_charge=0;c.base.equipment.panels=[]
    s=Simulation(c);w=s.logistics
    packs=w.inventory.select(BatteryModule,L.BASE_STORAGE)
    before=[b.energy_j for b in packs]
    s.step()
    assert [b.energy_j for b in packs]==before
    assert all(w.battery_service.slots[b.id].state=='WAITING_POWER' for b in packs)
    assert w.battery_service.snapshot()['module_input_power_w']==0
    assert abs(s.report()['energy_error_j'])<1e-4


def test_battery_in_field_or_reserved_for_transport_has_no_base_charge():
    s=reserve_world(drone_spare_initial_soc=.2)
    w=s.logistics;a,b=w.inventory.select(BatteryModule,L.BASE_STORAGE)[:2]
    w.inventory.move(a.id,L.WORKSITE_RACK,'worksite',(16,16,0),'STORED_READY')
    w.inventory.reserve([b.id],'fixture_transport')
    before=(a.energy_j,b.energy_j)
    s.step()
    assert (a.energy_j,b.energy_j)==before
    assert w.battery_service.slots[a.id].state=='EMPTY'
    assert w.battery_service.slots[b.id].state=='RESERVED'
    assert abs(s.report()['energy_error_j'])<1e-4


def test_slot_mode_persists_and_light_telemetry_omits_slot_rows(tmp_path):
    c=experiment_config(Config(),drone_spare_sets=2)
    path=tmp_path/'slots.json';save_configuration(path,c)
    clone=load_configuration(path)
    assert clone.logistics.individual_chargers
    s=Simulation(clone);s.run(5)
    assert 'slots' not in s.report(full=False)['battery_service']
    assert len(s.report()['battery_service']['slots'])==len(s.logistics.inventory.select(ChargingSlot))
    identities=set(s.logistics.inventory.items)
    with pytest.raises(ValueError):
        duplicate=deepcopy(s.logistics.inventory.select(ChargingSlot)[0]);duplicate.id='duplicate_slot'
        s.logistics.inventory.add(duplicate)
    assert set(s.logistics.inventory.items)==identities
    from ats.experiments import benchmark_configs
    port_experiments=[config for name,config in benchmark_configs(c) if name.startswith('S7_ports_')]
    assert [config.logistics.charger_ports for config in port_experiments]==[1,2,4]
    assert all(not config.logistics.individual_chargers for config in port_experiments)


def test_all_individual_slots_share_a_finite_inverter_limit():
    s=reserve_world(drone_spare_initial_soc=.2,inverter_power_w=5000.)
    s.step();service=s.report()['battery_service']
    assert len(service['ports'])==12
    assert service['module_input_power_w']==pytest.approx(5000.,abs=1e-7)
    assert all(p['input_power_w']>0 for p in service['slots'] if p['module_id'] in service['ports'])
    assert abs(s.report()['energy_error_j'])<1e-4


@pytest.mark.parametrize('inverter,module_power',[(100000.,6000.),(10000.,2000.)])
def test_landing_dock_and_individual_slots_have_separate_limited_circuits(inverter,module_power):
    c=experiment_config(Config(),drones=2)
    c.logistics.excavator_enabled=False;c.logistics.container_count=0
    c.logistics.swap_station=False;c.logistics.drone_spare_initial_soc=.2
    c.logistics.inverter_power_w=inverter
    for model in c.fleet.models:model.initial_charge=.2
    s=Simulation(c);before=sum(d.battery for d in s.drones)
    s.step()
    assert sum(d.battery for d in s.drones)-before==pytest.approx(8000*.9)
    assert s.logistics.battery_service.snapshot()['module_input_power_w']==pytest.approx(module_power,abs=1e-7)
    assert s.base.charging_input==pytest.approx(8000+module_power)
    assert abs(s.report()['energy_error_j'])<1e-4
