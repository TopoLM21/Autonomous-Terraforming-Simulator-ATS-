"""A batch courier must not serialize the diggers or strand a partial final load."""
import json
import subprocess
import sys

from ats.drone import DroneState as DS
from ats.experiments import experiment_config
from ats.inventory import Location as L
from ats.scenarios import scenario
from ats.simulation import Simulation


def world(**values):
    config=experiment_config(scenario('medium'))
    for name,value in values.items():setattr(config.logistics,name,value)
    return Simulation(config)


def balanced(sim):
    r=sim.report(full=False)
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<1e-4
    assert not sim.logistics.direct_ids.intersection(sim.logistics.active)
    assert len(sim.logistics.active)<=1
    sim.logistics.inventory.validate()


def fill(sim, container):
    index=sim.logistics.containers.index(container)
    container.soil_kg=sim.terrain.scoop(14.25+index*.5,16,container.assigned_limit_kg)
    assert container.soil_kg>=container.assigned_limit_kg-1e-8
    sim.logistics.inventory.move(container.id,L.FILLED_QUEUE,'worksite',sim.logistics.rack,'FULL_READY')
    sim._terrain_active=True


def test_default_launch_is_ats2_and_old_mode_is_explicit():
    for flags,ats2 in [([],True),(['--mode','legacy'],False)]:
        p=subprocess.run([sys.executable,'-m','ats','--headless','--steps','0',*flags],
                         capture_output=True,text=True,encoding='utf-8',timeout=30)
        assert p.returncode==0,p.stderr
        report=json.loads(p.stdout)
        assert ('logistics_version' in report)==ats2
        if ats2:
            assert report['drone_count']==4 and report['courier_strategy']=='batch'
            assert all(d['attachment']=='GRABBER' for d in report['drones'])


def test_all_four_drones_dig_before_the_batch_threshold():
    sim=world();w=sim.logistics
    assert all(c.location==L.WORKSITE_RACK for c in w.containers)
    sim.run(25)
    diggers={j.executor for j in w.jobs if j.kind=='EXCAVATE_DIRECT'}
    assert diggers=={d.serial for d in sim.drones}
    assert not w.active and not w.courier_active and w.courier_id is None
    assert all(d.attachment=='GRABBER' for d in sim.drones)
    balanced(sim)


def test_six_of_eight_containers_recall_one_courier_while_others_keep_digging():
    sim=world();w=sim.logistics
    for c in w.containers[:5]:fill(sim,c)
    sim.step();assert not w.courier_active
    fill(sim,w.containers[6]);sim.step()
    assert w.courier_active and w.courier_reason=='FILLED_THRESHOLD'
    assert sum(target=='CONTAINER_HOOK' for target in w.tool_targets.values())==1
    changed=False;hauled=False;diggers=set()
    for _ in range(500):
        sim.step();balanced(sim)
        changed |= any(d.state==DS.CHANGING_TOOL for d in sim.drones)
        hauled |= bool(w.active)
        diggers.update(d.serial for d in sim.drones if d.identifier in w.direct_ids and d.identifier!=w.courier_id)
        if hauled and changed and len(diggers)>=2:break
    assert changed and hauled and len(diggers)>=2


def test_krot_below_ten_percent_requests_a_real_battery_service():
    above=world(excavator_initial_soc=.11);above.step()
    assert not above.logistics.courier_active
    assert not any(j.kind=='DELIVER_EXCAVATOR_BATTERY' for j in above.logistics.jobs)
    sim=world(excavator_initial_soc=.099);w=sim.logistics
    old=list(w.excavator.battery_ids);sim.step()
    assert w.courier_active and w.courier_reason=='LOW_SOC'
    assert any(j.kind=='DELIVER_EXCAVATOR_BATTERY' for j in w.jobs)
    for _ in range(1200):
        sim.step();balanced(sim)
        if w.excavator.swaps:break
    assert w.excavator.swaps==1
    assert set(old).isdisjoint(w.excavator.battery_ids)
    assert w.excavator.energy(w.inventory)>.8*w.excavator.capacity(w.inventory)


def test_courier_returns_to_digging_after_draining_a_batch():
    sim=world();w=sim.logistics
    for c in w.containers[:6]:fill(sim,c)
    sim.step();courier=w.courier_id
    assert courier is not None
    returned=False
    for _ in range(2000):
        sim.step();balanced(sim)
        if not w.courier_active and sim.drones[courier].attachment=='GRABBER' and courier in w.direct_ids:
            returned=True;break
    assert returned and sim.terrain.remaining>sim.config.terrain.completion_volume
    assert w.delivered_mass>0


def test_partial_final_batch_completes_instead_of_waiting_for_six_full_bins():
    sim=Simulation(experiment_config(scenario('demo')));r=sim.run(10000)
    assert r['state']=='COMPLETED',r['reason']
    assert r['container_cycles']>0 and r['excavator']['excavated_kg']>0
    assert r['courier']['batches']>=1 and not r['courier']['active']
    assert all(c.soil_kg==0 for c in sim.logistics.containers)
    assert not sim.logistics.active and all(d.grounded for d in sim.drones)
    balanced(sim)


def test_low_krot_without_spares_does_not_stop_independent_diggers():
    sim=world(excavator_initial_soc=.08,excavator_battery_count=2)
    sim.run(25)
    assert sim.state.value=='RUNNING' and sim.logistics.courier_active
    assert any(j.kind=='EXCAVATE_DIRECT' for j in sim.logistics.jobs)
    balanced(sim)
