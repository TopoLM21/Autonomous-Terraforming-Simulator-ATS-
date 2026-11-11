"""Shared physical inventory, parallel digging and addressed service of 1–10 Krots."""
from copy import deepcopy
import json
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.inventory import BatteryModule,Location as L
from ats.checkpoint import save_state,load_state
from ats.costs import default_catalog,quote,excavation_cost,project_excavation_cost
from ats.work_schedule import WorkSchedule


def config(robots=3,soil=False):
    c=experiment_config(with_map_size(Config(),128),drones=4,containers=8*robots)
    c.terrain.cell_size=.5;c.terrain.n=256;c.terrain.dump_rect=(80.,80.,40.,40.)
    c.logistics.excavator_count=robots;c.fleet.airspace_mode='shared_lanes'
    c.plan=PlanConfig([PlanSegment([[24.,60.],[104.,60.]],width=8.,height=-.3)])
    if soil:
        from ats.soil import SoilProfile
        c.soil_profile=SoilProfile.qattara()
    c.cost_catalog=default_catalog();return c


def balanced(s):
    r=s.report();assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002
    if 'soil' in r:assert r['soil']['max_material_mass_error_kg']<1e-7
    s.logistics.inventory.validate();s.logistics.sync_batteries()


@pytest.mark.parametrize('robots,soil',[(3,False),(10,True)])
def test_parallel_real_excavation_ids_materials_inventory_and_exact_resume(tmp_path,robots,soil):
    s=Simulation(config(robots,soil));w=s.logistics
    assert s.state.value=='RUNNING',s.reason
    assert len(w.excavators)==robots and w.excavator is w.excavators[0]
    assert len({e.id for e in w.excavators})==robots
    assert len({id(w.inventory) for e in w.excavators})==1
    before=set(w.inventory.items)
    for _ in range(55):s.step()
    assert all(e.excavated_kg>0 for e in w.excavators)
    assert sum(e.excavated_kg for e in w.excavators)<=s.excavated_mass+1e-7
    owners={o.reserved_by for o in w.containers if o.reserved_by and o.reserved_by.startswith('excavator_cycle')}
    assert len(owners)==robots
    assert len({b for e in w.excavators for b in e.battery_ids})==robots*w.config.excavator_battery_slots
    balanced(s)
    path=tmp_path/'fleet.atsstate';save_state(path,s);other=load_state(path).simulation
    for _ in range(70):s.step();other.step()
    assert s.report()==other.report() and set(w.inventory.items)==before
    assert all(e._world is other.logistics for e in other.logistics.excavators)
    balanced(s);balanced(other)


def test_shared_side_dock_services_correct_robot_and_survives_mid_swap(tmp_path):
    c=config();c.logistics.battery_swap_seconds=2.;c.start_hour=23.
    c.operating_schedule=WorkSchedule('hours',1,365,6.).to_data()
    s=Simulation(c);w=s.logistics
    capital=quote(s.config,layout=s.base.layout_spec if s.base.config.layout is not None else None,world=w)['total']
    for e in w.excavators:
        for id in e.battery_ids:
            b=w.inventory.items[id];old=b.energy_j;b.energy_j=b.capacity_j*.05;w.initial_extra_energy+=b.energy_j-old
        packs=w.battery_service.ready('excavator')[:c.logistics.excavator_battery_slots]
        for b in packs:w.inventory.move(b.id,L.WORKSITE_RACK,e.id,e.position,'STORED_READY')
    ids=set(w.inventory.items);s.step()
    assert w.battery_service.robot_swap is not None
    path=tmp_path/'swap.atsstate';save_state(path,s);other=load_state(path).simulation
    for _ in range(30):s.step();other.step()
    assert s.report()==other.report()
    assert all(e.swaps==1 for e in w.excavators)
    assert set(w.inventory.items)==ids
    assert all(w.inventory.items[id].holder_id==e.id for e in w.excavators for id in e.battery_ids)
    assert all(e.energy(w.inventory)>e.capacity(w.inventory)*.9 for e in w.excavators)
    assert quote(s.config,layout=s.base.layout_spec if s.base.config.layout is not None else None,world=w)['total']==capital
    balanced(s);balanced(other)


def test_jobs_address_separate_racks_and_reserve_each_spare_only_once():
    s=Simulation(config());w=s.logistics
    for e in w.excavators:
        for id in e.battery_ids:
            b=w.inventory.items[id];before=b.energy_j;b.energy_j=0;w.initial_extra_energy-=before
    w.battery_service.generate_jobs()
    jobs=[j for j in w.jobs if j.kind=='DELIVER_EXCAVATOR_BATTERY']
    assert len(jobs)==6 and len({j.resource_ids[0] for j in jobs})==6
    assert {j.excavator_serial for j in jobs}=={1,2,3}
    for j in jobs:
        e=w.robot(j.excavator_serial)
        assert j.destination==tuple(e.position) and w.inventory.items[j.resource_ids[0]].reserved_by==j.id
    for job in jobs:w.cancel_job(job.id)
    assert all(not w.inventory.items[j.resource_ids[0]].reserved_by for j in jobs)
    balanced(s)


def test_collective_lift_targets_second_machine_and_leaves_other_machine_in_place():
    c=config(2);c.logistics.idle_digging=False;c.logistics.direct_allowed=False;c.logistics.battery_swap_seconds=2.
    s=Simulation(c);w=s.logistics;first,second=w.excavators;original=first.position.copy()
    from ats.excavator import ExcavatorState as ES
    destination=second.position.copy();destination[1]+=8
    assert second.stable(destination,s.terrain)
    second.destination=destination.copy();second.state=ES.WAITING_RELOCATION
    w.inventory.move('group_rig',L.WORKSITE_RACK,second.id,second.position,'STORED_READY')
    for _ in range(600):
        s.step()
        assert s.state.value=='RUNNING',s.reason
        if w.lift_count:break
    assert w.lift_count==1 and second.relocations==1
    np.testing.assert_allclose(second.position,destination,atol=1e-7)
    np.testing.assert_array_equal(first.position,original)
    assert w.excavator is first
    balanced(s)


def test_robot_bodies_and_reserved_destinations_are_real_obstacles():
    s=Simulation(config());a,b,c=s.logistics.excavators
    assert not b.stable(a.position,s.terrain)
    destination=a.position.copy();destination[0]+=5
    a.destination=destination
    assert not b.stable(destination,s.terrain)
    from copy import copy
    probe=copy(a)
    assert all(e.serial!=a.serial for e in probe.peers())  # route probes must not collide with their own original


def test_capital_cost_per_m3_uses_excavation_and_no_haul_double_count(tmp_path):
    s=Simulation(config(3));initial=excavation_cost(s)
    assert initial['capital_per_m3'] is None and initial['currency']=='USD'
    for _ in range(100):s.step()
    metric=excavation_cost(s)
    assert metric['capital_per_m3']==pytest.approx(metric['capital_cost']/metric['excavated_m3'])
    before=deepcopy(metric)
    s.drone.productivity.record(s.time,0,10.,s.config.terrain.density,'dump')
    s.logistics.delivered_mass+=10.  # a haul statistic is not another extraction
    assert excavation_cost(s)==before
    f=tmp_path/'cost.atsstate';save_state(f,s);other=load_state(f).simulation
    assert excavation_cost(other)==before
    assert json.loads(json.dumps(s.report(),allow_nan=False))['excavation_cost']['capital_per_m3']==metric['capital_per_m3']


def test_layered_volume_and_group_weighting_and_unknown_currency():
    a=Simulation(config(1,True));b=Simulation(config(3,True))
    for _ in range(70):a.step();b.step()
    metric=project_excavation_cost([a,b]);qa,qb=quote(a.config),quote(b.config)
    assert metric['excavated_m3']==a.excavated_volume+b.excavated_volume
    assert metric['capital_per_m3']==pytest.approx((qa['total']+qb['total'])/(a.excavated_volume+b.excavated_volume))
    b.config.cost_catalog['currency']='RUB'
    assert project_excavation_cost([a,b])['capital_per_m3'] is None
    assert excavation_cost(b)['currency']=='RUB'
    a.config.cost_catalog=None
    assert excavation_cost(a)['capital_per_m3'] is None


def test_ten_machine_power_recipe_has_all_packs_chargers_mass_and_prices():
    from ats.complex_setup import SetupRequest,calculate,place
    from ats.power_design import calculate_power,PowerRequest
    from ats.scheduled_power import day_request
    c=config();c.base.latitude_deg=31.078;schedule=WorkSchedule('hours',172,172,6)
    one=calculate(c,SetupRequest(excavators=1,containers=8),schedule)
    ten=calculate(c,SetupRequest(excavators=10,containers=80),schedule)
    request=day_request(PowerRequest(**ten.design['request']),schedule,31.078,172)
    a=calculate_power(one.config,request);b=calculate_power(ten.config,request)
    assert b['robot_charging_kw']==pytest.approx(10*a['robot_charging_kw'])
    assert b['excavator_spare_modules_total']==10*b['excavator_spare_modules']
    assert ten.design['equipment_mass_kg']>one.design['equipment_mass_kg']
    assert ten.estimate['total']>one.estimate['total']
    assert all(audit['repeatable'] for audit in ten.design['season_checks'])
    session=place(ten);s=session.simulation;w=s.logistics
    packs=[p for p in w.inventory.select(BatteryModule) if p.compatibility=='excavator']
    assert len(packs)==10*w.config.excavator_battery_count
    assert w.battery_service.capacity==b['module_charging_slots']
    assert quote(s.config,layout=s.base.layout_spec,world=w)['total']==pytest.approx(ten.estimate['total'])
    assert next(row for row in ten.estimate['rows'] if row['key']=='excavator')['quantity']==10
    balanced(s)


def test_template_draft_prices_and_calendar_do_not_change_the_active_complex():
    from ats.complexes import ComplexTemplate
    s=Simulation(config());s.config.operating_schedule=WorkSchedule().to_data()
    template=ComplexTemplate.capture(s.config)
    template.config.cost_catalog['prices']['excavator']=5.
    template.config.operating_schedule['hours']=6.
    assert s.config.cost_catalog['prices']['excavator']==70000.
    assert s.config.operating_schedule['hours']==8.


def test_maximum_mixed_fleet_is_physical_and_balanced():
    from ats.complex_setup import SetupRequest,calculate,place
    c=config(10);c.base.latitude_deg=31.078
    result=calculate(c,SetupRequest(drones=16,excavators=10,containers=80),WorkSchedule('hours',172,172,6))
    session=place(result);s=session.simulation
    assert len(s.drones)==16 and len(s.logistics.excavators)==10
    for _ in range(20):session.complex_project.step()
    assert s.state.value=='RUNNING',s.reason
    balanced(s)


def test_excess_machines_park_off_landing_pads_without_inventing_excavation():
    c=config(10);c.plan=PlanConfig([])
    s=Simulation(c);assert s.state.value=='RUNNING',s.reason
    assert len(s.logistics.excavators)==10
    pads=[s.slot_position(i) for i in range(len(s.slots))]+[s.wait_position(i) for i in range(len(s.drones))]
    for e in s.logistics.excavators:
        assert e.stable(e.position,s.terrain)
        assert all(np.linalg.norm(e.position[:2]-p[:2])>s.config.fleet.separation for p in pads)
    s.step();assert s.excavated_mass==0
    balanced(s)


@pytest.mark.parametrize('count',[-1,11,True,2.5])
def test_invalid_fleet_quantity_is_rejected(count):
    c=config();c.logistics.excavator_count=count
    with pytest.raises(ValueError):c.validate()
