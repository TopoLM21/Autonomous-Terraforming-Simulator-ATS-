from copy import deepcopy
import numpy as np
import pytest
from ats.config import Config
from ats.simulation import Simulation
from ats.logistics_config import LogisticsConfig
from ats.inventory import Inventory, BatteryModule, SoilContainer, Location as L


def logistics_config(count=1, **values):
    # Strict logistics fixtures isolate Krot/container physics. Work sharing has
    # dedicated integration coverage below and is enabled in the application.
    values.setdefault('idle_digging',False)
    values.setdefault('individual_chargers',False)  # retained shared-port scenarios
    c=Config(logistics=LogisticsConfig(**values))
    c.fleet.count=count
    return c


def test_container_payload_with_tare():
    c=SoilContainer('c',5.)
    assert c.capacity_for(85,2,1600)==78
    assert c.capacity_for(60,2,1600)==53
    assert c.capacity_for(1000,2,1600)==85


def test_inventory_reservation():
    inv=Inventory();inv.add(SoilContainer('c',5.));inv.add(BatteryModule('b',11.))
    assert inv.reserve(['c'],'a')
    assert not inv.reserve(['b','c'],'other')
    assert inv.items['b'].reserved_by is None
    with pytest.raises(ValueError):inv.move('c',L.DRONE_CARGO,'d',(0,0,0),'FULL_IN_TRANSIT','other')
    inv.release('a');assert inv.reserve(['c'],'other')


def test_container_cycle_mass():
    s=Simulation(logistics_config(excavator_enabled=False,container_count=1))
    l=s.logistics;c=l.containers[0]
    mass=s.terrain.scoop(16,16,53)
    c.soil_kg=mass
    c.location=L.FILLED_QUEUE;c.holder_id='worksite';c.position=(16.,16.,0.)
    s._terrain_active=True
    visited=set()
    for _ in range(600):
        s.step();visited.add(c.state)
        assert abs(s.report()['mass_error_kg'])<1e-7
        assert abs(s.report()['energy_error_j'])<1e-4
        assert s.drone.cargo<=l.payload_limit(s.drone)+1e-8
        if c.cycles and not l.active:break
    assert c.cycles==1 and c.soil_kg==0 and c.location==L.WORKSITE_RACK
    assert {'FULL_IN_TRANSIT','EMPTY_AT_EXCAVATOR'}<=visited
    assert l.delivered_mass==pytest.approx(mass)
    assert s.drone.battery>=s.drone.config.battery*s.drone.config.reserve


def test_legacy_regression():
    s=Simulation();r=s.run(6000)
    assert s.logistics is None and r['state']=='COMPLETED'
    assert r['time_seconds']==2662 and r['trips']==20
    assert abs(r['energy_error_j'])<1e-4


def test_battery_slots_not_inventory():
    s=Simulation(logistics_config());e=s.logistics.excavator
    assert len([b for b in s.logistics.inventory.select(BatteryModule) if b.compatibility=='excavator'])==4
    assert len(e.battery_ids)==2
    assert e.capacity(s.logistics.inventory)==3*3_600_000
    assert e.mass(s.logistics.inventory)==182


def test_excavator_waits_for_empty_container():
    s=Simulation(logistics_config(container_count=0))
    for _ in range(20):s.logistics.excavator.tick(s.logistics)
    e=s.logistics.excavator
    assert e.state.value=='WAITING_EMPTY_CONTAINER' and e.excavated_kg==0
    assert s.terrain.remaining==s.terrain.initial_excavation
    assert abs(s.report()['energy_error_j'])<1e-4


def test_excavator_short_mission():
    c=logistics_config(count=2)
    c.trench.length=c.trench.width=.5;c.trench.depth=.1
    c.terrain.completion_volume=.002
    s=Simulation(c);r=s.run(10000)
    assert r['state']=='COMPLETED', (r['reason'],r.get('excavator'),r['jobs'][-5:])
    assert r['container_cycles']>0 and r['excavator']['cycles']>0
    assert abs(r['mass_error_kg'])<1e-7
    assert abs(r['energy_error_j'])<1e-4


def test_excavator_reach_and_slope():
    s=Simulation(logistics_config());e=s.logistics.excavator;t=s.terrain
    far=e.position+np.array([10.,0.,0.])
    t.height=t.target.copy()
    assert not e.can_drive(np.array([16.,16.,-1.]),t)
    mask=e.reachable_mask(t,s.logistics.inventory)
    assert not mask[int(e.position[1]/t.cell),int(e.position[0]/t.cell)]
    assert not mask[int(far[1]/t.cell),int(far[0]/t.cell)]
    t.height[:]=10;t.height[int(e.position[1]/t.cell),int(e.position[0]/t.cell)]=0
    assert not e.stable(e.position,t)


def test_battery_swap_atomic():
    c=logistics_config(count=2,excavator_initial_soc=.28)
    c.trench.length=c.trench.width=.75;c.trench.depth=.15
    s=Simulation(c);ids=set(s.logistics.inventory.items)
    for _ in range(10000):
        s.step()
        inv=s.logistics.inventory;e=s.logistics.excavator
        assert set(inv.items)==ids
        assert len(e.battery_ids)==2 and len(set(e.battery_ids))==2
        assert all(inv.items[id].location==L.EXCAVATOR_SLOT for id in e.battery_ids)
        assert abs(s.report()['energy_error_j'])<1e-4
        assert abs(s.report()['mass_error_kg'])<1e-7
        if s.state.value!='RUNNING':break
    r=s.report()
    assert r['state']=='COMPLETED',r['excavator']
    assert e.swaps>=1
    recovered=[j for j in s.logistics.jobs if j.kind=='RECOVER_EMPTY_EXCAVATOR_BATTERY' and j.status=='COMPLETED']
    assert len(recovered)>=2
    assert any(b.charged_energy>0 for b in inv.select(BatteryModule) if b.compatibility=='excavator')


def test_charging_energy_conservation():
    c=logistics_config(excavator_enabled=False,drone_spare_sets=0,charger_ports=1)
    c.base.initial_charge=0;c.base.panel_area=0;c.drone.initial_charge=.2
    s=Simulation(c);l=s.logistics
    l.inventory.add(BatteryModule('fixture_charge_pack',11.,energy_j=0.))
    before=[b.energy_j for b in l.inventory.select(BatteryModule)]
    consumers=l.battery_service.charging()
    assert len(consumers)==1
    s.base.tick(1,0,s.drones,[False],modules=consumers)
    assert [b.energy_j for b in l.inventory.select(BatteryModule)]==before
    s.base.battery=10000
    initial=sum(before)+10000
    s.base.tick(1,0,s.drones,[False],modules=consumers)
    after=sum(b.energy_j for b in l.inventory.select(BatteryModule))
    assert after-sum(before)==pytest.approx(1500*.9)
    assert after+s.base.battery+s.base.charging_losses==pytest.approx(initial)


def test_drone_swap_keeps_physical_energy():
    c=logistics_config();c.drone.initial_charge=.25
    s=Simulation(c);old=list(s.drone.battery_ids);l=s.logistics
    s.run(65)
    assert l.battery_service.drone_swap_count==1
    assert not set(old)&set(s.drone.battery_ids)
    assert all(l.inventory.items[id].location in (L.BASE_STORAGE,L.BASE_CHARGER) for id in old)
    assert abs(s.report()['energy_error_j'])<1e-4


def test_deadlock_recovery():
    s=Simulation(logistics_config(container_count=0));r=s.run(100)
    assert r['state']=='BLOCKED' and 'NO_CONTAINERS' in r['reason']
    s=Simulation(logistics_config(excavator_battery_count=2,excavator_initial_soc=.21))
    r=s.run(100)
    assert r['state']=='BLOCKED' and 'NO_SPARE_BATTERIES' in r['reason']


def test_collective_lift():
    from ats.cooperative_lift import lift_check
    s=Simulation(logistics_config(count=4));w=s.logistics;e=w.excavator
    check=lift_check(e,w.inventory,s.drones,w.config)
    assert check['feasible'] and check['mass_kg']==194
    assert check['equivalent_kg']==pytest.approx(194/(4*np.cos(np.pi/6)))
    for d in s.drones:d.config.equipment.payload_limit_kg=20
    assert lift_check(e,w.inventory,s.drones,w.config)['reason']=='BLOCKED_INSUFFICIENT_LIFT'
    for d in s.drones:d.config.equipment.payload_limit_kg=120;d.battery=1
    assert lift_check(e,w.inventory,s.drones,w.config)['reason']=='WAITING_GROUP_SOC'


def test_collective_lift_physical_chain():
    from ats.excavator import ExcavatorState as ES
    c=logistics_config(count=4);c.fleet.charge_slots=4
    s=Simulation(c);w=s.logistics;e=w.excavator
    original=e.position.copy();e.destination=e.position+np.array([3.,-1.,0.]);e.state=ES.WAITING_RELOCATION
    removed=False;flying=False
    arrival=None
    for _ in range(1200):
        s.step()
        assert abs(s.report()['energy_error_j'])<1e-4
        if w.group is not None and w.group.phase in ('LIFT','FLY','LOWER'):
            flying=True;assert not e.battery_ids
        if not e.battery_ids:removed=True
        if w.lift_count and arrival is None:arrival=e.position.copy()
        if w.lift_count and len(e.battery_ids)==2 and not w.active:break
        assert s.state.value=='RUNNING',s.reason
    assert flying and removed and w.lift_count>=1
    assert np.linalg.norm(arrival-original)>2
    assert e.battery_ids and any(j.kind=='DELIVER_EXCAVATOR_BATTERY' and j.status=='COMPLETED' for j in w.jobs)
    assert any(j.kind=='DELIVER_RIGGING' and j.status=='COMPLETED' for j in w.jobs)


@pytest.mark.parametrize('excavator',[False,True])
def test_direct_drone_digging(excavator):
    c=logistics_config(count=4,direct_allowed=True,excavator_enabled=excavator)
    c.fleet.charge_slots=4
    if excavator:c.trench.length=4
    s=Simulation(c);r=s.run(2000)
    assert any(j.kind=='EXCAVATE_DIRECT' and j.executor==s.drones[-1].serial for j in s.logistics.jobs)
    assert s.drones[-1].productivity.total_mass>0
    assert abs(r['mass_error_kg'])<1e-7
    assert abs(r['energy_error_j'])<1e-4


def test_completed_small_scenario():
    c=logistics_config(count=4,excavator_initial_soc=.28);c.fleet.charge_slots=4
    s=Simulation(c);r=s.run(10000)
    assert r['state']=='COMPLETED',(r['reason'],r['excavator'])
    assert r['excavator']['swaps']>=1 and r['container_cycles']>=1
    assert r['excavator']['bucket_kg']==0
    assert all(c.soil_kg==0 for c in s.logistics.containers)
    assert not s.logistics.active and all(d.grounded and d.cargo==0 for d in s.drones)
    assert abs(r['energy_error_j'])<1e-4 and abs(r['mass_error_kg'])<1e-7


def test_deterministic_headless():
    c=logistics_config(count=2,excavator_initial_soc=.28)
    c.trench.length=c.trench.width=.75;c.trench.depth=.15
    a=Simulation(deepcopy(c));b=Simulation(deepcopy(c))
    assert a.run(4000)==b.run(4000)
    np.testing.assert_array_equal(a.terrain.height,b.terrain.height)


def test_repeated_swaps_do_not_lock_empty_bucket():
    c=logistics_config(count=4,excavator_initial_soc=.28,digging_power_w=6000.)
    c.fleet.charge_slots=4;c.trench.length=3.;c.trench.depth=.3
    s=Simulation(c);r=s.run(20000)
    assert r['state']=='COMPLETED',(r['reason'],r['excavator'])
    assert r['excavator']['swaps']>=2
    assert r['excavator']['bucket_kg']==0
    assert abs(r['energy_error_j'])<1e-4 and abs(r['mass_error_kg'])<1e-7


def test_partially_used_relocation_packs_return_to_charger():
    s=Simulation(logistics_config(count=4));w=s.logistics
    b=w.inventory.items['excavator_battery_03']
    b.energy_j=b.capacity_j*.35
    w.inventory.move(b.id,L.WORKSITE_RACK,'excavator',w.rack,'STORED_READY')
    w.generate_jobs()
    assert any(j.kind=='RECOVER_EMPTY_EXCAVATOR_BATTERY' and b.id in j.resource_ids for j in w.jobs)


def test_takeoff_keeps_pad_reserved_until_cruise():
    c=logistics_config(count=2);c.fleet.charge_slots=2
    s=Simulation(c);w=s.logistics
    s.step();job=w.active[0]
    assert s.slots[0]==0 and s.drone.slot==0
    for _ in range(30):
        s.step()
        if job.phase=='ORIGIN':break
        assert s.slots[0]==0
    assert s.drone.slot is None and s.slots[0] is None


def test_unported_module_does_not_charge_invisibly():
    c=logistics_config(charger_ports=1,drone_spare_initial_soc=.2)
    s=Simulation(c);w=s.logistics
    a,b=[w.inventory.items[f'drone_1_spare_1_{i}'] for i in (1,2)]
    before_b=b.energy_j;before_a=a.energy_j
    s.step()
    assert a.energy_j>before_a and b.energy_j==before_b
    assert len(w.battery_service.ports)==1
    assert abs(s.report()['energy_error_j'])<1e-4


def test_dock_preserves_legacy_charge_bus_without_spare_ports():
    c=logistics_config(drone_spare_sets=0);c.drone.initial_charge=.2;c.start_hour=0
    s=Simulation(c);before=s.drone.battery;s.step()
    assert s.drone.battery-before==pytest.approx(8000*.9)
    assert abs(s.report()['energy_error_j'])<1e-4


@pytest.mark.parametrize('idle_digging',[False,True])
def test_container_brush_fill_and_leftover_mass(idle_digging):
    from ats.plan import PlanConfig,PlanSegment
    c=logistics_config(count=4,idle_digging=idle_digging);c.fleet.charge_slots=4
    c.plan=PlanConfig([PlanSegment([[15,16],[17,16]],width=1.,height=-.25),
                       PlanSegment([[12,12]],mode='fill',width=.5,height=.1)])
    s=Simulation(c);r=s.run(20000)
    assert r['state']=='COMPLETED',(r['reason'],r['excavator'])
    assert r['filled_mass_kg']>0 and s.logistics.delivered_mass>r['filled_mass_kg']
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<1e-4


def test_large_field_has_finite_deployment_and_footprint():
    from ats.experiments import experiment_config
    s=Simulation(experiment_config(Config(),panel_area=200))
    w=s.logistics
    assert w.field_rect is not None and not w.field_ready
    s.run(100)
    assert s.base.generated==0 and w.field_elapsed==100
    assert w.station_consumed>=200*100
    s.run(210)
    assert w.field_ready and s.base.generated>0
    assert abs(s.report()['energy_error_j'])<1e-4


def test_mechanical_dock_can_use_sun_without_base_storage():
    from ats.equipment import resolve_config
    c=logistics_config();c.drone.initial_charge=.25;c.start_hour=12
    resolve_config(c);c.base.equipment.batteries=[]
    s=Simulation(c);r=s.run(65)
    assert s.logistics.battery_service.drone_swap_count==1
    assert s.base.battery==0 and s.logistics.station_consumed>=12000
    assert abs(r['energy_error_j'])<1e-4


def test_swap_is_visible_atomic_and_removed_modules_use_real_ports():
    from ats.drone import DroneState as DS
    c=logistics_config(drone_swap_seconds=10.,charger_ports=1)
    c.drone.initial_charge=.25
    s=Simulation(c);w=s.logistics;d=s.drone;old=list(d.battery_ids)
    initial=[w.inventory.items[id].energy_j for id in old]
    identities=set(w.inventory.items)
    for second in range(1,10):
        s.step();r=s.report(full=False)
        assert d.state==DS.SWAPPING_BATTERIES and d.grounded
        swap=r['battery_service']['drone_swaps'][d.serial]
        assert swap['remaining_seconds']==10-second
        assert r['battery_service']['dock_charging_drones']==[]
        assert d.battery_ids==old
        assert [w.inventory.items[id].energy_j for id in old]==initial
        assert abs(r['energy_error_j'])<1e-4
    s.step()
    assert w.battery_service.drone_swap_count==1
    assert w.battery_service.drone_swap_counts=={d.serial:1}
    assert set(d.battery_ids).isdisjoint(old) and set(w.inventory.items)==identities
    assert not d.charge_requested and not w.battery_service.drone_swaps
    # Dispatch may already spend the first second of winch preparation.
    assert sum(w.inventory.items[id].energy_j for id in d.battery_ids)+d.consumed==pytest.approx(d.config.battery)
    # Both removed packs persist; only one gets a port and input energy.
    before=[w.inventory.items[id].energy_j for id in old]
    s.step()
    after=[w.inventory.items[id].energy_j for id in old]
    assert sum(a>b for a,b in zip(after,before))==1
    assert abs(s.report()['energy_error_j'])<1e-4


def test_idle_carriers_dig_after_physical_tool_change_with_one_service_pad():
    from ats.experiments import experiment_config
    c=experiment_config(Config(),drones=4,landing_slots=1,batch_courier=False)
    s=Simulation(c);w=s.logistics;ids=set(w.inventory.items)
    changed=set();diggers=set();saw_tool_change=False
    for _ in range(6000):
        s.step()
        for d in s.drones:
            if d.state.value=='CHANGING_TOOL':
                saw_tool_change=True
                assert d.slot is not None and s.slots[d.slot]==d.identifier
                assert w.inventory.items[f'grabber_{d.serial}'].reserved_by==f'tool_{d.serial}'
            if d.attachment=='GRABBER':
                changed.add(d.serial)
                assert w.inventory.items[f'grabber_{d.serial}'].location==L.DRONE_MOUNT
                assert w.inventory.items[f'hook_{d.serial}'].location==L.BASE_STORAGE
            if d.productivity.total_mass>0 and d.attachment=='GRABBER':diggers.add(d.serial)
        assert not w.direct_ids.intersection(w.active)
        assert set(w.inventory.items)==ids
        r=s.report(full=False)
        assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<1e-4
        if s.state.value!='RUNNING':break
    assert s.state.value=='COMPLETED',s.reason
    assert saw_tool_change and len(changed)>=2 and len(diggers)>=2
    assert w.delivered_mass>0 and w.excavator.excavated_kg>0
    assert not w.tool_targets and not any(o.reserved_by for o in w.inventory.items.values())


def test_only_carrier_returns_to_hook_when_a_load_becomes_ready():
    from ats.drone import DroneState as DS
    c=logistics_config(direct_allowed=True,excavator_enabled=False,container_count=1,
                       attachment_seconds=3.)
    c.trench.length=6
    s=Simulation(c);w=s.logistics;d=s.drone
    s.run(15)
    assert d.identifier in w.direct_ids and not d.grounded
    # A new physical load is announced while the sole carrier is digging.
    container=w.containers[0]
    container.soil_kg=s.terrain.scoop(17,16,30)
    w.inventory.move(container.id,L.FILLED_QUEUE,'worksite',(17,16,0),'FULL_READY')
    w.generate_jobs()
    job=next(j for j in w.jobs if j.kind=='PICKUP_FULL_CONTAINER')
    saw_return=False;saw_change=False
    for _ in range(1500):
        s.step()
        saw_return |= d.state in (DS.RETURN,DS.LANDING)
        saw_change |= d.state==DS.CHANGING_TOOL
        assert not w.direct_ids.intersection(w.active)
        r=s.report(full=False)
        assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<1e-4
        if job.status=='COMPLETED':break
    assert job.status=='COMPLETED' and saw_return and saw_change
    assert d.attachment=='CONTAINER_HOOK'
    assert w.inventory.items[f'grabber_{d.serial}'].location==L.BASE_STORAGE
    assert w.inventory.items[f'hook_{d.serial}'].location==L.DRONE_MOUNT
    # A dedicated digger returns to its original role after urgent cargo duty,
    # even when opportunistic digging by ordinary carriers is disabled.
    before=sum(j.kind=='EXCAVATE_DIRECT' for j in w.jobs)
    for _ in range(150):
        s.step()
        if sum(j.kind=='EXCAVATE_DIRECT' for j in w.jobs)>before:break
    assert d.attachment=='GRABBER' and d.identifier in w.direct_ids
    assert abs(s.report()['mass_error_kg'])<1e-7 and abs(s.report()['energy_error_j'])<1e-4
