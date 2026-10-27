"""Safety and reproducibility of bounded planning, route caches and charging."""
import numpy as np
import pytest
from ats.config import Config
from ats.maps import with_map_size
from ats.experiments import experiment_config
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.checkpoint import save_state,load_state
from ats.inventory import BatteryModule,Item,Location as L
from ats.excavator import ExcavatorState
from ats.terrain import Terrain
from ats.drone import DroneState


def channel(layers=False):
    c=experiment_config(with_map_size(Config(),64))
    c.plan=PlanConfig([PlanSegment([[20,32],[50,32]],'dig',1,-.2)],channel=dict(
        start=[20.,32.],tangent=[1.,0.],operations=dict(front_length_m=16.,setback_m=6.,bank_width_m=8.,
        base_chainage_m=0.,base_sites=[[0.,12.,20.],[30.,52.,20.]])))
    if layers:
        from ats.soil import SoilProfile
        c.soil_profile=SoilProfile.qattara();c.soil_profile.thicknesses=[.02,.05]
    s=Simulation(c);assert s.state.value=='RUNNING',s.reason
    return s


def test_shared_dispatch_searches_one_unloading_point_and_reserves_both_ends(monkeypatch):
    from tests.test_shared_site import site
    p,t=site();s=p.active.simulation;p.enable_shared()
    calls=[];original=Terrain.dump_location
    def record(self,*args,**kwargs):calls.append(1);return original(self,*args,**kwargs)
    monkeypatch.setattr(Terrain,'dump_location',record)
    a,b=s.drones[:2];trip=s._choose_trip(a)
    assert trip is not None and len(calls)==1
    assert trip.mass>=a.config.payload*.9
    a.trip=trip;a.state=DroneState.TO_DIG;other=s._choose_trip(b)
    assert other is not None
    assert np.linalg.norm(np.subtract(other.dig,trip.dig))>=s.config.fleet.separation
    assert np.linalg.norm(np.subtract(other.dump,trip.dump))>=s.config.fleet.separation
    assert other.energy<=b.battery-b.config.battery*b.config.reserve


@pytest.mark.parametrize('layers',[False,True])
def test_local_unloading_preserves_material_and_capacity_with_dense_reservations(layers):
    s=channel(layers);t=s.terrain
    first=t.dump_location(.02,near=[30.,32.]);assert first is not None
    reserved=[first,*[(x,y) for x in (24.,28.,32.,36.) for y in (22.,42.)]]
    point=t.dump_location(.02,reserved,2.,near=[30.,32.]);assert point is not None
    assert all(np.linalg.norm(np.subtract(point,p))>=2. for p in reserved)
    source=[30.,32.];before=t.mass
    if layers:
        load=t.scoop_load(*source,60.);mass=load.mass
    else:load=None;mass=t.scoop(*source,60.)
    t.deposit(*point,mass,load=load)
    assert t.mass==pytest.approx(before,abs=1e-7)
    assert t.stockpile_volume>0
    height=t.height.copy();too_much=(t.local_dump_space(*point)+1)*s.config.terrain.density
    if not layers:
        with pytest.raises(ValueError):t.deposit(*point,too_much)
        np.testing.assert_array_equal(t.height,height)
    assert np.all(t.height[t.dump_mask]<=t.dump_cap[t.dump_mask]+1e-12)
    assert abs(s.report()['mass_error_kg'])<1e-7


def test_ground_route_rechecks_relevant_edits_and_always_checks_next_step(monkeypatch):
    c=experiment_config(with_map_size(Config(),64));c.terrain.dump_rect=(40.,38.,22.,24.)
    s=Simulation(c);e=s.logistics.excavator;t=s.terrain
    e.position=np.array([43.,45.,0.]);e.destination=np.array([60.,45.,0.])
    assert e.can_drive(e.destination,t)
    calls=[];original=e._path_clear
    def record(target,terrain):calls.append(tuple(target));return original(target,terrain)
    monkeypatch.setattr(e,'_path_clear',record)
    assert e._safe_drive_step(t) and len(calls)==1  # only the next physical step
    t.deposit(52.,56.,60.)
    calls.clear();assert e._safe_drive_step(t) and len(calls)==1
    t.deposit(52.,45.,3000.,radius=.25)
    calls.clear();assert not e._safe_drive_step(t)
    assert calls[0]==tuple(e.destination)
    np.testing.assert_array_equal(e.position,[43.,45.,0.])


def test_route_invalidates_for_a_foreign_base_but_not_distant_geometry(monkeypatch):
    from tests.test_shared_site import site
    p,t=site();p.add_shared(t);a,b=[u.simulation for u in p.units];e=a.logistics.excavator
    e.position=np.array([15.,75.,0.]);e.destination=np.array([65.,75.,0.])
    assert e.can_drive(e.destination,a.terrain)
    calls=[];original=e._path_clear
    def record(target,terrain):calls.append(tuple(target));return original(target,terrain)
    monkeypatch.setattr(e,'_path_clear',record)
    b.base.position[:2]=[100.,100.];p.shared_site.invalidate_geometry()
    assert e._safe_drive_step(a.terrain) and len(calls)==1
    b.base.position[:2]=[45.,75.];p.shared_site.invalidate_geometry();calls.clear()
    assert not e._safe_drive_step(a.terrain) and calls[0]==tuple(e.destination)


def test_moving_excavator_cache_roundtrips_with_exact_continuation(tmp_path):
    s=Simulation(experiment_config(with_map_size(Config(),64)));e=s.logistics.excavator
    e.position=np.array([30.,20.,0.]);e.destination=np.array([40.,20.,0.]);e.state=ExcavatorState.MOVING_LOCAL
    s.run(3);assert e.state==ExcavatorState.MOVING_LOCAL and e._drive_cache
    path=tmp_path/'route.atsstate';save_state(path,s);other=load_state(path).simulation
    for _ in range(40):s.step();other.step()
    assert s.report()==other.report()
    np.testing.assert_array_equal(s.terrain.height,other.terrain.height)
    assert abs(s.report()['energy_error_j'])<.002


def test_stationary_excavator_keeps_mining_from_its_reachable_worksite():
    c=experiment_config(with_map_size(Config(),64));c.logistics.idle_digging=False;c.logistics.direct_allowed=False
    c.terrain.dump_rect=(44.,38.,18.,24.)
    c.plan=PlanConfig([PlanSegment([[20,20],[22,20]],'dig',1,-.3)])
    s=Simulation(c);w=s.logistics;e=w.excavator
    e.position=np.array([20.,17.5,0.]);w.rack=e.position.copy()
    for item in w.containers:
        item.location=L.WORKSITE_RACK;item.position=tuple(e.position+[.8,.3,0])
    start=e.position.copy();assert e.work_point(s.terrain,w.inventory) is not None
    s.run(300)
    np.testing.assert_array_equal(e.position,start)
    assert e.excavated_kg>100 and e.cycles>=6 and e.moving_seconds==0
    assert abs(s.report()['mass_error_kg'])<1e-7 and abs(s.report()['energy_error_j'])<.002


def test_charging_keeps_tick_integration_sun_reservations_and_hot_slot_events(tmp_path):
    s=Simulation(experiment_config(with_map_size(Config(),64)));w=s.logistics
    s.config.start_hour=0.;w.field_ready=True
    pack=w.replenish_batteries(s.drone.battery_ids[0],1,.2)[0]
    pack=w.inventory.items[pack] if isinstance(pack,str) else pack
    service=w.battery_service
    assert len(service.modules())==len(w.inventory.select(BatteryModule))-len(w.base_battery_ids)
    a=service.charging();b=service.charging()
    assert next(port for port,_ in a if port.pack is pack) is next(port for port,_ in b if port.pack is pack)
    w.inventory.reserve([pack.id],'test_delivery');before=pack.energy_j;s.step()
    assert pack.energy_j==before and pack.id not in service.ports
    w.inventory.release('test_delivery');s.config.start_hour=12.;before=pack.energy_j;s.step()
    assert pack.energy_j>before and pack.energy_j<=pack.capacity_j
    assert 0<service.slots[pack.id].input_power_w<=pack.max_charge_w
    assert abs(s.report()['mass_error_kg'])<1e-7 and abs(s.report()['energy_error_j'])<.002
    path=tmp_path/'charging.atsstate';save_state(path,s);other=load_state(path).simulation
    for _ in range(30):s.step();other.step()
    assert s.report()==other.report()
    id=w.replenish_batteries(s.drone.battery_ids[0],1,1-1e-8)[0]
    nearly_full=w.inventory.items[id];s.step()
    assert nearly_full.energy_j==nearly_full.capacity_j and service.slots[id].state=='FULL'
    s.step();assert id not in service.ports
    assert abs(s.report()['energy_error_j'])<.002


def test_cached_structure_still_checks_energy_mass_and_charging_identity():
    s=Simulation(experiment_config(Config()));inv=s.logistics.inventory;inv.validate(dynamic=True)
    pack=s.logistics.battery_service.modules()[0];energy=pack.energy_j;mass=pack.mass_kg
    pack.energy_j=pack.capacity_j+1
    with pytest.raises(ValueError):inv.validate(dynamic=True)
    pack.energy_j=energy;pack.mass_kg=-1
    with pytest.raises(ValueError):inv.validate(dynamic=True)
    pack.mass_kg=mass;inv.validate(dynamic=True)
    slot=s.logistics.battery_service.slots[pack.id];slot.input_power_w=slot.rated_w+1
    with pytest.raises(ValueError):inv.validate(dynamic=True)
    slot.input_power_w=0;inv.items[pack.id]=Item(pack.id,mass,L.BASE_STORAGE,'FULL','base')
    with pytest.raises(ValueError):inv.validate(dynamic=True)
