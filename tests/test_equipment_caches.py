"""Local equipment caches retain live ground, motion and service semantics."""
import numpy as np
import pytest
from ats.checkpoint import save_state,load_state
from ats.inventory import Location as L


def solar():
    from tests.test_battery_platforms import configuration
    from ats.simulation import Simulation
    from ats.soil import SoilProfile
    c=configuration(area=452);c.base.layout={'heading_deg':22.,'auto_heading':0}
    c.soil_profile=SoilProfile.qattara()
    s=Simulation(c);s.request_landing();s.logistics.solar_array.audit(s.logistics)
    assert not s.logistics.solar_array.invalid_sections
    return s


def test_support_cache_checks_only_changed_local_ground_and_material(monkeypatch):
    import ats.self_solar as pv
    from ats.site_ground import rectangle_patch
    s=solar();w=s.logistics;a=w.solar_array;t=s.terrain;calls=[]
    original=pv.support_safe
    def counted(spec,module,*args):calls.append(module['id']);return original(spec,module,*args)
    monkeypatch.setattr(pv,'support_safe',counted)
    # A real distant edit changes the global height/material revisions, while
    # every solar support still stands on exactly the same ground.
    t.height[-2,-2]-=.01;t.revision+=1;a.audit(w)
    assert not calls and not a.invalid_sections
    q=a.sections[0];patch=rectangle_patch(t,q.rect)
    t.height[patch][0,0]+=.01;t.revision+=1;a.audit(w)
    assert calls==[q.platform_id] and not a.invalid_sections
    # Replace a thin native layer by an equal thickness of weaker loose soil:
    # surface elevation stays unchanged, but one foot loses bearing capacity.
    t.soil.profile.materials[1].bearing_kpa=.01
    index=int(a._support_indices[0,0]);xx,yy=t.soil._coordinates([index],center=True)
    values=t.soil._read(xx,yy);values[0]+=.001;values[2]+=.001
    before=t.height.copy();t.soil._write(xx,yy,values);calls.clear();a.audit(w)
    np.testing.assert_array_equal(t.height,before)
    assert q.id in a.invalid_sections and calls==[q.platform_id]
    values[0]-=.001;values[2]-=.001;t.soil._write(xx,yy,values);a.audit(w)
    assert not a.invalid_sections


def test_local_protection_and_water_changes_are_not_hidden_by_ground_cache():
    import base64
    from ats.site_ground import rectangle_patch
    s=solar();w=s.logistics;a=w.solar_array;t=s.terrain;q=a.sections[0]
    patch=rectangle_patch(t,q.rect);row,col=patch[0].start+1,patch[1].start+1
    t.protected=np.zeros_like(t.height,dtype=bool);t.protected[row,col]=True
    a.audit(w);assert q.id in a.invalid_sections
    t.protected[row,col]=False;t.revision+=1;a.audit(w);assert not a.invalid_sections
    bits=np.zeros_like(t.height,dtype=np.uint8);bits[row,col]=1;t.reference[row,col]=-1
    s.config.georeference={'marine':dict(bits=base64.b64encode(np.packbits(bits).tobytes()).decode(),
        origin=[0.,0.],spacing=[t.cell,t.cell],shape=list(bits.shape))}
    a.audit(w);assert q.id in a.invalid_sections


def test_stationary_battery_coordinates_reuse_and_invalidate_without_moving_cargo(monkeypatch):
    from tests.test_battery_platforms import stationary
    from ats.inventory import BatteryModule
    from ats.base_layout import rotate_offset
    s=stationary();w=s.logistics;p=w.battery_platforms;calls=[]
    original=p._slot_coordinates
    def counted(id):calls.append(id);return original(id)
    monkeypatch.setattr(p,'_slot_coordinates',counted)
    for _ in range(20):p.position_items()
    s.terrain.height[-2,-2]+=.1;s.terrain.revision+=1;p.position_items()
    assert not calls
    module=next(b for b in w.inventory.select(BatteryModule) if b.holder_id=='base')
    entry=p.bindings[module.id];carrier=s.base.layout_spec['platforms'][entry['platform_id']-1]
    xy=s.base.position[:2]+rotate_offset(s.base.config.layout,carrier['offset_m'])
    row,col=(xy[::-1]/s.terrain.cell).astype(int)
    old=module.position;s.terrain.height[row,col]+=.1;s.terrain.revision+=1;p.position_items()
    assert calls and module.position[2]==pytest.approx(old[2]+.1)
    calls.clear();p.position_items();assert not calls
    entry['offset_m'][0]+=.1;p.position_items()
    np.testing.assert_array_equal(module.position,original(module.id))
    # In-transfer modules keep the handler's real position; a returned module
    # gets its fixed mount again even when the carrier cache itself is reused.
    module.location=L.IN_TRANSFER;module.holder_id='test_handler';module.position=(1.,2.,3.)
    p.position_items();assert module.position==(1.,2.,3.)
    module.location=L.BASE_STORAGE;module.holder_id='base';p.position_items()
    np.testing.assert_array_equal(module.position,original(module.id))


def test_shared_geometry_rebuilds_one_unit_and_retains_real_contours(monkeypatch):
    from scripts.benchmark_battery_platforms import build
    p=build(3);site=p.shared_site;site.obstacle_top();calls=[]
    original=site._unit_geometry
    def counted(unit):calls.append(unit.id);return original(unit)
    monkeypatch.setattr(site,'_unit_geometry',counted)
    for _ in range(20):site.obstacle_top()
    assert not calls
    s=p.units[1].simulation;site.invalidate_geometry(s);site.obstacle_top()
    assert calls==[p.units[1].id]
    calls.clear();s.base.position[0]+=.5;site.obstacle_top()
    assert calls==[p.units[1].id]
    calls.clear()
    s.terrain.height[-2,-2]+=.1;s.terrain.revision+=1;site.obstacle_top();assert not calls
    from ats.base_layout import platform_polygons
    from ats.site_ground import polygon_bounds
    x,y,width,length=polygon_bounds(next(platform_polygons(s.base.layout_spec,s.base.position)))
    point=np.array([x+width/2,y+length/2])
    row,col=(point[::-1]/s.terrain.cell).astype(int)
    top=site.obstacle_height(point);s.terrain.height[row,col]+=.1;s.terrain.revision+=1
    assert site.obstacle_height(point)>=top+.099999 and calls==[p.units[1].id]


def test_support_actuation_does_not_rebuild_unchanged_navigation_geometry(monkeypatch):
    from scripts.benchmark_battery_platforms import build
    p=build(3);site=p.shared_site;site.obstacle_top();calls=[]
    original=site._unit_geometry
    def counted(unit):calls.append(unit.id);return original(unit)
    monkeypatch.setattr(site,'_unit_geometry',counted)
    for _ in range(10):p.step()
    assert all(0<u.simulation.logistics.solar_array.sections[0].supports<1 for u in p.units)
    assert not calls
    for _ in range(65):p.step()
    assert all(u.simulation.logistics.solar_array.sections[0].open_fraction>0 for u in p.units)
    # Updates during real hinge motion stay proportional to the units that
    # actually move, rather than rebuilding every neighbour after each drive.
    assert len(calls)<=65*len(p.units)*3


def test_foreign_moving_excavator_invalidates_only_nearby_solar_clearance():
    from scripts.benchmark_battery_platforms import build
    p=build(2);s,other=[u.simulation for u in p.units]
    w=s.logistics;a=w.solar_array;a.audit(w);q=a.sections[0]
    assert q.id not in a.invalid_sections
    e=other.logistics.excavator;previous=e.position.copy()
    e.position[:]=a.centre(w,q);a.audit(w)
    assert q.id in a.invalid_sections
    e.position[:]=previous;a.audit(w)
    assert q.id not in a.invalid_sections


@pytest.mark.parametrize('folding',[False,True])
def test_incremental_geometry_matches_full_rebuild_during_hinges_and_checkpoint(tmp_path,folding):
    from scripts.benchmark_battery_platforms import build
    from ats.self_solar import deployment_polygons
    p=build(2)
    for u in p.units:u.simulation.request_landing()
    for _ in range(100):p.step()
    if folding:
        for u in p.units:u.simulation.logistics.solar_array.request_fold()
    path=tmp_path/'incremental.atsstate'
    save_state(path,p.active.simulation,complex_project=p);other=load_state(path).complex_project
    for _ in range(12):
        p.step();other.step()
        points=[point for u in p.units for polygon in deployment_polygons(u.simulation.base.layout_spec,u.simulation.base.position)
            for point in [*polygon,np.mean(polygon,axis=0)]]
        site=p.shared_site;before=[site.obstacle_height(point) for point in points]
        tokens=[site.ground_token(None,(point[0]-1,point[1]-1,point[0]+1,point[1]+1)) for point in points]
        site.invalidate_geometry()
        assert before==[site.obstacle_height(point) for point in points]
        assert tokens==[site.ground_token(None,(point[0]-1,point[1]-1,point[0]+1,point[1]+1)) for point in points]
    assert p.report()==other.report()
    assert all(a.simulation.report()==b.simulation.report() for a,b in zip(p.units,other.units))
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002
