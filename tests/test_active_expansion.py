"""Neighbour joining preserves one live map and global physical coordinates."""
from copy import deepcopy
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.landscape import ElevationRaster,Landscape,workspace_config
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.complexes import ComplexProject,ComplexTemplate
from ats.checkpoint import save_state,load_state


def region(soil=False,second=False):
    raster=ElevationRaster(np.zeros((401,401)),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=1.,dy=1.))
    land=Landscape(raster,cell_size=.5)
    if soil:
        from ats.soil import SoilProfile
        land.soil_profile=SoilProfile.qattara();land.soil_profile.thicknesses=[.02,.05]
    c=workspace_config(land,experiment_config(Config(),panel_area=20),[80.,80.],128.)
    c.terrain.dump_rect=(90.,80.,30.,40.)
    c.plan=PlanConfig([PlanSegment([[35.,40.],[85.,40.]],width=2.,height=-.25)])
    sim=Simulation(c);assert sim.state.value=='RUNNING',sim.reason
    p=ComplexProject(sim,land);p.enable_shared()
    if second:p.add_shared(ComplexTemplate.capture(c),position=(100.,20.))
    return p


@pytest.mark.parametrize('soil',[False,True])
def test_prepend_keeps_edited_columns_global_positions_inventory_and_balances(tmp_path,soil):
    p=region(soil,second=True);site=p.shared_site;old=site.core
    # Move real material with the common conservative APIs before joining.
    load=old.scoop_load(50.,40.,100.,radius=1.) if soil else None
    mass=load.mass if soil else old.scoop(50.,40.,100.,radius=1.)
    old.deposit(105.,100.,mass,load=load,radius=2.)
    before=p.report();height=old.height.copy();target=old.target.copy();ids=p.inventory_ids()
    bases=[u.simulation.base.position.copy()+[80.,80.,0.] for u in p.units]
    items=[{key:np.array(o.position)+[80.,80.,0.] for key,o in u.simulation.logistics.inventory.items.items()} for u in p.units]
    site.enlarge(192.,origin_m=[48.,48.],preserve_resolution=True)
    fresh=site.core;overlap=np.s_[64:320,64:320]
    assert fresh.cell==.5 and fresh.height.shape==(384,384)
    np.testing.assert_array_equal(fresh.height[overlap],height)
    np.testing.assert_array_equal(fresh.target[overlap],target)
    assert p.inventory_ids()==ids and p.time==before['time_seconds']
    for u,base,inventory in zip(p.units,bases,items):
        np.testing.assert_array_equal(u.simulation.base.position+[48.,48.,0.],base)
        for key,point in inventory.items():np.testing.assert_array_equal(np.array(u.simulation.logistics.inventory.items[key].position)+[48.,48.,0.],point)
    assert p.report()['mass_error_kg']==pytest.approx(before['mass_error_kg'],abs=1e-7)
    assert p.report()['energy_error_j']==before['energy_error_j']
    if soil:np.testing.assert_allclose(site.material_errors(),0.,atol=1e-7)
    p.validate()
    path=tmp_path/'joined.atsstate';save_state(path,p.active.simulation,landscape=p.landscape,complex_project=p)
    q=load_state(path).complex_project
    assert q.report()==p.report()
    for _ in range(5):p.step();q.step()
    assert q.report()==p.report()


def test_dirty_relaxation_and_original_target_are_retained_exactly():
    p=region();old=p.shared_site.core
    old.target[100,90]-=.01;old._relax_dirty[90:93,100:104]=True
    dirty=old._relax_dirty.copy();target=old.target.copy()
    p.shared_site.enlarge(192.,origin_m=[48.,64.],preserve_resolution=True)
    new=p.shared_site.core
    np.testing.assert_array_equal(new.target[32:288,64:320],target)
    assert np.all(new._relax_dirty[32:288,64:320][dirty])
    assert new._dirty_bounds is not None and new.revision>old.revision


@pytest.mark.parametrize('failure',['airborne','service','outside','not_contained'])
def test_busy_or_invalid_reframe_is_atomic(failure):
    from ats.drone import DroneState
    p=region();site=p.shared_site;s=p.active.simulation;origin=[48.,48.]
    if failure=='airborne':s.drone.state=DroneState.TAKEOFF
    elif failure=='service':s.logistics.tool_changes[1]=2.
    elif failure=='outside':origin=[-10.,48.]
    elif failure=='not_contained':origin=[81.,48.]
    before=deepcopy(p.report());core=site.core;position=s.base.position.copy();height=core.height.copy()
    with pytest.raises(ValueError):site.enlarge(192.,origin_m=origin,preserve_resolution=True)
    assert p.report()==before and site.core is core
    np.testing.assert_array_equal(s.base.position,position);np.testing.assert_array_equal(core.height,height)


def test_append_native_resolution_preserves_live_drone_trip_and_cargo():
    p=region(soil=True);s=p.active.simulation
    for _ in range(150):
        p.step()
        if any(d.trip is not None for d in s.drones):break
    assert any(d.trip is not None for d in s.drones)
    positions=[d.position.copy() for d in s.drones];trips=[d.trip for d in s.drones];before=p.report()
    p.shared_site.enlarge(192.,preserve_resolution=True)
    assert [d.trip for d in s.drones]==trips
    for d,point in zip(s.drones,positions):np.testing.assert_array_equal(d.position,point)
    assert p.report()['energy_error_j']==before['energy_error_j']
    assert p.report()['mass_error_kg']==pytest.approx(before['mass_error_kg'],abs=1e-7)
    for _ in range(5):p.step()
    p.validate()


def test_parked_conventional_road_and_machines_follow_frame_without_moving(tmp_path):
    from ats.conventional import ConventionalConfig
    p=region(soil=True);s=p.active.simulation
    fleet=s.install_conventional(ConventionalConfig(base_x=12.,base_y=100.,work_rect=(35.,30.,20.,20.),truck_count=1))
    positions=[m.position.copy()+[80.,80.,0.] for m in fleet.machines]
    old_surface=fleet.ramp_surface.copy();before=p.report();machines=list(fleet.machines)
    p.shared_site.enlarge(192.,origin_m=[48.,48.],preserve_resolution=True)
    assert fleet.machines==machines and fleet.terrain is p.shared_site.core
    np.testing.assert_array_equal(fleet.ramp_surface[64:320,64:320],old_surface)
    for m,point in zip(fleet.machines,positions):np.testing.assert_array_equal(m.position+[48.,48.,0.],point)
    assert fleet.config.work_rect==(67.,62.,20.,20.)
    assert p.report()['energy_error_j']==before['energy_error_j']
    p.validate()
    path=tmp_path/'road.atsstate';save_state(path,s,landscape=p.landscape,complex_project=p)
    q=load_state(path).complex_project
    assert q.report()==p.report()
    for _ in range(5):p.step();q.step()
    assert q.report()==p.report()


def test_regional_sector_state_legacy_pickups_and_round_trip_after_expansion(tmp_path):
    from tests.test_window_boundaries import window,finish_visible,travel
    p,ch=window(index=3,soil=True);s=p.active.simulation
    finish_visible(p)
    old=p.shared_site.core;sector=old.front.sector_id
    origin=np.asarray(s.config.georeference['origin_m']);new_origin=origin-[32.,32.]
    old.front.legacy_pickups=[(35.,40.,1.5)];s.terrain.front.legacy_pickups=[(35.,40.,1.5)]
    p.shared_site.enlarge(192.,origin_m=new_origin,preserve_resolution=True)
    assert p.shared_site.core.front.sector_id==sector
    assert s.terrain.front.sector_id==sector
    assert p.shared_site.core.front.legacy_pickups==[(67.,72.,1.5)]
    assert s.terrain.front.legacy_pickups==[(67.,72.,1.5)]
    height=p.shared_site.core.height.copy();ids=p.inventory_ids()
    path=tmp_path/'channel-joined.atsstate';save_state(path,s,landscape=p.landscape,complex_project=p)
    q=load_state(path).complex_project
    assert q.report()==p.report()
    travel(q,None,origin=(new_origin+[16.,0.]).tolist())
    travel(q,None,origin=new_origin.tolist())
    np.testing.assert_array_equal(q.shared_site.core.height,height)
    assert q.inventory_ids()==ids and q.shared_site.core.cell==1.
    assert q.shared_site.core.height.shape==(192,192)
    assert abs(q.report()['mass_error_kg'])<1e-7 and abs(q.report()['energy_error_j'])<.002


def test_normal_landing_retains_queued_jobs_then_resumes_after_reframe(tmp_path):
    from ats.active_expansion import _stationary
    p=region(soil=True);s=p.active.simulation;w=s.logistics
    for _ in range(50):p.step()
    s.request_landing()
    for _ in range(300):
        try:_stationary(p.shared_site);break
        except ValueError:p.step()
    else:pytest.fail('Normal landing never reached a safe expansion boundary')
    queued={j.id:(j,tuple(j.origin),tuple(j.destination),tuple(j.resource_ids)) for j in w.jobs if j.status=='QUEUED'}
    assert queued and any(e.state.value=='IDLE' and e.destination is not None for e in w.excavators)
    reserved={key:o.reserved_by for key,o in w.inventory.items.items()};before=p.report()
    p.shared_site.enlarge(192.,origin_m=[48.,48.],preserve_resolution=True)
    assert {key:o.reserved_by for key,o in w.inventory.items.items()}==reserved
    for id,(job,origin,destination,resources) in queued.items():
        assert job.id==id and job.status=='QUEUED' and tuple(job.resource_ids)==resources
        np.testing.assert_array_equal(np.asarray(job.origin)[:2],np.asarray(origin)[:2]+32.)
        np.testing.assert_array_equal(np.asarray(job.destination)[:2],np.asarray(destination)[:2]+32.)
        assert not hasattr(job,'region_origin_m')
    assert p.report()['energy_error_j']==before['energy_error_j']
    assert p.report()['mass_error_kg']==pytest.approx(before['mass_error_kg'],abs=1e-7)
    path=tmp_path/'queued-joined.atsstate';save_state(path,s,landscape=p.landscape,complex_project=p)
    q=load_state(path).complex_project;assert q.report()==p.report()
    s.landing_requested=False;q.active.simulation.landing_requested=False
    flight_energy=sum(d.consumed for d in s.drones)
    for _ in range(600):
        p.step();q.step()
        if any(j.status=='COMPLETED' for j,_,_,_ in queued.values()):break
    assert any(j.status=='COMPLETED' for j,_,_,_ in queued.values())
    assert sum(d.consumed for d in s.drones)>flight_energy and q.report()==p.report()
    p.validate();assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002


@pytest.mark.parametrize('code,retry',[
    ('BLOCKED_SECTOR_SITE',True),('BLOCKED_SECTOR_RANGE',True),
    ('BLOCKED_SECTOR_NO_PROGRESS',False),('UNRELATED_FAULT',False)])
def test_expansion_retries_only_window_sector_blocks_and_rechecks_real_range(code,retry):
    from tests.test_window_boundaries import window
    from ats.simulation import MissionState
    p,ch=window(tiny=True);s=p.active.simulation
    s.block(code+': previous diagnosis');s._sector_range_key=('stale',);s._work_window_cache=('stale',)
    s._relocation_attempt_time=s.time
    p.shared_site.enlarge(160.,preserve_resolution=True)
    if retry:
        assert s.state==MissionState.RUNNING and s.reason==''
        assert s._sector_range_key is None and s._work_window_cache is None
        assert s._relocation_attempt_time==-float('inf')
        s.step()
        # The tiny real battery remains insufficient: expanding land must not
        # bypass the physical range test or manufacture propulsion energy.
        assert s.state==MissionState.BLOCKED and s.reason.startswith('BLOCKED_SECTOR_RANGE:')
    else:
        assert s.state==MissionState.BLOCKED and s.reason==code+': previous diagnosis'


def test_sector_named_block_without_sector_model_is_not_cleared():
    from ats.simulation import MissionState
    p=region();s=p.active.simulation;s.block('BLOCKED_SECTOR_SITE: unrelated local brush')
    p.shared_site.enlarge(192.,preserve_resolution=True)
    assert s.state==MissionState.BLOCKED and s.reason=='BLOCKED_SECTOR_SITE: unrelated local brush'
