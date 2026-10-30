"""Fixed canal sectors: persisted scheduling, physical loads and real moves."""
from copy import deepcopy
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import zipfile
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.plan import PlanConfig,PlanSegment
from ats.terrain import Terrain
from ats.simulation import Simulation
from ats.checkpoint import save_state,load_state


def sector_config(*,length=430.,sector=200.,remnants=False):
    c=experiment_config(with_map_size(Config(),512),drones=4,panel_area=20.)
    c.terrain.cell_size=1.;c.terrain.n=512;c.terrain.grab_radius=1.;c.local_reference=True
    c.terrain.completion_volume=1e-5
    c.logistics.excavator_enabled=False;c.logistics.self_solar=None
    c.reference_height=np.zeros((c.terrain.n,c.terrain.n)).tolist()
    c.base.x=150.;c.base.y=160.;c.base.latitude_deg=31.
    c.base.battery=100*3_600_000
    c.drone.hover_power=1000.
    for d in c.fleet.models:d.hover_power=1000.
    ops=dict(front_length_m=32.,sector_length_m=sector,setback_m=20.,bank_width_m=48.,base_chainage_m=0.,
        base_sites=[[float(x-50),float(x),160.] for x in np.arange(50.,50.+length,5.)])
    c.plan=PlanConfig([PlanSegment([[50.,256.],[50.+length,256.]],'dig',2.,-.001)],
        channel=dict(start=[50.,256.],tangent=[1.,0.],operations=ops))
    if remnants:
        t=Terrain(c);h=t.target.copy()
        for x in (180.,270.,460.):
            if x<50.+length:
                k=int(np.argmin((t.x-x)**2+(t.y-256.)**2));h.ravel()[k]=t.reference.ravel()[k]
        c.initial_height=h.tolist()
    return c


def finish_sector(s):
    """Test fixture: a conservative cut and deposit, not a change of target."""
    t=s.terrain;f=t.front;f.update()
    for index in f.active_indices.copy():
        x,y=t.x.ravel()[index],t.y.ravel()[index]
        mass=t.scoop(x,y,1e8,radius=t.cell*.49)
        if mass:
            dump=t.dump_location(mass/s.config.terrain.density,near=(x,y))
            assert dump is not None
            t.deposit(*dump,mass)
    f.update();assert f.complete
    s._terrain_active=False


def v10_copy(path,replacement):
    with zipfile.ZipFile(path) as a,zipfile.ZipFile(replacement,'w') as b:
        for name in a.namelist():
            data=a.read(name)
            if name=='manifest.json':
                manifest=json.loads(data);manifest['version']=10;data=json.dumps(manifest).encode()
            b.writestr(name,data)


def test_front_advances_inside_fixed_two_hundred_metres_without_moving_base():
    s=Simulation(sector_config());f=s.terrain.front;before=s.base.position.copy()
    t=s.terrain
    # Excavate first 30 m with the same conservative cut/deposit operations.
    f.update()
    for k in f.active_indices[f.along[np.isin(f.indices,f.active_indices)]<30]:
        x,y=t.x.ravel()[k],t.y.ravel()[k];mass=t.scoop(x,y,1e8,radius=t.cell*.49)
        if mass:t.deposit(*t.dump_location(mass/s.config.terrain.density,near=(x,y)),mass)
    s._update_workfront()
    assert f.position>16 and not f.complete
    assert s.relocation_pending is None and s.base.relocations==0
    np.testing.assert_array_equal(s.base.position,before)
    assert f.sector_start==0 and f.sector_end==200


def test_load_and_unfinished_flight_prevent_sector_advance_then_one_move():
    s=Simulation(sector_config());f=s.terrain.front;finish_sector(s)
    d=s.drones[0];d.cargo=1.
    s._update_workfront();assert f.sector_id==0 and s.relocation_pending is None
    d.cargo=0.;s._update_workfront()
    assert f.sector_id==1 and s.relocation_pending is not None and not f.stationed
    target=s.relocation_pending.copy()
    assert abs(target[0]-350)<6 and np.linalg.norm(target-s.base.position[:2])>190
    for _ in range(100):s._update_workfront()
    np.testing.assert_array_equal(s.relocation_pending,target)
    assert not len(f.active_indices)


def test_last_partial_sector_and_exact_boundary_membership():
    s=Simulation(sector_config());f=s.terrain.front
    finish_sector(s);assert f.advance_sector();f.stationed=True;f.revision=-1
    finish_sector(s);assert f.advance_sector();f.stationed=True;f.revision=-1;f.update()
    assert f.sector_start==400 and 429<f.sector_end<432
    assert 414<f.sector_centre<416
    assert np.all(f.along[np.isin(f.indices,f.active_indices)]>=400-1e-9)
    finish_sector(s);assert not f.advance_sector()
    assert abs(s.report()['mass_error_kg'])<1e-7


def test_already_suitable_arrival_position_does_not_fold_for_fifteen_metres():
    c=sector_config(length=210.);c.base.x=240.
    s=Simulation(c);finish_sector(s);s._update_workfront();f=s.terrain.front
    assert f.sector_id==1 and f.stationed and s.relocation_pending is None
    assert s.base.position[0]==240. and s.base.relocations==0


def test_footprint_cannot_excavate_across_fixed_sector_boundary():
    s=Simulation(sector_config());t=s.terrain;f=t.front
    future=f.chainage>=200
    before=t.height[future].copy()
    mass=t.scoop(250.,256.,1e6,radius=5.)
    assert mass>0
    np.testing.assert_array_equal(t.height[future],before)


def test_unreachable_sector_is_diagnosed_without_short_move():
    # Equipment resolution restores rated battery capacity, so reduce the actual
    # models after constructing the physically valid initial inventory.
    s=Simulation(sector_config())
    for d in s.drones:d.config.battery=1.
    s._update_workfront()
    assert s.state.value=='BLOCKED' and 'BLOCKED_SECTOR_RANGE' in s.reason
    assert s.relocation_pending is None and s.base.relocations==0


def test_early_range_screen_runs_on_entry_and_equipment_change_not_every_tick(monkeypatch):
    s=Simulation(sector_config());calls=[];original=s._sector_range_reason
    def check(point):calls.append(1);return original(point)
    monkeypatch.setattr(s,'_sector_range_reason',check)
    for _ in range(20):s._update_workfront()
    assert len(calls)==1
    s.drones[0].config.hover_power*=1.01;s._update_workfront()
    assert len(calls)==2


def test_no_safe_next_site_is_explicit_and_retains_ground_and_equipment():
    s=Simulation(sector_config());f=s.terrain.front;finish_sector(s)
    # User-provided sites are constrained to an excavated canal: never disable
    # safety or fall back to following the moving face in small increments.
    f.settings['base_sites']=[[300.,350.,256.]]
    before=s.base.position.copy();s._update_workfront()
    assert s.state.value=='BLOCKED' and 'BLOCKED_SECTOR_SITE' in s.reason
    np.testing.assert_array_equal(s.base.position,before)


@pytest.mark.parametrize('slots',[1,4])
def test_normal_autonomy_completes_all_sectors_and_drives_only_between_them(tmp_path,slots):
    c=sector_config(remnants=True);c.base.equipment.landing_slots=slots
    s=Simulation(c);moves=[];saved=False
    for _ in range(7000):
        was=s.base.destination is not None;old=s.base.position.copy();s.step()
        if not was and s.base.destination is not None:
            moves.append(float(np.linalg.norm(s.base.destination-old[:2])))
            assert all(d.grounded and not d.cargo for d in s.drones)
            assert all(not c.soil_kg for c in s.logistics.containers)
            if not saved:
                path=tmp_path/'driving.atsstate';save_state(path,s)
                q=load_state(path).simulation
                for _ in range(20):s.step();q.step()
                assert s.report()==q.report();saved=True
        if s.state.value!='RUNNING':break
    assert s.state.value=='COMPLETED',s.reason
    assert len(moves)==2 and min(moves)>100
    assert all(np.linalg.norm(d.position[:2]-s.base.position[:2])<20 for d in s.drones)
    assert abs(s.report()['mass_error_kg'])<1e-7 and abs(s.report()['energy_error_j'])<.002


def test_working_sector_snapshot_resumes_exactly(tmp_path):
    s=Simulation(sector_config(remnants=True));s.run(20)
    path=tmp_path/'working.atsstate';save_state(path,s);q=load_state(path).simulation
    for _ in range(40):s.step();q.step()
    assert s.report()==q.report()
    assert s.terrain.front.report()==q.terrain.front.report()


def test_partial_container_forces_final_courier_batch_before_sector_move():
    from tests.test_workfront import canal
    from ats.inventory import Location as L
    s=canal(sector_length=20.);finish_sector(s);w=s.logistics;f=s.terrain.front
    t=s.terrain;k=np.flatnonzero(t.height-t.reference>1e-9)[0]
    x,y=t.x.ravel()[k],t.y.ravel()[k]
    mass=t.scoop(x,y,1.,source='stockpile',radius=t.cell*.49)
    c=w.containers[0];c.soil_kg=mass;c.location=L.FILLED_QUEUE;c.state='FULL_READY'
    s._update_workfront();assert f.sector_id==0 and s.relocation_pending is None
    w.generate_jobs();jobs=[j for j in w.jobs if j.kind=='PICKUP_FULL_CONTAINER' and j.status=='QUEUED']
    assert jobs and mass>0
    w.batch_tool_requests(jobs,False)
    assert w.courier_active and w.courier_reason=='FINISH_BATCH'
    assert abs(s.report()['mass_error_kg'])<1e-7


def test_ground_follower_can_get_occupied_charge_slot_before_packing():
    c=sector_config();c.base.equipment.landing_slots=1;c.logistics.swap_station=False
    s=Simulation(c);finish_sector(s);d=s.drones[1]
    before=d.battery;d.battery=d.config.battery*d.config.reserve+s._transfer_energy(d,0)+100.
    s._initial_energy-=before-d.battery;s.logistics.sync_batteries()
    for _ in range(800):
        s.step()
        if s.base.destination is not None or s.state.value!='RUNNING':break
    assert s.state.value=='RUNNING',s.reason
    assert s.base.destination is not None and s.base.relocations==1
    assert abs(s.report()['mass_error_kg'])<1e-7 and abs(s.report()['energy_error_j'])<.002


def test_multiple_complexes_keep_exclusive_work_and_sector_state_on_repartition(tmp_path):
    from ats.complexes import ComplexProject,ComplexTemplate
    s=Simulation(sector_config(remnants=True));p=ComplexProject(s);p.enable_shared()
    p.add_shared(ComplexTemplate.capture(s.config))
    a,b=[u.simulation for u in p.units]
    assert not np.any(a.terrain.excavation&b.terrain.excavation)
    bounds=[(u.simulation.terrain.front.sector_start,u.simulation.terrain.front.sector_end) for u in p.units]
    positions=[u.simulation.base.position.copy() for u in p.units]
    for _ in range(30):p.step()
    assert not p.blocked,p.report()
    for i,(u,pos,bound) in enumerate(zip(p.units,positions,bounds)):
        assert (u.simulation.terrain.front.sector_start,u.simulation.terrain.front.sector_end)==bound
        if i==0:np.testing.assert_array_equal(u.simulation.base.position,pos)
        else:assert u.simulation.base.relocations<=1  # initial positioning of its newly assigned work
    path=tmp_path/'group.atsstate';save_state(path,p.active.simulation,complex_project=p)
    q=load_state(path).complex_project
    for _ in range(20):p.step();q.step()
    assert p.report()==q.report()
    for u,v in zip(p.units,q.units):assert u.simulation.report()==v.simulation.report()
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002


def test_sector_and_remaining_volume_are_visible_in_current_ui():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.checkpoint import Session
    from tests.test_workfront import canal
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s=canal(sector_length=20.);w=MainWindow(s.config,session=Session(s));w.timer.stop()
    try:
        w.refresh(force=True)
        assert 'Сектор' in w.channel_status.text() and 'осталось' in w.channel_status.text()
        assert w.workfront_outline.getData()[0] is not None
        outline=w._sector_outline_data;s.terrain.revision+=1;w.refresh(force=True)
        assert w._sector_outline_data is outline
        s.terrain.front._set_sector(s.terrain.front.sector_end+.1);w.refresh(force=True)
        assert w._sector_outline_data is not outline
    finally:w.close();app.processEvents()


def test_v10_migration_preserves_actor_positions_and_committed_move(tmp_path):
    from tests.test_workfront import canal
    s=canal(sector_length=False);s.run(10)
    path=tmp_path/'old.atsstate';save_state(path,s)
    replacement=tmp_path/'v10.atsstate'
    v10_copy(path,replacement)
    q=load_state(replacement).simulation
    assert q.terrain.front.sectors and q.terrain.front.stationed
    np.testing.assert_array_equal(s.base.position,q.base.position)
    np.testing.assert_array_equal(s.terrain.height,q.terrain.height)
    assert s.time==q.time
    assert list(s.logistics.inventory.items)==list(q.logistics.inventory.items)
    assert q.terrain._spoil_front_key is None
    for a,b in zip(s.drones,q.drones):np.testing.assert_array_equal(a.position,b.position)


def test_v10_flight_across_new_boundary_finishes_its_committed_grab(tmp_path):
    c=sector_config();t=Terrain(c);h=t.target.copy()
    for x in (240.5,260.5):
        k=int(np.argmin((t.x-x)**2+(t.y-255.5)**2));h.ravel()[k]=t.reference.ravel()[k]
    c.initial_height=h.tolist();s=Simulation(c);f=s.terrain.front
    f.settings.pop('sector_length_m');f.revision=-1;f.update();s._base_front_anchor=f.position
    for _ in range(150):
        s.step()
        if any(d.trip and d.trip.dig[0]>250 for d in s.drones):break
    assert any(d.trip and d.trip.dig[0]>250 for d in s.drones)
    path=tmp_path/'flight.atsstate';save_state(path,s)
    replacement=tmp_path/'flight-v10.atsstate';v10_copy(path,replacement)
    q=load_state(replacement).simulation;g=q.terrain.front
    assert g.sector_id==0 and g.legacy_pickups
    assert q.terrain.available_mass(260.5,255.5)>0
    grants=g.legacy_pickups;g.legacy_pickups=[]
    assert q.terrain.available_mass(260.5,255.5)==0
    g.legacy_pickups=grants;q.run(300)
    assert q.state.value=='COMPLETED',q.reason
    assert q.terrain.sample(260.5,255.5)<0
    assert abs(q.report()['mass_error_kg'])<1e-7 and abs(q.report()['energy_error_j'])<.002


def test_regional_roundtrip_adopts_arrival_sector_without_extra_short_relocation(tmp_path):
    from ats.channel import ChannelProject
    from ats.landscape import Landscape,ElevationRaster
    from ats.complexes import ComplexProject
    from tests.test_regional import advance
    y,x=np.indices((601,601));h=np.full(x.shape,.02);h[x<80]=-.02
    h[(x-520)**2+(y-300)**2<56**2]=-.02
    raster=ElevationRaster(h,dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=1.,dy=1.))
    channel=ChannelProject.create(raster,[[50,300],[540,300]],width_m=1,depth_m=.05,side_m=128)
    land=Landscape(raster,cell_size=1.);land.channel_project=channel.to_data()
    c=experiment_config(Config(),drones=2,containers=2,landing_slots=2)
    c.logistics.self_solar=None
    origin=np.floor(channel.origin(raster,3)).tolist()
    s=Simulation(channel.configuration(land,c,index=3,origin=origin));s.request_landing()
    p=ComplexProject(s,land);p.enable_shared();ids=p.inventory_ids()
    original=s.terrain.height.copy();original_origin=s.config.georeference['origin_m'].copy()
    p.move_window(section=4);advance(p,lambda:p.transition is None)
    f=s.terrain.front;assert f.sectors and f.stationed
    count=s.base.relocations;s.landing_requested=False;s._update_workfront()
    assert s.relocation_pending is None and s.base.relocations==count
    s.request_landing();p.move_window(section=3);advance(p,lambda:p.transition is None)
    assert s.config.georeference['origin_m']==original_origin and p.inventory_ids()==ids
    np.testing.assert_array_equal(s.terrain.height,original)
    assert s.terrain.front.stationed and not s.terrain.front.moving
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002
