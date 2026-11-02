"""World-coordinate relief and bounded, route-independent window preparation."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
import threading
import zipfile
import numpy as np
import pytest
from ats.landscape import Landscape,ElevationRaster,workspace_config
from ats.procedural import default_generation,sample
from ats.soil import SoilProfile,SoilField
from ats.config import Config
from ats.experiments import experiment_config
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.complexes import ComplexProject,ComplexTemplate
from ats.checkpoint import save_state,load_state
from ats.window_prefetch import WindowPrefetch
from ats.world_regions import interest_windows
from tests.test_landscape import raster
from tests.test_regional import advance


def generated_group(count=2):
    land=Landscape(raster(10.,200.),cell_size=.5,soil_profile=SoilProfile.qattara(),
        generation=default_generation(amplitude=.03),cache_bytes=0)
    c=workspace_config(land,experiment_config(Config(),drones=2,containers=1),[128.,128.],128.)
    c.logistics.field_deployment_seconds=2.
    c.plan=PlanConfig([PlanSegment([[20,40],[26,40]],width=2,height=9.8)])
    p=ComplexProject(Simulation(c),land);p.enable_shared()
    for _ in range(count-1):p.add_shared(ComplexTemplate.capture(c))
    return p


def test_world_seed_overlap_and_read_order_are_independent():
    a=Landscape(raster(),generation=default_generation(seed=123))
    b=Landscape(raster(),generation=default_generation(seed=124))
    left=a.read_window([16.,24.],128)[0]
    right=a.read_window([24.,32.],128)[0]
    np.testing.assert_array_equal(left[32:,32:],right[:-32,:-32])
    assert np.ptp(left)>0 and np.max(abs(left-10))<=.1
    assert not np.array_equal(left,b.read_window([16.,24.],128)[0])
    np.testing.assert_array_equal(a.read_window([16.,24.],128)[0],left)
    assert not a.tiles and not a.soil_tiles


def test_source_nodes_coast_and_nodata_are_preserved():
    r=ElevationRaster(np.array([[-10.,0.,10.],[-10.,0.,10.],[-10.,0.,10.]]),
        dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=100.,dy=100.))
    g=default_generation(amplitude=1.)
    np.testing.assert_array_equal(sample(r,np.array([0.,100.,200.]),np.array([0.,100.,200.]),g),[-10.,0.,10.])
    x=np.linspace(.25,199.75,500);y=np.full(500,47.25)
    assert np.array_equal(np.sign(sample(r,x,y,g)),np.sign(r.sample(x,y)))
    r.height.flags.writeable=True;r.height[1,1]=np.nan;r.height.flags.writeable=False
    with pytest.raises(ValueError,match='NoData'):sample(r,100.,100.,g)


@pytest.mark.parametrize('factor',[2,4])
def test_coarse_relief_is_mean_of_persistent_fine_columns(factor):
    land=Landscape(raster(),generation=default_generation())
    fine=land.read_window([16.,24.],64*factor)[0]
    coarse=land.read_window([16.,24.],64,cell_size=.25*factor)[0]
    np.testing.assert_allclose(coarse,fine.reshape(64,factor,64,factor).mean((1,3)),rtol=0,atol=4e-15)
    fine[3,7]-=.2;fine[8,11]+=.2
    land.commit_window([16.,24.],fine)
    reread=land.read_window([16.,24.],64,cell_size=.25*factor)[1]
    assert float((reread-coarse).sum())*(.25*factor)**2==pytest.approx(0,abs=1e-10)


def test_legacy_baseline_and_generator_lock_are_explicit(tmp_path):
    land=Landscape(raster());origin=[16.,24.]
    plain=land.read_window(origin,128)[0]
    np.testing.assert_array_equal(plain,np.full((128,128),10.))
    land.configure_generation(default_generation());land.prefetch_mode='world'
    assert not np.array_equal(land.read_window(origin,128)[0],plain)
    workspace_config(land,Config(),origin,32.)
    with pytest.raises(ValueError,match='закреплён'):land.configure_generation(default_generation(seed=99))
    path=tmp_path/'generated.atsregion';land.save(path);clone=Landscape.load(path)
    assert clone.generation==land.generation and clone.baseline_locked and clone.prefetch_mode=='world'
    np.testing.assert_array_equal(clone.read_window(origin,128)[0],land.read_window(origin,128)[0])
    legacy=Landscape(raster());path=tmp_path/'legacy.atsregion';legacy.save(path)
    with np.load(path,allow_pickle=False) as z:assert json.loads(str(z['metadata']))['version']==2
    old=Landscape.load(path);assert old.generation is None
    np.testing.assert_array_equal(old.read_window(origin,128)[0],plain)


@pytest.mark.parametrize('bad',[dict(version=2,seed=42,amplitude_m=.1,scale_m=16.),
    dict(version=1,seed=-1,amplitude_m=.1,scale_m=16.),
    dict(version=1,seed=42,amplitude_m=float('nan'),scale_m=16.)])
def test_unsupported_generation_fails_before_creating_a_world(bad):
    with pytest.raises(ValueError,match='Генератор'):Landscape(raster(),generation=bad)


def test_batch_prefetch_is_bounded_and_preserves_highest_priority_window():
    land=Landscape(raster(),generation=default_generation())
    keys=[[x,32.,64,.25] for x in (16.,32.,48.,64.)]
    runtime=WindowPrefetch();budget=2*64*64*8*2
    runtime.request_many(land.snapshot(),keys,budget)
    status=runtime.wait(10.)
    assert status['state']=='READY' and status['ready_windows']==2 and status['prepared_bytes']<=budget
    assert runtime.prepared(keys[0]) is not None
    for key in keys:
        p=runtime.prepared(key)
        if p:np.testing.assert_array_equal(land.read_window(key[:2],64)[1],p.current)
    assert not land.tiles


def test_batch_reuses_unchanged_windows_and_cancels_without_late_publish(monkeypatch):
    land=Landscape(raster(),generation=default_generation());keys=[[x,32.,64,.25] for x in (16.,32.,48.)]
    land.prefetch_windows(keys);assert land._prefetch_runtime.wait(10.)['state']=='READY'
    cached=land.prepared_window(keys[0][:2],64,.25);calls=[];original=Landscape.prepare_window
    def count(self,*a,**k):calls.append(a[0]);return original(self,*a,**k)
    with monkeypatch.context() as m:
        m.setattr(Landscape,'prepare_window',count)
        land.prefetch_windows([keys[0],keys[1],[64.,32.,64,.25]])
        assert land._prefetch_runtime.wait(10.)['state']=='READY'
    assert len(calls)==1 and land.prepared_window(keys[0][:2],64,.25) is cached
    start=threading.Event();release=threading.Event()
    def gate(self,*a,**k):start.set();assert release.wait(10.);return original(self,*a,**k)
    with monkeypatch.context() as m:
        m.setattr(Landscape,'prepare_window',gate)
        land.prefetch_windows([[80.,32.,64,.25]]);assert start.wait(5.)
        worker=land._prefetch_runtime._thread;land.cancel_prefetch();release.set();worker.join(10.)
        assert not worker.is_alive()
    assert land._prefetch_runtime.report()['state']=='CANCELLED'
    assert land._prefetch_runtime.report()['prepared_bytes']==0


def test_automatic_interest_works_without_a_canal_and_in_both_axes(tmp_path):
    p=generated_group();land=p.landscape;s=p.active.simulation
    assert land.channel_project is None
    keys=interest_windows(land,s)
    assert len(keys)==4 and all(land._grid(k[:2],k[2],k[3]) is not None for k in keys)
    for xy in ((5,5),(120,120),(120,5),(5,120)):
        s.base.position[:2]=xy
        for d in s.drones:d.position[:2]=xy
        assert interest_windows(land,s)
    land.ensure_next_prefetch(s);assert land._prefetch_runtime.wait(10.)['state']=='READY'
    assert len(land.prefetch_request['keys'])==4
    first=land.prefetch_request.copy();land.cancel_prefetch()
    for _ in range(5):
        p.time+=5
        for u in p.units:u.simulation.time+=5
        land.ensure_next_prefetch(s)
    assert land.prefetch_request is None
    path=tmp_path/'cancelled.atsstate';save_state(path,s,land,complex_project=p)
    q=load_state(path).complex_project;q.landscape.ensure_next_prefetch(q.active.simulation)
    assert q.landscape.prefetch_request is None and q.landscape.generation==land.generation


def test_generated_ground_and_materials_survive_cross_direction_round_trip_and_resume(tmp_path):
    p=generated_group();t=p.shared_site.core;land=p.landscape
    load=t.scoop_load(22,40,35.,radius=1.);assert load.mass>0
    t.deposit(100,100,load.mass,load=load)
    ground=t.height.copy();columns={k:a.copy() for k,a in t.soil.tiles.items()};ids=p.inventory_ids();start=p.time
    land.prefetch_windows([[160.,128.,256,.5],[128.,160.,256,.5]])
    assert land._prefetch_runtime.wait(10.)['state']=='READY'
    p.move_window([160.,128.]);advance(p,lambda:p.transition.phase=='DRIVE')
    path=tmp_path/'procedural-drive.atsstate';save_state(path,p.active.simulation,land,complex_project=p)
    p=load_state(path).complex_project;land=p.landscape
    assert land.generation==default_generation(amplitude=.03)
    advance(p,lambda:p.transition is None)
    p.move_window([128.,160.]);advance(p,lambda:p.transition is None)
    p.move_window([128.,128.]);advance(p,lambda:p.transition is None)
    np.testing.assert_array_equal(p.shared_site.core.height,ground)
    for k,a in columns.items():np.testing.assert_array_equal(p.shared_site.core.soil.tiles[k],a)
    assert ids==p.inventory_ids() and p.time>start
    r=p.report();assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002
    assert abs(land.report()['soil_delta_m3'])<1e-9
    assert land.cache_report()['resident_bytes']==0


def test_generated_snapshot_resumes_multi_window_request(tmp_path):
    p=generated_group(count=1);land=p.landscape
    land.prefetch_windows([[160.,128.,256,.5],[128.,160.,256,.5]])
    assert land._prefetch_runtime.wait(10.)['state']=='READY'
    path=tmp_path/'reading.atsstate';save_state(path,p.active.simulation,land,complex_project=p)
    q=load_state(path).complex_project
    assert q.landscape._prefetch_runtime is None and len(q.landscape.prefetch_request['keys'])==2
    q.landscape.resume_prefetch();assert q.landscape._prefetch_runtime.wait(10.)['ready_windows']==2
    assert q.report()==p.report()


def test_v13_snapshot_migrates_without_regenerating_old_ground(tmp_path):
    from tests.test_regional import region_group
    p=region_group(count=1,soil=True);ground=p.shared_site.core.height.copy()
    path=tmp_path/'old.atsstate';save_state(path,p.active.simulation,p.landscape,complex_project=p)
    with zipfile.ZipFile(path) as z:entries={k:z.read(k) for k in z.namelist()}
    manifest=json.loads(entries['manifest.json']);manifest['version']=13
    for node in manifest['nodes']:
        if node.get('type')=='ats.landscape.Landscape':
            for key in ('generation','baseline_locked','prefetch_mode','prefetch_cancelled_keys'):node['attrs'].pop(key,None)
    entries['manifest.json']=json.dumps(manifest).encode()
    with zipfile.ZipFile(path,'w') as z:
        for k,v in entries.items():z.writestr(k,v)
    q=load_state(path).complex_project
    assert q.landscape.generation is None and q.landscape.baseline_locked
    np.testing.assert_array_equal(q.shared_site.core.height,ground)
    with pytest.raises(ValueError,match='закреплён'):q.landscape.configure_generation(default_generation())


def test_failed_background_generation_and_retry_preserve_live_mission(monkeypatch):
    p=generated_group(count=1);land=p.landscape;before=p.report();ground=p.shared_site.core.height.copy()
    original=Landscape.prepare_window
    with monkeypatch.context() as m:
        def fail(*a,**k):raise OSError('test procedural read error')
        m.setattr(Landscape,'prepare_window',fail)
        land.prefetch_windows([[160.,128.,256,.5],[128.,160.,256,.5]])
        status=land._prefetch_runtime.wait(10.)
        assert status['state']=='ERROR' and 'test procedural read error' in status['error']
    assert p.report()==before
    np.testing.assert_array_equal(p.shared_site.core.height,ground)
    land.prefetch_windows([[160.,128.,256,.5]])
    assert land._prefetch_runtime.wait(10.)['state']=='READY'


def test_arbitrary_lateral_channel_window_preserves_global_plan_and_group():
    from tests.test_channel import coastal_dem
    from ats.channel import ChannelProject
    land=Landscape(coastal_dem(gap=100),generation=default_generation(amplitude=.03))
    ch=ChannelProject.create(land.raster,[[50,150],[250,150]],width_m=1,depth_m=.01,side_m=128)
    land.channel_project=ch.to_data()
    c=ch.configuration(land,experiment_config(Config(),drones=2,containers=1));c.logistics.field_deployment_seconds=2
    p=ComplexProject(Simulation(c),land);p.enable_shared();ids=p.inventory_ids()
    origin=np.array(c.georeference['origin_m']);height=p.shared_site.core.height.copy()
    p.move_window((origin+[0.,16.]).tolist())
    advance(p,lambda:p.transition is None)
    assert p.active.simulation.config.plan.channel['project_id']==ch.identity
    assert p.active.simulation.config.georeference['channel']['custom_window']
    p.move_window(origin.tolist());advance(p,lambda:p.transition is None)
    np.testing.assert_array_equal(p.shared_site.core.height,height)
    assert p.inventory_ids()==ids and p.landscape.channel_project['completed']==[]
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002


def test_window_planning_keeps_enlarged_cross_channel_extent():
    from tests.test_channel import coastal_dem
    from ats.channel import ChannelProject
    from ats.regional import target_configuration
    land=Landscape(coastal_dem(gap=100),generation=default_generation(amplitude=.03))
    ch=ChannelProject.create(land.raster,[[50,150],[250,150]],width_m=1,depth_m=.01,side_m=128)
    land.channel_project=ch.to_data()
    c=ch.configuration(land,experiment_config(Config(),drones=2,containers=1));c.logistics.field_deployment_seconds=2
    p=ComplexProject(Simulation(c),land);site=p.enable_shared();site.enlarge(160)
    target=target_configuration(p,c.georeference['origin_m'],section=0)
    assert target.terrain.n*target.terrain.cell_size==160
    assert target.georeference['channel']['project_id']==ch.identity
    assert target.georeference['channel']['custom_window']


def test_ui_new_world_settings_selected_point_and_existing_region_lock():
    from PySide6 import QtCore,QtWidgets
    from ats.gui import MainWindow
    from ats.landscape_ui import LandscapeDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(Config());w.timer.stop();dialog=LandscapeDialog(w)
    try:
        assert dialog.region.generation==default_generation() and dialog.region.prefetch_mode=='world'
        def edit():
            modal=app.activeModalWidget()
            assert isinstance(modal,QtWidgets.QDialog)
            # The settings panel's seed uses an integer-valued double spinbox.
            seed=next(q for q in modal.findChildren(QtWidgets.QDoubleSpinBox) if q.maximum()==2**32-1)
            seed.setValue(2026)
            modal.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
        QtCore.QTimer.singleShot(0,edit);dialog.edit_generation()
        assert dialog.region.generation['seed']==2026
        dialog.x.setValue(1000);dialog.y.setValue(2000)
        # This point is independent of a drawn route and can belong to a new DEM.
        dialog.prefetch_selected();assert dialog.region._prefetch_runtime.wait(10.)['state']=='READY'
        assert dialog.region.prefetch_request['key'][:2]==[1000.,2000.]
        dialog.cancel_prefetch();assert not dialog.selection_timer.isActive()
        assert dialog.region._prefetch_runtime.report()['state']=='CANCELLED'
    finally:
        dialog.close();w.close();app.processEvents()


def test_return_after_map_enlarge_keeps_grid_manual_plan_and_updated_marine_mask():
    from tests.test_channel import coastal_dem
    from ats.channel import ChannelProject,marine_window
    land=Landscape(coastal_dem(gap=100),generation=default_generation(amplitude=.03))
    ch=ChannelProject.create(land.raster,[[50,150],[250,150]],width_m=1,depth_m=.01,side_m=128)
    land.channel_project=ch.to_data()
    c=ch.configuration(land,experiment_config(Config(),drones=2,containers=1));c.logistics.field_deployment_seconds=2
    p=ComplexProject(Simulation(c),land);site=p.enable_shared();ids=p.inventory_ids()
    origin=np.array(c.georeference['origin_m'])
    p.move_window((origin+[0.,8.]).tolist());advance(p,lambda:p.transition is None)
    assert tuple(origin) in p.window_plans
    original=site.core.height.copy();site.enlarge(160)
    assert site.core.config.georeference['marine']==marine_window(land.raster,(origin+[0.,8.]).tolist(),160)
    p.move_window(origin.tolist());advance(p,lambda:p.transition is None)
    assert p.active.simulation.config.terrain.n*p.active.simulation.config.terrain.cell_size==160
    assert p.active.simulation.config.georeference['marine']==marine_window(land.raster,origin.tolist(),160)
    assert p.inventory_ids()==ids and p.active.simulation.config.georeference['channel']['custom_window']
    assert p.shared_site.core.height.shape==(640,640)
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002
