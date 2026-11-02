"""Bounded tile storage, isolated background reads and unchanged physical moves."""
from copy import copy
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import threading
import zipfile
import numpy as np
import pytest
from ats.landscape import Landscape
from ats.soil import SoilProfile,SoilField
from ats.tile_cache import TileCache
from ats.checkpoint import save_state,load_state
from tests.test_landscape import raster
from tests.test_regional import region_group,advance


def prepared(land,origin,n,cell):
    land.prefetch_window(origin,n,cell_size=cell)
    status=land._prefetch_runtime.wait(15.)
    assert status['state']=='READY',status
    return land.prepared_window(origin,n,cell)


def excavate(p):
    t=p.shared_site.core
    load=t.scoop_load(22,40,35.,radius=1.)
    assert load.mass>0;t.deposit(100,100,load.mass,load=load)


def test_shared_lru_evicts_heights_and_materials_without_losing_blocks():
    land=Landscape(raster(),cache_bytes=64*64*8)
    source=np.ones((64,64));land.tiles[0,0]=source
    source[0,0]=99.
    np.testing.assert_array_equal(land.tiles[0,0],np.ones((64,64)))
    with pytest.raises(ValueError):land.tiles[0,0][0,0]=8.
    land.soil_tiles[0,0]=np.full((4,64,64),2.)
    land.tiles[1,0]=np.full((64,64),3.)
    for _ in range(2):
        for key in land.tiles:assert np.isfinite(land.tiles[key]).all()
        assert float(land.soil_tiles[0,0].sum())==4*64*64*2
        assert land.cache_report()['resident_bytes']<=64*64*8
    assert land.cache_report()['evictions']>0
    assert len(land.tiles)==2 and len(land.soil_tiles)==1
    assert land.report()['fine_storage_bytes']==2*64*64*8


def test_frozen_versions_survive_replacement_deletion_and_eviction():
    land=Landscape(raster(),cache_bytes=0);land.tiles[0,0]=np.ones((64,64))
    before=land.snapshot()
    land.tiles[0,0]=np.full((64,64),2.);land.tiles[1,0]=np.full((64,64),3.)
    del land.tiles[0,0]
    assert np.all(before.tiles[0,0]==1.) and (1,0) not in before.tiles
    assert (0,0) not in land.tiles and np.all(land.tiles[1,0]==3.)
    before.tiles[0,0]=np.zeros((64,64))
    assert np.all(land.tiles[1,0]==3.)


def test_window_commit_is_atomic_when_disk_fills_midway(monkeypatch):
    land=Landscape(raster(),cache_bytes=0)
    _,height=land.read_window([0.,0.],128);height[1,1]-=1;land.commit_window([0.,0.],height)
    before=land.report();expected=land.read_window([0.,0.],128)[1]
    height[1,1]-=1;height[70,70]+=2
    real=land._tile_cache._files.write;calls=[]
    def fail(array):
        calls.append(1)
        if len(calls)==2:raise OSError('test disk full')
        return real(array)
    with monkeypatch.context() as m:
        m.setattr(land._tile_cache._files,'write',fail)
        with pytest.raises(OSError,match='disk full'):land.commit_window([0.,0.],height)
    assert land.report()==before
    np.testing.assert_array_equal(land.read_window([0.,0.],128)[1],expected)


@pytest.mark.parametrize('factor',[1,2,4])
def test_stale_prefetch_reloads_changed_height_and_material_columns(factor):
    profile=SoilProfile.qattara();land=Landscape(raster(0,1000),soil_profile=profile,cache_bytes=32*1024)
    origin=[8.25,12.75];n=64;cell=.25*factor
    _,height=land.read_window(origin,n,cell_size=cell)
    field=SoilField(profile,n,cell,land.soil_window(origin,n,cell))
    field.extract([600],[5.],commit=True)
    h=height+field.height_delta(np.arange(n*n)).reshape(n,n)
    land.commit_window(origin,h,cell_size=cell,soil=field)
    target=[origin[0]+8*cell,origin[1]]
    old=prepared(land,target,n,cell)
    _,load=field.extract([601],[9.],commit=True)
    field.add([900],[load.volume/cell**2],load)
    h=height+field.height_delta(np.arange(n*n)).reshape(n,n)
    # Active ground has changed since the read; it is not yet committed.
    preview=land.snapshot();preview.commit_window(origin,h,cell_size=cell,soil=field)
    a=preview.read_window(target,n,cell_size=cell,prepared=old)
    b=preview.read_window(target,n,cell_size=cell)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    stale=preview.soil_window(target,n,cell,prepared=old);fresh=preview.soil_window(target,n,cell)
    assert [(x,y) for x,y,_ in stale['tiles']]==[(x,y) for x,y,_ in fresh['tiles']]
    for (_,_,x),(_,_,y) in zip(stale['tiles'],fresh['tiles']):np.testing.assert_array_equal(x,y)
    assert land.read_window(origin,n,cell_size=cell)[1][9,25]!=h[9,25] or not np.array_equal(old.current,a[1])


def test_cancel_and_coalesced_requests_cannot_publish_late_data(monkeypatch):
    land=Landscape(raster());start=threading.Event();release=threading.Event()
    original=Landscape.prepare_window;calls=[]
    def gate(self,*args,**kwargs):
        calls.append(args[0])
        if len(calls)==1:
            start.set();assert release.wait(10.)
        return original(self,*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(Landscape,'prepare_window',gate)
        land.prefetch_window([0.,0.],64);assert start.wait(5.)
        runtime=land._prefetch_runtime;thread=runtime._thread
        land.cancel_prefetch();assert runtime.report()['state']=='CANCELLED'
        for x in range(1,20):land.prefetch_window([float(x),0.],64)
        assert runtime._thread is thread
        release.set();assert runtime.wait(10.)['state']=='READY'
    assert len(calls)==2 and runtime.prepared((19.,0.,64,.25)) is not None
    assert not land.tiles and land.report()['soil_delta_m3']==0


def test_reader_error_and_memory_limit_do_not_touch_live_ground(monkeypatch):
    land=Landscape(raster());before=land.report()
    with monkeypatch.context() as m:
        def fail(*a,**k):raise OSError('test unreadable tile')
        m.setattr(Landscape,'prepare_window',fail)
        land.prefetch_window([0.,0.],64)
        status=land._prefetch_runtime.wait(5.)
        assert status['state']=='ERROR' and 'unreadable' in status['error']
    assert land.report()==before
    assert prepared(land,[0.,0.],64,.25) is not None
    snapshot=land.snapshot()
    with pytest.raises(ValueError,match='лимит памяти'):snapshot.prepare_window([0.,0.],64,.25,budget=1,check=lambda *a:None)


def test_thread_start_failure_is_visible_and_can_be_retried(monkeypatch):
    land=Landscape(raster());before=land.report()
    with monkeypatch.context() as m:
        def fail(*a,**k):raise RuntimeError('test thread resource limit')
        m.setattr(threading.Thread,'start',fail)
        land.prefetch_window([0.,0.],64)
        result=land._prefetch_runtime.wait(1.)
        assert result['state']=='ERROR' and 'resource limit' in result['error']
    assert land.report()==before
    assert prepared(land,[0.,0.],64,.25) is not None


@pytest.mark.parametrize('limit',[0,32*1024])
def test_prefetched_group_round_trip_preserves_volume_composition_ids_and_energy(limit):
    p=region_group(soil=True);land=p.landscape;land._tile_cache.set_limit(limit)
    excavate(p);t=p.shared_site.core;heights=t.height.copy()
    columns={k:a.copy() for k,a in t.soil.tiles.items()};ids=p.inventory_ids()
    prepared(land,[96.,64.],256,.5)
    p.move_window([96.,64.]);advance(p,lambda:p.transition is None)
    assert abs(land.report()['soil_delta_m3'])<1e-9
    assert land.cache_report()['resident_bytes']<=limit
    prepared(land,[64.,64.],256,.5)
    p.move_window([64.,64.]);advance(p,lambda:p.transition is None)
    np.testing.assert_array_equal(p.shared_site.core.height,heights)
    for k,a in columns.items():np.testing.assert_array_equal(p.shared_site.core.soil.tiles[k],a)
    assert p.inventory_ids()==ids
    report=p.report()
    assert abs(report['mass_error_kg'])<1e-7 and abs(report['energy_error_j'])<.002
    assert report['shared_worksite']['max_material_mass_error_kg']<1e-7


def test_snapshot_while_reader_runs_is_self_contained_and_resumes_request(tmp_path,monkeypatch):
    p=region_group(count=1,soil=True);land=p.landscape;excavate(p);p.persist_window()
    land._tile_cache.set_limit(0);start=threading.Event();release=threading.Event()
    original=Landscape.prepare_window
    def gate(self,*args,**kwargs):
        if self.raster is land.raster:
            start.set();assert release.wait(10.)
        return original(self,*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(Landscape,'prepare_window',gate)
        land.prefetch_window([96.,64.],256,cell_size=.5);assert start.wait(5.)
        path=tmp_path/'reading.atsstate';save_state(path,p.active.simulation,land,complex_project=p)
        with zipfile.ZipFile(path) as archive:
            manifest=json.loads(archive.read('manifest.json'));assert manifest['version']==13
            assert 'ats-window-read' not in json.dumps(manifest) and land._tile_cache._files.directory.name not in json.dumps(manifest)
        q=load_state(path).complex_project
        assert q.report()==p.report() and q.landscape._prefetch_runtime is None
        q.landscape.resume_prefetch();assert q.landscape._prefetch_runtime.wait(10.)['state']=='READY'
        land.cancel_prefetch();release.set()
    q.move_window([96.,64.]);advance(q,lambda:q.transition is None)
    assert abs(q.report()['mass_error_kg'])<1e-7 and q.landscape.cache_report()['resident_bytes']==0


def test_exact_resume_mid_drive_with_evicted_tiles_and_prepared_window(tmp_path):
    p=region_group(count=1,soil=True);land=p.landscape;excavate(p);p.persist_window()
    land._tile_cache.set_limit(0);prepared(land,[96.,64.],256,.5)
    p.move_window([96.,64.]);advance(p,lambda:p.transition.phase=='DRIVE')
    path=tmp_path/'drive.atsstate';save_state(path,p.active.simulation,land,complex_project=p)
    q=load_state(path).complex_project
    assert q.report()==p.report() and len(q.landscape.tiles)>0
    for _ in range(3000):
        if p.transition is None and q.transition is None:break
        p.step();q.step()
    assert p.transition is None and q.transition is None
    assert q.report()==p.report()
    np.testing.assert_array_equal(q.shared_site.core.height,p.shared_site.core.height)
    for key,a in p.shared_site.core.soil.tiles.items():np.testing.assert_array_equal(q.shared_site.core.soil.tiles[key],a)


def test_cache_io_failure_at_handoff_retains_active_window_and_can_retry(monkeypatch,tmp_path):
    p=region_group(count=1,soil=True);land=p.landscape;excavate(p)
    old=p.shared_site.core;origin=p.active.simulation.config.georeference['origin_m'].copy();ids=p.inventory_ids()
    p.move_window([96.,64.])
    with monkeypatch.context() as m:
        def fail(*a,**k):raise OSError('test disk full')
        m.setattr(land._tile_cache._files,'write',fail)
        advance(p,lambda:bool(getattr(p.transition,'window_error',None)))
    assert p.shared_site.core is old and p.active.simulation.config.georeference['origin_m']==origin
    assert 'disk full' in p.transition.reason and not land.tiles and p.inventory_ids()==ids
    path=tmp_path/'retry.atsstate';save_state(path,p.active.simulation,land,complex_project=p)
    q=load_state(path).complex_project;assert q.transition.window_error
    q.transition.retry_window();advance(q,lambda:q.transition is None)
    assert q.active.simulation.config.georeference['origin_m']==[96.,64.] and q.inventory_ids()==ids
    assert abs(q.report()['mass_error_kg'])<1e-7 and abs(q.report()['energy_error_j'])<.002


def test_region_archive_streams_evicted_tiles_and_failed_save_keeps_existing_file(tmp_path,monkeypatch):
    land=Landscape(raster(),cache_bytes=0)
    for i in range(20):land.tiles[i,0]=np.full((64,64),float(i))
    path=tmp_path/'world.atsregion';land.save(path);original=path.read_bytes()
    clone=Landscape.load(path)
    assert clone.report()==land.report() and clone.cache_report()['resident_bytes']==0
    for key in land.tiles:np.testing.assert_array_equal(clone.tiles[key],land.tiles[key])
    with monkeypatch.context() as m:
        def fail(*a,**k):raise OSError('test save error')
        m.setattr(np.lib.format,'write_array',fail)
        with pytest.raises(OSError):land.save(path)
    assert path.read_bytes()==original


def test_cached_report_does_not_read_all_backing_files(monkeypatch):
    land=Landscape(raster(),cache_bytes=0)
    for i in range(20):land.tiles[i,0]=np.ones((64,64))
    with monkeypatch.context() as m:
        from ats.tile_cache import TileVersion
        def fail(*a,**k):raise AssertionError('report must use derived metadata')
        m.setattr(TileVersion,'read',fail)
        r=land.report();assert r['soil_delta_m3']==20*64*64*.25**2
        assert land.cache_report()['resident_bytes']==0


def test_v12_dictionary_snapshot_migrates_without_losing_tile_values(tmp_path):
    from ats.simulation import Simulation
    land=Landscape(raster(),soil_profile=SoilProfile.qattara())
    field=SoilField(land.soil_profile,64,.25,land.soil_window([0.,0.],64,.25))
    field.extract([131],[.1],commit=True)
    baseline,_=land.read_window([0.,0.],64)
    land.commit_window([0.,0.],baseline+field.height_delta(np.arange(4096)).reshape(64,64),soil=field)
    old=copy(land);old.tiles={k:a.copy() for k,a in land.tiles.items()};old.soil_tiles={k:a.copy() for k,a in land.soil_tiles.items()}
    for key in ('_tile_cache','prefetch_enabled','prefetch_request','prefetch_cancelled','_prefetch_runtime','_prefetch_auto_key'):
        delattr(old,key)
    path=tmp_path/'old.atsstate';save_state(path,Simulation(),old)
    with zipfile.ZipFile(path) as archive:contents={n:archive.read(n) for n in archive.namelist()}
    manifest=json.loads(contents['manifest.json']);manifest['version']=12
    contents['manifest.json']=json.dumps(manifest).encode()
    with zipfile.ZipFile(path,'w') as archive:
        for name,data in contents.items():archive.writestr(name,data)
    restored=load_state(path).landscape
    assert restored.report()==land.report()
    for key,array in land.tiles.items():np.testing.assert_array_equal(restored.tiles[key],array)
    for key,array in land.soil_tiles.items():np.testing.assert_array_equal(restored.soil_tiles[key],array)


def test_cancelled_automatic_prefetch_stays_cancelled_after_save_load(tmp_path):
    from tests.test_window_boundaries import window
    p,ch=window();land=p.landscape
    land.ensure_next_prefetch(p.active.simulation)
    assert land.prefetch_request is not None
    land._prefetch_runtime.wait(10.);land.cancel_prefetch()
    path=tmp_path/'cancelled.atsstate';save_state(path,p.active.simulation,land,complex_project=p)
    q=load_state(path).complex_project
    for _ in range(10):q.landscape.ensure_next_prefetch(q.active.simulation)
    assert q.landscape.prefetch_request is None and q.landscape._prefetch_runtime is None
    assert q.landscape.cache_report()['prefetch']['state']=='CANCELLED'


def test_prepared_data_from_another_raster_cannot_replace_new_region():
    a=Landscape(raster(10.));b=Landscape(raster(20.))
    old=prepared(a,[0.,0.],64,.25)
    baseline,current=b.read_window([0.,0.],64,prepared=old)
    assert np.all(baseline==20.) and np.all(current==20.)


def test_disk_corruption_is_an_error_and_never_a_zero_height_tile():
    land=Landscape(raster(),cache_bytes=0);land.tiles[0,0]=np.ones((64,64))
    record=land._tile_cache.version(('height',0,0))
    with record.path.open('wb') as stream:np.save(stream,np.full((64,64),2.),allow_pickle=False)
    with pytest.raises(OSError,match='файлового кэша'):land.read_window([0.,0.],64)
    land.tiles[0,0]=np.ones((64,64))
    assert land.read_window([0.,0.],64)[1][0,0]==11.


@pytest.mark.parametrize('factor',[1,2,4])
def test_bounded_interpolation_strips_match_original_full_sampling_exactly(factor):
    from ats.landscape import ElevationRaster
    rng=np.random.default_rng(72);r=ElevationRaster(rng.uniform(-12,34,(5,5)),raster().metadata)
    land=Landscape(r);origin=[13.25,24.5];n=128;cell=.25*factor
    axis=(np.arange(n)+.5)*cell;x,y=np.meshgrid(axis+origin[0],axis+origin[1])
    np.testing.assert_array_equal(land.read_window(origin,n,cell_size=cell)[0],r.sample(x,y))


def test_ui_can_cancel_a_read_while_paused_and_set_shared_ram_budget(monkeypatch):
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.landscape_ui import LandscapeDialog
    from ats.checkpoint import Session
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p=region_group(count=1);w=MainWindow(p.active.simulation.config,session=Session(p.active.simulation,p.landscape,{},complex_project=p));w.timer.stop()
    d=LandscapeDialog(w);d.x.setValue(96);d.y.setValue(64)
    started=threading.Event();release=threading.Event();original=Landscape.prepare_window
    def gate(self,*a,**k):started.set();assert release.wait(10.);return original(self,*a,**k)
    try:
        with monkeypatch.context() as m:
            m.setattr(Landscape,'prepare_window',gate)
            d.prefetch_button.click();assert started.wait(5.)
            d.refresh_prefetch();assert d.prefetch_cancel.isEnabled()
            d.cache_limit.setValue(1);d.prefetch_cancel.click()
            assert p.landscape.cache_report()['prefetch']['state']=='CANCELLED'
            assert p.landscape._tile_cache.limit_bytes==1024**2 and p.time==0
            release.set()
    finally:release.set();d.close();w.close();app.processEvents()
