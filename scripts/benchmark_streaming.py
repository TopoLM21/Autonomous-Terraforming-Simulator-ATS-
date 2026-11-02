"""Reproducible bounded-cache, Qt-heartbeat and physical-handoff controls.

Run this same script against --root checkouts of the baseline and new code.
RSS is available on Linux; Qt uses offscreen. These are not user-PC FPS or
evidence about a 23-hour mission freeze. No network or user save is required.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time


def rss(peak=False):
    if platform.system()!='Linux':return None
    if peak:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024


def window_digest(data):
    baseline,current,soil=data;digest=hashlib.sha256(baseline);digest.update(current)
    for x,y,a in (soil or {}).get('tiles',[]):
        digest.update(f'{x},{y}:'.encode());digest.update(a)
    return digest.hexdigest()


def heartbeat(operation,done,timeout=60.):
    from PySide6 import QtCore,QtWidgets
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    widget=QtWidgets.QLabel('Подготовка окна · контроль очереди интерфейса')
    widget.resize(500,70);widget.show();app.processEvents()
    gaps=[];last=[time.perf_counter()]
    timer=QtCore.QTimer();timer.setInterval(5)
    def beat():
        now=time.perf_counter();gaps.append(now-last[0]);last[0]=now
    timer.timeout.connect(beat);timer.start();start=time.perf_counter()
    QtCore.QTimer.singleShot(0,operation)
    while True:
        app.processEvents()
        if done():break
        if time.perf_counter()-start>timeout:raise TimeoutError('Qt control deadline')
        time.sleep(.001)
    gaps.append(time.perf_counter()-last[0]);timer.stop();widget.close();widget.deleteLater();app.processEvents()
    return dict(wall_seconds=time.perf_counter()-start,max_event_gap_seconds=max(gaps),heartbeats=len(gaps)-1)


def memory_control(args):
    import numpy as np
    from ats.landscape import Landscape,ElevationRaster
    from ats.soil import SoilProfile
    metadata=dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=500.,dy=500.,source='paired synthetic tiles')
    native=SoilProfile.qattara()
    kwargs=dict(soil_profile=native)
    streaming='cache_bytes' in inspect.signature(Landscape).parameters
    if streaming:kwargs['cache_bytes']=args.cache_mib*1024**2
    start_rss=rss();land=Landscape(ElevationRaster(np.zeros((5,5)),metadata),**kwargs)
    begin=time.perf_counter()
    for i in range(args.tiles):
        key=(i%64,i//64);negative=i%2==0
        h=np.full((64,64),-.125 if negative else .125)
        s=np.zeros((len(native.materials)+1,64,64));s[0 if negative else 1]=.125
        land.tiles[key]=h;land.soil_tiles[key]=s
    del h,s
    build_seconds=time.perf_counter()-begin;gc.collect()
    built_rss=rss();built_peak=rss(True)
    for key in land.tiles:
        assert np.isfinite(land.tiles[key]).all() and np.isfinite(land.soil_tiles[key]).all()
    report=land.report();assert abs(report['soil_delta_m3'])<1e-12
    storage=(land.cache_report() if streaming else dict(resident_bytes=report['fine_storage_bytes']+report['soil_storage_bytes'],limit_bytes=None))
    with tempfile.TemporaryDirectory(prefix='ats-stream-benchmark-') as directory:
        path=Path(directory)/'world.atsregion';start=time.perf_counter();land.save(path)
        save_seconds=time.perf_counter()-start;start=time.perf_counter();other=Landscape.load(path)
        load_seconds=time.perf_counter()-start;assert other.report()==report
        restore_peak=rss(True);del other;gc.collect()
    after_restore_rss=rss()
    origin=[8.,8.];n=1024;cell=.25;result=[]
    def read(prepared=None):
        if prepared is None:return (*land.read_window(origin,n,cell_size=cell),land.soil_window(origin,n,cell))
        return (*land.read_window(origin,n,cell_size=cell,prepared=prepared),land.soil_window(origin,n,cell,prepared=prepared))
    if streaming:
        def request():land.prefetch_window(origin,n,cell_size=cell)
        waiting=heartbeat(request,lambda:land.cache_report()['prefetch']['state'] in ('READY','ERROR'))
        status=land.cache_report()['prefetch'];assert status['state']=='READY',status
        ready=land.prepared_window(origin,n,cell)
    else:
        waiting=heartbeat(lambda:result.append(read()),lambda:bool(result));ready=None
        status=None
    after_prefetch_rss=rss()
    start=time.perf_counter();warm=read(ready);warm_seconds=time.perf_counter()-start
    digest=window_digest(warm);del warm;result.clear()
    # A conservative pair changes after reading. Refresh must use both current
    # heights and native/spoil columns, without reusing stale prepared soil.
    for key,sign,layer in (((0,2),-1,0),((1,2),1,1)):
        h=land.tiles[key].copy();h+=sign*.03125;land.tiles[key]=h
        s=land.soil_tiles[key].copy();s[layer]+=.03125;land.soil_tiles[key]=s
    start=time.perf_counter();refreshed=read(ready);refresh_seconds=time.perf_counter()-start
    refreshed_digest=window_digest(refreshed);fresh=read()
    assert refreshed_digest==window_digest(fresh)
    assert abs(land.report()['soil_delta_m3'])<1e-12
    return dict(streaming=streaming,modified_height_tiles=args.tiles,modified_material_tiles=args.tiles,
        initial_rss_bytes=start_rss,after_build_rss_bytes=built_rss,after_build_peak_rss_bytes=built_peak,
        after_restore_peak_rss_bytes=restore_peak,after_restore_rss_bytes=after_restore_rss,
        after_prefetch_rss_bytes=after_prefetch_rss,end_peak_rss_bytes=rss(True),
        build_seconds=build_seconds,save_seconds=save_seconds,
        load_seconds=load_seconds,cache=storage,window_grid=[n,n],window_cell_m=cell,
        first_read_ui=waiting,prefetch=status,warm_window_data_seconds=warm_seconds,
        stale_window_data_seconds=refresh_seconds,window_digest=digest,refreshed_digest=refreshed_digest,
        region_volume_error_m3=land.report()['soil_delta_m3'])


def physical_control():
    import numpy as np
    from ats.landscape import Landscape,ElevationRaster,workspace_config
    from ats.soil import SoilProfile
    from ats.config import Config
    from ats.experiments import experiment_config
    from ats.plan import PlanConfig,PlanSegment
    from ats.simulation import Simulation
    from ats.complexes import ComplexProject
    r=ElevationRaster(np.zeros((6,6)),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=100.,dy=100.,latitude_deg=30.))
    land=Landscape(r,cell_size=.5,soil_profile=SoilProfile.qattara())
    c=workspace_config(land,experiment_config(Config(),drones=2,containers=2,landing_slots=2),[64.,64.],128)
    c.logistics.field_deployment_seconds=4.;c.plan=PlanConfig([PlanSegment([[20,40],[26,40]],width=2,height=-.2)])
    p=ComplexProject(Simulation(c),land);p.enable_shared();t=p.shared_site.core
    load=t.scoop_load(22,40,35.,radius=1.);t.deposit(100,100,load.mass,load=load)
    heights=t.height.copy();columns={k:a.copy() for k,a in t.soil.tiles.items()};ids=p.inventory_ids()
    handoffs=[];preflights=[];max_step=0.;start=time.perf_counter()
    for origin in ([96.,64.],[64.,64.]):
        if hasattr(land,'prefetch_window'):
            land.prefetch_window(origin,256,cell_size=.5)
            assert land._prefetch_runtime.wait(30.)['state']=='READY'
        at=time.perf_counter();move=p.move_window(origin);preflights.append(time.perf_counter()-at)
        original=move._finish
        def finish():
            begin=time.perf_counter()
            try:return original()
            finally:handoffs.append(time.perf_counter()-begin)
        move._finish=finish
        for _ in range(4000):
            if p.transition is None:break
            at=time.perf_counter();p.step();max_step=max(max_step,time.perf_counter()-at)
            assert not p.blocked,p.report()
        assert p.transition is None
    np.testing.assert_array_equal(p.shared_site.core.height,heights)
    for key,a in columns.items():np.testing.assert_array_equal(p.shared_site.core.soil.tiles[key],a)
    assert p.inventory_ids()==ids
    report=p.report();assert abs(report['mass_error_kg'])<1e-7 and abs(report['energy_error_j'])<.002
    return dict(wall_seconds=time.perf_counter()-start,model_seconds=p.time,preflight_seconds=preflights,
        physical_handoff_seconds=handoffs,max_step_seconds=max_step,ids_preserved=True,
        height_bitwise_equal=True,columns_bitwise_equal=True,mass_error_kg=report['mass_error_kg'],
        material_error_kg=report['shared_worksite']['max_material_mass_error_kg'],energy_error_j=report['energy_error_j'],
        physical_sha256=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--case',choices=('memory','transfer','all'),default='all')
    parser.add_argument('--tiles',type=int,default=1024);parser.add_argument('--cache-mib',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 128<=args.tiles<=4096 or args.tiles%2 or not 0<=args.cache_mib<=4096:parser.error('even tiles 128..4096; cache 0..4096 MiB')
    sys.path.insert(0,str(args.root.resolve()));os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    result=dict(python=platform.python_version(),platform=platform.system(),source_root=str(args.root.resolve()),
        memory=memory_control(args) if args.case in ('memory','all') else None,
        transfer=physical_control() if args.case in ('transfer','all') else None,
        note='Qt heartbeat is a small offscreen widget, not full map FPS. RSS is Linux only. Synchronous route planning is unchanged.')
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
