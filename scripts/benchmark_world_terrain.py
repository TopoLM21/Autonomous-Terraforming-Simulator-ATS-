"""Bounded procedural/world-reader control; Qt offscreen, not user-PC FPS.

Compare --root checkouts, and separately --procedural. The latter deliberately
changes baseline heights; it must not be called bitwise-equivalent physics.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def runtime_fixtures(folder):
    """Create the exact 14×4 DEM controls under the selected engine checkout."""
    import numpy as np
    from ats.landscape import Landscape,ElevationRaster,workspace_config
    from ats.soil import SoilProfile
    from ats.config import Config
    from ats.experiments import experiment_config
    from ats.plan import PlanConfig,PlanSegment
    from ats.complexes import ComplexProject,ComplexTemplate
    from ats.simulation import Simulation
    from ats.checkpoint import save_state
    folder.mkdir(parents=True,exist_ok=True)
    land=Landscape(ElevationRaster(np.zeros((6,6)),dict(crs='LOCAL_METRES',units='metres',
        vertical_datum='MSL',x0=0.,y0=0.,dx=1000.,dy=1000.)),cell_size=1.,soil_profile=SoilProfile.qattara())
    c=workspace_config(land,experiment_config(Config(),drones=4,containers=2),[1024.,1024.],512.)
    c.logistics.field_deployment_seconds=2.
    c.plan=PlanConfig([PlanSegment([[20,100],[450,100]],width=4,height=-.5)])
    p=ComplexProject(Simulation(c),land);p.enable_shared()
    for _ in range(13):p.add_shared(ComplexTemplate.capture(c))
    save_state(folder/'world14-start.atsstate',p.active.simulation,land,complex_project=p)
    for _ in range(45):p.step()
    if not p.running:raise RuntimeError(p.report())
    save_state(folder/'world14-work.atsstate',p.active.simulation,land,complex_project=p)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--procedural',action='store_true')
    parser.add_argument('--runtime-fixtures',type=Path,help='Create 14 complexes × 4 drones at model times 0 and 45 s')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();sys.path.insert(0,str(args.root.resolve()))
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    if args.runtime_fixtures:
        if args.procedural:parser.error('Runtime controls use the unchanged baseline for paired comparisons')
        runtime_fixtures(args.runtime_fixtures);return
    if args.output is None:parser.error('--output or --runtime-fixtures required')
    import numpy as np
    from ats.landscape import Landscape,ElevationRaster
    from ats.soil import SoilProfile
    from benchmark_streaming import heartbeat,rss,window_digest
    kwargs=dict(soil_profile=SoilProfile.qattara(),cache_bytes=2*1024**2)
    if args.procedural:
        from ats.procedural import default_generation
        kwargs['generation']=default_generation()
    land=Landscape(ElevationRaster(np.full((5,5),10.),dict(crs='LOCAL_METRES',units='metres',
        vertical_datum='MSL',x0=0.,y0=0.,dx=2000.,dy=2000.)),**kwargs)
    for i in range(128):
        key=(i%16,i//16);h=np.full((64,64),.125 if i%2 else -.125)
        s=np.zeros((4,64,64));s[1 if i%2 else 0]=.125
        land.tiles[key]=h;land.soil_tiles[key]=s
    keys=[[128.,128.,1024,.5],[384.,128.,1024,.5],[128.,384.,1024,.5],[384.,384.,1024,.5]]
    baseline_rss=rss();start=time.perf_counter()
    def read(key,prepared=None):
        x,y,n,cell=key
        return (*land.read_window([x,y],n,cell_size=cell,prepared=prepared),land.soil_window([x,y],n,cell,prepared=prepared))
    cold=read(keys[0]);cold_seconds=time.perf_counter()-start;digest=window_digest(cold);del cold
    def request():
        if hasattr(land,'prefetch_windows'):land.prefetch_windows(keys)
        else:land.prefetch_window(keys[0][:2],1024,cell_size=.5)
    wait=heartbeat(request,lambda:land.cache_report()['prefetch']['state'] in ('READY','ERROR'),timeout=90.)
    report=land.cache_report();assert report['prefetch']['state']=='READY',report
    ready=land.prepared_window(keys[0][:2],1024,.5);assert ready is not None
    start=time.perf_counter();warm=read(keys[0],ready);warm_seconds=time.perf_counter()-start
    assert window_digest(warm)==digest;del warm
    result=dict(root=str(args.root),procedural=args.procedural,windows_requested=len(keys) if hasattr(land,'prefetch_windows') else 1,
        cold_read_seconds=cold_seconds,warm_read_seconds=warm_seconds,ui=wait,cache=report,
        before_rss_bytes=baseline_rss,after_rss_bytes=rss(),peak_rss_bytes=rss(True),window_digest=digest,
        volume_error_m3=land.report()['soil_delta_m3'])
    assert abs(result['volume_error_m3'])<1e-9
    land.cancel_prefetch();args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
