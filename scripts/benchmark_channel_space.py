"""Bounded, cold coastal startup control; report construction time and peak RSS.

Compare --root checkouts in separate processes. Shore goals intentionally change
between versions; these startup reports are not bitwise-equivalent earthworks.
Use benchmark_runtime.py with the same old snapshot for unchanged-goal timing.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--width',type=float,default=20.)
    parser.add_argument('--depth',type=float,default=5.)
    parser.add_argument('--side',type=float,default=512.)
    parser.add_argument('--procedural',action='store_true',help='Include default deterministic fine terrain detail')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.root.resolve()))
    from ats.config import Config
    from ats.experiments import experiment_config
    from ats.maps import with_map_size
    from ats.season_power import design_season,QATTARA_LATITUDE
    from ats.channel import ChannelProject
    from ats.landscape import Landscape,qattara_dem
    from ats.soil import SoilProfile
    from ats.simulation import Simulation
    start=time.perf_counter()
    c=experiment_config(with_map_size(Config(),64));c.base.latitude_deg=QATTARA_LATITUDE
    c,_=design_season(c);c.terrain.cut_policy='natural';c.fleet.airspace_mode='shared_lanes'
    from ats.procedural import default_generation
    raster=qattara_dem();land=Landscape(raster,soil_profile=SoilProfile.qattara(),generation=default_generation() if args.procedural else None)
    route=[list(map(float,raster.to_metres(*p))) for p in [(28.3,31.2),(27.5,29.5)]]
    project=ChannelProject.create(raster,route,width_m=args.width,depth_m=args.depth,side_m=args.side)
    prepared=time.perf_counter();c=project.configuration(land,c);configured=time.perf_counter()
    s=Simulation(c);initialized=time.perf_counter();r=s.report()
    assert r['state']=='RUNNING',r.get('reason')
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002
    peak=None
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except ImportError:pass
    data=(c.plan.channel or {}).get('shore_data',{})
    result=dict(root=str(args.root.resolve()),python=platform.python_version(),os=platform.system(),procedural=args.procedural,
        preparation_seconds=prepared-start,configuration_seconds=configured-prepared,
        simulation_construction_seconds=initialized-configured,total_seconds=initialized-start,
        peak_process_rss_bytes=peak,project_version=project.version,shore_reference=getattr(project,'shore_reference','axis'),
        requested_side_m=args.side,actual_side_m=r['map_side_m'],cell_size_m=r['cell_size_m'],
        platforms=s.base.layout_spec['platform_count'],panels_m2=r['panel_area_m2'],state=r['state'],
        mass_error_kg=r['mass_error_kg'],energy_error_j=r['energy_error_j'],
        encoded_shore_bytes=sum(len(data.get(k,'')) for k in ('protected','cap')),
        window_fit=(c.plan.channel or {}).get('window_fit'),modified_tiles=land.report()['modified_tiles'])
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
