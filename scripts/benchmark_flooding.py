"""Short reproducible global-reservoir controls; not a 2-D flow benchmark."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.years<=1000:parser.error('--years: 1..1000')
    from ats.landscape import Landscape,qattara_dem
    from ats.soil import SoilProfile
    from ats.channel import ChannelProject
    from ats.hydrology import create_experiment
    region=Landscape(qattara_dem(),soil_profile=SoilProfile.qattara())
    spec=json.loads((Path(__file__).resolve().parents[1]/'docs/ats2/qattara-channel.json').read_text())
    points=[[float(v) for v in region.raster.to_metres(*point)] for point in spec['points']]
    params={k:spec[k] for k in ('width_m','depth_m','shore_offset_m','angle_deg','side_m','spoil_height_m') if k in spec}
    project=ChannelProject.create(region.raster,points,**params)
    begin=time.perf_counter();model=create_experiment(region,project);prepare=time.perf_counter()-begin
    model.channel_open=True;before=region.report();times=[]
    for _ in range(args.years):
        begin=time.perf_counter();model.advance(365*86400.);times.append(time.perf_counter()-begin)
    model.validate();assert region.report()==before and not region.tiles and not region.soil_tiles
    empty=create_experiment(region,project);begin=time.perf_counter();empty.advance(1000*365*86400.);idle=time.perf_counter()-begin
    assert empty.calculations==0 and empty.volume_m3==0
    coarse=create_experiment(region,project);coarse.channel_open=True
    fine=deepcopy(coarse)
    coarse.advance(365*86400.);fine.advance(365*86400.,max_step=5400.)
    result=dict(python=sys.version.split()[0],source='bundled NOAA Qattara DEM',grid=list(region.raster.height.shape),
        curve_nodes=len(model.curve.bottom),preparation_seconds=prepare,years=args.years,
        per_year_seconds=times,median_year_seconds=statistics.median(times),active=model.report(),
        idle_1000_year_seconds=idle,idle_calculations=empty.calculations,
        refinement=dict(default_max_step_seconds=21600,reference_max_step_seconds=5400,
            year_volume_relative_difference=abs(coarse.volume_m3-fine.volume_m3)/max(1,fine.volume_m3),
            year_level_difference_m=abs(coarse.level-fine.level)),soil_unchanged=True,
        note='Conditional instant-channel experiment, single communicating basin, seasonal evaporation. No resolved water propagation or ground erosion. Timings are short headless controls, not desktop FPS or a 23-hour mission reproduction.')
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
