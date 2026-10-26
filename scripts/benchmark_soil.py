"""Whole-engine comparison of uniform soil and conditional layers, without Qt."""
import argparse
import gc
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter
import numpy as np
from ats.scenarios import scenario
from ats.maps import with_map_size
from ats.experiments import experiment_config
from ats.simulation import Simulation
from ats.soil import SoilProfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--warmup',type=int,default=150)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--sides',type=float,nargs='+',default=[32.,160.])
    parser.add_argument('--output',default='soil-performance.json')
    args=parser.parse_args()
    if args.steps<=0 or args.warmup<0 or args.repeats<=0:parser.error('Неверное число шагов или повторов')
    records=[]
    for side in args.sides:
        for repeat in range(args.repeats):
            for layers in ((False,True) if repeat%2==0 else (True,False)):
                c=experiment_config(with_map_size(scenario('medium'),side))
                if layers:c.soil_profile=SoilProfile.qattara()
                s=Simulation(c);s.run(args.warmup);gc.collect()
                model_start=s.time;start=perf_counter();s.run(args.steps);wall=perf_counter()-start;r=s.report()
                elapsed=s.time-model_start
                row=dict(map_side_m=side,grid_n=c.terrain.n,mode='three_layers' if layers else 'uniform',repeat=repeat+1,
                    model_steps=int(elapsed),wall_seconds=wall,speed=elapsed/wall,mass_error_kg=r['mass_error_kg'],
                    energy_error_j=r['energy_error_j'],material_mass_error_kg=r.get('soil',{}).get('max_material_mass_error_kg'),
                    soil_bytes=r.get('soil',{}).get('storage_bytes',0),state=r['state'],excavated_m3=r['excavated_m3'])
                if abs(r['mass_error_kg'])>1e-7 or abs(r['energy_error_j'])>1e-4:raise RuntimeError('Нарушен физический баланс')
                records.append(row);print(json.dumps(row,ensure_ascii=False),flush=True);del s;gc.collect()
    summary=[]
    for side in args.sides:
        old=median(r['wall_seconds'] for r in records if r['map_side_m']==side and r['mode']=='uniform')
        new=median(r['wall_seconds'] for r in records if r['map_side_m']==side and r['mode']=='three_layers')
        summary.append(dict(map_side_m=side,uniform_median_seconds=old,three_layers_median_seconds=new,
                            wall_time_change_percent=(new/old-1)*100))
    result=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,warmup_steps=args.warmup,
        methodology='One process, alternating order; medium geometry, 4 drones, 20 m2 panels, fixed 1 s steps. Final report included, no profiler/GUI. Standard short runs stay in top 2 m sand; near-surface material properties match. Scheduling may diverge on floating-point ties; this compares scenarios, not identical operation traces.',records=records,summary=summary)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
