"""Repeatable headless canal timing, including initialization and haul lengths."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path
import numpy as np
from ats.channel import ChannelProject
from ats.config import Config
from ats.experiments import experiment_config
from ats.landscape import Landscape,qattara_dem
from ats.simulation import Simulation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=180)
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.steps<1 or not 1<=args.trials<=100:parser.error('steps >= 1; trials: 1..100')
    raster=qattara_dem()
    points=[list(raster.to_metres(*p)) for p in [(28.3,31.2),(27.5,29.5)]]
    project=ChannelProject.create(raster,points)
    rows=[]
    for _ in range(args.trials):
        gc.collect();begin=time.perf_counter()
        config=project.configuration(Landscape(raster),experiment_config(Config()))
        prepared=time.perf_counter();sim=Simulation(config);started=time.perf_counter()
        report=sim.run(args.steps);end=time.perf_counter()
        lengths=[float(np.linalg.norm(np.asarray(j.destination[:2])-j.origin[:2]))
                 for j in sim.logistics.jobs if j.kind=='EXCAVATE_DIRECT']
        row=dict(layout_seconds=prepared-begin,initialization_seconds=started-prepared,
            run_seconds=end-started,actual_speed=report['time_seconds']/(end-started),
            mean_planned_loaded_haul_m=statistics.mean(lengths) if lengths else 0.,
            max_planned_loaded_haul_m=max(lengths,default=0.))
        row.update({key:report[key] for key in ('state','trips','excavated_m3','mass_error_kg','energy_error_j')})
        rows.append(row);del sim,config;gc.collect()
    median={key:statistics.median(row[key] for row in rows) for key in rows[0] if key!='state'}
    median['state']=rows[-1]['state']
    result=dict(trials=rows,median=median)
    if args.output:args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(median,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
