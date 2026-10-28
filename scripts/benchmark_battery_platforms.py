"""Bounded, reproducible shared-site benchmark of physical battery/PV carriers.

Run after installing the project: python scripts/benchmark_battery_platforms.py
--output battery-benchmark.json. This is a synthetic workload, not a user save.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.complexes import ComplexProject,ComplexTemplate


def build(count):
    c=experiment_config(with_map_size(Config(),512),panel_area=68)
    c.terrain.n=512;c.terrain.cell_size=1.;c.terrain.grab_radius=1.
    c.terrain.dump_rect=(20.,390.,470.,100.)
    c.plan=PlanConfig([PlanSegment([[20.,256.],[490.,256.]],width=4.,height=-.25)])
    c.base.x=50.;c.base.y=160.;c.base.layout={}
    s=Simulation(c)
    if s.state.value!='RUNNING':raise ValueError(s.reason)
    p=ComplexProject(s);p.enable_shared();template=ComplexTemplate.capture(c)
    for _ in range(count-1):p.add_shared(template)
    return p


def trial(count=14,steps=300,max_wall=180.):
    started=time.perf_counter();p=build(count)
    for _ in range(1000):
        if all(u.simulation.logistics.field_ready and u.simulation.logistics.battery_platforms.ready for u in p.units):break
        if not p.running or time.perf_counter()-started>max_wall:raise ValueError('Не завершено развёртывание; увеличьте лимит или исследуйте сценарий')
        p.step()
    else:raise ValueError('Не завершено развёртывание за 1000 модельных секунд')
    for _ in range(10):p.step()
    model_start=p.time;durations=[];started=time.perf_counter()
    for _ in range(steps):
        if not p.running:raise ValueError('Один из комплексов остановлен')
        if time.perf_counter()-started>max_wall:raise ValueError('Превышен лимит времени измерения')
        tick=time.perf_counter();p.step();durations.append(time.perf_counter()-tick)
    wall=time.perf_counter()-started;p.validate();r=p.report()
    result=dict(python=platform.python_version(),system=platform.system(),complexes=count,drones=count*4,
        map_side_m=512.,cell_m=1.,physical_dt_s=1.,solar_m2_per_complex=68.,soil='uniform',
        platforms=sum(u.simulation.base.layout_spec['platform_count'] for u in p.units),
        model_start_s=model_start,model_end_s=p.time,wall_s=wall,factor=(p.time-model_start)/wall,
        max_step_s=max(durations),median_step_s=statistics.median(durations),
        ground_moved_kg=sum(u.simulation.excavated_mass for u in p.units),
        mass_error_kg=r['mass_error_kg'],equipment_mass_error_kg=r['equipment_mass_error_kg'],energy_error_j=r['energy_error_j'])
    return p,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complexes',type=int,default=14);parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--max-wall-seconds',type=float,default=180.)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--save-state',type=Path)
    args=parser.parse_args()
    if not 1<=args.complexes<=20 or args.steps<1 or args.max_wall_seconds<=0:parser.error('complexes 1..20, steps и лимит времени > 0')
    p,result=trial(args.complexes,args.steps,args.max_wall_seconds)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.save_state:
        from ats.checkpoint import save_state
        save_state(args.save_state,p.active.simulation,complex_project=p)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
