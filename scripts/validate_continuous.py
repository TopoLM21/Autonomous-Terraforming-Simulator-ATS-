"""Operate real dig/fill cycles across successive nights without resetting gear.

This commissioning workload alternates two earthwork plans once every vehicle
has landed and all soil is deposited. It exercises the ordinary autonomous
dispatcher, charging controllers, excavator and mass/energy ledgers.
"""
import argparse
import json
from pathlib import Path
import time
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.power_design import PowerRequest
from ats.base_layout import design_continuous


def run(steps,latitude=31.,day=172,checkpoint=None,resume=None,cell=.5,max_wall=None):
    c=experiment_config(with_map_size(Config(),128));c.base.latitude_deg=latitude;c.base.start_day_of_year=day;c.start_hour=6.
    c.terrain.cell_size=cell;c.terrain.n=round(128/cell);c.terrain.grab_radius=max(c.terrain.grab_radius,cell)
    c.trench.x=52.;c.trench.y=64.
    c.plan=PlanConfig([PlanSegment([[48,64],[56,64]],'dig',2,-.3)])
    c,design=design_continuous(c,PowerRequest(drone_utilization=.85,excavator_utilization=1.,energy_margin=.3))
    s=Simulation(c);cycles=0;samples=[];minimum=s.base.battery/c.base.battery
    peak_error=0.;peak_mass=0.;mode='dig';night_idle=0.;night_excavated=0.
    if resume:
        from ats.checkpoint import load_state
        session=load_state(resume);s=session.simulation;c=s.config
        state=session.ui;design=state['design'];cycles=state['cycles'];mode=state['mode'];samples=state['samples']
        minimum=state['minimum'];peak_error=state['peak_error'];peak_mass=state['peak_mass'];night_idle=state['night_idle'];night_excavated=state['night_excavated']
    start=time.perf_counter();start_time=s.time
    def save_progress():
        if checkpoint:
            from ats.checkpoint import save_state
            save_state(checkpoint,s,ui=dict(design=design,cycles=cycles,mode=mode,samples=samples,minimum=minimum,
                peak_error=peak_error,peak_mass=peak_mass,night_idle=night_idle,night_excavated=night_excavated))
    while s.time<steps and (max_wall is None or time.perf_counter()-start<max_wall):
        if s.state.value=='BLOCKED':break
        if s.state.value=='COMPLETED':
            next_mode='fill' if mode=='dig' else 'dig'
            if s.logistics.excavator:s.logistics.excavator.active_container=None
            try:s.apply_plan(PlanConfig([PlanSegment([[48,64],[56,64]],next_mode,2,0. if next_mode=='fill' else -.3)]))
            except ValueError:
                save_progress();print(json.dumps(dict(time=s.time,mode=mode,base_destination=None if s.base.destination is None else s.base.destination.tolist(),
                    drones=[dict(state=d.state.value,cargo=d.cargo,grounded=d.grounded) for d in s.drones])),flush=True);raise
            mode=next_mode;cycles+=1
        before_mass=s.excavated_mass;s.step();minimum=min(minimum,s.base.battery/c.base.battery)
        if s.base.solar_power<1:night_excavated+=s.excavated_mass-before_mass
        if s.base.solar_power<1 and all(d.grounded for d in s.drones):night_idle+=1.
        if s.time%3600==0 or s.time>=steps:
            r=s.report(full=False);peak_error=max(peak_error,abs(r['energy_error_j']));peak_mass=max(peak_mass,abs(r['mass_error_kg']))
            samples.append(dict(time=s.time,hour=s.hour,base_soc=s.base.battery/c.base.battery,
                excavated_m3=r['excavated_m3'],consumed_kwh=r['consumed_kwh'],solar_generated_kwh=r['solar_generated_kwh']))
            print(json.dumps(samples[-1]),flush=True)
            save_progress()
    save_progress()
    report=s.report(full=False);elapsed=time.perf_counter()-start
    return dict(design=design,steps=s.time,measurement_start_seconds=start_time,wall_seconds=elapsed,actual_speed=(s.time-start_time)/elapsed,plan_changes=cycles,
        minimum_base_soc=minimum,night_grounded_seconds=night_idle,night_excavated_kg=night_excavated,max_mass_error_kg=max(peak_mass,abs(report['mass_error_kg'])),
        max_energy_error_j=max(peak_error,abs(report['energy_error_j'])),samples=samples,report=report,
        cell_m=s.terrain.cell,wall_limit_reached=s.time<steps and s.state.value=='RUNNING')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=172800)
    parser.add_argument('--latitude',type=float,default=31.);parser.add_argument('--day',type=int,default=172)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--resume',type=Path)
    parser.add_argument('--cell',type=float,choices=(.5,1.),default=.5)
    parser.add_argument('--max-wall-seconds',type=float,help='Bound this run and checkpoint progress for continuation')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.steps<1:parser.error('steps > 0')
    if args.max_wall_seconds is not None and args.max_wall_seconds<=0:parser.error('max wall > 0')
    r=run(args.steps,args.latitude,args.day,args.checkpoint,args.resume,args.cell,args.max_wall_seconds);args.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:r[k] for k in ('steps','wall_seconds','actual_speed','plan_changes','minimum_base_soc','max_mass_error_kg','max_energy_error_j')}))
    if r['report']['state']=='BLOCKED':raise SystemExit(r['report']['reason'])


if __name__=='__main__':main()
