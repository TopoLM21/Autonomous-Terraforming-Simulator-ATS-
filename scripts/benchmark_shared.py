"""Compare real shared-map configurations with identical terrain and task."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.complexes import ComplexProject,ComplexTemplate
from ats.base_layout import design_continuous
from ats.power_design import PowerRequest


def trial(count,steps,continuous=False,cell=1.,layers=False,state_path=None,max_wall=180.):
    c=experiment_config(with_map_size(Config(),512));c.terrain.cell_size=cell;c.terrain.n=round(512/cell);c.terrain.grab_radius=1.
    if layers:
        from ats.soil import SoilProfile
        c.soil_profile=SoilProfile.qattara();c.initial_soil=dict(fine_cell=.25,tile_cells=64,origin_m=[0.,0.],tiles=[])
    c.terrain.dump_rect=(350.,50.,150.,150.);c.trench.x=128.;c.trench.y=256.
    c.base.x=100.;c.base.y=235.;c.base.latitude_deg=31.
    operations=dict(front_length_m=32.,setback_m=20.,bank_width_m=48.,base_chainage_m=0.,
        base_sites=[[float(x-128),float(x),243.5] for x in range(128,385,8)])
    c.plan=PlanConfig([PlanSegment([[128,256],[384,256]],'dig',4,-.3)],
        channel=dict(start=[128.,256.],tangent=[1.,0.],operations=operations))
    if continuous:c,_=design_continuous(c,PowerRequest(drone_utilization=.85,excavator_utilization=1.,energy_margin=.3))
    started=time.perf_counter();s=Simulation(c)
    if s.state.value=='BLOCKED':raise ValueError(s.reason)
    p=ComplexProject(s);p.enable_shared();template=ComplexTemplate.capture(c)
    for _ in range(count-1):p.add_shared(template)
    ready=time.perf_counter()
    for _ in range(steps):
        if not p.running or time.perf_counter()-ready>=max_wall:break
        p.step()
    elapsed=time.perf_counter()-ready;r=p.report()
    ground=p.shared_site.core
    if state_path:
        from ats.checkpoint import save_state
        save_state(state_path,p.active.simulation,complex_project=p)
    arrays=sum(v.nbytes for v in vars(ground).values() if hasattr(v,'nbytes'))
    return dict(complexes=count,continuous_template=continuous,initialization_seconds=ready-started,
        wall_seconds=elapsed,model_seconds=p.time,actual_speed=p.time/elapsed,physical_dt_seconds=1,cell_m=cell,layers=layers,
        wall_limit_reached=p.time<steps and p.running,
        terrain_arrays_bytes=arrays,height_arrays=len({id(u.simulation.terrain.height) for u in p.units}),
        inventory_items=sum(len(u.simulation.logistics.inventory.items) for u in p.units),
        teams_excavating=sum(u.simulation.excavated_mass>0 for u in p.units),
        ground_sha256=hashlib.sha256(ground.height.tobytes()).hexdigest(),report=r)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--counts',default='1,4,20');parser.add_argument('--continuous',action='store_true')
    parser.add_argument('--cell',type=float,choices=(.5,1.),default=1.)
    parser.add_argument('--layers',action='store_true');parser.add_argument('--save-state',type=Path)
    parser.add_argument('--max-wall-seconds',type=float,default=180.)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rows=[]
    if args.save_state and len(args.counts.split(','))!=1:parser.error('--save-state requires one count')
    if args.max_wall_seconds<=0:parser.error('--max-wall-seconds > 0')
    for n in map(int,args.counts.split(',')):
        if not 1<=n<=20 or args.steps<1:parser.error('counts 1..20; steps > 0')
        row=trial(n,args.steps,args.continuous,args.cell,args.layers,args.save_state,args.max_wall_seconds);rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='report'}),flush=True)
    args.output.write_text(json.dumps(dict(python=platform.python_version(),platform=platform.system(),cases=rows),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if any(r['report']['state']=='BLOCKED' for r in rows):raise SystemExit('A complex was blocked; inspect the report')


if __name__=='__main__':main()
