"""Reproducible physical conventional worksite; editable model, not a quotation."""
import argparse
import json
import time
from pathlib import Path
from ats.config import Config
from ats.maps import with_map_size
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.conventional import ConventionalConfig
from ats.checkpoint import save_state


def run(steps=1800,trucks=2,soil=False):
    c=with_map_size(Config(),128);c.terrain.cell_size=.5;c.terrain.n=256
    c.terrain.dump_rect=(80.,80.,40.,40.);c.start_hour=6.
    c.plan=PlanConfig([PlanSegment([[40.,50.],[90.,50.]],width=8.,height=-.3)])
    if soil:
        from ats.soil import SoilProfile
        c.soil_profile=SoilProfile.qattara()
    s=Simulation.conventional_only(c)
    f=s.install_conventional(ConventionalConfig(base_x=12.,base_y=12.,work_rect=(35.,40.,60.,20.),truck_count=trucks))
    maximum=0.;started=time.perf_counter()
    for _ in range(steps):
        tick=time.perf_counter();s.step();maximum=max(maximum,time.perf_counter()-tick)
    wall=time.perf_counter()-started;f.validate()
    report=s.report();report['benchmark']=dict(steps=steps,wall_seconds=wall,max_step_seconds=maximum,
        scope='128 m synthetic site, 0.5 m cells, normal initial empty tanks and one 30 L/min pump; headless only')
    return s,report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=1800);p.add_argument('--trucks',type=int,default=2)
    p.add_argument('--soil',action='store_true');p.add_argument('--json',type=Path);p.add_argument('--save-state',type=Path)
    a=p.parse_args()
    if a.steps<0 or not 1<=a.trucks<=16:p.error('Nonnegative steps, 1..16 trucks required')
    s,r=run(a.steps,a.trucks,a.soil)
    if a.save_state:save_state(a.save_state,s)
    text=json.dumps(r,ensure_ascii=False,indent=2)+'\n'
    if a.json:a.json.write_text(text,encoding='utf-8')
    print(text,end='')
