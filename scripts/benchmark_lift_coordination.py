"""Reproduce a finite courier backlog around one requested Krot relocation.

Run from an installed checkout: python scripts/benchmark_lift_coordination.py
Set PYTHONPATH to a parent checkout to compare the same fixture on older code.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.excavator import ExcavatorState as ES
from ats.inventory import Location as L


def run(steps=5400):
    c=experiment_config(with_map_size(Config(),128),drones=4,containers=24,drone_spare_sets=7)
    c.terrain.cell_size=.5;c.terrain.n=256;c.terrain.dump_rect=(80.,80.,40.,40.)
    c.logistics.excavator_count=3;c.fleet.airspace_mode='shared_lanes'
    c.plan=PlanConfig([PlanSegment([[24.,60.],[104.,60.]],width=8.,height=-.3)])
    s=Simulation(c);w=s.logistics;e=w.excavators[-1]
    e.destination=e.position+np.array([0.,8.,0.])
    e.destination[2]=s.terrain.sample(*e.destination[:2])
    assert e.stable(e.destination,s.terrain)
    e.state=ES.WAITING_RELOCATION;e.supports=False
    for i,box in enumerate(w.containers):
        box.soil_kg=s.terrain.scoop(30.+i*2.,60.,60.)
        assert box.soil_kg>0
        w.inventory.move(box.id,L.FILLED_QUEUE,w.robot(box.excavator_serial).id,box.position,'FULL_READY')
    w.generate_jobs();ids=set(w.inventory.items);start=time.perf_counter();first_group=None;productive=0
    for _ in range(steps):
        s.step()
        if w.group is not None and first_group is None:first_group=s.time
        if first_group is None:productive+=len(w.direct_ids)
        if w.lift_count or s.state.value!='RUNNING':break
    r=s.report(full=False)
    return dict(model_seconds=s.time,wall_seconds=time.perf_counter()-start,state=s.state.value,reason=s.reason,
        drones=4,excavators=3,initial_filled_containers=24,drone_spare_modules=56,
        first_group_seconds=first_group,completed_lifts=w.lift_count,target_relocations=e.relocations,
        direct_digger_seconds_before_group=productive,
        queued_soil_jobs=sum(j.kind=='PICKUP_FULL_CONTAINER' and j.status=='QUEUED' for j in w.jobs),
        mass_error_kg=r['mass_error_kg'],energy_error_j=r['energy_error_j'],ids_preserved=set(w.inventory.items)==ids)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=5400)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.steps<1:parser.error('--steps must be positive')
    result=run(args.steps);text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    print(text,end='')
