"""Count expensive equipment computations on an unchanged physical snapshot.

Use the same --state/--steps with the baseline --root and current checkout.
Timings belong in benchmark_runtime.py: wrappers here are for call counts only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.steps<1:parser.error('steps must be positive')
    sys.path.insert(0,str(args.root.resolve()))
    from ats import self_solar
    from ats.battery_platforms import BatteryPlatforms
    from ats.shared_site import SharedSite
    from ats.checkpoint import load_state
    session=load_state(args.state);sim=session.simulation;project=session.complex_project
    actor=project or sim;counts={};originals=[]
    def watch(owner,name,label):
        if not hasattr(owner,name):return
        original=getattr(owner,name);originals.append((owner,name,original));counts[label]=0
        def wrapped(*args,**kwargs):
            counts[label]+=1;return original(*args,**kwargs)
        setattr(owner,name,wrapped)
    watch(SharedSite,'_build_geometry','geometry_passes')
    watch(SharedSite,'_unit_geometry','geometry_units_rebuilt')
    watch(self_solar,'support_safe','support_ground_checks')
    watch(BatteryPlatforms,'slot_position','slot_queries')
    watch(BatteryPlatforms,'_slot_coordinates','slot_coordinates')
    watch(self_solar.SelfSolarArray,'panel_pose','panel_pose_calculations')
    start=actor.time
    try:
        for _ in range(args.steps):
            actor.step()
            if actor.time<=start or (not project.running if project else sim.state.value!='RUNNING'):break
    finally:
        for owner,name,original in originals:setattr(owner,name,original)
    report=dict(project.report(),reports=[u.simulation.report() for u in project.units]) if project else sim.report()
    units=len(project.units) if project else 1
    if 'geometry_units_rebuilt' not in counts:counts['geometry_units_rebuilt']=counts['geometry_passes']*units
    if 'slot_coordinates' not in counts:counts['slot_coordinates']=counts['slot_queries']
    result=dict(source_root=str(args.root.resolve()),snapshot=args.state.name,
        snapshot_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
        model_seconds=actor.time-start,counts=counts,state=report['state'],
        physical_sha256=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest(),
        mass_error_kg=report['mass_error_kg'],energy_error_j=report['energy_error_j'])
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
