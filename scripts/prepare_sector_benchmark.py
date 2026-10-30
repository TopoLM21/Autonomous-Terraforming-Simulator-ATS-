"""Prepare identical old-format coastal/14x4 inputs for sector-controller timing.

Use --root pointing to f250d84, then profile the snapshots with each checkout.
Construction and the 235/600-second warmups are outside measured intervals.
"""
import argparse
import json
from pathlib import Path
import sys


def prepare(root,directory):
    sys.path.insert(0,str(root.resolve()))
    from ats.config import Config
    from ats.experiments import experiment_config
    from ats.maps import with_map_size
    from ats.season_power import design_season,QATTARA_LATITUDE
    from ats.channel import ChannelProject
    from ats.landscape import Landscape,qattara_dem
    from ats.soil import SoilProfile
    from ats.simulation import Simulation
    from ats.checkpoint import save_state
    from ats.plan import PlanConfig,PlanSegment
    from ats.complexes import ComplexProject,ComplexTemplate
    folder=directory;folder.mkdir(parents=True,exist_ok=True)
    c=experiment_config(with_map_size(Config(),64));c.base.latitude_deg=QATTARA_LATITUDE
    c,_=design_season(c)
    r=qattara_dem();land=Landscape(r,soil_profile=SoilProfile.qattara())
    route=[list(map(float,r.to_metres(*p))) for p in [(28.3,31.2),(27.5,29.5)]]
    channel=ChannelProject.create(r,route,side_m=512)
    s=Simulation(channel.configuration(land,c));s.run(600)
    assert s.state.value=='RUNNING',s.reason
    save_state(folder/'coast45-v10.atsstate',s,landscape=land)
    print(json.dumps(dict(case='coast45',time=s.time,platforms=s.base.layout_spec['platform_count'])),flush=True)
    c=with_map_size(c,1024);c.terrain.cell_size=2.;c.terrain.n=512;c.terrain.grab_radius=2.
    c.base.x=100.;c.base.y=350.;c.trench.x=128.;c.trench.y=512.;c.trench.width=4.;c.trench.length=6.
    c.soil_profile=SoilProfile.qattara();c.terrain.dump_rect=(900.,20.,100.,100.)
    c.plan=PlanConfig([PlanSegment([[100.,512.],[924.,512.]],'dig',4.,-.3)],
     channel=dict(start=[100.,512.],tangent=[1.,0.],operations=dict(front_length_m=32.,sector_length_m=200.,
     setback_m=24.,bank_width_m=112.,base_chainage_m=0.,base_sites=[[float(x-100),float(x),y] for x in range(100,925,8) for y in (360.,664.)])))
    s=Simulation(c);assert s.state.value=='RUNNING',s.reason
    p=ComplexProject(s);p.enable_shared();template=ComplexTemplate.capture(c)
    for _ in range(13):p.add_shared(template)
    save_state(folder/'heavy14-start-v10.atsstate',p.active.simulation,complex_project=p)
    print(json.dumps(dict(case='heavy14',phase='start',platforms=sum(u.simulation.base.layout_spec['platform_count'] for u in p.units))),flush=True)
    for _ in range(235):
     p.step()
     assert not p.blocked,p.report()
    save_state(folder/'heavy14-work-v10.atsstate',p.active.simulation,complex_project=p)
    print(json.dumps(dict(case='heavy14',phase='work',time=p.time,ready=all(u.simulation.logistics.field_ready for u in p.units))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='Checkout f250d84 for v10 snapshots')
    parser.add_argument('--output',type=Path,required=True,help='Directory for generated .atsstate inputs')
    args=parser.parse_args();prepare(args.root,args.output)


if __name__=='__main__':main()
