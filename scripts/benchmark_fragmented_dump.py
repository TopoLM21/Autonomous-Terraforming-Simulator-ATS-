"""Artificial late-bank condition: total free room, no footprint fits one load.

Does not load or alter a user save. This isolates a confirmed slow operation;
it is not a reproduction of the user's unspecified 23-hour mission.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from ats.config import Config
from ats.maps import with_map_size
from ats.terrain import Terrain


def probe():
    c=with_map_size(Config(),512);c.terrain.dump_rect=(150,80,250,300)
    seed=Terrain(c)
    seed.height[seed.dump_mask]=seed.dump_cap[seed.dump_mask]-1e-5
    c.initial_height=seed.height.tolist();del seed
    t=Terrain(c);t.front=SimpleNamespace(signed=t.y-200)
    before=t.height.copy();checks=[0];original=t.local_dump_space
    def counted(*point):
        checks[0]+=1;return original(*point)
    t.local_dump_space=counted
    times=[]
    for _ in range(3):
        started=time.perf_counter();point=t.dump_location(.03,near=(40,40));times.append(time.perf_counter()-started)
        assert point is None
    assert checks[0]<20 and np.array_equal(t.height,before) and t.mass_change==0
    return dict(scenario='Prescribed fragmented bank, not a user snapshot',grid_cells=c.terrain.n,
        map_side_m=512,bank_rectangle_m=list(c.terrain.dump_rect),free_height_m=1e-5,
        requested_volume_m3=.03,total_free_m3=t.dump_space,searches=3,
        scalar_local_capacity_checks=checks[0],search_wall_seconds=times,max_search_wall_seconds=max(times),
        terrain_unchanged=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=probe();args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
