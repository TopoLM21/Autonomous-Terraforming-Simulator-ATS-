"""A natural late worksite must resume bulk work instead of tiny-haul churn."""
from pathlib import Path
import numpy as np
from ats.checkpoint import load_state
from tests.test_conventional import balanced


def test_existing_midstage_ramp_digs_bulk_without_tiny_haul_cycle():
    path=Path(__file__).parent/'fixtures/conventional-v23/churn.atsstate'
    s=load_state(path).simulation;f=s.conventional;e=f.excavator
    before=f.excavated_m3;distance=e.distance_m;fuel=f.fuel_used_l
    departures=[];previous={m.id:m.state for m in f.machines}
    for _ in range(300):
        s.step()
        for truck in f.machines[1:]:
            if truck.state=='DRIVING' and truck.after_drive=='UNLOADING' and previous[truck.id]!='DRIVING':
                departures.append(truck.cargo_kg)
            previous[truck.id]=truck.state
        for i,a in enumerate(f.machines):
            for b in f.machines[i+1:]:
                assert np.linalg.norm(a.position[:2]-b.position[:2])>=a.radius+b.radius-.01
    gained=f.excavated_m3-before
    assert gained>1.,(gained,e.state,e.reason)
    assert e.distance_m-distance<gained*5.
    assert f.fuel_used_l>fuel
    assert all(cargo>1000. for cargo in departures),departures
    assert f.carried_mass_kg>1000. or f.delivered_m3>21.
    balanced(s,f)
