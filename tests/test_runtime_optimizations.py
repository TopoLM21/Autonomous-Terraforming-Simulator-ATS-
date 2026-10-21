"""Dispatch reservations and cached dust transport retain their model meaning."""
import math
import numpy as np
from ats.config import Config
from ats.dust import Dust
from ats.experiments import experiment_config
from ats.inventory import Location as L
from ats.simulation import Simulation


def test_assigned_cargo_keeps_route_until_cancelled_without_searching_again(monkeypatch):
    s=Simulation(experiment_config(Config()));w=s.logistics
    container=w.containers[0]
    container.soil_kg=s.terrain.scoop(16,16,10)
    container.location=L.FILLED_QUEUE
    w.generate_jobs()
    job=next(j for j in w.jobs if container.id in j.resource_ids)
    destination=job.destination
    original=s.terrain.dump_location
    def unavailable(*args,**kwargs):raise AssertionError('Assigned cargo searched for another dump')
    monkeypatch.setattr(s.terrain,'dump_location',unavailable)
    for _ in range(100):w.generate_jobs()
    assert s.state.value=='RUNNING' and job.destination==destination
    assert len([j for j in w.jobs if container.id in j.resource_ids])==1
    w.cancel_job(job.id)
    monkeypatch.setattr(s.terrain,'dump_location',original)
    w.generate_jobs()
    assert container.reserved_by!=job.id
    assert len([j for j in w.jobs if container.id in j.resource_ids and j.status=='QUEUED'])==1


def test_fractional_dust_advection_matches_analytic_impulse_after_wind_and_step_changes():
    c=Config();c.dust.diffusion=0;c.dust.strength=0;c.dust.lifetime=30
    dust=Dust(c)
    # Repeated values exercise cache reuse; changed values require a new stencil.
    for dx,dy,dt in [(.25,.5,1),(.25,.5,1),(-.25,.25,1),(.8,-.3,.5),(.8,-.3,2)]:
        c.dust.wind_x=dx*dust.cell;c.dust.wind_y=dy*dust.cell
        dust.field[:]=0;dust.field[10,10]=1
        dust.tick(dt,(0,0,0),0,0,False)
        sx,sy=dx*dt,dy*dt
        x,y=math.floor(sx),math.floor(sy);fx,fy=sx-x,sy-y
        expected=np.zeros_like(dust.field)
        for ox,wx in [(0,1-fx),(1,fx)]:
            for oy,wy in [(0,1-fy),(1,fy)]:expected[10+y+oy,10+x+ox]=wx*wy*math.exp(-dt/30)
        np.testing.assert_allclose(dust.field,expected,atol=2e-15,rtol=0)
