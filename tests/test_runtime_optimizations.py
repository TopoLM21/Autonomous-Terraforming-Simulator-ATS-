"""Dispatch reservations and cached dust transport retain their model meaning."""
import math
import numpy as np
from copy import deepcopy
import pytest
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


@pytest.mark.parametrize('wind',[(.3,-.7),(1.,0.),(1.-1e-15,-2.+1e-15),(200.,-200.)])
def test_rectangular_advection_matches_the_original_gather_bit_for_bit(wind):
    c=Config();c.dust.strength=0;c.dust.diffusion=0
    dust=Dust(c);c.dust.wind_x=wind[0]*dust.cell;c.dust.wind_y=wind[1]*dust.cell
    dust.field=np.random.default_rng(47).random(dust.field.shape)
    reference=dust.field.copy()
    for dt in (1.,1.,.5):
        x=dust.x-c.dust.wind_x*dt/dust.cell;y=dust.y-c.dust.wind_y*dt/dust.cell
        ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);fx=x-ix;fy=y-iy
        new=np.zeros_like(reference)
        for ox,oy,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
            xx,yy=ix+ox,iy+oy;inside=(xx>=0)&(xx<dust.n)&(yy>=0)&(yy<dust.n)
            new[inside]+=weight[inside]*reference[yy[inside],xx[inside]]
        reference=new*math.exp(-dt/c.dust.lifetime)
        dust.tick(dt,(0,0,0),0,0,False)
        np.testing.assert_array_equal(dust.field,reference)


def test_dump_cache_matches_fresh_search_after_cut_spoil_reclaim_and_avalanches():
    from tests.test_workfront import canal
    s=canal();a=s.terrain;b=deepcopy(a)
    start=np.asarray(s.config.plan.segments[0].points[0])
    for offset in range(3,15):
        point=start+[offset,0.]
        ma=a.scoop(*point,60.);mb=b.scoop(*point,60.);assert ma==mb
        b._dump_revision=None
        pa=a.dump_location(ma/s.config.terrain.density,near=point)
        pb=b.dump_location(mb/s.config.terrain.density,near=point)
        assert pa==pb and pa is not None
        a.deposit(*pa,ma);b.deposit(*pb,mb)
        a.relax(3);b.relax(3)
        b._dump_revision=None
        assert a.dump_location(.02,near=point)==b.dump_location(.02,near=point)
        if offset%3==0:
            assert a.scoop(*pa,10.,source='stockpile')==b.scoop(*pb,10.,source='stockpile')
            b._dump_revision=None
            assert a.dump_location(.02,near=point)==b.dump_location(.02,near=point)
        np.testing.assert_array_equal(a.height,b.height)
