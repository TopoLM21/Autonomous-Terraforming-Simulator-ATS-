"""Canal haul distances, moving work fronts and conservative local physics."""
from copy import deepcopy
import numpy as np
import pytest
from ats.channel import ChannelProject
from ats.config import Config
from ats.experiments import experiment_config
from ats.landscape import Landscape
from ats.simulation import Simulation
from tests.test_channel import coastal_dem


def canal(*, logistics=True, slots=1, panels=None,self_deploying=False):
    r=coastal_dem(80)
    project=ChannelProject.create(r,[[50,150],[215,150]],width_m=1,depth_m=.05,side_m=128)
    c=Config()
    # Lower assumed motor power makes this scheduling test finish without days
    # of charging; production physics is tested with the normal power elsewhere.
    c.drone.hover_power=1000
    if logistics:c=experiment_config(c,landing_slots=slots,panel_area=panels)
    if logistics and not self_deploying:c.logistics.self_solar=None
    return Simulation(project.configuration(Landscape(r),c))


def test_bilateral_spoil_is_parallel_and_clear_of_final_cut_and_base_starts_at_face():
    s=canal();t=s.terrain;f=t.front
    f.update()
    assert abs(float(f.chainage[int(s.base.position[1]/t.cell),int(s.base.position[0]/t.cell)])-f.start)<10
    assert np.any(t.dump_mask & (f.signed>0)) and np.any(t.dump_mask & (f.signed<0))
    assert not np.any(t.dump_mask & t.design_work)
    iy,ix=np.nonzero(t.design_work)
    bank=np.flatnonzero(t.dump_mask)
    for k in bank[::max(1,len(bank)//100)]:
        x,y=t.x.ravel()[k],t.y.ravel()[k]
        distance=np.hypot(t.axis[ix]-x,t.axis[iy]-y).min()
        assert distance>=f.settings['setback_m']-t.cell*1.5
    for d in s.drones:
        trip=s._choose_trip(d,unreserved=True)
        assert trip is not None
        along=f.chainage[int(trip.dig[1]/t.cell),int(trip.dig[0]/t.cell)]
        assert along<=f.position+f.settings['front_length_m']+t.cell
        assert np.linalg.norm(np.asarray(trip.dump)-trip.dig)<35
        assert abs(trip.dump[0]-trip.dig[0])<3  # horizontal canal, local banks


def test_unloading_is_local_and_uses_both_banks_without_backfilling_canal():
    s=canal();t=s.terrain;f=t.front
    original=t.height.copy();signs=set();mass=0.
    start=np.asarray(s.config.plan.segments[0].points[0])
    for offset in np.arange(3,28,1.):
        dig=start+[offset,0]
        taken=t.scoop(*dig,60)
        if taken<=0:continue
        point=t.dump_location(taken/s.config.terrain.density,near=dig)
        assert point is not None
        signs.add(np.sign(f.signed[int(point[1]/t.cell),int(point[0]/t.cell)]))
        before=t.height.copy();t.deposit(*point,taken)
        changed=np.abs(t.height-before)>1e-14
        assert np.all((t.x[changed]-point[0])**2+(t.y[changed]-point[1])**2<=4+1e-10)
        mass+=taken
    assert signs=={-1,1} and mass>100
    after_cut=t.height.copy();t.relax(100)
    # No soil from either bank returns to the designed canal.
    np.testing.assert_allclose(t.height[t.design_work],after_cut[t.design_work],atol=1e-12)
    assert abs(t.mass_change)<1e-7
    with pytest.raises(ValueError,match='Локальная'):
        t.deposit(*start,60)
    assert np.all(t.height[~(t.design_work|t.dump_mask)]==original[~(t.design_work|t.dump_mask)])


def test_completed_cut_is_still_forbidden_to_base_on_reopen():
    s=canal(logistics=False);t=s.terrain;c=deepcopy(s.config)
    c.initial_height=np.where(t.design_work,t.target,t.height).tolist()
    reopened=Simulation(c);q=reopened.terrain
    assert np.any(q.design_work) and not np.any(q.excavation)
    target=reopened.base.position[:2].copy();target[1]+=30
    assert not reopened.base.safe_path(target,q)


def test_local_avalanche_matches_full_disjoint_sweeps_and_conserves_mass():
    from ats.terrain import Terrain
    c=Config();c.reference_height=np.zeros((128,128)).tolist()
    c.georeference=dict(vertical_datum='MSL')
    a=Terrain(c);a.deposit(27,25,2000,radius=.25)
    b=deepcopy(a);b.config.reference_height=None
    before=a.mass
    a.relax(5);b.relax(5)
    np.testing.assert_allclose(a.height,b.height,atol=2e-14,rtol=0)
    assert a.mass==pytest.approx(before,abs=1e-7)
    assert b.mass==pytest.approx(before,abs=1e-7)


def test_base_uphill_motion_includes_potential_energy_and_follows_ground():
    from ats.terrain import Terrain
    from ats.solar_base import SolarBase
    from ats.drone import Drone
    c=Config();axis=(np.arange(128)+.5)*.25
    c.reference_height=np.broadcast_to(axis*.1,(128,128)).tolist()
    c.georeference=dict(vertical_datum='MSL')
    t=Terrain(c);base=SolarBase(c.base,np.array([8.125,2.,t.sample(8.125,2.)]))
    drone=Drone(c.drone,base.position.copy())
    mass=c.base.mass+drone.config.empty_mass
    target=np.array([13.125,2.]);distance=5.;climb=.5
    expected=base.motion_power([drone])*distance/c.base.speed+mass*9.80665*climb/.8
    assert base.route_energy(target,t,[drone])==pytest.approx(expected)
    base.destination=target
    for _ in range(30):
        base.tick(1.,12.,[drone],[False],terrain=t)
        assert base.position[2]==t.sample(*base.position[:2])
        if base.destination is None:break
    assert base.consumed==pytest.approx(expected)


def test_fleet_with_one_dock_finishes_and_base_advances_without_teleporting_or_free_energy():
    s=canal();w=s.logistics;t=s.terrain
    initial_base=s.base.position.copy();moved=0
    for _ in range(8000):
        old=s.base.position.copy()
        positions={id:np.asarray(item.position) for id,item in w.inventory.items.items() if item.holder_id=='base'}
        s.step()
        delta=s.base.position-old
        assert np.linalg.norm(delta[:2])<=s.config.base.speed+1e-9
        if np.linalg.norm(delta)>0:
            moved+=1
            assert all(d.grounded for d in s.drones)
            assert s.base.solar_power==0
            assert s.base.position[2]==t.sample(*s.base.position[:2])
            for id,position in positions.items():
                if w.inventory.items[id].holder_id=='base':
                    np.testing.assert_allclose(w.inventory.items[id].position,position+delta,atol=1e-12)
        if s.state.value!='RUNNING':break
    r=s.report()
    assert r['state']=='COMPLETED',r['reason']
    assert s.base.relocations>=2 and moved>100
    assert s.base.position[0]>initial_base[0]+25
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002
    assert not w.active and not w.direct_ids and not any(o.reserved_by for o in w.inventory.items.values())


def test_large_solar_field_folds_moves_and_redeploys_with_finite_service_energy():
    s=canal(panels=21);w=s.logistics
    assert w.field_rect is not None
    w.field_ready=True;w.field_elapsed=w.config.field_deployment_seconds
    # A controlled base-only relocation isolates panel handling from mining.
    site=next(p for p in s.terrain.front.settings['base_sites']
              if p[0]>20 and all(s.base.safe_path(p[1:],s.terrain,offset)
                    for offset in [(0,0),(0,6),(0,-6),(-3,6),(-3,-6)]))
    s.relocation_pending=np.asarray(site[1:]);s._pending_front_anchor=site[0]
    before=s.base.position.copy();consumed=s.base.consumed
    for _ in range(1000):
        s.step()
        if s.base.relocations and s.base.destination is None and w.field_ready:break
    assert s.base.relocations==1 and np.linalg.norm(s.base.position-before)>1
    assert w.field_ready and w.field_rect is not None
    assert s.base.consumed-consumed>=2*200*w.config.field_deployment_seconds
    assert abs(s.report()['energy_error_j'])<.002


def test_manual_landing_cancels_pending_base_move_and_drains_existing_soil_jobs():
    from ats.inventory import Location
    s=canal();w=s.logistics
    container=w.containers[0]
    point=np.asarray(s.config.plan.segments[0].points[0])+[6,0]
    container.soil_kg=s.terrain.scoop(*point,10.)
    container.location=Location.FILLED_QUEUE
    s.relocation_pending=s.base.position[:2]+[20,0]
    s._pending_front_anchor=20.
    s.request_landing()
    assert s.relocation_pending is None
    s.run(1000)
    assert container.soil_kg==0 and not w.active
    assert all(d.grounded for d in s.drones) and s.base.relocations==0
    assert abs(s.report()['mass_error_kg'])<1e-7
    assert abs(s.report()['energy_error_j'])<.002
