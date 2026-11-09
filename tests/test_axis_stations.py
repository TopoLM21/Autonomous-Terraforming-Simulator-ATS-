"""Axis stations use intact ground, real travel, and a common excavation face."""
from copy import deepcopy
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pytest
from ats.channel import ChannelProject
from ats.config import Config
from ats.simulation import Simulation
from ats.terrain import Terrain
from ats.canal_station import candidates,work_exclusion,slope_run
from ats.channel_site import equipment_spec
from ats.checkpoint import save_state,load_state
from ats.complexes import ComplexProject,ComplexTemplate
from tests.test_sectors import sector_config,finish_sector
from tests.test_channel import coastal_dem


def axis_config(**kwargs):
    c=sector_config(**kwargs)
    c.plan.channel['operations']['base_strategy']='axis_ahead'
    c.plan.channel['operations']['base_sites']=[p for p in c.plan.channel['operations']['base_sites'] if p[1]<512]
    t=Terrain(c);spec=equipment_spec(c)
    point=candidates(t,spec,t.front.global_sector_end)[0]
    _,c.base.x,c.base.y=point
    return c


def test_base_is_on_intact_future_axis_and_stationary_during_work():
    s=Simulation(axis_config());t=s.terrain;f=t.front
    assert s.state.value=='RUNNING',s.reason
    assert s.base.position[1]==pytest.approx(256.)
    iy,ix=int(s.base.position[1]/t.cell),int(s.base.position[0]/t.cell)
    assert t.design_work[iy,ix] and t.height[iy,ix]==t.reference[iy,ix]
    assert not work_exclusion(t)[iy,ix]
    assert f.chainage[iy,ix]>f.global_sector_end+slope_run(t)
    before=s.base.position.copy();s.run(40)
    np.testing.assert_array_equal(s.base.position,before)
    assert s.relocation_pending is None and s.base.relocations==0
    assert s.state.value=='RUNNING',s.reason
    assert all(d.trip is None or d.trip.dig[0]<250 for d in s.drones)


def test_axis_forecast_rejects_face_old_cut_and_terrain_change():
    s=Simulation(axis_config());t=s.terrain;f=t.front
    assert work_exclusion(t)[256,240]
    assert not s.base.safe_path([240.,256.],t)
    p=s.base.position[:2].copy();ix=int(p[0]);iy=int(p[1])
    assert not work_exclusion(t)[iy,ix]
    t.height[iy,ix]-=.01;t.revision+=1
    assert work_exclusion(t)[iy,ix]
    assert not s.base.safe_path(p,t)


def test_completed_sector_moves_physically_then_releases_previous_station(tmp_path):
    s=Simulation(axis_config());old=s.base.position.copy();finish_sector(s)
    s._update_workfront();f=s.terrain.front
    assert f.sector_id==1 and s.relocation_pending is not None
    assert not f.cut_enabled and not len(f.active_indices)
    assert not work_exclusion(s.terrain)[int(old[1]),int(old[0])]
    assert work_exclusion(s.terrain,future=True)[int(old[1]),int(old[0])]
    before=s.base.consumed
    path=tmp_path/'drive.atsstate';saved=False
    for _ in range(2000):
        previous=s.base.position.copy();s.step()
        assert np.linalg.norm(s.base.position[:2]-previous[:2])<=s.config.base.speed+1e-8
        assert s.state.value=='RUNNING',s.reason
        if s.base.destination is not None and not saved:
            save_state(path,s);saved=True
        if s.base.relocations and s.base.destination is None and f.stationed:break
    assert f.stationed and saved and s.base.relocations==1
    assert s.base.position[0]>old[0]+150 and s.base.consumed>before
    assert f.cut_mask((slice(int(old[1]),int(old[1])+1),slice(int(old[0]),int(old[0])+1))).item()
    q=load_state(path).simulation
    for _ in range(2000):
        q.step()
        if q.base.destination is None and q.terrain.front.stationed:break
    np.testing.assert_allclose(q.base.position,s.base.position,atol=1e-12)
    assert abs(s.report()['mass_error_kg'])<1e-7
    assert abs(s.report()['energy_error_j'])<.002
    assert q.report()==s.report()


def test_shared_teams_keep_common_sector_and_wait_for_leading_convoy():
    c=axis_config();p=ComplexProject(Simulation(c));p.enable_shared()
    p.add_shared(ComplexTemplate.capture(c));a,b=[u.simulation for u in p.units]
    assert a.terrain.front.sector_id==b.terrain.front.sector_id==0
    for s in (a,b):s.terrain.front.update()
    assert all(s.terrain.front.sector_remaining>0 for s in (a,b))
    finish_sector(a);a._update_workfront()
    assert a.relocation_pending is None and a.base_move_wait=='WAITING_SECTOR_TEAMS'
    finish_sector(b)
    for s in (a,b):s._update_workfront()
    assert p.shared_site.core.front.sector_id==1
    assert not any(s.terrain.front.cut_enabled for s in (a,b))
    ahead=max((a,b),key=lambda s:s.base.position[0]);behind=b if ahead is a else a
    assert ahead.relocation_pending is not None and behind.relocation_pending is None
    assert behind.base_move_wait=='WAITING_SECTOR_TEAMS'
    for _ in range(3000):
        p.step()
        assert not p.blocked,p.report()
        if p.shared_site.core.front.stationed:break
    assert p.shared_site.core.front.stationed
    assert all(s.terrain.front.stationed and s.terrain.front.cut_enabled for s in (a,b))
    assert a.base.relocations==b.base.relocations==1
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002


def test_legacy_channel_identity_and_strategy_are_not_silently_changed():
    r=coastal_dem(80)
    p=ChannelProject.create(r,[[50,150],[215,150]],width_m=1,depth_m=.05,side_m=128)
    old=p.to_data();old.pop('base_strategy');q=ChannelProject.from_data(old)
    assert p.identity==q.identity and q.base_strategy=='bank'
    axis=ChannelProject.create(r,[[50,150],[215,150]],width_m=1,depth_m=.05,side_m=128,base_strategy='axis_ahead')
    assert axis.version==3 and axis.identity!=p.identity
    assert ChannelProject.from_data(axis.to_data()).identity==axis.identity


@pytest.mark.parametrize('stage',['work','waiting'])
def test_axis_checkpoint_and_cancellation_preserve_support_guard(tmp_path,stage):
    s=Simulation(axis_config())
    if stage=='waiting':finish_sector(s);s._update_workfront();s.cancel_relocation()
    guard=s.terrain.front.support_guard_end
    path=tmp_path/'axis.atsstate';save_state(path,s);q=load_state(path).simulation
    assert q.terrain.front.axis and q.terrain.front.support_guard_end==guard
    np.testing.assert_array_equal(work_exclusion(s.terrain),work_exclusion(q.terrain))
    for _ in range(5):s.step();q.step()
    assert s.report()==q.report()


def axis_region():
    from ats.landscape import Landscape,ElevationRaster
    from ats.soil import SoilProfile
    r=ElevationRaster(np.zeros((9,13)),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=128.,dy=128.))
    land=Landscape(r,cell_size=1.,soil_profile=SoilProfile.qattara())
    channel=ChannelProject([[40.,300.],[1000.,300.]],[[50.,300.],[1000.,300.]],[40.,300.],
        width_m=2.,depth_m=.05,side_m=512.,version=3,base_strategy='axis_ahead')
    land.channel_project=channel.to_data()
    c=axis_config();c=channel.configuration(land,c,origin=[0.,0.])
    s=Simulation(c);s.request_landing();p=ComplexProject(s,land);p.enable_shared()
    return p


def finish_axis_material(s):
    t=s.terrain;f=t.front;f.update()
    for k in f.active_indices.copy():
        x,y=t.x.ravel()[k],t.y.ravel()[k]
        load=t.scoop_load(x,y,1e8,radius=t.cell*.49)
        if load.mass:t.deposit(*t.dump_location(load.volume,near=(x,y)),load.mass,load=load)
    s._terrain_active=False


def test_axis_regional_roundtrip_with_soil_and_save_in_transit(tmp_path):
    from tests.test_regional import advance
    p=axis_region();s=p.active.simulation;land=p.landscape;t=s.terrain
    finish_axis_material(s)
    original=t.height.copy();ids=p.inventory_ids();before=p.time;position=s.base.position.copy()
    p.move_window([128.,0.]);advance(p,lambda:p.transition.phase=='DRIVE')
    path=tmp_path/'regional-axis.atsstate';save_state(path,s,landscape=land,complex_project=p)
    q=load_state(path).complex_project
    advance(p,lambda:p.transition is None);advance(q,lambda:q.transition is None)
    assert p.report()==q.report()
    assert s.terrain.front.sector_id==1 and s.terrain.front.axis
    assert p.inventory_ids()==ids and p.time>before
    assert np.linalg.norm(np.asarray(s.config.georeference['origin_m'])+s.base.position[:2]-position[:2])>150
    p.move_window([0.,0.]);advance(p,lambda:p.transition is None)
    assert p.inventory_ids()==ids and s.config.georeference['origin_m']==[0.,0.]
    np.testing.assert_allclose(s.terrain.height,original,rtol=0,atol=1e-12)
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002
    assert p.report()['shared_worksite']['max_material_mass_error_kg']<1e-7


def test_boundary_staging_keeps_cut_disabled_until_final_station_after_handoff(tmp_path):
    from tests.test_regional import advance
    p=axis_region();s=p.active.simulation;s.landing_requested=False
    finish_axis_material(s)
    advance(p,lambda:s.base.relocations==1 and p.shared_site.core.front.stationed)
    finish_axis_material(s);s._update_workfront()
    assert s.state.value=='BLOCKED' and s.reason.startswith('BLOCKED_SECTOR_SITE')
    assert s.terrain.front.sector_id==2 and not s.terrain.front.cut_enabled
    s.request_landing();p.move_window([256.,0.])
    advance(p,lambda:p.transition is None)
    assert s.config.georeference['origin_m']==[256.,0.]
    assert not p.shared_site.core.front.stationed and not s.terrain.front.cut_enabled
    assert p.shared_site.core.front.support_guard_end==400.
    # A save on the staging apron must not promote it to a working station.
    path=tmp_path/'staging.atsstate';save_state(path,s,landscape=p.landscape,complex_project=p)
    q=load_state(path).complex_project
    for project in (p,q):
        project.active.simulation.landing_requested=False
        advance(project,lambda:project.shared_site.core.front.stationed)
        actor=project.active.simulation
        assert actor.terrain.front.sector_id==2 and actor.terrain.front.cut_enabled
        assert actor.base.position[0]+256>650
        assert abs(project.report()['mass_error_kg'])<1e-7 and abs(project.report()['energy_error_j'])<.002
    assert p.report()==q.report()


def test_unreachable_next_station_reports_space_instead_of_undercutting():
    c=axis_config(length=470.);s=Simulation(c)
    finish_sector(s);s._update_workfront()
    for _ in range(1500):
        s.step()
        if s.terrain.front.stationed and s.base.relocations:break
    finish_sector(s);s._update_workfront()
    assert s.terrain.front.sector_id==2
    # The last sector extends beyond the map; an off-map station is forbidden.
    assert s.state.value=='BLOCKED' and s.reason.startswith('BLOCKED_SECTOR_SITE')
    assert not s.terrain.front.cut_enabled and s.base.destination is None


def test_hairpin_near_future_axis_remains_excluded_by_spatial_face_distance():
    c=sector_config();c.terrain.floor=-20.
    c.plan.channel['operations']['base_strategy']='axis_ahead'
    c.plan.segments[0].points=[[50.,256.],[250.,256.],[250.,272.],[60.,272.]]
    c.plan.segments[0].height=-10.
    t=Terrain(c)
    # This is far ahead in chainage, but lies next to the current cut in space.
    assert t.front.chainage[272,150]>300
    assert work_exclusion(t)[272,150]


def test_natural_layered_end_slump_does_not_freeze_soil_or_reach_station():
    from ats.soil import SoilProfile
    c=axis_config();c.soil_profile=SoilProfile.qattara();c.soil_profile.thicknesses=[.02,.05]
    c.terrain.cut_policy='natural';c.plan.segments[0].height=-2.
    t=Terrain(c);spec=equipment_spec(c);_,c.base.x,c.base.y=candidates(t,spec,200.)[0]
    s=Simulation(c);t=s.terrain
    original=t.height.copy();mass=t.mass
    load=t.scoop_load(249.5,256.5,1e8,radius=.49)
    assert load.mass>0
    t.deposit(*t.dump_location(load.volume,near=(249.5,256.5)),load.mass,load=load)
    assert t.relax(100)>0
    assert abs(t.mass-mass)<1e-7 and abs(s.report()['mass_error_kg'])<1e-7
    ix,iy=(s.base.position[:2]/t.cell).astype(int)
    np.testing.assert_array_equal(t.height[iy-2:iy+3,ix-2:ix+3],original[iy-2:iy+3,ix-2:ix+3])
    assert t.protected is None or not t.protected[iy,ix]  # No frozen supports.


def test_completed_sector_releases_unstarted_krot_relocation_only():
    from ats.excavator import ExcavatorState as ES
    c=axis_config();c.logistics.excavator_enabled=True;s=Simulation(c);e=s.logistics.excavator
    finish_sector(s);e.state=ES.WAITING_RELOCATION;e.target=np.array([180.,256.,-.001])
    s.logistics.try_lift();assert s.logistics.group is None and not s.logistics.tool_targets
    e.tick(s.logistics)
    assert e.state==ES.IDLE and e.target is None and e.reason=='SECTOR_COMPLETE'
    assert s._sector_drained()
