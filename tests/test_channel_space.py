"""Crest setback, complete equipment envelope and initial window sizing."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from ats.channel import ChannelProject,marine_window
from ats.channel_site import SiteError,slope_runs,equipment_fits
from ats.config import Config
from ats.landscape import ElevationRaster,Landscape
from ats.plan import target_surface
from ats.shore import spread
from ats.simulation import Simulation
from tests.test_channel import coastal_dem


def oblique():
    h=np.full((601,601),2.);h[:,:80]=-2.
    y,x=np.indices(h.shape);h[(x-470)**2+(y-400)**2<80**2]=-2.
    r=ElevationRaster(h,dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=1.,dy=1.))
    p=ChannelProject.create(r,[[40,100],[120,400],[470,400]],width_m=50,depth_m=10,side_m=512)
    c=Config();c.terrain.cut_policy='natural';c.terrain.n=600;c.terrain.cell_size=1.
    x,y=np.meshgrid(np.arange(600)+.5,np.arange(600)+.5,sparse=True);reference=r.sample(x,y)
    t=SimpleNamespace(height=reference.copy(),reference=reference,x=x,y=y,slope=np.tan(np.deg2rad(35)),config=c,soil=None)
    return r,p,t


@pytest.mark.parametrize('shape',[(1,8),(8,1),(5,6)])
def test_linear_spread_matches_brute_force_including_different_crest_heights(shape):
    rng=np.random.default_rng(12);v=rng.normal(size=shape)*5;y,x=np.indices(shape)
    expected=np.full(shape,-np.inf)
    for yy,xx in np.ndindex(shape):expected=np.maximum(expected,v[yy,xx]-.7*np.maximum(abs(y-yy),abs(x-xx)))
    np.testing.assert_allclose(spread(v,.7),expected,rtol=0,atol=1e-12)


def test_oblique_channel_keeps_the_entire_cut_away_from_the_sea():
    r,p,t=oblique();plan=p.plan([0,0],side_m=600)
    plan.channel.update(marine=marine_window(r,[0,0],600),cell_size=1.)
    original=t.height.copy();target,labels=target_surface(t,plan);cut=labels>0
    assert np.any(cut)
    # The exact vertical synthetic coastline is x=79.5 m. Check the full
    # footprint of changed cells, not just the centre-line or dry-cell centres.
    x=np.broadcast_to(t.x,t.height.shape)
    assert float(x[cut].min()-.5-79.5)>=p.shore_offset_m
    assert np.max(np.abs(np.diff(target,axis=1))[...,90:120])<=2.+1e-9
    np.testing.assert_array_equal(t.height,original)
    legacy=deepcopy(plan);legacy.channel.pop('shore_reference');legacy.channel.pop('shore_offset_m')
    _,old_labels=target_surface(t,legacy)
    assert float(x[old_labels>0].min()-79.5)<p.shore_offset_m


def test_buffer_cap_anchors_to_dry_height_instead_of_dropping_from_crest_to_sea_level():
    from ats.shore import crest_bank
    h=np.full((20,40),10.);h[:,:3]=-1.;t=SimpleNamespace(reference=h,height=h.copy(),slope=1.)
    protected,cap=crest_bank(t,h<0,20.,5.,1.)
    last=int(np.flatnonzero(protected[10])[-1])
    assert cap[10,last+1]==pytest.approx(10.-.8)
    assert np.max(np.abs(np.diff(cap[10,last:])))<=.8+1e-12


def test_old_project_hash_and_geometry_remain_unchanged_without_new_metadata():
    r=coastal_dem();old=ChannelProject.create(r,[[50,150],[150,150]],width_m=1,depth_m=.05,side_m=32,version=1)
    assert old.shore_reference=='axis' and old.identity=='fdede5948389e20c'
    data=old.to_data();data.pop('shore_reference');p=ChannelProject.from_data(data)
    assert p.identity==old.identity and p.shore_reference=='axis'
    assert 'shore_reference' not in p.plan([0,0]).channel
    c=p.configuration(Landscape(r),Config())
    assert c.terrain.n*c.terrain.cell_size==32


def test_new_mission_enlarges_workspace_without_moving_soil_or_changing_project_identity():
    land=Landscape(coastal_dem(gap=100));p=ChannelProject.create(land.raster,[[50,150],[250,150]],width_m=20,depth_m=5,side_m=32)
    identity=p.identity;before=deepcopy(p.to_data());c=p.configuration(land,Config());s=Simulation(c)
    assert c.terrain.n*c.terrain.cell_size==128 and c.georeference['channel']['auto_window']
    assert p.identity==identity and p.side_m==32 and p.active_section==before['active_section']
    assert not land.tiles and s.terrain.mass_change==0
    fit=c.plan.channel['window_fit']
    assert fit['cross_section']['width_m']>p.width_m and fit['cross_section']['fits']
    assert fit['spoil']['sufficient'] and fit['spoil']['available_free_m3']>=fit['spoil']['required_free_m3']
    assert s.state.value=='RUNNING' and abs(s.report()['mass_error_kg'])<1e-7


def test_an_explicit_window_is_not_silently_enlarged_and_shortage_is_explained():
    land=Landscape(coastal_dem(gap=100));p=ChannelProject.create(land.raster,[[50,150],[250,150]],width_m=20,depth_m=5,side_m=32)
    origin=p.origin(land.raster)
    with pytest.raises(SiteError):p.configuration(land,Config(),origin=origin,side_m=32)
    assert p.side_m==32 and not land.tiles


def test_finite_spoil_capacity_is_checked_before_starting_a_deep_cut():
    land=Landscape(coastal_dem(gap=100));p=ChannelProject.create(land.raster,[[50,150],[250,150]],width_m=60,depth_m=10,side_m=64)
    with pytest.raises(SiteError,match='вместимости отвала.*нужно.*доступно'):p.configuration(land,Config())
    assert not land.tiles and not p.completed


def test_layered_sizing_distinguishes_native_profile_loose_forecast_and_explicit_angle():
    from ats.soil import SoilProfile
    c=Config();c.soil_profile=SoilProfile.qattara();depth=np.array([20.])
    native=slope_runs(c,c.soil_profile,depth)
    c.terrain.cut_policy='natural';loose=slope_runs(c,c.soil_profile,depth)
    assert native<loose
    c.trench.angle_deg=20.;assert slope_runs(c,c.soil_profile,depth)==pytest.approx(depth/np.tan(np.deg2rad(20)))


def test_full_carrier_footprint_and_open_wing_space_are_both_checked():
    from ats.base_layout import configure_layout
    from ats.base_layout import layout_footprints
    from ats.experiments import experiment_config
    from ats.equipment import resolve_config
    from ats.terrain import Terrain
    c=resolve_config(experiment_config(Config()));c.base.layout={}
    c.terrain.n=256;c.terrain.cell_size=1.;c.terrain.grab_radius=1.
    t=Terrain(c);t.design_work[:]=False;t.dump_mask[:]=False
    spec=configure_layout(c);point=[100.,100.]
    assert equipment_fits(c,spec,point,t)
    rect=next(iter(layout_footprints(spec,point)))
    x,y,w,h=rect;t.design_work[int(y+h-.5),int(x+w-.5)]=True
    assert not equipment_fits(c,spec,point,t)


def test_new_region_and_snapshot_preserve_crest_policy_and_selected_grid(tmp_path):
    from ats.checkpoint import save_state,load_state
    land=Landscape(coastal_dem());p=ChannelProject.create(land.raster,[[50,150],[150,150]],width_m=1,depth_m=.05,side_m=32)
    c=p.configuration(land,Config());s=Simulation(c);s.run(25);land.channel_project=p.to_data()
    region=tmp_path/'crest.atsregion';land.save(region);q=Landscape.load(region)
    assert ChannelProject.from_data(q.channel_project).identity==p.identity
    state=tmp_path/'crest.atsstate';save_state(state,s,land);loaded=load_state(state)
    a=s.run(25);b=loaded.simulation.run(25)
    assert a==b and loaded.landscape.channel_project['shore_reference']=='crest'
    assert a['georeference']['channel']['window_fit']['actual_side_m']==32


def test_opening_legacy_project_in_ui_does_not_silently_replace_its_goal():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.landscape_ui import LandscapeDialog
    from ats.checkpoint import Session
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    land=Landscape(coastal_dem());p=ChannelProject.create(land.raster,[[50,150],[150,150]],width_m=1,depth_m=.05,side_m=32,version=1)
    land.channel_project=p.to_data();c=p.configuration(land,Config());s=Simulation(c)
    window=MainWindow(c,Session(s,land));dialog=LandscapeDialog(window)
    assert dialog.prepare_channel(),dialog.result.text()
    assert dialog.project.identity==p.identity and dialog.project.shore_reference=='axis'
    assert 'прежний отступ по оси' in dialog.project_info.text()
    dialog.close();window.close();app.processEvents()


def test_shore_protection_matches_overlap_when_sea_is_outside_the_next_window():
    from tests.test_window_boundaries import window
    from ats.plan import channel_shore
    from ats.terrain import Terrain
    p,ch=window(side=512);s=p.active.simulation;old=s.terrain.core
    source=s.config;origin=np.asarray(source.georeference['origin_m']);other=origin+[88.,0.]
    c=ch.configuration(p.landscape,source,origin=other,side_m=512)
    new=Terrain(c)
    np.testing.assert_array_equal(new.protected[:,:424],old.protected[:,88:])
    _,a=channel_shore(old,source.plan);_,b=channel_shore(new,c.plan)
    np.testing.assert_allclose(a[:,88:],b[:,:424],atol=1e-12,rtol=0.)
    np.testing.assert_allclose(old.target[:,88:],new.target[:,:424],atol=1e-12,rtol=0.)


def test_a_shore_packet_cannot_silently_move_to_a_different_frame():
    land=Landscape(coastal_dem());p=ChannelProject.create(land.raster,[[50,150],[150,150]],width_m=1,depth_m=.05,side_m=32)
    c=p.configuration(land,Config());c.georeference['origin_m'][0]+=.25
    with pytest.raises(ValueError,match='другому рабочему окну'):Simulation(c)


def test_base_swept_path_keeps_clear_of_aircraft_parked_between_endpoints():
    from ats.drone import DroneState as DS
    from ats.traffic import ground_path_clear
    s=Simulation(Config());d=s.drone;old=s.base.position.copy();target=old[:2]+[10.,0.]
    d.state=DS.WAIT_GROUND;d.slot=None;d.position[:2]=old[:2]+[5.,0.]
    assert not ground_path_clear(s,target)
    d.position[1]+=s.base.config.radius+s.config.fleet.separation+1
    assert ground_path_clear(s,target)
    np.testing.assert_array_equal(s.base.position,old)


def test_late_ground_landing_cancels_pending_move_without_moving_any_actor():
    from tests.test_workfront import canal
    from ats.drone import DroneState as DS
    s=canal(logistics=False);d=s.drone;target=s.base.position[:2]+[10.,0.]
    s.relocation_pending=target;d.state=DS.WAIT_GROUND;d.trip=None;d.slot=None;s.slots[:]=[None]*len(s.slots)
    d.position[:2]=s.base.position[:2]+[5.,0.]
    base=s.base.position.copy();position=d.position.copy();energy=s.base.battery
    assert not s._start_front_move() and s.relocation_pending is None
    assert s.base_move_wait=='WAITING_GROUNDED_AIRCRAFT_CLEARANCE'
    np.testing.assert_array_equal(s.base.position,base);np.testing.assert_array_equal(d.position,position)
    assert s.base.battery==energy and s.base.destination is None


def test_route_remaining_inside_crest_buffer_reports_no_work_without_empty_reductions():
    y,x=np.indices((1001,1001));h=np.full(x.shape,.02);h[x<80]=-.02
    h[(x-470)**2+(y-900)**2<56**2]=-.02
    r=ElevationRaster(h,dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=1.,dy=1.))
    land=Landscape(r)
    p=ChannelProject.create(r,[[40,100],[82,200],[82,900],[470,900]],width_m=1,depth_m=.05,side_m=32)
    with pytest.raises(SiteError,match='защищённой береговой полосе'):p.configuration(land,Config())
    assert not land.tiles and not p.completed
