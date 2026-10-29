"""Display effects are reversible UI preferences, independent of model physics."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
import numpy as np
import pytest
from ats.config import Config
from ats.simulation import Simulation
from ats.map_visuals import sky,direction,relief_light,terrain_rgb


@pytest.fixture
def app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def window(config=None,session=None):
    from ats.gui import MainWindow
    w=MainWindow(config or Config(),session=session);w.timer.stop();return w


@pytest.mark.parametrize('latitude,day',[(31.078,172),(31.078,355),(-31.078,172),(90.,172),(90.,355)])
def test_sky_uses_real_calendar_and_finite_solar_vector(latitude,day):
    for hour in (0.,6.,12.,18.,23.99):
        elevation,night,warm=sky(latitude,day,hour)
        v=direction(latitude,day,hour)
        assert np.linalg.norm(v)==pytest.approx(1.)
        assert np.sin(np.deg2rad(elevation))==pytest.approx(v[2])
        assert 0<=night<=1 and 0<=warm<=1
    assert sky(31.078,172,12)[0]>sky(31.078,355,12)[0]


def test_visual_sampling_uses_model_quarters_and_year_rollover():
    c=Config();c.start_hour=23.75;c.base.start_day_of_year=365
    s=Simulation(c);a,_=relief_light(s,True)
    s.time=899.9;b,_=relief_light(s,True);assert a==b
    s.time=900.;b,_=relief_light(s,True);assert b!=a and b[1]==1
    s.time=86400*4.;assert relief_light(s,False)[0]==('fixed',)
    s.base.config.latitude_deg=-55;assert relief_light(s,False)[0]==('fixed',)
    # Near sunrise there are no visual jumps across a render-time boundary.
    before=sky(31.078,172,5.)[1];after=sky(31.078,172,5.+1/3600)[1]
    assert abs(before-after)<.002


def test_palette_distinguishes_dry_negative_ground_sea_cut_and_fill():
    h=np.full((8,8),-20.);reference=h.copy();sea=np.zeros_like(h,dtype=bool);sea[:2,:2]=True
    h[3,3]-=1.;h[5,5]+=1.;original=h.copy()
    rgb=terrain_rgb(h,1.,5.,1.,reference=reference,marine=sea)
    assert rgb.dtype==np.uint8
    assert rgb[0,0,2]>rgb[0,0,0] and rgb[7,7,0]>rgb[7,7,2]
    assert not np.array_equal(rgb[3,3],rgb[7,7]) and not np.array_equal(rgb[5,5],rgb[7,7])
    np.testing.assert_array_equal(h,original)


def test_switches_keep_physics_and_protected_layers_unchanged(app,tmp_path):
    from ats.checkpoint import load_state
    c=Config();c.start_hour=23.
    w=window(c)
    try:
        w.show();app.processEvents();before=w.sim.report();height=w.sim.terrain.height.copy()
        assert w.visual_night>.9 and w.night_overlay.isVisible() and w.light_points.isVisible()
        assert len(w.light_points.data)==len(w.sim.drones)
        for item in [w.design_overlay,w.brush_overlay,w.profile_line,w.route,w.brush_cursor,*w.drone_labels,w.base_label]:
            assert item.zValue()>w.night_overlay.zValue()
        w.day_night.setChecked(False)
        assert not w.night_overlay.isVisible() and w.light_points.isVisible()
        w.drone_lights.setChecked(False);assert not w.light_points.isVisible()
        assert w.sim.report()==before;np.testing.assert_array_equal(w.sim.terrain.height,height)
        path=tmp_path/'display.atsstate';w.save_session(path);w.timer.stop()
        q=load_state(path);assert q.ui['day_night'] is False and q.ui['drone_lights'] is False
        assert q.simulation.report()==before
        w.day_night.setChecked(True);w.drone_lights.setChecked(True)
        w.load_session(path);w.timer.stop()
        assert not w.day_night.isChecked() and not w.drone_lights.isChecked()
        assert w.sim.report()==before
    finally:w.close();app.processEvents()


def test_fixed_light_date_and_latitude_changes_and_pause(app):
    w=window()
    try:
        w.day_night.setChecked(False);w.drone_lights.setChecked(False);w.refresh(True);image=w.image.image.copy()
        w.sim.config.start_hour=23.;w.sim.base.config.latitude_deg=60.;w.sim.base.config.start_day_of_year=355
        w.refresh(True);np.testing.assert_array_equal(w.image.image,image)
        assert not w.night_overlay.isVisible()
        w.day_night.setChecked(True);assert w.night_overlay.brush().color().alpha()>100
        key=w._last_light_key;frozen=w.night_overlay.brush().color().rgba();w.refresh()
        assert w._last_light_key==key and w.night_overlay.brush().color().rgba()==frozen
        w.sim.base.config.latitude_deg=-60.;w.refresh(True)
        assert w._last_light_key!=key
        # Old saves may carry no UI dictionary at all.
        w.restore_session_ui(None);assert w.day_night.isChecked()
    finally:w.close();app.processEvents()


def test_fixed_geometry_is_cached_and_partial_unfolding_and_pose_invalidate(app,monkeypatch):
    import ats.site_ui as site
    from tests.test_self_solar import world
    from ats.checkpoint import Session
    s=world();w=window(session=Session(s,ui={}))
    try:
        calls=[];build=site.geometry
        def counted(sim):calls.append(1);return build(sim)
        monkeypatch.setattr(site,'geometry',counted)
        w.refresh(True);assert not calls
        s.run(100);w.refresh(True);assert calls
        partial=np.column_stack(w.solar_active.getData());s.run(135);w.refresh(True)
        assert s.logistics.field_ready and not np.array_equal(np.column_stack(w.solar_deployed.getData()),partial,equal_nan=True)
        calls.clear();w.refresh(True);assert not calls
        original=w.solar_deployed.getData()[0].copy();s.base.position[0]+=1;w.refresh(True)
        assert calls;np.testing.assert_allclose(w.solar_deployed.getData()[0],original+1,equal_nan=True)
        calls.clear();s.base.layout_spec['access_lanes_local'][0]=list(s.base.layout_spec['access_lanes_local'][0])
        s.base.layout_spec['access_lanes_local'][0][0]+=.1;w.refresh(True);assert calls
        s.logistics.solar_array.request_fold();s.run(30);w.refresh(True)
        assert w.solar_active.getData()[0] is not None
    finally:w.close();app.processEvents()


def test_terrain_and_profile_refresh_after_digging_without_static_geometry_rebuild(app,monkeypatch):
    import ats.site_ui as site
    w=window()
    try:
        original=w.image.image.copy();calls=[]
        monkeypatch.setattr(site,'geometry',lambda s:calls.append(s))
        w.sim.run(90);w.refresh(True)
        assert not calls and w.last_revision==w.sim.terrain.revision
        assert not np.array_equal(original,w.image.image)
        assert w._profile_key[2]==w.sim.terrain.revision
        assert abs(w.sim.report()['mass_error_kg'])<1e-7
    finally:w.close();app.processEvents()


def test_effects_and_hidden_telemetry_do_not_change_accelerated_model(app,monkeypatch):
    from ats.experiments import experiment_config
    c=experiment_config(Config());c.start_hour=23.
    w=window(c);reference=Simulation(deepcopy(c))
    try:
        w.running=True
        for i in range(120):
            w.sim.step();reference.step()
            if i%7==0:
                w.day_night.setChecked(not w.day_night.isChecked());w.drone_lights.setChecked(not w.drone_lights.isChecked());w.running=True
            if i%5==0:w.refresh()
        assert w.sim.report()==reference.report()
        before=w.sim.report();w.zoom_in.click();w.refresh(True);assert w.sim.report()==before
    finally:w.close();app.processEvents()


def test_shared_beacons_and_zoom_bound_the_vector_cache(app):
    from PySide6 import QtWidgets
    from ats.checkpoint import Session
    from tests.test_battery_platforms import platform_group
    p=platform_group()
    for u in p.units:u.simulation.config.start_hour=23.
    w=window(session=Session(p.active.simulation,p.landscape,{},complex_project=p))
    try:
        w.show();app.processEvents();w.refresh(True)
        assert len(w.light_points.data)==sum(len(u.simulation.drones) for u in p.units)
        assert all(label.zValue()>w.night_overlay.zValue() for label in w.peer_labels)
        w.map.setRange(xRange=(0,1),yRange=(0,1),padding=0);w.refresh(True)
        assert w.solar_planned.curve.cacheMode()==QtWidgets.QGraphicsItem.CacheMode.NoCache
        w.fit_map();w.refresh(True)
        assert w.solar_planned.curve.cacheMode()==QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache
    finally:w.close();app.processEvents()


def test_repeated_frames_reuse_marker_styles_instead_of_growing_the_symbol_atlas(app):
    w=window()
    try:
        w.show();app.processEvents();w.running=True
        for _ in range(3):w.refresh();app.processEvents()
        before=len(w.drone_marker.fragmentAtlas)
        for _ in range(40):w.refresh()
        assert len(w.drone_marker.fragmentAtlas)==before
    finally:w.close();app.processEvents()


def test_display_preferences_and_caches_survive_group_roundtrip_and_load(app,tmp_path):
    from ats.checkpoint import Session,save_state
    from tests.test_battery_platforms import platform_group
    from tests.test_regional import advance
    p=platform_group()
    for u in p.units:u.simulation.request_landing()
    advance(p,lambda:all(u.simulation.logistics.field_ready and u.simulation.logistics.battery_platforms.ready for u in p.units))
    w=window(session=Session(p.active.simulation,p.landscape,{},complex_project=p))
    try:
        w.day_night.setChecked(False);ids=p.inventory_ids();p.move_window([112.,64.])
        advance(p,lambda:p.transition.phase=='DRIVE');w.refresh(True)
        path=tmp_path/'moving-display.atsstate'
        save_state(path,p.active.simulation,p.landscape,w.session_ui(),complex_project=p)
        w.load_session(path);w.timer.stop();p=w.complex_project
        assert not w.day_night.isChecked() and w.drone_lights.isChecked()
        advance(p,lambda:p.transition is None)
        w.running=True;w.credit=1;w.on_frame();w.running=False
        assert w.config.georeference['origin_m']==[112.,64.]
        assert w.last_revision==w.sim.terrain.revision
        p.move_window([64.,64.]);advance(p,lambda:p.transition is None)
        w.running=True;w.credit=1;w.on_frame();w.running=False
        assert w.config.georeference['origin_m']==[64.,64.]
        assert not w.day_night.isChecked() and p.inventory_ids()==ids
        from ats.site_ui import geometry
        arrays=[geometry(u.simulation)[1] for u in p.units]
        expected=np.concatenate(arrays);actual=np.column_stack(w.solar_deployed.getData())
        np.testing.assert_allclose(actual,expected,equal_nan=True)
        assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002
    finally:w.close();app.processEvents()
