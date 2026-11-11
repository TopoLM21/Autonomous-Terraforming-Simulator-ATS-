"""Exercise the public planning window without creating physical equipment."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import threading
import time
import numpy as np
import pytest
from PySide6 import QtCore,QtWidgets,QtTest
from ats.config import Config
from ats.landscape import Landscape
from ats.complex_setup_ui import WorkspaceWindow
from tests.test_landscape_display import raster


@pytest.fixture
def app():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    for widget in app.topLevelWidgets():widget.close()
    app.processEvents()


def wait(app,predicate,seconds=5):
    end=time.perf_counter()+seconds
    while not predicate() and time.perf_counter()<end:
        app.processEvents();QtTest.QTest.qWait(10)
    assert predicate()


def planning(app):
    h=np.full((31,41),20.);h[:,:6]=-30.;h[11:21,15:26]=-30.
    w=WorkspaceWindow(Config());w._landscape_draft=Landscape(raster(h,dx=10.,dy=10.))
    w.show_landscape();d=w.landscape_window
    app.processEvents()
    return w,d,d.elevation_display


def test_msl_and_horizontal_slices_preserve_route_project_zoom_and_physical_state(app,monkeypatch):
    from ats.simulation import Simulation
    from ats.inventory import Inventory
    def fail(*a,**kw):raise AssertionError('Read-only display created equipment')
    monkeypatch.setattr(Simulation,'__init__',fail);monkeypatch.setattr(Inventory,'__init__',fail)
    w,d,e=planning(app)
    d.points=[[30.,150.],[210.,150.]];d.route.setData(np.array(d.points))
    d.width.setValue(1);d.depth.setValue(.1);d.project_side.setCurrentIndex(d.project_side.findData(128))
    assert d.prepare_channel(),d.result.text()
    before=d.region.report();project=d.project;points=[p.copy() for p in d.points]
    raw=d.region.raster.height.copy();route=tuple(a.copy() for a in d.route.getData())
    d.map.setRange(xRange=(90,280),yRange=(70,230),padding=0);bounds=np.array(d.map.viewRange())
    assert e.mode.currentData()=='msl' and e.zero_line.isChecked()
    for mode in ('height','below','band','msl'):
        e.mode.setCurrentIndex(e.mode.findData(mode));e.level.setValue(-20.);e.thickness.setValue(4.);e.draw();app.processEvents()
        assert d.project is project and d.points==points
        for a,b in zip(d.route.getData(),route):np.testing.assert_array_equal(a,b)
        # The fixed-aspect view adapts its vertical span when slice controls
        # take space; toggling must keep the chosen centre and horizontal zoom.
        current=np.array(d.map.viewRange())
        np.testing.assert_allclose(current[0],bounds[0])
        np.testing.assert_allclose(current.mean(axis=1),bounds.mean(axis=1))
        assert d.region.report()==before and w.sim is None
    np.testing.assert_array_equal(d.region.raster.height,raw)
    assert not d.region.tiles and not d.region.soil_tiles
    assert 'сухая впадина' in e.legend.text() and '0 м MSL' in e.legend.text()


def test_sections_and_shift_click_inspect_landscape_without_editing_the_route(app):
    w,d,e=planning(app);d.points=[[60.,0.],[300.,300.]];d.route.setData(np.array(d.points));e.show_sections()
    s=e.section_window;app.processEvents()
    assert s.direction.currentData()=='route' and not s.position.isEnabled()
    assert s.ground.getData()[0][-1]==pytest.approx(np.hypot(240,300))
    for mode,coordinate in (('x',90.),('y',120.)):
        s.direction.setCurrentIndex(s.direction.findData(mode));s.position.setValue(coordinate);app.processEvents()
        line=e.section_line.getData()
        np.testing.assert_allclose(line[1 if mode=='x' else 0],coordinate)
    s.direction.setCurrentIndex(s.direction.findData('x'))
    previous=[p.copy() for p in d.points]
    point=QtCore.QPointF(150.,110.);scene=d.map.getViewBox().mapViewToScene(point)
    QtTest.QTest.mouseClick(d.map.viewport(),QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ShiftModifier,d.map.mapFromScene(scene))
    app.processEvents()
    assert s.position.value()==pytest.approx(110.,abs=2.) and d.points==previous
    e.zero_line.setChecked(False);e.draw();s.refresh()
    assert not e.contour.isVisible() and not e.profile_zero.isVisible() and not s.sea.isVisible()
    s.close();assert e.section_line.getData()[0] is None


def test_hover_distinguishes_sea_dry_depression_and_nodata(app):
    w,d,e=planning(app)
    def hover(x,y):e.hover((d.map.getViewBox().mapViewToScene(QtCore.QPointF(x,y)),));return e.probe.text()
    assert 'морское дно' in hover(20.,150.)
    assert 'сухая впадина' in hover(200.,150.)
    assert '-30.00 м MSL' in e.probe.text()
    assert 'выше моря' in hover(320.,150.)
    h=np.full((31,41),20.);h[13:18,18:23]=np.nan
    d.region=Landscape(raster(h,dx=10.,dy=10.));d.show_region();e.draw()
    assert 'NoData' in hover(200.,150.)


def test_nodata_viewport_source_change_and_reopening_restore_display_only(app):
    w,d,e=planning(app)
    e.mode.setCurrentIndex(e.mode.findData('band'));e.level.setValue(-25.);e.thickness.setValue(8.)
    w._landscape_draft=d.region;d.close();w.show_landscape();d=w.landscape_window;e=d.elevation_display
    assert e.mode.currentData()=='band' and e.level.value()==-25. and e.thickness.value()==8.
    h=np.full((31,41),np.nan);h[0,0]=1.
    d.region=Landscape(raster(h,dx=10.,dy=10.));d.show_region()
    d.map.setRange(xRange=(180,220),yRange=(130,170),padding=0);e.draw()
    assert np.isnan(e._grid[0]).all() and np.isfinite(d.image.image).all()
    assert e.contour.getData()[0] is None and 'NoData' in e.legend.text()
    assert e.level.minimum()==0 and e.level.maximum()==1


def test_larger_dem_sea_preparation_does_not_block_gui_and_discards_stale_result(app,monkeypatch):
    import ats.landscape_display_ui as module
    original=module.ocean_nodes;started=threading.Event();release=threading.Event()
    def delayed(r):
        if r.height.size>262144:started.set();release.wait(5)
        return original(r)
    monkeypatch.setattr(module,'ocean_nodes',delayed)
    w,d,e=planning(app);small=d.region
    beats=[];timer=QtCore.QTimer();timer.setInterval(5);timer.timeout.connect(lambda:beats.append(1));timer.start()
    try:
        large=Landscape(raster(np.ones((520,520))));d.region=large;d.show_region()
        generation=e.generation;cancel=e.cancel;wait(app,started.is_set);QtTest.QTest.qWait(40)
        assert beats and e.marine is None and 'вычисляется' in e.legend.text()
        d.region=small;d.show_region();mask=e.marine
        assert cancel.is_set() and mask is not None
        release.set();e.marine_ready(generation,large.raster,np.ones((520,520),bool),'')
        app.processEvents();assert e.raster is small.raster and e.marine is mask
        d.close();assert e.cancel.is_set() and e.closed and not e.range_timer.isActive()
    finally:release.set();timer.stop()


def test_failed_background_mask_reports_error_without_changing_the_region(app,monkeypatch):
    import ats.landscape_display_ui as module
    w,d,e=planning(app)
    def broken(*a):raise OSError('test DEM read failed')
    monkeypatch.setattr(module,'ocean_nodes',broken)
    d.region=Landscape(raster(np.ones((520,520))));before=d.region.report();d.show_region()
    wait(app,lambda:bool(e.marine_error))
    assert 'test DEM read failed' in e.legend.text() and e.marine is None
    assert d.region.report()==before and w.sim is None


def test_planning_map_stays_large_and_controls_scroll_on_a_small_screen(app):
    w,d,e=planning(app);d.resize(1024,768);app.processEvents()
    assert d.map.height()>400
    assert d.controls_scroll.verticalScrollBar().maximum()>0
    for mode in ('msl','band','below'):
        e.mode.setCurrentIndex(e.mode.findData(mode));app.processEvents()
        assert d.minimumSizeHint().width()<=1024
        assert e.sections_button.geometry().right()<=e.toolbar.width()
        assert d.map.height()>350
    previous=d.map.width();e.parameters_button.click();app.processEvents()
    assert not d.controls_scroll.isVisible() and d.map.width()>previous
    e.parameters_button.click();app.processEvents();assert d.controls_scroll.isVisible()


def test_real_qattara_default_is_msl_and_dry_destination_is_not_marine(app):
    from ats.channel import is_ocean
    w=WorkspaceWindow(Config());w.show_landscape();d=w.landscape_window;e=d.elevation_display
    d.example_corridor();app.processEvents()
    assert e.mode.currentData()=='msl' and d.project is not None
    assert not bool(is_ocean(d.region.raster,*d.project.points[-1]))
    assert d.image.image.ndim==3 and len(e.contour.getData()[0])>0
    e.show_sections();s=e.section_window
    assert len(s.ground.getData()[0])>2 and s.sea.value()==0
    assert 'без проектного прокопа' in s.info.text()
