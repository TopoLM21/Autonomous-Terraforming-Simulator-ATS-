"""Real scale, heading, reusable map items and visible equipment passports."""
import gc
import math
import os
from types import SimpleNamespace
import numpy as np
import pytest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6 import QtCore,QtWidgets
from ats.conventional_visuals import body_polygons,refresh_geometry


_APP=None


@pytest.fixture
def app():
    global _APP
    _APP=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gc.collect();yield _APP
    _APP.sendPostedEvents(None,QtCore.QEvent.Type.DeferredDelete);gc.collect()


def test_body_uses_meter_dimensions_and_heading_instead_of_pixel_marker_size():
    m=SimpleNamespace(id='truck_1',kind='truck',position=np.array([20.,30.,0.]),path=[],
                      dimensions_m=(8.,3.,4.),heading_rad=math.pi/2)
    fleet=SimpleNamespace(base_position=np.array([5.,5.,0.]),machines=[m])
    base,truck=body_polygons(fleet)
    assert np.ptp(base['polygon'][:,0])==8. and np.ptp(base['polygon'][:,1])==8.
    assert np.ptp(truck['polygon'][:,0])==pytest.approx(3.)
    assert np.ptp(truck['polygon'][:,1])==pytest.approx(8.)
    np.testing.assert_allclose(truck['polygon'].mean(axis=0),m.position[:2])
    del m.heading_rad;m.path=[np.array([20.,40.])]
    # A version-22 machine without heading still points along its saved path.
    np.testing.assert_allclose(body_polygons(fleet)[1]['polygon'],truck['polygon'])


def test_shared_planned_ramp_geometry_invalidates_cache_without_growing_scene(app):
    import pyqtgraph as pg
    host=SimpleNamespace();view=pg.PlotWidget()
    fleet=SimpleNamespace(base_position=np.array([5.,5.,0.]),machines=[],
        ramp=dict(entry_m=[10.,10.],end_m=[20.,10.],width_m=9.))
    try:
        refresh_geometry(host,view,fleet);item=host._ground_geometry_item;picture=item._picture
        count=len(view.getViewBox().addedItems)
        assert np.ptp(item.ramp_polygon[:,0])==10. and np.ptp(item.ramp_polygon[:,1])==9.
        assert host._ground_ramp_label.toPlainText()=='Проектный съезд'
        refresh_geometry(host,view,fleet);assert item._picture is picture
        fleet.ramp['end_m']=[30.,10.];refresh_geometry(host,view,fleet)
        assert item._picture is not picture and np.ptp(item.ramp_polygon[:,0])==20.
        assert len(view.getViewBox().addedItems)==count
        refresh_geometry(host,view,None)
        assert not item.isVisible() and not host._ground_ramp_label.isVisible()
    finally:view.close();app.processEvents()


def test_workspace_draws_actual_bodies_and_reuses_the_cached_picture(app,tmp_path):
    from ats.conventional import ConventionalConfig
    from ats.conventional_workspace import ConventionalWorkWindow
    from tests.test_conventional_workspace import workspace_config
    w=ConventionalWorkWindow(workspace_config());w.timer.stop()
    try:
        cfg=ConventionalConfig(base_x=12.,base_y=12.,work_rect=(35.,40.,60.,20.),truck_count=2,
            excavator_dimensions_m=(6.5,3.2,3.1),truck_dimensions_m=(7.2,2.8,3.3))
        w.install_conventional(cfg);fleet=w.sim.conventional
        item=w._ground_geometry_item;picture=item._picture;rows=item.polygons
        assert len(rows)==4 and rows[1]['dimensions_m']==cfg.excavator_dimensions_m
        refresh_geometry(w,w.map,fleet)
        assert w._ground_geometry_item is item and item._picture is picture
        fleet.machines[1].heading_rad=math.pi/2;w.refresh()
        truck=next(row for row in item.polygons if row['id']==fleet.machines[1].id)
        assert np.ptp(truck['polygon'][:,0])==pytest.approx(2.8)
        assert np.ptp(truck['polygon'][:,1])==pytest.approx(7.2)
        assert item._picture is not picture
        path=tmp_path/'bodies.atsstate';assert w.save_session(path);assert w.load_session(path)
        assert w._ground_geometry_item is item
        truck=next(row for row in item.polygons if row['id']=='truck_1')
        assert truck['heading_rad']==math.pi/2
    finally:w.close();app.processEvents()


def test_passport_edits_and_displays_installed_dimensions(app):
    from ats.conventional_workspace import ConventionalWorkWindow
    from tests.test_conventional_workspace import workspace_config
    w=ConventionalWorkWindow(workspace_config());w.timer.stop()
    try:
        w.show_conventional();d=w.equipment_window;d.timer.stop()
        assert not d.auto_area.isEnabled() and not d.auto_area.isChecked()
        for widget,value in zip(d.dimension_controls['truck_dimensions_m'],(7.2,2.8,3.3)):widget.setValue(value)
        assert d.config_value().truck_dimensions_m==(7.2,2.8,3.3)
        d.controls['base_x'].setValue(12.);d.controls['base_y'].setValue(12.)
        for widget,value in zip(d.rect,(35.,40.,60.,20.)):widget.setValue(value)
        d.create_button.click()
        assert w.sim.conventional is not None,d.message.text()
        assert '7.2 × 2.8 × 3.3 м' in d.model_info.text()
        assert d.table.item(1,6).text()=='7.2 × 2.8 × 3.3'
        assert not d.dimension_controls['truck_dimensions_m'][0].isEnabled()
        assert w.sim.conventional.config.truck_dimensions_m==(7.2,2.8,3.3)
    finally:w.close();app.processEvents()


def test_default_channel_dialog_automatically_allocates_then_displays_actual_site(app):
    from ats.conventional_workspace import ConventionalWorkWindow
    from tests.test_conventional_siting import channel_site
    sim,_=channel_site();w=ConventionalWorkWindow(sim.config);w.timer.stop()
    try:
        w.show_conventional();d=w.equipment_window;d.timer.stop()
        assert d.auto_area.isEnabled() and d.auto_area.isChecked()
        assert not d.controls['base_x'].isEnabled() and not d.rect[0].isEnabled()
        assert w.sim.conventional is None and w.sim.time==0.
        d.create_button.click();fleet=w.sim.conventional
        assert fleet is not None,d.message.text()
        assert not fleet.config.auto_work_area and not d.auto_area.isChecked()
        assert tuple(widget.value() for widget in d.rect)==pytest.approx(fleet.config.work_rect,abs=.005)
        assert d.controls['base_x'].value()==pytest.approx(fleet.config.base_x,abs=.005)
        assert d.controls['base_y'].value()==pytest.approx(fleet.config.base_y,abs=.005)
        assert len(w._ground_geometry_item.polygons)==len(fleet.machines)+1
        assert not d.create_button.isEnabled() and w.sim.time==0.
    finally:w.close();app.processEvents()
