"""Larger physical fields keep SI coordinates; zoom changes presentation only."""
import json
import os
import subprocess
import sys
import numpy as np
import pytest
from ats.config import Config
from ats.maps import with_map_size
from ats.experiments import experiment_config,save_configuration,load_configuration
from ats.simulation import Simulation


@pytest.mark.parametrize('side,cells',[(32,128),(64,256),(128,512),(256,1024)])
def test_map_extent_preserves_resolution_and_equipment_coordinates(side,cells):
    original=Config();changed=with_map_size(original,side)
    assert original.terrain.n==128
    assert changed.terrain.n==cells and changed.terrain.cell_size==.25
    assert changed.base.x==original.base.x and changed.base.y==original.base.y
    assert changed.trench==original.trench and changed.terrain.dump_rect==original.terrain.dump_rect


@pytest.mark.parametrize('side',[0,31,257,64.1,float('nan'),float('inf')])
def test_invalid_map_size_does_not_mutate_config(side):
    c=Config()
    with pytest.raises(ValueError):with_map_size(c,side)
    assert c.terrain.n==128


def test_custom_height_grid_cannot_be_silently_discarded():
    c=Config();c.initial_height=np.zeros((128,128)).tolist()
    with pytest.raises(ValueError,match='рельеф'):with_map_size(c,64)
    assert with_map_size(c,32).initial_height==c.initial_height


def test_large_field_brush_mass_and_configuration_roundtrip(tmp_path):
    from ats.plan import PlanConfig,PlanSegment
    c=with_map_size(Config(),64)
    c.plan=PlanConfig([PlanSegment([[46,46],[47,46]],width=1.,height=-.25)])
    c.trench.x=c.trench.y=46
    path=tmp_path/'large-map.json';save_configuration(path,c)
    clone=load_configuration(path)
    s=Simulation(clone);t=s.terrain
    assert t.height.shape==(256,256) and s.report()['map_side_m']==64
    mass=t.scoop(46,46,30);assert mass>0
    t.deposit(26,24,mass)
    assert abs(t.mass-t.initial_volume*c.terrain.density)<1e-7
    assert clone.terrain.n==c.terrain.n and clone.plan==c.plan


def test_cli_new_default_map_and_loaded_size_are_explicit(tmp_path):
    saved=tmp_path/'existing.json';save_configuration(saved,Config())
    for flags,side in [([],64),(['--configuration',str(saved)],32),(['--map-size','128'],128),
                       (['--configuration',str(saved),'--map-size','64'],64)]:
        r=subprocess.run([sys.executable,'-m','ats','--headless','--steps','0',*flags],
                         capture_output=True,text=True,encoding='utf-8',timeout=30)
        assert r.returncode==0,r.stderr
        data=json.loads(r.stdout)
        assert data['map_side_m']==side and data['cell_size_m']==.25
        assert data['grid_cells']==side/.25


def test_real_zoom_buttons_wheel_fullscreen_and_brush_use_large_coordinates():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets,QtCore,QtGui,QtTest
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c=with_map_size(Config(),64);c.trench.x=c.trench.y=48
    w=MainWindow(c);w.timer.stop();w.show();QtTest.QTest.qWait(100)
    sim=w.sim;terrain=sim.terrain.height.copy();plot=w.map
    span=lambda:min(b-a for a,b in w.map.viewRange())
    w.zoom_fit.click();app.processEvents();initial=span()
    w.zoom_in.click();app.processEvents();assert span()<initial*.85
    w.zoom_out.click();app.processEvents();assert span()==pytest.approx(initial,rel=.02)
    viewport=w.map.viewport();point=viewport.rect().center()
    wheel=QtGui.QWheelEvent(QtCore.QPointF(point),QtCore.QPointF(viewport.mapToGlobal(point)),
        QtCore.QPoint(0,0),QtCore.QPoint(0,120),QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,QtCore.Qt.ScrollPhase.ScrollUpdate,False)
    before=span();QtWidgets.QApplication.sendEvent(viewport,wheel);app.processEvents()
    assert span()<before
    w.map_fullscreen.click();app.processEvents()
    dialog=w.map_dialog;assert dialog is not None and dialog.isFullScreen()
    QtTest.QTest.keyClick(dialog,QtCore.Qt.Key.Key_Escape);app.processEvents()
    assert w.map_dialog is None and w.map is plot and w.sim is sim and sim.time==0
    np.testing.assert_array_equal(terrain,sim.terrain.height)
    w.activateWindow();QtTest.QTest.qWait(30)
    QtTest.QTest.keyClick(w,QtCore.Qt.Key.Key_F11);app.processEvents()
    assert w.map_dialog is not None
    QtTest.QTest.keyClick(w.map_dialog,QtCore.Qt.Key.Key_F11);app.processEvents()
    assert w.map_dialog is None
    w.brush_button.setChecked(True);app.processEvents()
    w.map.setRange(xRange=(40,56),yRange=(40,56),padding=0);app.processEvents()
    target=w.map.mapFromScene(w.view_box.mapViewToScene(QtCore.QPointF(48,48)))
    assert viewport.rect().contains(target)
    QtTest.QTest.mouseClick(viewport,QtCore.Qt.MouseButton.LeftButton,pos=target)
    app.processEvents()
    assert len(w.plan_editor.segments)==1
    np.testing.assert_allclose(w.plan_editor.segments[0].points[0],[48,48],atol=.15)
    w.map_fullscreen.click();app.processEvents();assert w.map_dialog is not None
    w.close();app.processEvents();assert w.map_dialog is None


def test_new_scenario_and_module_slot_tabs_follow_selected_map_and_mode():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.logistics_ui import LogisticsWindow,ExperimentDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(experiment_config(Config()));w.timer.stop()
    view=LogisticsWindow(w);view.timer.stop();view.tabs.setCurrentIndex(3)
    assert view.charging.rowCount()==20  # 16 drone modules + four Krot modules
    assert any(view.charging.item(row,2).text()=='FULL' for row in range(20))
    dialog=ExperimentDialog(w)
    assert dialog.charge_mode.currentData() and not dialog.ports.isEnabled()
    dialog.charge_mode.setCurrentIndex(1);assert dialog.ports.isEnabled()
    dialog.map_size.setValue(64);candidate=dialog.candidate()
    assert candidate.terrain.n==256 and not candidate.logistics.individual_chargers
    w.map_size.setValue(128);w.cx.setValue(80);w.cy.setValue(80);w.new_scenario()
    assert w.sim.terrain.height.shape==(512,512) and w.config.trench.x==80 and w.config.trench.y==80
    assert w.cx.maximum()>100 and w.map_size.value()==128
    dialog.close();view.close();w.close();app.processEvents()
