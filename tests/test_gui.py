import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from ats.config import Config


def test_offscreen_window_start_pause_reset_and_new_scenario():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(Config())
    window.show()
    app.processEvents()
    window.toggle_play()
    assert window.running
    window.credit = 4
    window.on_frame()
    assert window.sim.time > 0
    window.toggle_play()
    assert not window.running
    window.reset()
    assert window.sim.time == 0
    window.preset.setCurrentIndex(1)
    window.new_scenario()
    assert window.sim.config.trench.length == 5
    window.wind_x.setValue(2)
    assert window.sim.dust.config.wind_x == 2
    window.latitude.setValue(-35)
    window.season.setCurrentIndex(3)
    window.panels.setValue(30)
    window.base_mass.setValue(1500)
    window.new_scenario()
    assert window.sim.config.base.latitude_deg == -35
    assert window.sim.config.base.start_day_of_year == 355
    assert window.sim.config.base.panel_area == 30
    assert window.sim.config.base.mass == 1500
    from PySide6 import QtCore
    window.calendar.setDate(QtCore.QDate(2025, 7, 3))
    assert window.season.currentIndex() == -1
    window.sim.time = 2572
    window.refresh()
    assert "09:42:52" in window.telemetry.text()
    assert "0 д 00:42:52" in window.telemetry.text()
    from ats.fleet_ui import FleetDialog,ComparisonWindow
    dialog=FleetDialog(window.config,window)
    dialog.count.setValue(3);dialog.slots.setValue(2);dialog.preset(True);dialog.save()
    assert dialog.result_config.fleet.count==3
    assert dialog.result_config.fleet.models[0].payload==30
    window.replace_simulation(dialog.result_config)
    window.sim.run(20);window.refresh(True)
    assert window.fleet_table.rowCount()==3
    assert len(window.drone_labels)>=3
    comparison=ComparisonWindow(window.config,window);comparison.timer.stop();comparison.tick()
    assert comparison.sim is not None
    comparison.cancel();assert not comparison.timer.isActive();comparison.close()
    window.close()


def test_real_mouse_brush_apply_segments_and_profile_switches():
    import numpy as np
    from PySide6 import QtWidgets,QtCore,QtGui,QtTest
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window=MainWindow(Config());window.show();window.brush_button.setChecked(True)
    QtTest.QTest.qWait(150);window.timer.stop()
    viewport=window.map.viewport()
    a=window.map.mapFromScene(window.view_box.mapViewToScene(QtCore.QPointF(10,10)))
    b=window.map.mapFromScene(window.view_box.mapViewToScene(QtCore.QPointF(12,11)))
    QtTest.QTest.mouseMove(viewport,a)
    QtTest.QTest.mousePress(viewport,QtCore.Qt.MouseButton.LeftButton,pos=a)
    for i in range(1,11):
        QtTest.QTest.qWait(20)  # actual scene event-rate limit; no mocked brush callbacks
        p=QtCore.QPoint(round(a.x()+(b.x()-a.x())*i/10),round(a.y()+(b.y()-a.y())*i/10))
        event=QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove,QtCore.QPointF(p),QtCore.QPointF(viewport.mapToGlobal(p)),QtCore.Qt.MouseButton.NoButton,QtCore.Qt.MouseButton.LeftButton,QtCore.Qt.KeyboardModifier.NoModifier)
        QtWidgets.QApplication.sendEvent(viewport,event)
    QtTest.QTest.mouseRelease(viewport,QtCore.Qt.MouseButton.LeftButton,pos=b);app.processEvents()
    assert len(window.plan_editor.segments)==1
    assert len(window.plan_editor.segments[0].points)>1
    assert np.linalg.norm(np.asarray(window.plan_editor.segments[0].points[-1])-window.plan_editor.segments[0].points[0])>1
    before=window.sim.terrain.height.copy()
    button=next(b for b in window.plan_editor.findChildren(QtWidgets.QPushButton) if b.text()=='Применить план')
    button.click();app.processEvents()
    assert window.config.plan is not None
    assert window.sim.terrain.remaining_cut>0
    np.testing.assert_array_equal(before,window.sim.terrain.height)
    window.add_brush_stroke([[16,12],[16,15]])
    button.click()
    assert window.profile_segment.count()==3
    window.profile_segment.setCurrentIndex(1);window.profile_position.setValue(0)
    first=window.profile_line.getData()[0].copy()
    window.profile_position.setValue(100)
    assert not np.array_equal(first,window.profile_line.getData()[0],equal_nan=True)
    window.profile_segment.setCurrentIndex(0);window.profile_direction.setCurrentIndex(1)
    assert np.count_nonzero(np.isnan(window.current_curve.getData()[1]))==1
    assert len(window.profile_boundaries)==1
    window.plan_editor.clear();button.click()
    assert window.profile_segment.count()==1
    assert window.sim.terrain.remaining==0
    window.close()
