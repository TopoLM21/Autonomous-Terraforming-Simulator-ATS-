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
    assert not window.panels.isReadOnly() and window.base_mass.isReadOnly()
    window.new_scenario()
    assert window.sim.config.base.latitude_deg == -35
    assert window.sim.config.base.start_day_of_year == 355
    assert window.sim.config.base.panel_area == 20
    assert window.sim.config.base.mass == 1000
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
    window.apply_equipment(dialog.result_config)
    window.sim.run(20);window.refresh(True)
    assert window.fleet_table.rowCount()==3
    assert len(window.drone_labels)>=3
    comparison=ComparisonWindow(window.config,window);comparison.timer.stop();comparison.tick()
    assert comparison.sim is not None
    comparison.cancel();assert not comparison.timer.isActive();comparison.close()
    window.close()


def test_equipment_editor_live_apply_and_daily_history():
    from copy import deepcopy
    import numpy as np
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.equipment_ui import EquipmentDialog,DailyStatsWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(Config());w.timer.stop();w.sim.run(90);w.refresh(True)
    sim=w.sim;time=sim.time;height=sim.terrain.height.copy();logs=w.log.toPlainText()
    dialog=EquipmentDialog(w.config,w)
    dialog.base.panels.table.item(0,1).setText('15');dialog.base.slots.setValue(2)
    dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
    assert dialog.result()==QtWidgets.QDialog.DialogCode.Accepted
    assert w.apply_equipment(dialog.result_config)=='applied'
    assert w.sim is sim and w.sim.time==time and w.panels.value()==30
    np.testing.assert_array_equal(height,sim.terrain.height)
    assert w.log.toPlainText().startswith(logs)
    assert w.fleet_table.item(0,10).text()==f"{sim.drone.productivity.total_volume:.4f}"
    profile=w.profile_position.value();view=w.map.viewRange()
    c=deepcopy(w.config);c.fleet.models[0].equipment.bucket.volume_m3*=2
    assert w.apply_equipment(c)=='pending' and w.running
    w.toggle_play();w.toggle_play()
    assert sim.landing_requested  # pause/resume must preserve pending maintenance
    for _ in range(100):
        w.credit=4;w.on_frame()
        if sim.pending_equipment is None:break
    assert sim.pending_equipment is None and w.config is sim.config
    assert not w.running and w.config.fleet.models[0].payload==120
    assert w.profile_position.value()==profile and w.map.viewRange()==view
    stats=DailyStatsWindow(w);stats.timer.stop();stats.refresh()
    assert stats.table.rowCount()==1
    assert float(stats.table.item(0,2).text())==sim.drone.productivity.total_mass
    c=deepcopy(w.config);c.base.equipment.batteries=[];w.apply_equipment(c);w.refresh()
    assert 'Без аккумулятора' in w.telemetry.text()
    stats.close();dialog.close();w.close();app.processEvents()


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


def test_high_speeds_real_timer_fixed_physics_and_laser_display():
    from copy import deepcopy
    from PySide6 import QtWidgets,QtCore,QtTest
    from ats.gui import MainWindow
    from ats.simulation import Simulation
    from ats.equipment import resolve_config,LaserEmitter
    from ats.equipment_ui import EquipmentDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c=resolve_config(Config());c.base.equipment.emitters=[LaserEmitter()];c.fleet.models[0].equipment.receiver.installed=True
    w=MainWindow(c);w.show()
    assert [w.speed.itemData(i) for i in range(w.speed.count())][-2:]==[1000,2000]
    w.speed.setCurrentIndex(w.speed.findData(1000));assert w.speed.currentData()==1000
    w.speed.setCurrentIndex(w.speed.findData(2000));w.toggle_play()
    deadline=QtCore.QElapsedTimer();deadline.start()
    while w.sim.drone.laser_received==0 and deadline.elapsed()<3000:
        QtTest.QTest.qWait(20)
    w.running=False;w.timer.stop();w.refresh(True)
    assert w.sim.time>0 and w.config.dt==1
    reference=Simulation(deepcopy(c));reference.run(int(w.sim.time))
    assert w.sim.drone.battery==pytest.approx(reference.drone.battery,abs=1e-6)
    assert w.sim.base.battery==pytest.approx(reference.base.battery,abs=1e-6)
    assert w.sim.drone.laser_received>0
    assert 'Лазер: база' in w.telemetry.text()
    assert 'Получено за миссию' in w.fleet_table.item(0,11).toolTip()
    assert w.laser_beams.getData()[0] is not None
    dialog=EquipmentDialog(w.config,w);table=dialog.base.emitters.table
    table.item(0,5).setCheckState(QtCore.Qt.CheckState.Unchecked)
    dialog.save();assert not dialog.result_config.base.equipment.emitters[0].enabled
    dialog.close();w.close();app.processEvents()


def test_equipment_and_brush_signal_callbacks_allow_clean_process_exit():
    # CPython 3.11 detects PySide bool reference errors only during finalization;
    # checking a fresh interpreter's exit code catches failures after pytest.
    import subprocess,sys
    code='''
from PySide6 import QtWidgets,QtCore
from ats.gui import MainWindow
from ats.config import Config
from ats.equipment import LaserEmitter
from ats.equipment_ui import EquipmentDialog
app=QtWidgets.QApplication([])
w=MainWindow(Config());w.timer.stop()
w.config.base.equipment.emitters=[LaserEmitter()]
d=EquipmentDialog(w.config,w);item=d.base.emitters.table.item(0,5)
for _ in range(10):
    item.setCheckState(QtCore.Qt.CheckState.Unchecked)
    item.setCheckState(QtCore.Qt.CheckState.Checked)
button=next(b for b in w.plan_editor.findChildren(QtWidgets.QPushButton) if b.text()=='Посадить флот')
button.click();app.processEvents()
d.close();w.close()
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr.decode('utf-8',errors='replace')


def test_logistics_inventory_and_experiments_ui(tmp_path):
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.experiments import experiment_config,save_configuration,load_configuration
    from ats.logistics_ui import LogisticsWindow,ExperimentDialog
    from ats.simulation import Simulation
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c=experiment_config(Config(),drones=4,laser='realistic_reference')
    path=tmp_path/'scenario.json';save_configuration(path,c)
    clone=load_configuration(path)
    assert Simulation(c).run(50)==Simulation(clone).run(50)
    w=MainWindow(clone);w.timer.stop();w.sim.run(90);w.refresh(True)
    view=LogisticsWindow(w);view.timer.stop();view.tabs.setCurrentIndex(1);view.refresh()
    assert view.inventory.rowCount()==len(w.sim.logistics.inventory.items)
    view.tabs.setCurrentIndex(2)
    assert view.jobs.rowCount()>0 and 'Крот:' in view.status.text()
    view.tabs.setCurrentIndex(1)
    assert w.excavator_marker.data.size>0 and w.container_markers.data.size==8
    view.filter.setText('excavator_battery')
    assert view.inventory.rowCount()==8  # four modules and their four physical slots
    dialog=ExperimentDialog(w);dialog.mode.setCurrentIndex(1)
    assert dialog.candidate().logistics.direct_allowed
    dialog.idle_digging.setChecked(False)
    assert not dialog.candidate().logistics.idle_digging
    from ats.equipment_ui import EquipmentDialog
    gear=EquipmentDialog(w.config,w);gear.save()
    assert gear.result_config.base.equipment.emitters[0].profile=='realistic_reference'
    assert gear.result_config.fleet.models[0].equipment.receiver.optical_cap_kw==1.5
    gear.result_config.base.equipment.emitters[0].enabled=False
    assert w.sim.request_equipment(gear.result_config)=='applied'
    assert not w.sim.config.base.equipment.emitters[0].enabled
    gear.close()
    dialog.close();view.close();w.close();app.processEvents()


def test_inventory_refresh_does_not_walk_archive_and_dialog_is_reused():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.experiments import experiment_config
    from ats.logistics import TransportJob
    class Archive(list):
        def __iter__(self):
            raise AssertionError('GUI traversed the full task archive')
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(experiment_config(Config()));w.timer.stop()
    job=TransportJob('historic','DELIVER_EMPTY_CONTAINER',[],(0,0,0),(0,0,0),4,0,status='COMPLETED')
    w.sim.logistics.jobs=Archive([job]*100000)
    w.refresh();w.show_logistics();view=w.logistics_window
    view.tabs.setCurrentIndex(1);app.processEvents()
    assert view.inventory.rowCount()==len(w.sim.logistics.inventory.items)
    original=view.inventory.item(0,0)
    view.refresh()
    assert view.inventory.item(0,0) is original
    view.tabs.setCurrentIndex(2);app.processEvents()
    assert view.jobs.rowCount()==200
    view.close();assert not view.timer.isActive()
    w.show_logistics();assert w.logistics_window is view and view.timer.isActive()
    view.close();w.close();app.processEvents()
