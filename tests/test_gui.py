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
