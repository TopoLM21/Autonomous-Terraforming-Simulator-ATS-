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
    window.close()
