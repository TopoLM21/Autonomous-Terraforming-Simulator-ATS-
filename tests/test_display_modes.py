"""Display detail and rendering rate must not change the physical experiment."""
import os
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.simulation import Simulation


def test_display_crop_averages_large_pixels_and_restores_cells_when_zoomed():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from ats.gui import terrain_view
    height=np.arange(1024*1024,dtype=float).reshape(1024,1024)
    original=height.copy()
    view,rect,stride=terrain_view(height,.5,[[0,512],[0,512]])
    assert stride==2 and view.shape==(512,512) and rect==(0,0,512,512)
    assert view[0,0]==np.mean(height[:2,:2])
    detail,rect,stride=terrain_view(height,.5,[[100,120],[100,120]])
    assert stride==1 and np.shares_memory(detail,height)
    np.testing.assert_array_equal(height,original)


def test_no_map_mode_advances_the_same_simulation_and_does_not_rebuild_the_image(monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(experiment_config(Config()));w.timer.stop();reference=Simulation(experiment_config(Config()))
    w.render_mode.setCurrentIndex(w.render_mode.findData('off'))
    monkeypatch.setattr(w.image,'setImage',lambda *a,**k:pytest.fail('Disabled map rebuilt its image'))
    w.running=True
    for _ in range(50):w.sim.step();reference.step()
    w.refresh()
    assert w.sim.report()==reference.report() and w.map_frame.isHidden()
    assert w.sim.config.dt==1
    w.close();app.processEvents()


def test_loaded_wind_is_preserved_beyond_the_default_control_range(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.checkpoint import save_state
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s=Simulation(experiment_config(Config()));s.config.dust.wind_x=73.123456
    s.config.dust.wind_y=-81.5;s.run(10);before=s.report()
    path=tmp_path/'wind.atsstate';save_state(path,s)
    host=MainWindow(experiment_config(Config()));host.timer.stop();host.load_session(path);host.timer.stop()
    assert host.sim.config.dust.wind_x==73.123456 and host.sim.config.dust.wind_y==-81.5
    assert host.sim.report()==before
    host.close();app.processEvents()
