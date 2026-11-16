"""Both equipment families are delivered into one real Qt worksite."""
import os
import time
import gc
import pytest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6 import QtCore,QtWidgets
from ats.maps import with_map_size
from ats.conventional import ConventionalConfig
from ats.conventional_workspace import ConventionalWorkWindow
from tests.test_conventional_workspace import workspace_config


@pytest.fixture(autouse=True)
def gui_lifetime():
    # Other UI cases close top-level windows without deleting their Python
    # cycles. Reclaim these on the GUI thread before starting a real worker,
    # and keep QApplication alive until that worker and queued events finish.
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gc.collect();app.sendPostedEvents(None,QtCore.QEvent.Type.DeferredDelete)
    yield app
    assert QtCore.QThreadPool.globalInstance().waitForDone(30000)
    app.sendPostedEvents(None,QtCore.QEvent.Type.DeferredDelete);gc.collect()


def test_ground_first_composer_delivers_drones_on_the_same_existing_map():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c=with_map_size(workspace_config(),256)
    w=ConventionalWorkWindow(c);w.timer.stop();w.show()
    try:
        fleet=w.sim.install_conventional(ConventionalConfig(base_x=12.,base_y=12.,work_rect=(35.,40.,20.,20.)))
        old_terrain=w.sim.terrain;stock=fleet.report()['fuel_stock_l'];old_ids=[m.id for m in fleet.machines]
        w.autonomous_button.click();app.processEvents();d=w.setup_window
        assert d.isVisible() and not d.source.conventional_only and d.source.fleet.count==4
        d.drones.setValue(2);d.name.setText('Новые копатели')
        d.period.setCurrentIndex(d.period.findData('custom'))
        date=QtCore.QDate(2025,1,1).addDays(171);d.first.setDate(date);d.last.setDate(date)
        d.mode.setCurrentIndex(d.mode.findData('hours'));d.hours.setValue(1.)
        d.auto_place.setChecked(False);d.x.setValue(180.);d.y.setValue(180.);d.start_after.setChecked(False)
        d.calculate_button.click();deadline=time.monotonic()+30
        while d.job is not None and time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
        assert d.job is None and d.result is not None,d.status.text()
        d.place_button.click();app.processEvents()
        assert hasattr(w,'mixed_window'),d.status.text()
        mixed=w.mixed_window;mixed.timer.stop()
        assert mixed.isVisible() and not mixed.running and not w.isVisible()
        assert mixed.sim.shared_site.core is old_terrain and len(mixed.sim.drones)==2
        assert mixed.sim.conventional is fleet and fleet.sim is mixed.sim and w.sim.conventional is None
        assert fleet.report()['fuel_stock_l']==stock and [m.id for m in fleet.machines]==old_ids
        assert mixed.sim.time==0 and mixed.complex_project.time==0
        mixed.show_conventional();mixed.conventional_window.timer.stop()
        assert not mixed.conventional_window.create_button.isEnabled()
        assert mixed.sim.report()['conventional']['capital_cost_usd']==225000
    finally:
        if hasattr(w,'mixed_window'):w.mixed_window.close()
        w.close();app.processEvents()


def test_refused_ground_purchase_keeps_the_original_autonomous_dispatcher():
    from ats.gui import MainWindow
    from ats.config import Config
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window=MainWindow(Config());window.timer.stop()
    try:
        terrain=window.sim.terrain;before=window.sim.report()
        assert window.complex_project is None and window.sim.shared_site is None
        with pytest.raises(ValueError):
            window.install_conventional(ConventionalConfig(base_x=1.,base_y=1.,work_rect=(13.,13.,8.,8.),truck_count=1))
        assert window.complex_project is None and window.sim.shared_site is None
        assert window.sim.terrain is terrain and window.sim.conventional is None
        assert window.sim.report()==before
    finally:window.close();app.processEvents()
