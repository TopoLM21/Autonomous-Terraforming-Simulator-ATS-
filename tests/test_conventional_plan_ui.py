"""The work-plan controls submit a policy without resetting physical work."""
import gc
import os
import numpy as np
import pytest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6 import QtCore,QtWidgets

_APP=None


@pytest.fixture
def host():
    global _APP
    from ats.simulation import Simulation
    from tests.test_conventional_workspace import workspace_config
    _APP=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gc.collect()
    class Host(QtWidgets.QWidget):
        landscape=None
        def __init__(self):
            super().__init__();self.sim=Simulation.conventional_only(workspace_config())
        def refresh(self,*args):pass
        def install_conventional(self,config):return self.sim.install_conventional(config)
    value=Host()
    yield value
    value.close();value.deleteLater()
    _APP.sendPostedEvents(None,QtCore.QEvent.Type.DeferredDelete);gc.collect()


def test_new_fleet_policy_is_selected_before_purchase(host):
    from ats.conventional_ui import ConventionalWindow
    dialog=ConventionalWindow(host);dialog.timer.stop()
    try:
        assert dialog.config_value().dump_order=='far_to_near'
        assert not dialog.live_dump_order.isEnabled()
        assert not dialog.plan_apply_button.isEnabled()
        assert 'Конечный профиль' in dialog.plan_summary.text()
        dialog.controls['dump_order'].setCurrentIndex(dialog.controls['dump_order'].findData('nearest'))
        assert dialog.config_value().dump_order=='nearest'
        assert host.sim.conventional is None and host.sim.time==0.
    finally:dialog.close();dialog.deleteLater()


def test_live_order_applies_only_after_explicit_command_and_preserves_trip(host,tmp_path):
    from ats.conventional import ConventionalConfig
    from ats.conventional_ui import ConventionalWindow
    from ats.checkpoint import save_state,load_state
    fleet=host.install_conventional(ConventionalConfig(base_x=12.,base_y=12.,work_rect=(35.,40.,60.,20.)))
    truck=fleet.machines[1]
    truck.state='DRIVING';truck.cargo_kg=500.;truck.path=[np.array([20.,20.])]
    truck.target=(90.,90.)
    before=(truck.state,truck.cargo_kg,truck.path[0].copy(),truck.target,host.sim.time)
    dialog=ConventionalWindow(host);dialog.timer.stop()
    try:
        assert 'полос съезда' in dialog.plan_summary.text()
        assert 'точек' in dialog.plan_summary.text()
        assert 'от дальнего края' in dialog.plan_summary.text()
        assert not dialog.plan_apply_button.isEnabled()
        dialog.live_dump_order.setCurrentIndex(dialog.live_dump_order.findData('nearest'))
        dialog.refresh()
        assert fleet.config.dump_order=='far_to_near' and dialog.plan_apply_button.isEnabled()
        dialog.plan_apply_button.click()
        assert fleet.config.dump_order=='nearest'
        assert not dialog.plan_apply_button.isEnabled()
        assert 'Текущие маршруты, грузы' in dialog.message.text()
        assert (truck.state,truck.cargo_kg,truck.target,host.sim.time)==(before[0],before[1],before[3],before[4])
        np.testing.assert_array_equal(truck.path[0],before[2])
        # Use an unloaded valid fleet for snapshot validation; UI must restore
        # the saved policy rather than its new-fleet default.
        truck.state='PARKED';truck.cargo_kg=0.;truck.path=[];truck.target=None
        state=tmp_path/'plan.atsstate';save_state(state,host.sim)
        host.sim=load_state(state).simulation
        dialog.refresh()
        assert dialog.live_dump_order.currentData()=='nearest'
        assert dialog.controls['dump_order'].currentData()=='nearest'
        assert not dialog.plan_apply_button.isEnabled()
    finally:dialog.close();dialog.deleteLater()
