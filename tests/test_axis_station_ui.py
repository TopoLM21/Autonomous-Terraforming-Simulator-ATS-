"""The real coastal GUI launches the complete self-deploying axis formation."""
from copy import deepcopy
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from ats.checkpoint import Session,save_state,load_state
from ats.canal_station import work_exclusion
from tests.test_coastal_compact import seasonal_config


def test_default_coastal_ui_axis_choice_full_formation_and_legacy_choice(seasonal_config,tmp_path):
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.landscape_ui import LandscapeDialog
    from ats.winter_power import ready_design
    from ats.channel_site import equipment_fits
    from ats.workfront import coordinates
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    initial,_=ready_design(deepcopy(seasonal_config),seasonal_config.energy_design)
    w=MainWindow(initial.config,session=Session(initial,ui={}));w.timer.stop()
    dialog=LandscapeDialog(w)
    try:
        assert dialog.base_strategy.currentData()=='axis_ahead'
        dialog.example_corridor();assert dialog.project.base_strategy=='axis_ahead'
        dialog.start_button.click();w.timer.stop();s=w.sim;t=s.terrain
        assert s.state.value=='RUNNING',s.reason
        assert s.base.layout_spec['platform_count']==45 and t.front.axis
        assert equipment_fits(s.config,s.base.layout_spec,s.base.position[:2],t)
        _,along,signed=coordinates(s.config.plan,np.array(s.base.position[0]),np.array(s.base.position[1]))
        assert abs(float(signed))<1e-6 and float(along)>t.front.global_sector_end
        ix,iy=(s.base.position[:2]/t.cell).astype(int)
        assert t.design_work[iy,ix] and not work_exclusion(t)[iy,ix]
        original=s.base.position.copy()
        for _ in range(240):
            s.step();assert s.state.value=='RUNNING',s.reason
        np.testing.assert_array_equal(s.base.position,original)
        assert s.logistics.solar_array.ready
        assert all(d.last_wait!='BLOCKED_STATIC_OBSTACLE' for d in s.drones)
        path=tmp_path/'coastal-axis.atsstate';save_state(path,s,landscape=w.landscape)
        q=load_state(path).simulation
        for _ in range(3):s.step();q.step()
        assert s.report()==q.report()
        assert abs(s.report()['mass_error_kg'])<1e-7 and abs(s.report()['energy_error_j'])<.002
        w.running=False;w.show();w.refresh(True);app.processEvents()
        assert 'база на оси' in w.channel_status.text()
        assert w.grab().save(str(tmp_path/'coastal-axis.png'))
        # Loading an old project must display the actual bank strategy.
        old=deepcopy(dialog.project.to_data());old['version']=2;old.pop('base_strategy')
        w.landscape.channel_project=old
        other=LandscapeDialog(w)
        try:assert other.base_strategy.currentData()=='bank'
        finally:other.close()
    finally:
        w.running=False;w.timer.stop();dialog.close();w.close();app.processEvents()
