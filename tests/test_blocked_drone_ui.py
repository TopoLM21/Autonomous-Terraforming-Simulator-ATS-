"""A mixed site's running diesel fleet must not hide stopped drone work."""
import os
from pathlib import Path
import subprocess
import sys


def _probe_mixed_block_reason():
    from PySide6 import QtWidgets
    from ats.checkpoint import Session
    from ats.drone import DroneState
    from ats.gui import MainWindow
    from ats.simulation import MissionState
    from tests.test_conventional_shared import mixed
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    project,_,_=mixed()
    sim=project.active.simulation
    phases=[DroneState.SWAPPING_BATTERIES,DroneState.LANDING,DroneState.RETURN]
    for drone,phase in zip(sim.drones,phases):drone.state=phase
    reason='Траектория дрона 2 пересекает рельеф или базу <контроль>'
    sim.block(reason)
    # The conventional fleet intentionally keeps the shared project's clock
    # running; that must not turn the autonomous error into "Группа работает".
    assert project.running and project.blocked
    session=Session(sim,complex_project=project)
    window=MainWindow(sim.config,session=session);window.timer.stop()
    try:
        before=sim.report(full=False)
        window.labels_view.setChecked(True);window.refresh(True)
        assert 'заблокированные комплексы' in window.status.text()
        assert 'Комплекс 1' in window.status.text()
        assert 'Траектория дрона 2' in window.status.text()
        assert '&lt;контроль&gt;' in window.status.text()
        for row,phase in enumerate(phases):
            item=window.fleet_table.item(row,1)
            assert item.text()=='Остановлен · '+phase.value
            assert reason in item.toolTip() and 'не продвигается' in item.toolTip()
            label=window.drone_labels[row]
            assert 'Остановлен · '+phase.value in label.toPlainText()
            assert reason in label.toolTip()
        assert sim.report(full=False)==before

        # Selecting another healthy complex must retain the group's diagnostic,
        # without incorrectly marking that complex's aircraft as stopped.
        other=project.select('complex_2')
        window.install_simulation(other,complex_project=project);window.refresh(True)
        assert 'Траектория дрона 2' in window.status.text()
        assert 'Остановлен' not in window.fleet_table.item(0,1).text()
        assert not window.drone_labels[0].toolTip()

        sim.state=MissionState.RUNNING;sim.reason=''
        project.select('complex_1')
        window.install_simulation(sim,complex_project=project);window.refresh(True)
        assert 'заблокированные комплексы' not in window.status.text()
        assert window.fleet_table.item(0,1).text()==phases[0].value
        assert reason not in window.fleet_table.item(0,1).toolTip()
        assert not window.drone_labels[0].toolTip()
    finally:
        window.close();app.processEvents()


def test_mixed_block_reason_remains_visible_for_stopped_drone_operations():
    # Keep this complete MainWindow tree in its own QApplication lifetime.
    # Qt 6.11/Python 3.11 may otherwise collect its native plot objects during
    # a later, unrelated GUI test. The child executes every real assertion.
    code='from tests.test_blocked_drone_ui import _probe_mixed_block_reason; _probe_mixed_block_reason()'
    result=subprocess.run([sys.executable,'-c',code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ,'QT_QPA_PLATFORM':'offscreen','PYTHONIOENCODING':'utf-8'},
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
