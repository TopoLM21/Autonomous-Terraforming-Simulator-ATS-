"""The separate conventional site never buys or simulates autonomous machinery."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from ats.config import Config
from ats.maps import with_map_size
from ats.plan import PlanConfig,PlanSegment


def workspace_config():
    c=with_map_size(Config(),128)
    c.terrain.cell_size=.5;c.terrain.n=256
    c.terrain.dump_rect=(80.,80.,40.,40.)
    c.plan=PlanConfig([PlanSegment([[40.,50.],[90.,50.]],width=8.,height=-.3)])
    c.start_hour=6.
    return c


def app_and_window():
    from PySide6 import QtWidgets
    from ats.conventional_workspace import ConventionalWorkWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window=ConventionalWorkWindow(workspace_config());window.timer.stop()
    return app,window


def test_empty_conventional_window_has_no_autonomous_purchase_or_clock_advance():
    app,w=app_and_window()
    try:
        assert not w.sim.drones and w.sim.logistics is None
        assert getattr(w.sim,'conventional',None) is None
        assert not w.play.isEnabled() and not w.step_button.isEnabled()
        assert len(w.machines.points())==0 and len(w.camp_marker.points())==0
        before=w.sim.time;w.toggle_play();w.single_step();w.tick()
        assert w.sim.time==before and not w.running
        assert 'отсутствует' in w.overview.text()
        w.show_conventional();w.equipment_window.timer.stop()
        assert w.equipment_window.isVisible()
        assert getattr(w.sim,'conventional',None) is None
    finally:w.close();app.processEvents()


def test_conventional_window_renders_physical_machines_and_exact_snapshot(tmp_path):
    from ats.conventional import ConventionalConfig
    from ats.checkpoint import load_state
    from ats.map_visuals import terrain_rgb
    from ats.site_ground import marine_mask
    app,w=app_and_window()
    try:
        c=ConventionalConfig(base_x=12.,base_y=12.,work_rect=(35.,40.,60.,20.),truck_count=2)
        w.install_conventional(c)
        assert w.play.isEnabled() and not w.sim.drones
        for _ in range(30):w.single_step()
        r=w.sim.conventional.report()
        assert len(r['machines'])==3
        actual=np.asarray([[point.pos().x(),point.pos().y()] for point in w.machines.points()])
        np.testing.assert_allclose(actual,np.asarray([m['position_m'][:2] for m in r['machines']]))
        assert 'Экскаватор' in w.details.text() and 'Самосвал' in w.details.text()
        assert w.sim.time==30.
        state=tmp_path/'traditional.atsstate';assert w.save_session(state)
        restored=load_state(state).simulation
        assert restored.report()==w.sim.report()
        for _ in range(25):w.sim.step();restored.step()
        assert restored.report()==w.sim.report()
        assert w.load_session(state) and w.sim.time==30.
        assert not w.running and not w.sim.drones and len(w.machines.points())==3
        assert abs(w.sim.report()['mass_error_kg'])<1e-7
        # Repaint the saved site after a local cut, then an edit touching the
        # raster edge. Relief shading at neighbours must match a full render.
        terrain=w.sim.terrain
        assert terrain.scoop(45.,50.,50.,source='conventional',radius=.5)>0.
        for edge in (False,True):
            if edge:
                terrain.height[0,:2]+=.15
                terrain.revision+=1
            w.refresh()
            expected=terrain_rgb(terrain.height,terrain.cell,w.config.trench.depth,
                w.config.terrain.dump_max_height,reference=terrain.reference,marine=marine_mask(terrain))
            np.testing.assert_array_equal(w.image.image,expected)
    finally:w.close();app.processEvents()


def test_invalid_load_keeps_the_current_physical_worksite(tmp_path):
    app,w=app_and_window()
    try:
        before=w.sim;path=tmp_path/'bad.atsstate';path.write_bytes(b'broken')
        assert not w.load_session(path)
        assert w.sim is before and 'Ошибка загрузки' in w.statusBar().currentMessage()
    finally:w.close();app.processEvents()
