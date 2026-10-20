import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pytest
from ats.config import Config
from ats.landscape import ElevationRaster,Landscape,workspace_config
from ats.plan import PlanConfig,PlanSegment


def test_region_overview_draws_real_corridor_and_opens_msl_workspace():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.landscape_ui import LandscapeDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(Config());w.timer.stop();d=LandscapeDialog(w)
    d.example_corridor()
    assert len(d.points)==2 and len(d.region.corridors)==1
    assert d.ground.getData()[0][-1]>100000
    assert 'млн м³' in d.result.text() and not d.region.tiles
    d.open_workspace();w.timer.stop()
    assert w.landscape is d.region and w.config.georeference['vertical_datum']=='MSL'
    assert w.plan_editor.height.suffix()==' м MSL'
    assert w.plan_editor.height.value()==-5 and w.config.plan.segments==[]
    assert w.latitude.value()==pytest.approx(w.config.base.latitude_deg,abs=.01)
    assert w.sim.terrain.remaining==0 and not w.landscape.tiles
    w.commit_landscape();assert not w.landscape.tiles
    d.close();w.close();app.processEvents()


def test_switching_regions_preserves_excavation_and_refuses_inflight_soil():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.landscape_ui import LandscapeDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    r=ElevationRaster(np.full((5,5),10.),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=100.,dy=100.))
    land=Landscape(r);c=workspace_config(land,Config(),[64.,64.],32.)
    c.plan=PlanConfig([PlanSegment([[15,16],[17,16]],height=9.75)])
    w=MainWindow(c);w.landscape=land;w.timer.stop()
    w.sim.run(30)
    with pytest.raises(ValueError,match='посадите'):w.commit_landscape()
    assert w.sim.run(10000)['state']=='COMPLETED'
    expected=w.sim.terrain.height.copy();w.commit_landscape()
    assert land.tiles and abs(land.report()['soil_delta_m3'])<1e-9
    dialog=LandscapeDialog(w);dialog.side.setCurrentIndex(0)
    dialog.x.setValue(128);dialog.y.setValue(128);dialog.open_workspace();w.timer.stop()
    assert w.config.georeference['origin_m']==[128.,128.]
    assert w.landscape is land
    np.testing.assert_array_equal(land.read_window([64.,64.],128)[1],expected)
    dialog.close();w.close();app.processEvents()
