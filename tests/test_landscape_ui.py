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
    assert 50000<d.ground.getData()[0][-1]<100000
    assert 'млн м³' in d.result.text() and not d.region.tiles
    d.side.setCurrentIndex(1)  # the separate manual workspace stays a bounded test
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


def test_channel_start_uses_drawn_route_dimensions_and_ready_excavation_plan():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.landscape_ui import LandscapeDialog
    from tests.test_channel import coastal_dem
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(Config());w.timer.stop();d=LandscapeDialog(w)
    d.region=Landscape(coastal_dem());d.show_region();d.points=[[50,150],[150,150]]
    d.width.setValue(1);d.depth.setValue(.05)
    d.project_side.setCurrentIndex(d.project_side.findData(64))
    assert d.prepare_channel()
    d.start_button.click();w.timer.stop()
    assert w.running and w.sim.time==0 and w.config.plan.segments
    assert w.sim.terrain.remaining_cut>0
    assert w.config.georeference['channel']['depth_m']==.05
    assert w.plan_editor.height.value()==-.05 and w.plan_editor.width.value()==1
    assert w.sim.run(20000)['state']=='COMPLETED'
    w.commit_landscape()
    assert w.landscape.channel_project['completed']==[0]
    assert len(w.landscape.channel_project['history'])==1
    assert abs(w.landscape.report()['soil_delta_m3'])<1e-9
    w.running=False;d.close();w.close();app.processEvents()


def test_geographic_latitude_is_readonly_and_follows_actual_site_in_gui():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    r=ElevationRaster(np.full((3,3),0.),dict(crs='EPSG:4326',units='degrees',vertical_datum='MSL',x0=27.,y0=30.,dx=.02,dy=.02))
    c=workspace_config(Landscape(r),Config(),[100.,100.],32.)
    w=MainWindow(c);w.timer.stop()
    assert not w.latitude.isEnabled()
    assert w.latitude.value()==pytest.approx(c.base.latitude_deg,abs=.0001)
    assert c.base.latitude_deg>30 and c.base.latitude_deg<30.01
    w.close();app.processEvents()


def test_next_section_keeps_landscape_and_completed_section_history():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.landscape_ui import LandscapeDialog
    from tests.test_channel import coastal_dem
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(Config());w.timer.stop();d=LandscapeDialog(w)
    land=Landscape(coastal_dem(gap=50));d.region=land;d.show_region()
    d.points=[[50,150],[200,150]];d.width.setValue(1);d.depth.setValue(.01)
    d.project_side.setCurrentIndex(d.project_side.findData(64));d.start_channel();w.timer.stop()
    assert w.config.georeference['channel']['sections']==2
    assert w.sim.run(30000)['state']=='COMPLETED'
    mass=w.sim.excavated_mass;w.running=False
    next_dialog=LandscapeDialog(w);next_dialog.next_section();w.timer.stop()
    assert w.landscape is land and land.channel_project['completed']==[0]
    assert w.config.georeference['channel']['section']==1
    assert land.channel_project['history'][0]['excavated_m3']==pytest.approx(mass/1600)
    assert abs(land.report()['soil_delta_m3'])<1e-9
    w.running=False;next_dialog.close();d.close();w.close();app.processEvents()
