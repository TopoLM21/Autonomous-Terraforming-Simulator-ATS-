"""Explicit adjacent-window loading preserves resolution and caller drafts."""
import os
import gc
import subprocess
import sys
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from PySide6 import QtCore,QtWidgets
from ats.active_expansion_ui import ActiveExpansionDialog,expansion_proposal,_translated_ui


_APP=None


@pytest.fixture
def app():
    # Keep Qt alive and collect closed mock dialogs on the GUI thread.
    global _APP
    _APP=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gc.collect()
    yield _APP
    _APP.sendPostedEvents(None,QtCore.QEvent.Type.DeferredDelete)
    gc.collect()


def context(origin=(100.,100.),cell=.5,n=128,extent=(1000.,1000.)):
    terrain=SimpleNamespace(height=np.zeros((n,n)),reference=np.zeros((n,n)),cell=cell)
    sim=SimpleNamespace(terrain=terrain,config=SimpleNamespace(georeference={'origin_m':list(origin)}),landing_requested=False)
    sim.requests=0
    def land():sim.landing_requested=True;sim.requests+=1
    sim.request_landing=land
    land=SimpleNamespace(raster=SimpleNamespace(extent_m=extent))
    return sim,land


def test_centered_proposal_contains_current_window_on_same_nonzero_grid():
    sim,land=context(origin=(100.25,100.25))
    before=deepcopy(sim.config.georeference)
    p=expansion_proposal(sim,land,128.)
    assert p['origin_m']==[68.25,68.25] and p['offset_m']==[32.,32.]
    assert p['n']==256 and p['cell_m']==.5
    assert p['added_area_m2']==128**2-64**2 and p['dense_bytes']>=64*256**2
    assert sim.config.georeference==before and sim.terrain.height.shape==(128,128)


def test_proposal_clips_at_dem_edges_without_coarsening():
    sim,land=context(origin=(0.,936.))
    p=expansion_proposal(sim,land,128.)
    assert p['origin_m']==[0.,872.] and p['offset_m']==[0.,64.]
    with pytest.raises(ValueError,match='разрешение'):expansion_proposal(sim,land,2048.)
    with pytest.raises(ValueError,match='кратен'):expansion_proposal(sim,land,128.1)
    with pytest.raises(ValueError,match='региональный'):expansion_proposal(sim,None,128.)


class Host(QtWidgets.QWidget):
    def __init__(self,error=None):
        super().__init__();self.sim,self.landscape=context();self.running=True
        self._session_path='original.atsstate';self.installed=[];self.restored=None;self.calls=[]
        self.ui=dict(brush_segments=[dict(points=[[3.,4.],[5.,6.]])],brush_dirty=True,
                     view_range=[[0.,64.],[0.,64.]],speed=200,day_night=False)
        self.project=SimpleNamespace(shared_site=object(),units=[SimpleNamespace(simulation=self.sim)],
                                     active=SimpleNamespace(simulation=self.sim),landscape=self.landscape)
        def enlarge(side,**kw):
            self.calls.append((side,kw))
            if error:raise ValueError(error)
            self.sim.config.georeference['origin_m']=kw['origin_m']
        self.site=SimpleNamespace(enlarge=enlarge)
        self.project.enable_shared=lambda:self.site
        self.complex_project=self.project
    def install_simulation(self,*args):
        self.installed.append(args);self.running=False;self._session_path=None
    def session_ui(self):return deepcopy(self.ui)
    def restore_session_ui(self,ui):self.restored=ui
    def _after_file_action(self,running):self.running=running


def test_ui_success_calls_explicit_growth_preserves_path_running_and_dirty_draft(app):
    host=Host();dialog=ActiveExpansionDialog(host);dialog.side.setValue(128.)
    assert 'ячеек по 0.5' in dialog.preview.text() and 'временные буферы' in dialog.preview.text()
    dialog.apply_button.click()
    assert dialog.result()==QtWidgets.QDialog.DialogCode.Accepted
    assert host.calls==[(128.,dict(origin_m=[68.,68.],preserve_resolution=True))]
    assert host.running and host._session_path=='original.atsstate'
    assert host.restored['brush_dirty'] and host.restored['brush_segments'][0]['points']==[[35.,36.],[37.,38.]]
    assert host.restored['view_range']==[[32.,96.],[32.,96.]]
    assert host.ui['brush_segments'][0]['points']==[[3.,4.],[5.,6.]]
    assert dialog.offset_m==(32.,32.) and len(host.installed)==1
    host.close();dialog.close();app.processEvents()


def test_ui_refusal_stays_open_and_landing_is_an_explicit_time_advancing_request(app):
    host=Host(error='Для смены начала окна посадите все комплексы и завершите грузовые операции')
    dialog=ActiveExpansionDialog(host);dialog.side.setValue(128.);dialog.show()
    before=deepcopy(host.sim.config.georeference)
    dialog.apply_button.click()
    assert dialog.isVisible() and not host.installed and host.running
    assert 'посадите' in dialog.message.text() and host.sim.config.georeference==before
    assert host.sim.requests==0
    dialog.landing_button.click()
    assert host.sim.requests==1 and host.sim.landing_requested and host.running
    assert 'дождитесь' in dialog.message.text()
    host.close();dialog.close();app.processEvents()


def test_ui_no_growth_available_is_explained_and_disabled(app):
    host=Host();host.sim,host.landscape=context(origin=(0.,0.),cell=1.,n=64,extent=(64.,64.))
    dialog=ActiveExpansionDialog(host)
    assert not dialog.apply_button.isEnabled() and 'больше текущего' in dialog.preview.text()
    host.close();dialog.close();app.processEvents()


def test_translation_accepts_real_plan_segments_without_touching_input():
    from ats.plan import PlanSegment
    segment=PlanSegment([[1.,2.],[3.,4.]])
    result=_translated_ui(dict(brush_segments=[segment]),(10.,20.))
    assert result['brush_segments'][0].points==[[11.,22.],[13.,24.]]
    assert segment.points==[[1.,2.],[3.,4.]]


def test_first_shared_dispatcher_is_rolled_back_on_refused_growth(app,monkeypatch):
    from ats import complexes
    host=Host();host.complex_project=None
    host.sim.config.terrain=SimpleNamespace(completion_volume=.02)
    original_terrain=host.sim.terrain
    original_keys=set(vars(host.sim))
    class Project:
        shared_site=None
        def __init__(self,sim,landscape):
            self.units=[SimpleNamespace(simulation=sim,committed=True)];self.templates=[]
        def enable_shared(self):
            sim=self.units[0].simulation
            sim.shared_site=self;sim.terrain=SimpleNamespace(core=original_terrain)
            sim.config.terrain.completion_volume=.001
            def refused(*args,**kwargs):raise ValueError('Есть груз в незавершённой операции')
            self.shared_site=SimpleNamespace(enlarge=refused)
            return self.shared_site
    monkeypatch.setattr(complexes,'ComplexProject',Project)
    dialog=ActiveExpansionDialog(host);dialog.side.setValue(128.);dialog.apply_button.click()
    assert host.complex_project is None and not host.installed
    assert host.sim.terrain is original_terrain and set(vars(host.sim))==original_keys
    assert host.sim.config.terrain.completion_volume==.02
    assert 'Есть груз' in dialog.message.text() and host.running
    host.close();dialog.close();app.processEvents()


def regional_config():
    from ats.config import Config
    from ats.landscape import Landscape,ElevationRaster,workspace_config
    from ats.plan import PlanConfig,PlanSegment
    raster=ElevationRaster(np.full((5,5),10.),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',
        x0=0.,y0=0.,dx=100.,dy=100.))
    land=Landscape(raster);config=workspace_config(land,Config(),[128.,128.],32.)
    config.plan=PlanConfig([PlanSegment([[14.,16.],[18.,16.]],height=9.75)])
    return config,land


def _probe_main_window_growth(app):
    from ats.gui import MainWindow
    config,land=regional_config();host=MainWindow(config);host.landscape=land;host.timer.stop()
    host._session_path='kept.atsstate';host.running=False
    sim=host.sim;position=sim.base.position.copy();drones=[d.position.copy() for d in sim.drones]
    ground=sim.terrain.height.copy();energy=sim.report()['initial_energy_kwh']
    dialog=ActiveExpansionDialog(host);dialog.side.setValue(64.);dialog.apply_button.click();host.timer.stop()
    assert dialog.result()==QtWidgets.QDialog.DialogCode.Accepted,dialog.message.text()
    assert host.sim is sim and host._session_path=='kept.atsstate' and not host.running
    assert host.complex_project.shared_site is sim.shared_site
    assert sim.config.georeference['origin_m']==[112.,112.] and sim.terrain.cell==.25
    np.testing.assert_array_equal(sim.terrain.height[64:192,64:192],ground)
    np.testing.assert_array_equal(sim.base.position,position+[16.,16.,0.])
    for drone,before in zip(sim.drones,drones):np.testing.assert_array_equal(drone.position,before+[16.,16.,0.])
    assert sim.time==0. and sim.report()['initial_energy_kwh']==energy
    host.close();dialog.close();app.processEvents()


def _probe_conventional_window_growth(app,tmp_path):
    from ats.conventional_workspace import ConventionalWorkWindow
    from ats.conventional import ConventionalConfig
    config,land=regional_config();host=ConventionalWorkWindow(config,landscape=land);host.timer.stop()
    dialog=ActiveExpansionDialog(host);dialog.side.setValue(64.);dialog.apply_button.click();host.timer.stop()
    assert dialog.result()==QtWidgets.QDialog.DialogCode.Accepted,dialog.message.text()
    assert host.complex_project.shared_site is host.sim.shared_site
    assert host.config.georeference['origin_m']==[112.,112.]
    assert not host.sim.drones and host.sim.time==host.complex_project.time==0.
    fleet_config=ConventionalConfig.for_simulation(host.sim);fleet_config.enabled=False
    host.install_conventional(fleet_config);host.single_step()
    assert host.sim.time==host.complex_project.time==1.
    path=tmp_path/'expanded-empty-ordinary.atsstate'
    assert host.save_session(path)
    assert host.load_session(path)
    assert host.complex_project.shared_site.core.height.shape==(256,256)
    assert host.sim.config.georeference['origin_m']==[112.,112.]
    assert host.sim.time==host.complex_project.time==1.
    host.single_step();assert host.sim.time==host.complex_project.time==2.
    host.close();dialog.close();app.processEvents()


def _window_process(kind,directory=None):
    # Qt 6.11/Python 3.11 can corrupt native view ownership when the complete
    # main-window and standalone save/load trees are collected in one pytest
    # process. Exercise each real entry point with its own QApplication life
    # cycle; all physical, snapshot and widget assertions still run below.
    code='''
import sys
from pathlib import Path
from PySide6 import QtWidgets
from tests.test_active_expansion_ui import _probe_main_window_growth,_probe_conventional_window_growth
app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
if sys.argv[1]=='main':_probe_main_window_growth(app)
else:_probe_conventional_window_growth(app,Path(sys.argv[2]))
'''
    result=subprocess.run([sys.executable,'-c',code,kind,str(directory or '')],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ,'QT_QPA_PLATFORM':'offscreen','PYTHONIOENCODING':'utf-8'},
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
    assert result.returncode==0,result.stdout+'\n'+result.stderr


def test_runtime_main_window_growth_preserves_physical_actors_clock_and_file():
    _window_process('main')


def test_runtime_conventional_only_window_can_expand_and_keep_one_shared_clock(tmp_path):
    _window_process('conventional',tmp_path)
