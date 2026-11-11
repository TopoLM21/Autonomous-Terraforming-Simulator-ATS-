"""The real water dialog also works before a physical complex is delivered."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
import numpy as np
import pytest
from PySide6 import QtWidgets
from ats.config import Config
from ats.landscape import Landscape,workspace_config
from ats.landscape_ui import LandscapeDialog
from ats.complex_setup_ui import WorkspaceWindow
from ats.hydrology_ui import FloodingDialog
from tests.test_channel import coastal_dem


@pytest.fixture
def app():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    for widget in app.topLevelWidgets():widget.close()
    app.processEvents()


def channel(owner):
    owner.points=[[50,150],[250,150]];owner.width.setValue(2);owner.depth.setValue(1)
    owner.project_side.setCurrentIndex(owner.project_side.findData(128))
    assert owner.prepare_channel(),owner.result.text()


@pytest.mark.parametrize('selected_window',[False,True])
def test_public_open_calculate_save_and_reload_without_vehicles(app,monkeypatch,tmp_path,selected_window):
    from ats.simulation import Simulation
    from ats.inventory import Inventory
    def no_hardware(*a,**k):raise AssertionError('Water preview must not create physical equipment')
    monkeypatch.setattr(Simulation,'__init__',no_hardware);monkeypatch.setattr(Inventory,'__init__',no_hardware)
    land=Landscape(coastal_dem(gap=100),cell_size=.5)
    c=workspace_config(land,Config(),[80.,70.],128) if selected_window else Config()
    host=WorkspaceWindow(c,land if selected_window else None);owner=LandscapeDialog(host)
    owner.region=land;owner.show_region();before=host.preview.height.copy()
    owner.open_flooding()  # no route yet: useful validation, no empty modal
    assert owner.result.text() and land.flooding is None
    channel(owner)
    if selected_window:
        def no_preview_commit(*a,**k):raise AssertionError('Planning terrain must not enter the earthworks ledger')
        monkeypatch.setattr(land,'commit_window',no_preview_commit)
    destination=tmp_path/'water-only.atsregion'
    monkeypatch.setattr(QtWidgets.QFileDialog,'getSaveFileName',lambda *a,**k:(str(destination),''))
    captured=[]
    def use_dialog(dialog):
        dialog.timer.stop();dialog.show();app.processEvents()
        assert host.sim is None and not hasattr(host,'play')
        assert dialog.view() is land
        dialog.inputs['evaporation_mm_day'].setValue(0.)
        dialog.instant.click();dialog.day.click()
        assert dialog.model.volume_m3>0 and dialog.model.elapsed_seconds==86400.
        assert abs(dialog.model.balance_error)<1e-6 and abs(dialog.model.salt_balance_error)<1e-6
        assert dialog.water_image.image is not None
        dialog.save_button.click();assert destination.exists(),dialog.error.text()
        captured.append(deepcopy(dialog.model.report()));dialog.close();return 0
    monkeypatch.setattr(FloodingDialog,'exec',use_dialog)
    owner.open_flooding()
    assert len(captured)==1 and host.sim is None and host.complex_project is None
    assert not host.running and not hasattr(host,'credit') and not hasattr(host,'last_wall')
    np.testing.assert_array_equal(host.preview.height,before)
    assert not land.tiles and not land.soil_tiles
    loaded=Landscape.load(destination);assert loaded.flooding.report()==captured[0]
    # Reopening a persisted water scenario must keep volume, salt and its clock.
    other=WorkspaceWindow(Config(),loaded);reader=LandscapeDialog(other)
    def reopen(dialog):
        dialog.timer.stop();r=dialog.model.report()
        for key in ('volume_m3','elapsed_seconds','salt_mass_kg','inflow_m3','evaporated_m3'):
            assert r[key]==captured[0][key]
        dialog.close();return 0
    monkeypatch.setattr(FloodingDialog,'exec',reopen);reader.open_flooding()
    assert other.sim is None and not hasattr(other,'play')


def test_default_qattara_button_opens_water_before_building_any_complex(app,monkeypatch):
    from ats.simulation import Simulation
    def no_sim(*a,**k):raise AssertionError('An offline water calculation needs no earthworks simulation')
    monkeypatch.setattr(Simulation,'__init__',no_sim)
    host=WorkspaceWindow(Config());host.show_landscape();owner=host.landscape_window
    owner.example_corridor();opened=[]
    def inspect(dialog):
        dialog.timer.stop();assert dialog.model.curve.capacity>0
        assert 29<dialog.model.config.latitude<32 and not dialog.model.channel_open
        opened.append(dialog.model);dialog.close();return 0
    monkeypatch.setattr(FloodingDialog,'exec',inspect);owner.open_flooding()
    assert len(opened)==1 and host.sim is None and host.complex_project is None


@pytest.mark.parametrize('running',[False,True])
def test_live_georeferenced_earthworks_are_paused_and_resumed_without_changes(app,running):
    from ats.gui import MainWindow
    from ats.experiments import experiment_config
    from ats.plan import PlanConfig,PlanSegment
    from ats.soil import SoilProfile
    land=Landscape(coastal_dem(gap=100),cell_size=.5,soil_profile=SoilProfile.qattara())
    c=workspace_config(land,experiment_config(Config(),drones=2),[80.,70.],128)
    c.plan=PlanConfig([PlanSegment([[30,40],[50,40]],width=2,height=-.1)])
    host=MainWindow(c);host.timer.stop();host.landscape=land
    terrain=host.sim.terrain
    load=terrain.scoop_load(35.,40.,10.,radius=1.)
    point=terrain.dump_location(load.volume);assert point is not None
    terrain.deposit(*point,load.mass,load=load)
    host.running=running;owner=LandscapeDialog(host);channel(owner)
    before=host.sim.report();height=terrain.height.copy();water=FloodingDialog(owner)
    water.timer.stop();assert not host.running
    view=water.view();assert view is not land and view.tiles and not land.tiles
    water.inputs['evaporation_mm_day'].setValue(0.);water.instant.click();water.day.click()
    water.close();water.finish(0)  # both close paths must resume at most once
    assert host.running is running and host.sim.report()==before
    np.testing.assert_array_equal(terrain.height,height)
    assert not land.tiles and not land.soil_tiles
    host.running=False


@pytest.mark.parametrize('live',[False,True])
def test_failed_construction_closes_cleanly_and_preserves_host_playback(app,monkeypatch,live):
    import ats.hydrology_ui as module
    from ats.gui import MainWindow
    host=MainWindow(Config()) if live else WorkspaceWindow(Config())
    if live:
        host.timer.stop();host.running=True;host.credit=17.
        stamp=host.last_wall;text=host.play.text()
    owner=LandscapeDialog(host);owner.region=Landscape(coastal_dem(gap=100));owner.show_region();channel(owner)
    def unreadable(*a,**k):raise ValueError('Не удалось прочитать рельеф впадины')
    monkeypatch.setattr(module,'create_experiment',unreadable)
    owner.open_flooding()
    assert 'Не удалось прочитать' in owner.result.text() and owner.region.flooding is None
    # A partially constructed native dialog still receives close/done events.
    for dialog in owner.findChildren(FloodingDialog):dialog.close()
    app.processEvents()
    if live:
        assert host.running and host.credit==17. and host.last_wall==stamp and host.play.text()==text
        host.running=False
    else:
        assert not host.running and not hasattr(host,'credit') and not hasattr(host,'play')
