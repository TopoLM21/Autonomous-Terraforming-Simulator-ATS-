"""Actual Qt setup, cancellation and placement; previews own no physical actors."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import threading
import time
import pytest
from PySide6 import QtCore,QtWidgets,QtTest
from ats.config import Config
from ats.complex_setup import SetupRequest,calculate,place
from ats.work_schedule import WorkSchedule
from ats.complex_setup_ui import WorkspaceWindow,ComplexSetupDialog,CalendarReviewDialog


@pytest.fixture
def app():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    QtCore.QThreadPool.globalInstance().waitForDone(30000)
    for widget in app.topLevelWidgets():widget.close()
    app.processEvents()


@pytest.fixture(scope='module')
def design():
    c=Config();c.base.latitude_deg=31.078
    return calculate(c,SetupRequest(name='Мой дневной комплекс'),WorkSchedule('hours',172,172,6))


def wait_until(app,predicate,seconds=30):
    end=time.perf_counter()+seconds
    while not predicate() and time.perf_counter()<end:
        app.processEvents();QtTest.QTest.qWait(10)
    assert predicate()


def test_empty_site_and_landscape_preview_create_no_equipment(app,monkeypatch):
    from ats.simulation import Simulation
    from ats.inventory import Inventory
    def fail(*a,**k):raise AssertionError('Empty preview created equipment')
    monkeypatch.setattr(Simulation,'__init__',fail);monkeypatch.setattr(Inventory,'__init__',fail)
    w=WorkspaceWindow(Config());w.show();w.show_composer();app.processEvents()
    assert w.sim is None and w.complex_project is None
    assert '0 комплексов' in w.inventory_status.text() and w.builder.drones.value()==4
    assert not w.builder.place_button.isEnabled()
    w.show_landscape();d=w.landscape_window
    d.example_corridor();d.side.setCurrentIndex(1);d.open_workspace()
    assert w.sim is None and w.landscape is d.region and w.config.georeference
    assert not d.move_button.isEnabled() and not d.move_next_button.isEnabled()


def test_background_review_does_not_block_events_and_stale_result_cannot_be_placed(app,design,monkeypatch):
    import ats.complex_setup_ui as module
    started=threading.Event();release=threading.Event()
    def delayed(*a,**kw):started.set();release.wait(5);return design
    monkeypatch.setattr(module,'calculate',delayed)
    w=WorkspaceWindow(Config());w.show_composer();d=w.builder
    beats=[];timer=QtCore.QTimer();timer.setInterval(5);timer.timeout.connect(lambda:beats.append(1));timer.start()
    try:
        d.calculate();wait_until(app,started.is_set)
        QtTest.QTest.qWait(50);assert beats and w.sim is None
        d.drones.setValue(3);release.set();wait_until(app,lambda:d.job is None)
        assert d.result is None and not d.place_button.isEnabled() and w.sim is None
        d.calculate();d.cancel_calculation();wait_until(app,lambda:d.job is None)
        assert d.result is None and w.sim is None
    finally:release.set();timer.stop()


def test_real_calculate_place_and_energy_review_preserve_the_selected_graph(app,tmp_path):
    c=Config();c.base.latitude_deg=31.078
    w=WorkspaceWindow(c);w.show();w.show_composer();d=w.builder
    d.mode.setCurrentIndex(d.mode.findData('hours'));d.hours.setValue(6)
    d.period.setCurrentIndex(d.period.findData('custom'))
    d.first.setDate(QtCore.QDate(2025,6,21));d.last.setDate(d.first.date())
    d.name.setText('Мой дневной комплекс');d.start_after.setChecked(False)
    d.calculate();wait_until(app,lambda:d.job is None)
    assert d.result is not None,d.status.text()
    assert w.sim is None and 'Стоимость' not in w.inventory_status.text()
    assert d.place_button.isEnabled() and d.save_button.isEnabled()
    template=tmp_path/'review.atstemplate';d.save_template(template);assert template.exists()
    for tab in range(3):
        d.tabs.setCurrentIndex(tab);app.processEvents()
        assert d.place_button.geometry().right()<d.width() and d.place_button.geometry().bottom()<d.height()
    d.place();app.processEvents();live=w.work_window;live.timer.stop()
    assert live.sim.config.operating_schedule['hours']==6 and len(live.sim.drones)==4
    assert live.complex_project.active.name=='Мой дневной комплекс'
    assert not live.running and not w.isVisible()
    live.show_power_design();assert isinstance(live.power_window,CalendarReviewDialog)
    assert '6 ч днём' in live.power_window.summary.toPlainText()
    assert 'Смета оборудования' in live.procurement_status.text()


def test_template_load_keeps_the_selected_site_and_rejects_silent_loss_of_extra_stock(app,design,tmp_path):
    from ats.complexes import ComplexTemplate
    from ats.plan import PlanConfig,PlanSegment
    c=Config();c.base.latitude_deg=-25.;c.plan=PlanConfig([PlanSegment([[16.,16.],[20.,16.]],height=-1.)])
    w=WorkspaceWindow(c);path=tmp_path/'recipe.atstemplate';design.template().save(path)
    w.load_template(path);d=w.builder
    assert d.source.plan==w.config.plan and d.latitude.value()==-25.
    assert d.source.base.equipment==design.config.base.equipment and d.drones.value()==4
    assert d.schedule().mode=='hours' and d.schedule().hours==6 and w.sim is None
    template=ComplexTemplate.load(path)
    template.stock=[dict(count=1,compatibility='excavator',capacity_kwh=1.5,mass_kg=11.,
        max_charge_w=1500.,max_discharge_w=6000.,initial_soc=1.)]
    template.save(path);w.load_template(path)
    assert 'дополнительный физический запас' in w.site_info.text() and w.sim is None


def test_opening_existing_save_does_not_rebuild_its_equipment_or_reset_time(app,design,tmp_path):
    from ats.checkpoint import save_state
    session=place(design);p=session.complex_project
    for _ in range(20):p.step()
    path=tmp_path/'continue.atsstate';save_state(path,session.simulation,complex_project=p)
    w=WorkspaceWindow(Config());w.load_session(path);live=w.work_window;live.timer.stop()
    assert live.sim.report()==session.simulation.report()
    assert live.complex_project.report()==p.report() and not live.running


def test_escape_cancels_a_pending_delivery_and_restores_previous_commands(app,design):
    from ats.gui import MainWindow
    session=place(design);live=MainWindow(session.simulation.config,session=session);live.timer.stop()
    d=ComplexSetupDialog(live);d.show();d.result=design;d._pending=True
    original=live.sim.landing_requested;d._delivery_previous=(False,[(live.sim,original)])
    live.sim.request_landing();live.running=True;d.delivery_timer.start()
    QtTest.QTest.keyClick(d,QtCore.Qt.Key.Key_Escape);app.processEvents()
    assert not d._pending and not d.delivery_timer.isActive()
    assert live.sim.landing_requested==original and not live.running


def test_builder_ten_real_krots_and_per_m3_statistic_are_visible(app):
    from tests.test_excavator_fleet import config
    from ats.equipment_ui import DailyStatsWindow
    c=config(10);c.base.latitude_deg=31.078
    w=WorkspaceWindow(c);w.show_composer();d=w.builder
    assert d.krot.maximum()==10 and d.krot.value()==0
    d.krot.setValue(10);assert d.containers.value()==80
    d.mode.setCurrentIndex(d.mode.findData('hours'));d.hours.setValue(6)
    d.period.setCurrentIndex(d.period.findData('custom'))
    d.first.setDate(QtCore.QDate(2025,6,21));d.last.setDate(d.first.date());d.start_after.setChecked(False)
    d.calculate();wait_until(app,lambda:d.job is None)
    assert d.result is not None,d.status.text()
    assert d.result.config.logistics.robot_count==10
    assert next(row for row in d.result.estimate['rows'] if row['key']=='excavator')['quantity']==10
    assert w.sim is None
    d.place();app.processEvents();live=w.work_window;live.timer.stop()
    assert len(live.sim.logistics.excavators)==10
    for _ in range(55):live.complex_project.step()
    live.refresh(True);app.processEvents()
    assert len(live.excavator_marker.points())==10
    stats=DailyStatsWindow(live);stats.timer.stop();stats.refresh()
    assert stats.robots.rowCount()==10
    assert len({stats.robots.item(row,0).text() for row in range(10)})==10
    assert 'USD/м³' in stats.economics.text() and 'Капитальная оценка' in stats.economics.text()
    assert 'USD/м³' in live.procurement_status.text()
    assert stats.robots.item(9,0).text()=='excavator_10'
