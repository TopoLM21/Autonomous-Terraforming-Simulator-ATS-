"""The conventional equipment window drives the real separate worksite."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from tests.test_conventional_workspace import app_and_window


def test_dialog_opens_without_purchase_and_explains_three_shift_hours():
    from PySide6 import QtWidgets
    app,host=app_and_window()
    try:
        host.show_conventional();dialog=host.equipment_window;dialog.timer.stop()
        assert host.sim.conventional is None and not host.sim.drones
        assert dialog.create_button.isEnabled() and not dialog.pause_button.isEnabled()
        assert dialog.tabs.count()==4
        assert all(isinstance(dialog.tabs.widget(i),QtWidgets.QScrollArea) for i in range(3))
        dialog.controls['shifts_per_day'].setValue(3)
        config=dialog.config_value()
        assert config.shift_span_hours==8 and config.lunch_hours==1
        assert '7 ч труда' in dialog.shift_note.text() and 'бригады' in dialog.shift_note.text()
        assert host.sim.time==0 and host.sim.conventional is None
        dialog.create_button.click()
        assert host.sim.conventional is not None,dialog.message.text()
        assert host.sim.conventional.report()['crew_count']==6
        assert host.sim.time==0 and not host.sim.drones
    finally:host.close();app.processEvents()


def test_real_placement_pause_supplies_and_loaded_dialog_keep_a_separate_account(tmp_path):
    app,host=app_and_window()
    try:
        host.show_conventional();dialog=host.equipment_window;dialog.timer.stop()
        dialog.controls['base_x'].setValue(12);dialog.controls['base_y'].setValue(12)
        dialog.controls['truck_count'].setValue(2)
        for widget,value in zip(dialog.rect,(35.,40.,60.,20.)):widget.setValue(value)
        before=host.sim.time;dialog.create_button.click()
        fleet=host.sim.conventional;assert fleet is not None,dialog.message.text()
        assert host.sim.time==before and not host.sim.drones and host.sim.logistics is None
        assert dialog.table.rowCount()==3 and not dialog.create_button.isEnabled()
        assert all(not page.isEnabled() for page in dialog.setup_pages)
        assert 'USD/м³' not in dialog.summary.text()  # no invented unit cost before extraction
        assert 'до начала добычи' in dialog.summary.text() and 'автономный комплекс' in dialog.summary.text()
        report=fleet.report();capital=report['capital_cost_usd']
        dialog.supply_fuel.setValue(50);dialog.supply_water.setValue(10);dialog.supply_food.setValue(2)
        dialog.replenish_button.click();stock=fleet.report()
        assert stock['fuel_stock_l']==pytest.approx(report['fuel_stock_l']+50)
        assert stock['water_stock_l']==pytest.approx(report['water_stock_l']+10)
        assert stock['food_stock_units']==pytest.approx(report['food_stock_units']+2)
        assert stock['capital_cost_usd']==capital and host.sim.time==before
        dialog.pause_button.click();assert not fleet.report()['enabled']
        assert 'Возобновить' in dialog.pause_button.text()
        dialog.pause_button.click();assert fleet.report()['enabled']
        state=tmp_path/'dialog.atsstate';assert host.save_session(state)
        assert host.load_session(state)
        host.show_conventional();restored=host.equipment_window;restored.timer.stop()
        assert restored is not dialog and not restored.create_button.isEnabled()
        assert restored.controls['truck_count'].value()==2 and restored.table.rowCount()==3
        assert tuple(widget.value() for widget in restored.rect)==(35.,40.,60.,20.)
        assert host.sim.conventional.report()['capital_cost_usd']==capital
        assert host.sim.time==before and not host.sim.drones
    finally:host.close();app.processEvents()


def test_selecting_another_shared_complex_keeps_one_conventional_fleet_and_account():
    from PySide6 import QtWidgets
    from ats.checkpoint import Session
    from ats.conventional import ConventionalConfig
    from ats.conventional_ui import ConventionalWindow
    from ats.gui import MainWindow
    from tests.test_shared_site import site
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    project,template=site();project.add_shared(template,position=(100.,20.));project.select('complex_1')
    host=MainWindow(project.active.simulation.config,
                    session=Session(project.active.simulation,complex_project=project));host.timer.stop()
    fresh=None
    try:
        cfg=ConventionalConfig(base_x=12.,base_y=100.,work_rect=(35.,30.,20.,20.),truck_count=2)
        fleet=host.install_conventional(cfg)
        # The UI host need not return the engine object when submitting commands.
        if fleet is None:fleet=host.sim.conventional
        host.show_conventional();dialog=host.conventional_window;dialog.timer.stop()
        assert dialog.fleet() is fleet
        owners=[getattr(unit.simulation,'conventional',None) for unit in project.units]
        capital=fleet.report()['capital_cost_usd'];clock=project.time
        host.select_complex('complex_2');host.timer.stop();host.show_conventional()
        other=host.conventional_window;other.timer.stop();other.refresh()
        assert other.fleet() is fleet and not other.create_button.isEnabled()
        assert all(not page.isEnabled() for page in other.setup_pages)
        assert tuple(widget.value() for widget in other.rect)==cfg.work_rect
        assert 'Зарезервированный участок' in other.patch_status.text()
        assert 'дроны работают на других участках' in other.patch_status.text()
        assert other.table.rowCount()==3
        other.create()  # also exercise the command guard, not just a disabled button
        assert 'уже' in other.message.text().lower()
        assert [getattr(unit.simulation,'conventional',None) for unit in project.units]==owners
        assert fleet.report()['capital_cost_usd']==capital and project.time==clock
        fuel=fleet.report()['fuel_stock_l']
        other.supply_fuel.setValue(10);other.supply_water.setValue(0);other.supply_food.setValue(0)
        other.replenish_button.click()
        assert fleet.report()['fuel_stock_l']==pytest.approx(fuel+10)
        other.pause_button.click();assert not fleet.report()['enabled']
        other.pause_button.click();assert fleet.report()['enabled']
        host.select_complex('complex_1');host.timer.stop();other.refresh()
        assert other.fleet() is fleet and not other.create_button.isEnabled()
        assert tuple(widget.value() for widget in other.rect)==cfg.work_rect
        host.select_complex('complex_2');host.timer.stop()
        fresh=ConventionalWindow(host);fresh.timer.stop()
        assert fresh.fleet() is fleet and not fresh.create_button.isEnabled()
        assert fresh.controls['truck_count'].value()==2 and fresh.table.rowCount()==3
        assert project.time==clock
    finally:
        if fresh is not None:fresh.close()
        host.close();app.processEvents()


def test_released_patch_updates_the_open_dialog_without_another_purchase():
    from PySide6 import QtWidgets
    from ats.checkpoint import Session
    from ats.conventional import ConventionalConfig
    from ats.gui import MainWindow
    from tests.test_shared_site import site
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    project,_=site();project.enable_shared()
    host=MainWindow(project.active.simulation.config,
                    session=Session(project.active.simulation,complex_project=project));host.timer.stop()
    try:
        fleet=host.install_conventional(ConventionalConfig(base_x=108.,base_y=20.,work_rect=(35.,30.,15.,20.)))
        host.show_conventional();dialog=host.conventional_window;dialog.timer.stop()
        assert 'Зарезервированный участок' in dialog.patch_status.text()
        capital=fleet.report()['capital_cost_usd'];clock=project.time
        # Presentation fixture only: natural handoff is covered by core tests.
        # Keep the same fleet and mask objects to catch stale widget caches.
        terrain=project.shared_site.core
        terrain.target[fleet.work_mask]=terrain.height[fleet.work_mask]
        terrain.revision+=1
        assert not fleet._choose_excavator_site(fleet.excavator)
        assert fleet.ground_stage_drained
        project.shared_site.core._conventional_mask.fill(False)
        dialog.refresh()
        assert 'Участок передан дронам' in dialog.patch_status.text()
        assert 'Зарезервированный участок' not in dialog.patch_status.text()
        assert dialog.fleet() is fleet and not dialog.create_button.isEnabled()
        assert fleet.report()['capital_cost_usd']==capital and project.time==clock
    finally:host.close();app.processEvents()
