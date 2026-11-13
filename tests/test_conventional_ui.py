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
