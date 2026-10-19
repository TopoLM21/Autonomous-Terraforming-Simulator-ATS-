"""Compact counts and grouped views must preserve physical recipes and instances."""
from copy import deepcopy
import os
import pytest
from ats.config import Config,KWH
from ats.equipment import BaseEquipment,PanelModule,BatteryModule,resolve_config
from ats.equipment import set_panel_area
from ats.experiments import experiment_config
from ats.inventory import BatteryModule as PhysicalBattery,Location as L
from ats.simulation import Simulation


@pytest.fixture
def app():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    instance=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield instance
    instance.processEvents()


def test_base_quantities_control_real_capacity_area_and_mass(app):
    from ats.equipment_ui import BaseEditor
    original=BaseEquipment();editor=BaseEditor(original)
    assert editor.advanced.isHidden() and editor.panel_quantities.controls[0].value()==20
    assert original.panels[0].area_m2==1 and original.panels[0].mass_kg==4
    editor.panel_quantities.controls[0].setValue(100)
    editor.battery_quantities.controls[0].setValue(8)
    changed=editor.value()
    assert changed.panel_area==100 and changed.capacity==40*KWH
    assert changed.mass-original.mass==pytest.approx(320+100)
    assert original.panel_area==20 and original.capacity==30*KWH
    resized=deepcopy(original);set_panel_area(resized,200)
    assert resized.panels[0].quantity==200 and resized.panels[0].area_m2==1 and resized.panels[0].mass_kg==4
    editor.details.setChecked(True);assert not editor.advanced.isHidden()
    editor.panels.table.item(0,2).setText('1.5')
    assert editor.value().panel_area==150 and '1.5 м²' in editor.panel_quantities.labels[0].text()
    editor.close()


def test_custom_types_and_hidden_fields_survive_compact_editing(app):
    from ats.equipment_ui import BaseEditor
    original=BaseEquipment(panels=[PanelModule(name='Custom',quantity=7,area_m2=2,mass_kg=9,efficiency=.31)],
                           batteries=[BatteryModule(quantity=2,capacity_kwh=7,mass_kg=60),BatteryModule(quantity=3,capacity_kwh=2,mass_kg=12)])
    editor=BaseEditor(original)
    assert editor.value()==original
    editor.panel_quantities.controls[0].setValue(14);editor.battery_quantities.controls[1].setValue(5)
    changed=editor.value()
    assert changed.panels[0].area_m2==2 and changed.panels[0].efficiency==.31 and changed.panels[0].mass_kg==9
    assert changed.batteries[0]==original.batteries[0] and changed.batteries[1].quantity==5
    assert editor.value().panel_area==28
    editor.close()


def test_zero_equipment_can_be_added_from_compact_controls(app):
    from ats.equipment_ui import BaseEditor
    editor=BaseEditor(BaseEquipment(panels=[],batteries=[]))
    assert editor.value().panels==[] and editor.value().batteries==[]
    editor.panel_quantities.controls[0].setValue(30);editor.battery_quantities.controls[0].setValue(2)
    assert editor.value().panel_area==30 and editor.value().capacity==10*KWH
    editor.close()


def test_fleet_list_opens_only_the_selected_drone_and_preserves_others(app):
    from ats.equipment_ui import EquipmentDialog
    config=experiment_config(Config(),drones=3)
    dialog=EquipmentDialog(config)
    assert dialog.tabs.count()==2 and dialog.drone_table.rowCount()==3 and dialog.drone_stack.isHidden()
    before=dialog.editors[0].value()
    dialog.open_drone(1)
    assert dialog.show_drone_details.isChecked() and dialog.drone_stack.currentIndex()==1
    dialog.editors[1].battery_quantities.controls[0].setValue(3)
    result=dialog.value()
    assert result.fleet.models[0]==before
    assert result.fleet.models[1].battery==7.5*KWH
    assert result.fleet.models[2].battery==5*KWH
    assert dialog.editors[1].advanced.isHidden()
    dialog.preset(True)
    assert dialog.value().fleet.models[1].payload==30 and dialog.value().fleet.models[0].payload==60
    dialog.close()


def test_grouped_inventory_partitions_all_mass_energy_and_reserved_items():
    config=experiment_config(Config());config.logistics.drone_spare_initial_soc=.05
    sim=Simulation(config);world=sim.logistics
    world.replenish_batteries(sim.drone.battery_ids[0],2,.5)
    pack=world.inventory.select(PhysicalBattery,L.BASE_STORAGE)[0]
    assert world.inventory.reserve([pack.id],'existing_delivery')
    before=world.inventory.snapshot();groups=world.inventory.grouped()
    assert sum(g['count'] for g in groups)==len(world.inventory.items)
    assert sum(g['mass_kg'] for g in groups)==pytest.approx(sum(i.gross_mass for i in world.inventory.items.values()))
    assert sum(g['energy_j'] for g in groups)==pytest.approx(sum(b.energy_j for b in world.inventory.select(PhysicalBattery)))
    assert sum(g['reserved'] for g in groups)==1
    assert len({id for g in groups for id in g['member_ids']})==len(world.inventory.items)
    assert any('Разряженные' in g['label'] for g in groups) and any('Частично' in g['label'] for g in groups)
    assert world.inventory.snapshot()==before


def test_inventory_defaults_to_groups_and_drills_into_a_selected_group(app,monkeypatch):
    from ats.gui import MainWindow
    from ats.logistics_ui import LogisticsWindow
    config=experiment_config(Config(),drone_spare_sets=20)
    w=MainWindow(config);w.timer.stop();view=LogisticsWindow(w);view.timer.stop()
    inventory=w.sim.logistics.inventory
    monkeypatch.setattr(inventory,'snapshot',lambda:pytest.fail('grouping serialized all instances'))
    assert view.tabs.currentIndex()==1 and view.inventory.isHidden()
    assert view.inventory_groups.rowCount()<30 and len(inventory.items)>300
    index=next(i for i,g in enumerate(view._groups) if 'Батареи дронов' in g['label'] and 'На складе' in g['label'])
    view.inventory_groups.selectRow(index);view.show_inventory_details.setChecked(True)
    assert view.inventory.rowCount()==view._groups[index]['count']==160
    assert all(view.inventory.item(i,0).text() in view._groups[index]['member_ids'] for i in range(160))
    view.detail_scope.setCurrentIndex(1);assert view.inventory.rowCount()==len(inventory.items)
    view.show_inventory_details.setChecked(False);assert view.inventory.isHidden()
    view.close();w.close()
