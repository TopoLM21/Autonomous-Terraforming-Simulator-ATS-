"""Hot base storage changes preserve physical inventory, soil and energy."""
from copy import deepcopy
import os
import numpy as np
import pytest
from ats.config import Config,KWH
from ats.equipment import BatteryModule,BASE_BATTERY_LIMIT
from ats.experiments import experiment_config,save_configuration,load_configuration
from ats.inventory import Location as L
from ats.simulation import Simulation


def world():return Simulation(experiment_config(Config()))


def balanced(s):
    r=s.report(full=False)
    assert abs(r['mass_error_kg'])<1e-7
    assert abs(r['energy_error_j'])<.002


def test_installing_empty_base_packs_in_flight_preserves_world_jobs_and_charge():
    s=world();s.run(25);w=s.logistics
    before=s.report();terrain=s.terrain.height.copy();time=s.time
    positions=[d.position.copy() for d in s.drones]
    items=dict(w.inventory.items);reservations={id:o.reserved_by for id,o in items.items()}
    jobs=[(j.id,j.status,j.executor,j.destination) for j in w.jobs]
    slots=dict(w.battery_service.slots)
    proposed=deepcopy(s.config);proposed.base.equipment.batteries[0].quantity=100
    assert s.request_equipment(proposed)=='applied'
    r=s.report()
    assert r['base_battery_count']==100 and r['base_capacity_kwh']==500
    assert s.base.battery==before['base_kwh']*KWH and s.time==time
    assert r['initial_energy_kwh']==before['initial_energy_kwh'] and r['equipment_imported_kwh']==0
    assert r['base_mass_kg']-before['base_mass_kg']==94*50
    assert r['inventory_imported_mass_kg']==94*50
    assert len(w.base_battery_ids)==100 and w.battery_service.slots==slots
    assert all(w.inventory.items[id] is item and item.reserved_by==reservations[id] for id,item in items.items())
    assert all(w.inventory.items[id].energy_j==0 for id in w.base_battery_ids if id not in items)
    assert jobs==[(j.id,j.status,j.executor,j.destination) for j in w.jobs]
    np.testing.assert_array_equal(s.terrain.height,terrain)
    for d,p in zip(s.drones,positions):np.testing.assert_array_equal(d.position,p)
    s.run(400);balanced(s)


def test_removal_exports_charge_and_replacements_have_new_ids_and_no_free_energy():
    c=experiment_config(Config());c.base.initial_charge=.5
    c.base.equipment.batteries=[BatteryModule(quantity=2),BatteryModule(quantity=3,capacity_kwh=2,mass_kg=12)]
    s=Simulation(c);w=s.logistics;ids=list(w.base_battery_ids)
    w.configure_base_batteries([BatteryModule(),BatteryModule(capacity_kwh=2,mass_kg=12),
                               BatteryModule(capacity_kwh=2.5,mass_kg=15)])
    assert w.base_battery_ids[:2]==[ids[0],ids[2]]
    assert not any(id in w.inventory.items for id in (ids[1],ids[3],ids[4]))
    assert w.base_battery_ids[-1] not in ids
    assert s.base.battery==3.5*KWH and s.base.config.battery==9.5*KWH
    assert s.equipment_exported_energy==4.5*KWH and s.equipment_imported_energy==0
    assert w.imported_mass_kg==15 and w.exported_mass_kg==74
    assert all(w.inventory.items[id].location==L.BASE_MOUNT for id in w.base_battery_ids)
    s.run(100);balanced(s)


def test_storage_can_be_removed_and_reinstalled_and_saved_as_large_configuration(tmp_path):
    s=world();w=s.logistics
    w.configure_base_batteries([])
    assert s.base.battery==0 and s.base.config.battery==0 and not w.base_battery_ids
    balanced(s);s.step();balanced(s)
    w.configure_base_batteries([BatteryModule(quantity=BASE_BATTERY_LIMIT)])
    assert s.base.battery==0 and s.base.config.battery==5000*KWH
    s.step();balanced(s)
    path=tmp_path/'large-base.json';save_configuration(path,s.config)
    loaded=load_configuration(path)
    assert loaded.base.battery==5000*KWH and loaded.base.equipment.batteries[0].quantity==1000


@pytest.mark.parametrize('case',['quantity','total','moving','reserved','panel_site'])
def test_invalid_base_storage_edit_is_atomic(case):
    from ats.equipment import PanelModule
    s=world();w=s.logistics;proposed=deepcopy(s.config)
    proposed.base.equipment.batteries[0].quantity=8
    if case=='quantity':proposed.base.equipment.batteries[0].quantity=1001
    elif case=='total':proposed.base.equipment.batteries=[BatteryModule(quantity=600),BatteryModule(quantity=600)]
    elif case=='moving':s.base.destination=s.base.position[:2]+[1,0]
    elif case=='reserved':
        w.inventory.reserve([w.base_battery_ids[-1]],'existing_owner')
        proposed.base.equipment.batteries[0].quantity=0
    elif case=='panel_site':proposed.base.equipment.panels=[PanelModule(quantity=1,area_m2=100000)]
    before=s.report();equipment=deepcopy(s.config.base.equipment);items=dict(w.inventory.items)
    with pytest.raises(ValueError):s.request_equipment(proposed)
    assert s.report()==before and s.config.base.equipment==equipment
    assert w.inventory.items==items


def test_base_count_ui_allows_thousand_without_changing_drone_limits():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.equipment_ui import EquipmentDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s=world();dialog=EquipmentDialog(s.config)
    assert dialog.base.battery_quantities.controls[0].maximum()==1000
    assert dialog.editors[0].battery_quantities.controls[0].maximum()==64
    dialog.base.battery_quantities.controls[0].setValue(1000)
    assert s.request_equipment(dialog.value())=='applied'
    assert s.report()['base_battery_count']==1000
    assert s.base.config.battery==5000*KWH
    balanced(s);dialog.close();app.processEvents()
