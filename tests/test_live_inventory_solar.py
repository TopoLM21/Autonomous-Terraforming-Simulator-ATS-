"""External resources have explicit mass/energy; live edits preserve the world."""
from copy import deepcopy
import json
import os
import subprocess
import sys
import numpy as np
import pytest

from ats.config import Config,KWH
from ats.experiments import experiment_config,save_configuration,load_configuration
from ats.inventory import BatteryModule,ChargingSlot,Location as L
from ats.maps import with_map_size
from ats.scenarios import scenario
from ats.simulation import Simulation


def make_world():
    return Simulation(experiment_config(with_map_size(scenario('medium'),64)))


def balanced(sim):
    r=sim.report(full=False)
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<1e-4


def test_live_battery_shipment_preserves_flights_reservations_and_initial_energy():
    s=make_world();s.run(25);w=s.logistics
    existing=dict(w.inventory.items);reservations={id:o.reserved_by for id,o in existing.items()}
    time=s.time;terrain=s.terrain.height.copy();positions=[d.position.copy() for d in s.drones]
    before=s.report();template=s.drone.battery_ids[0]
    ids=w.replenish_batteries(template,3,.5);r=s.report()
    assert s.time==time and s.report()['initial_energy_kwh']==before['initial_energy_kwh']
    np.testing.assert_array_equal(s.terrain.height,terrain)
    for d,p in zip(s.drones,positions):np.testing.assert_array_equal(d.position,p)
    assert all(w.inventory.items[id] is obj and obj.reserved_by==reservations[id] for id,obj in existing.items())
    assert len(set(ids))==3 and len(w.inventory.items)==len(existing)+6
    assert all(id in w.battery_service.slots for id in ids)
    assert r['equipment_imported_kwh']==pytest.approx(3*2.5*.5)
    assert r['base_mass_kg']-before['base_mass_kg']==pytest.approx(3*(15+1))
    assert r['battery_deliveries_count']==1 and r['inventory_imported_mass_kg']==48
    s.run(100);balanced(s)


def test_empty_delivered_modules_use_real_slot_power_and_import_no_energy():
    s=make_world();w=s.logistics
    ids=w.replenish_batteries(s.drone.battery_ids[0],2,0.)
    assert s.equipment_imported_energy==0
    assert all(w.inventory.items[id].energy_j==0 for id in ids)
    s.step()
    assert all(w.inventory.items[id].energy_j==pytest.approx(1500*.9) for id in ids)
    balanced(s)


@pytest.mark.parametrize('count,soc',[(0,1),(129,1),(True,1),(2,-.1),(2,1.1),(2,float('nan'))])
def test_invalid_shipment_is_atomic(count,soc):
    s=make_world();before=s.report();items=dict(s.logistics.inventory.items)
    with pytest.raises(ValueError):s.logistics.replenish_batteries(s.drone.battery_ids[0],count,soc)
    assert s.report()==before and s.logistics.inventory.items==items


def test_hot_drone_spares_are_used_in_an_actual_timed_swap():
    c=experiment_config(Config(),drones=1,drone_spare_sets=0)
    c.fleet.models[0].initial_charge=.25
    s=Simulation(c);w=s.logistics;old=list(s.drone.battery_ids)
    s.step();ids=w.replenish_batteries(old[0],2)
    s.run(61)
    assert w.battery_service.drone_swap_count==1 and set(s.drone.battery_ids)==set(ids)
    assert all(id in w.inventory.items for id in old)
    assert s.equipment_imported_energy==pytest.approx(5*KWH)
    balanced(s)


def test_hot_krot_spares_recover_a_no_spares_block_without_reset():
    c=experiment_config(scenario('medium'),excavator_spare_batteries=0)
    c.logistics.excavator_initial_soc=.08
    c.logistics.idle_digging=False  # strict logistics has no independent diggers
    s=Simulation(c);w=s.logistics
    # Pre-staged empty containers isolate an actual no-spares block from
    # unrelated initial deliveries, which correctly keep logistics running.
    for container in w.containers:
        w.inventory.move(container.id,L.WORKSITE_RACK,'worksite',w.rack,'EMPTY_AT_EXCAVATOR')
    s.run(20)
    assert s.state.value=='BLOCKED' and 'NO_SPARE_BATTERIES' in s.reason
    time=s.time;ids=w.replenish_batteries(w.excavator.battery_ids[0],2)
    assert s.state.value=='RUNNING' and s.time==time
    for _ in range(1200):
        s.step();balanced(s)
        if w.excavator.swaps:break
    assert w.excavator.swaps==1 and set(w.excavator.battery_ids)==set(ids)


def test_live_solar_area_changes_mass_generation_and_saved_configuration(tmp_path):
    s=make_world();s.run(25);before=s.report();terrain=s.terrain.height.copy();time=s.time
    old_items={id:o for id,o in s.logistics.inventory.items.items() if o.kind!='SOLAR_PANEL'}
    s.set_panel_area(123.5)
    assert s.time==time and s.base.config.panel_area==123.5
    assert s.base.config.mass-before['base_mass_kg']==pytest.approx(4*(123.5-20))
    assert s.equipment_imported_energy==s.equipment_exported_energy==0
    assert all(s.logistics.inventory.items[id] is obj for id,obj in old_items.items())
    np.testing.assert_array_equal(s.terrain.height,terrain)
    assert not s.logistics.field_ready and s.logistics.field_rect is not None
    s.run(301);assert s.logistics.field_ready and s.base.solar_power>0
    s.set_panel_area(0);s.step();assert s.base.solar_power==0
    s.set_panel_area(37.25);s.run(301);assert s.base.solar_power>0
    path=tmp_path/'solar.json';save_configuration(path,s.config)
    assert load_configuration(path).base.panel_area==37.25
    balanced(s)


@pytest.mark.parametrize('area',[-1,float('nan'),float('inf'),1000000])
def test_invalid_or_unplaceable_solar_area_keeps_the_current_world(area):
    s=make_world();before=s.report();equipment=deepcopy(s.base.config.equipment)
    with pytest.raises(ValueError):s.set_panel_area(area)
    assert s.report()==before and s.base.config.equipment==equipment


def test_panel_cli_overrides_a_loaded_configuration_and_can_start_from_zero(tmp_path):
    c=experiment_config(Config(),panel_area=0);path=tmp_path/'empty-solar.json';save_configuration(path,c)
    p=subprocess.run([sys.executable,'-m','ats','--headless','--configuration',str(path),
                      '--panel-area','123.5','--steps','0'],capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert p.returncode==0,p.stderr
    r=json.loads(p.stdout)
    assert r['panel_area_m2']==123.5 and r['logistics_version']==2


def test_real_gui_controls_apply_panels_and_deliver_batteries_without_reset():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.logistics_ui import LogisticsWindow,ExperimentDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(experiment_config(with_map_size(Config(),64)));w.timer.stop();w.sim.run(10)
    original=w.sim;time=original.time;w.running=True
    w.panels.setValue(73.5);w.apply_panels_button.click()
    assert w.sim is original and w.sim.time==time and w.running
    assert w.sim.base.config.panel_area==73.5
    view=LogisticsWindow(w);view.timer.stop();view.tabs.setCurrentIndex(1)
    view.show_inventory_details.setChecked(True);view.detail_scope.setCurrentIndex(1)
    count=len(original.logistics.inventory.items)
    view.delivery_count.setValue(2);view.delivery_soc.setValue(25);view.replenish_button.click()
    assert w.sim is original and original.time==time and w.running
    assert view.inventory.rowCount()==count+4 and original.equipment_imported_energy==pytest.approx(1.25*KWH)
    view.tabs.setCurrentIndex(3);assert view.charging.rowCount()==22
    dialog=ExperimentDialog(w)
    assert dialog.mode.count()==2 and all('Старый' not in dialog.mode.itemText(i) for i in range(2))
    dialog.solar.setValue(87.25);assert dialog.candidate().base.panel_area==87.25
    dialog.close();view.close();w.close();app.processEvents();balanced(original)
