"""Large physical panel inventories and the live base passport stay consistent."""
from copy import deepcopy
import os
import numpy as np
import pytest
from ats.checkpoint import load_state,save_state
from ats.config import Config
from ats.equipment import BaseEquipment,PanelModule,PANEL_QUANTITY_LIMIT,set_panel_area
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.simulation import Simulation


@pytest.mark.parametrize('empty',[False,True])
def test_ten_thousand_square_metres_keep_standard_panel_size_mass_and_efficiency(empty):
    equipment=BaseEquipment(panels=[] if empty else [PanelModule(efficiency=.27)])
    set_panel_area(equipment,10000,.27);equipment.validate()
    assert equipment.panels==[PanelModule(quantity=10000,efficiency=.27)]
    assert equipment.panel_area==10000 and equipment.breakdown()['panels_kg']==40000


@pytest.mark.parametrize('quantity',[PANEL_QUANTITY_LIMIT+1,True,-1])
def test_invalid_panel_quantities_are_rejected(quantity):
    with pytest.raises(ValueError):PanelModule(quantity=quantity).validate()


def test_large_solar_installation_in_flight_preserves_jobs_energy_and_saved_state(tmp_path):
    s=Simulation(experiment_config(with_map_size(Config(),256)));s.run(25)
    assert any(not d.grounded for d in s.drones)
    before=s.report();terrain=s.terrain.height.copy()
    positions=[d.position.copy() for d in s.drones];active=dict(s.logistics.active)
    physical={id:item for id,item in s.logistics.inventory.items.items() if item.kind!='SOLAR_PANEL'}
    s.set_panel_area(10000);r=s.report()
    support=sum(o.mass_kg for o in s.logistics.inventory.items.values() if o.kind in ('SOLAR_FRAME','SOLAR_INSTALLER'))
    assert s.time==25 and r['base_mass_kg']-before['base_mass_kg']==39920+support
    assert r['inventory_imported_mass_kg']==39920+support
    assert s.config.base.equipment.panels==[PanelModule(quantity=10000)]
    panels=[p for p in s.logistics.inventory.items.values() if p.kind=='SOLAR_PANEL']
    assert len(panels)==10000 and sum(p.mass_kg for p in panels)==40000
    assert r['base_kwh']==before['base_kwh'] and r['equipment_imported_kwh']==0
    assert np.prod(r['base_specification']['solar_field_dimensions_m'])>10000
    assert len(s.logistics.solar_array.sections)>400
    assert all(s.logistics.inventory.items[id] is item for id,item in physical.items())
    assert s.logistics.active==active
    np.testing.assert_array_equal(s.terrain.height,terrain)
    for d,p in zip(s.drones,positions):np.testing.assert_array_equal(d.position,p)
    path=tmp_path/'large-field.atsstate';save_state(path,s)
    loaded=load_state(path).simulation
    assert loaded.report()==r
    for _ in range(10):s.step();loaded.step()
    assert loaded.report()==s.report()
    s.run(301);r=s.report()
    assert not r['solar_field_deployed'] and 20<=r['solar_generating_area_m2']<10000
    assert r['solar_power_w']>0 and abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002


def test_unplaceable_large_field_does_not_change_the_existing_base():
    s=Simulation(experiment_config(Config()));before=s.report();equipment=deepcopy(s.config.base.equipment)
    with pytest.raises(ValueError,match='BLOCKED_SOLAR'):s.set_panel_area(10000)
    assert s.report()==before and s.config.base.equipment==equipment


def test_passport_tracks_real_storage_and_docked_mass_without_resizing_the_body():
    c=experiment_config(Config(),landing_slots=4);c.base.radius=1.25;c.base.roof_height=2.1
    s=Simulation(c);before=s.report()['base_specification']
    assert before['body_dimensions_m']==[2.5,2.5,2.1]
    assert before['landing_zone_dimensions_m']==[2.5,11.5]
    s.drone.cargo=s.terrain.scoop(16,16,10)
    s.logistics.direct_ids.add(s.drone.identifier)
    s.logistics.replenish_batteries(s.drone.battery_ids[0],2,0.)
    r=s.report();spec=r['base_specification']
    assert spec['body_dimensions_m']==before['body_dimensions_m']
    assert spec['stored_items_mass_kg']-before['stored_items_mass_kg']==32
    docked=sum(d.config.empty_mass+d.cargo for d in s.drones if d.docked)
    assert spec['docked_drones_mass_kg']==docked
    assert spec['gross_mass_kg']==r['base_mass_kg']+docked
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002


def test_gui_accepts_ten_thousand_panels_and_refreshes_the_live_passport(monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.equipment_ui import EquipmentDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(QtWidgets.QMessageBox,'warning',lambda *args:pytest.fail(str(args[2])))
    host=MainWindow(experiment_config(with_map_size(Config(),256)));host.timer.stop()
    dialog=EquipmentDialog(host.config);original=host.sim
    try:
        assert dialog.base.panel_quantities.controls[0].maximum()==PANEL_QUANTITY_LIMIT
        dialog.base.panel_quantities.controls[0].setValue(10000)
        assert host.apply_equipment(dialog.value())=='applied'
        assert host.sim is original and original.time==0 and host.panels.value()==10000
        assert host.base_mass.value()==pytest.approx(original.base.config.mass,abs=.05)
        field=original.base_specification()['solar_field_dimensions_m']
        assert f'{field[0]:.2f} × {field[1]:.2f} м' in host.base_dimensions.text()
        x,y=host.base_outline.getData()
        assert np.ptp(x)==np.ptp(y)==2*original.base.config.radius
        mass=host.base_mass.value();host.render_mode.setCurrentIndex(host.render_mode.findData('off'))
        original.logistics.replenish_batteries(original.drone.battery_ids[0],2,0.)
        host.refresh()
        assert host.base_mass.value()-mass==32
        assert f"{original.base_specification()['gross_mass_kg']:.1f} кг" in host.base_dimensions.text()
        assert 'Корпус' in host.telemetry.text() and 'Масса с дронами' in host.telemetry.text()
    finally:
        dialog.close();host.close();app.processEvents()
