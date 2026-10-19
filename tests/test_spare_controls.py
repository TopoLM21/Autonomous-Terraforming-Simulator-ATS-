"""Starting inventory controls must create finite mass and accounted energy."""
import json
import os
import subprocess
import sys

import pytest

from ats.config import Config, KWH
from ats.experiments import experiment_config, load_configuration, save_configuration
from ats.inventory import BatteryModule, Location
from ats.simulation import Simulation


@pytest.mark.parametrize('slots,total', [(2,4),(4,8)])
def test_default_reserve_keeps_existing_experiment_inventory(slots,total):
    config=experiment_config(Config(),robot_slots=slots)
    assert config.logistics.drone_spare_sets==1
    assert config.logistics.excavator_battery_count==total


def test_extra_spares_are_physical_items_with_initial_mass_and_energy():
    baseline=experiment_config(Config(),drones=2,battery_modules=1,
                               drone_spare_sets=0,excavator_spare_batteries=0)
    configured=experiment_config(baseline,drones=2,battery_modules=1,
                                 drone_spare_sets=3,excavator_spare_batteries=5)
    original=Simulation(baseline);sim=Simulation(configured)
    batteries=sim.logistics.inventory.select(BatteryModule)
    drone_spares=[b for b in batteries if '_spare_' in b.id]
    robot_spares=[b for b in batteries if b.compatibility=='excavator'
                  and b.location==Location.BASE_STORAGE]
    assert len(drone_spares)==6 and len(robot_spares)==5
    assert len(sim.drone.battery_ids)==1
    assert len(sim.logistics.excavator.battery_ids)==2
    expected_energy=sum(b.energy_j for b in drone_spares+robot_spares)
    assert sim.report()['initial_energy_kwh']-original.report()['initial_energy_kwh']==pytest.approx(expected_energy/KWH)
    inventory_mass=lambda s:sum(b.mass_kg for b in s.logistics.inventory.select(BatteryModule))
    added_mass=sum(b.mass_kg for b in drone_spares+robot_spares)
    assert inventory_mass(sim)-inventory_mass(original)==pytest.approx(added_mass)
    assert sim.time==original.time==0
    added_slots=len(drone_spares)+len(robot_spares)
    assert sim.base.config.mass-original.base.config.mass==pytest.approx(
        added_mass+added_slots*configured.logistics.charger_slot_mass_kg)
    assert sim.equipment_imported_energy==sim.equipment_exported_energy==0
    for _ in range(60):sim.step()
    assert sim.equipment_imported_energy==sim.equipment_exported_energy==0
    assert abs(sim.report()['energy_error_j'])<1e-4
    assert abs(sim.report()['mass_error_kg'])<1e-7


def test_configuration_roundtrip_and_reconfiguration_preserve_spares(tmp_path):
    from dataclasses import asdict
    config=experiment_config(Config(),drones=2,robot_slots=4,
                             drone_spare_sets=3,excavator_spare_batteries=7)
    snapshot=asdict(config)
    path=tmp_path/'reserves.json';save_configuration(path,config)
    assert asdict(config)==snapshot  # serialization cannot change running configuration
    restored=load_configuration(path)
    assert restored.logistics.drone_spare_sets==3
    assert restored.logistics.excavator_battery_count==11
    assert Simulation(config).run(10)==Simulation(restored).run(10)
    changed=experiment_config(restored,robot_slots=2)
    assert changed.logistics.drone_spare_sets==3
    assert changed.logistics.excavator_battery_count==9  # seven spare plus two mounted


@pytest.mark.parametrize('flags,sets,spares', [([],2,3),(['--drone-spare-sets','4'],4,3),
    (['--excavator-spare-batteries','7'],2,7),
    (['--drone-spare-sets','0','--excavator-spare-batteries','0'],0,0)])
def test_cli_explicit_reserve_flags_override_loaded_configuration(tmp_path,flags,sets,spares):
    config=experiment_config(Config(),drones=2,robot_slots=4,
                             drone_spare_sets=2,excavator_spare_batteries=3)
    path=tmp_path/'scenario.json';save_configuration(path,config)
    saved=tmp_path/'saved.json'
    result=subprocess.run([sys.executable,'-m','ats','--headless','--steps','3',
        '--configuration',str(path),'--save-configuration',str(saved),*flags],
        capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    physical=report['inventory']
    assert len([b for b in physical if b['type']=='BatteryModule' and '_spare_' in b['id']])==2*2*sets
    assert len([b for b in physical if b.get('compatibility')=='excavator'])==4+spares
    assert abs(report['energy_error_j'])<1e-4
    assert report['equipment_imported_kwh']==report['equipment_exported_kwh']==0
    saved_config=load_configuration(saved)
    assert saved_config.logistics.drone_spare_sets==sets
    assert saved_config.logistics.excavator_battery_count==4+spares


def test_cli_new_experiment_accepts_spare_inventory_and_rejects_legacy():
    flags=['--drone-spare-sets','2','--excavator-spare-batteries','5']
    result=subprocess.run([sys.executable,'-m','ats','--headless','--steps','0',
        '--mode','logistics','--drones','1',*flags],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert len([b for b in report['inventory'] if b['type']=='BatteryModule' and '_spare_' in b['id']])==4
    assert len([b for b in report['inventory'] if b.get('compatibility')=='excavator'])==7
    legacy=subprocess.run([sys.executable,'-m','ats','--headless','--mode','legacy','--steps','0',*flags],
                          capture_output=True,text=True,timeout=30)
    assert legacy.returncode==2 and 'requires --mode logistics' in legacy.stderr


@pytest.mark.parametrize('flag,value', [('--drone-spare-sets','-1'),('--drone-spare-sets','65'),
    ('--excavator-spare-batteries','-1'),('--excavator-spare-batteries','129')])
def test_cli_invalid_inventory_cannot_write_a_configuration(tmp_path,flag,value):
    path=tmp_path/'invalid.json'
    result=subprocess.run([sys.executable,'-m','ats','--headless','--steps','0','--mode','logistics',
        flag,value,'--save-configuration',str(path)],capture_output=True,text=True,timeout=30)
    assert result.returncode==2 and not path.exists()


@pytest.mark.parametrize('values', [{'drone_spare_sets':-1},{'drone_spare_sets':65},
                                  {'excavator_spare_batteries':-1},{'excavator_spare_batteries':129}])
def test_experiment_rejects_out_of_range_reserve(values):
    with pytest.raises(ValueError):experiment_config(Config(),**values)


def test_experiment_dialog_spares_initialize_save_load_and_apply(tmp_path,monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.logistics_ui import ExperimentDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class Host(QtWidgets.QWidget):
        def replace_simulation(self,config):self.config=config

    host=Host();host.config=experiment_config(Config(),drones=2,robot_slots=4,
        drone_spare_sets=3,excavator_spare_batteries=7)
    dialog=ExperimentDialog(host)
    assert dialog.drone_spares.value()==3 and dialog.robot_spares.value()==7
    assert dialog.robot_slots.currentIndex()==1 and dialog.drones.value()==2
    dialog.drone_spares.setValue(5);dialog.robot_spares.setValue(11)
    path=tmp_path/'gui-spares.json'
    monkeypatch.setattr(QtWidgets.QFileDialog,'getSaveFileName',lambda *args:(str(path),'JSON (*.json)'))
    dialog.save()
    assert host.config.logistics.drone_spare_sets==3  # saving cannot inject runtime batteries
    saved=load_configuration(path)
    assert saved.logistics.drone_spare_sets==5 and saved.logistics.excavator_battery_count==15
    monkeypatch.setattr(QtWidgets.QFileDialog,'getOpenFileName',lambda *args:(str(path),'JSON (*.json)'))
    dialog.load()
    reopened=ExperimentDialog(host)
    assert reopened.drone_spares.value()==5 and reopened.robot_spares.value()==11
    reopened.drone_spares.setValue(1);reopened.apply()
    assert host.config.logistics.drone_spare_sets==1
    assert host.config.logistics.excavator_battery_count==15
    reopened.close();dialog.close();host.close();app.processEvents()
