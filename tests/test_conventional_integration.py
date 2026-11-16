"""The diesel fleet shares physical soil, but never autonomous purchases/costs."""
import json
import subprocess
import sys
import numpy as np
import pytest
from ats.simulation import Simulation,MissionState
from ats.conventional import ConventionalConfig
from ats.checkpoint import save_state,load_state
from tests.test_conventional_workspace import workspace_config


def fleet_config():return ConventionalConfig(base_x=12.,base_y=12.,work_rect=(35.,40.,60.,20.))


def test_zero_fleet_is_explicit_and_does_not_allocate_phantom_equipment():
    c=workspace_config();c.fleet.count=0;c.fleet.charge_slots=0
    with pytest.raises(ValueError):c.validate()
    c=workspace_config();c.conventional_only=True
    with pytest.raises(ValueError):c.validate()
    c.conventional_only='yes'
    with pytest.raises(ValueError):c.validate()
    s=Simulation.conventional_only(workspace_config())
    assert not s.drones and not s.slots and s.logistics is None
    from ats.site_ground import equipment_mask
    assert not np.any(equipment_mask(s.terrain,[s]))
    assert s.base.battery==0 and s.report()['autonomous_equipment_cost_usd']==0
    s.install_conventional(fleet_config());s.step()
    assert s.time==1 and s.report()['mass_error_kg']==0


def test_separate_patch_excludes_new_autonomous_jobs_without_changing_physical_plan():
    s=Simulation(workspace_config());before=s.terrain.remaining_cut
    target=s.terrain.target.copy();s.install_conventional(fleet_config())
    t=s.terrain;ids=t.source_indices()
    assert not np.any(t._conventional_mask.ravel()[ids])
    np.testing.assert_array_equal(target,t.target)
    assert t.remaining_cut==before
    assert s.excavated_mass==0 and s.conventional.report()['capital_cost_usd']>0
    s.state=MissionState.COMPLETED;clock=s.time;s.step()
    assert s.time==clock+1 and s.conventional.payroll_usd>0


@pytest.mark.parametrize('layered',[False,True])
def test_checkpoint_keeps_loaded_soil_materials_and_fuel_in_one_ledger(tmp_path,layered):
    c=workspace_config()
    if layered:
        from ats.soil import SoilProfile
        c.soil_profile=SoilProfile.qattara()
    s=Simulation.conventional_only(c);f=s.install_conventional(fleet_config());truck=f.machines[-1]
    if s.terrain.soil is not None:
        truck.soil=s.terrain.scoop_load(55.,50.,120.)
        truck.cargo_kg=truck.soil.mass
    else:truck.cargo_kg=s.terrain.scoop(55.,50.,120.)
    f.excavated_mass_kg=truck.cargo_kg
    f.excavated_m3=truck.soil.volume if layered else truck.cargo_kg/c.terrain.density
    if layered:f.extracted_materials+=truck.soil.masses
    assert truck.cargo_kg>0 and abs(s.report()['mass_error_kg'])<1e-7
    if layered:assert s.report()['soil']['max_material_mass_error_kg']<1e-7
    path=tmp_path/'diesel.atsstate';save_state(path,s);other=load_state(path).simulation
    assert other.conventional.sim is other and other.conventional.terrain is other.terrain
    assert other.report()==s.report()
    for _ in range(30):s.step();other.step()
    assert other.report()==s.report()
    assert abs(s.report()['conventional']['fuel_balance_error_l'])<1e-8


def test_live_composition_and_plan_cannot_silently_delete_a_loaded_fleet():
    s=Simulation.conventional_only(workspace_config());f=s.install_conventional(fleet_config())
    with pytest.raises(ValueError):s.install_conventional(fleet_config())
    with pytest.raises(ValueError):s.apply_plan(s.config.plan)
    assert s.conventional is f and s.report()['conventional']['capital_cost_usd']>0


def test_existing_headless_resume_path_accepts_an_empty_autonomous_fleet(tmp_path):
    s=Simulation.conventional_only(workspace_config());s.install_conventional(fleet_config())
    path=tmp_path/'diesel.atsstate';out=tmp_path/'report.json';save_state(path,s)
    result=subprocess.run([sys.executable,'-m','ats','--headless','--load-state',str(path),
        '--steps','5','--json',str(out)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    r=json.loads(out.read_text(encoding='utf-8'));assert r['time_seconds']==5 and r['drone_count']==0
    assert 'conventional' in r and abs(r['mass_error_kg'])<1e-7


def test_existing_ats2_window_shows_a_separate_fleet_without_combining_costs():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.experiments import experiment_config
    from ats.costs import default_catalog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c=experiment_config(workspace_config(),drones=4);c.cost_catalog=default_catalog()
    w=MainWindow(c);w.timer.stop()
    try:
        auto_cost=w.sim.report().get('equipment_cost_estimate')
        assert auto_cost['total']>0
        from dataclasses import replace
        w.install_conventional(replace(fleet_config(),base_x=108.,base_y=20.))
        w.refresh(True);w.show_conventional()
        w.conventional_window.timer.stop()
        assert w.conventional_window.isVisible()
        assert len(w.ground_vehicle_markers.points())==3  # excavator, truck, camp
        assert 'отдельная смета' in w.compact_telemetry.text()
        assert w.sim.report().get('equipment_cost_estimate')==auto_cost
        assert w.sim.report()['conventional']['capital_cost_usd']==225000
        assert abs(w.sim.report()['mass_error_kg'])<1e-7
    finally:w.close();app.processEvents()


def test_soil_still_slumps_when_only_the_conventional_fleet_keeps_working():
    from ats.plan import PlanConfig,PlanSegment
    c=workspace_config();c.plan=PlanConfig([PlanSegment([[40.,50.],[90.,50.]],width=2.,height=-1.)])
    s=Simulation(c);f=s.install_conventional(fleet_config());truck=f.machines[-1]
    truck.cargo_kg=s.terrain.scoop(55.25,50.25,400.,radius=.01)
    assert truck.cargo_kg==400
    f.excavated_mass_kg=truck.cargo_kg;f.excavated_m3=truck.cargo_kg/c.terrain.density
    before=s.terrain.height.copy();s._terrain_active=True;s.state=MissionState.COMPLETED
    s.step()
    assert s.time==1 and not np.array_equal(before,s.terrain.height)
    assert abs(s.report()['mass_error_kg'])<1e-7
