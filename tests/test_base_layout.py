from copy import deepcopy
import os
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.simulation import Simulation
from ats.base_layout import solve_layout,design_continuous,support_report
from ats.checkpoint import save_state,load_state


def config(side=128):
    c=experiment_config(with_map_size(Config(),side));c.base.layout={}
    c.logistics.battery_platforms=None  # retain the historical aggregate-layout regressions
    c.logistics.self_solar=None
    return c


def test_layout_carries_equipment_with_real_frames_volume_and_contact_area():
    c=config();small=solve_layout(c)
    c.base.equipment.batteries[0].quantity=200;large=solve_layout(c)
    assert large['platform_count']>small['platform_count']
    assert large['packed_volume_m3']>small['packed_volume_m3']
    assert large['packed_gross_mass_kg']==pytest.approx(large['packed_payload_kg']+large['chassis_mass_kg'])
    for p in large['platforms']:
        assert p['payload_kg']<=large['parameters']['platform_payload_kg']
        assert p['pressure_kpa']<=large['parameters']['max_pressure_kpa']+1e-9
        assert p['pressure_kpa']==pytest.approx(p['gross_mass_kg']*9.80665/p['contact_area_m2']/1000)
    s=Simulation(config());w=s.logistics
    chassis=sum(o.mass_kg for o in w.inventory.items.values() if o.kind in ('CHASSIS','PLATFORM_CHASSIS'))
    assert chassis==pytest.approx(s.base.config.equipment.chassis_mass_kg)
    assert len([o for o in w.inventory.items.values() if o.kind in ('CHASSIS','PLATFORM_CHASSIS')])==s.base.layout_spec['platform_count']
    assert w.imported_mass_kg==0


def test_ground_bearing_can_reject_a_heavy_base_and_restores_when_tracks_are_sufficient():
    from ats.soil import SoilProfile
    c=config();c.soil_profile=SoilProfile.qattara();s=Simulation(c)
    assert support_report(s.base,s.terrain,s.drones)['safe']
    s.terrain.soil.profile.materials[0].bearing_kpa=.1
    assert not support_report(s.base,s.terrain,s.drones)['safe']
    assert not s.base.safe_path(s.base.position[:2]+[1,0],s.terrain)


def test_hot_storage_recomposes_platforms_without_fabricating_charge_and_roundtrips(tmp_path):
    s=Simulation(config(160));s.run(15);before=s.report();inventory=s.logistics.inventory
    s.set_base_battery_count(200);after=s.report()
    assert s.time==before['time_seconds'] and after['base_kwh']==before['base_kwh']
    assert after['base_specification']['layout']['platform_count']>before['base_specification']['layout']['platform_count']
    assert abs(after['energy_error_j'])<.002 and after['inventory_imported_mass_kg']>0
    path=tmp_path/'layout.atsstate';save_state(path,s);other=load_state(path).simulation
    assert other.report()==s.report()
    for _ in range(20):s.step();other.step()
    assert other.report()==s.report()


def test_continuous_design_closes_energy_geometry_and_chassis_idle_iteration():
    c=experiment_config(Config());c.base.latitude_deg=31
    from ats.complexes import ComplexTemplate
    designed,r=design_continuous(c)
    assert r['request']['work_hours']==24 and r['base_layout']['platform_count']>1
    assert designed.base.idle_power==r['base_layout']['control_power_w']
    assert r['storage_check']['daily_storage_drift_kwh']>=-1e-7
    template=ComplexTemplate.capture(designed,power_result=r);s=Simulation(template.config)
    assert s.state.value=='RUNNING',s.reason
    assert s.report()['initial_energy_kwh']==pytest.approx(r['initial_stored_kwh'])
    assert s.base.layout_spec['platform_count']==r['base_layout']['platform_count']


def test_relocation_spends_the_same_energy_as_its_flat_route_estimate():
    s=Simulation(config());b=s.base
    target=next(b.position[:2]+delta for delta in ([1.,0.],[-1.,0.],[0.,1.],[0.,-1.]) if b.safe_path(b.position[:2]+delta,s.terrain))
    assert b.safe_path(target,s.terrain)
    energy=b.route_energy(target,s.terrain,s.drones);before=b.battery
    b.destination=target.copy()
    while b.destination is not None:b.tick(1.,9.,s.drones,[False]*len(s.drones),terrain=s.terrain)
    assert before-b.battery==pytest.approx(energy,abs=1e-6)


def test_two_platform_base_advances_along_the_canal_and_finishes():
    from tests.test_workfront import canal
    c=canal().config;c.base.layout={'platform_payload_kg':1000.}
    s=Simulation(c);start=s.base.position.copy()
    assert s.base.layout_spec['platform_count']>=2
    for _ in range(8000):
        old=s.base.position.copy();s.step()
        distance=np.linalg.norm(s.base.position-old)
        assert distance<=s.config.base.speed+1e-9
        if distance>0:assert all(d.grounded for d in s.drones)
        if s.state.value!='RUNNING':break
    r=s.report()
    assert r['state']=='COMPLETED',r['reason']
    assert s.base.relocations>=2 and s.base.position[0]>start[0]+25
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002


def test_compact_main_window_fits_small_desktop_and_details_open_separately():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(config());w.timer.stop();w.resize(1024,700);w.show();app.processEvents()
    assert w.minimumSizeHint().width()<=1024 and w.minimumSizeHint().height()<=700
    assert not w.settings_window.isVisible() and not w.fleet_window.isVisible()
    w.settings_window.show();w.fleet_window.show();app.processEvents()
    assert w.settings_tabs.count()==3 and w.fleet_table.isVisible()
    assert w.map.isVisible() and w.map.height()>250
    w.close();app.processEvents()


def test_completion_waits_for_an_atomic_battery_swap_and_plan_accepts_roundoff():
    from ats.plan import PlanConfig
    c=experiment_config(Config());c.plan=PlanConfig([]);s=Simulation(c);d=s.drone
    old=d.battery;d.battery=d.config.battery*.25;s._initial_energy-=old-d.battery
    d.charge_requested=True;s.logistics.sync_batteries();s.step()
    assert s.logistics.battery_service.drone_swaps and s.state.value=='RUNNING'
    s.run(300)
    assert s.state.value=='COMPLETED' and not s.logistics.battery_service.drone_swaps
    d.cargo=2.1316282072803006e-14;s.apply_plan(PlanConfig([]))
    assert d.cargo==2.1316282072803006e-14  # the numerical residue is retained


def test_gui_continuous_preset_starts_a_real_mission_and_keeps_the_template():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c=experiment_config(Config());c.base.latitude_deg=31
    w=MainWindow(c);w.timer.stop();w.show_power_design();dialog=w.power_window
    dialog.calculate_continuous();assert dialog.result['request']['drone_utilization']==.85
    result=dialog.result;original=w.sim;dialog.start_template();w.timer.stop()
    assert w.sim is not original and w.sim.state.value=='RUNNING'
    assert w.config.base.layout['heading_deg']==pytest.approx(-90)
    assert w.config.base.panel_area==result['panel_area_m2'] and len(w.template_library)==1
    w.sim.run(20);assert abs(w.sim.report()['energy_error_j'])<.002
    w.close();app.processEvents()


def test_platforms_and_docks_follow_the_route_direction():
    c=config();s=Simulation(c)
    assert s.base.config.layout['heading_deg']==pytest.approx(-90)
    difference=s.slot_position(1)-s.slot_position(0)
    assert difference[0]==pytest.approx(c.fleet.pad_spacing) and abs(difference[1])<1e-10
    from ats.base_layout import platform_polygons
    polygons=list(platform_polygons(s.base.layout_spec,s.base.position))
    assert np.ptp(polygons[0][:,0])==pytest.approx(6.) and np.ptp(polygons[0][:,1])==pytest.approx(3.)
