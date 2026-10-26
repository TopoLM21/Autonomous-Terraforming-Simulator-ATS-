"""Independent hardware, common regional time, exclusive earthworks and saves."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.landscape import Landscape,ElevationRaster,workspace_config
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.complexes import ComplexTemplate,ComplexProject
from ats.checkpoint import save_state,load_state


def regional():
    raster=ElevationRaster(np.zeros((11,11)),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=100.,dy=100.))
    land=Landscape(raster)
    c=workspace_config(land,experiment_config(Config()),[64.,64.],64.)
    c.plan=PlanConfig([PlanSegment([[30,32],[34,32]],'dig',1,-.25)])
    return Simulation(c),land


def test_template_never_copies_terrain_cargo_reservations_or_current_soc(tmp_path):
    s,land=regional();s.run(150);before=s.report()
    t=ComplexTemplate.capture(s.config,'Два комплекса')
    assert t.config.initial_height is None and t.config.reference_height is None and t.config.plan is None and t.config.georeference is None
    path=tmp_path/'копия.atstemplate';t.save(path);other=ComplexTemplate.load(path)
    assert other.to_data()==t.to_data()
    assert s.report()==before
    other.config.base.equipment.panels[0].quantity=77
    assert t.config.base.equipment.panels[0].quantity!=77


def test_clones_have_separate_batteries_inventory_and_balanced_delivery_energy():
    s,land=regional();s.run(50);p=ComplexProject(s,land);t=ComplexTemplate.capture(s.config)
    u=p.add(t,origin=[140.,64.]);other=u.simulation
    assert other is not s and not np.shares_memory(s.terrain.height,other.terrain.height)
    assert s.logistics.inventory is not other.logistics.inventory
    assert s.logistics.inventory.items['base_battery_01'] is not other.logistics.inventory.items['base_battery_01']
    assert p.delivered_energy_kwh==pytest.approx(other.report()['initial_energy_kwh'])
    initial=other.base.battery;other.set_base_battery_count(20)
    assert other.base.battery==initial and len(s.logistics.base_battery_ids)==6
    for _ in range(100):p.step()
    assert s.hour==pytest.approx(other.hour,abs=1e-12)
    assert s.day_of_year==other.day_of_year and p.time==s.time==u.born_at+other.time
    r=p.report();assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<1e-4
    assert all(row['excavated_m3']>0 for row in r['complexes'])


def test_overlapping_work_zones_are_rejected_before_importing_any_hardware():
    s,land=regional();p=ComplexProject(s,land);template=ComplexTemplate.capture(s.config)
    before=p.report();items=set(s.logistics.inventory.items)
    for origin in ([64.,64.],[126.,64.]):
        with pytest.raises(ValueError,match='пересекаются'):p.add(template,origin=origin)
    assert p.report()==before and set(s.logistics.inventory.items)==items


def test_whole_project_snapshot_keeps_selected_complex_and_exact_continuation(tmp_path):
    s,land=regional();s.run(35);p=ComplexProject(s,land);t=ComplexTemplate.capture(s.config)
    p.templates.append(t);u=p.add(t,origin=[140.,64.]);p.select(u.id)
    for _ in range(100):p.step()
    path=tmp_path/'all.atsstate';save_state(path,u.simulation,land,{'complex_templates':p.templates},p)
    session=load_state(path);other=session.complex_project
    assert other.active.simulation is session.simulation and other.landscape is session.landscape
    assert other.templates is session.ui['complex_templates']
    for _ in range(100):p.step();other.step()
    assert p.report()==other.report()
    for a,b in zip(p.units,other.units):
        assert a.simulation.report()==b.simulation.report()
        np.testing.assert_array_equal(a.simulation.terrain.height,b.simulation.terrain.height)


def test_one_blocked_complex_pauses_the_whole_project():
    s,land=regional();p=ComplexProject(s,land);u=p.add(ComplexTemplate.capture(s.config),origin=[140.,64.])
    u.simulation.block('Контрольная остановка');before=p.time
    p.step();assert p.time==before and not p.running and p.report()['state']=='BLOCKED'


def test_energy_sizing_gui_template_library_and_project_snapshot(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s,land=regional();host=MainWindow(s.config);host.timer.stop();host.landscape=land
    host.show_power_design();dialog=host.power_window
    assert dialog.result is not None and dialog.result['storage_check']['unserved_kwh']<1e-7
    original=host.sim;original_report=original.report()
    path=tmp_path/'design.atstemplate';dialog.save_template(path)
    assert path.exists() and host.sim is original and host.sim.report()==original_report
    assert len(host.template_library)==1
    host.show_complexes();manager=host.complex_window;manager.timer.stop()
    # Use the actual small current recipe for a compact work-window test.
    p=host.complex_project;p.templates.append(ComplexTemplate.capture(host.config,'Исходная комплектация'))
    manager.refresh();manager.template.setCurrentIndex(1);manager.x.setValue(140);manager.y.setValue(64);manager.clone_unit()
    assert len(p.units)==2
    host.select_complex(p.units[1].id);host.timer.stop()
    assert host.sim is p.units[1].simulation
    save=tmp_path/'gui.atsstate';host.save_session(save);host.timer.stop()
    host.load_session(save);host.timer.stop()
    assert len(host.complex_project.units)==2 and len(host.template_library)==2
    host.close();app.processEvents()


def test_finished_complex_is_powered_off_while_its_calendar_keeps_up():
    s,land=regional();s.apply_plan(PlanConfig([]));s.step()
    assert s.state.value=='COMPLETED'
    p=ComplexProject(s,land);u=p.add(ComplexTemplate.capture(s.config),origin=[140.,64.])
    u.simulation.apply_plan(PlanConfig([PlanSegment([[30,32],[34,32]],'dig',1,-.25)]))
    energy=s.base.battery;generated=s.base.generated
    for _ in range(30):p.step()
    assert s.base.battery==energy and s.base.generated==generated
    assert s.time==p.time==u.born_at+u.simulation.time
    assert s.hour==pytest.approx(u.simulation.hour)
    assert s.drone.activity.report()['total_seconds']==s.time


def test_cli_energy_report_and_calculated_template_are_executable(tmp_path):
    report=tmp_path/'power.json';template=tmp_path/'plant.atstemplate'
    def run(*args):
        r=subprocess.run([sys.executable,'-m','ats','--headless',*map(str,args)],capture_output=True,text=True,encoding='utf-8',timeout=45)
        assert r.returncode==0,r.stderr
        return json.loads(r.stdout)
    run('--steps',0,'--power-report',report,'--save-complex-template',template,'--work-hours',12,'--work-start-hour',6)
    design=json.loads(report.read_text(encoding='utf-8'));r=run('--complex-template',template,'--steps',10)
    assert r['panel_area_m2']==design['panel_area_m2'] and r['base_capacity_kwh']==design['base_capacity_kwh']


def test_real_version_two_layered_save_keeps_soil_and_exact_continuation(tmp_path):
    import base64
    def check_historical_report(actual,expected):
        actual=json.loads(json.dumps(actual))
        # Python 3.12 changed float sum() reduction. Only the two rolling
        # statistics may differ by a few ULP across runtimes; recorded loads,
        # physics and the rest of the historical report must remain exact.
        for kind in ('drones','retired_drones'):
            for row,want in zip(actual.get(kind,[]),expected.get(kind,[])):
                for key in ('last24_mass_kg','last24_volume_m3'):
                    assert row[key]==pytest.approx(want[key],rel=0,abs=1e-11)
                    row[key]=want[key]
        assert actual==expected
    fixture=Path(__file__).parent/'fixtures'/'soil-v2'/'layered'
    path=tmp_path/'v2.atsstate';path.write_bytes(base64.b64decode(fixture.with_suffix('.b64').read_text(encoding='ascii')))
    expected=json.loads(fixture.with_suffix('.json').read_text(encoding='utf-8'))
    original=path.read_bytes();session=load_state(path);s=session.simulation
    assert session.complex_project is None and s.terrain.soil is not None
    check_historical_report(s.report(),expected['initial'])
    s.run(expected['steps']);check_historical_report(s.report(),expected['continued'])
    save_state(tmp_path/'v3.atsstate',s)
    assert path.read_bytes()==original and load_state(tmp_path/'v3.atsstate').simulation.report()==s.report()


def test_project_delivery_ledger_cannot_omit_new_hardware_or_energy(tmp_path):
    s,land=regional();p=ComplexProject(s,land);p.add(ComplexTemplate.capture(s.config),origin=[140.,64.])
    r=p.report();assert r['delivered_equipment_mass_kg']>0 and abs(r['equipment_mass_error_kg'])<1e-7
    ids=p.inventory_ids();assert len(ids)==len(set(ids))
    p.delivered_energy_kwh=0
    assert abs(p.report()['energy_error_j'])>1e6
    path=tmp_path/'bad.atsstate';save_state(path,s,land,complex_project=p)
    with pytest.raises(ValueError,match='энергии'):load_state(path)
