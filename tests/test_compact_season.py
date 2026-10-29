import json
import os
from copy import deepcopy
import subprocess
import sys
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.season_power import design_season,ready_season_simulation,QATTARA_LATITUDE
from ats.winter_power import design_winter,ready_simulation,design_status,default_request
from ats.checkpoint import save_state,load_state
from ats.self_solar import deployment_rectangles
from ats.compact_layout import rectangles,safe
from ats.base_layout import layout_footprints


def source():
    c=experiment_config(with_map_size(Config(),64));c.base.latitude_deg=QATTARA_LATITUDE
    return c


def overlaps(a,b):
    x,y,w,h=a;xx,yy,ww,hh=b
    return x<xx+ww-1e-8 and x+w>xx+1e-8 and y<yy+hh-1e-8 and y+h>yy+1e-8


def test_full_winter_formation_is_square_with_real_deployed_clearances():
    s,d=ready_simulation(source());spec=s.base.layout_spec;g=spec['grid']
    assert d['platform_count']==63 and g['deployed_dimensions_local_m']==[117.,117.]
    spaces=deployment_rectangles(spec,s.base.position);lanes=rectangles(spec,s.base.position)
    assert safe(spec,s.base.position,s.terrain)
    for i,a in enumerate(spaces):
        assert not any(overlaps(a,b) for b in spaces[i+1:])
        assert not any(overlaps(a,b) for b in lanes)
    services=[a for a in spec['platforms'] if a['role']=='SERVICE']
    assert len({a['offset_m'][0] for a in services})==1
    assert len({a['offset_m'][0] for a in spec['platforms'] if a['role']!='PV'})>1
    for i,xy in enumerate(spec['dock_offsets']):
        from ats.base_layout import rotate_offset
        np.testing.assert_allclose(s.slot_position(i)[:2],s.base.position[:2]+rotate_offset(spec['parameters'],xy))
        assert np.linalg.norm(s.slot_position(i)[:2]-s.wait_position(i)[:2])>=6.5


def test_season_audits_every_day_night_charge_ports_and_repeatability():
    c=source();before=deepcopy(c);sized,d=design_season(c)
    assert c==before and sized.base.start_day_of_year==172
    assert d['checked_days']==214 and d['season_start_day']==60 and d['season_end_day']==273
    assert d['limiting_date']=='01.03' and d['longest_night_day']==60
    assert d['panel_area_m2']==4866 and d['base_capacity_kwh']==2375 and d['platform_count']==45
    assert d['rotation_feasible'] and sized.logistics.individual_chargers
    assert all(p['required_kw']<=p['available_kw'] for p in d['service_charging'])
    for a in d['season_checks']:
        assert a['repeatable'] and a['one_day_covered']
        assert a['solar_kwh']>=a['demand_kwh']
        assert a['storage_check']['unserved_kwh']<1e-7
        assert a['storage_check']['daily_storage_drift_kwh']>=-1e-7
        assert a['inverter_kw']>=a['connected_peak_kw']
    winter,w=design_winter(c)
    assert d['panel_area_m2']<w['panel_area_m2'] and d['base_capacity_kwh']<w['base_capacity_kwh']
    assert sized.base.battery/3600000<d['limiting']['demand_kwh']*2  # nights, not months


def test_latitude_and_dates_require_review_without_changing_equipment():
    c,d=design_season(source());before=deepcopy(c)
    for day in (60,172,273):assert design_status(c,day_of_year=day)[0]=='ready'
    for day in (59,274,355):assert design_status(c,day_of_year=day)[0]=='stale'
    assert design_status(c,latitude=40)[0]=='stale' and c==before
    c.base.start_day_of_year=300
    sized,d=design_season(c);assert sized.base.start_day_of_year==300
    assert design_status(sized)[0]=='stale'


@pytest.mark.parametrize('latitude',[0.,-31.,55.])
def test_other_latitudes_are_checked_instead_of_reusing_qattara(latitude):
    c=source();c.base.latitude_deg=latitude;c,d=design_season(c)
    assert d['latitude_deg']==latitude and all(a['repeatable'] for a in d['season_checks'])


def test_insufficient_service_bus_cannot_be_hidden_by_full_starting_batteries():
    c=source();c.logistics.battery_platforms={'service_bus_kw':10.}
    with pytest.raises(ValueError,match='сервисной платформы'):design_season(c)


def test_impossible_krot_full_time_duty_is_reported():
    request=default_request();request.excavator_utilization=1.
    with pytest.raises(ValueError,match='оборота'):design_season(source(),request)


def test_compact_season_save_and_exact_continuation(tmp_path):
    s,d=ready_season_simulation(source());s.request_landing();s.run(95)
    positions={a['id']:deepcopy(a['offset_m']) for a in s.base.layout_spec['platforms']}
    path=tmp_path/'season.atsstate';save_state(path,s);q=load_state(path).simulation
    assert s.report()==q.report()
    for _ in range(140):s.step();q.step()
    assert s.report()==q.report() and s.logistics.field_ready
    assert {a['id']:a['offset_m'] for a in s.base.layout_spec['platforms']}==positions
    assert abs(s.report()['energy_error_j'])<.002 and abs(s.report()['mass_error_kg'])<1e-7
    assert s.base.layout_spec['formation']=='compact_grid_v1'


def test_two_days_on_the_one_second_bus_with_real_converter_limits():
    from scripts.validate_winter_power import audit_bus
    c,d=design_season(source());r=audit_bus(c,d)
    assert r['unserved_kwh']<1e-7 and r['daily_storage_drift_kwh']>=-1e-6
    assert r['minimum_soc']>=d['request']['reserve_soc']-1e-6
    assert abs(r['energy_error_j'])<.2


def test_season_group_roundtrip_and_resume_preserve_layout_and_ground(tmp_path):
    from ats.landscape import Landscape,ElevationRaster,workspace_config
    from ats.complexes import ComplexProject,ComplexTemplate
    from ats.simulation import Simulation
    from ats.plan import PlanConfig,PlanSegment
    from tests.test_regional import advance
    r=ElevationRaster(np.zeros((8,8)),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=150.,dy=150.,latitude_deg=QATTARA_LATITUDE))
    land=Landscape(r,cell_size=1.)
    c=workspace_config(land,experiment_config(Config(),drones=2,landing_slots=2,containers=2),[64.,64.],320)
    c.plan=PlanConfig([PlanSegment([[20,80],[26,80]],width=2,height=-.2)])
    request=default_request();request.drone_utilization=.05
    c,d=design_season(c,request);p=ComplexProject(Simulation(c),land);p.enable_shared();p.add_shared(ComplexTemplate.capture(c))
    for u in p.units:u.simulation.request_landing()
    advance(p,lambda:all(u.simulation.logistics.field_ready for u in p.units))
    t=p.shared_site.core;mass=t.scoop(22.,80.,20.);t.deposit(250,250,mass)
    ground=t.height.copy();ids=p.inventory_ids();poses={u.id:{a['id']:deepcopy(a['offset_m']) for a in u.simulation.base.layout_spec['platforms']} for u in p.units}
    p.move_window([112.,64.]);advance(p,lambda:p.transition.phase=='DRIVE')
    path=tmp_path/'compact-moving.atsstate';save_state(path,p.active.simulation,p.landscape,complex_project=p);q=load_state(path).complex_project
    for _ in range(5000):
        if p.transition is None:break
        p.step();q.step();assert not p.blocked,p.report()
    assert p.transition is None and q.report()==p.report()
    p.move_window([64.,64.]);advance(p,lambda:p.transition is None,5000)
    np.testing.assert_array_equal(p.shared_site.core.height,ground)
    assert p.inventory_ids()==ids
    for u in p.units:
        assert {a['id']:a['offset_m'] for a in u.simulation.base.layout_spec['platforms']}==poses[u.id]
        assert u.simulation.config.energy_design['kind']=='season'
    r=p.report();assert abs(r['mass_error_kg'])<1e-7 and abs(r['equipment_mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002


def test_access_strip_rejects_water_or_new_cut_under_its_edge():
    s,d=ready_season_simulation(source());spec=s.base.layout_spec
    from ats.site_ground import rectangle_patch
    rect=rectangles(spec,s.base.position)[0];patch=rectangle_patch(s.terrain,rect)
    s.terrain.design_work[patch]=True;s.terrain.revision+=1
    assert not safe(spec,s.base.position,s.terrain)


def test_old_v9_without_grid_is_preserved_on_load_and_small_shipment(tmp_path):
    import zipfile
    from tests.test_self_solar import world
    s=world(68);spec=s.base.layout_spec
    for key in ('formation','access_lanes_local','grid','hub_slots_local','service_header_y','service_aisles_x'):spec.pop(key,None)
    positions={a['id']:deepcopy(a['offset_m']) for a in spec['platforms']}
    path=tmp_path/'previous-v9.atsstate';save_state(path,s)
    with zipfile.ZipFile(path) as archive:files={n:archive.read(n) for n in archive.namelist()}
    manifest=json.loads(files['manifest.json']);manifest['version']=9;files['manifest.json']=json.dumps(manifest).encode()
    with zipfile.ZipFile(path,'w') as archive:
        for name,value in files.items():archive.writestr(name,value)
    q=load_state(path).simulation
    assert 'formation' not in q.base.layout_spec and q.report()==s.report()
    q.logistics.replenish_batteries(q.drone.battery_ids[0],2,0.)
    assert {a['id']:a['offset_m'] for a in q.base.layout_spec['platforms']}==positions


def test_season_cli_and_winter_option_are_explicit(tmp_path):
    path=tmp_path/'season.json'
    p=subprocess.run([sys.executable,'-m','ats','--headless','--energy-preset','qattara-season','--steps','0','--json',str(path)],capture_output=True,text=True,encoding='utf-8',timeout=40)
    assert p.returncode==0,p.stderr
    r=json.loads(path.read_text(encoding='utf-8'));d=r['energy_design']
    assert r['latitude_deg']==QATTARA_LATITUDE and r['day_of_year']==172 and d['kind']=='season'
    assert r['energy_design_status']['state']=='ready' and d['checked_days']==214


def test_new_gui_default_and_manual_configuration_preservation(monkeypatch,tmp_path):
    from ats import __main__ as cli
    from ats import gui
    from ats.experiments import save_configuration
    captured={}
    def launch(config,**kwargs):captured.update(config=config,kwargs=kwargs);return 0
    monkeypatch.setattr(gui,'launch',launch);monkeypatch.setattr(sys,'argv',['ats'])
    assert cli.main()==0
    c=captured['config'];assert c.base.latitude_deg==QATTARA_LATITUDE and c.base.start_day_of_year==172
    assert c.energy_design['kind']=='season' and captured['kwargs']['session'].simulation.base.layout_spec['formation']=='compact_grid_v1'
    manual=source();manual.base.latitude_deg=23.;manual.base.start_day_of_year=300
    path=tmp_path/'manual.json';save_configuration(path,manual)
    monkeypatch.setattr(sys,'argv',['ats','--configuration',str(path)])
    assert cli.main()==0
    assert captured['config'].energy_design is None and captured['config'].base.latitude_deg==23.
    assert captured['config'].base.start_day_of_year==300
    monkeypatch.setattr(sys,'argv',['ats','--configuration',str(path),'--energy-preset','qattara-season'])
    assert cli.main()==0
    assert captured['config'].base.latitude_deg==23. and captured['config'].base.start_day_of_year==300


def test_ui_shows_season_coordinates_limiting_day_and_outside_preview(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets,QtCore
    from ats.gui import MainWindow
    s,d=ready_season_simulation(source());app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host=MainWindow(s.config);host.timer.stop();host.show_power_design();dialog=host.power_window
    assert dialog.climate_scope.currentData()=='season' and '01.03' in dialog.info.text() and '31.078' in dialog.info.text()
    host.calendar.setDate(QtCore.QDate(2025,10,1));host.update_energy_status()
    assert host.energy_status.property('designState')=='stale'
    assert host.sim.config.base.start_day_of_year==172
    dialog.climate_scope.setCurrentIndex(dialog.climate_scope.findData('winter'));dialog.calculate_winter()
    assert 'winter_design' in dialog.result and 'Зимний' in dialog.info.text()
    host.close();app.processEvents()
