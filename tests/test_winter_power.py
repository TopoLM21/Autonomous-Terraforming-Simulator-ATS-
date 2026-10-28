import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
import json
import subprocess
import sys
import numpy as np
import pytest
from ats.config import Config
from ats.maps import with_map_size
from ats.experiments import experiment_config,save_configuration,load_configuration
from ats.winter_power import design_winter,ready_simulation,design_status,default_request,audit_day,winter_day
from ats.checkpoint import save_state,load_state,Session


def source(latitude=31):
    c=experiment_config(with_map_size(Config(),64));c.base.latitude_deg=latitude
    return c


@pytest.mark.parametrize('latitude',[0,31,-31,55,-55])
def test_winter_design_is_repeatable_in_both_hemispheres_and_keeps_actual_date(latitude):
    c=source(latitude);before=deepcopy(c)
    sized,d=design_winter(c)
    assert c==before and sized.base.start_day_of_year==c.base.start_day_of_year
    assert d['mode']=='renewable' and d['rotation_feasible']
    for row in (d['current'],d['winter']):
        assert row['one_day_covered'] and row['repeatable']
        assert row['storage_check']['unserved_kwh']<1e-7
        assert row['storage_check']['daily_storage_drift_kwh']>=-1e-7
        assert abs(row['storage_check']['energy_error_kwh'])<1e-7
        assert row['inverter_kw']>=row['connected_peak_kw']
    assert d['winter']['solar_kwh']>=d['winter']['demand_kwh']
    assert d['winter_day_of_year']==winter_day(latitude)
    if latitude:assert (d['winter_day_of_year']>300)==(latitude>0)
    assert design_status(sized)[0]=='ready'
    json.dumps(d,allow_nan=False)


@pytest.mark.parametrize('latitude',[70,90,-70,-90])
def test_polar_night_has_finite_one_day_storage_without_renewable_claim(latitude):
    c,d=design_winter(source(latitude));w=d['winter']
    assert d['mode']=='one_day_storage'
    assert w['solar_kwh']==0 and w['one_day_covered'] and not w['repeatable']
    assert w['no_sun_base_autonomy_hours']>=24
    assert w['storage_check']['daily_storage_drift_kwh']<0
    assert d['base_battery_count']<=1000 and d['panels']==20
    assert design_status(c)[0]=='limited'


def test_ready_default_fits_real_panels_platforms_inventory_and_survives_tool_changes():
    s,d=ready_simulation(source(55));assert s.state.value=='RUNNING',s.reason
    assert d['panel_area_m2']>10000 and all(p.area_m2==1 for p in s.config.base.equipment.panels)
    assert s.logistics.field_rect is not None and s.base.layout_spec['platform_count']==d['platform_count']
    assert s.config.terrain.n<=1024 and s.config.terrain.n*s.terrain.cell>=d['required_map_side_m']
    before=s.drones[0].config.empty_mass
    s.logistics._tool_mass(s.drones[0],2.)
    mass=s.drones[0].config.empty_mass
    assert mass!=before and design_status(s.config)[0]=='ready'
    assert s.drones[0].config.empty_mass==mass
    assert s.report()['energy_design_status']['state']=='ready'
    assert abs(s.report()['energy_error_j'])<.002


def test_changed_latitude_or_manual_hardware_marks_review_without_mutating_inventory():
    s,d=ready_simulation(source());before=s.report();packs=list(s.logistics.inventory.items)
    assert design_status(s.config,40.)[0]=='stale'
    assert s.report()==before and list(s.logistics.inventory.items)==packs
    c=deepcopy(s.config);c.base.equipment.panels[0].quantity+=1
    assert design_status(c)[0]=='stale'
    assert s.config.base.equipment.panels[0].quantity==d['panels']


def test_saved_run_and_hardware_template_retain_the_design(tmp_path):
    from ats.complexes import ComplexTemplate
    from ats.simulation import Simulation
    s,d=ready_simulation(source())
    path=tmp_path/'winter.atsstate';save_state(path,s)
    q=load_state(path).simulation
    assert q.report()==s.report() and q.config.energy_design==d
    s.run(20);q.run(20);assert s.report()==q.report()
    path=tmp_path/'winter.json';save_configuration(path,s.config)
    c=load_configuration(path);assert design_status(c)[0]=='ready'
    template=ComplexTemplate.capture(c);template.save(tmp_path/'winter.atstemplate')
    t=ComplexTemplate.load(tmp_path/'winter.atstemplate');fresh=Simulation(t.config)
    assert fresh.state.value=='RUNNING',fresh.reason
    assert fresh.config.energy_design==d


def test_cli_winter_preset_reports_actual_equipment_and_does_not_change_manual_mode(tmp_path):
    path=tmp_path/'power.json'
    result=subprocess.run([sys.executable,'-m','ats','--headless','--energy-preset','winter24h','--latitude','31','--steps','0','--power-report',str(path)],capture_output=True,check=True)
    r=json.loads(result.stdout);power=json.loads(path.read_text())
    assert r['energy_design_status']['state']=='ready'
    assert r['panel_area_m2']==power['panel_area_m2']
    assert r['day_of_year']==172 and power['day_of_year']==354
    result=subprocess.run([sys.executable,'-m','ats','--headless','--energy-preset','manual','--steps','0'],capture_output=True,check=True)
    assert json.loads(result.stdout)['panel_area_m2']==20


def test_default_gui_entrypoint_prepares_winter_hardware_and_manual_entrypoint_keeps_it(monkeypatch):
    from ats import __main__,gui
    launches=[]
    monkeypatch.setattr(gui,'launch',lambda config,**kwargs:launches.append((config,kwargs)) or 0)
    monkeypatch.setattr(sys,'argv',['ats','--latitude','31'])
    assert __main__.main()==0
    config,options=launches[-1]
    assert config.energy_design['mode']=='renewable'
    assert config.base.start_day_of_year==172
    assert options['session'].simulation.config is config
    assert options['session'].simulation.state.value=='RUNNING'
    monkeypatch.setattr(sys,'argv',['ats','--latitude','31','--panel-area','100'])
    assert __main__.main()==0
    config,options=launches[-1]
    assert config.energy_design is None and config.base.panel_area==100
    assert 'session' not in options


def test_gui_climate_review_is_explicit_and_preserves_the_running_complex():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.power_ui import PowerDesignDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sim,d=ready_simulation(source());w=MainWindow(sim.config,session=Session(sim,ui={}));w.timer.stop()
    assert w.energy_status.property('designState')=='ready'
    before=sim.report();w.latitude.setValue(-31)
    assert w.energy_status.property('designState')=='stale' and sim.report()==before
    dialog=PowerDesignDialog(w)
    assert dialog.result is not None,dialog.info.text()
    assert dialog.result['winter_design']['latitude_deg']==-31
    assert dialog.result['winter_design']['winter_day_of_year']==171
    assert w.sim is sim and sim.report()==before
    dialog.calculate_continuous()
    assert not dialog.winter_check.isChecked() and 'winter_design' not in dialog.result
    assert dialog.result['day_of_year']==172
    dialog.calculate_winter()
    assert dialog.result['winter_design']['winter_day_of_year']==171
    reviewed=deepcopy(dialog.result['winter_design']['request'])
    dialog.controls['drone_utilization'].setValue(50)  # unapplied control edit
    dialog.start_template();w.timer.stop()
    assert w.sim is not sim and w.config.base.latitude_deg==-31
    assert w.config.energy_design['mode']=='renewable'
    assert w.config.energy_design['request']==reviewed
    assert w.energy_status.property('designState')=='ready'
    dialog.close();w.close();app.processEvents()


def test_materials_change_digging_effort_but_not_the_common_slope_limit():
    from ats.soil import SoilProfile,SoilMaterial
    from ats.terrain import Terrain
    fields=[];factors=[]
    for profile in (SoilProfile([SoilMaterial('Песок',1600,1)],[]),SoilProfile.qattara()):
        if len(profile.materials)>1:profile.thicknesses=[.01,.02]
        c=Config();c.soil_profile=profile;t=Terrain(c)
        load=t.scoop_load(16,16,1e6,radius=.5,max_volume=.1)
        factors.append(load.factor);t.deposit(26,25,load.mass,radius=.25,load=load)
        for _ in range(12):t.relax()
        fields.append(t.height.copy());assert abs(t.mass_change)<1e-7
    assert factors[1]>2*factors[0]
    np.testing.assert_allclose(fields[0],fields[1],rtol=0,atol=1e-12)


@pytest.mark.parametrize('latitude',[31,55,70])
def test_two_winter_days_on_the_actual_one_second_power_bus(latitude):
    from scripts.validate_winter_power import audit_bus
    c,d=design_winter(source(latitude));r=audit_bus(c,d)
    assert r['first_day_unserved_kwh']<1e-7
    assert r['first_day_final_kwh']>=c.base.battery/3_600_000*.2-1e-6
    assert abs(r['energy_error_j'])<.2
    if d['mode']=='renewable':
        assert r['unserved_kwh']<1e-7 and r['daily_storage_drift_kwh']>=-1e-6
    else:assert r['unserved_kwh']>0 and r['daily_storage_drift_kwh']<0


def test_group_window_transfer_retains_energy_policy_and_requires_geographic_review():
    from ats.landscape import Landscape,ElevationRaster,workspace_config
    from ats.complexes import ComplexProject
    from ats.simulation import Simulation
    from tests.test_regional import advance
    # A lower duty case fits a small regional fixture while testing real transfer.
    request=default_request();request.drone_utilization=.05
    r=ElevationRaster(np.zeros((8,8)),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=100.,dy=100.,latitude_deg=31.))
    land=Landscape(r,cell_size=1.)
    cfg=workspace_config(land,source(),[64.,64.],256)
    c,d=design_winter(cfg,request);c.logistics.field_deployment_seconds=2.
    # Changing deployment assumptions is a deliberate edit and needs review.
    assert design_status(c)[0]=='stale'
    c,d=design_winter(c,request);s=Simulation(c)
    assert s.state.value=='RUNNING',s.reason
    p=ComplexProject(s,land);p.move_window([96.,64.]);advance(p,lambda:p.transition is None,limit=8000)
    assert s.config.energy_design==d and s.config.base.latitude_deg==31
    assert design_status(s.config)[0]=='ready'
    assert abs(p.report()['energy_error_j'])<.002
    assert design_status(s.config,40.)[0]=='stale'
