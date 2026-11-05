"""Passive, conservative salt tracing of the conditional reservoir experiment."""
import json
import math
import os
import subprocess
import sys
import zipfile
import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from ats.hydrology import BasinCurve,Flooding
from ats.landscape import Landscape
from tests.test_hydrology import flat


def legacy_data(model):
    data=model.to_data();data['version']=data['state']['version']=1
    for name in list(data['state']):
        if 'salt' in name:del data['state'][name]
    for name in ('inflow_salinity_g_l','solubility_g_l'):del data['config'][name]
    for row in data['state']['history']:
        row.pop('salinity_g_l',None);row.pop('salt_mass_kg',None)
    return data


def assert_balanced(m):
    m.validate()
    assert abs(m.balance_error)<1e-6
    assert abs(m.salt_balance_error)<max(1e-8,m.salt_mass_kg*1e-12)


def test_fresh_basin_fills_at_source_concentration_and_source_change_affects_future_only():
    m=flat();m.advance(100.)
    assert m.salt_mass_kg==pytest.approx(3800.)
    assert m.salinity_g_l==pytest.approx(38.)
    m.config.inflow_salinity_g_l=20.
    assert m.salt_mass_kg==pytest.approx(3800.)
    m.advance(100.)
    assert m.salinity_g_l==pytest.approx(29.)
    assert m.imported_salt_kg==pytest.approx(5800.) and m.exported_salt_kg==0
    assert_balanced(m)


def test_evaporation_concentrates_salt_then_leaves_dry_residue_and_refill_redissolves():
    m=flat(volume=500,evap=100,flow=0.);m.initialize_salt(38.)
    m.advance(25*86400.)
    assert m.volume_m3==pytest.approx(250.) and m.salinity_g_l==pytest.approx(76.)
    assert m.salt_mass_kg==19000. and m.exported_salt_kg==0
    m.advance(30*86400.)
    assert m.volume_m3==0 and m.salinity_g_l is None
    assert m.dissolved_salt_kg==0 and m.precipitated_salt_kg==19000.
    m.config.evaporation_mm_day=0;m.config.flow_limit_m3_s=1.;m.config.inflow_salinity_g_l=0
    m.advance(100.)
    assert m.volume_m3==pytest.approx(100.) and m.salinity_g_l==pytest.approx(190.)
    assert m.precipitated_salt_kg==0 and m.salt_mass_kg==19000.
    assert_balanced(m)


def test_solubility_changes_partition_without_changing_total_salt():
    m=flat(volume=500.);m.initialize_salt(100.)
    m.config.solubility_g_l=50.
    assert m.salinity_g_l==50 and m.dissolved_salt_kg==25000 and m.precipitated_salt_kg==25000
    m.config.solubility_g_l=300.
    assert m.salinity_g_l==100 and m.precipitated_salt_kg==0
    assert_balanced(m)


def test_dry_throughflow_deposits_salt_despite_zero_water_and_no_numeric_solves():
    m=flat(area=1000,evap=100,flow=.0001);m.advance(365*86400.)
    assert m.volume_m3==0 and m.calculations==0
    assert m.salt_mass_kg==pytest.approx(m.inflow_m3*38.)
    assert m.precipitated_salt_kg==m.salt_mass_kg and m.salinity_g_l is None
    assert_balanced(m)


def test_constant_water_equilibrium_still_accumulates_salt_analytically():
    m=flat(volume=400.,evap=10.);m.initialize_salt(38.)
    q=m.config.evaporation(0)*100;m.config.flow_limit_m3_s=q
    m.advance(365*86400.)
    assert m.calculations==0 and m.volume_m3==400.
    assert m.salinity_g_l==pytest.approx(38.+q*365*86400*38/400)
    assert m.history[-1]['salt_mass_kg']==m.salt_mass_kg
    assert_balanced(m)


@pytest.mark.parametrize('limit',[300.,100.])
def test_reduced_basin_capacity_exports_only_dissolved_well_mixed_salt(limit):
    m=flat(volume=500.);m.initialize_salt(300.);m.config.solubility_g_l=limit
    old=m.salt_mass_kg;solid=m.precipitated_salt_kg
    m.replace_curve(BasinCurve.create((3,3),[4],[-2.5],[100.]))
    assert m.volume_m3==250 and m.spilled_m3==250
    assert m.exported_salt_kg==pytest.approx(limit*250)
    assert m.salt_mass_kg==pytest.approx(old-m.exported_salt_kg)
    assert m.precipitated_salt_kg==pytest.approx(solid)
    assert_balanced(m)


def test_manual_assumption_starts_new_salt_ledger_without_rewriting_water_history():
    m=flat();m.advance(100.);water=(m.volume_m3,m.inflow_m3,m.elapsed_seconds,m.evaporated_m3)
    m.initialize_salt(12.)
    assert (m.volume_m3,m.inflow_m3,m.elapsed_seconds,m.evaporated_m3)==water
    assert m.initial_salt_kg==pytest.approx(1200.) and m.imported_salt_kg==0
    assert m.salt_origin=='manual' and m.salt_baseline_seconds==100.
    m.advance(100.)
    assert m.imported_salt_kg==pytest.approx(3800.)
    assert_balanced(m)


def test_salt_tracer_leaves_all_water_fluxes_and_solver_decisions_unchanged():
    a=flat(volume=400,evap=10,flow=.0001);a.initialize_salt(38.)
    b=Flooding.from_data(a.to_data());b.config.inflow_salinity_g_l=0.;b.initialize_salt(0.)
    a.config.season_amplitude=b.config.season_amplitude=.5
    a.advance(80*86400.);b.advance(80*86400.)
    for key in ('volume_m3','inflow_m3','evaporated_m3','spilled_m3','elapsed_seconds','calculations','skipped_seconds'):
        assert getattr(a,key)==getattr(b,key)
    assert a.salinity_g_l>b.salinity_g_l
    assert_balanced(a);assert_balanced(b)


@pytest.mark.parametrize('volume,spill,known',[(0.,0.,True),(100.,0.,False),(0.,1.,False)])
def test_legacy_water_import_does_not_invent_measured_salt(volume,spill,known):
    m=flat(volume=volume);m.advance(100.)
    if spill:m.volume_m3-=spill;m.spilled_m3+=spill
    old=legacy_data(m);water={k:getattr(m,k) for k in ('volume_m3','inflow_m3','spilled_m3','elapsed_seconds')}
    q=Flooding.from_data(old)
    assert all(getattr(q,k)==v for k,v in water.items())
    assert q.version==2 and q.salt_known==known
    assert all('salinity_g_l' not in r for r in q.history)
    if known:
        assert q.salt_origin=='legacy_assumed_inflow'
        assert q.salt_mass_kg==pytest.approx(q.inflow_m3*38.)
        assert_balanced(q)
    else:
        assert q.salt_origin=='legacy_unknown' and q.report()['salt_mass_kg'] is None
        q.advance(10.);assert q.salinity_g_l is None
        q.initialize_salt(20.);assert_balanced(q)
    q.validate()


def test_salted_region_and_checkpoint_resume_exactly_and_preserve_soil_and_equipment(tmp_path):
    from ats.checkpoint import save_state,load_state
    from tests.test_regional import region_group
    p=region_group(count=1,soil=True);land=p.landscape
    land.flooding=m=flat();m.curve.shape=land.raster.height.shape;m.advance(123.)
    source=p.report();ids=p.inventory_ids();height=p.shared_site.core.height.copy()
    path=tmp_path/'salt.atsregion';land.save(path);r=Landscape.load(path)
    assert r.flooding.to_data()==m.to_data()
    path=tmp_path/'salt.atsstate';save_state(path,p.active.simulation,land,complex_project=p)
    q=load_state(path).complex_project
    q.landscape.flooding.advance(100.);r.flooding.advance(100.);m.advance(100.)
    assert q.landscape.flooding.to_data()==r.flooding.to_data()==m.to_data()
    assert q.report()==source and q.inventory_ids()==ids
    np.testing.assert_array_equal(q.shared_site.core.height,height)
    assert_balanced(m)


def test_previous_state_v15_and_region_v4_migrate_without_mutating_source_files(tmp_path):
    from ats.checkpoint import save_state,load_state
    from tests.test_regional import region_group
    p=region_group(count=1,soil=True);p.landscape.flooding=m=flat();m.curve.shape=p.landscape.raster.height.shape;m.advance(100.)
    new=tmp_path/'new.atsstate';old=tmp_path/'old.atsstate';save_state(new,p.active.simulation,p.landscape,complex_project=p)
    with zipfile.ZipFile(new) as a,zipfile.ZipFile(old,'w') as b:
        for name in a.namelist():
            payload=a.read(name)
            if name=='manifest.json':
                manifest=json.loads(payload);manifest['version']=15
                for node in manifest['nodes']:
                    if node.get('type')=='ats.hydrology.Flooding':
                        node['attrs']['version']=1
                        node['attrs']={k:v for k,v in node['attrs'].items() if 'salt' not in k}
                    elif node.get('type')=='ats.hydrology.FloodConfig':
                        for k in ('inflow_salinity_g_l','solubility_g_l'):node['attrs'].pop(k,None)
                    elif node['kind']=='dict':
                        node['items']=[kv for kv in node['items'] if kv[0] not in ('salinity_g_l','salt_mass_kg')]
                payload=json.dumps(manifest).encode()
            b.writestr(name,payload)
    original=old.read_bytes();q=load_state(old)
    assert old.read_bytes()==original and q.landscape.flooding.salt_origin=='legacy_assumed_inflow'
    assert q.landscape.flooding.salt_mass_kg==pytest.approx(3800.)
    assert q.complex_project.report()==p.report()
    region=tmp_path/'old.atsregion';p.landscape.save(region)
    with np.load(region,allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
    metadata=json.loads(str(arrays['metadata']));metadata['version']=4;metadata['flooding']=legacy_data(m)
    arrays['metadata']=np.asarray(json.dumps(metadata))
    with region.open('wb') as f:np.savez_compressed(f,**arrays)
    original=region.read_bytes();r=Landscape.load(region)
    assert region.read_bytes()==original and r.flooding.salt_mass_kg==pytest.approx(3800.)
    assert r.flooding.report()==q.landscape.flooding.report()


@pytest.mark.parametrize('field,value',[('inflow_salinity_g_l',-1.),('solubility_g_l',0.),('inflow_salinity_g_l',math.nan)])
def test_invalid_salt_parameters_are_rejected(field,value):
    m=flat();setattr(m.config,field,value)
    with pytest.raises(ValueError):m.config.validate()


def test_salt_balance_corruption_and_inconsistent_versions_are_rejected():
    m=flat();m.advance(100.);bad=m.to_data();bad['state']['salt_mass_kg']+=1
    with pytest.raises(ValueError,match='баланс соли'):Flooding.from_data(bad)
    bad=m.to_data();bad['version']=1
    with pytest.raises(ValueError,match='версии'):Flooding.from_data(bad)


def test_ui_salt_settings_graph_and_cancel_do_not_reset_water_or_earthworks(monkeypatch):
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.config import Config
    from ats.landscape_ui import LandscapeDialog
    from ats.hydrology_ui import FloodingDialog,SaltDialog
    from tests.test_channel import coastal_dem
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(Config());w.timer.stop();owner=LandscapeDialog(w)
    owner.region=Landscape(coastal_dem(gap=100));owner.show_region()
    owner.points=[[50,150],[250,150]];owner.width.setValue(2);owner.depth.setValue(1)
    owner.project_side.setCurrentIndex(owner.project_side.findData(128));assert owner.prepare_channel()
    water=FloodingDialog(owner);water.timer.stop();water.show();app.processEvents()
    try:
        m=water.model;before=w.sim.report()
        water.instant.click();water.inputs['evaporation_mm_day'].setValue(0.);water.day.click()
        assert 'Солёность:' in water.status.text() and m.salinity_g_l==pytest.approx(38.)
        assert len(water.salt_line.getData()[0])>0
        mass=m.salt_mass_kg;clock=m.elapsed_seconds;volume=m.volume_m3
        dialog=SaltDialog(water);dialog.source.setValue(100.);dialog.reject()
        assert m.config.inflow_salinity_g_l==38. and m.salt_mass_kg==mass
        def apply(dialog):
            dialog.source.setValue(40.);dialog.limit.setValue(250.)
            dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
            return dialog.result()
        monkeypatch.setattr(SaltDialog,'exec',apply);water.salt_button.click()
        assert m.config.inflow_salinity_g_l==40. and m.config.solubility_g_l==250.
        assert m.salt_mass_kg==mass and m.elapsed_seconds==clock and m.volume_m3==volume
        assert w.sim.report()==before
    finally:water.close();owner.close();w.close();app.processEvents()


def test_cli_salt_report_and_resume_match_continuous_run(tmp_path):
    from tests.test_channel import coastal_dem
    from ats.channel import ChannelProject
    land=Landscape(coastal_dem(gap=100));project=ChannelProject.create(land.raster,[[50,150],[250,150]],width_m=2,depth_m=1,side_m=128)
    land.channel_project=project.to_data();region=tmp_path/'start.atsregion';land.save(region)
    state=tmp_path/'first.atsstate';first=tmp_path/'first.json';resumed=tmp_path/'resumed.json';continuous=tmp_path/'continuous.json'
    def run(*args):
        p=subprocess.run([sys.executable,'-m','ats','--headless','--steps','0',*map(str,args)],capture_output=True,text=True,encoding='utf-8',timeout=40)
        assert p.returncode==0,p.stderr
    base=('--landscape',region,'--channel','--instant-channel','--evaporation-mm-day','0','--salt-inflow-g-l','40')
    run(*base,'--flood-days','2','--save-state',state,'--flood-report',first)
    run('--load-state',state,'--flood-days','3','--flood-report',resumed)
    run(*base,'--flood-days','5','--flood-report',continuous)
    a=json.loads(resumed.read_text());b=json.loads(continuous.read_text())
    assert a['report']['salinity_g_l']==pytest.approx(40.)
    for key in ('volume_m3','inflow_m3','salt_mass_kg','imported_salt_kg'):
        assert a['report'][key]==pytest.approx(b['report'][key],rel=1e-12,abs=1e-7)
    assert a['report']['salt_balance_error_kg']==pytest.approx(0.,abs=1e-7)
