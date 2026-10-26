"""Material mass, bucket volume, real V1 migration and deterministic resumes."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import base64
import json
import zipfile
import numpy as np
import pytest
from ats.config import Config
from ats.terrain import Terrain
from ats.soil import SoilField,SoilLoad,SoilProfile
from ats.simulation import Simulation
from ats.experiments import experiment_config,save_configuration,load_configuration
from ats.checkpoint import save_state,load_state
from ats.landscape import Landscape,workspace_config
from tests.test_landscape import raster


def layered(logistics=False):
    c=Config();c.soil_profile=SoilProfile.qattara();c.soil_profile.thicknesses=[.03,.05]
    return experiment_config(c) if logistics else c


def test_native_contacts_and_mass_inverse_are_exact():
    profile=SoilProfile.qattara();field=SoilField(profile,16,.25)
    assert not field.tiles
    _,preview=field.extract([17],[12.])
    np.testing.assert_allclose(preview.masses,[2*1600/16,6*1850/16,4*2200/16])
    assert not field.tiles
    for limit in (1.,200.,300.,893.75,1200.):
        change,load=field.extract([17],[12.],max_mass=limit)
        assert load.mass==pytest.approx(limit,abs=1e-10)
        assert change[0]*.25**2==pytest.approx(load.volume,abs=1e-13)


@pytest.mark.parametrize('factor',[1,2,4])
def test_random_mixed_columns_respect_both_limits_and_each_material(factor):
    rng=np.random.default_rng(234);profile=SoilProfile.qattara()
    # Also test decreasing density at a contact; the mass curve stays monotone.
    profile.materials[1].density=1200.
    field=SoilField(profile,16,.25*factor,dict(fine_cell=.25));ids=np.array([17,18,19])
    field.extract(ids,[1.,2.1,9.],commit=True)
    load=SoilLoad(profile,np.array([15.,25.,40.]));field.add(ids,np.full(3,load.volume/(3*field.cell**2)),load)
    before=field.volume_delta_by_material*profile.densities
    for _ in range(30):
        maximum=float(rng.uniform(.01,200));volume=float(rng.uniform(.00001,.1))
        change,load=field.extract(ids,rng.uniform(0.,2.,3),maximum,volume,commit=True)
        assert load.mass<=maximum+1e-9 and load.volume<=volume+1e-13
        after=field.volume_delta_by_material*profile.densities
        np.testing.assert_allclose(before-after,load.masses,atol=1e-8,rtol=0)
        before=after


def test_mixed_excavation_dump_reclaim_and_avalanches_conserve_materials():
    t=Terrain(layered());initial=t.mass
    for _ in range(70):
        excess=np.where(t.excavation,t.height-t.target,-np.inf);y,x=np.unravel_index(np.argmax(excess),excess.shape)
        load=t.scoop_load(t.axis[x],t.axis[y],60,max_volume=.0375)
        assert load.volume<=.0375+1e-13 and load.mass<=60+1e-9
        if load.mass>0:
            point=t.dump_location(load.volume);t.deposit(*point,load.mass,load=load)
        t.relax(4)
        np.testing.assert_allclose(t.soil.volume_delta_by_material,[0,0,0],atol=1e-12)
        assert abs(t.mass_change)<1e-8
    assert t.mass==pytest.approx(initial,abs=1e-7)
    # Actual spoil composition, rather than the native layer below the dump.
    ids=np.flatnonzero(t.dump_mask&(t.height>t.reference+1e-8))
    i=ids[np.argmax(t.height.ravel()[ids]-t.reference.ravel()[ids])]
    preview=t.preview_load(t.x.ravel()[i],t.y.ravel()[i],10,'stockpile')
    assert preview.mass>0 and preview.masses[1:].sum()>0
    rows,cols=np.indices(t.height.shape)
    np.testing.assert_allclose(t.soil.height_delta((rows*t.height.shape[0]+cols).ravel()).reshape(t.height.shape),t.height-t.reference,atol=1e-12)


def test_partial_fill_uses_cargo_volume_and_keeps_remainder():
    t=Terrain(layered());load=SoilLoad(t.soil.profile,np.array([0.,0.,22.]))
    t.fill_mask[:]=False;t.fill_mask[20,20]=True;t.target[20,20]=t.height[20,20]+.01
    capacity=.01*t.area
    before=load.masses.copy();mass=t.deposit_fill(t.axis[20],t.axis[20],load.mass,load=load)
    assert mass==pytest.approx(capacity*2200.)
    delivered=load.remove(mass)
    assert delivered.volume==pytest.approx(capacity)
    np.testing.assert_allclose(load.masses+delivered.masses,before)
    assert t.height[20,20]==pytest.approx(t.target[20,20])


def test_failed_dump_does_not_change_materials_or_height():
    t=Terrain(layered());before=t.height.copy()
    load=SoilLoad(t.soil.profile,np.array([0.,0.,(t.dump_space+1)*2200.]))
    with pytest.raises(ValueError,match='Отвал заполнен'):t.deposit(26,24,load.mass,load=load)
    np.testing.assert_array_equal(t.height,before);assert not t.soil.tiles


def test_material_cannot_be_discarded_or_relabelled_through_scalar_apis():
    t=Terrain(layered());before=t.height.copy()
    with pytest.raises(ValueError,match='scoop_load'):t.scoop(16,16,20)
    with pytest.raises(ValueError,match='состава'):t.deposit(26,24,20)
    other=SoilLoad(SoilProfile.qattara(),np.array([20.,0.,0.]))
    with pytest.raises(ValueError,match='состава'):t.deposit(26,24,20,load=other)
    np.testing.assert_array_equal(t.height,before)


@pytest.mark.parametrize('logistics',[False,True])
def test_fleet_material_energy_balances_and_exact_active_snapshot(tmp_path,logistics):
    s=Simulation(layered(logistics));s.run(175);path=tmp_path/'layered.atsstate'
    save_state(path,s);other=load_state(path).simulation
    assert other.report()==s.report()
    for _ in range(250):s.step();other.step()
    assert other.report()==s.report()
    r=s.report();assert r['state']=='RUNNING' and r['excavated_m3']>0
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<1e-4
    assert r['soil']['max_material_mass_error_kg']<1e-7
    assert r['excavated_m3']!=pytest.approx(r['excavated_mass_kg']/1600)
    for d in s.drones:
        if d.soil is not None:
            d.soil.validate();assert d.soil.volume<=d.config.equipment.bucket.volume_m3+1e-9
    json.dumps(r,ensure_ascii=False)


@pytest.mark.parametrize('mode',['legacy','logistics'])
def test_real_v1_files_keep_cargo_density_and_original_continuation(tmp_path,mode):
    # Captured with unmodified ac55b95, before the soil classes/fields existed.
    fixtures=Path(__file__).parent/'fixtures'/'soil-v1';path=tmp_path/'v1.atsstate'
    path.write_bytes(base64.b64decode((fixtures/(mode+'.b64')).read_text(encoding='ascii')))
    expected=json.loads((fixtures/(mode+'.json')).read_text(encoding='utf-8'))
    original=path.read_bytes();s=load_state(path).simulation
    assert s.config.soil_profile is None and s.terrain.soil is None
    assert json.loads(json.dumps(s.report()))==expected['initial']
    s.run(expected['steps']);assert json.loads(json.dumps(s.report()))==expected['continued']
    converted=tmp_path/'converted.atsstate';save_state(converted,s)
    assert path.read_bytes()==original
    assert load_state(converted).simulation.report()==s.report()
    with zipfile.ZipFile(converted) as archive:assert json.loads(archive.read('manifest.json'))['version']==2


def test_regional_fine_columns_survive_coarsening_and_overlapping_windows(tmp_path):
    region=Landscape(raster(),soil_profile=SoilProfile.qattara())
    # Non-aligned window shares partial persistent tiles with its neighbours.
    origin=[65.,65.];c=workspace_config(region,Config(),origin,32);t=Terrain(c)
    indices=np.array([20*128+20,20*128+21]);changes,load=t.soil.extract(indices,[2.1,9.],commit=True)
    t.height.ravel()[indices]-=changes
    t.soil.add(np.array([21*128+20]),[load.volume/t.area],load);t.height[21,20]+=load.volume/t.area
    region.commit_window(origin,t.height,soil=t.soil)
    np.testing.assert_allclose(t.soil.volume_delta_by_material,[0,0,0],atol=1e-14)
    path=tmp_path/'layers.atsregion';region.save(path);restored=Landscape.load(path)
    # Open the same changed area under 1 m cells, then return to 0.25 m.
    fine,current=restored.read_window([64.,64.],128,cell_size=1.)
    initial=restored.soil_window([64.,64.],128,1.)
    coarse=SoilField(restored.soil_profile,128,1.,initial)
    np.testing.assert_allclose(coarse.height_delta(np.arange(128**2)).reshape(128,128),current-fine,atol=1e-12)
    before=coarse.volume_delta_by_material.copy()
    change,parcel=coarse.extract([6*128+6],[.1],100,.03,True)
    current.ravel()[6*128+6]-=change[0]
    restored.commit_window([64.,64.],current,cell_size=1.,soil=coarse)
    fine_again=SoilField(restored.soil_profile,512,.25,restored.soil_window([64.,64.],512,.25))
    np.testing.assert_allclose((before-fine_again.volume_delta_by_material)*coarse.profile.densities,parcel.masses,atol=1e-8)
    baseline,height=restored.read_window([64.,64.],512)
    np.testing.assert_allclose(fine_again.height_delta(np.arange(512**2)).reshape(512,512),height-baseline,atol=1e-12)
    config=workspace_config(restored,Config(),[64.,64.],32.)
    config_path=tmp_path/'regional-config.json';save_configuration(config_path,config)
    reloaded=Terrain(load_configuration(config_path))
    np.testing.assert_allclose(reloaded.soil.height_delta(np.arange(128**2)).reshape(128,128),reloaded.height-reloaded.reference,atol=1e-12)


def test_old_region_and_config_remain_uniform_and_layered_config_roundtrips(tmp_path):
    region=Landscape(raster());path=tmp_path/'old.atsregion';region.save(path)
    with np.load(path) as data:
        metadata=json.loads(str(data['metadata']));height=data['height'].copy()
    metadata['version']=1;metadata.pop('soil_profile')
    with open(path,'wb') as file:np.savez_compressed(file,height=height,metadata=json.dumps(metadata))
    assert Landscape.load(path).soil_profile is None
    config=layered();path=tmp_path/'configuration.json';save_configuration(path,config)
    restored=load_configuration(path)
    assert asdict(restored.soil_profile)==asdict(config.soil_profile)


@pytest.mark.parametrize('damage',['density','negative','height'])
def test_corrupt_material_snapshot_is_rejected(tmp_path,damage):
    s=Simulation(layered());s.run(50)
    if damage=='density':s.config.soil_profile.materials[0].density=-1
    elif damage=='negative':next(iter(s.terrain.soil.tiles.values()))[0,0,0]=-1
    else:s.terrain.height[0,0]+=1
    path=tmp_path/'bad.atsstate';save_state(path,s)
    with pytest.raises(ValueError):load_state(path)


@pytest.mark.parametrize('robot',[False,True])
def test_harder_material_takes_longer_and_consumes_more_energy(robot):
    results=[]
    for factor in (1.,3.):
        c=Config();c.soil_profile=SoilProfile.qattara()
        c.soil_profile.materials=c.soil_profile.materials[:1];c.soil_profile.thicknesses=[]
        c.soil_profile.materials[0].digging_factor=factor
        s=Simulation(experiment_config(c) if robot else c)
        for _ in range(500):
            s.step()
            if (s.logistics.excavator.cycles if robot else s.trips):break
        assert s.logistics.excavator.cycles if robot else s.trips
        energy=s.logistics.excavator.consumed_j if robot else s.drone.consumed
        results.append((s.time,energy))
    assert results[1][0]>=results[0][0]+20
    assert results[1][1]>results[0][1]


def test_mixed_cut_and_fill_finishes_with_volume_reservations():
    from ats.plan import PlanConfig,PlanSegment
    c=layered();c.fleet.count=3;c.fleet.charge_slots=2
    c.plan=PlanConfig([PlanSegment([[12,12],[14,12]],'dig',1,-.3),PlanSegment([[16,12],[18,12]],'fill',1,.2)])
    s=Simulation(c);r=s.run(12000)
    assert r['state']=='COMPLETED',r['reason']
    assert r['filled_m3']>.3
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<1e-4
    assert r['soil']['max_material_mass_error_kg']<1e-7


def test_material_gui_profile_settings_and_snapshot_restore(tmp_path):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.soil_ui import SoilDialog
    from ats.landscape_ui import LandscapeDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    editor=SoilDialog(SoilProfile.qattara());editor.densities[1].setValue(1700)
    editor.thicknesses[0].setValue(1.25);editor.apply()
    assert editor.result_profile.materials[1].density==1700 and editor.result_profile.thicknesses[0]==1.25
    host=MainWindow(layered());host.timer.stop();host.sim.run(70);host.refresh(True)
    ordinary=host.image.image.copy();host.soil_view.setChecked(True)
    assert host.soil_view.isEnabled() and not host.soil_legend.isHidden()
    assert not np.array_equal(ordinary,host.image.image)
    assert len(host.profile_extra)>=2
    before=host.sim.report();path=tmp_path/'soil-gui.atsstate';host.save_session(path)
    host.soil_view.setChecked(False);host.sim.run(30);host.load_session(path);host.timer.stop()
    assert host.soil_view.isChecked() and host.sim.report()==before
    regional=LandscapeDialog(host)
    assert regional.region.soil_profile is not None
    regional.close();host.close();editor.close();app.processEvents()


def test_cli_material_snapshot_continuation_matches_straight_run(tmp_path):
    import subprocess,sys
    def run(*args):
        result=subprocess.run([sys.executable,'-m','ats','--headless',*map(str,args)],capture_output=True,text=True,encoding='utf-8',timeout=40)
        assert result.returncode==0,result.stderr
        return json.loads(result.stdout)
    path=tmp_path/'cli.atsstate'
    run('--soil-profile','qattara','--steps',30,'--save-state',path)
    assert run('--load-state',path,'--steps',70)==run('--soil-profile','qattara','--steps',100)


def test_strict_courier_preserves_mixed_container_cargo_through_resume_and_unload(tmp_path):
    from ats.inventory import Location as L
    c=experiment_config(layered(),drones=2,containers=2,batch_courier=False,idle_digging=False)
    s=Simulation(c);w=s.logistics
    for _ in range(1800):
        s.step()
        if any(c.soil_kg>0 and c.location==L.DRONE_CARGO for c in w.containers):break
    carried=next(c for c in w.containers if c.soil_kg>0 and c.location==L.DRONE_CARGO)
    assert carried.soil.masses[1:].sum()>0
    snapshot=tmp_path/'mixed-container.atsstate';save_state(snapshot,s);other=load_state(snapshot).simulation
    for _ in range(400):s.step();other.step()
    assert s.report()==other.report()
    assert w.delivered_mass>0 and sum(d.productivity.total_volume for d in s.drones)>0
    r=s.report();assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<1e-4
    assert r['soil']['max_material_mass_error_kg']<1e-7
    for container in w.containers:
        if container.soil is not None:container.soil.validate(container.soil_kg)
        assert container.soil_volume(1600)<=container.volume_m3+1e-9
