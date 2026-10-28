"""Layer faces, loose mixtures, bounded failures and old-state compatibility."""
from copy import deepcopy
import math
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pytest
from ats.config import Config
from ats.terrain import Terrain
from ats.soil import SoilProfile,SoilMaterial,SoilLoad
from ats.slopes import pair_transfer


def layered():
    c=Config();c.trench.depth=2.;c.terrain.floor=-5.
    c.soil_profile=SoilProfile([SoilMaterial('sand',1600,native_angle_deg=30,loose_angle_deg=30),
        SoilMaterial('rock',2200,native_angle_deg=60,loose_angle_deg=35)],[.5])
    return c


def expose_target(t):
    ids=np.flatnonzero(t.excavation);dh=(t.height-t.target).ravel()[ids]
    _,load=t.soil.extract(ids,dh,commit=True)
    t.height.ravel()[ids]-=dh;t._relax_dirty[:]=True;t._dirty_bounds=(0,t.height.shape[0],0,t.height.shape[1])
    t.revision+=1
    return load


def test_auto_minimum_uses_steep_lower_layer_and_shallow_sand_cap_without_avalanching():
    c=layered();t=Terrain(c)
    flat=deepcopy(c);flat.soil_profile.materials[1].native_angle_deg=30;flat.soil_profile.materials[1].loose_angle_deg=30
    wider=Terrain(flat)
    assert t.initial_excavation<wider.initial_excavation*.75
    assert np.max(np.abs(np.diff(t.target,axis=0)))>t.cell*math.tan(math.radians(50))
    load=expose_target(t);before=t.height.copy()
    assert t.relax(10)==0
    np.testing.assert_array_equal(t.height,before)
    assert t.max_slope_excess()<1e-7
    assert abs(t.mass_change+load.mass)<1e-7
    assert set(t.soil.surface_ids(np.flatnonzero(t.excavation)))=={0,1}


def test_project_angle_is_independent_of_material_and_unstable_profile_is_rejected():
    c=layered();c.trench.angle_deg=25;t=Terrain(c)
    assert t.slope==pytest.approx(math.tan(math.radians(30)))
    before=[(m.native_angle_deg,m.loose_angle_deg) for m in c.soil_profile.materials]
    expose_target(t);assert t.relax(5)==0
    c=layered();c.trench.angle_deg=45
    with pytest.raises(ValueError,match='раскрытых материалов'):Terrain(c)
    assert [(m.native_angle_deg,m.loose_angle_deg) for m in c.soil_profile.materials]==before


def test_intact_rock_face_is_stable_but_removed_rock_is_loose_and_mixture_uses_volume():
    c=Config();c.soil_profile=SoilProfile([SoilMaterial('rock',2200,native_angle_deg=60,loose_angle_deg=35)],[])
    t=Terrain(c);a=np.array([20*t.height.shape[0]+20]);b=a+1
    taken=t.soil.extract(b,np.array([.4]),commit=True);t.height.ravel()[b]-=.4
    assert pair_transfer(t,a,b,t.cell)[0]==0
    # On another flat pair the same rock is now transported loose material.
    u=a+10;v=u+1
    load=SoilLoad(c.soil_profile,np.array([.4*t.area*2200]))
    t.soil.add(u,np.array([.4]),load);t.height.ravel()[u]+=.4
    assert pair_transfer(t,u,v,t.cell)[0]>.1
    c=layered();q=Terrain(c);ids=np.array([1000])
    mix=SoilLoad(c.soil_profile,np.array([1600.,2200.])*.1)
    q.soil.add(ids,np.array([.2/q.area]),mix);q.height.ravel()[ids]+=.2/q.area
    from ats.slopes import _columns
    _,slope=_columns(q,ids)
    assert slope[0,0]==pytest.approx((math.tan(math.radians(30))+math.tan(math.radians(35)))/2)


def test_layered_avalanche_finishes_in_bounded_sweeps_and_conserves_each_material():
    c=layered();t=Terrain(c);load=expose_target(t)
    # Re-deposit a portion to create a fresh, initially steep spoil heap.
    moved=load.scaled(.03);t.deposit(26,25,moved.mass,radius=.25,load=moved)
    before=t.soil.volume_delta_by_material.copy()
    settled=False
    for _ in range(150):
        if t.relax(3)<=c.terrain.avalanche_stop_volume:settled=True;break
    assert settled
    np.testing.assert_allclose(t.soil.volume_delta_by_material,before,atol=1e-10,rtol=0)
    assert t.max_slope_excess()<1e-7


def test_auto_angle_and_physical_parameters_survive_configuration_plan_and_snapshot(tmp_path):
    from ats.experiments import save_configuration,load_configuration,experiment_config
    from ats.checkpoint import save_state,load_state
    from ats.simulation import Simulation
    from ats.plan import PlanConfig,PlanSegment,save_plan,load_plan
    c=layered();c.trench.depth=.25;c.plan=PlanConfig([PlanSegment([[15,16],[17,16]],'dig',1,-.25)])
    save_configuration(tmp_path/'cfg.json',c);q=load_configuration(tmp_path/'cfg.json')
    assert q.trench.angle_deg is None and q.slope_model=='materials' and q.soil_profile==c.soil_profile
    save_plan(tmp_path/'plan.json',c);q=load_plan(tmp_path/'plan.json',Config())
    assert q.trench.angle_deg is None and q.slope_model=='materials' and q.soil_profile==c.soil_profile
    s=Simulation(experiment_config(c));s.run(100);save_state(tmp_path/'s.atsstate',s)
    q=load_state(tmp_path/'s.atsstate').simulation
    assert q.config.slope_model=='materials' and q.config.soil_profile==c.soil_profile
    assert s.run(40)==q.run(40)


def test_fragmented_dump_rejects_without_per_cell_scans_and_capacity_cache_invalidates(monkeypatch):
    from types import SimpleNamespace
    from ats.maps import with_map_size
    c=with_map_size(Config(),128);c.terrain.dump_rect=(40,40,80,80)
    t=Terrain(c);t.front=SimpleNamespace(signed=t.y-64)
    t.height[t.dump_mask]=t.dump_cap[t.dump_mask]-1e-5;t.revision+=1;t._dump_content_revision+=1
    original=t.local_dump_space;calls=[]
    def check(*point):calls.append(point);return original(*point)
    monkeypatch.setattr(t,'local_dump_space',check)
    assert t.dump_space>.03 and t.dump_location(.03,near=(16,16)) is None
    assert len(calls)<10
    cap=t._local_dump_capacities().copy();y,x=160,160
    t.height[y:y+5,x:x+5]-=.1;t._dump_content_revision+=1
    assert np.max(np.abs(t._local_dump_capacities()-cap))>0


def test_cached_disk_capacities_match_scalar_capacity_at_edges_and_fresh_edits():
    from types import SimpleNamespace
    t=Terrain(Config());t.front=SimpleNamespace(signed=t.y-20)
    rng=np.random.default_rng(42);t.height[t.dump_mask]+=rng.random(np.count_nonzero(t.dump_mask))*.5
    t._dump_content_revision+=1;cap=t._local_dump_capacities()
    for iy,ix in [(0,0),(127,127),(80,100),(90,105),(110,120)]:
        assert cap[iy,ix]==pytest.approx(t.local_dump_space(t.axis[ix],t.axis[iy]),abs=1e-11)


def test_piecewise_face_solution_matches_independent_bisection_for_random_mixed_columns():
    from ats.slopes import _columns,_run
    c=layered();t=Terrain(c);rng=np.random.default_rng(17)
    a=2000+np.arange(50)*2;b=a+1
    values=np.vstack((rng.uniform(0,2,50),rng.uniform(0,.7,50),rng.uniform(0,.7,50)))
    difference=rng.uniform(.1,1.,50)
    after,_=t.soil._removed(values,difference)
    xx,yy=t.soil._coordinates(np.r_[a,b]);t.soil._write(xx,yy,np.c_[values,after])
    t.height.ravel()[a]=values[1:].sum(axis=0)-values[0]
    t.height.ravel()[b]=after[1:].sum(axis=0)-after[0]
    data=_columns(t,a);low=np.zeros(50);high=difference/2
    threshold=t.cell+c.terrain.avalanche_epsilon/math.tan(math.radians(30))
    for _ in range(60):
        mid=(low+high)/2;unstable=_run(difference-mid,data)-_run(mid,data)>threshold
        low=np.where(unstable,mid,low);high=np.where(unstable,high,mid)
    expected=np.where(_run(difference,data)>threshold,high,0)
    np.testing.assert_allclose(pair_transfer(t,a,b,t.cell),expected,atol=1e-10,rtol=0)


def test_version_five_graph_retains_legacy_angles_and_exact_continuation(tmp_path):
    import json,zipfile
    from ats.checkpoint import save_state,load_state
    from ats.simulation import Simulation
    c=Config();c.slope_model='legacy';c.trench.angle_deg=35.
    c.soil_profile=SoilProfile([SoilMaterial('sand',1600.)],[])
    s=Simulation(c);s.run(75);save_state(tmp_path/'new.atsstate',s)
    with zipfile.ZipFile(tmp_path/'new.atsstate') as original,zipfile.ZipFile(tmp_path/'v5.atsstate','w') as old:
        m=json.loads(original.read('manifest.json'));m['version']=5
        for node in m['nodes']:
            if node.get('kind')=='object':
                for key in ('slope_model','dump_slope','native_angle_deg','loose_angle_deg'):node['attrs'].pop(key,None)
        for name in original.namelist():old.writestr(name,json.dumps(m) if name=='manifest.json' else original.read(name))
    q=load_state(tmp_path/'v5.atsstate').simulation
    assert q.config.slope_model=='legacy' and q.config.trench.angle_deg==35
    assert s.report()==q.report() and s.run(100)==q.run(100)


def test_project_angle_controls_and_soil_angles_are_available_in_the_real_gui():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.soil_ui import SoilDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(Config());w.timer.stop()
    assert w.angle.value()==0 and 'минимальная ширина' in w.angle.specialValueText()
    dialog=SoilDialog(SoilProfile.qattara(),w)
    dialog.angles[2][0].setValue(55);dialog.apply()
    assert dialog.result_profile.materials[2].native_angle_deg==55
    assert dialog.result_profile.materials[2].loose_angle_deg==38
    dialog.close();w.close();app.processEvents()


def test_hardware_clone_uses_the_shared_sites_material_physics():
    from tests.test_regional import region_group
    from ats.complexes import ComplexTemplate
    p=region_group(count=1,soil=True)
    recipe=ComplexTemplate.capture(p.active.simulation.config);recipe.config.slope_model='legacy'
    u=p.add_shared(recipe)
    assert u.simulation.config.slope_model==p.shared_site.core.config.slope_model=='materials'
    assert u.simulation.config.trench.angle_deg is None


def test_channel_base_sites_keep_every_landing_footprint_inside_the_map():
    from tests.test_workfront import canal
    s=canal(slots=6);c=s.config;side=c.terrain.n*c.terrain.cell_size
    offsets=[(i+1)//2*(1 if i%2 else -1)*c.fleet.pad_spacing for i in range(1,6)]+[0.]
    clearance=c.fleet.separation+c.terrain.cell_size
    for _,x,y in c.plan.channel['operations']['base_sites']:
        assert clearance<=x-c.fleet.waiting_offset<=side-clearance
        for dy in offsets:assert clearance<=y+dy<=side-clearance


def test_auto_profile_accounts_for_preexisting_loose_soil_above_strong_native_material():
    from ats.plan import PlanConfig,PlanSegment
    c=Config();c.soil_profile=SoilProfile([SoilMaterial('rubble',2200.,native_angle_deg=60,loose_angle_deg=30)],[])
    c.reference_height=np.zeros((128,128)).tolist();c.initial_height=np.ones((128,128)).tolist()
    c.georeference=dict(vertical_datum='MSL')
    c.plan=PlanConfig([PlanSegment([[15,16],[17,16]],'dig',1,-.5)])
    t=Terrain(c);iy=64
    assert np.count_nonzero(t.target[:,iy]<t.height[:,iy]-1e-10)*t.cell>4.
    # Existing loose material requires a wider opening than intact rock alone.
    ids=np.flatnonzero(t.excavation);after=t.target
    for step in (1,128):
        a=ids[(ids+step<128**2)];b=a+step
        assert pair_transfer(t,a,b,t.cell,height=after).max(initial=0)<1e-6
