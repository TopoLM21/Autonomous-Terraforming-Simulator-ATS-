"""Bottom-only digging, natural faces, conservative progress and old missions."""
from copy import deepcopy
import json
import os
import zipfile
import numpy as np
import pytest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from ats.config import Config
from ats.terrain import Terrain
from ats.simulation import Simulation
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.checkpoint import save_state,load_state


def pit(layered=False):
    c=Config();c.terrain.n=40;c.terrain.cell_size=1.;c.terrain.grab_radius=1.;c.terrain.cut_policy='natural'
    c.trench.x=c.trench.y=16.5;c.trench.length=c.trench.width=1.;c.trench.depth=1.
    if layered:
        from tests.test_material_slopes import layered as profile
        c.soil_profile=profile().soil_profile;c.trench.depth=2.;c.terrain.floor=-5.
    return Terrain(c)


@pytest.mark.parametrize('layered',[False,True])
def test_one_bottom_cell_collects_slumps_without_excavating_forecast_faces(layered):
    t=pit(layered);centre=t.core_mask
    assert centre.sum()==1 and t.dig_mask.sum()==1
    assert np.any(t.excavation&~t.dig_mask)
    removed=0.;cargo=None
    if t.soil is not None:
        from ats.soil import SoilLoad
        cargo=SoilLoad.empty(t.soil.profile)
    for _ in range(300):
        if cargo is None:removed+=t.scoop(16.5,16.5,1e9,radius=.49)
        else:
            load=t.scoop_load(16.5,16.5,1e9,radius=.49);cargo.add(load);removed+=load.mass
        moved=t.relax(12)
        if t.remaining<1e-7 and moved<1e-8:break
    assert t.remaining<1e-7 and t.max_slope_excess()<1e-7
    assert float(np.maximum(t.height-t.target,0)[t.excavation&~t.dig_mask].sum()*t.area)>.01
    assert abs(t.mass_change+removed)<1e-7
    if cargo is not None:
        volumes=cargo.masses/np.array([m.density for m in cargo.profile.materials])
        np.testing.assert_allclose(t.soil.volume_delta_by_material+volumes,0.,atol=1e-9,rtol=0.)
        assert np.count_nonzero(cargo.masses)==2
    else:assert removed/t.config.terrain.density>1.5


def test_explicit_project_angle_retains_strict_profile_even_with_natural_preference():
    c=pit().config;c.trench.angle_deg=25.
    t=Terrain(c)
    assert not t.natural_excavation and np.array_equal(t.dig_mask,t.excavation)
    assert t.remaining>float(t.core_mask.sum()*t.area)


def test_excavation_footprint_does_not_allow_a_grab_on_only_the_predicted_face():
    t=pit();k=np.flatnonzero(t.excavation&~t.dig_mask)[0]
    x,y=t.x.ravel()[k],t.y.ravel()[k]
    before=t.height.copy();assert t.scoop(x,y,1e9,radius=.49)==0
    np.testing.assert_array_equal(t.height,before)


def test_autonomous_natural_mission_finishes_after_slumps_and_keeps_mass_energy():
    c=experiment_config(with_map_size(Config(),64),drones=2,panel_area=20.)
    c.terrain.cut_policy='natural';c.fleet.airspace_mode='shared_lanes';c.logistics.excavator_enabled=False
    s=Simulation(c);r=s.run(12000)
    assert r['state']=='COMPLETED',r.get('reason')
    assert not s._terrain_active and s.terrain.remaining<=c.terrain.completion_volume
    assert all(d.grounded and d.cargo==0 for d in s.drones)
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002


def natural_group():
    from tests.test_regional import region_group
    from ats.complexes import ComplexProject,ComplexTemplate
    original=region_group(count=1,soil=True)
    c=deepcopy(original.active.simulation.config);c.terrain.cut_policy='natural';c.fleet.airspace_mode='shared_lanes'
    p=ComplexProject(Simulation(c),original.landscape);p.enable_shared();p.add_shared(ComplexTemplate.capture(c))
    return p


def test_shared_bottom_cells_have_one_owner_and_forecast_is_still_reserved():
    p=natural_group();t=p.shared_site.core
    masks=[u.simulation.terrain.dig_mask for u in p.units]
    np.testing.assert_array_equal(np.sum(masks,axis=0),t.dig_mask.astype(int))
    assert all(u.simulation.terrain.natural_excavation for u in p.units)
    assert np.any(t.design_work&~t.dig_mask)


def test_natural_group_window_roundtrip_and_save_in_transit_preserve_goal(tmp_path):
    from tests.test_regional import advance
    p=natural_group();ids=p.inventory_ids();ground=p.shared_site.core.height.copy()
    p.move_window([96.,64.]);advance(p,lambda:p.transition.phase=='DRIVE')
    path=tmp_path/'natural-driving.atsstate';save_state(path,p.active.simulation,p.landscape,complex_project=p)
    q=load_state(path).complex_project
    advance(p,lambda:p.transition is None);advance(q,lambda:q.transition is None)
    assert p.report()==q.report() and p.inventory_ids()==ids
    assert p.shared_site.core.natural_excavation and q.shared_site.core.natural_excavation
    p.move_window([64.,64.]);advance(p,lambda:p.transition is None)
    np.testing.assert_array_equal(p.shared_site.core.height,ground)
    assert p.inventory_ids()==ids
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002


def test_v16_profile_mission_reads_without_silent_goal_or_airspace_change(tmp_path):
    from tests.test_regional import region_group
    p=region_group(count=2,soil=True);before=p.report();path=tmp_path/'previous.atsstate'
    save_state(path,p.active.simulation,p.landscape,complex_project=p)
    with zipfile.ZipFile(path) as archive:files={n:archive.read(n) for n in archive.namelist()}
    data=json.loads(files['manifest.json']);data['version']=16
    for node in data['nodes']:
        attrs=node.get('attrs',{})
        if node.get('type') in ('ats.terrain.Terrain','ats.shared_site.WorkTerrain'):
            for key in ('core_mask','_dig_mask','_clearance_revision'):attrs.pop(key,None)
        if node.get('type')=='ats.config.TerrainConfig':attrs.pop('cut_policy',None)
        if node.get('type')=='ats.config.FleetConfig':attrs.pop('airspace_mode',None)
    files['manifest.json']=json.dumps(data).encode()
    with zipfile.ZipFile(path,'w') as archive:
        for name,payload in files.items():archive.writestr(name,payload)
    original=path.read_bytes();q=load_state(path).complex_project
    assert path.read_bytes()==original and q.report()==before
    for u in q.units:
        s=u.simulation
        assert not s.terrain.natural_excavation and s.config.fleet.airspace_mode=='per_drone'
        np.testing.assert_array_equal(s.terrain.dig_mask,s.terrain.excavation)


def test_channel_progress_keeps_its_completion_policy_between_windows():
    from ats.channel import ChannelProject
    from ats.landscape import Landscape
    from tests.test_channel import coastal_dem
    land=Landscape(coastal_dem(gap=100));c=experiment_config(Config());c.terrain.cut_policy='natural'
    project=ChannelProject.create(land.raster,[[50,150],[250,150]],width_m=2,depth_m=1,side_m=128)
    identity=project.identity;project.configuration(land,c)
    assert project.progress_policy=='natural' and project.identity==identity
    project.finished_spans=[[0.,10.]]
    loaded=ChannelProject.from_data(project.to_data());c.terrain.cut_policy='profile'
    next_window=loaded.configuration(land,c)
    assert next_window.terrain.cut_policy=='natural' and loaded.finished_spans==[[0.,10.]]


def test_natural_robot_and_transport_fleet_finish_without_trimming_the_forecast():
    c=experiment_config(with_map_size(Config(),64),drones=4,panel_area=20.)
    c.terrain.cut_policy='natural';c.fleet.airspace_mode='shared_lanes'
    s=Simulation(c);r=s.run(20000)
    assert r['state']=='COMPLETED',r.get('reason')
    assert r['excavator']['excavated_kg']>0
    assert not s._terrain_active and all(d.grounded and d.cargo==0 for d in s.drones)
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002


def test_new_gui_defaults_are_natural_and_shared_but_configuration_files_keep_their_choices(monkeypatch,tmp_path):
    import sys
    from ats import __main__ as cli,gui
    from ats.experiments import save_configuration
    captured={}
    monkeypatch.setattr(gui,'launch',lambda config,**kwargs:captured.update(config=config,kwargs=kwargs) or 0)
    monkeypatch.setattr(sys,'argv',['ats']);assert cli.main()==0
    c=captured['config'];assert c.terrain.cut_policy=='natural' and c.fleet.airspace_mode=='shared_lanes'
    assert 'session' not in captured['kwargs']  # the new GUI starts with an empty site
    from ats.terrain import Terrain
    assert Terrain(c).natural_excavation
    path=tmp_path/'profile.json';save_configuration(path,experiment_config(with_map_size(Config(),64)))
    monkeypatch.setattr(sys,'argv',['ats','--configuration',str(path)]);assert cli.main()==0
    assert captured['config'].terrain.cut_policy=='profile' and captured['config'].fleet.airspace_mode=='per_drone'


def test_sector_progress_checks_bottom_instead_of_the_unused_forecast():
    from tests.test_sectors import sector_config
    from ats.sector_progress import ready_bins
    c=sector_config(length=170.);c.terrain.cut_policy='natural';c.plan.segments[0].height=-1.
    s=Simulation(c);t=s.terrain
    # Controlled conservative removal into an external load. Keep the forecast
    # unchanged to ensure its unexcavated faces are not counted as bottom work.
    removed=0.
    for k in t._cut_indices:
        removed+=t.scoop(t.x.ravel()[k],t.y.ravel()[k],1e8,radius=.49)
    assert t.remaining==0 and abs(t.mass_change+removed)<1e-7
    assert np.any(t.height[t.excavation&~t.dig_mask]>t.target[t.excavation&~t.dig_mask])
    g=ready_bins(t,0.,170.)
    assert np.any(g['loaded']) and np.all(g['ready'][g['loaded']])


def test_shared_plan_with_physical_platforms_can_be_cleared_without_losing_inventory():
    from tests.test_self_solar import world
    from ats.complexes import ComplexProject
    from ats.plan import PlanConfig
    s=world(164);p=ComplexProject(s);p.enable_shared();ids=p.inventory_ids();height=s.terrain.height.copy()
    p.shared_site.apply_plan(PlanConfig([]))
    assert p.inventory_ids()==ids and p.shared_site.core.remaining==0
    np.testing.assert_array_equal(s.terrain.height,height)
