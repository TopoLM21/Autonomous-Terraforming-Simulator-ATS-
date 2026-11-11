"""Every robot is a physical actor during regional relocation, including resume."""
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.landscape import Landscape,ElevationRaster,workspace_config
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.complexes import ComplexProject
from ats.checkpoint import save_state,load_state
from ats.soil import SoilProfile


def group(robots):
    land=Landscape(ElevationRaster(np.zeros((6,6)),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',
        x0=0.,y0=0.,dx=100.,dy=100.,latitude_deg=30.)),cell_size=.5,soil_profile=SoilProfile.qattara())
    c=experiment_config(Config(),drones=4,containers=robots);c.logistics.excavator_count=robots
    c.logistics.field_deployment_seconds=4.
    c=workspace_config(land,c,[64.,64.],128)
    c.plan=PlanConfig([PlanSegment([[24,60],[104,60]],width=8,height=-.3)])
    p=ComplexProject(Simulation(c),land);p.enable_shared();return p


def advance(p,predicate,limit=14000):
    for _ in range(limit):
        if predicate():return
        p.step();assert not p.blocked,p.active.simulation.reason
    pytest.fail(str(p.transition.report() if p.transition else p.report()))


@pytest.mark.parametrize('robots',[3,10])
def test_every_robot_roundtrips_with_materials_and_resumes_in_drive(tmp_path,robots):
    p=group(robots);s=p.active.simulation;t=p.shared_site.core
    load=t.scoop_load(40,60,35.,radius=1.);t.deposit(26,26,load.mass,load=load)
    height=t.height.copy();materials={k:v.copy() for k,v in t.soil.tiles.items()}
    ids=p.inventory_ids();serials=[e.id for e in s.logistics.excavators]
    objects=list(s.logistics.excavators)
    p.move_window([96.,64.]);advance(p,lambda:p.transition.phase=='DRIVE')
    assert len(p.transition.routes)==robots+1
    assert {kind for owner,kind in p.transition.routes}=={'base',*serials}
    # Save after actual progress by a robot other than the public primary alias.
    advance(p,lambda:p.transition.actor_index>=1)
    path=tmp_path/'all-robots-moving.atsstate';save_state(path,s,p.landscape,complex_project=p)
    q=load_state(path).complex_project
    advance(p,lambda:p.transition is None);advance(q,lambda:q.transition is None)
    assert p.report()==q.report() and s.report()==q.active.simulation.report()
    assert all(e is obj for e,obj in zip(s.logistics.excavators,objects))
    q.move_window([64.,64.]);advance(q,lambda:q.transition is None)
    assert q.inventory_ids()==ids and [e.id for e in q.active.simulation.logistics.excavators]==serials
    np.testing.assert_array_equal(q.shared_site.core.height,height)
    for key,array in materials.items():np.testing.assert_array_equal(q.shared_site.core.soil.tiles[key],array)
    r=q.report();assert q.time>p.time and len(q.window_history)==2
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002
    assert abs(r['equipment_mass_error_kg'])<1e-7
    assert r['shared_worksite']['max_material_mass_error_kg']<1e-7
    q.validate()


def test_two_shared_complexes_own_separate_robots_and_one_ground_ledger():
    from ats.complexes import ComplexTemplate
    p=group(3);first=p.active.simulation
    p.add_shared(ComplexTemplate.from_simulation(first,'Тройка'))
    assert len({id(e) for u in p.units for e in u.simulation.logistics.excavators})==6
    # Existing robots retain their positions; one may need a real transfer to
    # the revised assignment before the rest of its team resumes excavation.
    advance(p,lambda:all(any(e.excavated_kg>0 for e in u.simulation.logistics.excavators) for u in p.units),limit=1600)
    r=p.report();assert r['state']=='RUNNING',r['reason']
    assert r['shared_worksite']['shared_height_arrays']==1
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002
    assert abs(r['equipment_mass_error_kg'])<1e-7
    assert all(any(e.excavated_kg>0 for e in u.simulation.logistics.excavators) for u in p.units)
    p.validate()


def test_v20_flat_primary_pose_cache_is_rebuilt_without_losing_site_or_state(tmp_path):
    import json,zipfile
    from ats.season_power import ready_season_simulation
    from ats.config import Config
    from ats.experiments import experiment_config
    from ats.complexes import ComplexProject
    c=experiment_config(Config());c.base.latitude_deg=31.078
    s,_=ready_season_simulation(c);p=ComplexProject(s);p.enable_shared()
    bp=s.logistics.battery_platforms;bp.audit_site()
    signature=bp._audit_signature
    old_actors=tuple((u.id,tuple(u.simulation.base.position),tuple(u.simulation.logistics.excavator.position)) for u in p.units)
    bp._audit_signature=(tuple(s.base.position),s.logistics.field_rect,old_actors,signature[-1])
    path=tmp_path/'v20-cache.atsstate';save_state(path,s,complex_project=p)
    with zipfile.ZipFile(path) as archive:files={name:archive.read(name) for name in archive.namelist()}
    manifest=json.loads(files['manifest.json']);manifest['version']=20;files['manifest.json']=json.dumps(manifest).encode()
    with zipfile.ZipFile(path,'w') as archive:
        for name,data in files.items():archive.writestr(name,data)
    q=load_state(path).complex_project
    assert q.report()==p.report()
    assert q.active.simulation.logistics.battery_platforms._audit_signature[0]=='robot_fleet_v1'
    for _ in range(25):p.step();q.step()
    assert q.report()==p.report()
    q.validate()


def test_previous_failed_route_cache_can_retry_after_resume(tmp_path):
    p=group(1);p.move_window([96.,64.])
    advance(p,lambda:p.transition.phase=='ENERGY')
    tr=p.transition
    # A previous-format wait after failed preflight: no committed ground drive
    # yet. Keep the old flat primary-pose key as it was written in v20.
    tr.last_failed_route=(p.shared_site.core.revision,tuple((u.id,tuple(u.simulation.base.position),
        tuple(u.simulation.logistics.excavator.position),u.simulation.base.config.mass,
        repr(u.simulation.base.config.layout)) for u in p.units))
    tr.routes={};tr.actors=[];tr.retry_at=p.time
    path=tmp_path/'old-route-cache.atsstate';save_state(path,p.active.simulation,p.landscape,complex_project=p)
    q=load_state(path).complex_project;ids=q.inventory_ids();before=q.time
    q.step()  # must invalidate the obsolete key, rather than compare arrays or freeze
    advance(q,lambda:q.transition is None)
    assert q.time>before and q.inventory_ids()==ids
    assert abs(q.report()['mass_error_kg'])<1e-7 and abs(q.report()['energy_error_j'])<.002
