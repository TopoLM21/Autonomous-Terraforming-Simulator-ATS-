import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.complexes import ComplexProject,ComplexTemplate
from ats.checkpoint import save_state,load_state


def site(side=128,soil=False):
    c=experiment_config(with_map_size(Config(),side))
    if soil:
        from ats.soil import SoilProfile
        c.soil_profile=SoilProfile.qattara();c.soil_profile.thicknesses=[.02,.05]
    c.plan=PlanConfig([PlanSegment([[35,40],[95,40]],'dig',2,-.25)])
    s=Simulation(c);assert s.state.value=='RUNNING',s.reason
    return ComplexProject(s),ComplexTemplate.capture(c)


def test_twenty_complexes_own_one_ground_and_unique_physical_equipment():
    p,t=site()
    for _ in range(19):p.add_shared(t)
    assert len(p.units)==20
    assert len({id(u.simulation.terrain.height) for u in p.units})==1
    assert len({id(u.simulation.logistics.inventory) for u in p.units})==20
    masks=[u.simulation.terrain.excavation.astype(int) for u in p.units]
    np.testing.assert_array_equal(np.sum(masks,axis=0),p.shared_site.core.excavation.astype(int))
    assert p.report()['equipment_mass_error_kg']==pytest.approx(0,abs=1e-7)
    before=p.report()
    with pytest.raises(ValueError,match='20'):p.add_shared(t)
    assert p.report()==before;p.validate()


def test_concurrent_excavation_conserves_each_material_and_exact_whole_save(tmp_path):
    p,t=site(soil=True)
    for _ in range(2):p.add_shared(t)
    for _ in range(130):p.step()
    r=p.report();assert sum(u.simulation.excavated_mass>0 for u in p.units)>=2
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002
    assert r['shared_worksite']['max_material_mass_error_kg']<1e-7
    p.select('complex_2');path=tmp_path/'together.atsstate'
    save_state(path,p.active.simulation,complex_project=p)
    loaded=load_state(path);q=loaded.complex_project
    assert q.active.simulation is loaded.simulation
    assert len({id(u.simulation.terrain.height) for u in q.units})==1
    for _ in range(30):p.step();q.step()
    assert p.report()==q.report()
    assert all(a.simulation.report()==b.simulation.report() for a,b in zip(p.units,q.units))


def test_collision_checks_swept_paths_between_different_complexes():
    p,t=site();p.add_shared(t);a,b=[u.simulation for u in p.units]
    d,e=a.drone,b.drone;d.position=np.array([50.,60.,20.]);e.position=np.array([51.,60.,20.])
    p.shared_site.start_tick()
    assert p.shared_site.collision(a,d,d.position,np.array([52.,60.,20.])) is e
    e.position=np.array([51.,65.,20.]);p.shared_site.start_tick()
    assert p.shared_site.collision(a,d,d.position,np.array([52.,60.,20.])) is None


def test_finished_team_receives_more_work_without_new_hardware_or_energy():
    from ats.simulation import MissionState
    p,t=site();p.add_shared(t);s=p.units[0].simulation;ground=p.shared_site.core
    removed=s.terrain.scoop(64,40,1e9,radius=100)
    point=ground.dump_location(removed/s.config.terrain.density)
    assert removed>0 and point is not None
    ground.deposit(*point,removed,radius=100)
    assert s.terrain.remaining<1e-10
    s.state=MissionState.COMPLETED
    before=p.report();ids=set(s.logistics.inventory.items)
    p.shared_site.advance_tasks()
    assert s.state==MissionState.RUNNING and s.terrain.remaining>0
    assert set(s.logistics.inventory.items)==ids
    assert p.report()['delivered_energy_kwh']==before['delivered_energy_kwh']
    assert p.report()['energy_error_j']==before['energy_error_j']
    assert abs(p.report()['mass_error_kg'])<1e-7
    p.validate()


@pytest.mark.parametrize('soil',[False,True])
def test_live_map_extension_retains_all_cargo_energy_and_ground(tmp_path,soil):
    p,t=site(soil=soil);p.add_shared(t)
    for _ in range(80):p.step()
    before=p.report();height=p.shared_site.core.height.copy()
    p.shared_site.enlarge(192)
    np.testing.assert_array_equal(p.shared_site.core.height[:512,:512],height)
    assert p.report()['mass_error_kg']==pytest.approx(before['mass_error_kg'],abs=1e-7)
    assert p.report()['energy_error_j']==before['energy_error_j']
    assert p.time==before['time_seconds']
    path=tmp_path/'extended.atsstate';save_state(path,p.active.simulation,complex_project=p)
    q=load_state(path).complex_project
    for _ in range(10):p.step();q.step()
    assert p.report()==q.report()


def test_common_plan_is_atomic_and_cannot_pass_through_another_base():
    p,t=site();p.add_shared(t);s=p.active.simulation
    second=p.units[1].simulation.base.position
    before=p.shared_site.core.target.copy()
    plan=PlanConfig([PlanSegment([[second[0]-2,second[1]],[second[0]+2,second[1]]],'dig',1,-.25)])
    with pytest.raises(ValueError,match='посадочн'):s.apply_plan(plan)
    np.testing.assert_array_equal(p.shared_site.core.target,before)


def test_live_coarsening_preserves_fine_materials_and_all_stored_energy():
    p,t=site(soil=True);p.add_shared(t)
    for _ in range(100):p.step()
    for u in p.units:u.simulation.request_landing()
    for _ in range(3000):
        if all(d.grounded for u in p.units for d in u.simulation.drones):break
        p.step()
    assert all(d.grounded for u in p.units for d in u.simulation.drones)
    old=p.shared_site.core;before=p.report();height=old.height.copy()
    positions=[u.simulation.base.position.copy() for u in p.units]
    p.shared_site.enlarge(320)
    ground=p.shared_site.core;assert ground.cell==old.cell*2
    np.testing.assert_allclose(ground.height[:256,:256],height.reshape(256,2,256,2).mean(axis=(1,3)),atol=1e-12)
    after=p.report()
    assert after['time_seconds']==before['time_seconds']
    assert after['energy_error_j']==before['energy_error_j']
    assert abs(after['mass_error_kg'])<1e-7
    assert after['shared_worksite']['max_material_mass_error_kg']<1e-7
    for unit,point in zip(p.units,positions):np.testing.assert_array_equal(unit.simulation.base.position,point)
    p.validate()


def test_template_copies_hot_delivered_stock_as_real_initial_equipment(tmp_path):
    p,t=site();source=p.active.simulation;w=source.logistics
    w.replenish_batteries(source.drone.battery_ids[0],17,.2)
    template=ComplexTemplate.from_simulation(source)
    assert sum(r['count'] for r in template.stock)==17
    path=tmp_path/'stock.atstemplate';template.save(path);template=ComplexTemplate.load(path)
    u=p.add_shared(template);other=u.simulation
    from ats.inventory import BatteryModule
    a=[o for o in source.logistics.inventory.select(BatteryModule) if o.compatibility!='base']
    b=[o for o in other.logistics.inventory.select(BatteryModule) if o.compatibility!='base']
    assert len(a)==len(b) and all(o.soc==1 for o in b)
    assert other.logistics.imported_mass_kg==0 and other.equipment_imported_energy==0
    assert abs(p.report()['equipment_mass_error_kg'])<1e-7
    assert other.report()['initial_energy_kwh']>source.report()['initial_energy_kwh']


def test_authentic_version_three_regional_group_continues_without_conversion(tmp_path):
    import base64,json
    from pathlib import Path
    folder=Path(__file__).parent/'fixtures'/'complexes-v3'
    path=tmp_path/'old.atsstate';path.write_bytes(base64.b64decode((folder/'two.b64').read_text(encoding='ascii')))
    expected=json.loads((folder/'two.json').read_text(encoding='utf-8'))
    session=load_state(path);p=session.complex_project
    assert p.shared_site is None and len(p.units)==2
    def compare(want):
        assert json.loads(json.dumps(p.report()))==want['project']
        for u,target in zip(p.units,want['reports']):
            actual=json.loads(json.dumps(u.simulation.report()))
            # Platform libm can round this instantaneous power product by one
            # ulp; the stored energy balances below still match exactly.
            assert actual['solar_power_w']==pytest.approx(target['solar_power_w'],rel=0,abs=1e-10)
            actual['solar_power_w']=target['solar_power_w']
            for kind in ('drones','retired_drones'):
                for a,b in zip(actual.get(kind,[]),target.get(kind,[])):
                    for key in ('last24_mass_kg','last24_volume_m3'):
                        assert a[key]==pytest.approx(b[key],rel=0,abs=1e-11);a[key]=b[key]
            # Python 3.12 improved sum() accuracy. The fixture was generated
            # there; rolling statistics on 3.11 can differ by a few ulps.
            if target.get('excavator') is not None:
                for key in ('last24_excavated_kg','last24_excavated_m3'):
                    a,b=actual['excavator'],target['excavator']
                    assert a[key]==pytest.approx(b[key],rel=0,abs=1e-11);a[key]=b[key]
            assert actual==target
    compare(expected['initial'])
    for _ in range(expected['steps']):p.step()
    compare(expected['continued'])
    new=tmp_path/'new.atsstate';save_state(new,p.active.simulation,session.landscape,complex_project=p)
    assert load_state(new).complex_project.report()==p.report()
