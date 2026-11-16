"""Physical tipping, bounded routes and genuinely excavated access roads."""
import numpy as np
import pytest
from ats.checkpoint import save_state,load_state
from tests.test_conventional import world,fuel_tanks,balanced


def loaded_truck(soil=False,**changes):
    s,f=world(trucks=1,soil=soil,truck_body_m3=12.,**changes)
    fuel_tanks(f);truck=f.machines[1];t=s.terrain
    if t.soil is not None:
        truck.soil=t.scoop_load(55.,50.,18000.,source='conventional',radius=5.)
        truck.cargo_kg=truck.soil.mass;f.extracted_materials+=truck.soil.masses
        volume=truck.soil.volume
    else:
        truck.cargo_kg=t.scoop(55.,50.,18000.,source='conventional',radius=5.)
        volume=truck.cargo_kg/t.config.terrain.density
    f.excavated_mass_kg=truck.cargo_kg;f.excavated_m3=volume
    truck.position[:]=(94.,100.,t.sample(94.,100.));truck.target=(100.,100.)
    truck.state='UNLOADING';truck.timer=0.
    assert f.passable(truck.position[:2],truck.radius,truck)
    return s,f,truck


@pytest.mark.parametrize('soil',[False,True])
def test_full_legal_load_tips_without_burying_its_wheels_or_return_route(soil):
    s,f,truck=loaded_truck(soil);before=s.terrain.height.copy();mass=truck.cargo_kg
    assert f._unload(truck)
    assert 0.<f.delivered_mass_kg<=mass and truck.cargo_kg==pytest.approx(mass-f.delivered_mass_kg)
    assert f.delivered_m3<=f.config.truck_body_m3
    support=s.terrain.patch(*truck.position[:2],truck.radius)
    wheel=(s.terrain.x[support]-truck.position[0])**2+(s.terrain.y[support]-truck.position[1])**2<=truck.radius**2
    np.testing.assert_array_equal(s.terrain.height[support][wheel],before[support][wheel])
    for _ in range(8):s.terrain.relax()
    assert f.passable(truck.position[:2],truck.radius,truck)
    assert f.route(truck,truck.home[:2]) is not None
    balanced(s,f)


def test_tipping_remains_local_with_large_drone_grab_radius():
    s,f,truck=loaded_truck();s.config.terrain.grab_radius=6.
    t=s.terrain;before=t.height.copy();target=np.asarray(truck.target)
    assert f._unload(truck)
    changed=np.abs(t.height-before)>1e-12
    assert np.all((t.x[changed]-target[0])**2+(t.y[changed]-target[1])**2<=2.**2+1e-10)
    for _ in range(8):t.relax()
    assert f.passable(truck.position[:2],truck.radius,truck)
    balanced(s,f)


@pytest.mark.parametrize('soil',[False,True])
def test_a_whole_gram_of_safe_remaining_cargo_is_physically_deposited(soil):
    s,f=world(trucks=1,soil=soil);t=s.terrain;truck=f.machines[1]
    if soil:
        truck.soil=t.scoop_load(55.,50.,.001,source='conventional',radius=.5)
        truck.cargo_kg=truck.soil.mass;f.extracted_materials+=truck.soil.masses;volume=truck.soil.volume
    else:
        truck.cargo_kg=t.scoop(55.,50.,.001,source='conventional',radius=.5)
        volume=truck.cargo_kg/t.config.terrain.density
    f.excavated_mass_kg=truck.cargo_kg;f.excavated_m3=volume;mass=truck.cargo_kg
    truck.position[:]=(94.,100.,0.);truck.target=(100.,100.)
    assert mass>0. and f._unload(truck)
    assert truck.cargo_kg==0. and truck.soil is None
    assert f.delivered_mass_kg==pytest.approx(mass,abs=1e-12)
    balanced(s,f)


def test_empty_unloading_state_finishes_without_inventing_a_deposit():
    s,f=world(trucks=1);truck=f.machines[1];truck.state='UNLOADING'
    before=s.terrain.height.copy()
    assert f._unload(truck) and truck.state=='PARKED' and truck.target is None
    assert f.delivered_mass_kg==0. and truck.cargo_kg==0.
    np.testing.assert_array_equal(s.terrain.height,before);balanced(s,f)


def test_partial_tipping_and_its_remaining_material_resume_exactly(tmp_path):
    s,f,truck=loaded_truck(True)
    s.terrain.dump_cap=np.minimum(s.terrain.dump_cap,s.terrain.height+.2)
    assert f._unload(truck) and truck.cargo_kg>0.
    path=tmp_path/'partial-tip.atsstate';save_state(path,s)
    other=load_state(path).simulation
    assert other.report()==s.report()
    for _ in range(35):s.step();other.step()
    assert other.report()==s.report()
    balanced(s,f);balanced(other,other.conventional)


def test_route_rejects_unsafe_edges_and_keeps_searching_for_a_detour():
    s,f=world(trucks=1,truck_dimensions_m=(.5,.5,.5),excavator_dimensions_m=(.5,.5,.5))
    truck=f.machines[1];truck.position[:]=(20.,20.,0.)
    # Neither end of the two-metre grid edge touches this narrow steep cell.
    # Its swept body does, so a safe planner must continue through another row.
    t=s.terrain;t.height[40,50]=4.;t.revision+=1
    goal=(32.,20.);path=f.route(truck,goal)
    assert not f.segment_safe(truck.position,goal,truck.radius,truck)
    assert path is not None
    assert all(f.segment_safe(a,b,truck.radius,truck) for a,b in zip([truck.position[:2],*path[:-1]],path))
    assert any(abs(p[1]-20.)>1. for p in path)


def test_route_budget_reports_uncertainty_and_preserves_cargo():
    s,f,truck=loaded_truck(max_route_nodes=100)
    t=s.terrain;t.height[:,120:122]=5.;t.revision+=1
    before=truck.cargo_kg;position=truck.position.copy()
    assert not f._assign_route(truck,(30.,100.),'PARKED')
    assert 'Лимит' in truck.reason or 'локальном' in truck.reason
    assert truck.cargo_kg==before
    np.testing.assert_array_equal(truck.position,position)


def test_existing_ramp_crest_is_not_cropped_out_of_a_deep_return_route():
    from ats.config import Config
    from ats.maps import with_map_size
    from ats.simulation import Simulation
    from ats.conventional import ConventionalConfig
    c=with_map_size(Config(),320);c.terrain.n=320;c.terrain.cell_size=1.
    c.terrain.floor=-100.;c.terrain.grab_radius=1.;c.terrain.dump_rect=(270.,270.,40.,40.)
    x,y=np.meshgrid(np.arange(320)+.5,np.arange(320)+.5)
    pit=(x>=100)&(x<200)&(y>=50)&(y<240)
    grade=-np.clip((240.-y)*.12,0.,20.)-.7*np.maximum(np.abs(x-150.)-4.5,0.)
    imported=np.where(pit,np.maximum(-20.,grade),0.)
    c.initial_height=imported;c.reference_height=imported;c.local_reference=True;c.plan=None
    s=Simulation.conventional_only(c);t=s.terrain
    # This is imported already formed ground, not a claimed runtime excavation.
    t.excavation=pit.copy();t.core_mask=pit.copy()
    t.target=np.where(pit,np.minimum(imported,-20.01),imported);t._set_dig_indices()
    f=s.install_conventional(ConventionalConfig(base_x=50.,base_y=100.,work_rect=(100.,50.,100.,190.),
        truck_count=1,max_route_nodes=5000,ramp_entry_m=(150.,240.),ramp_end_m=(150.,50.)))
    truck=f.machines[1];truck.position[:]=(150.,110.,t.sample(150.,110.))
    known=[truck.position[:2],(150.,244.),(94.,244.),(94.,truck.home[1]),truck.home[:2]]
    assert all(f.segment_safe(a,b,truck.radius,truck) for a,b in zip(known,known[1:]))
    assert f.ramp['entry_m'][1]>max(truck.position[1],truck.home[1])+32.
    path=f.route(truck,truck.home[:2])
    assert path is not None and max(p[1] for p in path)>200.
    assert all(f.segment_safe(a,b,truck.radius,truck) for a,b in zip([truck.position[:2],*path[:-1]],path))


def test_body_volume_and_ramp_clearance_are_validated():
    from ats.conventional import ConventionalConfig
    with pytest.raises(ValueError):ConventionalConfig(truck_dimensions_m=(10.,4.,4.)).validate()
    s,f=world(trucks=1);f.machines[1].cargo_kg=17000.;f.excavated_mass_kg=17000.
    with pytest.raises(ValueError,match='кузова'):f.validate()


def test_fuel_base_is_a_full_physical_square_including_its_corners():
    s,f=world(trucks=1);truck=f.machines[1];t=s.terrain
    assert not f.passable(f.base_position,truck.radius,truck)
    assert not f.segment_safe(truck.home,f.base_position,truck.radius,truck)
    assert not f.passable((18.,18.),truck.radius,truck)
    patch=t.patch(16.,16.,2.);t.dump_mask[patch]=True;t.dump_cap[patch]=t.height[patch]+10.
    area,capacity=f._dump_capacity(truck,(16.,16.),truck.position[:2],[])
    corners=(t.x[area]<=16.)&(t.y[area]<=16.)
    assert not np.any(capacity[corners])


def test_full_dump_backoff_keeps_cargo_and_waits_before_planning_again():
    s,f,truck=loaded_truck();t=s.terrain;capacity=t.dump_cap.copy()
    t.dump_cap=t.height.copy();before=truck.cargo_kg
    assert f._dump_point(truck) is None and truck.retry_route_at==30.
    calls=f.route_calculations
    for i in range(1,30):
        s.time=float(i)
        assert f._dump_point(truck) is None
    assert f.route_calculations==calls and truck.cargo_kg==before
    s.time=30.;t.dump_cap=capacity
    assert f._dump_point(truck) is not None and truck.cargo_kg==before
    assert f.route_calculations>calls


def test_failed_stance_search_has_a_shared_budget_and_persistent_cursor(monkeypatch,tmp_path):
    from ats.conventional import ConventionalFleet
    s,f=world(trucks=1);e=f.excavator;original=ConventionalFleet.route
    probes=[]
    def blocked_outbound(owner,m,destination):
        if np.linalg.norm(np.asarray(destination)-m.home[:2])<1e-9:
            return original(owner,m,destination)
        owner.route_calculations+=1;probes.append(tuple(destination));return None
    position=e.position.copy();before=f.route_calculations
    with monkeypatch.context() as patch:
        patch.setattr(ConventionalFleet,'route',blocked_outbound)
        assert not f._choose_excavator_site(e)
    assert 1<len(probes) and f.route_calculations-before<=8
    assert e.site_search_cursor>0 and 'Лимит' in e.reason
    np.testing.assert_array_equal(e.position,position)
    path=tmp_path/'search-cursor.atsstate';save_state(path,s)
    other=load_state(path).simulation
    assert other.conventional.excavator.site_search_cursor==e.site_search_cursor
    assert other.report()==s.report()


def test_large_bodies_load_at_their_real_clearance_not_a_fixed_eight_metres():
    s,f=world(trucks=1,excavator_dimensions_m=(8.,3.,3.),truck_dimensions_m=(9.,3.,3.),
              ramp_width_m=11.5,bucket_cycle_s=.5)
    fuel_tanks(f);e=f.excavator;truck=f.machines[1]
    e.position[:]=(35.,42.,0.);e.state='WAIT_TRUCK';e.target=f._available_target(e)
    assert e.target is not None
    distance=f._loading_distance(truck)
    truck.position[:]=(35.,42.-distance,0.);truck.state='WAIT_LOAD'
    assert distance>8.5 and f.passable(truck.position,truck.radius,truck)
    s.step()
    assert f.excavated_mass_kg>0. and truck.cargo_kg>0.
    balanced(s,f)


def test_real_access_excavation_descends_then_returns_and_resumes_exactly(tmp_path):
    from ats.plan import PlanConfig,PlanSegment
    from ats.simulation import Simulation
    from ats.conventional import ConventionalConfig
    from tests.test_conventional_workspace import workspace_config
    c=workspace_config()
    c.plan=PlanConfig([PlanSegment([[35.,60.],[85.,60.]],width=24.,height=-1.5)])
    # Keep the test haul local; the long-crest detour is a separate case.
    c.terrain.dump_rect=(2.,35.,19.,24.)
    s=Simulation.conventional_only(c)
    f=s.install_conventional(ConventionalConfig(base_x=12.,base_y=12.,work_rect=(30.,40.,65.,40.),
        truck_count=1,refuel_l_min=3000.,bucket_cycle_s=.5,excavator_dig_depth_m=.5))
    minimum=0.;visits=0;previous='PARKED';levels=[];last_part_cut=None
    for tick in range(3000):
        s.step();minimum=min(minimum,f.excavator.position[2]);truck=f.machines[1]
        if truck.state=='WAIT_LOAD' and previous!='WAIT_LOAD':visits+=1
        previous=truck.state
        if f.excavator.state in ('WAIT_TRUCK','DIGGING') and f.excavator.position[2]<(levels[-1] if levels else 0.)-.05:
            levels.append(float(f.excavator.position[2]))
        if tick==1999:last_part_cut=f.excavated_m3
    save_state(tmp_path/'access-progress.atsstate',s)
    assert len(levels)>=2,(levels,minimum,f.excavator.position,f.excavator.reason)
    assert f.excavated_m3>last_part_cut+.1
    assert visits>=2 and f.delivered_m3>0. and f.excavated_m3>0.
    assert f.excavator.distance_m>20.
    other=load_state(tmp_path/'access-progress.atsstate').simulation
    assert other.report()==s.report()
    f.set_enabled(False);other.conventional.set_enabled(False)
    cargo=f.carried_mass_kg
    for _ in range(300):
        s.step();other.step()
        if all(np.linalg.norm(m.position-m.home)<.1 and not m.path for m in f.machines):break
    assert all(np.linalg.norm(m.position-m.home)<.1 and not m.path for m in f.machines)
    assert f.carried_mass_kg==pytest.approx(cargo)
    assert other.report()==s.report()
    balanced(s,f);balanced(other,other.conventional)
