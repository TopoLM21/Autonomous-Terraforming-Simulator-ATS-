"""Traditional and autonomous crews share one physical height/material ledger."""
import numpy as np
import pytest
from ats.conventional import ConventionalConfig
from ats.conventional_integration import fleet_for
from ats.checkpoint import save_state,load_state
from tests.test_shared_site import site


def mixed(soil=False):
    project,template=site(soil=soil)
    project.add_shared(template,position=(100.,20.))
    config=ConventionalConfig(base_x=12.,base_y=100.,work_rect=(35.,30.,20.,20.),truck_count=2,refuel_l_min=3000.)
    fleet=project.units[0].simulation.install_conventional(config)
    return project,template,fleet


def balanced(project):
    project.validate();report=project.report()
    assert abs(report['mass_error_kg'])<1e-7
    assert abs(report['energy_error_j'])<.002
    error=project.shared_site.material_errors()
    if error is not None:np.testing.assert_allclose(error,0.,atol=1e-7)
    for fleet in project.shared_site.conventional_fleets():fleet.validate()


@pytest.mark.parametrize('soil',[False,True])
def test_two_autonomous_crews_and_diesel_machines_really_work_on_one_heightfield(soil):
    p,template,f=mixed(soil);site=p.shared_site;t=site.core
    assert sum(len(u.simulation.drones) for u in p.units)==8
    assert len({id(u.simulation.terrain.height) for u in p.units}|{id(f.terrain.height)})==1
    assert set(np.unique(site.owners))=={0,1,2}
    assert np.all(site.owners[t._conventional_mask]==0)
    for _ in range(1500):
        p.step()
        for unit in p.units:
            s=unit.simulation;ids=s.terrain.source_indices()
            assert not np.any(t._conventional_mask.ravel()[ids])
            if int(p.time)%100==0:
                for drone in s.drones:
                    if drone.trip is not None and drone.trip.source=='cut':
                        x,y=drone.trip.dig;assert not t._conventional_mask[int(y/t.cell),int(x/t.cell)]
        if f.delivered_m3>.05 and all(u.simulation.excavated_mass>0 for u in p.units):break
    assert f.excavated_m3>0 and f.delivered_m3>.05
    assert all(u.simulation.excavated_mass>0 for u in p.units)
    assert f.sim.time==p.time and f._last_time==p.time-1
    expected=(f.config.excavator_wage_egp_month+2*f.config.driver_wage_egp_month)*p.time/(30*86400*f.config.egp_per_usd)
    assert f.payroll_usd==pytest.approx(expected)
    balanced(p)


def test_selected_autonomous_unit_finds_the_same_fleet_without_duplication():
    p,template,f=mixed()
    a,b=[u.simulation for u in p.units]
    assert a.conventional is f and b.conventional is None
    p.select('complex_2')
    assert fleet_for(p.active.simulation) is f
    assert p.shared_site.conventional_fleets()==[f]
    with pytest.raises(ValueError,match='приобретена'):b.install_conventional(f.config)
    fleet_for(b).replenish(fuel_l=150.,water_l=10.,food_units=2.)
    assert f.fuel_imported_l==150
    before=f.payroll_usd
    for _ in range(10):p.step()
    assert f.payroll_usd>before
    assert b.conventional is None and len(p.shared_site.conventional_fleets())==1
    balanced(p)


def test_new_complex_does_not_copy_conventional_assets_or_reallocate_the_reserved_patch():
    p,template,f=mixed();reserved=p.shared_site.core._conventional_mask.copy();ids=[m.id for m in f.machines]
    p.add_shared(template,position=(100.,100.))
    assert sum(getattr(u.simulation,'conventional',None) is not None for u in p.units)==1
    np.testing.assert_array_equal(p.shared_site.core._conventional_mask,reserved)
    assert [m.id for m in f.machines]==ids
    assert np.all(p.shared_site.owners[reserved]==0)
    for u in p.units:assert not np.any(u.simulation.terrain.excavation&reserved)
    balanced(p)


def test_shared_install_refuses_an_entirely_reserved_project_before_purchase():
    p,template=site();p.add_shared(template,position=(100.,20.))
    owners=p.shared_site.owners.copy();before=p.report()
    config=ConventionalConfig(base_x=12.,base_y=100.,work_rect=(30.,35.,68.,10.))
    with pytest.raises(ValueError,match='не остаётся работы'):p.active.simulation.install_conventional(config)
    assert all(u.simulation.conventional is None for u in p.units)
    np.testing.assert_array_equal(p.shared_site.owners,owners)
    assert p.report()==before


def test_live_reserved_drone_work_cannot_be_taken_by_the_ground_fleet():
    p,template=site();p.add_shared(template,position=(100.,20.))
    for _ in range(20):
        p.step()
        drone=next((d for u in p.units for d in u.simulation.drones if d.trip is not None),None)
        if drone is not None:break
    assert drone is not None
    x,y=drone.trip.dig
    owners=p.shared_site.owners.copy();height=p.shared_site.core.height.copy()
    config=ConventionalConfig(base_x=12.,base_y=100.,work_rect=(x-4,y-4,8.,8.))
    with pytest.raises(ValueError,match='рейсом|зарезервирован|цикл'):p.active.simulation.install_conventional(config)
    assert all(u.simulation.conventional is None for u in p.units)
    np.testing.assert_array_equal(p.shared_site.owners,owners);np.testing.assert_array_equal(p.shared_site.core.height,height)
    balanced(p)


def test_unrelated_flights_continue_when_a_free_patch_is_reserved():
    p,template=site();p.add_shared(template,position=(100.,20.))
    # Start the two autonomous crews exclusively in the right-hand future free
    # area, then reserve a different portion without resetting those flights.
    mask=p.shared_site.core.x<60
    for u in p.units:
        u.simulation.terrain.assign(u.simulation.terrain.excavation&~mask)
    for _ in range(10):
        p.step()
        trips=[d.trip for u in p.units for d in u.simulation.drones if d.trip is not None]
        if trips:break
    assert trips
    f=p.units[0].simulation.install_conventional(ConventionalConfig(base_x=12,base_y=100,work_rect=(35,30,15,20),refuel_l_min=3000))
    assert all(d.trip in trips for u in p.units for d in u.simulation.drones if d.trip is not None)
    for _ in range(10):p.step()
    assert f.sim.time==p.time
    balanced(p)


def test_ground_obstacle_tokens_include_moving_trucks_but_router_ignores_its_own_fleet():
    p,template,f=mixed();s=p.shared_site;truck=f.machines[1]
    bounds=(truck.position[0]-10,truck.position[1]-10,truck.position[0]+15,truck.position[1]+10)
    before=s.ground_token('complex_2',bounds)
    assert any(row[0]=='conventional' for row in before)
    assert not any(row[0]=='conventional' for row in s.ground_token('conventional',bounds))
    assert f.passable(truck.position,truck.radius,truck)
    assert s.obstacle_height(truck.position[:2])>=truck.position[2]+truck.dimensions_m[2]
    stock=20.;f.fuel_stock_l-=stock;truck.fuel_l+=stock
    destination=truck.position[:2]+[8.,0.]
    assert f._assign_route(truck,destination,'PARKED')
    owner_geometry=s._geometry_units['complex_1']
    f._move(truck)
    after=s.ground_token('complex_2',bounds)
    assert after!=before
    assert s._geometry_units['complex_1'] is owner_geometry
    balanced(p)


def test_common_checkpoint_keeps_the_one_loaded_fleet_when_selected_unit_has_none(tmp_path):
    p,template,f=mixed(soil=True)
    for _ in range(800):
        p.step()
        if f.carried_mass_kg>50:break
    assert f.carried_mass_kg>50
    p.select('complex_2');path=tmp_path/'all-crews.atsstate';save_state(path,p.active.simulation,complex_project=p)
    q=load_state(path).complex_project;g=fleet_for(q.active.simulation)
    assert q.active.simulation.conventional is None and g is not None
    assert g.terrain is q.shared_site.core
    assert q.report()==p.report()
    for _ in range(25):p.step();q.step()
    assert q.report()==p.report()
    balanced(p);balanced(q)


def test_same_resolution_growth_preserves_the_ground_fleet_and_its_original_ramp():
    p,template,f=mixed(soil=True)
    for _ in range(400):
        p.step()
        if f.carried_mass_kg>10:break
    before=p.report();height=p.shared_site.core.height.copy();ramp=f.ramp_surface.copy()
    points=[m.position.copy() for m in f.machines];ids=[m.id for m in f.machines]
    p.shared_site.enlarge(192)
    assert f.terrain is p.shared_site.core and f.sim.terrain.core is f.terrain
    np.testing.assert_array_equal(f.terrain.height[:height.shape[0],:height.shape[1]],height)
    np.testing.assert_array_equal(f.ramp_surface[:ramp.shape[0],:ramp.shape[1]],ramp)
    for old,m in zip(points,f.machines):np.testing.assert_array_equal(old,m.position)
    assert [m.id for m in f.machines]==ids and p.time==before['time_seconds']
    for _ in range(5):p.step()
    balanced(p)


def test_resolution_change_or_plan_replacement_with_road_machines_is_rejected_atomically():
    p,template,f=mixed();core=p.shared_site.core;before=p.report();target=core.target.copy()
    with pytest.raises(ValueError,match='разрешения'):p.shared_site.enlarge(320)
    with pytest.raises(ValueError,match='обычная техника'):p.shared_site.apply_plan(core.config.plan)
    assert p.shared_site.core is core and f.terrain is core and p.report()==before
    np.testing.assert_array_equal(core.target,target)
    balanced(p)


def test_completed_ground_stage_hands_ramp_and_buffer_back_to_aircraft_naturally():
    from ats.config import Config
    from ats.maps import with_map_size
    from ats.experiments import experiment_config
    from ats.plan import PlanConfig,PlanSegment
    from ats.simulation import Simulation
    from ats.complexes import ComplexProject,ComplexTemplate
    c=experiment_config(with_map_size(Config(),128))
    c.plan=PlanConfig([PlanSegment([[42.,43.],[44.,43.]],width=2.,height=-.05),
                       PlanSegment([[48.,43.],[48.5,43.]],width=1.,height=-.1),
                       PlanSegment([[65.,60.],[90.,60.]],width=2.,height=-.25)])
    p=ComplexProject(Simulation(c));template=ComplexTemplate.capture(c);p.add_shared(template,position=(100.,20.))
    f=p.units[0].simulation.install_conventional(ConventionalConfig(base_x=12.,base_y=100.,work_rect=(40.,40.,6.,6.),truck_count=1,refuel_l_min=3000.))
    reserved=p.shared_site.core._conventional_mask.copy();released=False;cleanup_flight=False
    for _ in range(1800):
        p.step()
        if not np.any(p.shared_site.core._conventional_mask):
            released=True
            assert f.ground_stage_drained and f.carried_mass_kg==0
            assert all(m.state in ('PARKED','REFUELING','WAIT_FUEL') and not m.path and np.linalg.norm(m.position[:2]-m.home[:2])<.1 for m in f.machines)
            for unit in p.units:
                for drone in unit.simulation.drones:
                    if drone.trip is not None and drone.trip.source=='cut':
                        x,y=drone.trip.dig
                        if reserved[int(y/p.shared_site.core.cell),int(x/p.shared_site.core.cell)]:cleanup_flight=True
            if cleanup_flight:break
    assert released and cleanup_flight and f.excavated_m3>0 and f.delivered_m3>0
    assert np.all(p.shared_site.owners[p.shared_site.core.work_mask]>0)
    balanced(p)


def test_pause_or_loaded_cargo_cannot_release_the_ground_allocation():
    p,template,f=mixed();site=p.shared_site;before=site.core._conventional_mask.copy()
    f.set_enabled(False)
    assert not f.ground_stage_drained and not site.release_conventional_stage()
    np.testing.assert_array_equal(before,site.core._conventional_mask)
    f._work_complete=True;f.machines[0].state='DRIVING';f.machines[0].path=[tuple(f.machines[0].home[:2])]
    assert not f.ground_stage_drained and not site.release_conventional_stage()
    np.testing.assert_array_equal(before,site.core._conventional_mask)
    f.machines[0].state='PARKED';f.machines[0].path=[]
    truck=f.machines[1]
    truck.cargo_kg=site.core.scoop(45.,40.,40.,source='conventional',radius=.5)
    assert truck.cargo_kg>0
    f.excavated_mass_kg+=truck.cargo_kg
    f.excavated_m3+=truck.cargo_kg/site.core.config.terrain.density
    assert not f.ground_stage_drained and not site.release_conventional_stage()
    np.testing.assert_array_equal(before,site.core._conventional_mask)
    balanced(p)


def test_empty_home_bay_remains_reserved_without_an_air_obstacle_roof():
    p,template,f=mixed();site=p.shared_site;truck=f.machines[1];home=truck.home.copy()
    amount=20.;f.fuel_stock_l-=amount;truck.fuel_l+=amount
    assert f._assign_route(truck,truck.position[:2]+[12.,0.],'PARKED')
    for _ in range(3):f._move(truck)
    assert np.linalg.norm(truck.position[:2]-home[:2])>truck.radius*2
    assert site.ground_occupied('complex_2',(home[0]-.2,home[1]-.2,.4,.4))
    assert site.obstacle_height(home[:2])==pytest.approx(site.core.sample(*home[:2]))
    assert not site.ground_occupied('conventional',(home[0]-.2,home[1]-.2,.4,.4))
    balanced(p)
