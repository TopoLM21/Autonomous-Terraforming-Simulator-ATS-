"""Conservative physical diesel earthmoving, independent from autonomous costs."""
import math
import numpy as np
import pytest
from ats.simulation import Simulation
from ats.conventional import ConventionalConfig,ConventionalFleet
from ats.soil import SoilProfile
from tests.test_conventional_workspace import workspace_config


def world(trucks=2,soil=False,**changes):
    config=workspace_config()
    if soil:config.soil_profile=SoilProfile.qattara()
    s=Simulation.conventional_only(config)
    cfg=ConventionalConfig(base_x=12.,base_y=12.,work_rect=(35.,40.,60.,20.),truck_count=trucks,refuel_l_min=3000.)
    for key,value in changes.items():setattr(cfg,key,value)
    s.conventional=ConventionalFleet(s,cfg)
    return s,s.conventional


def fuel_tanks(f):
    for m in f.machines:
        amount=f._tank(m)-m.fuel_l
        f.fuel_stock_l-=amount;m.fuel_l+=amount


def balanced(s,f):
    f.validate()
    assert abs(s.terrain.mass_change+f.carried_mass_kg)<1e-7
    if s.terrain.soil is not None:
        np.testing.assert_allclose(s.terrain.soil.volume_delta_by_material*s.terrain.soil.profile.densities+f.carried_soil.masses,0.,atol=1e-7)
    report=f.report()
    assert abs(report['fuel_balance_error_l'])<1e-7
    assert abs(report['water_balance_error_l'])<1e-7
    assert abs(report['food_balance_error_units'])<1e-7


@pytest.mark.parametrize('soil',[False,True])
def test_two_trucks_really_excavate_haul_dump_and_preserve_materials(soil):
    s,f=world(soil=soil)
    initial=[m.position.copy() for m in f.machines]
    for _ in range(1800):
        s.step()
        if f.delivered_m3>=1.:break
    assert f.delivered_m3>=1. and f.excavated_m3>=f.delivered_m3
    assert any(m.distance_m>20 for m in f.machines)
    assert all(m.engine_seconds>0 for m in f.machines)
    assert not all(np.array_equal(a,m.position) for a,m in zip(initial,f.machines))
    assert f.fuel_used_l>0 and f.water_used_l>0 and f.food_used_units==3
    assert s.terrain.revision>0
    balanced(s,f)


def test_time_is_spent_refueling_and_nothing_is_created_for_free():
    s,f=world(trucks=1,refuel_l_min=30.)
    for _ in range(120):s.step()
    assert f.machines[0].state=='REFUELING' and f.machines[0].fuel_l==pytest.approx(60.)
    assert f.machines[1].state=='WAIT_FUEL' and f.machines[1].fuel_l==0.
    assert f.excavated_m3==0 and f.fuel_stock_l==pytest.approx(4940.)
    assert f.fuel_used_l==0
    balanced(s,f)


def test_routes_detour_barriers_and_do_not_cross_water_under_footprint():
    s,f=world(trucks=1);truck=f.machines[1];fuel_tanks(f)
    t=s.terrain;t.height[:50,90:92]=2.;t.revision+=1
    goal=(70.,6.);path=f.route(truck,goal)
    assert path is not None and any(p[1]>25 for p in path)
    assert all(f.segment_safe(a,b,truck.radius,truck) for a,b in zip([truck.position[:2],*path[:-1]],path))
    # Water only beneath the outer part of the body is still an obstacle.
    f._marine=np.zeros(t.height.shape,dtype=bool)
    f._marine[12,145]=True
    assert not f.passable((70.,6.),truck.radius,truck)


def test_closed_route_is_finite_and_leaves_loaded_cargo_untouched():
    s,f=world(trucks=1);truck=f.machines[1];fuel_tanks(f)
    t=s.terrain;t.height[:,60:62]=4.;t.revision+=1
    assert f.route(truck,(70.,6.)) is None
    assert f.route_calculations==1
    assert truck.cargo_kg==0 and np.array_equal(truck.position,truck.home)


def test_return_preflight_refuses_a_late_outbound_trip():
    s,f=world(trucks=1);fuel_tanks(f)
    s.time=4*3600-60;truck=f.machines[1]
    assert f.shift()['seconds_left']==60
    assert not f._assign_route(truck,(70.,6.),'WAIT_LOAD')
    assert 'времени' in truck.reason and truck.state=='PARKED' and not truck.path
    balanced(s,f)


def test_lunch_weekly_rest_and_pause_do_not_start_refueling_or_work():
    s,f=world(trucks=1)
    for when in (4*3600,18*3600,4*86400):
        s.time=when;before=f.fuel_stock_l;old=[m.fuel_l for m in f.machines]
        s.step()
        assert [m.fuel_l for m in f.machines]==old and f.fuel_stock_l==before
        assert all(m.state=='PARKED' for m in f.machines)
    assert f.payroll_usd>0 and f.excavated_m3==0
    balanced(s,f)


def test_calendar_payroll_accrues_on_paid_rest_and_crews_are_distinct():
    s,f=world(trucks=1,shifts_per_day=3,shift_span_hours=8.)
    s.time=4*86400
    for _ in range(100):s.step()
    expected=(f.config.excavator_wage_egp_month+f.config.driver_wage_egp_month)*3*100/(30*86400*f.config.egp_per_usd)
    assert f.payroll_usd==pytest.approx(expected) and f.report()['crew_count']==6
    assert f.fuel_used_l==0 and f.excavated_m3==0
    balanced(s,f)


def test_pause_keeps_real_cargo_then_resume_can_unload_it():
    s,f=world(trucks=1)
    for _ in range(1200):
        s.step()
        if f.carried_mass_kg>200.:break
    assert f.carried_mass_kg>0
    before=f.carried_mass_kg;f.set_enabled(False)
    for _ in range(400):s.step()
    assert f.carried_mass_kg==pytest.approx(before)
    assert all(np.linalg.norm(m.position[:2]-m.home[:2])<.1 for m in f.machines)
    f.set_enabled(True)
    for _ in range(1500):
        s.step()
        if f.delivered_m3>0:break
    assert f.delivered_m3>0
    balanced(s,f)


def test_save_load_resumes_loaded_truck_and_physical_routes_exactly(tmp_path):
    from ats.checkpoint import save_state,load_state
    s,f=world(soil=True)
    for _ in range(1200):
        s.step()
        if f.carried_mass_kg>100.:break
    assert f.carried_mass_kg>100
    path=tmp_path/'diesel.atsstate';save_state(path,s);other=load_state(path).simulation
    assert other.report()==s.report()
    assert other.conventional.terrain is other.terrain
    for _ in range(50):s.step();other.step()
    assert other.report()==s.report()
    balanced(s,f);balanced(other,other.conventional)


def test_per_cell_ramp_and_machine_support_guards_survive_actual_relaxation():
    s,f=world(trucks=1);e=f.excavator;truck=f.machines[1]
    for _ in range(600):
        s.step()
        if f.excavated_m3>.1:break
    assert f.excavated_m3>.1
    for m in f.machines:assert f.passable(m.position[:2],m.radius,m)
    t=s.terrain
    floor=np.maximum(t.target,f.ramp_surface)
    assert np.all(t.height[f.ramp_mask]>=floor[f.ramp_mask]-1e-6)
    assert f.report()['ramp']['remaining_m3']>0
    balanced(s,f)


def test_cost_per_m3_counts_purchase_once_and_keeps_stock_value_separate():
    s,f=world(trucks=1)
    initial=f.report()
    assert initial['cost_per_m3_usd'] is None and initial['capital_cost_usd']==225000.
    assert initial['inventory_purchases_usd']>0
    for _ in range(700):
        s.step()
        if f.excavated_m3>.5:break
    r=f.report()
    assert r['cost_per_m3_usd']==pytest.approx((r['capital_cost_usd']+r['operating_cost_usd'])/r['excavated_m3'])
    assert r['capital_cost_usd']==initial['capital_cost_usd']
    assert r['inventory_purchases_usd']==initial['inventory_purchases_usd']
    before=f.fuel_stock_l;f.replenish(fuel_l=500.,water_l=20.,food_units=3.)
    assert f.fuel_stock_l==before+500 and f.report()['inventory_purchases_usd']>r['inventory_purchases_usd']
    balanced(s,f)


def test_full_swept_footprints_and_invalid_schedule_are_not_silently_accepted():
    s,f=world(trucks=1)
    for m in f.machines:
        length,width,_=m.dimensions_m
        assert m.radius>=math.hypot(length,width)/2
    with pytest.raises(ValueError):ConventionalConfig(shifts_per_day=3).validate()
    with pytest.raises(ValueError):f.replenish(fuel_l=-1.)
    f.machines[1].cargo_kg=100.
    with pytest.raises(ValueError):f.validate()


def test_both_trucks_complete_repeated_load_cycles_without_soil_loss():
    s,f=world()
    for _ in range(2400):
        s.step()
        if all(m.delivered_m3>.2 for m in f.machines[1:]):break
    assert all(m.delivered_m3>.2 for m in f.machines[1:])
    assert f.excavator.working_seconds>=300
    balanced(s,f)


def test_finishing_the_cut_still_unloads_remaining_cargo_and_returns_home():
    s,f=world(trucks=1)
    for _ in range(900):
        s.step()
        if f.carried_mass_kg>100:break
    assert f.carried_mass_kg>0
    # This isolates completion after a finite final bucket, preserving the real
    # cargo and unmodified terrain instead of fabricating extra soil.
    f._work_complete=True
    for _ in range(900):
        s.step()
        if f.carried_mass_kg==0 and all(np.linalg.norm(m.position[:2]-m.home[:2])<.1 for m in f.machines):break
    assert f.carried_mass_kg==0 and f.delivered_m3>0
    assert all(np.linalg.norm(m.position[:2]-m.home[:2])<.1 for m in f.machines)
    balanced(s,f)


def test_a_real_small_patch_finishes_unloads_and_parks_without_forcing_completion():
    from ats.plan import PlanConfig,PlanSegment
    c=workspace_config();c.plan=PlanConfig([PlanSegment([[42.,43.],[44.,43.]],width=2.,height=-.05)])
    s=Simulation.conventional_only(c)
    f=ConventionalFleet(s,ConventionalConfig(base_x=12.,base_y=12.,work_rect=(40.,40.,6.,6.),truck_count=1,refuel_l_min=3000.))
    s.conventional=f
    for _ in range(2000):
        s.step()
        if f._work_complete and f.carried_mass_kg==0 and all(np.linalg.norm(m.position[:2]-m.home[:2])<.1 for m in f.machines):break
    assert f._work_complete and f.report()['remaining_work_m3']<1e-7
    assert f.delivered_m3>0 and f.carried_mass_kg==0
    assert all(np.linalg.norm(m.position[:2]-m.home[:2])<.1 for m in f.machines)
    balanced(s,f)


def test_four_trucks_do_not_wall_the_excavator_into_its_parking_bay():
    s,f=world(trucks=4)
    for _ in range(900):
        s.step()
        if f.excavated_m3>.5:break
    assert f.excavated_m3>.5 and f.excavator.distance_m>0
    assert all(np.linalg.norm(a.home[:2]-b.home[:2])>=a.radius+b.radius+min(a.radius,b.radius)*2 for i,a in enumerate(f.machines) for b in f.machines[i+1:])
    balanced(s,f)


def test_a_failed_route_is_retried_at_bounded_intervals_without_skipping_resources(monkeypatch):
    s,f=world(trucks=1);fuel_tanks(f);calls=[]
    monkeypatch.setattr(f,'_return_budget',lambda machine,extra_s=0.: calls.append(machine.id))
    for _ in range(20):s.step()
    assert len(calls)==2  # one failed attempt per machine, not per second
    assert f.water_used_l>0 and f.payroll_usd>0 and f.food_used_units==2
    for _ in range(12):s.step()
    assert len(calls)==4
    balanced(s,f)


def test_one_excavator_has_one_loading_bay_and_extra_trucks_wait_at_camp():
    s,f=world(trucks=4)
    seen=False
    for _ in range(650):
        s.step()
        users=[m for m in f.machines[1:] if m.state=='WAIT_LOAD' or m.state=='DRIVING' and m.after_drive=='WAIT_LOAD']
        assert len(users)<=1
        if users:
            seen=True
            assert any(m.state in ('PARKED','WAIT_FUEL','REFUELING') for m in f.machines[1:] if m not in users)
    assert seen and f.excavated_m3>0
    balanced(s,f)


def test_access_ramp_is_wider_than_the_full_loaded_truck_footprint():
    s,f=world(trucks=1)
    truck=f.machines[1]
    assert f.ramp['width_m']>=truck.radius*2+1.
    with pytest.raises(ValueError):ConventionalConfig(ramp_width_m=6.).validate()


@pytest.mark.parametrize('soil',[False,True])
def test_independent_work_patch_does_not_inherit_an_autonomous_sector_block(soil):
    from types import SimpleNamespace
    s,f=world(trucks=1,soil=soil)
    for _ in range(700):
        s.step()
        if f.excavated_m3>.1:break
    assert f.excavated_m3>.1
    t=s.terrain;e=f.excavator;truck=f.machines[1]
    t.front=SimpleNamespace(sectors=True,legacy_pickups=[],cut_mask=lambda patch:np.zeros(t.height[patch].shape,dtype=bool))
    target=f._available_target(e);assert target is not None
    if soil:assert t.scoop_load(*target[:2],100.,radius=.5).mass==0
    else:assert t.scoop(*target[:2],100.,radius=.5)==0
    before=f.excavated_mass_kg
    assert f._cut(e,truck) and f.excavated_mass_kg>before
    balanced(s,f)
