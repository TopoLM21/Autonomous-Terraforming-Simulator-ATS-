from copy import deepcopy
import math
import numpy as np
import pytest
from ats.config import Config
from ats.complex_setup import SetupRequest,calculate,place
from ats.costs import default_catalog,quote,project_quote
from ats.work_schedule import WorkSchedule
from ats.checkpoint import save_state,load_state
from ats.complexes import ComplexTemplate
from ats.inventory import BatteryModule
from ats.complex_setup import recipe_config


def source():
    c=Config();c.base.latitude_deg=31.078;return c


def day(mode='hours',hours=6.):return WorkSchedule(mode,172,172,hours)


def balanced(s):
    r=s.report(full=False)
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002
    s.logistics.inventory.validate()


def test_reviewing_a_recipe_does_not_create_vehicles_or_inventory(monkeypatch):
    from ats.simulation import Simulation
    from ats.inventory import Inventory
    def fail(*a,**k):raise AssertionError('Review must not create physical actors')
    monkeypatch.setattr(Simulation,'__init__',fail);monkeypatch.setattr(Inventory,'__init__',fail)
    c=source();original=deepcopy(c)
    result=calculate(c,SetupRequest(),day())
    assert c==original and result.config.fleet.count==4
    assert result.estimate['total']>0 and result.design['limiting']['repeatable']
    assert not result.config.logistics.excavator_enabled


def test_recipe_preserves_edited_drone_and_logistics_parameters():
    from ats.experiments import experiment_config
    c=experiment_config(source());c.logistics.cycle_seconds=75.;c.logistics.pack_mass_kg=19.
    c.fleet.models[0].equipment.frame_mass_kg=63.
    other=recipe_config(c,SetupRequest(drones=7,excavator=True,containers=12))
    assert other.logistics.cycle_seconds==75. and other.logistics.pack_mass_kg==19.
    assert other.fleet.models[0].equipment.frame_mass_kg==63.
    assert other.logistics.container_count==12 and len(other.fleet.models)==7
    assert c.logistics.container_count==8 and len(c.fleet.models)==4


def test_day_shift_needs_less_hardware_and_all_selected_days_are_checked():
    period=WorkSchedule('continuous',60,65)
    full=calculate(source(),SetupRequest(),period)
    shift=calculate(source(),SetupRequest(),WorkSchedule('hours',60,65,6))
    assert shift.design['checked_days']==6
    assert shift.config.base.panel_area<full.config.base.panel_area
    assert shift.config.base.battery<full.config.base.battery
    assert shift.estimate['total']<full.estimate['total']
    for result in (full,shift):
        for audit in result.design['season_checks']:
            assert audit['repeatable'] and audit['storage_check']['unserved_kwh']<1e-7
            assert audit['storage_check']['daily_storage_drift_kwh']>=-1e-7
            assert abs(audit['storage_check']['energy_error_kwh'])<1e-7


def test_southern_hemisphere_calendar_can_cross_new_year():
    c=source();c.base.latitude_deg=-31.
    result=calculate(c,SetupRequest(),WorkSchedule('daylight',364,2))
    assert result.design['checked_days']==4
    assert [r['day_of_year'] for r in result.design['season_checks']]==[364,365,1,2]
    assert all(r['repeatable'] for r in result.design['season_checks'])


def test_entire_year_checks_winter_and_robot_equipment_is_real():
    from ats.winter_power import design_report
    result=calculate(source(),SetupRequest(excavator=True),WorkSchedule('hours',1,365,6))
    assert result.design['checked_days']==365
    assert result.design['calendar']['min_work_hours']==6
    assert result.design['limiting_day_of_year'] not in range(120,240)
    assert all(a['repeatable'] for a in result.design['season_checks'])
    s=place(result).simulation
    assert s.logistics.excavator is not None and len(s.logistics.containers)==8
    assert quote(s.config,layout=s.base.layout_spec,world=s.logistics)['total']==pytest.approx(result.estimate['total'])
    report=design_report(s.config,result.design)
    assert report['daily_energy_kwh']==pytest.approx(result.design['limiting']['demand_kwh'])
    assert 'calendar_design' in report
    s.config.operating_schedule=WorkSchedule('hours',1,365,1).to_data()
    s.run(30)
    assert s.logistics.excavator.excavated_kg==0
    balanced(s)


def test_daylight_shortage_and_polar_night_are_explicit():
    with pytest.raises(ValueError,match='световой день'):
        calculate(source(),SetupRequest(),WorkSchedule('hours',1,365,15))
    c=source();c.base.latitude_deg=80.
    with pytest.raises(ValueError,match='без солнечной генерации'):
        calculate(c,SetupRequest(),WorkSchedule('continuous',355,355))


def test_price_edit_does_not_modify_hardware_or_energy_calculations():
    c=source();result=calculate(c,SetupRequest(),day());before=quote(result.config)
    catalog=deepcopy(result.config.cost_catalog);catalog['prices']['solar_m2']+=10.
    after=quote(result.config,catalog=catalog)
    assert after['total']-before['total']==pytest.approx(result.config.base.panel_area*10)
    assert result.config.cost_catalog['prices']['solar_m2']==60.
    assert c.cost_catalog is None


def test_costs_count_owned_packs_once_during_swap_and_include_hot_stock():
    result=calculate(source(),SetupRequest(),day());session=place(result);s=session.simulation;w=s.logistics
    before=quote(s.config,layout=s.base.layout_spec,world=w)
    battery=next(b for b in w.inventory.select(BatteryModule) if b.compatibility.startswith('drone:'))
    w.replenish_batteries(battery.id,1,1.)
    after=quote(s.config,layout=s.base.layout_spec,world=w)
    assert after['total']>before['total']
    owned=[o.id for o in w.inventory.items.values()]
    s.request_landing();s.run(10);balanced(s)
    assert set(owned)==set(w.inventory.items)
    assert quote(s.config,layout=s.base.layout_spec,world=w)['total']==after['total']


def test_named_complex_is_deployed_only_after_place_and_resumes_exactly(tmp_path):
    result=calculate(source(),SetupRequest(name='Дневной комплекс'),day())
    session=place(result);s=session.simulation;p=session.complex_project
    assert p.units[0].name=='Дневной комплекс' and len(p.units)==1
    assert len(s.drones)==4 and len(s.slots)==4 and s.state.value=='RUNNING'
    assert quote(s.config,layout=s.base.layout_spec,world=s.logistics)['total']==pytest.approx(result.estimate['total'])
    for _ in range(25):p.step()
    path=tmp_path/'new-complex.atsstate';save_state(path,s,complex_project=p)
    other=load_state(path)
    for _ in range(60):p.step();other.complex_project.step()
    assert s.report()==other.simulation.report()
    assert p.report()==other.complex_project.report()
    assert s.config.cost_catalog==other.simulation.config.cost_catalog
    balanced(s);balanced(other.simulation)


def test_sixteen_aircraft_have_real_slots_ids_energy_and_correct_cost(tmp_path):
    from ats.maps import with_map_size
    from ats.plan import PlanConfig,PlanSegment
    c=with_map_size(source(),256);c.terrain.n=512;c.terrain.cell_size=.5
    c.terrain.dump_rect=(165.,24.,48.,48.);c.fleet.airspace_mode='shared_lanes';c.terrain.cut_policy='natural'
    c.plan=PlanConfig([PlanSegment([[96.,120.],[144.,120.]],width=20.,height=-.25)])
    result=calculate(c,SetupRequest(drones=16),day())
    assert result.design['platform_count']>16
    session=place(result);s=session.simulation
    assert s.state.value=='RUNNING',s.reason
    assert len(s.drones)==len(s.slots)==16
    assert len({d.serial for d in s.drones})==16
    ids=set(s.logistics.inventory.items)
    for _ in range(40):session.complex_project.step()
    assert set(s.logistics.inventory.items)==ids and all(d.cargo>0 for d in s.drones)
    assert {j.executor for j in s.logistics.jobs if j.kind=='EXCAVATE_DIRECT'}==set(range(1,17))
    assert s.state.value in ('RUNNING','COMPLETED'),s.reason
    assert all(d.battery>=d.config.battery*d.config.reserve-1e-7 for d in s.drones)
    balanced(s)
    path=tmp_path/'sixteen.atsstate';save_state(path,s,complex_project=session.complex_project)
    other=load_state(path).complex_project
    for _ in range(10):session.complex_project.step();other.step()
    assert s.report()==other.active.simulation.report()
    balanced(s);balanced(other.active.simulation)


def test_templates_keep_prices_and_calendar_without_terrain_or_vehicles(tmp_path):
    result=calculate(source(),SetupRequest(),day());template=result.template()
    path=tmp_path/'day-shift.atstemplate';template.save(path);other=ComplexTemplate.load(path)
    assert other.config.operating_schedule==result.config.operating_schedule
    assert other.config.cost_catalog==result.config.cost_catalog
    assert other.config.initial_height is None and other.config.georeference is None
    assert not hasattr(other,'simulation')


@pytest.mark.parametrize('value',[-1.,math.inf,math.nan])
def test_invalid_prices_are_rejected(value):
    book=default_catalog();book['prices']['crawler']=value
    with pytest.raises(ValueError):calculate(source(),SetupRequest(),day(),catalog=book)


def test_price_changes_and_addition_preserve_the_existing_complex_and_common_ground():
    first=calculate(source(),SetupRequest(),day());session=place(first);p=session.complex_project;s=session.simulation
    p.enable_shared();p.shared_site.enlarge(256)
    s.request_landing();ids=set(p.inventory_ids());height=p.shared_site.core.height.copy();time=p.time
    second=calculate(s.config,SetupRequest(drones=1,name='Помощник'),day())
    unit=p.add_shared(second.template());unit.simulation.request_landing()
    assert p.time==time and set(p.inventory_ids())>ids
    np.testing.assert_array_equal(p.shared_site.core.height,height)
    total=project_quote(u.simulation for u in p.units)
    assert total['priced_complexes']==2 and total['totals']['USD']>first.estimate['total']
    report=p.report();assert abs(report['equipment_mass_error_kg'])<1e-7 and abs(report['energy_error_j'])<.002
