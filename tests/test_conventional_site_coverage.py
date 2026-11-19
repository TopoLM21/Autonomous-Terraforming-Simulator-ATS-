"""The representative ramp/edge stances do not cover every safe working position."""
import numpy as np

from ats.checkpoint import save_state,load_state
from ats.conventional import ConventionalConfig,ConventionalFleet
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from tests.test_conventional import balanced,fuel_tanks
from tests.test_conventional_workspace import workspace_config


def off_axis_work(*,closed=False,side_work=False):
    config=workspace_config()
    segments=[PlanSegment([[80.2,44.25],[80.3,44.25]],width=.5,height=-.3)]
    if side_work:
        segments.append(PlanSegment([[46.2,35.25],[46.3,35.25]],width=.5,height=-.3))
    config.plan=PlanConfig(segments)
    axis=(np.arange(config.terrain.n)+.5)*config.terrain.cell_size
    x,y=np.meshgrid(axis,axis)
    height=np.zeros_like(x)
    # Four raised cells block the cardinal approaches while leaving a real
    # diagonal stance. A complete ring instead blocks all reachable stances.
    for angle in np.linspace(0.,2*np.pi,32 if closed else 4,endpoint=False):
        ix=np.argmin((x-80.25-7.2*np.cos(angle))**2+(y-44.25-7.2*np.sin(angle))**2)
        height.ravel()[ix]=.4
    config.initial_height=height;config.reference_height=height.copy();config.local_reference=True
    sim=Simulation.conventional_only(config)
    # This reachability fixture requires the shallow perimeter of its
    # single-cell cut as well as the central 0.075 m³.
    crew=ConventionalConfig(base_x=12.,base_y=12.,work_rect=(35.,30.,60.,40.),cut_tolerance_m=0.)
    if side_work:
        crew.ramp_entry_m=(35.,44.25);crew.ramp_end_m=(95.,44.25)
    fleet=ConventionalFleet(sim,crew)
    if side_work:
        fleet.excavator.position[:]=(39.25,35.25,0.)
    sim.conventional=fleet;fuel_tanks(fleet)
    return sim,fleet


def test_missed_diagonal_working_site_is_reached_and_actually_excavated():
    sim,fleet=off_axis_work()
    excavator,truck=fleet.machines
    before=sim.terrain.height.copy()
    assert fleet._choose_excavator_site(excavator),excavator.reason
    assert excavator.state=='DRIVING' and fleet.route_calculations<=8
    assert all(fleet.segment_safe(a,b,excavator.radius,excavator)
               for a,b in zip([excavator.position[:2],*excavator.path[:-1]],excavator.path))
    np.testing.assert_array_equal(sim.terrain.height,before)
    for _ in range(300):
        sim.step()
        if fleet.excavated_m3:break
    assert fleet.excavated_m3>.1 and truck.cargo_kg>0
    assert excavator.distance_m>0 and excavator.fuel_used_l>0
    balanced(sim,fleet)


def test_closed_stances_finish_a_bounded_search_and_resume_partway(tmp_path):
    sim,fleet=off_axis_work(closed=True);excavator=fleet.excavator
    before=fleet._work_remaining().copy()
    assert not fleet._choose_excavator_site(excavator)
    assert 'Лимит' in excavator.reason and excavator.site_search_cursor>0
    path=tmp_path/'partial-stance-search.atsstate';save_state(path,sim)
    restored=load_state(path).simulation.conventional
    assert restored.excavator.site_search_cursor==excavator.site_search_cursor
    for _ in range(80):
        previous=excavator.site_search_cursor
        for fleet_to_check in (fleet,restored):
            calls=fleet_to_check.route_calculations
            assert not fleet_to_check._choose_excavator_site(fleet_to_check.excavator)
            assert fleet_to_check.route_calculations-calls<=8
        assert excavator.reason==restored.excavator.reason
        assert excavator.site_search_cursor==restored.excavator.site_search_cursor
        if not excavator.site_search_cursor:break
        assert excavator.site_search_cursor>previous
    assert not excavator.site_search_cursor and 'Рабочая позиция не найдена' in excavator.reason
    assert not fleet._work_complete and not fleet.excavated_m3 and not excavator.distance_m
    np.testing.assert_array_equal(fleet._work_remaining(),before)
    balanced(sim,fleet)


def test_unreachable_access_front_does_not_pin_reachable_side_cut(tmp_path):
    sim,fleet=off_axis_work(closed=True,side_work=True);excavator=fleet.excavator
    road=fleet.access_mask.copy();before=sim.terrain.height[road].copy()
    assert road.any() and fleet._available_target(excavator) is None
    assert fleet._choose_excavator_site(excavator)
    assert excavator.side_bucket and excavator.state=='WAIT_TRUCK'
    assert excavator.expose_target==excavator.target[:2]
    path=tmp_path/'side-cut.atsstate';save_state(path,sim)
    other=load_state(path).simulation
    for _ in range(120):
        sim.step();other.step()
        if fleet.excavated_m3:break
    assert fleet.excavated_m3>.1 and fleet.machines[1].cargo_kg>0
    assert not excavator.distance_m and not fleet._work_complete
    assert sim.report()==other.report()
    np.testing.assert_array_equal(sim.terrain.height[road],before)
    balanced(sim,fleet)
