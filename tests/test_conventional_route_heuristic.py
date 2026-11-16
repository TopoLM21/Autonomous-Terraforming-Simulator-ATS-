"""Long grid routes retain metre-scale clearance and exact endpoint costs."""
import math
import numpy as np
import pytest

from ats.config import Config
from ats.conventional import ConventionalConfig
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation


def diagonal_haul(*,barrier=False,start_offset=(0.,0.)):
    c=Config();c.terrain.n=512;c.terrain.cell_size=1.;c.terrain.grab_radius=1.
    c.plan=PlanConfig([PlanSegment([[260.,400.],[310.,400.]],width=30.,height=-.5)])
    if barrier:
        imported=np.zeros((512,512));imported[:,340:342]=10.
        c.reference_height=imported;c.initial_height=imported.copy();c.local_reference=True
    sim=Simulation.conventional_only(c)
    fleet=sim.install_conventional(ConventionalConfig(base_x=384.,base_y=116.,work_rect=(250.,370.,80.,80.)))
    e,truck=fleet.machines
    goal=np.array([270.25,400.35])
    # The camp obstructs the long straight departure. The excavator's body
    # remains beside the loading point, outside the real truck clearance.
    e.position[:]=(goal[0]-6.25,goal[1]+6.25,0.)
    truck.position[:2]+=start_offset
    return sim,fleet,truck,goal


def length(start,path):
    return sum(float(np.linalg.norm(np.asarray(b)-a))
               for a,b in zip([np.asarray(start),*map(np.asarray,path[:-1])],path))


def test_long_diagonal_detours_around_camp_with_the_original_search_budget():
    sim,fleet,truck,goal=diagonal_haul();before=sim.terrain.height.copy();position=truck.position.copy()
    assert fleet.config.max_route_nodes==1200
    assert fleet.passable(goal,truck.radius,truck)
    assert not fleet.segment_safe(truck.position,goal,truck.radius,truck)
    path=fleet.route(truck,goal)
    assert path is not None,fleet._route_failure
    assert path[-1]==tuple(goal)
    assert all(fleet.segment_safe(a,b,truck.radius,truck)
               for a,b in zip([truck.position[:2],*path[:-1]],path))
    # This camp can be passed by ordering the same NW/N steps differently.
    # The independent obstacle-free lower bound is therefore attainable. Its
    # last leg goes to the exact off-grid goal, not a snapped substitute.
    terminals=[np.array([x,y]) for x in range(266,275,2) for y in range(396,405,2)
               if np.linalg.norm(np.array([x,y])-goal)<=3.]
    def relaxed(q):
        dx,dy=np.abs(q-truck.position[:2])
        return max(dx,dy)+(math.sqrt(2.)-1.)*min(dx,dy)+float(np.linalg.norm(q-goal))
    assert length(truck.position[:2],path)==pytest.approx(min(map(relaxed,terminals)),abs=1e-9)
    np.testing.assert_array_equal(truck.position,position)
    np.testing.assert_array_equal(sim.terrain.height,before)
    assert sim.time==0. and fleet.excavated_m3==0. and truck.distance_m==0.


def test_non_grid_start_keeps_a_physically_checked_first_and_last_connector():
    sim,fleet,truck,goal=diagonal_haul(start_offset=(.35,.15));position=truck.position.copy()
    assert not fleet.segment_safe(truck.position,goal,truck.radius,truck)
    path=fleet.route(truck,goal)
    assert path is not None,fleet._route_failure
    assert path[-1]==tuple(goal)
    assert all(fleet.segment_safe(a,b,truck.radius,truck)
               for a,b in zip([truck.position[:2],*path[:-1]],path))
    np.testing.assert_array_equal(truck.position,position)
    assert fleet.config.max_route_nodes==1200 and sim.time==0.


def test_better_search_order_does_not_drive_through_an_actual_steep_barrier():
    sim,fleet,truck,goal=diagonal_haul(barrier=True)
    before=sim.terrain.height.copy();position=truck.position.copy()
    assert fleet.passable(goal,truck.radius,truck)
    assert fleet.route(truck,goal) is None
    assert 'Лимит' in fleet._route_failure or 'Нет безопасного пути' in fleet._route_failure
    assert fleet.config.max_route_nodes==1200
    np.testing.assert_array_equal(truck.position,position)
    np.testing.assert_array_equal(sim.terrain.height,before)
    assert sim.time==0. and fleet.excavated_m3==0.


def reverse_search_uncertain(monkeypatch,fleet,truck,goal):
    """Represent a bounded coarse search, never a physical obstacle waiver."""
    from ats.conventional import ConventionalFleet
    original=ConventionalFleet.route;calls=[]
    def route(owner,m,destination):
        if (owner is fleet and m is truck and np.linalg.norm(m.position[:2]-goal)<1e-8
                and np.linalg.norm(np.asarray(destination)[:2]-m.home[:2])<1e-8):
            owner.route_calculations+=1;calls.append(True)
            owner._route_failure='Лимит узлов локального поиска исчерпан; отсутствие пути не доказано'
            return None
        return original(owner,m,destination)
    monkeypatch.setattr(ConventionalFleet,'route',route)
    return calls


def stock_and_resources(fleet,truck):
    return (fleet.fuel_stock_l,fleet.fuel_used_l,fleet.water_stock_l,fleet.food_stock_units,
            truck.fuel_l,truck.fuel_used_l,truck.cargo_kg,truck.distance_m,truck.engine_seconds)


def test_assignment_uses_the_checked_reverse_when_fresh_search_cannot_prove_it(monkeypatch):
    from tests.test_conventional import fuel_tanks,balanced
    sim,fleet,truck,goal=diagonal_haul();fuel_tanks(fleet)
    position=truck.position.copy();before=stock_and_resources(fleet,truck)
    calls=reverse_search_uncertain(monkeypatch,fleet,truck,goal)
    # A fresh search fails at the destination; that does not invalidate the
    # physically symmetric path we are about to drive there.
    truck.position[:2]=goal
    assert fleet.route(truck,truck.home[:2]) is None and calls
    truck.position[:]=position
    assert fleet._assign_route(truck,goal,'WAIT_LOAD'),truck.reason
    cache=truck.budget_cache
    assert cache['position']==tuple(goal) and cache['path'][-1]==tuple(truck.home[:2])
    assert all(fleet.segment_safe(a,b,truck.radius,truck)
               for a,b in zip([goal,*cache['path'][:-1]],cache['path']))
    assert len(calls)==1  # The assignment needed no uncertain reverse search.
    assert stock_and_resources(fleet,truck)==before and sim.time==0.
    np.testing.assert_array_equal(truck.position,position)
    balanced(sim,fleet)


def test_assignment_rejects_a_reverse_whose_ground_has_changed(monkeypatch):
    from tests.test_conventional import fuel_tanks
    sim,fleet,truck,goal=diagonal_haul();fuel_tanks(fleet)
    outbound=fleet.route(truck,goal);assert outbound
    point=np.asarray(outbound[len(outbound)//2]);row,col=(point[::-1]/sim.terrain.cell).astype(int)
    # A real new obstruction invalidates the previously checked swept road.
    sim.terrain.height[row,col]+=10.;sim.terrain.revision+=1
    sim.terrain._touch_route_cells([row],[col])
    calls=reverse_search_uncertain(monkeypatch,fleet,truck,goal)
    position=truck.position.copy();before=stock_and_resources(fleet,truck);height=sim.terrain.height.copy()
    assert not fleet._assign_route(truck,goal,'WAIT_LOAD',outbound_path=outbound)
    assert calls and 'обратного пути' in truck.reason
    assert truck.state=='PARKED' and not truck.path and truck.destination is None
    assert stock_and_resources(fleet,truck)==before and sim.time==0.
    np.testing.assert_array_equal(truck.position,position)
    np.testing.assert_array_equal(sim.terrain.height,height)


def test_saved_reverse_cache_is_rechecked_when_its_terrain_token_changes(tmp_path,monkeypatch):
    from ats.checkpoint import save_state,load_state
    from tests.test_conventional import fuel_tanks
    sim,fleet,truck,goal=diagonal_haul();fuel_tanks(fleet)
    assert fleet._assign_route(truck,goal,'WAIT_LOAD')
    path=tmp_path/'checked-backtrack.atsstate';save_state(path,sim)
    restored=load_state(path).simulation
    assert restored.report()==sim.report()
    other=restored.conventional;m=other.machines[1]
    assert m.budget_cache['path']==truck.budget_cache['path']
    # Inspect the cache at its stated endpoint, without claiming a drive.
    m.position[:2]=goal
    back=other._home_route(m,validate=True);assert back
    token=m.budget_cache['token'];point=np.asarray(back[len(back)//2])
    row,col=(point[::-1]/restored.terrain.cell).astype(int)
    restored.terrain.height[row,col]+=10.;restored.terrain.revision+=1
    restored.terrain._touch_route_cells([row],[col])
    assert token!=restored.terrain.route_token(m.budget_cache['bounds'])
    calls=reverse_search_uncertain(monkeypatch,other,m,goal)
    before=stock_and_resources(other,m)
    assert other._home_route(m) is None and calls
    assert stock_and_resources(other,m)==before
