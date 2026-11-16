"""A newly allocated long access road must expose its first real bucket cut."""
from copy import deepcopy
import numpy as np
import pytest
from ats.checkpoint import save_state,load_state
from ats.config import Config
from ats.conventional import ConventionalConfig,ConventionalFleet
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from tests.test_conventional import fuel_tanks,balanced
from tests.test_conventional_siting import channel_site
from tests.test_conventional_workspace import workspace_config


def deep_site(angle=0.,depth=20.,width=None):
    sim,config=channel_site(angle=angle,depth=depth,side=512.)
    if width is not None:
        # Rebuild the physical target after setting the user's real width.
        sim.config.plan.segments[0].width=width
        sim=Simulation.conventional_only(sim.config)
        config=ConventionalConfig.for_simulation(sim)
    # Physical stock/pump and bucket cycles remain finite. These values only
    # shorten the test's wait before the first normal assignment and scoop.
    config.refuel_l_min=3000.;config.bucket_cycle_s=.5
    return sim,sim.install_conventional(config)


def start_approach(sim,fleet):
    before=sim.terrain.height.copy()
    for _ in range(20):
        sim.step()
        if fleet.excavator.state=='DRIVING':break
    e=fleet.excavator
    assert e.state=='DRIVING',(e.state,e.reason)
    assert e.after_drive=='WAIT_TRUCK' and e.path
    assert fleet.excavated_m3==0.
    np.testing.assert_array_equal(sim.terrain.height,before)
    assert all(fleet.segment_safe(a,b,e.radius,e)
               for a,b in zip([e.position[:2],*e.path[:-1]],e.path))
    return before


@pytest.mark.parametrize('angle,depth,width',[(0.,20.,None),(90.,20.,None),(45.,12.,None),(0.,40.,60.)])
def test_new_deep_auto_patch_physically_reaches_first_cut_without_preexcavation(angle,depth,width):
    sim,fleet=deep_site(angle,depth,width);e=fleet.excavator
    start_approach(sim,fleet)
    checked_working_escape=False
    for _ in range(550):
        sim.step()
        if e.state=='WAIT_TRUCK' and e.target is not None and not checked_working_escape:
            escape=fleet._home_route(e,validate=True)
            assert escape is not None
            assert all(fleet.segment_safe(a,b,e.radius,e)
                       for a,b in zip([e.position[:2],*escape[:-1]],escape))
            checked_working_escape=True
        if fleet.excavated_m3>0.:break
    assert checked_working_escape
    assert fleet.excavated_m3>0.,(sim.time,e.state,e.reason)
    assert fleet.machines[1].cargo_kg>0.
    assert e.distance_m>10. and e.fuel_used_l>0.
    assert fleet.terrain.mass_change<0.
    assert 'Нет доступной рабочей полки' not in e.reason
    balanced(sim,fleet)


@pytest.mark.parametrize('stage',['before','approach'])
def test_new_deep_patch_resume_before_first_site_and_during_approach_is_exact(tmp_path,stage):
    sim,fleet=deep_site()
    if stage=='approach':
        start_approach(sim,fleet)
        for _ in range(5):sim.step()
        assert fleet.excavator.state=='DRIVING' and fleet.excavator.distance_m>0.
    assert fleet.excavated_m3==0.
    path=tmp_path/(stage+'.atsstate');save_state(path,sim)
    other=load_state(path).simulation
    assert other.report()==sim.report()
    for _ in range(350):
        sim.step();other.step()
        if fleet.excavated_m3>0.:break
    assert fleet.excavated_m3>0.,fleet.excavator.reason
    assert other.report()==sim.report()
    np.testing.assert_array_equal(other.terrain.height,sim.terrain.height)
    balanced(sim,fleet);balanced(other,other.conventional)


@pytest.mark.parametrize('wait',['fuel','shift'])
def test_reachable_first_shelf_reports_dispatch_wait_instead_of_missing_geometry(wait):
    sim,fleet=deep_site();e=fleet.excavator;before=sim.terrain.height.copy()
    if wait=='shift':
        fuel_tanks(fleet)
        sim.time=(fleet.config.lunch_after_hours+fleet.config.lunch_hours/2)*3600
    calls=fleet.route_calculations;position=e.position.copy()
    assert not fleet._choose_excavator_site(e)
    expected='Недостаточно топлива' if wait=='fuel' else 'Недостаточно времени'
    assert e.reason.startswith(expected),e.reason
    assert fleet.reason==e.reason and fleet.route_calculations-calls<=8
    np.testing.assert_array_equal(e.position,position)
    np.testing.assert_array_equal(sim.terrain.height,before)
    assert fleet.excavated_m3==0.
    balanced(sim,fleet)


def test_previously_stranded_saved_excavator_replans_and_clears_its_stale_reason(tmp_path):
    sim,fleet=deep_site();fuel_tanks(fleet);e=fleet.excavator
    old='Нет доступной рабочей полки с безопасным выездом; рампа и остаток сохранены'
    e.reason=fleet.reason=old;e.next_plan_at=60.
    path=tmp_path/'old-stalled-start.atsstate';save_state(path,sim)
    restored=load_state(path).simulation;new=restored.conventional
    for _ in range(65):
        restored.step()
        if new.excavator.state=='DRIVING':break
    assert new.excavator.state=='DRIVING',new.excavator.reason
    assert old not in new.excavator.reason and old not in new.reason
    assert restored.time>=60. and new.excavated_m3==0.
    balanced(restored,new)


def test_real_steep_barrier_keeps_excavation_blocked_and_search_bounded():
    config=workspace_config();n=config.terrain.n;cell=config.terrain.cell_size
    axis=(np.arange(n)+.5)*cell
    heights=np.broadcast_to(np.where(axis<30.,0.,100.),(n,n)).copy()
    config.initial_height=heights;config.reference_height=heights.copy()
    config.local_reference=True
    config.terrain.floor=-100.;config.terrain.dump_rect=(5.,90.,20.,25.)
    config.plan=PlanConfig([PlanSegment([[40.,50.],[90.,50.]],width=8.,height=-.3)],
        channel=dict(start=[40.,50.],tangent=[1.,0.],cell_size=cell))
    sim=Simulation.conventional_only(config)
    fleet=sim.install_conventional(ConventionalConfig(base_x=12.,base_y=12.,work_rect=(35.,40.,60.,20.)))
    fuel_tanks(fleet);e=fleet.excavator;before=deepcopy(sim.report());surface=sim.terrain.height.copy()
    approach=np.asarray(fleet.ramp['entry_m'])
    assert not fleet.segment_safe(e.position,approach,e.radius,e)
    calls=fleet.route_calculations
    assert not fleet._choose_excavator_site(e)
    assert fleet.route_calculations-calls<=8
    assert e.state=='PARKED' and not fleet._work_complete
    assert ('Лимит' in e.reason or 'Нет доступной рабочей полки' in e.reason),e.reason
    assert fleet.excavated_m3==0. and sim.time==before['time_seconds']
    np.testing.assert_array_equal(sim.terrain.height,surface)
    balanced(sim,fleet)


def sloping_regional_channel(gradient):
    config=Config();config.terrain.n=1024;config.terrain.cell_size=.5
    config.terrain.floor=-100.;config.terrain.dump_rect=(5.,472.,25.,35.)
    axis=(np.arange(config.terrain.n)+.5)*config.terrain.cell_size
    surface=np.broadcast_to(gradient*axis,(config.terrain.n,config.terrain.n)).copy()
    config.initial_height=surface;config.reference_height=surface.copy();config.local_reference=True
    config.start_hour=6.
    # The regional route continues outside this physical window. Later anchors
    # cannot hide a failed length solve by silently skipping to the route end.
    config.plan=PlanConfig([PlanSegment([[20.,256.],[1200.,256.]],width=60.,height=-40.)],
        channel=dict(start=[20.,256.],tangent=[1.,0.],cell_size=.5))
    return Simulation.conventional_only(config)


def test_ramp_length_converges_on_rising_ground_before_choosing_its_footprint():
    from ats.conventional_siting import prepare
    sim=sloping_regional_channel(.03);surface=sim.terrain.height.copy();before=sim.report()
    config=ConventionalConfig.for_simulation(sim)
    selected=prepare(sim,config)
    fleet=ConventionalFleet(sim,selected);entry=np.asarray(fleet.ramp['entry_m'])
    end=np.asarray(fleet.ramp['end_m']);length=float(np.linalg.norm(end-entry))
    full_depth=float(np.maximum(sim.terrain.height-sim.terrain.target,0.)[
        fleet.work_mask&sim.terrain.dig_mask].max())
    assert length>=full_depth/config.ramp_gradient
    assert length>=fleet.ramp['required_length_m']
    assert selected.work_rect[0]+selected.work_rect[2]<512.
    approach=entry-(end-entry)/length*(fleet.excavator.radius+sim.terrain.cell)
    route=fleet.route(fleet.excavator,approach)
    assert route is not None
    assert all(fleet.segment_safe(a,b,fleet.excavator.radius,fleet.excavator)
               for a,b in zip([fleet.excavator.position[:2],*route[:-1]],route))
    assert sim.conventional is None and sim.report()==before and sim.time==0.
    np.testing.assert_array_equal(sim.terrain.height,surface)


def test_rising_ground_without_room_for_required_ramp_refuses_without_purchase_or_cut():
    sim=sloping_regional_channel(.12);surface=sim.terrain.height.copy();before=sim.report()
    config=ConventionalConfig.for_simulation(sim)
    with pytest.raises(ValueError,match='окно|[Рр]амп|карт'):
        sim.install_conventional(config)
    assert sim.conventional is None and sim.report()==before and sim.time==0.
    assert not hasattr(sim.terrain,'_conventional_mask')
    np.testing.assert_array_equal(sim.terrain.height,surface)
