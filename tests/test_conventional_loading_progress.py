"""Loading progress and safe cooperative bays, rather than tiny timed hauls."""
import numpy as np
import pytest
import json
import zipfile
from pathlib import Path
from copy import deepcopy

from ats.checkpoint import save_state,load_state
from tests.test_conventional import world,fuel_tanks,balanced


def waiting_load(kg=30.,trucks=1,**changes):
    s,f=world(trucks=trucks,soil=True,**changes);e=f.excavator;truck=f.machines[1];t=s.terrain
    for _ in range(250):
        s.step()
        if truck.state=='WAIT_LOAD' and e.target is not None:break
    assert truck.state=='WAIT_LOAD' and truck.cargo_kg==0.
    x,y,_=f._available_target(e);radius=max(t.cell*.71,.5);patch=t.patch(x,y,radius)
    floor,allowed=f._bucket_geometry(e,patch)
    mask=np.zeros(t.height.shape,dtype=bool);mask[patch]=allowed
    load=t.scoop_load(x,y,kg,source='conventional',radius=radius,max_volume=f.config.bucket_m3,
                      allowed_mask=mask,minimum_height=floor)
    assert load.mass==pytest.approx(kg)
    truck.soil=load;truck.cargo_kg=load.mass
    f.extracted_materials+=load.masses;f.excavated_mass_kg+=load.mass
    f.excavated_m3+=load.volume;e.excavated_m3+=load.volume;f._terrain_changed()
    truck.load_started=s.time;e.timer=0.
    balanced(s,f)
    return s,f,e,truck


def test_empty_truck_starts_idle_clock_at_actual_arrival_not_dispatch():
    s,f=world(trucks=1);fuel_tanks(f);truck=f.machines[1]
    truck.load_started=-1000.
    assert f._assign_route(truck,(70.,6.),'WAIT_LOAD')
    while truck.state=='DRIVING':
        s.time+=1;f._move(truck)
    assert truck.state=='WAIT_LOAD' and truck.load_started==s.time
    balanced(s,f)


def test_every_real_bucket_refreshes_idle_clock_and_keeps_partial_load():
    s,f,e,truck=waiting_load();truck.load_started=s.time-1000.
    before=truck.cargo_kg
    assert f._cut(e,truck)
    assert truck.cargo_kg>before and truck.load_started==s.time
    s.step()
    assert not (truck.state=='DRIVING' and truck.after_drive=='UNLOADING')
    balanced(s,f)


def test_one_missed_cut_replans_without_sending_a_tiny_load(monkeypatch):
    s,f,e,truck=waiting_load();cargo=truck.cargo_kg
    e.timer=f.config.bucket_cycle_s-1.
    monkeypatch.setattr(f,'_cut',lambda *args:False)
    s.step()
    assert e.target is None and truck.state=='WAIT_LOAD'
    assert truck.cargo_kg==cargo
    balanced(s,f)


def test_true_loading_idle_timeout_delivers_the_remaining_cargo(monkeypatch):
    s,f,e,truck=waiting_load();cargo=truck.cargo_kg
    e.target=None;e.state='PARKED';truck.load_started=s.time-299.
    monkeypatch.setattr(f,'_choose_excavator_site',lambda *args,**kwargs:False)
    s.step();assert truck.state=='WAIT_LOAD'
    s.step();assert truck.state=='WAIT_LOAD'
    s.step()
    assert truck.state=='DRIVING' and truck.after_drive=='UNLOADING'
    assert truck.cargo_kg==cargo
    balanced(s,f)


def test_partial_truck_can_follow_a_real_excavator_relocation():
    s,f,e,truck=waiting_load();cargo=truck.cargo_kg;clock=truck.load_started
    assert f._assign_route(e,(42.,60.),'WAIT_TRUCK')
    while e.state=='DRIVING':
        s.time+=1;f._move(e)
    e.target=f._available_target(e);assert e.target is not None
    assert np.linalg.norm(truck.position[:2]-e.position[:2])>f._loading_distance(truck)+.5
    s.step()
    assert truck.state=='DRIVING' and truck.after_drive=='WAIT_LOAD',truck.reason
    assert truck.cargo_kg==cargo and truck.load_started==clock
    assert all(f.segment_safe(a,b,truck.radius,truck)
               for a,b in zip([truck.position[:2],*truck.path[:-1]],truck.path))
    balanced(s,f)


@pytest.mark.parametrize('ending',['full','complete'])
def test_full_or_final_load_leaves_without_waiting_for_idle_timeout(ending):
    changes={'truck_body_m3':.02} if ending=='full' else {}
    s,f,e,truck=waiting_load(**changes);cargo=truck.cargo_kg
    if ending=='complete':
        s.terrain.target[f.work_mask]=s.terrain.height[f.work_mask]
        s.terrain.revision+=1
        assert not f._choose_excavator_site(e) and f._work_complete
    s.step()
    assert truck.state=='DRIVING' and truck.after_drive=='UNLOADING'
    assert truck.cargo_kg==cargo
    balanced(s,f)


def test_loading_progress_timestamp_and_cargo_resume_exactly(tmp_path):
    s,f,e,truck=waiting_load();path=tmp_path/'partial-loading.atsstate'
    save_state(path,s);other=load_state(path).simulation
    assert other.conventional.machines[1].load_started==truck.load_started
    assert other.report()==s.report()
    for _ in range(35):s.step();other.step()
    assert other.report()==s.report()
    np.testing.assert_array_equal(s.terrain.height,other.terrain.height)
    balanced(s,f);balanced(other,other.conventional)


def test_failed_new_loading_bay_retries_boundedly_then_hauls_idle_load(monkeypatch):
    s,f,e,truck=waiting_load();cargo=truck.cargo_kg
    assert f._assign_route(e,(42.,60.),'WAIT_TRUCK')
    while e.state=='DRIVING':s.time+=1;f._move(e)
    e.target=f._available_target(e);assert e.target is not None
    truck.load_started=s.time-295.;calls=[]
    monkeypatch.setattr(f,'_loading_point',lambda *args,**kwargs:calls.append(s.time) or False)
    for _ in range(35):
        s.step()
        if truck.state=='DRIVING' and truck.after_drive=='UNLOADING':break
    assert len(calls)==1
    assert truck.state=='DRIVING' and truck.after_drive=='UNLOADING'
    assert truck.cargo_kg==cargo
    balanced(s,f)


@pytest.mark.parametrize('incoming',[False,True])
def test_productive_reposition_waits_for_the_incoming_truck_to_arrive(monkeypatch,incoming):
    s,f,e,truck=waiting_load();position=e.position.copy();calls=[]
    if incoming:
        assert f._assign_route(truck,truck.position[:2]+[1.,0.],'WAIT_LOAD')
    monkeypatch.setattr(f,'target_yield',lambda *args:0.)
    monkeypatch.setattr(f,'_choose_excavator_site',lambda *args,**kwargs:calls.append(kwargs) or False)
    s.step()
    assert calls==([] if incoming else [dict(productive_only=True)])
    np.testing.assert_array_equal(e.position,position)
    balanced(s,f)


def test_clearing_a_bay_physically_withdraws_partial_load_without_dumping():
    s,f,e,truck=waiting_load();cargo=truck.cargo_kg
    old_truck=truck.position.copy();old_e=e.position.copy();height=s.terrain.height.copy()
    assert f._release_loading_bay(e)
    assert truck.state=='DRIVING' and truck.after_drive=='PARKED'
    assert truck.cargo_kg==cargo and e.budget_cache is None
    assert e.wait_for_truck==truck.id and truck.loading_hold_for==e.id
    np.testing.assert_array_equal(truck.position,old_truck)
    np.testing.assert_array_equal(e.position,old_e)
    np.testing.assert_array_equal(s.terrain.height,height)
    assert np.linalg.norm(np.asarray(truck.destination)-truck.position[:2])>=.5
    assert all(f.segment_safe(a,b,truck.radius,truck)
               for a,b in zip([truck.position[:2],*truck.path[:-1]],truck.path))
    balanced(s,f)


def test_clear_exit_does_not_invent_a_withdrawal_without_a_joint_proof(monkeypatch):
    s,f,e,truck=waiting_load();cargo=truck.cargo_kg
    position=truck.position.copy()
    monkeypatch.setattr(f,'_cooperative_return',lambda *args,**kwargs:None)
    assert not f._release_loading_bay(e)
    assert truck.state=='WAIT_LOAD' and truck.cargo_kg==cargo
    np.testing.assert_array_equal(truck.position,position)
    balanced(s,f)


def single_lane_exit(monkeypatch,f):
    """Conservative test obstruction: E's exit opens only after truckhome.

    The virtual proof, truck withdrawal and subsequent E routes still use the
    real physical router. This narrows the fixture without editing terrain or
    granting any otherwise unsafe movement.
    """
    cls=type(f);original=cls._home_route
    def route(self,m,validate=False):
        truck=self.machines[1]
        if m is self.excavator and np.linalg.norm(truck.position[:2]-truck.home[:2])>=.1:return None
        return original(self,m,validate)
    monkeypatch.setattr(cls,'_home_route',route)


def test_static_excavator_can_budget_a_verified_joint_return_without_tipping(monkeypatch):
    s,f,e,truck=waiting_load();single_lane_exit(monkeypatch,f)
    budget=f._return_budget(e,30.)
    assert budget is not None and e.budget_cache is None
    plan=f._cooperative_return_cache
    assert budget[0]>=f._route_seconds(truck,plan['truck_path'])+f._route_seconds(e,plan['path'])+30.+15.
    calculations=f.route_calculations
    for _ in range(8):assert f._return_budget(e,30.) is not None
    assert f.route_calculations==calculations
    s.step()
    assert truck.state=='WAIT_LOAD' and truck.cargo_kg==30.
    assert not getattr(e,'wait_for_truck',None)
    balanced(s,f)


def test_cooperative_cache_ignores_distant_trucks_but_invalidates_near_ones():
    s,f,e,truck=waiting_load(trucks=2);other=f.machines[2]
    other.position[:]=(110.,110.,0.)
    plan=f._cooperative_return(e,truck);assert plan is not None
    calculations=f.route_calculations
    other.position[:2]+=[2.,0.]
    assert f._cooperative_return(e,truck) is plan
    assert f.route_calculations==calculations
    other.position[:]=(20.,40.,0.)
    assert f._cached_cooperative_return(e,truck) is None
    balanced(s,f)


@pytest.mark.parametrize('command',['return','relocate'])
def test_cooperative_withdrawal_and_excavator_motion_resume_exactly(monkeypatch,tmp_path,command):
    s,f,e,truck=waiting_load();single_lane_exit(monkeypatch,f)
    before_e=e.position.copy();before_t=truck.position.copy();cargo=truck.cargo_kg
    if command=='return':assert f._go_home(e)
    else:assert f._prepare_productive_relocation(e,(42.,60.))
    assert e.wait_for_truck==truck.id and truck.state=='DRIVING' and truck.after_drive=='PARKED'
    np.testing.assert_array_equal(e.position,before_e)
    np.testing.assert_array_equal(truck.position,before_t)
    assert truck.cargo_kg==cargo
    state=tmp_path/(command+'.atsstate');save_state(state,s)
    with zipfile.ZipFile(state) as archive:manifest=json.loads(archive.read('manifest.json'))
    assert '_cooperative_return_cache' not in json.dumps(manifest)
    other=load_state(state).simulation
    assert other.report()==s.report()
    seen_e_motion=False
    for _ in range(180):
        s.step();other.step()
        assert s.report()==other.report()
        if np.linalg.norm(truck.position[:2]-truck.home[:2])>=.1:
            np.testing.assert_array_equal(e.position,before_e)
        else:seen_e_motion|=not np.array_equal(e.position,before_e)
        goal=e.home[:2] if command=='return' else np.array([42.,60.])
        if seen_e_motion and np.linalg.norm(e.position[:2]-goal)<.1:break
    assert seen_e_motion and e.distance_m>0.
    assert np.linalg.norm(e.position[:2]-goal)<.1
    assert truck.cargo_kg==cargo and f.delivered_mass_kg==0.
    balanced(s,f);balanced(other,other.conventional)


def test_warm_natural_joint_return_hints_resume_without_extra_astar(tmp_path):
    fixture=Path(__file__).parent/'fixtures'/'conventional-v23'/'churn.atsstate'
    s=load_state(fixture).simulation;f=s.conventional
    for _ in range(150):
        s.step()
        if any(m.state=='WAIT_LOAD' for m in f.machines[1:]) and f._cooperative_return(f.excavator) is not None:break
    assert getattr(f,'cooperative_return_hints',None) is not None
    assert f._cooperative_return_cache is not None
    state=tmp_path/'warm-joint-plan.atsstate';save_state(state,s)
    with zipfile.ZipFile(state) as archive:manifest=json.loads(archive.read('manifest.json'))
    encoded=json.dumps(manifest)
    assert '_cooperative_return_cache' not in encoded and 'cooperative_return_hints' in encoded
    q=load_state(state).simulation
    assert q.report()==s.report()
    for _ in range(10):
        s.step();q.step()
        assert q.conventional.route_calculations==f.route_calculations
        assert q.report()==s.report()
    np.testing.assert_array_equal(q.terrain.height,s.terrain.height)
    balanced(s,f);balanced(q,q.conventional)


def test_a_serialized_hint_is_rejected_when_its_real_segments_are_blocked(monkeypatch):
    s,f,e,truck=waiting_load();assert f._cooperative_return(e,truck) is not None
    f._cooperative_return_cache=None
    monkeypatch.setattr(f,'segment_safe',lambda *args:False)
    assert f._checked_cooperative_hint(e,truck) is None
    assert f._cooperative_return_cache is None
    balanced(s,f)


def test_joint_return_duration_counts_every_discrete_waypoint_tick():
    s,f=world(trucks=1);truck=f.machines[1];p=truck.position[:2].copy()
    # _move consumes a tick even for an already reached explicit waypoint.
    assert f._route_seconds(truck,[p,p+[1.,0.],p+[1.,0.]])==3.


def test_rebinding_translates_pending_goal_without_mutating_the_live_draft():
    from ats.active_expansion import _fleet_preview
    from ats.simulation import Simulation
    s,f,e,truck=waiting_load();assert f._cooperative_return(e,truck) is not None
    e.pending_relocation=dict(goal=(42.,60.),requested_at=s.time)
    e.wait_for_truck=truck.id;truck.loading_hold_for=e.id
    config=deepcopy(s.config);config.terrain.n+=32
    replacement=Simulation.conventional_only(config).terrain
    translated=_fleet_preview(f,replacement,np.array([4.,6.]))
    assert translated.excavator.pending_relocation==dict(goal=(46.,66.),requested_at=s.time)
    assert e.pending_relocation==dict(goal=(42.,60.),requested_at=s.time)
    assert translated.excavator.wait_for_truck==truck.id
    assert translated.machines[1].loading_hold_for==e.id
    assert translated._cooperative_return_cache is None and translated.cooperative_return_hints is None
    assert f._cooperative_return_cache is not None and f.cooperative_return_hints is not None
    balanced(s,f)
