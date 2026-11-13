"""A relocation request must not idle an otherwise productive multi-Krot fleet."""
import numpy as np
import pytest
from ats.simulation import Simulation
from ats.excavator import ExcavatorState as ES
from ats.drone import DroneState as DS
from ats.inventory import Location as L
from tests.test_excavator_fleet import config,balanced
from tests.test_scheduler_utilization import _initial_hook


def waiting_world(robots=3):
    c=config(robots);c.logistics.drone_spare_sets=7
    s=Simulation(c);w=s.logistics;e=w.excavators[-1]
    for dx,dy in ((0.,8.),(0.,-8.),(8.,0.),(-8.,0.)):
        destination=e.position+np.array([dx,dy,0.])
        destination[2]=s.terrain.sample(*destination[:2])
        if e.stable(destination,s.terrain):break
    else:raise AssertionError('Fixture needs a safe relocation destination')
    e.destination=destination;e.supports=False;e.state=ES.WAITING_RELOCATION
    return s,e


def deliver_rig(s,e,owner=None):
    s.logistics.inventory.move('group_rig',L.WORKSITE_RACK,e.id,e.position,'STORED_READY',owner)


@pytest.mark.parametrize('robots',[3,10])
def test_preparation_keeps_free_diggers_and_other_krots_working(robots):
    s,e=waiting_world(robots);w=s.logistics
    for _ in range(55):s.step()
    assert w.group is None and e.state==ES.WAITING_RELOCATION
    assert len(w.direct_ids)>=2
    assert any(robot.excavated_kg>0 for robot in w.excavators if robot is not e)
    assert any(j.kind=='DELIVER_RIGGING' and j.status in ('QUEUED','ACTIVE') for j in w.jobs)
    balanced(s)


def test_rig_delivery_is_requested_while_ongoing_digging_finishes():
    s,e=waiting_world();w=s.logistics
    # Dispatch a real flight first, then introduce the relocation request.
    e.state=ES.IDLE;w.direct_dispatch();assert len(w.direct_ids)==4
    e.state=ES.WAITING_RELOCATION;w.try_lift()
    jobs=[j for j in w.jobs if j.kind=='DELIVER_RIGGING']
    assert len(jobs)==1 and jobs[0].excavator_serial==e.serial
    assert jobs[0].priority<=2 and w.group is None
    assert w.inventory.items['group_rig'].location==L.BASE_STORAGE
    balanced(s)


def test_ready_rig_stops_new_courier_work_until_the_team_has_assembled():
    s,e=waiting_world();w=s.logistics;deliver_rig(s,e)
    d=s.drones[0];_initial_hook(s,d);w.courier_active=True;w.courier_id=d.identifier
    box=next(c for c in w.containers if c.excavator_serial!=e.serial)
    box.soil_kg=s.terrain.scoop(60.,60.,40.)
    assert box.soil_kg>0
    w.inventory.move(box.id,L.FILLED_QUEUE,w.robot(box.excavator_serial).id,
                     box.position,'FULL_READY')
    w.generate_jobs();job=next(j for j in w.jobs if j.kind=='PICKUP_FULL_CONTAINER')
    w.dispatch();w.direct_dispatch()
    assert job.status=='QUEUED' and not w.active and not w.direct_ids
    assert box.reserved_by==job.id and box.soil_kg>0
    balanced(s)


def test_preparation_keeps_its_owner_and_promotes_an_older_queued_rig_job():
    s,e=waiting_world();w=s.logistics
    with w.robot_context(e):job=w.new_job('DELIVER_RIGGING',w.inventory.items['group_rig'],s.base.position,e.position,6)
    # A second request must not redirect a rig already reserved for this one.
    other=w.excavators[0];other.destination=other.position+np.array([0.,8.,0.]);other.state=ES.WAITING_RELOCATION
    assert w.relocation_robot is e
    w.try_lift()
    assert job.priority==2 and job.excavator_serial==e.serial
    assert w.relocation_robot is e and w.lift_muster_robot is None
    deliver_rig(s,e,job.id);w.inventory.release(job.id);job.status='COMPLETED'
    assert w.relocation_robot is e and w.lift_muster_robot is e
    balanced(s)


def test_rig_in_a_couriers_cargo_does_not_prematurely_reserve_the_fleet():
    s,e=waiting_world();w=s.logistics;w.try_lift()
    d=s.drones[0];_initial_hook(s,d);w.courier_active=True;w.courier_id=d.identifier
    w.dispatch();job=w.active[d.identifier];assert job.kind=='DELIVER_RIGGING'
    w._load(d,job);s._clear_slot(d)
    d.state=DS.TO_DIG
    # Even directly over the target, a reserved airborne rig is not available.
    rig=w.inventory.items['group_rig'];rig.position=tuple(e.position)
    assert w.lift_muster_robot is None and not w.direct_should_return(s.drones[1])
    w.direct_dispatch();assert len(w.direct_ids)>=2
    assert w.group is None and rig.reserved_by==job.id
    balanced(s)


def test_no_safe_destination_does_not_ground_the_fleet_or_start_a_lift(monkeypatch):
    s,e=waiting_world();w=s.logistics;deliver_rig(s,e)
    e.destination=w.excavators[0].position.copy()  # occupied by another Krot
    monkeypatch.setattr(e,'find_site',lambda *a:None)
    w.try_lift();w.direct_dispatch()
    assert e.destination is None and e.reason=='WAITING_SAFE_RELOCATION_SITE'
    assert w.group is None and w.lift_muster_robot is None and len(w.direct_ids)>=2
    assert 'площадки' in w.lift_preparation()['reason']
    balanced(s)


def congested_world():
    s,e=waiting_world();w=s.logistics
    for i,box in enumerate(w.containers):
        box.soil_kg=s.terrain.scoop(30.+i*2.,60.,60.)
        assert box.soil_kg>0
        w.inventory.move(box.id,L.FILLED_QUEUE,w.robot(box.excavator_serial).id,box.position,'FULL_READY')
    w.generate_jobs()
    return s,e


def test_relocation_overtakes_routine_haul_backlog_without_discarding_soil():
    s,e=congested_world();w=s.logistics;ids=set(w.inventory.items);dig_seconds=0
    for _ in range(900):
        s.step();dig_seconds+=len(w.direct_ids)
        assert s.state.value=='RUNNING',s.reason
        assert not w.direct_ids.intersection(w.active)
        if int(s.time)%30==0:balanced(s)
        if w.lift_count:break
    assert w.lift_count==1 and e.relocations==1 and dig_seconds>0
    assert any(j.kind=='PICKUP_FULL_CONTAINER' and j.status=='QUEUED' for j in w.jobs)
    assert any(c.soil_kg>0 for c in w.containers) and set(w.inventory.items)==ids
    np.testing.assert_allclose(e.position,e.destination,atol=1e-7)
    balanced(s)


def test_save_load_exactly_resumes_preparation_rendezvous_and_group_flight(tmp_path):
    from ats.checkpoint import save_state,load_state
    s,e=congested_world();seen=set()
    for _ in range(900):
        s.step();w=s.logistics
        assert s.state.value=='RUNNING',s.reason
        phase=(w.group.phase if w.group is not None else w.lift_preparation()['phase'] if w.lift_preparation() else '')
        if phase not in ('PREPARATION','RENDEZVOUS','FLY') or phase in seen:continue
        seen.add(phase);path=tmp_path/(phase+'.atsstate');save_state(path,s);other=load_state(path).simulation
        assert other.report()==s.report()
        for _ in range(25):s.step();other.step()
        assert other.report()==s.report()
        balanced(s);balanced(other)
        if len(seen)==3:break
    assert seen=={'PREPARATION','RENDEZVOUS','FLY'}


def test_resetting_a_rig_courier_keeps_its_cargo_and_finishes_the_relocation():
    s,e=waiting_world();w=s.logistics;ids=set(w.inventory.items);reset=False
    for _ in range(1200):
        s.step()
        assert s.state.value=='RUNNING',s.reason
        if not reset:
            job=next((j for j in w.active.values() if j.kind=='DELIVER_RIGGING' and j.phase=='DESTINATION'),None)
            if job is not None:
                w.reset_drone_order(job.executor);reset=True
        if w.lift_count:break
    assert reset and e.relocations==1 and w.lift_count==1 and not w.recall_ids
    assert set(w.inventory.items)==ids
    balanced(s)


def test_work_finished_during_preparation_cancels_only_the_unstarted_rig_trip():
    s,e=waiting_world();w=s.logistics
    s.config.terrain.completion_volume=.0001
    from ats.plan import PlanConfig,PlanSegment
    s.apply_plan(PlanConfig([PlanSegment([[60.25,60.25]],width=.5,height=-.02),
                            PlanSegment([[70.,90.],[71.,90.]],mode='fill',width=1.,height=.002)]))
    assert s.terrain.remaining_cut>s.config.terrain.completion_volume
    e.state=ES.WAITING_RELOCATION;e.destination=e.position+np.array([0.,8.,0.])
    w.try_lift();rig_job=next(j for j in w.jobs if j.kind=='DELIVER_RIGGING')
    # Finish the small cut with real soil, leaving a loaded box and fill work.
    box=w.containers[0];box.soil_kg=s.terrain.scoop(60.25,60.25,80.,radius=2.)
    assert box.soil_kg>0
    w.inventory.move(box.id,L.FILLED_QUEUE,w.robot(box.excavator_serial).id,box.position,'FULL_READY')
    assert s.terrain.remaining_cut==0 and s.terrain.remaining_fill>0
    w.try_lift()
    assert rig_job.status=='CANCELLED' and not w.inventory.items['group_rig'].reserved_by
    assert w.group is None and w.lift_muster_robot is None and e.destination is None
    balanced(s)


def test_successive_krot_moves_have_productive_work_between_them():
    s,last=waiting_world();w=s.logistics;second=w.excavators[-2]
    second.destination=second.position+np.array([0.,8.,0.]);assert second.stable(second.destination,s.terrain)
    second.state=ES.WAITING_RELOCATION;second.supports=False
    workers_between=0;targets=set()
    for _ in range(2000):
        s.step()
        assert s.state.value=='RUNNING',s.reason
        if w.group:targets.add(w.group.excavator_serial)
        if w.lift_count==1:workers_between+=len(w.direct_ids)
        if int(s.time)%60==0:balanced(s)
        if w.lift_count==2:break
    assert w.lift_count==2 and targets=={second.serial,last.serial}
    assert second.relocations==1 and last.relocations==1 and workers_between>0
    balanced(s)


def test_ui_shows_the_target_krot_and_current_rendezvous_blocker():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.checkpoint import Session
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s,e=waiting_world();w=MainWindow(s.config,session=Session(s));w.timer.stop()
    try:
        w.refresh(True)
        assert f'Крот {e.serial}' in w.compact_telemetry.text()
        assert 'строп' in w.compact_telemetry.text()
        deliver_rig(s,e);s.drones[0].charge_requested=True;w.refresh(True)
        assert 'Д1' in w.compact_telemetry.text() and 'заряд' in w.compact_telemetry.text()
        assert f'Крот {e.serial}' in w.fleet_table.item(1,1).toolTip()
        w.show_logistics();w.logistics_window.timer.stop();w.logistics_window.tabs.setCurrentIndex(0)
        assert f'Крот {e.serial}' in w.logistics_window.status.text()
        assert 'Д1' in w.logistics_window.status.text()
    finally:w.close();app.processEvents()


def test_manual_landing_postpones_the_lift_and_still_drains_soil_cargo():
    s,e=waiting_world();w=s.logistics;deliver_rig(s,e)
    box=w.containers[0];box.soil_kg=s.terrain.scoop(60.,60.,40.)
    assert box.soil_kg>0
    w.inventory.move(box.id,L.FILLED_QUEUE,w.robot(box.excavator_serial).id,box.position,'FULL_READY')
    s.request_landing();w.try_lift()
    assert w.group is None and w.lift_muster_robot is None
    for _ in range(500):
        s.step();assert s.state.value=='RUNNING',s.reason
        assert w.group is None
        if box.soil_kg==0 and not w.active and all(d.grounded for d in s.drones):break
    assert box.soil_kg==0 and w.lift_count==0 and s.landing_requested
    assert not w.active and all(d.grounded for d in s.drones)
    balanced(s)
