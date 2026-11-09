"""Contact charging must not strand a ready team beside an empty dock."""
import numpy as np
import pytest

from ats.checkpoint import save_state, load_state
from ats.config import Config
from ats.drone import DroneState as DS
from ats.experiments import experiment_config
from ats.simulation import Simulation


def service_fleet(*, count=4, slots=4):
    c=experiment_config(Config(),drones=count,landing_slots=slots,drone_spare_sets=0)
    c.logistics.excavator_enabled=False
    s=Simulation(c);s.request_landing()
    return s


def park(s,d,charge=.3):
    s._clear_slot(d);d.position[:]=s.wait_position(d.identifier)
    s.equipment_exported_energy+=d.battery-d.config.battery*charge
    d.battery=d.config.battery*charge;d.state=DS.WAIT_GROUND
    d.charge_requested=True;s._enqueue(d);s.logistics.sync_batteries()


def balanced(s):
    r=s.report(full=False)
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002
    assert all(d.battery>=d.config.battery*d.config.reserve-1e-7 for d in s.drones)
    assert len([i for i in s.slots if i is not None])==len(set(i for i in s.slots if i is not None))
    s.logistics.inventory.validate()


def test_empty_far_slot_does_not_hide_a_reachable_later_slot():
    s=service_fleet(count=2,slots=2);d=s.drones[0]
    park(s,d);s._clear_slot(s.drones[1])
    d.position[:2]=s.slot_position(1)[:2]+[0.,6.]
    far,near=(s._transfer_energy(d,i) for i in (0,1))
    assert far>near
    d.battery=d.config.battery*d.config.reserve+(far+near)/2
    s._assign_slots()
    assert d.slot==1 and s.slots==[None,d.identifier]
    assert d.state==DS.TRANSFER_TAKEOFF


def test_return_keeps_an_existing_reservation_instead_of_reserving_twice():
    s=service_fleet();d=s.drones[-1];s._clear_slot(s.drones[0])
    s._reserve_return(d)
    assert d.slot==3 and s.slots.count(d.identifier)==1
    np.testing.assert_array_equal(d.destination,s.slot_position(3))


def test_expired_charge_ticket_does_not_send_a_returning_drone_to_the_ground():
    s=service_fleet();d=s.drones[-1];s._clear_slot(d)
    s.queue=[s.drones[0].identifier]
    s._reserve_return(d)
    assert d.slot==3 and s.queue==[]
    np.testing.assert_array_equal(d.destination,s.slot_position(3))


@pytest.mark.parametrize('state',[DS.RETURN,DS.LANDING,DS.WAIT_LANDING])
def test_newly_free_slot_is_assigned_before_a_queued_aircraft_lands(state):
    s=service_fleet();d=s.drones[-1];park(s,d)
    d.position[2]=s.cruise(d);d.destination=s.wait_position(d.identifier)
    d.state=state
    s._assign_slots()
    assert d.slot==3 and s.queue==[] and d.state==DS.TO_SLOT
    np.testing.assert_array_equal(d.destination,s.slot_position(3))


def test_three_charging_aircraft_do_not_block_the_fourth_free_pad(tmp_path):
    s=service_fleet()
    for d in s.drones[:3]:
        old=d.battery;d.battery=d.config.battery*.3;d.charge_requested=True
        s.equipment_exported_energy+=old-d.battery
    d=s.drones[-1];park(s,d)
    s.run(4);assert d.slot==3 and not d.grounded
    path=tmp_path/'queue.atsstate';save_state(path,s);other=load_state(path).simulation
    for _ in range(70):s.step();other.step();balanced(s);balanced(other)
    assert d.docked and d.slot==3 and d.battery>d.config.battery*.3-s._transfer_energy_from_ground(d,3)
    assert s.report()==other.report()
    for a,b in zip(s.drones,other.drones):np.testing.assert_array_equal(a.position,b.position)


def test_insufficient_transfer_energy_is_explained_without_charging_on_the_ground():
    s=service_fleet();d=s.drones[-1];park(s,d,charge=.2001)
    before=d.battery;s.run(5)
    assert d.slot is None and s.slots[-1] is None and d.battery==before
    reason=s.logistics.idle_reason(d)
    assert 'перелёт' in reason and 'Вт·ч' in reason
    balanced(s)


def test_interrupted_descent_can_use_a_newly_free_pad_without_an_extra_ground_hop():
    s=service_fleet();d=s.drones[-1];park(s,d)
    d.position[2]=s.cruise(d);d.state=DS.WAIT_LANDING
    d.destination=s.wait_position(d.identifier)
    needed=s._transfer_energy(d,3)
    available=(needed+s._transfer_energy_from_ground(d,3))/2
    assert needed<available<s._transfer_energy_from_ground(d,3)
    old=d.battery;d.battery=d.config.battery*d.config.reserve+available
    s.equipment_exported_energy+=old-d.battery;s.logistics.sync_batteries()
    s.run(40)
    assert d.docked and d.slot==3 and s.state.value=='RUNNING'
    balanced(s)


def test_three_charging_aircraft_and_a_ground_waiter_eventually_form_the_lift_team():
    from ats.excavator import ExcavatorState as ES
    from ats.inventory import Location as L
    c=experiment_config(Config(),drone_spare_sets=0)
    for model in c.fleet.models:model.initial_charge=.95
    s=Simulation(c);w=s.logistics;e=w.excavator
    w.inventory.move('group_rig',L.WORKSITE_RACK,'worksite',e.position,'STORED_READY')
    e.destination=e.position+np.array([3.,-1.,0.]);e.state=ES.WAITING_RELOCATION
    for d in s.drones:d.charge_requested=True
    park(s,s.drones[-1],.95)
    for _ in range(2400):
        s.step()
        if w.group is not None:break
        assert s.state.value=='RUNNING',s.reason
    assert w.group is not None and len(w.group.participants)==4
    assert all(d.battery==d.config.battery for d in s.drones)
    assert all(d.docked for d in s.drones)
    balanced(s)


def test_a_queued_active_courier_keeps_its_cargo_route():
    s=service_fleet();d=s.drones[-1];park(s,d)
    d.state=DS.RETURN;s.logistics.active[d.identifier]=object()
    s._assign_slots()
    assert d.slot is None and d.state==DS.RETURN and s.queue==[d.identifier]


def test_descent_reassignment_accounts_for_the_climb_back_to_the_lane():
    s=service_fleet();d=s.drones[-1];park(s,d)
    before=s._transfer_energy(d,3);d.state=DS.WAIT_LANDING
    assert s._transfer_energy(d,3)==before


def test_returning_digger_reserves_the_real_hop_to_a_contact_charger():
    s=service_fleet(count=2,slots=1);d=s.drones[-1]
    s.landing_requested=False;d.charge_requested=False
    trip=s._choose_trip(d)
    assert trip is not None
    wait=s.wait_position(d.identifier)
    direct=s.planner.estimate(s.base,trip.dig,trip.dump,trip.altitude,trip.mass,
                              d.config,wait,None,s.terrain.minimum_surface)
    hop=max(s._transfer_energy_from_ground(d,i) for i in range(len(s.slots)))
    assert trip.energy>=direct+hop-1e-7
    assert trip.energy<=d.battery-d.config.battery*d.config.reserve


def test_courier_preflight_cannot_spend_its_future_contact_charging_hop():
    from ats.inventory import Location as L
    s=service_fleet();w=s.logistics;d=s.drones[-1]
    c=w.containers[0];c.soil_kg=s.terrain.scoop(16.,16.,30.)
    w.inventory.move(c.id,L.FILLED_QUEUE,'worksite',(16.,16.,0.),'FULL_READY')
    job=w.new_job('PICKUP_FULL_CONTAINER',c,c.position,(25.,25.,0.),3)
    energy,_=w.estimate(d,job)
    assert energy>=s._waiting_charge_reserve(d)
    # A nearly exhausted courier must remain docked rather than accepting a
    # cargo flight which would leave it unable to reach a contact charger.
    old=d.battery;d.battery=d.config.battery*d.config.reserve+energy-1.
    s.equipment_exported_energy+=old-d.battery
    d.attachment='CONTAINER_HOOK';s.landing_requested=False
    for other in s.drones[:3]:
        old=other.battery;other.battery=other.config.battery*.3
        s.equipment_exported_energy+=old-other.battery;other.charge_requested=True
    w.dispatch()
    assert job.status=='QUEUED' and d.identifier not in w.active and d.charge_requested
    w.sync_batteries();balanced(s)


def test_live_fleet_tooltip_explains_an_empty_but_unreachable_charge_pad(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host=MainWindow(service_fleet().config);host.timer.stop()
    try:
        s=host.sim;s.request_landing();d=s.drones[-1];park(s,d,charge=.2001)
        s.run(2);host.refresh(True)
        tooltip=host.fleet_table.item(3,1).toolTip()
        assert 'свободному доку' in tooltip and 'Вт·ч' in tooltip
        balanced(s)
    finally:
        host.close();app.processEvents()


def test_pending_dock_addition_does_not_allocate_not_yet_installed_pads():
    from copy import deepcopy
    s=service_fleet(count=2,slots=1);d=s.drones[-1];park(s,d)
    s._clear_slot(s.drones[0]);s.pending_equipment=deepcopy(s.config)
    s.pending_equipment.fleet.charge_slots=6
    s._assign_slots()
    assert len(s.slots)==1 and d.slot==0 and s.slots==[d.identifier]


def test_pending_dock_removal_cannot_allocate_a_disappearing_free_pad():
    from copy import deepcopy
    s=service_fleet();d=s.drones[-1];park(s,d)
    s.pending_equipment=deepcopy(s.config);s.pending_equipment.fleet.charge_slots=3
    s._assign_slots()
    assert d.slot is None and s.slots[-1] is None and s.queue==[d.identifier]


def test_old_snapshot_with_an_empty_but_orphaned_reservation_recovers(tmp_path):
    s=service_fleet();d=s.drones[-1];park(s,d)
    # Reconstruct the old, accepted save state after one of two reservations
    # was released on takeoff: reverse index present, no aircraft owns the pad.
    s.slots[3]=d.identifier
    path=tmp_path/'orphaned-pad.atsstate';save_state(path,s)
    other=load_state(path).simulation
    for _ in range(60):s.step();other.step();balanced(s);balanced(other)
    assert d.docked and d.slot==3 and s.slots.count(d.identifier)==1
    assert s.report()==other.report()


def test_orphan_cleanup_does_not_offer_a_pad_still_physically_occupied():
    s=service_fleet();d=s.drones[-1]
    d.slot=None;d.state=DS.WAIT_GROUND
    d.charge_requested=True;d.battery*=.3;s._enqueue(d)
    before=d.position.copy();s._assign_slots()
    assert s.slots[3]==d.identifier and d.slot is None
    np.testing.assert_array_equal(d.position,before)


def test_three_docked_charge_tickets_do_not_strand_a_low_soc_returner():
    c=experiment_config(Config(),drone_spare_sets=0);c.logistics.excavator_enabled=False
    for model in c.fleet.models[:3]:model.initial_charge=.3
    c.fleet.models[-1].initial_charge=.225
    s=Simulation(c);s.request_landing();d=s.drones[-1]
    # Follower/lift preparation can request charging from aircraft which
    # already own a dock. Those tickets must not divert the fourth to the ground.
    for other in s.drones[:3]:s._enqueue(other)
    s._clear_slot(d);d.position[:]=s.wait_position(d.identifier)
    d.position[2]=s.cruise(d);d.state=DS.RETURN;s._enqueue(d)
    s._reserve_return(d)
    assert d.slot==3
    for _ in range(60):s.step();balanced(s)
    assert s.slots==[0,1,2,3] and not s.queue and d.state==DS.CHARGING
