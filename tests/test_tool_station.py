"""Dock servicing is independent of external PV commissioning, with real tools."""
from copy import deepcopy
import pytest

from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.simulation import Simulation
from ats.drone import DroneState as DS
from ats.inventory import Location as L
from ats.checkpoint import save_state,load_state


def workshop(*,panels=20,drones=1,slots=None,soc=1.,digging=True):
    c=experiment_config(with_map_size(Config(),128),drones=drones,panel_area=panels,
                        landing_slots=drones if slots is None else slots,batch_courier=False)
    c.logistics.excavator_enabled=False;c.logistics.container_count=0
    c.logistics.direct_allowed=digging;c.logistics.idle_digging=digging
    for d in c.fleet.models:d.initial_charge=soc
    s=Simulation(c);s.landing_requested=True
    assert s.state.value=='RUNNING',s.reason
    return s


def balanced(s):
    report=s.report(full=False)
    assert abs(report['mass_error_kg'])<1e-7
    assert abs(report['energy_error_j'])<.002
    s.logistics.inventory.validate()


def test_hook_change_timer_continues_when_external_panels_are_added():
    s=workshop();w=s.logistics;d=s.drone
    w.request_tool(d,'CONTAINER_HOOK');s.run(10)
    assert d.state==DS.CHANGING_TOOL and w.tool_changes[d.serial]==10
    s.set_panel_area(48);ids=set(w.inventory.items);s.run(20)
    assert not w.field_ready and w.tool_changes[d.serial]==30
    s.run(15)
    assert not w.field_ready and d.attachment=='CONTAINER_HOOK'
    assert not w.tool_changes and not w.tool_targets
    assert w.inventory.items['hook_1'].location==L.DRONE_MOUNT
    assert w.inventory.items['grabber_1'].location==L.BASE_STORAGE
    assert set(w.inventory.items)==ids
    balanced(s)


def test_hook_can_be_fitted_before_field_commissioning_even_with_small_inverter():
    s=workshop(panels=48);w=s.logistics
    # Enough for the 100 W tool changer and idle controls, less than the
    # 200 W PV installer. Waiting for the whole field would be a deadlock.
    w.config.inverter_power_w=150.
    w.request_tool(s.drone,'CONTAINER_HOOK');s.run(45)
    assert s.drone.attachment=='CONTAINER_HOOK' and not w.field_ready
    assert w.station_consumed==pytest.approx(4500.)
    balanced(s)


def test_full_spares_are_swapped_while_external_panels_are_being_deployed():
    s=workshop(panels=48,soc=.2);w=s.logistics;d=s.drone
    old=list(d.battery_ids);ready=w.battery_service.ready(w.inventory.items[old[0]].compatibility)
    assert len(ready)==2 and all(p.soc==1 for p in ready)
    s.step()
    assert not w.field_ready and d.state==DS.SWAPPING_BATTERIES
    assert not w.battery_service.snapshot()['dock_charging_drones']
    assert w.battery_service.drone_swaps[d.serial]['elapsed']==1.
    s.run(60)
    assert w.battery_service.drone_swap_count==1 and d.battery_ids!=old
    assert all(w.inventory.items[id].location in (L.BASE_STORAGE,L.BASE_CHARGER) for id in old)
    assert all(w.inventory.items[id].location==L.DRONE_MOUNT for id in d.battery_ids)
    balanced(s)


def test_no_power_is_reported_without_advancing_the_tool_timer():
    c=deepcopy(workshop().config);c.start_hour=0.;c.base.initial_charge=0.
    s=Simulation(c);s.landing_requested=True;w=s.logistics
    w.request_tool(s.drone,'CONTAINER_HOOK');s.run(3)
    assert w.tool_changes.get(s.drone.serial,0)==0 and s.drone.attachment=='GRABBER'
    assert 'нет энергии' in w.idle_reason(s.drone)
    balanced(s)


def test_foreign_tool_reservation_is_preserved_and_explained_until_released():
    s=workshop(panels=48);w=s.logistics;d=s.drone
    w.inventory.reserve(['hook_1'],'other_job');w.request_tool(d,'CONTAINER_HOOK');s.run(3)
    assert d.attachment=='GRABBER' and not w.tool_changes
    assert w.inventory.items['hook_1'].reserved_by=='other_job'
    assert w.inventory.items['grabber_1'].reserved_by is None
    assert 'зарезервирован другим' in w.idle_reason(d)
    w.inventory.release('other_job');s.run(45)
    assert d.attachment=='CONTAINER_HOOK'
    balanced(s)


def test_hook_away_from_base_must_be_delivered_before_exchange():
    s=workshop();w=s.logistics;d=s.drone
    w.inventory.move('hook_1',L.GROUND,'worksite',(20.,20.,0.),'AWAITING_PICKUP')
    w.request_tool(d,'CONTAINER_HOOK');s.run(3)
    assert d.attachment=='GRABBER' and not w.tool_changes
    assert 'не доставлен' in w.idle_reason(d)
    assert w.inventory.items['hook_1'].location==L.GROUND
    balanced(s)


def test_four_drones_with_one_hook_do_not_create_or_borrow_missing_hardware():
    from ats.complexes import ComplexProject
    s=workshop(panels=48,drones=4);w=s.logistics;p=ComplexProject(s)
    # An explicit external withdrawal leaves one physical personal hook.
    for d in s.drones[1:]:
        item=w.inventory.items.pop(f'hook_{d.serial}');w.exported_mass_kg+=item.mass_kg
    ids=set(w.inventory.items)
    for d in s.drones:w.request_tool(d,'CONTAINER_HOOK')
    for _ in range(50):p.step()
    assert s.drones[0].attachment=='CONTAINER_HOOK'
    for d in s.drones[1:]:
        assert d.attachment=='GRABBER' and 'Нет инструмента' in w.idle_reason(d)
        assert d.serial not in w.tool_changes
    assert set(w.inventory.items)==ids
    assert p.report()['equipment_mass_error_kg']==pytest.approx(0,abs=1e-7)
    balanced(s)


def test_four_drones_share_one_service_pad_and_complete_both_exchange_directions():
    s=workshop(panels=48,drones=4,slots=1);w=s.logistics
    s.landing_requested=False;ids=set(w.inventory.items)
    hooks=set();returned=set()
    for d in s.drones:w.request_tool(d,'CONTAINER_HOOK')
    for _ in range(1500):
        s.step();balanced(s)
        for d in s.drones:
            if d.attachment=='CONTAINER_HOOK':hooks.add(d.serial)
            elif d.serial in hooks and d.attachment=='GRABBER':returned.add(d.serial)
        if len(returned)==4:break
    assert hooks==returned=={d.serial for d in s.drones}
    s.request_landing()
    for d in s.drones:w.reset_drone_order(d.serial)
    for _ in range(1000):
        if all(d.grounded for d in s.drones):break
        s.step();balanced(s)
    assert set(w.inventory.items)==ids and not w.tool_targets and not w.tool_changes
    assert not any(o.reserved_by for o in w.inventory.items.values())


def test_cancel_partial_hook_exchange_preserves_other_owners_and_hardware():
    s=workshop(panels=48);w=s.logistics;d=s.drone
    w.inventory.reserve(['base_chassis'],'other_job')
    w.request_tool(d,'CONTAINER_HOOK');s.run(10)
    assert w.tool_changes[d.serial]==10
    ids=set(w.inventory.items);mass=sum(o.mass_kg for o in w.inventory.items.values())
    w.reset_drone_order(d.serial)
    assert d.attachment=='GRABBER' and d.state==DS.DOCKED
    assert not w.tool_changes and not w.tool_targets
    assert w.inventory.items['base_chassis'].reserved_by=='other_job'
    assert all(w.inventory.items[id].reserved_by is None for id in ('hook_1','grabber_1'))
    assert set(w.inventory.items)==ids and sum(o.mass_kg for o in w.inventory.items.values())==mass
    balanced(s)


@pytest.mark.parametrize('service',['hook','battery'])
def test_exact_resume_during_dock_service_and_pv_installation(tmp_path,service):
    s=workshop(panels=48,soc=.2 if service=='battery' else 1.);w=s.logistics
    if service=='hook':w.request_tool(s.drone,'CONTAINER_HOOK')
    s.run(10)
    assert (w.tool_changes[s.drone.serial] if service=='hook' else w.battery_service.drone_swaps[s.drone.serial]['elapsed'])==10
    path=tmp_path/'dock-and-pv.atsstate';save_state(path,s);other=load_state(path).simulation
    assert other.report()==s.report()
    for _ in range(75):s.step();other.step()
    assert other.report()==s.report()
    balanced(s);balanced(other)


def test_started_swap_finishes_when_base_move_is_requested_without_starting_another():
    s=workshop(panels=48,drones=2,soc=.2);w=s.logistics
    # Start the first swap, then request a move before the second is requested.
    s.drones[1].charge_requested=False
    s.drones[1].battery=s.drones[1].config.battery
    # Keep this artificial control change explicitly in the energy ledger.
    s.equipment_imported_energy+=s.drones[1].config.battery*.8;w.sync_batteries()
    s.run(10);assert set(w.battery_service.drone_swaps)=={s.drones[0].serial}
    second=s.drones[1];removed=second.battery-second.config.battery*.2
    second.battery-=removed;s.equipment_exported_energy+=removed;w.sync_batteries()
    s.relocation_pending=s.base.position[:2]+[10.,0.]
    s.run(51)
    assert not w.battery_service.drone_swaps and w.battery_service.drone_swap_count==1
    assert second.charge_requested and second.serial not in w.battery_service.drone_swap_counts
    balanced(s)


def test_collective_lift_owns_its_participants_until_the_group_is_released():
    import numpy as np
    from ats.cooperative_lift import CollectiveLift
    from tests.test_logistics import logistics_config
    c=logistics_config(count=4);c.fleet.charge_slots=4
    s=Simulation(c);w=s.logistics;d=s.drone;e=w.excavator
    w.inventory.move('group_rig',L.WORKSITE_RACK,'worksite',e.position,'STORED_READY')
    w.group=CollectiveLift(w,e.position+np.array([3.,0.,0.]),list(s.drones))
    assert d.docked
    w.request_tool(d,'GRABBER');before=s.report()['energy_error_j'];w.tick_tool_changes()
    assert not w.tool_changes and d.attachment=='CONTAINER_HOOK'
    assert 'совместного переноса' in w.wait_reasons[d.identifier]
    assert w.inventory.items['group_rig'].reserved_by=='collective_lift'
    w.request_tool(d,d.attachment)
    assert w.inventory.items['group_rig'].reserved_by=='collective_lift'
    assert s.report()['energy_error_j']==before
    balanced(s)


def test_running_ui_shows_the_actual_reason_for_waiting_for_a_hook():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host=MainWindow(workshop(panels=48).config);host.timer.stop()
    s=host.sim;s.landing_requested=True;w=s.logistics
    w.inventory.reserve(['hook_1'],'other_job');w.request_tool(s.drone,'CONTAINER_HOOK')
    s.step();host.refresh(True)
    assert 'зарезервирован другим заданием' in host.fleet_table.item(0,1).toolTip()
    w.inventory.release('other_job');s.run(45);host.refresh(True)
    assert s.drone.attachment=='CONTAINER_HOOK'
    assert 'зарезервирован другим заданием' not in host.fleet_table.item(0,1).toolTip()
    host.close();app.processEvents();balanced(s)
