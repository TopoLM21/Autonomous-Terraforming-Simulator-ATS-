"""Dawn, service priority and recoverable orders must preserve physical balances."""
from copy import deepcopy
import os
import numpy as np
import pytest
from ats.config import Config
from ats.drone import DroneState as DS
from ats.experiments import experiment_config
from ats.inventory import Location as L
from ats.simulation import Simulation
from tests.test_workfront import canal


def balanced(s):
    r=s.report(full=False)
    assert abs(r['mass_error_kg'])<1e-7
    assert abs(r['energy_error_j'])<.002
    s.logistics.inventory.validate()


def service_world(*, soc=0., spares=2, hour=12.):
    c=experiment_config(Config(),drones=1,drone_spare_sets=spares,batch_courier=False)
    c.start_hour=hour;c.base.latitude_deg=0.;c.base.start_day_of_year=80
    c.base.initial_charge=soc
    c.logistics.excavator_enabled=False;c.logistics.container_count=0
    c.logistics.direct_allowed=True;c.logistics.drone_spare_initial_soc=0.
    s=Simulation(c);s.request_landing()
    return s


def pending_move(s):
    site=next(p for p in s.terrain.front.settings['base_sites']
              if p[0]>20 and s.base.safe_path(p[1:],s.terrain))
    s.relocation_pending=np.asarray(site[1:]);s._pending_front_anchor=site[0]


def test_night_relocation_reserves_energy_for_panels_and_restarts_generation_at_dawn():
    c=deepcopy(canal(panels=100,slots=4).config)
    c.start_hour=5.95;c.base.start_day_of_year=80;c.base.initial_charge=.015
    c.logistics.excavator_enabled=False;c.logistics.container_count=0
    c.logistics.drone_spare_initial_soc=0.;c.logistics.swap_station=False
    for model in c.fleet.models:model.initial_charge=.2
    s=Simulation(c);w=s.logistics
    w.field_ready=True;w.field_elapsed=w.config.field_deployment_seconds
    pending_move(s)
    for _ in range(1200):s.step()
    assert s.base.relocations==1 and s.base.destination is None
    assert w.field_ready and w.base_pack_elapsed==0
    assert s.base.solar_power>0 and s.relocation_pending is None
    balanced(s)


def test_tool_change_uses_sun_before_spare_chargers_can_starve_it():
    s=service_world();w=s.logistics;d=s.drone
    w.request_tool(d,'CONTAINER_HOOK')
    for _ in range(47):s.step();balanced(s)
    assert d.attachment=='CONTAINER_HOOK'
    assert not w.tool_targets and not w.tool_changes
    assert all(o.reserved_by!=f'tool_{d.serial}' for o in w.inventory.items.values())
    assert w.station_consumed==pytest.approx(100*w.config.attachment_seconds)


def test_deployed_panels_recharge_an_empty_base_after_night():
    s=service_world(spares=0,hour=5.95)
    s.run(180)
    assert s.base.battery==0 and s.base.solar_power==0
    s.run(600)
    assert s.base.battery>0 and s.base.solar_power>0
    assert s.state.value=='RUNNING'
    balanced(s)


@pytest.mark.parametrize('spares',[0,1])
def test_strict_logistics_waits_for_dawn_without_a_spare_module_charge_queue(spares):
    c=experiment_config(Config(),drones=1,idle_digging=False,drone_spare_sets=spares)
    c.start_hour=5.95;c.base.start_day_of_year=80;c.base.initial_charge=0.
    c.fleet.models[0].initial_charge=.2
    c.logistics.swap_station=False;c.logistics.stall_timeout_s=60.
    s=Simulation(c);before=s.drone.battery
    s.run(179)
    assert s.state.value=='RUNNING' and s.base.solar_power==0
    assert s.drone.charge_requested and not s.logistics.battery_service.queue
    assert s.logistics.wait_reason=='WAITING_SUN_OR_CHARGER: конечная зарядная очередь'
    s.run(600)
    assert s.state.value=='RUNNING' and s.base.solar_power>0 and s.drone.battery>before
    balanced(s)


def test_hot_added_panels_keep_the_existing_live_array_and_can_deploy_with_empty_storage():
    s=service_world(spares=0);w=s.logistics
    s.set_panel_area(100.)
    assert not w.field_ready and w.generating_area()==pytest.approx(20.)
    s.run(301)
    assert w.field_ready and s.base.battery>0 and w.generating_area()==100.
    balanced(s)


def test_partly_folded_field_recovers_at_dawn_even_if_storage_is_already_empty():
    c=deepcopy(canal(panels=100,slots=4).config)
    c.start_hour=5.95;c.base.start_day_of_year=80;c.base.initial_charge=0.
    c.logistics.excavator_enabled=False;c.logistics.container_count=0
    s=Simulation(c);w=s.logistics
    w.field_ready=True;w.base_pack_elapsed=67.
    pending_move(s)
    s.run(3600)
    assert s.base.relocations==1 and s.relocation_pending is None
    assert w.field_ready and s.base.solar_power>0
    balanced(s)


def test_reset_ground_tool_work_preserves_hardware_charge_and_other_owners():
    s=service_world(soc=.5,spares=0);w=s.logistics;d=s.drone
    w.request_tool(d,'CONTAINER_HOOK');s.run(10)
    assert d.state==DS.CHANGING_TOOL and w.tool_changes[d.serial]>0
    w.inventory.reserve(['base_chassis'],'other_job')
    before=s.report();items=dict(w.inventory.items);height=s.terrain.height.copy()
    mounted=(w.inventory.items[f'grabber_{d.serial}'].location,w.inventory.items[f'hook_{d.serial}'].location)
    w.reset_drone_order(d.serial)
    assert d.state==DS.DOCKED and d.attachment=='GRABBER'
    assert not w.tool_targets and not w.tool_changes and not w.recall_ids
    assert w.inventory.items['base_chassis'].reserved_by=='other_job'
    assert all(o.reserved_by!=f'tool_{d.serial}' for o in items.values())
    assert all(w.inventory.items[id] is item for id,item in items.items())
    assert mounted==(w.inventory.items[f'grabber_{d.serial}'].location,w.inventory.items[f'hook_{d.serial}'].location)
    after=s.report()
    for key in ('time_seconds','base_kwh','drone_kwh','consumed_kwh','initial_energy_kwh'):
        assert after[key]==before[key]
    np.testing.assert_array_equal(s.terrain.height,height);balanced(s)


def test_reset_an_unfinished_battery_swap_releases_both_real_sets_without_swapping():
    c=experiment_config(Config(),drones=1)
    c.fleet.models[0].initial_charge=.2
    s=Simulation(c);s.request_landing();w=s.logistics;d=s.drone
    old=list(d.battery_ids);s.run(10)
    swap=w.battery_service.drone_swaps[d.serial]
    assert 0<swap['elapsed']<w.config.drone_swap_seconds
    incoming=list(swap['ids']);before=s.report()
    w.reset_drone_order(d.serial)
    assert d.battery_ids==old and not w.battery_service.drone_swaps
    assert d.state==DS.CHARGING and d.charge_requested
    assert all(w.inventory.items[id].location==L.BASE_STORAGE for id in incoming)
    assert all(w.inventory.items[id].location==L.DRONE_MOUNT for id in old)
    assert all(w.inventory.items[id].reserved_by is None for id in incoming+old)
    assert s.report()['energy_error_j']==before['energy_error_j']
    assert w.battery_service.drone_swap_count==0
    balanced(s)


def flight_world():
    c=Config();c.drone.hover_power=1000.
    c=experiment_config(c,drones=1)
    c.logistics.excavator_enabled=False;c.logistics.container_count=0;c.logistics.direct_allowed=True
    return Simulation(c)


def test_recall_during_takeoff_does_not_leave_a_duplicate_slot_reservation():
    s=flight_world();w=s.logistics;d=s.drone
    s.step();assert d.state==DS.TAKEOFF and d.slot==0
    job=w.jobs[-1];w.reset_drone_order(d.serial)
    for _ in range(100):
        s.step();balanced(s)
        if not w.recall_ids:break
    assert d.grounded and d.cargo==0 and not w.recall_ids
    assert job.status=='CANCELLED' and d.identifier not in w.direct_ids
    assert s.slots.count(d.identifier)==1 and s.slots[d.slot]==d.identifier
    assert all(o.reserved_by!=job.id for o in w.inventory.items.values())


def test_recall_loaded_digger_delivers_soil_before_releasing_its_task():
    s=flight_world();w=s.logistics;d=s.drone
    for _ in range(500):
        s.step()
        if d.state==DS.TO_DUMP and d.cargo>0:break
    assert d.cargo>0 and d.identifier in w.direct_ids
    job=w.jobs[-1];cargo=d.cargo;w.reset_drone_order(d.serial)
    assert d.cargo==cargo and d.identifier in w.recall_ids
    for _ in range(500):
        s.step();balanced(s)
        if not w.recall_ids:break
    assert d.grounded and d.cargo==0 and not w.recall_ids
    assert d.trips>=1 and job.status=='CANCELLED'
    assert all(o.reserved_by!=job.id for o in w.inventory.items.values())


def test_recall_transport_keeps_the_container_route_and_releases_it_after_delivery():
    c=experiment_config(Config(),drones=1,containers=1,batch_courier=False,idle_digging=False)
    c.logistics.excavator_enabled=False
    s=Simulation(c);s.request_landing();w=s.logistics;d=s.drone
    container=w.containers[0]
    container.soil_kg=s.terrain.scoop(16.,16.,30.)
    assert container.soil_kg>0
    w.inventory.move(container.id,L.FILLED_QUEUE,'worksite',(16.,16.,0.),'FULL_READY')
    for _ in range(300):
        s.step()
        if d.identifier in w.active and w.active[d.identifier].phase=='DESTINATION':break
    job=w.active[d.identifier];route=(job.origin,job.destination);cargo=d.cargo
    assert container.location==L.DRONE_CARGO and cargo>container.soil_kg
    w.reset_drone_order(d.serial)
    assert route==(job.origin,job.destination) and d.cargo==cargo
    assert container.reserved_by==job.id and job.status=='ACTIVE'
    for _ in range(500):
        s.step();balanced(s)
        if not w.recall_ids:break
    assert d.grounded and d.cargo==0 and not w.recall_ids
    assert job.status=='COMPLETED' and container.soil_kg==0 and container.reserved_by is None
    assert not w.active and w.delivered_mass>0


def test_pending_move_clears_tool_changes_and_finishes_only_existing_battery_swaps():
    c=deepcopy(canal(panels=21,slots=4).config)
    c.logistics.excavator_enabled=False;c.logistics.container_count=0;c.logistics.drone_swap_seconds=3.
    for d in c.fleet.models[2:]:d.initial_charge=.2
    s=Simulation(c);w=s.logistics;w.field_ready=True;w.field_elapsed=w.config.field_deployment_seconds
    for d in s.drones[:2]:w.request_tool(d,'GRABBER')
    s.run(2)
    assert all(d.grounded for d in s.drones) and len(w.battery_service.drone_swaps)==2
    assert all(d.state==DS.CHANGING_TOOL for d in s.drones[:2])
    pending_move(s);s.step()
    assert s.base_move_wait=='WAITING_SERVICE_COMPLETION'
    assert 'смену батарей' in s.base_move_status
    assert not w.tool_targets and not w.tool_changes and not w.battery_service.drone_swaps
    assert all(not (o.reserved_by or '').startswith('tool_') for o in w.inventory.items.values())
    assert w.battery_service.drone_swap_count==2
    for _ in range(1000):
        s.step()
        if s.base.relocations and s.base.destination is None and w.field_ready:break
    assert s.base.relocations==1 and w.field_ready
    balanced(s)


def test_recall_of_a_group_member_keeps_the_load_supported_until_it_is_set_down():
    from ats.excavator import ExcavatorState as ES
    from tests.test_logistics import logistics_config
    c=logistics_config(count=4);c.fleet.charge_slots=4
    s=Simulation(c);w=s.logistics;e=w.excavator
    e.destination=e.position+np.array([3.,-1.,0.]);e.state=ES.WAITING_RELOCATION
    for _ in range(700):
        s.step()
        if w.group is not None and w.group.phase=='LIFT':break
    assert w.group is not None and w.group.phase=='LIFT'
    group=w.group;d=s.drones[-1];cargo=d.cargo;load=d.equivalent_load
    assert cargo>0 and load>0
    w.reset_drone_order(d.serial)
    assert w.group is group and d.cargo==cargo and d.equivalent_load==load
    assert w.inventory.items['group_rig'].reserved_by=='collective_lift'
    for _ in range(700):
        s.step();balanced(s)
        if not w.recall_ids:break
    assert not w.recall_ids and d.grounded and d.cargo==0 and d.equivalent_load is None
    assert w.lift_count==1 and w.group is None


def test_cancelling_partial_folding_preserves_deployed_area_and_costs_real_unfolding_energy():
    s=canal(panels=100,slots=4);w=s.logistics
    w.field_ready=True;w.field_elapsed=w.config.field_deployment_seconds
    pending_move(s);s.run(20)
    assert w.base_pack_elapsed==20
    area=w.generating_area();before=w.station_consumed
    assert s.cancel_relocation()
    assert s.relocation_pending is None and not w.field_ready and w.base_pack_elapsed==0
    assert w.generating_area()==pytest.approx(area)
    s.request_landing();s.run(21)
    assert w.field_ready and s.base.relocations==0
    assert w.station_consumed-before>=20*200.
    balanced(s)


def test_panel_mechanics_including_the_final_folding_second_are_accounted_before_driving():
    c=deepcopy(canal(panels=21,slots=4).config)
    c.start_hour=0.;c.base.idle_power=30.
    c.logistics.excavator_enabled=False;c.logistics.container_count=0
    s=Simulation(c);w=s.logistics
    w.field_ready=True;w.field_elapsed=w.config.field_deployment_seconds
    pending_move(s)
    drive=s.base.route_energy(s.relocation_pending,s.terrain,s.drones)
    for _ in range(1000):
        s.step()
        if s.base.relocations and s.base.destination is None and w.field_ready:break
    assert w.field_ready and s.base.relocations==1
    assert w.station_consumed==2*200*w.config.field_deployment_seconds
    assert s.base.consumed==pytest.approx(drive+2*(200+30)*w.config.field_deployment_seconds,abs=.002)
    balanced(s)


def test_real_dispatcher_buttons_reset_dock_work_and_cancel_only_the_pending_move():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.logistics_ui import LogisticsWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host=MainWindow(service_world(soc=.5,spares=0).config);host.timer.stop()
    s=host.sim;s.request_landing();w=s.logistics
    w.request_tool(s.drone,'CONTAINER_HOOK');s.run(5)
    assert s.drone.state==DS.CHANGING_TOOL
    view=LogisticsWindow(host);view.timer.stop();view.tabs.setCurrentIndex(2)
    time=s.time;view.reset_order_button.click()
    assert host.sim is s and s.time==time and not w.tool_targets
    s.relocation_pending=s.base.position[:2]+[1.,0.];view.refresh()
    assert view.cancel_move_button.isEnabled()
    view.cancel_move_button.click()
    assert s.relocation_pending is None and s.time==time and host.sim is s
    assert 'Поток в батарею базы' in host.telemetry.text()
    view.close();host.close();app.processEvents();balanced(s)
