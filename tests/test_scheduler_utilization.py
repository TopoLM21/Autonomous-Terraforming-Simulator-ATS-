"""Resource contention must not serialize independent earthmoving work."""
from copy import copy

import numpy as np
import pytest

from ats.config import Config
from ats.cooperative_lift import lift_check
from ats.drone import DroneState as DS
from ats.excavator import ExcavatorState as ES
from ats.experiments import experiment_config
from ats.inventory import Location as L
from ats.scenarios import scenario
from ats.simulation import Simulation


def _initial_hook(sim, drone):
    """Choose a coherent initial attachment, moving both finite inventory items."""
    world = sim.logistics
    world.inventory.move(f"grabber_{drone.serial}", L.BASE_STORAGE, "base",
                         sim.base.position, "STORED_READY")
    hook = world.inventory.items[f"hook_{drone.serial}"]
    world.inventory.move(hook.id, L.DRONE_MOUNT, f"drone_{drone.serial}",
                         drone.position, "INSTALLED")
    drone.attachment = "CONTAINER_HOOK"
    world._tool_mass(drone, hook.mass_kg)


def _assert_balanced(sim):
    report = sim.report(full=False)
    assert abs(report["mass_error_kg"]) < 1e-7
    assert abs(report["energy_error_j"]) < 1e-4
    assert not sim.logistics.direct_ids.intersection(sim.logistics.active)
    sim.logistics.inventory.validate()


def test_charging_carrier_does_not_serialize_ready_diggers_or_cargo():
    config = experiment_config(scenario("medium"), mode="hybrid", drones=4)
    config.logistics.excavator_enabled = False
    config.logistics.container_count = 1
    config.logistics.attachment_seconds = 3.
    config.logistics.drone_spare_sets = 0
    config.logistics.swap_station = False
    config.logistics.bus_power_w = 200.
    config.fleet.models[-1].initial_charge = .2
    sim = Simulation(config)
    world = sim.logistics
    original_carrier = sim.drones[-1]
    _initial_hook(sim, original_carrier)
    initial_ids = set(world.inventory.items)
    container = world.containers[0]
    # This is real terrain mass in a real container, not a synthetic payload.
    container.soil_kg = sim.terrain.scoop(16., 16., 40.)
    assert container.soil_kg > 0
    world.inventory.move(container.id, L.FILLED_QUEUE, "worksite",
                         (16., 16., sim.terrain.sample(16., 16.)), "FULL_READY")
    world.generate_jobs()
    job = next(j for j in world.jobs if j.kind == "PICKUP_FULL_CONTAINER")
    saw_tool_change = False
    independent_diggers = set()
    for _ in range(500):
        sim.step()
        saw_tool_change |= any(d.state == DS.CHANGING_TOOL for d in sim.drones[:-1])
        independent_diggers.update(world.direct_ids)
        assert original_carrier.charge_requested
        assert original_carrier.grounded
        assert original_carrier.identifier not in world.active
        assert set(world.inventory.items) == initial_ids
        _assert_balanced(sim)
        if job.status == "COMPLETED":
            break
    assert job.status == "COMPLETED"
    assert job.executor != original_carrier.serial
    assert saw_tool_change
    assert len(independent_diggers) >= 2
    assert original_carrier.battery < original_carrier.config.battery
    assert world.delivered_mass > 0


@pytest.mark.parametrize("initial_attachment", ["GRABBER", "CONTAINER_HOOK"])
def test_insufficient_next_dig_budget_requests_full_charge(initial_attachment):
    initial = Config()
    # A high work altitude makes even the smallest safe round trip exceed
    # the available .101 bank while SOC remains above the .30 threshold.
    initial.drone.work_height = 35.
    initial.drone.initial_charge = .301
    config = experiment_config(initial, drones=1)
    config.logistics.direct_allowed = True
    config.logistics.excavator_enabled = False
    config.logistics.container_count = 0
    config.logistics.idle_dig_delay_s = 1.
    config.logistics.attachment_seconds = 3.
    config.logistics.drone_swap_seconds = 3.
    sim = Simulation(config)
    world, drone = sim.logistics, sim.drone
    if initial_attachment == "CONTAINER_HOOK":
        _initial_hook(sim, drone)
    preview = copy(drone)
    preview.config = copy(drone.config)
    world._tool_mass(preview, world.inventory.items[f"grabber_{drone.serial}"].mass_kg)
    assert drone.battery > drone.config.battery * drone.config.charge_threshold
    assert sim._choose_trip(preview) is None
    assert sim._choose_trip(preview, battery=preview.config.battery) is not None
    initial_battery_ids = list(drone.battery_ids)
    # An initial hook may first finish its finite idle delay.
    for _ in range(3):
        sim.step()
        _assert_balanced(sim)
        if drone.charge_requested:
            break
    assert drone.charge_requested
    assert drone.grounded
    assert drone.identifier not in world.direct_ids
    for _ in range(30):
        consumed_before = drone.consumed
        sim.step()
        _assert_balanced(sim)
        if drone.charge_requested:
            assert drone.grounded
            assert drone.identifier not in world.direct_ids
        if drone.identifier in world.direct_ids:
            # The newly installed bank was exactly full before this second's
            # takeoff motor draw; no launch occurs at a merely sufficient SOC.
            assert drone.battery + drone.consumed - consumed_before == pytest.approx(drone.config.battery, rel=0, abs=1e-5)
            break
    assert drone.identifier in world.direct_ids
    assert not drone.charge_requested
    assert drone.battery_ids != initial_battery_ids
    assert world.battery_service.drone_swap_count == 1


def _relocation_world(last_soc):
    config = experiment_config(scenario("medium"), drones=4)
    config.logistics.container_count = 0
    config.logistics.drone_swap_seconds = 3.
    for model in config.fleet.models:
        model.initial_charge = .8
    config.fleet.models[-1].initial_charge = last_soc
    sim = Simulation(config)
    world, excavator = sim.logistics, sim.logistics.excavator
    excavator.state = ES.WAITING_RELOCATION
    excavator.destination = excavator.position + np.array([1., 0., 0.])
    excavator.destination[2] = sim.terrain.sample(*excavator.destination[:2])
    world.inventory.move("group_rig", L.WORKSITE_RACK, "excavator",
                         excavator.position, "STORED_READY")
    return sim


def test_collective_lift_only_requests_energy_from_undercharged_participant():
    sim = _relocation_world(.2)
    world = sim.logistics
    world.try_lift()
    assert world.group is None
    assert [d.charge_requested for d in sim.drones] == [False, False, False, True]
    for _ in range(3):
        sim.step()
        _assert_balanced(sim)
        if sim.drones[-1].charge_requested:
            assert world.group is None
            assert all(d.grounded for d in sim.drones)
    assert world.group is not None
    assert not sim.drones[-1].charge_requested
    assert sim.drones[-1].battery == pytest.approx(sim.drones[-1].config.battery, rel=0, abs=1e-5)
    assert all(not d.charge_requested for d in sim.drones[:-1])


def test_collective_lift_honors_existing_request_until_bank_is_full():
    sim = _relocation_world(.8)
    world = sim.logistics
    drone = sim.drones[-1]
    preview = copy(world.excavator)
    preview.battery_ids = []
    assert lift_check(preview, world.inventory, sim.drones, world.config, 1.)["feasible"]
    sim._request_charge(drone, "finish an already requested full charging cycle")
    world.try_lift()
    assert world.group is None
    for _ in range(3):
        sim.step()
        _assert_balanced(sim)
        if drone.charge_requested:
            assert world.group is None
            assert all(d.grounded for d in sim.drones)
    assert world.group is not None
    assert not drone.charge_requested
    assert drone.battery == pytest.approx(drone.config.battery, rel=0, abs=1e-5)


def test_foreign_grabber_reservation_prevents_direct_launch():
    config=experiment_config(Config(),drones=1)
    config.logistics.excavator_enabled=False
    config.logistics.direct_allowed=True
    config.logistics.container_count=0
    sim=Simulation(config);world=sim.logistics;drone=sim.drone
    grabber=world.inventory.items[f'grabber_{drone.serial}']
    assert world.inventory.reserve([grabber.id],'other_job')
    assert sim._choose_trip(drone) is not None
    world.direct_dispatch()
    assert drone.grounded and drone.trip is None
    assert not world.direct_ids and not world.jobs
    assert grabber.reserved_by=='other_job'
    assert 'занят' in world.idle_reason(drone)
    _assert_balanced(sim)
    world.inventory.release('other_job');world.direct_dispatch()
    assert drone.identifier in world.direct_ids and not drone.grounded
