"""A cheaper material preview must preserve the actual loaded bucket time."""
import pytest

from tests.test_conventional import balanced, fuel_tanks, world


def test_bucket_rechecks_current_material_when_the_base_cycle_is_elapsed(monkeypatch):
    sim, fleet = world(trucks=1, soil=True, bucket_cycle_s=30.)
    fuel_tanks(fleet)
    excavator, truck = fleet.machines
    material = sim.terrain.soil.profile.materials[0]
    material.digging_factor = 2.
    for _ in range(200):
        sim.step()
        if truck.state == 'WAIT_LOAD':
            break
    assert truck.state == 'WAIT_LOAD' and excavator.timer == 0.
    preview = sim.terrain.preview_load
    checked = []

    def measured(*args, **kwargs):
        checked.append(excavator.timer)
        return preview(*args, **kwargs)

    monkeypatch.setattr(sim.terrain, 'preview_load', measured)
    for _ in range(34):
        sim.step()
    assert truck.cargo_kg == 0.
    assert checked == list(range(30, 35))

    # The current material, not a factor frozen at the beginning of the
    # bucket, determines whether it can finish this second.
    material.digging_factor = 1.
    sim.step()
    assert checked[-1] == 35
    assert excavator.timer == 0. and truck.cargo_kg > 0.
    assert fleet.excavated_mass_kg == pytest.approx(truck.cargo_kg)
    balanced(sim, fleet)
