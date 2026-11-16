"""The shared work reservation never replaces physical equipment clearance."""
import numpy as np
from ats.power_design import equipment_copy
from ats.mixed_setup import adopt_first
from tests.test_mixed_setup import running_site,template
from tests.test_conventional_shared import mixed,balanced


def test_zero_autonomous_hardware_copy_preserves_the_empty_workspace():
    source,_=running_site()
    copied=equipment_copy(source.config)
    assert copied.conventional_only and copied.fleet.count==0 and copied.fleet.charge_slots==0
    assert copied.base is not source.config.base and copied.fleet is not source.config.fleet


def test_ground_first_delivery_then_same_cell_growth_keeps_the_live_loaded_fleet():
    source,fleet=running_site(soil=True)
    project=adopt_first(source,template(),position=(110.,20.)).complex_project
    terrain=project.shared_site.core;heights=terrain.height.copy();cell=terrain.cell
    cargo=fleet.carried_mass_kg;clock=project.time
    project.shared_site.enlarge(192.)
    assert fleet.terrain is project.shared_site.core and fleet.terrain.cell==cell
    np.testing.assert_array_equal(fleet.terrain.height[:heights.shape[0],:heights.shape[1]],heights)
    assert fleet.carried_mass_kg==cargo and project.time==clock
    project.step();balanced(project)


def test_autonomous_grab_cannot_cut_under_camp_or_vehicle_outside_work_reservation():
    project,_,fleet=mixed();terrain=project.shared_site.core
    for point in (fleet.base_position,fleet.machines[1].position):
        x,y=point[:2];patch=terrain.patch(x,y,2.)
        assert not np.any(terrain._conventional_mask[patch])
        # Also protects legacy/custom plans whose future cut crosses a body.
        terrain.dig_mask[patch]=True;terrain.target[patch]=terrain.height[patch]-1.
        before=terrain.height.copy()
        assert terrain.available_mass(x,y,radius=1.)==0.
        assert terrain.scoop(x,y,500.,radius=1.)==0.
        np.testing.assert_array_equal(terrain.height,before)
    truck=fleet.machines[1];home=truck.home.copy()
    fleet.fuel_stock_l-=20.;truck.fuel_l+=20.
    assert fleet._assign_route(truck,home[:2]+[8.,0.],'PARKED')
    for _ in range(10):
        if not truck.path:break
        fleet._move(truck)
    assert np.linalg.norm(truck.position[:2]-home[:2])>truck.radius+1.
    # Its empty return bay must survive too; it is ground, not an air obstacle.
    assert terrain.available_mass(*home[:2],radius=.5)==0.
    assert terrain.scoop(*home[:2],500.,radius=.5)==0.
