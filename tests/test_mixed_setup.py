"""First autonomous delivery joins the live diesel site without a new world."""
from copy import deepcopy
import numpy as np
import pytest
from ats.config import Config
from ats.maps import with_map_size
from ats.experiments import experiment_config
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.complexes import ComplexTemplate
from ats.conventional import ConventionalConfig
from ats.checkpoint import save_state,load_state
from ats.mixed_setup import adopt_first
from tests.test_conventional_workspace import workspace_config


def running_site(soil=False,extended=True):
    c=workspace_config()
    if extended:c.plan=PlanConfig([PlanSegment([[40.,50.],[118.,50.]],width=8.,height=-.3)])
    if soil:
        from ats.soil import SoilProfile
        c.soil_profile=SoilProfile.qattara()
    s=Simulation.conventional_only(c)
    fleet=s.install_conventional(ConventionalConfig(base_x=12.,base_y=12.,work_rect=(35.,40.,60.,20.),
        truck_count=2,refuel_l_min=3000.))
    for _ in range(1800):
        s.step()
        if s.time>=750 and fleet.carried_mass_kg>200:break
    assert s.time>=750 and fleet.carried_mass_kg>200
    return s,fleet


def template(robots=0):
    c=experiment_config(with_map_size(Config(),128),drones=2,containers=0)
    c.logistics.excavator_enabled=bool(robots);c.logistics.excavator_count=max(1,robots)
    if robots:c.logistics.container_count=8*robots
    return ComplexTemplate.capture(c,'Копатели на общей карте')


@pytest.mark.parametrize('soil',[False,True])
def test_first_delivery_preserves_live_cargo_history_ground_and_exact_resume(tmp_path,soil):
    old,fleet=running_site(soil);core=old.terrain;clock=old.time
    before=deepcopy(fleet.report());heights=core.height.copy();ground_soil=core.soil
    machines=tuple(fleet.machines);positions=[m.position.copy() for m in machines]
    recipe=template();recipe_data=deepcopy(recipe.to_data())
    session=adopt_first(old,recipe,position=(110.,20.));s=session.simulation;p=session.complex_project
    assert len(p.units)==1 and len(s.drones)==2 and not s.config.conventional_only
    assert p.time==s.time==clock and p.units[0].born_at==0.
    assert s.config.start_hour==old.config.start_hour and s.config.base.start_day_of_year==old.config.base.start_day_of_year
    assert s.terrain.core is core and core.height is old.terrain.height and core.soil is ground_soil
    np.testing.assert_array_equal(core.height,heights)
    assert s.conventional is fleet and old.conventional is None and fleet.sim is s and fleet.terrain is core
    assert tuple(fleet.machines)==machines and fleet.report()==before
    for machine,position in zip(machines,positions):np.testing.assert_array_equal(machine.position,position)
    assert recipe.to_data()==recipe_data
    assert all(d.created_at==clock and d.activity.report()['total_seconds']==0 for d in s.drones)
    assert s.excavated_mass==0. and s.base.consumed==0. and s.base.generated==0.
    assert p.initial_energy_kwh==0. and p.delivered_energy_kwh>0.
    assert p.initial_equipment_mass==0. and p.delivered_equipment_mass>0.
    assert any(timestamp==clock and 'поставлен' in message for _,timestamp,message in s.events)
    p.validate();r=p.report()
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002
    state=tmp_path/'mixed.atsstate';save_state(state,s,complex_project=p)
    other=load_state(state).complex_project
    assert other.report()==p.report()
    for _ in range(25):p.step();other.step()
    assert other.report()==p.report() and p.time==clock+25
    assert other.units[0].simulation.conventional.sim is other.units[0].simulation
    assert all(d.activity.report()['total_seconds']==25 for d in s.drones)
    assert fleet.payroll_usd>before['payroll_usd']
    assert abs(p.report()['mass_error_kg'])<1e-7
    if soil:assert p.report()['shared_worksite']['max_material_mass_error_kg']<1e-7


def test_failed_late_preflight_restores_the_existing_owner_fleet_and_caches(monkeypatch):
    import ats.mixed_setup as module
    s,fleet=running_site();core=s.terrain;owner=core._site_owner;report=deepcopy(s.report())
    core._mixed_test_cache={'value':[1.,2.,3.]}
    height=core.height;soil=core.soil;mask=core._conventional_mask
    def reject(project,live_fleet):
        assert live_fleet is fleet and fleet.sim is project.active.simulation
        assert project.shared_site.core is core
        core._mixed_test_cache['value'].append(99.)
        raise ValueError('Контрольная ошибка после полного размещения')
    monkeypatch.setattr(module,'_check_placement',reject)
    with pytest.raises(ValueError,match='Контрольная ошибка'):
        adopt_first(s,template(),position=(110.,20.))
    assert s.conventional is fleet and fleet.sim is s and core._site_owner is owner
    assert core.height is height and core.soil is soil and core._conventional_mask is mask
    assert core._mixed_test_cache=={'value':[1.,2.,3.]}
    assert s.report()==report and s.shared_site is None


def test_manual_collision_rejects_purchase_without_teleport_or_clock_reset():
    s,fleet=running_site();before=deepcopy(s.report());owner=s.terrain._site_owner
    with pytest.raises(ValueError):adopt_first(s,template(),position=tuple(fleet.base_position[:2]))
    assert s.report()==before and s.conventional is fleet and fleet.sim is s
    assert s.terrain._site_owner is owner


def test_automatic_placement_keeps_the_existing_map_and_is_outside_reserved_plot():
    s,fleet=running_site();t=s.terrain;session=adopt_first(s,template())
    p=session.complex_project;sim=session.simulation
    assert p.shared_site.core is t and fleet.terrain is t and fleet.sim is sim
    assert len(p.units)==1 and p.next_id==2 and p.time==s.time
    assert sim.terrain.remaining>sim.config.terrain.completion_volume
    p.validate()


def test_no_work_outside_the_conventional_plot_is_a_clear_non_destructive_error():
    s,fleet=running_site(extended=False);before=deepcopy(s.report());owner=s.terrain._site_owner
    with pytest.raises(ValueError,match='нет работы для дронов'):
        adopt_first(s,template(),position=(110.,20.))
    assert s.report()==before and s.conventional is fleet and fleet.sim is s and s.terrain._site_owner is owner


def test_new_krot_and_racks_are_delivered_beside_the_new_base_not_the_old_camp():
    s,fleet=running_site();session=adopt_first(s,template(robots=1),position=(110.,20.))
    sim=session.simulation;p=session.complex_project;e=sim.logistics.excavator
    assert np.linalg.norm(e.position[:2]-sim.base.position[:2])<10.
    assert np.linalg.norm(e.position[:2]-fleet.base_position[:2])>50.
    assert sim.logistics.inventory.items[e.battery_ids[0]].position==tuple(e.position)
    assert s.conventional is None and sim.conventional is fleet
    p.validate()
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002


def test_oversized_pv_is_rejected_before_any_simulation_constructor(monkeypatch):
    import ats.simulation as module
    s,fleet=running_site();before=deepcopy(s.report());owner=s.terrain._site_owner
    terrain=s.terrain;height=terrain.height;positions=[m.position.copy() for m in fleet.machines]
    c=experiment_config(with_map_size(Config(),128),drones=2,containers=0,panel_area=30000.)
    c.logistics.excavator_enabled=False;c.base.layout={}
    recipe=ComplexTemplate.capture(c,'Слишком большое солнечное поле');recipe_data=deepcopy(recipe.to_data())
    def forbidden(*args,**kwargs):raise AssertionError('Геометрически невозможный комплект не должен создавать Simulation')
    monkeypatch.setattr(module,'Simulation',forbidden)
    with pytest.raises(ValueError,match='Габариты развёрнутого комплекса'):
        adopt_first(s,recipe)
    assert s.report()==before and s.conventional is fleet and fleet.sim is s
    assert s.terrain is terrain and terrain.height is height and terrain._site_owner is owner
    assert recipe.to_data()==recipe_data
    for machine,position in zip(fleet.machines,positions):np.testing.assert_array_equal(machine.position,position)
