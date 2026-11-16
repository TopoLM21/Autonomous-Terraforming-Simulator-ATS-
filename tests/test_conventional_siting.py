"""Real channel allocation, full machine clearance and atomic dry preflight."""
import base64
from copy import deepcopy
from dataclasses import asdict
import math
import numpy as np
import pytest
from ats.config import Config
from ats.maps import with_map_size
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation
from ats.conventional import ConventionalConfig
from ats.conventional_siting import prepare


def channel_site(angle=0.,depth=.5,side=128.,marine_west=0.):
    """Small real heightfield/target with optional connected western sea mask."""
    c=with_map_size(Config(),side);c.terrain.cell_size=.5;c.terrain.n=round(side/.5)
    c.terrain.floor=-100.;c.terrain.dump_rect=(5.,side-40.,25.,35.)
    center=np.array([side/2,side/2]);direction=np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
    a=center-direction*(side/2-20);b=center+direction*(side/2-18)
    c.plan=PlanConfig([PlanSegment([a.tolist(),b.tolist()],width=12.,height=-depth)],
        channel=dict(start=a.tolist(),tangent=direction.tolist(),cell_size=c.terrain.cell_size))
    if marine_west:
        c.terrain.dump_rect=(side-35.,side-40.,30.,35.)
        n=c.terrain.n;axis=(np.arange(n)+.5)*c.terrain.cell_size
        h=np.broadcast_to(np.where(axis<marine_west,-.02,.02),(n,n)).copy()
        c.initial_height=h;c.reference_height=h.copy()
        c.local_reference=True
        nodes=np.broadcast_to(np.arange(n+1)*c.terrain.cell_size<marine_west,(n+1,n+1))
        c.plan.channel['marine']=dict(shape=[n+1,n+1],origin=[0.,0.],spacing=[c.terrain.cell_size]*2,
            bits=base64.b64encode(np.packbits(nodes.ravel()).tobytes()).decode())
    c.start_hour=6.
    sim=Simulation.conventional_only(c)
    config=ConventionalConfig.for_simulation(sim)
    return sim,config


@pytest.mark.parametrize('angle',[0.,90.,45.])
def test_automatic_channel_patch_is_real_deterministic_and_has_full_camp_clearance(angle):
    sim,config=channel_site(angle);before=deepcopy(sim.report());height=sim.terrain.height.copy()
    assert config.auto_work_area
    first=prepare(sim,config);second=prepare(sim,config)
    assert asdict(first)==asdict(second) and not first.auto_work_area
    assert first.work_rect!=config.work_rect
    assert sim.conventional is None and sim.report()==before
    np.testing.assert_array_equal(sim.terrain.height,height)
    fleet=sim.install_conventional(config)
    assert fleet.config.work_rect==first.work_rect
    assert np.any(fleet.work_mask&sim.terrain.dig_mask)
    assert fleet.passable(fleet.base_position,4.,ignore_vehicles=True)
    assert all(fleet.passable(m.position,m.radius,m) for m in fleet.machines)
    assert all(not np.any(sim.terrain.work_mask[sim.terrain.patch(*m.home[:2],m.radius)]) for m in fleet.machines)
    assert sim.time==0. and fleet.excavated_m3==0.


def test_connected_sea_cannot_be_under_selected_patch_camp_or_any_machine_edge():
    from ats.site_ground import marine_mask
    sim,config=channel_site(marine_west=40.)
    fleet=sim.install_conventional(config);water=marine_mask(sim.terrain)
    assert not np.any(water&fleet.work_mask)
    for p,radius in [(fleet.base_position,4.),*[(m.home,m.radius) for m in fleet.machines]]:
        patch=sim.terrain.patch(*p[:2],radius)
        assert not np.any(water[patch])


def test_deep_profile_without_required_ramp_length_is_explained_before_purchase():
    sim,config=channel_site(depth=40.,side=256.)
    height=sim.terrain.height.copy();before=sim.report()
    with pytest.raises(ValueError,match='[Рр]амп|окно|карт'):
        sim.install_conventional(config)
    assert sim.conventional is None and sim.report()==before
    assert not hasattr(sim.terrain,'_conventional_mask')
    np.testing.assert_array_equal(sim.terrain.height,height)


def test_manual_selection_is_not_silently_replaced_and_nonchannel_auto_is_rejected():
    sim,config=channel_site();manual=deepcopy(config);manual.auto_work_area=False
    assert prepare(sim,manual) is manual
    sim.config.plan.channel=None
    with pytest.raises(ValueError,match='проекта канала'):prepare(sim,config)
    assert sim.conventional is None


def test_channel_without_remaining_cut_does_not_buy_idle_fleet():
    sim,config=channel_site();sim.terrain.height[:]=sim.terrain.target
    with pytest.raises(ValueError,match='нет оставшейся выемки'):sim.install_conventional(config)
    assert sim.conventional is None


def test_six_metre_deep_diagonal_uses_real_channel_axis_not_bounding_box():
    sim,config=channel_site(angle=45.,depth=6.,side=256.)
    resolved=prepare(sim,config);fleet=sim.install_conventional(config)
    axis=np.asarray(fleet.ramp['end_m'])-np.asarray(fleet.ramp['entry_m'])
    np.testing.assert_allclose(axis/np.linalg.norm(axis),[2**-.5,2**-.5],atol=1e-8)
    assert np.linalg.norm(axis)>=6./config.ramp_gradient
    assert np.linalg.norm(np.asarray(resolved.ramp_end_m)-resolved.ramp_entry_m)>=6./config.ramp_gradient
    assert fleet.ramp['required_length_m']==pytest.approx(6./config.ramp_gradient)


def test_short_edges_around_bends_cannot_fake_one_long_straight_ramp():
    source,_=channel_site(depth=6.,side=256.);c=deepcopy(source.config)
    points=[[50.,50.],[80.,50.],[80.,80.],[110.,80.]]
    c.plan=PlanConfig([PlanSegment([a,b],width=12.,height=-6.) for a,b in zip(points,points[1:])],
        channel=dict(start=points[0],tangent=[1.,0.],cell_size=c.terrain.cell_size))
    sim=Simulation.conventional_only(c);config=ConventionalConfig.for_simulation(sim)
    with pytest.raises(ValueError,match='[Рр]амп'):sim.install_conventional(config)
    assert sim.conventional is None


def autonomous_channel():
    from ats.complexes import ComplexProject,ComplexTemplate
    source,_=channel_site();config=deepcopy(source.config)
    config.conventional_only=False;config.fleet.count=4;config.fleet.charge_slots=4;config.fleet.models=[]
    sim=Simulation(config);project=ComplexProject(sim)
    project.add_shared(ComplexTemplate.capture(config),position=(108.,16.))
    return project


def test_auto_patch_reserves_only_its_work_and_checkpoint_does_not_reallocate(tmp_path):
    from ats.checkpoint import save_state,load_state
    project=autonomous_channel();project.select('complex_2')
    config=ConventionalConfig.for_simulation(project.active.simulation)
    fleet=project.active.simulation.install_conventional(config)
    site=project.shared_site;reserved=site.core._conventional_mask
    assert np.all(site.owners[reserved]==0)
    assert any(np.any(unit.simulation.terrain.dig_mask) for unit in project.units)
    assert not fleet.config.auto_work_area
    path=tmp_path/'auto-channel.atsstate'
    save_state(path,project.active.simulation,complex_project=project)
    other=load_state(path).complex_project
    assert other.report()==project.report()
    for _ in range(20):project.step();other.step()
    assert other.report()==project.report()
    assert abs(project.report()['mass_error_kg'])<1e-7


def test_active_unrelated_flight_is_preserved_and_auto_patch_avoids_its_reservation():
    from ats.conventional_integration import _overlap_circle
    project=autonomous_channel()
    for _ in range(10):
        project.step()
        trips=[drone.trip for unit in project.units for drone in unit.simulation.drones if drone.trip is not None]
        if trips:break
    assert trips
    owner=project.units[0].simulation;config=ConventionalConfig.for_simulation(owner)
    fleet=owner.install_conventional(config);reserved=project.shared_site.core._conventional_mask
    for unit in project.units:
        for drone in unit.simulation.drones:
            if drone.trip is not None:
                assert any(drone.trip is trip for trip in trips)
                for point in (drone.trip.dig,drone.trip.dump):
                    assert not _overlap_circle(project.shared_site.core,reserved,point,drone.config.grab_radius or .5)
    assert fleet.config.work_rect!=config.work_rect
