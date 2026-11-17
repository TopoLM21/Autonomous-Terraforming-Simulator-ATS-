"""A service pad belongs to its level carrier, not the terrain below the pad."""
import numpy as np
import pytest
import ats.checkpoint as checkpoint

from ats.base_layout import rotate_offset
from ats.checkpoint import load_state,save_state
from ats.complexes import ComplexProject
from ats.config import Config
from ats.conventional import ConventionalConfig
from ats.drone import DroneState as DS
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.plan import PlanConfig,PlanSegment
from ats.simulation import Simulation


def sloped_mixed_site(slope=-.005,heading=0.):
    c=experiment_config(with_map_size(Config(),192),panel_area=20)
    c.base.layout={'heading_deg':heading,'auto_heading':0.}
    c.base.x=c.base.y=55.
    c.logistics.excavator_enabled=False
    c.logistics.container_count=0
    c.logistics.direct_allowed=c.logistics.idle_digging=False
    axis=(np.arange(c.terrain.n)+.5)*c.terrain.cell_size
    c.initial_height=(slope*(axis[None,:]+.37*axis[:,None])).tolist()
    c.reference_height=c.initial_height
    c.local_reference=True
    c.terrain.dump_rect=(120.,30.,60.,30.)
    c.plan=PlanConfig([PlanSegment([[105.,120.],[130.,120.]],width=4.,height=-2.),
                       PlanSegment([[145.,145.],[175.,145.]],width=4.,height=-2.)])
    s=Simulation(c)
    assert s.state.value=='RUNNING',s.reason
    s.request_landing()
    s.run(80)
    assert s.logistics.battery_platforms.ready
    p=ComplexProject(s)
    p.enable_shared()
    s.install_conventional(ConventionalConfig(base_x=12.,base_y=155.,work_rect=(100.,115.,35.,15.)))
    return p,s


def balanced(p):
    p.validate()
    report=p.report()
    assert abs(report['mass_error_kg'])<1e-7
    assert abs(report['energy_error_j'])<.002
    assert abs(report['equipment_mass_error_kg'])<1e-7


def deplete_for_swap(p,s,drone):
    old=drone.battery
    drone.battery=drone.config.battery*.25
    s._initial_energy-=old-drone.battery
    p.initial_energy_kwh-=(old-drone.battery)/3_600_000
    drone.charge_requested=True
    s.logistics.sync_batteries()


@pytest.mark.parametrize('slope',[-.005,.005])
@pytest.mark.parametrize('heading',[0.,90.,37.])
def test_all_mixed_docks_share_rigid_carrier_height_on_slopes(slope,heading):
    p,s=sloped_mixed_site(slope,heading)
    w=s.logistics
    for d in s.drones:
        goal=s.slot_position(d.slot)
        carrier_id=w.battery_platforms.dock_carrier(d.slot)
        carrier=next(a for a in s.base.layout_spec['platforms'] if a['id']==carrier_id)
        centre=s.base.position[:2]+rotate_offset(s.base.layout_spec['parameters'],carrier['offset_m'])
        expected=s.terrain.sample(*centre)+s.config.base.roof_height
        assert goal[2]==pytest.approx(expected,abs=1e-12)
        assert p.shared_site.obstacle_height(goal[:2])==pytest.approx(goal[2],abs=1e-12)
        wait=s.wait_position(d.identifier)
        assert wait[2]>=max(s.terrain.sample(*wait[:2]),p.shared_site.obstacle_height(wait[:2]))
        # Exercise the actual powered ascent/descent, including the initial
        # deck contact and other aircraft remaining parked on their own decks.
        d.destination=goal.copy()
        d.state=DS.TRANSFER_TAKEOFF
        s._starts={o.identifier:o.position.copy() for o in s.drones}
        p.shared_site.start_tick()
        s._airborne(d,1.)
        assert s.state.value=='RUNNING',s.reason
        assert d.position[2]>goal[2]
        d.state=DS.LANDING
        s._starts={o.identifier:o.position.copy() for o in s.drones}
        p.shared_site.start_tick()
        s._airborne(d,1.)
        assert s.state.value=='RUNNING',s.reason
        assert d.state==DS.DOCKED
        np.testing.assert_allclose(d.position,goal,atol=1e-12)
    w.sync_batteries()
    balanced(p)


def test_landing_does_not_freeze_another_real_swap_and_resumes_from_save(tmp_path):
    p,s=sloped_mixed_site()
    w=s.logistics
    swapping=s.drones[1]
    deplete_for_swap(p,s,swapping)
    p.step()
    swap=w.battery_service.drone_swaps[swapping.serial]
    assert swapping.state==DS.SWAPPING_BATTERIES and swap['phase']=='FETCH'
    landing=s.drones[0]
    landing.destination=s.slot_position(landing.slot)
    landing.position=landing.destination+np.array([0.,0.,.05])
    landing.state=DS.LANDING
    w.sync_batteries()
    path=tmp_path/'slope-swap-and-landing.atsstate'
    save_state(path,s,complex_project=p)
    q=load_state(path).complex_project
    ids=set(w.inventory.items)
    old_ids=list(swapping.battery_ids)
    before=p.time
    for _ in range(150):
        p.step()
        q.step()
        assert s.state.value=='RUNNING',s.reason
        assert q.active.simulation.state.value=='RUNNING',q.active.simulation.reason
    assert p.time==before+150 and p.report()==q.report()
    assert landing.docked
    assert swapping.serial not in w.battery_service.drone_swaps
    assert swapping.battery_ids!=old_ids
    assert set(w.inventory.items)==ids
    balanced(p)
    balanced(q)


@pytest.mark.parametrize('phase',['FETCH','SWAP','RETURN'])
def test_old_save_repairs_only_dock_anchors_and_resumes_proven_landing_block(tmp_path,monkeypatch,phase):
    from ats.inventory import BatteryModule,Location as L
    from ats.simulation import MissionState
    p,s=sloped_mixed_site()
    w=s.logistics
    swapping=s.drones[1]
    deplete_for_swap(p,s,swapping)
    for _ in range(200):
        p.step()
        swap=w.battery_service.drone_swaps.get(swapping.serial)
        if swap and swap['phase']==phase:break
    else:pytest.fail(f'Swap phase {phase} not reached')
    # Recreate the old serialized deck anchors and an interrupted swap, without
    # changing a single module ID, reservation, joule or item of equipment.
    old_pose={}
    for drone in s.drones:
        corrected=s.slot_position(drone.slot)
        old_pose[drone.serial]=np.r_[corrected[:2],s.terrain.sample(*corrected[:2])+s.config.base.roof_height]
        if drone.docked:drone.position[:]=old_pose[drone.serial]
    w.sync_batteries()
    if phase=='FETCH':w.battery_service._handling_leg(swapping,swap,phase)
    elif phase=='RETURN':
        for id in swap['old_ids']:w.inventory.items[id].position=tuple(swapping.position)
        w.battery_service._handling_leg(swapping,swap,phase)
    else:
        for id in swap['ids']:
            assert w.inventory.items[id].location==L.IN_TRANSFER
            w.inventory.items[id].position=tuple(swapping.position)
            swap['targets'][id]=swapping.position.tolist()
            swap['paths'][id][-1]=swapping.position.tolist()
    landing=s.drones[0]
    corrected=s.slot_position(landing.slot)
    landing.destination=old_pose[landing.serial].copy()
    landing.position=corrected+np.array([0.,0.,.05])
    landing.state=DS.LANDING
    s.block(f'Траектория дрона {landing.serial} пересекает рельеф или базу')
    w.sync_batteries()
    ids=set(w.inventory.items)
    energy={b.id:b.energy_j for b in w.inventory.select(BatteryModule)}
    airborne_position=landing.position.copy()
    transfer_positions={id:w.inventory.items[id].position for id in swap['ids']+swap['old_ids'] if w.inventory.items[id].location==L.IN_TRANSFER}
    old_ids=list(swapping.battery_ids)
    path=tmp_path/f'old-{phase}.atsstate'
    with monkeypatch.context() as patch:
        patch.setattr(checkpoint,'VERSION',22)
        save_state(path,s,complex_project=p)
    q=load_state(path).complex_project
    restored=q.active.simulation
    assert restored.state==MissionState.RUNNING
    assert restored.drones[0].state==DS.LANDING
    np.testing.assert_array_equal(restored.drones[0].position,airborne_position)
    np.testing.assert_array_equal(restored.drones[0].destination,restored.slot_position(0))
    np.testing.assert_array_equal(restored.drones[1].position,restored.slot_position(1))
    assert set(restored.logistics.inventory.items)==ids
    actual={b.id:b.energy_j for b in restored.logistics.inventory.select(BatteryModule)}
    assert actual==energy
    if phase!='SWAP':
        for id,position in transfer_positions.items():np.testing.assert_array_equal(restored.logistics.inventory.items[id].position,position)
    s.repair_legacy_dock_geometry()
    assert p.report()==q.report()
    assert s.repair_legacy_dock_geometry()==[]
    for _ in range(200):p.step();q.step()
    assert s.state==MissionState.RUNNING and p.report()==q.report()
    assert not w.battery_service.drone_swaps and landing.docked
    assert phase=='RETURN' or swapping.battery_ids!=old_ids
    balanced(p);balanced(q)
    # The new format must round-trip without running the migration twice.
    save_state(path,restored,complex_project=q)
    assert load_state(path).complex_project.report()==q.report()


@pytest.mark.parametrize('case',['unrelated_reason','unowned_slot','airborne_below_roof','arbitrary_ground_pose',
                                  'base_moving','regional_folded','regional_drive'])
def test_legacy_geometry_repair_does_not_resume_or_move_unproven_state(case):
    from ats.simulation import MissionState
    p,s=sloped_mixed_site()
    d=s.drone
    corrected=s.slot_position(d.slot)
    old=np.r_[corrected[:2],s.terrain.sample(*corrected[:2])+s.config.base.roof_height]
    d.position=corrected+np.array([0.,0.,.05])
    d.destination=old
    d.state=DS.LANDING
    s.block(f'Траектория дрона {d.serial} пересекает рельеф или базу')
    if case=='unrelated_reason':s.reason='Нет места для отвала'
    elif case=='unowned_slot':s.slots[d.slot]=None
    elif case=='airborne_below_roof':d.position[2]=corrected[2]-.01
    elif case=='arbitrary_ground_pose':d.state=DS.DOCKED;d.position=old+np.array([.1,0.,0.])
    elif case=='base_moving':s.base.destination=s.base.position[:2]+np.array([1.,0.])
    elif case=='regional_folded':s.logistics.regional_folded=True
    else:s.regional_stage='DRIVE'
    before=d.position.copy()
    s.repair_legacy_dock_geometry()
    assert s.state==MissionState.BLOCKED
    np.testing.assert_array_equal(d.position,before)
