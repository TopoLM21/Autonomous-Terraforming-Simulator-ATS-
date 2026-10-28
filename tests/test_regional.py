import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pytest
from ats.config import Config
from ats.landscape import Landscape,ElevationRaster,workspace_config
from ats.experiments import experiment_config
from ats.complexes import ComplexProject,ComplexTemplate
from ats.simulation import Simulation
from ats.checkpoint import save_state,load_state
from ats.inventory import Location as L
from ats.plan import PlanConfig,PlanSegment


def region_group(count=2,soil=False,slots=2,side=128,cell=.5,containers=2):
    from ats.soil import SoilProfile
    r=ElevationRaster(np.zeros((6,6)),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=max(100.,side/2),dy=max(100.,side/2),latitude_deg=30.))
    land=Landscape(r,cell_size=cell,soil_profile=SoilProfile.qattara() if soil else None)
    c=workspace_config(land,experiment_config(Config(),drones=2,containers=containers,landing_slots=slots),[64.,64.],side)
    c.logistics.field_deployment_seconds=4.
    c.plan=PlanConfig([PlanSegment([[20,40],[26,40]],width=2,height=-.2)])
    p=ComplexProject(Simulation(c),land);p.enable_shared()
    for _ in range(count-1):p.add_shared(ComplexTemplate.capture(c))
    return p


def advance(p,predicate,limit=4000):
    for _ in range(limit):
        if predicate():return
        p.step()
        assert not p.blocked,p.report()
    pytest.fail(str(p.transition.report() if p.transition else p.report()))


@pytest.mark.parametrize('soil',[False,True])
def test_group_roundtrip_preserves_ground_materials_ids_clocks_and_energy(soil):
    p=region_group(soil=soil);t=p.shared_site.core
    load=t.scoop_load(22,40,35.,radius=1.) if soil else None
    mass=load.mass if soil else t.scoop(22,40,35.,radius=1.)
    t.deposit(100,100,mass,load=load)
    soil_blocks={key:values.copy() for key,values in t.soil.tiles.items()} if soil else {}
    ground=t.height.copy();ids=p.inventory_ids();objects=[u.simulation for u in p.units]
    batteries={u.id:{id:o.energy_j for id,o in u.simulation.logistics.inventory.items.items() if hasattr(o,'energy_j')} for u in p.units}
    start=p.report();p.move_window([96.,64.]);advance(p,lambda:p.transition is None)
    assert all(u.simulation is s for u,s in zip(p.units,objects))
    assert p.inventory_ids()==ids and p.time>start['time_seconds']+10
    assert all(u.born_at+u.simulation.time==p.time for u in p.units)
    assert all(u.simulation.config.georeference['origin_m']==[96.,64.] for u in p.units)
    assert p.landscape.tiles
    assert p.landscape.report()['soil_delta_m3']==pytest.approx(0,abs=1e-9)
    np.testing.assert_array_equal(p.landscape.read_window([64.,64.],256,cell_size=.5)[1],ground)
    p.move_window([64.,64.]);advance(p,lambda:p.transition is None)
    np.testing.assert_array_equal(p.shared_site.core.height,ground)
    for key,values in soil_blocks.items():np.testing.assert_array_equal(p.shared_site.core.soil.tiles[key],values)
    assert p.inventory_ids()==ids and len(p.window_history)==2
    r=p.report();assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002
    assert r['equipment_mass_error_kg']==pytest.approx(0,abs=1e-7)
    assert r['consumed_kwh']>start['consumed_kwh']
    assert any(o.energy_j!=batteries[u.id][id] for u in p.units for id,o in u.simulation.logistics.inventory.items.items() if id in batteries[u.id])
    if soil:assert r['shared_worksite']['max_material_mass_error_kg']<1e-7
    assert len(p.shared_site.core.config.plan.segments)==1;p.validate()


@pytest.mark.parametrize('stage',['COLLECT','PACK','DRIVE','DEPLOY'])
def test_exact_resume_during_group_transfer(tmp_path,stage):
    p=region_group(soil=True);p.move_window([96.,64.])
    advance(p,lambda:p.transition.phase==stage)
    if stage=='DRIVE':
        advance(p,lambda:any(u.simulation.logistics.excavator.moving_seconds>3 for u in p.units))
    path=tmp_path/'travelling.atsstate';save_state(path,p.active.simulation,p.landscape,complex_project=p)
    q=load_state(path).complex_project
    assert p.report()==q.report()
    for _ in range(2200):
        if p.transition is None and q.transition is None:break
        p.step();q.step()
        assert p.time==q.time
    assert p.transition is None and q.transition is None
    assert p.report()==q.report()
    for a,b in zip(p.units,q.units):assert a.simulation.report()==b.simulation.report()
    np.testing.assert_array_equal(p.shared_site.core.height,q.shared_site.core.height)


def test_repeated_request_cancel_and_invalid_windows_are_non_destructive():
    p=region_group();ids=p.inventory_ids();before=p.report()
    for origin in ([64.,64.],[400.,64.],[-1.,64.],[192.,64.]):
        with pytest.raises(ValueError):p.move_window(origin)
        assert p.report()==before
    p.move_window([96.,64.])
    with pytest.raises(ValueError,match='уже'):p.move_window([96.,64.])
    advance(p,lambda:p.transition.phase=='DRIVE')
    advance(p,lambda:any(u.simulation.logistics.excavator.moving_seconds>5 for u in p.units))
    positions=[u.simulation.base.position.copy() for u in p.units];time=p.time
    assert p.cancel_window_move() and not p.cancel_window_move()
    assert p.time==time and p.inventory_ids()==ids
    for u,position in zip(p.units,positions):np.testing.assert_array_equal(u.simulation.base.position,position)
    assert all(u.simulation.config.georeference['origin_m']==[64.,64.] for u in p.units)
    assert all(not o.reserved_by for u in p.units for o in u.simulation.logistics.inventory.items.values())
    assert p.window_history[-1]['status']=='CANCELLED';p.validate()


def test_energy_shortage_waits_with_live_clock_and_no_free_movement():
    p=region_group(count=1);s=p.active.simulation
    s.base.battery=0.;s._initial_energy-=s.config.base.battery*s.config.base.initial_charge
    s.config.base.peak_irradiance=0.;s.logistics.sync_batteries()
    positions=[s.base.position.copy(),s.logistics.excavator.position.copy()]
    p.initial_energy_kwh=s.report(full=False)['initial_energy_kwh']
    p.move_window([96.,64.])
    for _ in range(30):p.step()
    assert p.time==30 and p.transition is not None
    np.testing.assert_array_equal(s.base.position,positions[0]);np.testing.assert_array_equal(s.logistics.excavator.position,positions[1])
    assert s.config.georeference['origin_m']==[64.,64.]
    assert abs(p.report()['energy_error_j'])<.002
    p.cancel_window_move()


def test_navigation_detours_and_does_not_cross_new_blocker():
    from ats.regional import ground_route
    p=region_group(count=1);s=p.active.simulation;e=s.logistics.excavator;t=s.terrain
    e.position[:]=[30.,70.,0.]
    mask=(abs(t.x-40)<1)&(abs(t.y-70)<2)
    t.height[mask]=3.;t.revision+=1;t._mark_dirty(np.s_[0:256,0:256],mask)
    path=ground_route(e,t,[50.,70.])
    assert path and len(path)>1
    previous=e.position.copy()
    for point in path:
        assert e.can_drive(np.r_[point,t.sample(*point)],t)
        e.position=np.r_[point,t.sample(*point)]
    e.position=previous
    assert not e.can_drive(np.array([50.,70.,0.]),t)


def test_ui_move_button_keeps_group_and_does_not_autopause_after_landing():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.landscape_ui import LandscapeDialog
    from ats.checkpoint import Session
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p=region_group();w=MainWindow(p.active.simulation.config,session=Session(p.active.simulation,p.landscape,{},complex_project=p));w.timer.stop()
    d=LandscapeDialog(w);assert d.side.currentData()==128
    d.x.setValue(96);d.y.setValue(64);d.move_button.click();w.timer.stop()
    assert w.complex_project is p and p.transition is not None and w.running
    w.credit=3.;w.on_frame();assert w.running and p.time>0
    advance(p,lambda:p.transition is None)
    w.credit=2.;w.on_frame();w.refresh(True)
    assert w.sim is p.active.simulation and w._display_origin==[96.,64.]
    assert w.config.georeference['origin_m']==[96.,64.]
    w.running=False;d.close();w.close();app.processEvents()


def test_move_drains_live_cargo_and_keeps_late_complex_clock():
    p=region_group(count=1)
    for _ in range(18):p.step()
    p.add_shared(ComplexTemplate.from_simulation(p.active.simulation))
    p.select('complex_2');ids=p.inventory_ids()
    advance(p,lambda:any(d.cargo>1e-6 for u in p.units for d in u.simulation.drones),limit=400)
    at=p.time;p.move_window([96.,64.]);advance(p,lambda:p.transition is None,limit=6000)
    assert p.active_id=='complex_2' and p.inventory_ids()==ids
    assert p.time>at and all(u.born_at+u.simulation.time==p.time for u in p.units)
    assert all(d.cargo==0 for u in p.units for d in u.simulation.drones)
    assert all(c.soil_kg==0 for u in p.units for c in u.simulation.logistics.containers)
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002
    p.validate()


def test_fewer_charging_slots_followers_fly_and_cancel_does_not_drop_equipment():
    p=region_group(count=1,slots=1);s=p.active.simulation;w=s.logistics
    p.move_window([96.,64.]);advance(p,lambda:p.transition.phase=='ARRIVE')
    follower=next(d for d in s.drones if not d.docked);before=follower.consumed
    height=follower.position[2]
    advance(p,lambda:follower.position[2]>height+.1)
    assert follower.consumed>before
    hoist=w.inventory.items[f'hoist_{follower.serial}'];assert hoist.location==L.DRONE_MOUNT
    p.cancel_window_move()
    assert hoist.location==L.DRONE_MOUNT and hasattr(follower,'regional_mass')
    for u in p.units:u.simulation.landing_requested=True
    advance(p,lambda:follower.grounded)
    assert hoist.location==L.GROUND and not hasattr(follower,'regional_mass')
    assert abs(p.report()['energy_error_j'])<.002;p.validate()


def test_base_stops_before_a_new_obstacle_and_retains_old_window():
    p=region_group(count=1);s=p.active.simulation
    p.move_window([96.,64.]);move=p.transition
    advance(p,lambda:move.phase=='DRIVE' and move.actor_index<len(move.actors) and move.actors[move.actor_index][1]=='base' and s.base.destination is not None)
    position=s.base.position.copy();destination=s.base.destination.copy();delta=destination-position[:2]
    point=position[:2]+delta/np.linalg.norm(delta)*2
    t=s.terrain;patch=t.patch(*point,2.)
    t.height[patch]=5.;t.revision+=1;t._mark_dirty(patch,np.ones(t.height[patch].shape,dtype=bool))
    for _ in range(3):p.step()
    assert p.transition is move and s.config.georeference['origin_m']==[64.,64.]
    assert np.linalg.norm(s.base.position[:2]-position[:2])<1e-7
    assert move.phase=='ENERGY' and 'препятствие' in move.reason
    p.cancel_window_move()


def test_channel_move_next_ui_preserves_existing_fleet_and_ready_plan():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.landscape_ui import LandscapeDialog
    from ats.channel import ChannelProject
    from ats.checkpoint import Session
    from tests.test_channel import coastal_dem
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    land=Landscape(coastal_dem(gap=100));channel=ChannelProject.create(land.raster,[[50,150],[250,150]],width_m=1,depth_m=.01,side_m=128)
    land.channel_project=channel.to_data()
    c=channel.configuration(land,experiment_config(Config(),drones=2,containers=1));c.logistics.field_deployment_seconds=2
    s=Simulation(c);p=ComplexProject(s,land);p.enable_shared();ids=p.inventory_ids()
    w=MainWindow(c,session=Session(s,land,{},complex_project=p));w.timer.stop();d=LandscapeDialog(w)
    d.move_next_button.click();w.timer.stop()
    assert p.transition is not None,d.result.text()
    advance(p,lambda:p.transition is None,limit=6000)
    assert s is p.active.simulation and p.inventory_ids()==ids
    assert s.config.georeference['channel']['section']==1 and s.config.plan.segments
    assert p.time>0 and land.channel_project['active_section']==1
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002
    p.validate();w.running=False;d.close();w.close();app.processEvents()


def test_window_alignment_preserves_world_positions_and_coarse_soil_grid():
    p=region_group(count=1,soil=True);p.move_window([96.25,64.])
    move=p.transition;assert move.new_origin==[96.,64.]
    advance(p,lambda:move.phase=='ARRIVE')
    s=p.active.simulation
    before=s.base.position+np.r_[move.old_origin,0.]
    advance(p,lambda:move.phase=='DEPLOY')
    np.testing.assert_array_equal(s.base.position+np.r_[move.new_origin,0.],before)
    assert s.terrain.cell==.5 and s.terrain.soil.fine_cell==.5
    advance(p,lambda:p.transition is None);p.validate()


def test_base_route_avoids_its_own_excavator():
    from ats.regional import ground_route,own_ground_clear
    p=region_group(count=1);s=p.active.simulation
    s.base.position[:]=[30.,70.,0.];s.logistics.excavator.position[:]=[50.,70.,0.]
    assert not own_ground_clear(s.base,s.terrain,[30.,70.],[70.,70.],base=True)
    path=ground_route(s.base,s.terrain,[70.,70.],base=True)
    assert path and len(path)>1
    previous=s.base.position[:2]
    for point in path:
        assert own_ground_clear(s.base,s.terrain,previous,point,base=True)
        previous=np.array(point)


def test_unavailable_route_is_retried_only_after_relevant_change(monkeypatch):
    import ats.regional as regional
    p=region_group(count=1);calls=[]
    monkeypatch.setattr(regional,'ground_route',lambda *args,**kwargs:calls.append(1))
    p.move_window([96.,64.])
    for _ in range(100):p.step()
    assert p.transition.phase=='ENERGY' and len(calls)==1
    p.shared_site.core.revision+=1
    p.step();assert len(calls)==2
    assert p.active.simulation.config.georeference['origin_m']==[64.,64.]
    p.cancel_window_move();p.validate()


def test_alignment_at_dem_edge_stays_inside_the_region():
    from ats.regional import target_configuration
    p=region_group(count=1);p.landscape.raster.metadata['dx']=100.18
    target=target_configuration(p,[372.75,64.])
    assert target.georeference['origin_m']==[372.5,64.]
    assert target.georeference['origin_m'][0]+128<=p.landscape.raster.extent_m[0]


def test_fourteen_complexes_plan_future_clearance_without_teleporting_actors():
    p=region_group(count=14,side=512,cell=1.,containers=0)
    poses=[(u.simulation.base.position.copy(),u.simulation.logistics.excavator.position.copy()) for u in p.units]
    p.move_window([320.,64.]);p.step()
    assert p.transition.phase=='ENERGY'
    assert len(p.transition.routes)==28,p.transition.reason
    for u,(base,robot) in zip(p.units,poses):
        np.testing.assert_array_equal(u.simulation.base.position,base)
        np.testing.assert_array_equal(u.simulation.logistics.excavator.position,robot)
    p.validate();p.cancel_window_move()


def test_old_window_is_retained_until_dry_soil_has_settled():
    p=region_group(count=1,containers=0);p.move_window([96.,64.])
    advance(p,lambda:p.transition.phase=='ARRIVE')
    s=p.active.simulation
    assert all(d.grounded for d in s.drones)
    s._terrain_active=True
    assert not p.transition._settled()
    assert s.config.georeference['origin_m']==[64.,64.]
    advance(p,lambda:p.transition is None)
    assert s.config.georeference['origin_m']==[96.,64.]


def test_follower_with_requested_charge_waits_for_full_battery():
    p=region_group(count=1,slots=1,containers=0);s=p.active.simulation
    p.move_window([96.,64.]);p.step()
    d=next(d for d in s.drones if not d.docked)
    d.battery*=.8;d.charge_requested=True
    assert not p.transition._energy_ready()
    assert d.identifier in s.queue
    assert 'полную зарядку' in p.transition.reason
    p.cancel_window_move()
