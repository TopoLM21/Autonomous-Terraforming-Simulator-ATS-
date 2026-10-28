from copy import deepcopy
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.simulation import Simulation
from ats.inventory import BatteryModule,Location as L
from ats.base_layout import solve_layout
from ats.checkpoint import save_state,load_state


def configuration(side=160,area=20):
    c=experiment_config(with_map_size(Config(),side),panel_area=area)
    c.base.layout={};c.logistics.excavator_enabled=False
    return c


def stationary(area=20):
    s=Simulation(configuration(area=area));assert s.state.value=='RUNNING',s.reason
    s.request_landing();s.run(80)
    assert s.logistics.battery_platforms.ready
    return s


def low_battery(s,d=None):
    d=d or s.drone
    old=d.battery;d.battery=d.config.battery*.25;s._initial_energy-=old-d.battery
    d.charge_requested=True;s.logistics.sync_batteries();return d


def test_manifest_carries_2550kwh_on_several_real_carriers():
    c=configuration();c.base.equipment.batteries[0].quantity=510
    spec=solve_layout(c);bess=[a for a in spec['platforms'] if a['role']=='BESS']
    assert len(bess)==6
    assert sum(e['mass_kg'] for a in bess for e in a['loads'] if e['kind']=='BATTERY')==25500
    assert sum(e['kwh'] for a in bess for e in a['loads'])==2550
    assert all(a['payload_kg']<=5000 for a in spec['platforms'])
    assert sum(a['gross_mass_kg'] for a in spec['platforms'])==pytest.approx(spec['packed_gross_mass_kg'])
    mounts=[e['mount_id'] for a in spec['platforms'] for e in a['loads']]
    assert len(mounts)==len(set(mounts))


def test_actual_module_positions_and_individual_charge_slots_are_bound():
    s=stationary();w=s.logistics;p=w.battery_platforms
    for b in w.inventory.select(BatteryModule):
        assert b.id in p.bindings
        if b.holder_id=='base':assert np.allclose(b.position,p.slot_position(b.id))
    for id,slot in w.battery_service.slots.items():
        assert p.bindings[id]['platform_id']==p.bindings[slot.id]['platform_id']
        assert slot.mass_kg>0
    assert sum(r['mass_kg'] for r in p.loads().values())==pytest.approx(s.base.config.mass+sum(d.config.empty_mass for d in s.drones if d.docked))
    assert abs(s.report()['energy_error_j'])<.002
    fixed=sum(o.mass_kg for o in w.inventory.items.values() if o.kind=='BATTERY_PLATFORM_HARDWARE')
    assert s.base_specification()['installed_equipment_mass_kg']==pytest.approx(s.base.config.equipment.mass+fixed)


def test_deployment_and_reverse_are_powered_partial_and_saved(tmp_path):
    s=Simulation(configuration());s.request_landing();p=s.logistics.battery_platforms
    s.run(12);assert all(0<c.supports<1 and c.apron==0 for c in p.carriers.values())
    path=tmp_path/'supports.atsstate';save_state(path,s);other=load_state(path).simulation
    assert other.report()==s.report()
    before=s.base.consumed;p.request('PACK');other.logistics.battery_platforms.request('PACK')
    for _ in range(20):s.step();other.step()
    assert p.stowed and other.report()==s.report() and s.base.consumed>before
    p.request('DEPLOY');s.run(80);assert p.ready


def test_zero_energy_does_not_open_supports_or_create_charge():
    c=configuration(area=0);c.base.initial_charge=0
    s=Simulation(c);s.request_landing();s.run(20)
    assert all(c.supports==0 and c.reason for c in s.logistics.battery_platforms.carriers.values())
    assert s.base.battery==0 and abs(s.report()['energy_error_j'])<.002


def test_real_swap_transfers_both_sets_and_packing_waits_for_it(tmp_path):
    s=stationary();w=s.logistics;p=w.battery_platforms;d=low_battery(s)
    before=set(w.inventory.items);old=list(d.battery_ids);s.step()
    swap=w.battery_service.drone_swaps[d.serial]
    assert swap['phase']=='FETCH' and all(w.inventory.items[id].location==L.IN_TRANSFER for id in swap['ids'])
    p.request('PACK');s.run(4)
    assert not p.stowed and any(c.state=='WAIT_SERVICE' for c in p.carriers.values())
    path=tmp_path/'exchange.atsstate';save_state(path,s);other=load_state(path).simulation
    for _ in range(250):s.step();other.step()
    assert not w.battery_service.drone_swaps and p.stowed
    assert d.battery_ids!=old and all(w.inventory.items[id].location==L.BASE_STORAGE for id in old)
    assert set(w.inventory.items)==before and other.report()==s.report()
    assert abs(s.report()['energy_error_j'])<.002


@pytest.mark.parametrize('phase',['FETCH','SWAP','RETURN'])
def test_reset_order_returns_handled_batteries_without_loss(phase):
    s=stationary();w=s.logistics;d=low_battery(s);ids=set(w.inventory.items)
    for _ in range(220):
        s.step()
        swap=w.battery_service.drone_swaps.get(d.serial)
        if swap and swap['phase']==phase:break
    else:pytest.fail('Swap phase was not reached')
    w.reset_drone_order(d.serial)
    for _ in range(240):
        s.step()
        if d.serial not in w.battery_service.drone_swaps:break
    assert not any(o.location==L.IN_TRANSFER for o in w.inventory.select(BatteryModule))
    assert set(w.inventory.items)==ids and not w.battery_platforms.handlers
    assert abs(s.report()['energy_error_j'])<.002


def test_pv_assembly_does_not_block_working_battery_platform():
    s=stationary(area=68);w=s.logistics;assert not w.field_ready
    w.request_tool(s.drone,'CONTAINER_HOOK');s.run(45)
    assert s.drone.attachment=='CONTAINER_HOOK' and not w.tool_changes
    assert np.allclose(w.inventory.items['grabber_1'].position,w.battery_platforms.slot_position('grabber_1'))
    d=low_battery(s)
    for _ in range(240):
        s.step()
        if w.battery_service.drone_swap_count:break
    assert w.battery_service.drone_swap_count==1
    assert abs(s.report()['energy_error_j'])<.002


def test_legacy_layout_and_service_remain_explicitly_available():
    c=configuration();c.logistics.battery_platforms=None
    s=Simulation(c)
    assert s.logistics.battery_platforms is None and 'bindings' not in s.base.layout_spec


@pytest.mark.parametrize('seconds,packing',[(12,False),(35,False),(52,False),(80,False),(2,True),(12,True),(35,True)])
def test_exact_resume_at_each_platform_stage(tmp_path,seconds,packing):
    s=stationary() if packing else Simulation(configuration());s.request_landing()
    if packing:s.logistics.battery_platforms.request('PACK')
    s.run(seconds);path=tmp_path/'stage.atsstate';save_state(path,s);q=load_state(path).simulation
    assert s.report()==q.report()
    for _ in range(80):s.step();q.step()
    assert s.report()==q.report()


def test_hot_shipments_keep_existing_mounts_ids_and_energy(tmp_path):
    s=stationary();w=s.logistics;before={id:(e['platform_id'],e['offset_m']) for id,e in w.battery_platforms.bindings.items() if id in w.inventory.items}
    energy=s.base.battery;s.set_base_battery_count(100)
    assert s.base.battery==energy
    new=w.replenish_batteries(s.drone.battery_ids[0],6,soc=.6)
    assert len(new)==6 and all(id in w.battery_platforms.bindings for id in new)
    for id,mount in before.items():
        e=w.battery_platforms.bindings[id];assert (e['platform_id'],e['offset_m'])==mount
    s.set_base_battery_count(4);s.set_base_battery_count(5)
    s.set_panel_area(44)
    assert all(id in w.battery_platforms.bindings for id in w.base_battery_ids)
    for q in w.solar_array.sections:assert np.allclose(w.inventory.items[q.id].position,w.battery_platforms.slot_position(q.id))
    path=tmp_path/'hot.atsstate';save_state(path,s);q=load_state(path).simulation
    assert s.report()==q.report() and abs(s.report()['energy_error_j'])<.002


def test_packing_pauses_charge_at_exact_energy_and_resumes():
    s=stationary();w=s.logistics;p=w.battery_platforms
    b=next(b for b in w.battery_service.modules() if b.location==L.BASE_STORAGE)
    w.initial_extra_energy-=b.energy_j;b.energy_j=0.
    s.run(5);assert b.energy_j>0
    p.request('PACK');s.run(80);assert p.stowed
    before=b.energy_j;s.run(20);assert b.energy_j==before
    p.request('DEPLOY');s.run(80);assert b.energy_j>before
    assert abs(s.report()['energy_error_j'])<.002


def test_one_service_carrier_can_work_while_another_is_packed():
    s=stationary();p=s.logistics.battery_platforms;first=p.dock_carrier(0)
    p.request('PACK',carrier_id=first);s.run(80)
    assert not p.service_ready(0) and p.service_ready(1)
    d=low_battery(s,s.drones[1]);s.run(220)
    assert s.logistics.battery_service.drone_swap_counts[d.serial]==1
    assert p.carriers[first].state=='STOWED'


def test_full_support_footprint_rejects_sea_under_its_edge_and_keeps_equipment(tmp_path):
    s=stationary();w=s.logistics;p=w.battery_platforms;t=s.terrain
    ids=set(w.inventory.items);rect=p.rectangles()[list(p.carriers).index(p.dock_carrier(0))];x,y,ww,hh=rect
    # The centre remains dry; only a support/access edge enters the sea mask.
    ix=int((x+.2)/t.cell);iy=int((y+.2)/t.cell)
    t.reference=t.reference.copy();t.reference[iy,ix]=-1
    t.config.georeference={'source':{'crs':'LOCAL_METRES','latitude_deg':55.},'origin_m':[0.,0.],'vertical_datum':'MSL','marine':dict(shape=[2,2],spacing=[160,160],origin=[0,0],bits='8A==')}
    t.revision+=1;s.step()
    assert p.issue.startswith('BATTERY_SITE_BLOCKED') and not p.service_ready(0)
    assert p.service_ready(1)  # the other dry service carrier remains available
    assert set(w.inventory.items)==ids
    path=tmp_path/'blocked.atsstate';save_state(path,s);q=load_state(path).simulation
    assert set(q.logistics.inventory.items)==ids and q.logistics.battery_platforms.issue


def test_driving_requires_stowed_platforms_and_respects_bess_power():
    s=stationary();b=s.base;before=b.position.copy();energy=b.battery
    b.destination=b.position[:2]+[.1,0]
    reason=b.tick(1.,9,s.drones,[False]*4,terrain=s.terrain)
    assert 'опоры' in reason and b.battery==energy and np.array_equal(b.position,before)
    s.logistics.battery_platforms.request('PACK');s.run(80)
    b.destination=b.position[:2]+[.1,0];energy=b.battery
    reason=b.tick(1.,9,s.drones,[False]*4,terrain=s.terrain,storage_discharge_w=1.)
    assert 'мощности' in reason and b.battery==energy
    b.destination=b.position[:2]+[.1,0]
    b.tick(1.,9,s.drones,[False]*4,terrain=s.terrain)
    assert b.battery<energy and not np.array_equal(b.position,before)


def test_overweight_module_and_live_geometry_change_are_rejected_atomically():
    c=configuration();c.base.equipment.batteries[0].mass_kg=6000
    with pytest.raises(ValueError,match='Модуль'):solve_layout(c)
    s=stationary();before=s.report()
    with pytest.raises(ValueError,match='мгновенно'):s.configure_platforms({'platform_length_m':7.})
    assert s.report()==before
    c=configuration();c.logistics.individual_chargers=False
    with pytest.raises(ValueError,match='индивидуальные'):solve_layout(c)


def platform_group():
    from ats.landscape import Landscape,ElevationRaster,workspace_config
    from ats.complexes import ComplexProject,ComplexTemplate
    from ats.plan import PlanConfig,PlanSegment
    from ats.soil import SoilProfile
    raster=ElevationRaster(np.zeros((6,6)),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=200.,dy=200.,latitude_deg=30.))
    land=Landscape(raster,cell_size=1.,soil_profile=SoilProfile.qattara())
    c=workspace_config(land,experiment_config(Config(),drones=2,containers=2,panel_area=44),[64.,64.],192)
    c.base.layout={};c.plan=PlanConfig([PlanSegment([[20,80],[26,80]],width=2,height=-.2)])
    p=ComplexProject(Simulation(c),land);p.enable_shared();p.add_shared(ComplexTemplate.capture(c))
    return p


def test_two_platform_complexes_roundtrip_with_ground_and_save_in_motion(tmp_path):
    from tests.test_regional import advance
    p=platform_group()
    for u in p.units:u.simulation.request_landing()
    advance(p,lambda:all(u.simulation.logistics.field_ready and u.simulation.logistics.battery_platforms.ready for u in p.units))
    t=p.shared_site.core;load=t.scoop_load(22,80,20,radius=1.);t.deposit(150,150,load.mass,load=load)
    ground=t.height.copy();ids=p.inventory_ids();start=p.time
    p.move_window([112.,64.]);advance(p,lambda:p.transition.phase=='DRIVE')
    path=tmp_path/'moving.atsstate';save_state(path,p.active.simulation,p.landscape,complex_project=p)
    q=load_state(path).complex_project;assert q.report()==p.report()
    for _ in range(3000):
        if p.transition is None:break
        p.step();q.step()
    assert p.transition is None and q.report()==p.report()
    p.move_window([64.,64.]);advance(p,lambda:p.transition is None)
    np.testing.assert_array_equal(p.shared_site.core.height,ground)
    assert p.inventory_ids()==ids and p.time>start+100
    r=p.report();assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002 and abs(r['equipment_mass_error_kg'])<1e-7
    assert all(u.simulation.logistics.battery_platforms.ready for u in p.units)
    p.validate()


def test_battery_platform_window_is_available_through_actual_ui():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host=MainWindow(configuration());host.timer.stop();host.show();app.processEvents()
    button=next(b for b in host.findChildren(QtWidgets.QPushButton) if b.text()=='Батарейные платформы')
    button.click();app.processEvents();window=host.battery_platforms_window
    assert window.isVisible() and window.table.rowCount()==5
    window.show_mounts.setChecked(True);assert 'base_battery_' in window.details.toPlainText()
    window.pack.click();assert all(c.mode=='PACK' for c in host.sim.logistics.battery_platforms.carriers.values())
    host.close();app.processEvents()


def test_recovered_pack_is_handled_into_its_rack_and_packing_waits(tmp_path):
    s=stationary();w=s.logistics;p=w.battery_platforms
    b=next(b for b in w.battery_service.modules() if b.location==L.BASE_STORAGE)
    w.inventory.move(b.id,L.BASE_STORAGE,'base',p.receiving_position(b.id),'QUEUED_FOR_CHARGE');p.receive(b.id)
    start=b.position;p.request('PACK');s.step()
    assert b.location==L.IN_TRANSFER and b.position!=start and not p.stowed
    path=tmp_path/'return.atsstate';save_state(path,s);q=load_state(path).simulation
    for _ in range(100):s.step();q.step()
    assert not p.returns and p.stowed and b.location==L.BASE_STORAGE
    assert np.allclose(b.position,p.slot_position(b.id)) and s.report()==q.report()
    assert abs(s.report()['energy_error_j'])<.002


def test_queued_battery_delivery_does_not_deadlock_packing():
    s=stationary();w=s.logistics;p=w.battery_platforms
    b=next(b for b in w.battery_service.modules() if b.location==L.BASE_STORAGE)
    job=w.new_job('DELIVER_EXCAVATOR_BATTERY',b,b.position,[100,100,0],1)
    assert job and b.reserved_by==job.id
    p.request('PACK');s.run(80)
    assert job.status=='CANCELLED' and b.reserved_by is None and p.stowed


def test_actual_overload_is_detected_on_its_own_carrier():
    from ats.base_layout import support_report
    s=stationary();s.drone.cargo=6000
    support=support_report(s.base,s.terrain,s.drones)
    id=s.logistics.battery_platforms.dock_carrier(s.drone.slot)
    assert not support['safe'] and next(r for r in support['platforms'] if r['id']==id)['overloaded']
