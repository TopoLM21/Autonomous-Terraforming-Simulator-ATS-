"""Attached PV: powered reversible hinges, dry swept space and persistent carriers."""
from copy import deepcopy
import json
import zipfile
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.simulation import Simulation
from ats.checkpoint import save_state,load_state
from ats.self_solar import SelfSolarArray,deployment_rectangles,parameters
from ats.inventory import Location as L
from ats.site_ground import rectangle_patch


def world(area=164,side=160):
    c=experiment_config(with_map_size(Config(),side),panel_area=area)
    c.base.layout={};c.logistics.excavator_enabled=False
    s=Simulation(c);s.request_landing()
    assert s.state.value=='RUNNING',s.reason
    assert isinstance(s.logistics.solar_array,SelfSolarArray)
    return s


def advance(s,predicate,limit=700):
    for _ in range(limit):
        if predicate():return
        s.step();assert s.state.value!='BLOCKED',s.reason
    pytest.fail(str(s.logistics.solar_array.report()))


def balanced(s):
    a=s.logistics.solar_array;a.validate(s.logistics)
    r=s.report();assert abs(r['energy_error_j'])<.002 and abs(r['mass_error_kg'])<1e-7
    assert sum(r['mass_kg'] for r in s.logistics.battery_platforms.loads().values())==pytest.approx(s.base.config.mass+sum(d.config.empty_mass+d.cargo for d in s.drones if d.docked))


def test_permanent_panels_real_hardware_and_reversible_own_drives():
    s=world();w=s.logistics;a=w.solar_array;q=a.sections[0]
    ids=set(w.inventory.items);m=s.base.config.mass
    assert len(a.sections)==1 and not a.crews and a.available_installers==0
    assert not any(o.kind=='SOLAR_INSTALLER' for o in w.inventory.items.values())
    module=a.module(w,q)
    assert module['dimensions_m']==[3.,6.,3.2]
    assert q.frame_mass_kg==524 and q.panel_mass_kg==576
    assert module['gross_mass_kg']==2670
    assert w.generating_area()==20
    s.run(100);assert q.supports==1 and 0<q.open_fraction<1 and w.generating_area()==20
    assert all(w.inventory.items[id].location==L.BASE_MOUNT for id in q.panel_ids)
    positions={id:w.inventory.items[id].position for id in q.panel_ids}
    advance(s,lambda:a.ready);assert w.generating_area()==164
    assert any(w.inventory.items[id].position!=positions[id] for id in q.panel_ids)
    assert q.actuation_energy_j>0
    consumed=q.actuation_energy_j;a.request_fold();s.run(70)
    assert 0<q.open_fraction<1 and q.connection==0 and q.supports==1
    a.request_deploy();advance(s,lambda:a.ready)
    a.request_fold();advance(s,lambda:a.stowed)
    assert q.actuation_energy_j>consumed and q.supports==q.open_fraction==q.connection==0
    assert set(w.inventory.items)==ids and s.base.config.mass==m
    balanced(s)


@pytest.mark.parametrize('seconds,folding',[(12,False),(90,False),(215,False),(2,True),(70,True),(215,True)])
def test_exact_checkpoint_at_partial_stages(tmp_path,seconds,folding):
    s=world();a=s.logistics.solar_array
    if folding:advance(s,lambda:a.ready);a.request_fold()
    s.run(seconds);path=tmp_path/'self-pv.atsstate';save_state(path,s);other=load_state(path).simulation
    assert other.report()==s.report()
    for _ in range(55):s.step();other.step()
    assert other.report()==s.report();balanced(s);balanced(other)


def test_low_power_pauses_and_recovers_without_consuming_fictional_energy():
    s=world();w=s.logistics;a=w.solar_array
    w.config.inverter_power_w=0.;s.run(10)
    q=a.sections[0];assert q.supports==0 and q.actuation_energy_j==0 and q.reason
    w.config.inverter_power_w=100000.;advance(s,lambda:a.ready)
    a.request_fold();s.run(20);before=q.open_fraction;energy=q.actuation_energy_j
    w.config.inverter_power_w=0.;s.run(10)
    assert q.open_fraction==before and q.actuation_energy_j==energy
    w.config.inverter_power_w=100000.;advance(s,lambda:a.stowed);balanced(s)


def test_full_swept_space_not_only_centre_and_retry_after_ground_change():
    s=world();w=s.logistics;a=w.solar_array;q=a.sections[0]
    rect=q.rect;patch=rectangle_patch(s.terrain,rect);row,col=patch[0].start,patch[1].start
    h=s.terrain.height[row,col];s.terrain.height[row,col]+=1.;s.terrain.revision+=1
    s.run(3);assert q.id in a.invalid_sections and q.supports==0 and q.reason
    s.terrain.height[row,col]=h;s.terrain.revision+=1
    advance(s,lambda:a.ready)
    assert not a.invalid_sections;balanced(s)


def test_partial_platforms_generate_independently_and_docks_stay_live():
    s=world(320);w=s.logistics;a=w.solar_array
    assert len(a.sections)==3
    # One drive is obstructed after placement; other carriers still unfold.
    patch=rectangle_patch(s.terrain,a.sections[-1].rect)
    s.terrain.protected=np.zeros_like(s.terrain.height,dtype=bool)
    s.terrain.protected[patch]=True;s.terrain.revision+=1
    s.run(80);assert w.battery_platforms.ready and not w.field_ready
    w.request_tool(s.drone,'CONTAINER_HOOK');s.run(45)
    assert s.drone.attachment=='CONTAINER_HOOK' and not w.tool_changes
    s.run(120);assert 20<w.generating_area()<320
    d=s.drone;old=d.battery;d.battery=d.config.battery*.2;s._initial_energy-=old-d.battery;d.charge_requested=True;w.sync_batteries()
    advance(s,lambda:w.battery_service.drone_swap_count>0)
    assert not w.field_ready;balanced(s)


def test_hot_extension_preserves_carriers_and_attached_mass(tmp_path):
    s=world(44);w=s.logistics;a=w.solar_array;advance(s,lambda:a.ready)
    before={q.platform_id:deepcopy(a.module(w,q)['offset_m']) for q in a.sections};ids=[q.id for q in a.sections]
    assert s.set_panel_area(68)=='pending' or getattr(w,'pending_solar_panels',None) is not None
    advance(s,lambda:getattr(w,'pending_solar_panels',None) is None and w.solar_array.ready,1000)
    a=w.solar_array
    assert [q.id for q in a.sections]==ids
    assert {q.platform_id:a.module(w,q)['offset_m'] for q in a.sections}==before
    balanced(s)
    a.request_fold();advance(s,lambda:a.stowed)
    assert s.set_panel_area(0)=='applied'
    assert w.solar_array.sections[0].id==ids[0]
    assert w.solar_array.sections[0].frame_mass_kg==524 and w.generating_area()==0
    assert {q.platform_id:w.solar_array.module(w,q)['offset_m'] for q in w.solar_array.sections}==before
    balanced(s)


def test_own_drive_route_cost_matches_motion_and_requires_stowed_wings():
    s=world(68);w=s.logistics;a=w.solar_array;b=s.base
    advance(s,lambda:a.ready)
    target=b.position[:2]+[1.,0.];b.destination=target.copy()
    reason=b.tick(1.,12.,s.drones,[False]*len(s.drones),terrain=s.terrain)
    assert b.destination is None and 'солнечные' in reason
    a.request_fold();w.battery_platforms.request('PACK');advance(s,lambda:a.stowed and w.battery_platforms.stowed)
    before=b.consumed;cost=b.route_energy(target,s.terrain,s.drones);b.destination=target.copy()
    while b.destination is not None:b.tick(1.,12.,s.drones,[False]*len(s.drones),terrain=s.terrain)
    assert b.consumed-before==pytest.approx(cost,abs=.0001)
    assert cost>0
    w.sync_batteries();balanced(s)


def test_sea_under_edge_blocks_the_full_hinge_sweep_without_losing_panels():
    import base64
    from ats.site_ground import marine_mask
    s=world();w=s.logistics;a=w.solar_array;q=a.sections[0]
    shape=s.terrain.height.shape;bits=np.zeros(shape,dtype=np.uint8);patch=rectangle_patch(s.terrain,q.rect)
    row,col=patch[0].start+1,patch[1].start+1;bits[row,col]=1
    s.terrain.reference[row,col]=-1
    s.config.georeference={'source':{'crs':'LOCAL_METRES','latitude_deg':55.},'origin_m':[0.,0.], 'marine':dict(bits=base64.b64encode(np.packbits(bits).tobytes()).decode(),origin=[0.,0.],spacing=[s.terrain.cell]*2,shape=list(shape))}
    s.terrain.revision+=1;ids=set(w.inventory.items)
    s.run(5);assert q.id in a.invalid_sections and q.supports==0
    assert set(w.inventory.items)==ids and np.any(marine_mask(s.terrain)[patch])


def test_new_coastal_channel_uses_own_carriers_and_no_installers():
    from tests.test_workfront import canal
    s=canal(panels=100,slots=4,self_deploying=True)
    assert s.state.value=='RUNNING',s.reason
    assert isinstance(s.logistics.solar_array,SelfSolarArray)
    s.request_landing();advance(s,lambda:s.logistics.field_ready)
    balanced(s)


def test_cancelling_group_move_reverses_actual_partial_folding():
    from tests.test_battery_platforms import platform_group
    from tests.test_regional import advance as group_advance
    p=platform_group()
    for u in p.units:u.simulation.request_landing()
    group_advance(p,lambda:all(u.simulation.logistics.field_ready for u in p.units))
    ids=p.inventory_ids();p.move_window([112.,64.])
    group_advance(p,lambda:p.transition.phase=='PACK');p.step()
    for _ in range(60):p.step()
    positions=[u.simulation.base.position.copy() for u in p.units]
    angles=[[q.open_fraction for q in u.simulation.logistics.solar_array.sections] for u in p.units]
    assert any(0<f<1 for values in angles for f in values)
    assert p.cancel_window_move()
    assert [[q.open_fraction for q in u.simulation.logistics.solar_array.sections] for u in p.units]==angles
    for u,position in zip(p.units,positions):np.testing.assert_array_equal(u.simulation.base.position,position)
    group_advance(p,lambda:all(u.simulation.logistics.field_ready and u.simulation.logistics.battery_platforms.ready for u in p.units),700)
    assert p.inventory_ids()==ids and p.window_history[-1]['status']=='CANCELLED'
    assert all(u.simulation.config.georeference['origin_m']==[64.,64.] for u in p.units)
    r=p.report();assert abs(r['energy_error_j'])<.002 and abs(r['equipment_mass_error_kg'])<1e-7


def test_old_v8_array_is_not_replaced_or_repacked(tmp_path):
    c=experiment_config(with_map_size(Config(),128),panel_area=48);c.logistics.self_solar=None
    s=Simulation(c);s.request_landing();s.run(30)
    path=tmp_path/'v8.atsstate';save_state(path,s);before=s.report()
    with zipfile.ZipFile(path) as archive:files={n:archive.read(n) for n in archive.namelist()}
    manifest=json.loads(files['manifest.json']);manifest['version']=8
    for node in manifest['nodes']:
        if node.get('type')=='ats.logistics_config.LogisticsConfig':node['attrs'].pop('self_solar',None)
    files['manifest.json']=json.dumps(manifest).encode()
    with zipfile.ZipFile(path,'w') as archive:
        for name,value in files.items():archive.writestr(name,value)
    other=load_state(path).simulation
    assert other.config.logistics.self_solar is None and not isinstance(other.logistics.solar_array,SelfSolarArray)
    assert before==other.report()
    for _ in range(20):s.step();other.step()
    assert other.report()==s.report()


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):parameters({'leaf_pairs':5})
    with pytest.raises(ValueError):parameters({'drive_efficiency':2})


def test_hot_compact_layout_keeps_pv_identity_when_cargo_carrier_count_increases():
    c=experiment_config(with_map_size(Config(),160),panel_area=20)
    c.base.x=c.base.y=60.;c.logistics.excavator_enabled=False
    s=Simulation(c);s.request_landing();s.set_panel_area(44)
    w=s.logistics;a=w.solar_array;advance(s,lambda:a.ready)
    ids=[q.id for q in a.sections];carrier_ids=[q.platform_id for q in a.sections]
    offsets=[deepcopy(a.module(w,q)['offset_m']) for q in a.sections]
    from ats.base_layout import solve_layout
    from ats.equipment import set_base_battery_count
    equipment=deepcopy(s.base.config.equipment);set_base_battery_count(equipment,300)
    before=s.report();spec=solve_layout(s.config,w,batteries=equipment.batteries)
    pv=[row for row in spec['platforms'] if row.get('role')=='PV']
    assert [row['id'] for row in pv]==carrier_ids
    assert [row['offset_m'] for row in pv]==offsets
    assert spec['platform_count']>s.base.layout_spec['platform_count']
    assert [q.id for q in a.sections]==ids and [q.platform_id for q in a.sections]==carrier_ids
    assert [a.module(w,q)['offset_m'] for q in a.sections]==offsets
    assert [q['id'] for q in spec['platforms']]==list(range(1,spec['platform_count']+1))
    assert s.report()==before
    a.validate(w);assert abs(s.report()['energy_error_j'])<.002


def test_actual_ui_buttons_and_batched_unfolding_geometry(tmp_path,monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host=MainWindow(world().config);host.timer.stop();host.show();host.settings_window.show();app.processEvents()
    host.site_button.click();app.processEvents();dialog=host.site_window
    assert dialog.isVisible() and not dialog.replan.isVisible()
    assert 'монтажники не используются' in dialog.details.text()
    before=host.solar_active.getData();host.sim.request_landing();host.sim.run(90);host.refresh(True);dialog.refresh()
    after=host.solar_active.getData()
    assert before[0] is not None and after[0] is not None
    assert not np.array_equal(before[0],after[0],equal_nan=True)
    assert len(host.solar_installers.data)==0
    dialog.fold.click();assert host.sim.logistics.solar_array.mode=='FOLD'
    dialog.deploy.click();assert host.sim.logistics.solar_array.mode=='DEPLOY'
    host.close();app.processEvents()
