"""Snapshots preserve physical state, ownership and deterministic continuation."""
from copy import deepcopy
import json
import os
import subprocess
import sys
import zipfile
import numpy as np
import pytest
from ats.checkpoint import save_state,load_state
from ats.config import Config
from ats.drone import DroneState as DS
from ats.experiments import experiment_config
from ats.inventory import Location as L
from ats.simulation import Simulation


def identical_continuation(path, sim, steps=150, landscape=None):
    before=sim.report();save_state(path,sim,landscape)
    session=load_state(path);other=session.simulation
    assert other.report()==before and sim.report()==before
    np.testing.assert_array_equal(other.terrain.height,sim.terrain.height)
    assert other.terrain.config is other.config and other.planner.config is other.config
    if other.logistics:
        w=other.logistics
        assert w.sim is other and w.battery_service.world is w
        assert all(any(job is entry for entry in w.jobs) for job in w.active.values())
        assert all(port.pack is w.inventory.items[port.pack.id] for port,_ in w.battery_service.charging())
    for _ in range(steps):sim.step();other.step()
    assert other.report()==sim.report()
    np.testing.assert_array_equal(other.terrain.height,sim.terrain.height)
    np.testing.assert_array_equal(other.dust.field,sim.dust.field)
    return session


@pytest.mark.parametrize('legacy',[False,True])
def test_mid_flight_snapshot_and_continuation_are_exact(tmp_path,legacy):
    s=Simulation(Config() if legacy else experiment_config(Config()));s.run(25)
    assert any(not d.grounded for d in s.drones)
    identical_continuation(tmp_path/'flight.atsstate',s)


def test_snapshot_during_real_battery_swap_preserves_each_reserved_set(tmp_path):
    c=experiment_config(Config(),drones=1);c.fleet.models[0].initial_charge=.2
    s=Simulation(c);s.request_landing();s.run(10)
    assert s.logistics.battery_service.drone_swaps
    identical_continuation(tmp_path/'swapping.atsstate',s,80)


def test_snapshot_preserves_active_dust_stencils_and_scalar_types(tmp_path):
    from ats.scenarios import scenario
    s=Simulation(experiment_config(scenario('medium')));s.run(175)
    assert s.dust.field.max()>0 and s.dust._advection
    identical_continuation(tmp_path/'dust.atsstate',s,175)


def test_loaded_container_retains_job_inventory_aliases_and_soil(tmp_path):
    c=experiment_config(Config(),drones=1,containers=1,batch_courier=False,idle_digging=False)
    c.logistics.excavator_enabled=False
    s=Simulation(c);s.request_landing();w=s.logistics;container=w.containers[0]
    container.soil_kg=s.terrain.scoop(16,16,30)
    w.inventory.move(container.id,L.FILLED_QUEUE,'worksite',(16,16,0),'FULL_READY')
    for _ in range(300):
        s.step()
        if w.active and container.location==L.DRONE_CARGO:break
    assert container.location==L.DRONE_CARGO and container.soil_kg>0
    identical_continuation(tmp_path/'cargo.atsstate',s,300)


def test_snapshot_keeps_unfinished_tool_exchange_and_hot_added_hardware(tmp_path):
    c=experiment_config(Config(),drones=1,drone_spare_sets=0)
    c.logistics.excavator_enabled=False;c.logistics.container_count=0;c.logistics.direct_allowed=True
    s=Simulation(c);s.request_landing();s.logistics.request_tool(s.drone,'CONTAINER_HOOK');s.run(10)
    assert s.drone.state==DS.CHANGING_TOOL
    old=s.base.battery;s.set_base_battery_count(100)
    assert s.base.battery==old and len(s.logistics.base_battery_ids)==100
    identical_continuation(tmp_path/'equipment.atsstate',s,60)


def test_snapshot_of_a_collective_lift_keeps_member_identity_and_load(tmp_path):
    from ats.excavator import ExcavatorState as ES
    from tests.test_logistics import logistics_config
    c=logistics_config(count=4);c.fleet.charge_slots=4
    s=Simulation(c);w=s.logistics;e=w.excavator
    e.destination=e.position+np.array([3.,-1.,0.]);e.state=ES.WAITING_RELOCATION
    for _ in range(700):
        s.step()
        if w.group is not None and w.group.phase=='LIFT':break
    assert w.group is not None and w.group.phase=='LIFT'
    session=identical_continuation(tmp_path/'team.atsstate',s,0)
    other=session.simulation
    assert all(any(member is d for d in other.drones) for member in other.logistics.group.participants)
    for _ in range(200):s.step();other.step()
    assert other.report()==s.report()


def test_snapshot_includes_sparse_landscape_dem_front_and_panel_progress(tmp_path):
    from ats.landscape import Landscape
    from tests.test_channel import coastal_dem
    from tests.test_workfront import canal
    s=canal(panels=100,slots=4);s.run(30)
    region=Landscape(coastal_dem(80));region.corridors=[{'name':'test','points':[[1,2],[3,4]]}]
    region.tiles[(0,0)]=np.zeros((region.tile_cells,)*2);region.tiles[(0,0)][2,3]=.2
    session=identical_continuation(tmp_path/'landscape.atsstate',s,60,region)
    assert session.landscape.report()==region.report() and session.landscape.corridors==region.corridors
    assert not session.landscape.raster.height.flags.writeable
    assert session.simulation.terrain.front.terrain is session.simulation.terrain


def test_pending_legacy_equipment_still_installs_after_loading(tmp_path):
    s=Simulation();s.run(15);candidate=deepcopy(s.config)
    candidate.fleet.models[0].equipment.bucket.volume_m3*=1.2
    assert s.request_equipment(candidate)=='pending'
    identical_continuation(tmp_path/'pending.atsstate',s,300)


def rewrite_manifest(path, change):
    with zipfile.ZipFile(path) as archive:files={name:archive.read(name) for name in archive.namelist()}
    manifest=json.loads(files['manifest.json']);change(manifest)
    files['manifest.json']=json.dumps(manifest).encode()
    with zipfile.ZipFile(path,'w') as archive:
        for name,body in files.items():archive.writestr(name,body)


@pytest.mark.parametrize('damage',['version','class','reference','terrain','plain'])
def test_damaged_or_unknown_state_is_rejected(tmp_path,damage):
    path=tmp_path/'bad.atsstate';s=Simulation(experiment_config(Config()));save_state(path,s)
    if damage=='plain':path.write_text('not an ATS archive')
    else:
        def change(m):
            if damage=='version':m['version']=999
            elif damage=='reference':m['root']={'ref':10**9}
            elif damage=='class':
                node=next(n for n in m['nodes'] if n.get('kind')=='object');node['type']='builtins.eval'
            elif damage=='terrain':
                node=next(n for n in m['nodes'] if n.get('type')=='ats.terrain.Terrain');node['attrs']['height']=None
        rewrite_manifest(path,change)
    with pytest.raises(ValueError):load_state(path)


def test_failed_save_keeps_the_previous_complete_file(tmp_path):
    path=tmp_path/'atomic.atsstate';s=Simulation();save_state(path,s);original=path.read_bytes()
    s.unsupported_callback=lambda:None
    with pytest.raises(ValueError):save_state(path,s)
    assert path.read_bytes()==original
    assert load_state(path).simulation.time==0
    assert sorted(p.name for p in tmp_path.iterdir())==['atomic.atsstate']


def test_cli_continuation_matches_one_uninterrupted_run(tmp_path):
    first=tmp_path/'first.atsstate';second=tmp_path/'second.atsstate'
    def run(*args):
        result=subprocess.run([sys.executable,'-m','ats','--headless',*map(str,args)],capture_output=True,text=True,encoding='utf-8',timeout=45)
        assert result.returncode==0,result.stderr
        return json.loads(result.stdout)
    run('--steps',25,'--save-state',first)
    resumed=run('--load-state',first,'--steps',150,'--save-state',second)
    straight=run('--steps',175)
    assert resumed==straight and load_state(second).simulation.time==175


def test_real_gui_save_load_pauses_and_restores_the_same_world_and_draft(tmp_path,monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host=MainWindow(experiment_config(Config()));host.timer.stop();host.sim.run(25)
    from ats.plan import PlanSegment
    host.plan_editor.segments.append(PlanSegment([[10.,8.],[12.,8.]],name='Несохранённый мазок'))
    host.plan_editor.dirty=True;host.plan_editor.rebuild()
    sim=host.sim;before=sim.report();path=tmp_path/'gui.atsstate'
    host.speed.setCurrentIndex(host.speed.findData(2000));host.running=True
    host.save_session(path);host.timer.stop()
    assert host.running and host.sim is sim and sim.report()==before
    host.sim.run(50)
    host.calendar.setDate(host.calendar.date().addDays(10));host.start_hour.setValue(22)
    host.load_session(path);host.timer.stop()
    assert not host.running and host.sim is not sim and host.sim.report()==before
    assert host.speed.currentData()==2000 and host._session_path==path
    assert host.calendar.date().dayOfYear()==host.config.base.start_day_of_year
    assert host.start_hour.value()==host.config.start_hour
    assert host.plan_editor.dirty and host.plan_editor.segments[-1].name=='Несохранённый мазок'
    current=host.sim;running=host.running;bad=tmp_path/'bad.atsstate';bad.write_text('invalid')
    warnings=[];monkeypatch.setattr(QtWidgets.QMessageBox,'warning',lambda *args:warnings.append(args))
    host.load_session(bad);host.timer.stop()
    assert host.sim is current and host.running==running and warnings
    host.base_batteries.setValue(100);host.apply_base_batteries_button.click()
    assert host.sim is current and host.sim.time==25 and len(host.sim.logistics.base_battery_ids)==100
    host.close();app.processEvents()
