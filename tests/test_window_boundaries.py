"""Window edges, global canal completion and safe recovery from sector blocks."""
from copy import deepcopy
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import zipfile
import numpy as np
import pytest
from ats.channel import ChannelProject
from ats.landscape import Landscape,ElevationRaster
from ats.config import Config
from ats.experiments import experiment_config
from ats.simulation import Simulation,MissionState
from ats.complexes import ComplexProject,ComplexTemplate
from ats.checkpoint import save_state,load_state,Session
from ats.sector_progress import record_window,merge
from tests.test_coastal_compact import seasonal_config


def window(index=0,side=128,soil=False,tiny=False):
    from ats.soil import SoilProfile
    y,x=np.indices((601,601));h=np.full(x.shape,.02);h[x<80]=-.02
    h[(x-520)**2+(y-300)**2<56**2]=-.02
    r=ElevationRaster(h,dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=1.,dy=1.))
    channel=ChannelProject.create(r,[[50,300],[540,300]],width_m=1,depth_m=.05,side_m=side)
    land=Landscape(r,cell_size=1.,soil_profile=SoilProfile.qattara() if soil else None);land.channel_project=channel.to_data()
    c=experiment_config(Config(),drones=2,containers=2,landing_slots=2)
    c.logistics.self_solar=None;c.logistics.excavator_enabled=False;c.logistics.field_deployment_seconds=4.
    if tiny:
        for d in c.fleet.models:
            for b in d.equipment.batteries:b.capacity_kwh=.0001
    origin=np.floor(channel.origin(r,index)).tolist()
    s=Simulation(channel.configuration(land,c,index=index,origin=origin))
    assert s.state==MissionState.RUNNING,s.reason
    p=ComplexProject(s,land);p.enable_shared()
    return p,channel


def finish_visible(p):
    """Conservative earthwork fixture; actual transfers are tested separately."""
    s=p.active.simulation;t=s.terrain;f=t.front;f.stationed=True;f.revision=-1
    for _ in range(20):
        f.update()
        for k in f.active_indices.copy():
            x,y=t.x.ravel()[k],t.y.ravel()[k]
            load=t.scoop_load(x,y,1e8,radius=t.cell*.49) if t.soil is not None else None
            mass=load.mass if load is not None else t.scoop(x,y,1e8,radius=t.cell*.49)
            if not mass:continue
            volume=load.volume if load is not None else mass/s.config.terrain.density
            dump=t.dump_location(volume,near=(x,y));assert dump is not None
            t.deposit(*dump,mass,load=load)
        s._terrain_active=False;record_window(p)
        if t.remaining<=s.config.terrain.completion_volume:return
        assert f.advance_sector()
        f.stationed=True;f.revision=-1
    pytest.fail('Visible fixture did not finish')


def travel(p,section,limit=2500,origin=None):
    p.move_window(origin,section=section)
    for _ in range(limit):
        if p.transition is None:return
        p.step();assert not p.blocked,p.report()
    pytest.fail(str(p.transition.report()))


@pytest.mark.parametrize('soil',[False,True])
def test_partial_window_waits_and_resumes_same_global_sector_without_losing_progress(tmp_path,soil):
    p,ch=window(soil=soil);s=p.active.simulation;finish_visible(p);p.step()
    f=s.terrain.front;r=f.report()
    assert s.state==MissionState.RUNNING and s.base_move_wait=='NEEDS_NEXT_WINDOW'
    assert r['visible_part_complete'] and not r['global_sector_complete'] and r['needs_next_window']
    assert r['global_sector_start_m']==0 and r['global_sector_end_m']==200 and r['sector_end_m']<100
    original=s.terrain.height.copy();origin=s.config.georeference['origin_m'].copy();ids=p.inventory_ids()
    for _ in range(10):p.step()
    assert s.time==11 and s.base_move_wait=='NEEDS_NEXT_WINDOW'
    path=tmp_path/'edge.atsstate';save_state(path,s,landscape=p.landscape,complex_project=p)
    q=load_state(path).complex_project
    assert q.active.simulation.report()==s.report()
    travel(q,1);u=q.active.simulation;r=u.terrain.front.report()
    assert r['sector_id']==0 and r['global_sector_end_m']==200 and u.terrain.remaining>0
    np.testing.assert_array_equal(q.landscape.read_window(origin,len(original),cell_size=s.terrain.cell)[1],original)
    travel(q,0)
    np.testing.assert_array_equal(u.terrain.height,original)
    assert q.inventory_ids()==ids
    assert abs(q.report()['mass_error_kg'])<1e-7 and abs(q.report()['energy_error_j'])<.002


def test_finished_fragments_merge_until_true_project_end_and_survive_region_save(tmp_path):
    p,ch=window();s=p.active.simulation
    for index in range(ch.section_count):
        finish_visible(p);p.step()
        spans=merge(p.landscape.channel_project['finished_spans'])
        if index<ch.section_count-1:
            assert s.state==MissionState.RUNNING and s.base_move_wait=='NEEDS_NEXT_WINDOW'
            assert not s.terrain.front.report()['project_profile_complete']
            if index>=2:assert spans[0][0]==0 and spans[0][1]>=200
            travel(p,index+1)
        else:
            assert s.state==MissionState.COMPLETED
            assert s.terrain.front.report()['global_sector_complete']
            assert s.terrain.front.report()['at_project_end']
            assert s.terrain.front.report()['project_profile_complete']
            assert spans==[[0.,ch.length]]
    p.persist_window();path=tmp_path/'complete.atsregion';p.landscape.save(path)
    restored=Landscape.load(path);copy=ChannelProject.from_data(restored.channel_project)
    assert copy.finished_spans==[[0.,ch.length]] and copy.identity==ch.identity
    assert copy.completed==list(range(ch.section_count))


def test_true_end_inside_window_finishes_without_spurious_next_window():
    from tests.test_workfront import canal
    from tests.test_sectors import finish_sector
    s=canal();finish_sector(s);s.step()
    assert s.state==MissionState.COMPLETED,s.reason
    assert s.terrain.front.report()['global_sector_complete'] and not s.terrain.front.report()['needs_next_window']


def test_exact_two_hundred_metre_project_does_not_create_an_empty_final_sector():
    from tests.test_sectors import sector_config,finish_sector
    s=Simulation(sector_config(length=200.));finish_sector(s);f=s.terrain.front
    assert f.global_sector_end==200 and not f.advance_sector()
    s.step()
    assert s.state==MissionState.COMPLETED and f.sector_id==0
    assert f.report()['project_profile_complete'] and not f.report()['needs_next_window']


def test_group_retains_unfinished_global_sector_across_window_round_trip(tmp_path):
    p,ch=window();p.add_shared(ComplexTemplate.capture(p.active.simulation.config))
    ids=p.inventory_ids()
    for u in p.units:
        p.select(u.id);finish_visible(p)
    p.step()
    assert all(u.simulation.base_move_wait=='NEEDS_NEXT_WINDOW' for u in p.units)
    original=p.shared_site.core.height.copy()
    path=tmp_path/'group.atsstate';save_state(path,p.active.simulation,landscape=p.landscape,complex_project=p)
    q=load_state(path).complex_project
    travel(q,1)
    assert all(u.simulation.terrain.front.sector_id==0 for u in q.units)
    assert q.shared_site.core.remaining>0
    travel(q,0)
    np.testing.assert_array_equal(q.shared_site.core.height,original)
    assert q.inventory_ids()==ids and all(u.simulation.time==q.time for u in q.units)
    assert abs(q.report()['mass_error_kg'])<1e-7 and abs(q.report()['energy_error_j'])<.002


def test_existing_window_tolerance_does_not_turn_into_an_unfinishable_sector():
    from tests.test_workfront import canal
    from tests.test_sectors import finish_sector
    s=canal();finish_sector(s);t=s.terrain
    # A conservative two-cell edit leaves a small residual. The existing
    # window completion tolerance allows it and retains all its mass.
    source=int(np.argmax(np.where(t.dump_mask,t.height-t.reference,-np.inf)))
    destination=int(t.front.indices[len(t.front.indices)//2])
    volume=min(s.config.terrain.completion_volume*.5,(t.height-t.reference).ravel()[source]*t.area*.9)
    assert volume>t.front.sector_tolerance
    t.height.ravel()[source]-=volume/t.area;t.height.ravel()[destination]+=volume/t.area;t.revision+=1
    assert t.remaining<s.config.terrain.completion_volume and not t.front.complete
    s.step()
    assert s.state==MissionState.COMPLETED and s.terrain.front.report()['global_sector_complete']
    assert abs(s.report()['mass_error_kg'])<1e-7


def test_cross_section_cut_by_side_edge_is_not_declared_complete():
    from tests.test_sectors import sector_config,finish_sector
    c=sector_config();c.plan.segments[0].points=[[50.,10.],[480.,10.]]
    c.plan.segments[0].width=80.;c.plan.channel['start']=[50.,10.]
    s=Simulation(c);finish_sector(s);s._update_workfront();r=s.terrain.front.report()
    assert r['sector_end_m']==200 and r['window_limited']
    assert r['visible_part_complete'] and not r['global_profile_complete']
    assert s.base_move_wait=='NEEDS_NEXT_WINDOW'


def test_real_site_block_can_enter_safe_regional_transition(tmp_path):
    from tests.test_sectors import finish_sector
    p,ch=window(side=512);s=p.active.simulation;finish_sector(s)
    f=s.terrain.front;x,y=s.base.position[:2];iy,ix=(s.base.position[1::-1]/s.terrain.cell).astype(int)
    f.settings['base_sites']=[[float(f.chainage[iy,ix]),float(x),float(y)]]
    s._update_workfront()
    assert s.state==MissionState.BLOCKED and s.reason.startswith('BLOCKED_SECTOR_SITE:')
    path=tmp_path/'blocked.atsstate';save_state(path,s,landscape=p.landscape,complex_project=p)
    p=load_state(path).complex_project;s=p.active.simulation;ids=p.inventory_ids()
    travel(p,1)
    assert s.state==MissionState.RUNNING and p.inventory_ids()==ids and s.terrain.front.sector_id==1
    assert abs(p.report()['mass_error_kg'])<1e-7 and abs(p.report()['energy_error_j'])<.002


def test_range_block_is_relocatable_but_hard_block_of_any_unit_is_not():
    p,ch=window(index=3,tiny=True);s=p.active.simulation;s._update_workfront()
    assert s.state==MissionState.BLOCKED and s.reason.startswith('BLOCKED_SECTOR_RANGE:')
    p.move_window(section=4);assert p.transition is not None and s.state==MissionState.RUNNING
    assert p.cancel_window_move();s._update_workfront()
    assert s.state==MissionState.BLOCKED and s.reason.startswith('BLOCKED_SECTOR_RANGE:')
    p,ch=window();p.add_shared(ComplexTemplate.capture(p.active.simulation.config))
    a,b=[u.simulation for u in p.units];a.block('BLOCKED_SECTOR_SITE: no local site');b.block('BLOCKED_NO_ENERGY: hard failure')
    before=p.report()
    with pytest.raises(ValueError,match='устраните блокировку'):p.move_window(section=1)
    assert p.report()==before and p.transition is None


def test_custom_window_origin_is_respected_and_does_not_complete_canonical_section():
    from ats.regional import RegionalTransition
    p,ch=window();s=p.active.simulation;old=np.asarray(s.config.georeference['origin_m'])
    requested=old+[16.,0.];move=RegionalTransition.prepare(p,requested,1)
    assert move.new_origin==requested.tolist() and move.target.georeference['channel']['custom_window']
    report=s.report();report['state']='COMPLETED';report['georeference']=move.target.georeference
    original=deepcopy(ch.completed);ch.record(report)
    assert ch.completed==original and p.transition is None


def test_v11_completed_fragment_migrates_to_explicit_window_wait(tmp_path):
    p,ch=window();s=p.active.simulation;finish_visible(p)
    s.state=MissionState.COMPLETED;s.base_move_wait=''
    path=tmp_path/'new.atsstate';save_state(path,s,landscape=p.landscape,complex_project=p)
    old=tmp_path/'old-v11.atsstate'
    with zipfile.ZipFile(path) as a,zipfile.ZipFile(old,'w') as b:
        for name in a.namelist():
            data=a.read(name)
            if name=='manifest.json':
                v=json.loads(data);v['version']=11;data=json.dumps(v).encode()
            b.writestr(name,data)
    q=load_state(old).complex_project;u=q.active.simulation
    assert u.state==MissionState.RUNNING and u.base_move_wait=='NEEDS_NEXT_WINDOW'
    np.testing.assert_array_equal(u.terrain.height,s.terrain.height)
    assert u.time==s.time and q.inventory_ids()==p.inventory_ids()


def test_new_window_wait_and_canonical_sector_are_visible_in_ui():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p,ch=window();s=p.active.simulation;finish_visible(p);p.step()
    w=MainWindow(s.config,session=Session(s,landscape=p.landscape,complex_project=p));w.timer.stop()
    try:
        w.refresh(True);text=w.channel_status.text()
        assert 'Сектор 0–200' in text and 'видимая часть готова' in text and 'нужно соседнее окно' in text
        assert 'весь сектор готов' not in text
    finally:w.close();app.processEvents()


def test_full_45_carrier_footprint_preflight_rejects_small_overlap_before_moving(seasonal_config):
    from ats.landscape import qattara_dem
    from ats.soil import SoilProfile
    from ats.regional import RegionalTransition
    r=qattara_dem();land=Landscape(r,soil_profile=SoilProfile.qattara())
    route=[list(map(float,r.to_metres(*v))) for v in [(28.3,31.2),(27.5,29.5)]]
    ch=ChannelProject.create(r,route,side_m=512);land.channel_project=ch.to_data()
    s=Simulation(ch.configuration(land,seasonal_config));s.request_landing()
    p=ComplexProject(s,land);p.enable_shared();before=p.report()
    assert s.base.layout_spec['platform_count']==45
    accepted=RegionalTransition.prepare(p,None,1)
    assert accepted.goals and p.transition is None
    with pytest.raises(ValueError,match='не помещаются в перекрытии'):RegionalTransition.prepare(p,None,2)
    assert p.transition is None and p.report()==before
