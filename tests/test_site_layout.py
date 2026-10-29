"""Physical solar deployment, dry footprints and growing spoil reservations."""
from copy import deepcopy
import numpy as np
import pytest

from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.simulation import Simulation
from ats.checkpoint import save_state,load_state
from ats.site_ground import footprint_safe,paint_rect,marine_mask
from ats.spoil import inside_distance,reserve


def solar_world(area=48.,crews=1):
    c=experiment_config(with_map_size(Config(),128),panel_area=area)
    c.logistics.solar={'installer_count':crews}
    c.logistics.self_solar=None  # historical detached-section mechanics stay readable
    s=Simulation(c);s.landing_requested=True
    assert s.state.value=='RUNNING',s.reason
    return s


def until(s,predicate,limit=6000):
    for _ in range(limit):
        if predicate():return
        s.step()
        assert s.state.value!='BLOCKED',s.reason
    pytest.fail(str(s.logistics.solar_array.report()))


def balanced(s):
    r=s.report(full=False)
    assert abs(r['mass_error_kg'])<1e-7
    assert abs(r['energy_error_j'])<1e-3
    s.logistics.inventory.validate()
    s.logistics.solar_array.validate(s.logistics)


def test_sections_have_real_mass_lanes_carriers_and_partial_power():
    s=solar_world(120);w=s.logistics;a=w.solar_array
    mass=sum(o.gross_mass for o in w.inventory.items.values())
    assert len(a.sections)==5 and len(a.crews)==1
    assert sum(o.mass_kg for o in w.inventory.items.values() if o.kind=='SOLAR_FRAME')==120
    assert w.generating_area()==20
    for q in a.sections:
        assert footprint_safe(s.terrain,q.rect,s.terrain.work_mask|s.terrain.dump_mask)
        assert q.carrier==0 and len(q.route)>=3
    until(s,lambda:20<w.generating_area()<120)
    assert not w.field_ready and w.generating_area()==44
    assert s.base.solar_power>0
    until(s,lambda:w.field_ready)
    assert s.time<=a.duration()+len(a.sections)*5
    assert w.generating_area()==120 and w.station_consumed>0
    assert all(w.inventory.items[q.id].holder_id==q.id for q in a.sections)
    assert sum(o.gross_mass for o in w.inventory.items.values())==pytest.approx(mass)
    balanced(s)


@pytest.mark.parametrize('phase',['LOAD_OUT','OUT','ASSEMBLE','BACK','FOLD','RETURN_LOAD','STORE'])
def test_exact_checkpoint_during_each_installation_and_return_phase(tmp_path,phase):
    s=solar_world();w=s.logistics;a=w.solar_array
    if phase in ('FOLD','RETURN_LOAD','STORE'):
        until(s,lambda:a.ready);a.request_fold()
    until(s,lambda:any(c.phase==phase for c in a.crews))
    s.step();path=tmp_path/'sections.atsstate';save_state(path,s)
    loaded=load_state(path).simulation
    assert loaded.report()==s.report()
    for _ in range(80):s.step();loaded.step()
    assert loaded.report()==s.report()
    for id,item in w.inventory.items.items():assert vars(loaded.logistics.inventory.items[id])==vars(item)
    balanced(s);balanced(loaded)


def test_cancel_outbound_and_reverse_partial_fold_without_teleportation():
    s=solar_world();w=s.logistics;a=w.solar_array
    until(s,lambda:a.crews[0].phase=='OUT');a.request_fold()
    ids=[q.id for q in a.sections];mass=sum(o.gross_mass for o in w.inventory.items.values())
    while not a.stowed:
        before={id:np.asarray(w.inventory.items[id].position) for id in ids}
        s.step()
        for id,p in before.items():assert np.linalg.norm(np.asarray(w.inventory.items[id].position)-p)<=a.settings['speed_m_s']+1e-8
    assert w.generating_area()==20  # integrated roof remains available for recovery
    a.request_deploy();until(s,lambda:a.ready);a.request_fold()
    until(s,lambda:a.crews[0].phase=='FOLD');s.run(10)
    q=a.sections[a.crews[0].active];z=w.inventory.items[q.panel_ids[0]].position[2]
    a.request_deploy();s.step()
    assert abs(w.inventory.items[q.panel_ids[0]].position[2]-z)<.1
    until(s,lambda:a.ready)
    assert sum(o.gross_mass for o in w.inventory.items.values())==pytest.approx(mass)
    balanced(s)


def test_hot_panel_replacement_waits_for_physical_return_and_keeps_other_items(tmp_path):
    s=solar_world();w=s.logistics;a=w.solar_array;until(s,lambda:a.ready)
    objects={id:o for id,o in w.inventory.items.items() if o.kind not in ('SOLAR_PANEL','SOLAR_FRAME','SOLAR_INSTALLER')}
    assert s.set_panel_area(72)=='waiting'
    assert s.base.config.panel_area==48 and a.mode=='FOLD'
    path=tmp_path/'pending-panels.atsstate';save_state(path,s);other=load_state(path).simulation
    until(s,lambda:s.base.config.panel_area==72)
    other.run(int(s.time-other.time));assert other.report()==s.report()
    assert all(w.inventory.items[id] is item for id,item in objects.items())
    until(s,lambda:w.field_ready)
    other.run(int(s.time-other.time));assert other.report()==s.report()
    assert w.generating_area()==72
    balanced(s)


def test_water_under_edge_blocks_full_footprint_and_no_space_is_atomic():
    s=solar_world();t=s.terrain
    t.config.georeference={'vertical_datum':'MSL','marine':dict(shape=[2,2],spacing=[128,128],origin=[0,0],bits='8A==')};t.reference=t.reference.copy();t.reference[80,80]=-1
    xy=80*t.cell
    assert not footprint_safe(t,(xy-.5,xy-.5,2,2))
    assert footprint_safe(t,(72,72,2,2))
    before=s.report();items=dict(s.logistics.inventory.items)
    # A pending removal retains the installed hardware until it can return.
    with pytest.raises(ValueError):s.set_panel_area(999999)
    assert s.report()==before and s.logistics.inventory.items==items


def test_terrain_change_stops_installer_without_losing_load():
    s=solar_world();w=s.logistics;a=w.solar_array
    until(s,lambda:a.crews[0].phase=='OUT');c=a.crews[0]
    p=np.asarray(c.path[c.cursor]);paint_rect(s.terrain,s.terrain.design_work,(p[0]-2,p[1]-2,4,4))
    ids=set(w.inventory.items);mass=sum(o.gross_mass for o in w.inventory.items.values())
    until(s,lambda:bool(a.issue),limit=300)
    assert 'ROUTE_CHANGED' in a.issue and set(w.inventory.items)==ids
    pos=c.position.copy();s.run(10);np.testing.assert_array_equal(c.position,pos)
    assert sum(o.gross_mass for o in w.inventory.items.values())==pytest.approx(mass)
    balanced(s)


def test_chessboard_distance_matches_erosion_oracle():
    from ats.workfront import expand
    rng=np.random.default_rng(27)
    for mask in [rng.random((31,31))>.2,np.pad(np.ones((18,12),bool),3)]:
        inside=mask.copy();expected=np.zeros_like(mask,dtype=int)
        for _ in range(32):
            inside=~expand(~inside);inside[[0,-1],:]=False;inside[:,[0,-1]]=False
            expected+=inside
        np.testing.assert_array_equal(inside_distance(mask),expected)


def test_installer_footprint_rejects_excessive_diagonal_grade():
    s=solar_world();t=s.terrain
    t.height=(t.x+t.y)*np.tan(np.deg2rad(7.))
    assert not footprint_safe(t,(70.,70.,2.,2.),grade_deg=8.,relief_m=10.)
    t.height=t.x*np.tan(np.deg2rad(7.))
    assert footprint_safe(t,(70.,70.,2.,2.),grade_deg=8.,relief_m=10.)


def test_compact_spoil_reserves_capacity_and_grows_without_releasing_piles():
    from tests.test_workfront import canal
    s=canal();t=s.terrain;g=t._spoil_geometry
    info=t.dump_reserve_info
    assert info['sufficient'] and info['available_free_m3']>=info['required_free_m3']
    assert info['reserved_area_m2']<info['allowed_area_m2']*.7
    original=t.dump_mask.copy();cut=s._choose_trip(s.drone,unreserved=True).dig
    mass=t.scoop(*cut,40);point=t.dump_location(mass/s.config.terrain.density,near=cut)
    t.deposit(*point,mass);t.relax(30)
    occupied=t.height>t.reference+1e-10
    # Advance the requested front; all existing allocations remain protected.
    mask,cap,info=reserve(t,**g,old_mask=t.dump_mask,old_cap=t.dump_cap,fronts=[60.])
    assert np.all(mask[original]) and np.all(mask[occupied])
    assert np.all(cap>=t.dump_cap) and abs(t.mass_change)<1e-7
    assert not np.any(mask&t.design_work)


def test_spoil_multiple_fronts_incoming_load_and_bounded_no_space_result():
    from tests.test_workfront import canal
    s=canal();t=s.terrain;g=t._spoil_geometry
    one=reserve(t,**g,fronts=[0.])[2]
    both=reserve(t,**g,fronts=[0.,70.],carried_volume=10.)[2]
    assert both['required_free_m3']>one['required_free_m3']+10
    mask,cap,info=reserve(t,**g,blocked=np.ones_like(t.dump_mask),carried_volume=10.)
    assert not np.any(mask) and not info['sufficient'] and info['available_free_m3']==0


def test_replan_is_transactional_and_keeps_support_ids():
    s=solar_world();w=s.logistics;a=w.solar_array
    a.request_fold();ids=set(w.inventory.items);w.plan_solar_field()
    assert set(w.inventory.items)==ids
    a=w.solar_array;a.request_fold();before=deepcopy(a.report());rects=deepcopy(a.reserved_rects)
    original=s.terrain.design_work;s.terrain.design_work=np.ones_like(original)
    with pytest.raises(ValueError):w.plan_solar_field()
    s.terrain.design_work=original
    assert w.solar_array is a and a.report()==before and a.reserved_rects==rects


def test_legacy_water_field_load_warns_and_preserves_equipment(tmp_path):
    s=solar_world();w=s.logistics
    # Authentic aggregate model remains supported, without inventing supports.
    w.solar_array=None;s.config.logistics.solar=None;w.field_ready=True
    w.field_rect=(70.,70.,8.,6.)
    s.config.georeference={'vertical_datum':'MSL','marine':dict(shape=[2,2],spacing=[128,128],origin=[0,0],bits='8A==')}
    s.terrain.reference=s.terrain.reference.copy();s.terrain.reference[int(70/s.terrain.cell):int(78/s.terrain.cell),int(70/s.terrain.cell):int(78/s.terrain.cell)]=-1
    path=tmp_path/'legacy-sea.atsstate';ids=set(w.inventory.items);save_state(path,s)
    loaded=load_state(path).simulation
    assert loaded.logistics.generating_area()==0
    assert 'BLOCKED_SOLAR_SITE' in loaded.logistics.solar_site_issue
    assert set(loaded.logistics.inventory.items)==ids


def test_empty_base_can_commission_extra_panels_from_integrated_roof():
    c=experiment_config(with_map_size(Config(),64),panel_area=72)
    c.logistics.self_solar=None
    c.base.initial_charge=0.;c.start_hour=5.95;c.base.latitude_deg=0.;c.base.start_day_of_year=80
    c.logistics.excavator_enabled=False;c.logistics.container_count=0;c.logistics.drone_spare_sets=0
    s=Simulation(c);s.landing_requested=True;a=s.logistics.solar_array
    position=a.crews[0].position.copy();s.run(150)
    np.testing.assert_array_equal(a.crews[0].position,position)
    assert s.logistics.station_consumed==0
    until(s,lambda:a.ready,limit=4000)
    assert s.base.solar_power>0 and s.logistics.generating_area()==72
    balanced(s)


def test_physical_sections_roundtrip_and_resume_keep_ids_and_changed_soil(tmp_path):
    from tests.test_regional import region_group,advance
    p=region_group(count=2,soil=True,side=128,cell=1.)
    for u in p.units:
        u.simulation.config.logistics.self_solar=None
        u.simulation.set_panel_area(48)
    t=p.shared_site.core;load=t.scoop_load(22,40,30.,radius=1.);t.deposit(100,100,load.mass,load=load)
    ground=t.height.copy();ids=p.inventory_ids()
    p.move_window([96.,64.]);advance(p,lambda:p.transition.phase=='PACK')
    p.step();path=tmp_path/'physical-moving.atsstate';save_state(path,p.active.simulation,p.landscape,complex_project=p)
    resumed=load_state(path).complex_project
    assert resumed.report()==p.report()
    for _ in range(4000):
        if p.transition is None:break
        p.step();resumed.step()
    assert p.transition is None and resumed.transition is None
    assert p.report()==resumed.report() and p.inventory_ids()==ids
    p.move_window([64.,64.]);advance(p,lambda:p.transition is None)
    np.testing.assert_array_equal(p.shared_site.core.height,ground)
    assert p.inventory_ids()==ids and p.time>100
    report=p.report()
    assert abs(report['energy_error_j'])<.002 and abs(report['mass_error_kg'])<1e-7
    assert report['equipment_mass_error_kg']==pytest.approx(0,abs=1e-7)
    assert all(u.simulation.logistics.solar_array.ready for u in p.units)


def test_site_ui_draws_all_sections_with_bounded_qt_items_and_controls():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host=MainWindow(solar_world(48).config);host.timer.stop();host.site_button.click();view=host.site_window;view.timer.stop()
    count=len(host.map.items())
    assert 'секций 2' in view.details.text() and 'проезды' in view.details.text().lower()
    assert host.solar_planned.getData()[0] is not None
    view.fold.click();assert host.sim.logistics.solar_array.mode=='FOLD'
    view.deploy.click();assert host.sim.logistics.solar_array.mode=='DEPLOY'
    host.sim.set_panel_area(240);host.refresh(True)
    assert len(host.map.items())==count
    view.close();host.close();app.processEvents()


def test_equipment_dialog_request_does_not_replace_panels_before_return():
    s=solar_world();w=s.logistics;until(s,lambda:w.field_ready)
    proposed=deepcopy(s.config)
    from ats.equipment import set_panel_area
    set_panel_area(proposed.base.equipment,72)
    assert s.request_equipment(proposed)=='waiting'
    assert s.base.config.equipment.panel_area==48 and s.base.config.panel_area==48
    until(s,lambda:s.base.config.panel_area==72)
    balanced(s)


def test_cancel_relocation_before_first_section_is_folded_keeps_panels_deployed():
    s=solar_world();w=s.logistics;a=w.solar_array;until(s,lambda:a.ready)
    a.request_fold();until(s,lambda:a.crews[0].phase=='FETCH')
    assert w.base_pack_elapsed==0
    s.relocation_pending=s.base.position[:2]+[1.,0.]
    assert s.cancel_relocation() and a.mode=='DEPLOY'
    until(s,lambda:a.ready)
    assert w.generating_area()==48 and a.sections[0].open_fraction==1.
    balanced(s)


def test_panel_service_after_completed_excavation_needs_no_new_mission():
    from ats.plan import PlanConfig
    s=solar_world();w=s.logistics;until(s,lambda:w.field_ready)
    s.apply_plan(PlanConfig([]));s.landing_requested=True;s.step()
    assert s.state.value=='COMPLETED'
    assert s.set_panel_area(72)=='waiting' and s.state.value=='RUNNING'
    until(s,lambda:s.base.config.panel_area==72 and w.field_ready)
    assert s.state.value=='COMPLETED'
    balanced(s)
