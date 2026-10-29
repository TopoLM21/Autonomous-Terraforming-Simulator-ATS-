"""A rotated compact field must not collide with its empty enclosing corners."""
from copy import deepcopy
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.maps import with_map_size
from ats.simulation import Simulation
from ats.site_ground import polygon_bounds,rectangle_polygon,polygons_overlap
from ats.self_solar import deployment_polygons,place
from ats.battery_platforms import deployment_safe
from ats.checkpoint import Session,save_state,load_state


def rotated(heading=22.):
    c=experiment_config(with_map_size(Config(),160),panel_area=452)
    c.base.layout=dict(heading_deg=heading,auto_heading=0)
    c.logistics.excavator_enabled=False
    s=Simulation(c);s.request_landing()
    assert s.state.value=='RUNNING',s.reason
    return s


@pytest.fixture(scope='module')
def seasonal_config():
    from ats.season_power import design_season,QATTARA_LATITUDE
    c=experiment_config(with_map_size(Config(),64));c.base.latitude_deg=QATTARA_LATITUDE
    return design_season(c)[0]


def test_rotated_separating_axes_keep_clearance_and_real_collisions():
    from ats.base_layout import rotate_offset
    def box(y):return np.array([rotate_offset(dict(heading_deg=35),v) for v in [(-13,y-3.25),(13,y-3.25),(13,y+3.25),(-13,y+3.25)]])
    a,b=box(0),box(8.5)
    assert polygons_overlap(rectangle_polygon(polygon_bounds(a)),rectangle_polygon(polygon_bounds(b)))
    assert not polygons_overlap(a,b) and not polygons_overlap(b,a,.25)
    assert polygons_overlap(a,box(6.)) and polygons_overlap(a,box(6.6),.25)
    assert not polygons_overlap(a,box(6.5))  # a shared boundary is not penetration


@pytest.mark.parametrize('heading',[-202.17292820374084,22.,45.])
def test_rotated_initial_grid_commissions_without_moving_manual_offsets(heading):
    s=rotated(heading);w=s.logistics;spec=s.base.layout_spec
    offsets={a['id']:list(a['offset_m']) for a in spec['platforms']}
    spaces=deployment_polygons(spec,s.base.position)
    assert any(polygons_overlap(rectangle_polygon(polygon_bounds(a)),rectangle_polygon(polygon_bounds(b)))
        and not polygons_overlap(a,b) for i,a in enumerate(spaces) for b in spaces[i+1:])
    s.run(235)
    assert w.field_ready and w.battery_platforms.ready,w.solar_site_issue
    assert not w.solar_array.invalid_sections and not w.battery_platforms.site_issues
    assert s.config.base.layout['heading_deg']==heading
    assert {a['id']:a['offset_m'] for a in s.base.layout_spec['platforms']}==offsets
    assert w.generating_area()==452 and abs(s.report()['energy_error_j'])<.002
    assert abs(s.report()['mass_error_kg'])<1e-7


def test_true_wing_overlap_is_rejected_at_delivery_and_runtime():
    from ats.compact_layout import move
    s=rotated();spec=deepcopy(s.base.layout_spec)
    pv=[a for a in spec['platforms'] if a['role']=='PV']
    move(pv[1],pv[0]['offset_m'])
    with pytest.raises(ValueError,match='пересекает собственную'):
        place(s.config,spec,s.base.position,s.terrain)
    s.base.layout_spec=spec;s.logistics.solar_array.audit(s.logistics)
    assert len(s.logistics.solar_array.invalid_sections)>=2
    # An actual service-apron/wing collision also stays prohibited.
    bess=next(a for a in spec['platforms'] if a['role']=='BESS')
    move(pv[0],bess['offset_m'])
    assert not deployment_safe(spec,s.base.position,s.terrain,solar_polygons=deployment_polygons(spec,s.base.position))


def test_rigid_geometry_cache_keeps_ground_checks_and_replanning_live(monkeypatch):
    import ats.site_ground as ground
    from ats.compact_layout import move
    from ats.site_ground import rectangle_patch
    s=rotated();w=s.logistics;a=w.solar_array;a.audit(w)
    calls=[];overlap=ground.polygons_overlap
    def counted(*args,**kwargs):calls.append(1);return overlap(*args,**kwargs)
    monkeypatch.setattr(ground,'polygons_overlap',counted)
    s.terrain.revision+=1;a.audit(w)
    assert not calls and not a.invalid_sections
    # A changed ground cell still stops unfolding without rescanning rigid rows.
    patch=rectangle_patch(s.terrain,a.sections[0].rect)
    s.terrain.height[patch][0,0]+=1.;s.terrain.revision+=1;a.audit(w)
    assert a.sections[0].id in a.invalid_sections and not calls
    s.terrain.height[patch][0,0]-=1.
    pv=[p for p in s.base.layout_spec['platforms'] if p['role']=='PV']
    move(pv[1],pv[0]['offset_m']);a.plan(w);a.audit(w)
    assert calls and len(a.invalid_sections)>=2


def test_real_coastal_seasonal_start_deploys_and_resumes_exactly(seasonal_config,tmp_path):
    from ats.channel import ChannelProject
    from ats.landscape import Landscape,qattara_dem
    from ats.soil import SoilProfile
    from ats.compact_layout import safe
    r=qattara_dem();land=Landscape(r,soil_profile=SoilProfile.qattara())
    points=[list(map(float,r.to_metres(*p))) for p in [(28.3,31.2),(27.5,29.5)]]
    project=ChannelProject.create(r,points,side_m=512)
    c=project.configuration(land,seasonal_config);s=Simulation(c);s.request_landing()
    assert s.state.value=='RUNNING',s.reason
    assert s.base.layout_spec['platform_count']==45 and safe(s.base.layout_spec,s.base.position,s.terrain)
    assert c.base.layout['heading_deg']==pytest.approx(-202.17292820374084)
    poses={a['id']:list(a['offset_m']) for a in s.base.layout_spec['platforms']}
    s.run(95);path=tmp_path/'coast.atsstate';save_state(path,s,landscape=land)
    q=load_state(path).simulation
    assert s.report()==q.report()
    for _ in range(140):s.step();q.step()
    assert s.report()==q.report()
    assert s.logistics.field_ready and s.logistics.battery_platforms.ready,s.logistics.solar_site_issue
    assert s.logistics.generating_area()==4866
    assert {a['id']:a['offset_m'] for a in s.base.layout_spec['platforms']}==poses
    r=s.report();assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<.002


@pytest.mark.parametrize('shared',[False,True])
def test_all_four_coastal_drones_take_off_resume_work_and_return(seasonal_config,tmp_path,shared):
    from ats.channel import ChannelProject
    from ats.landscape import Landscape,qattara_dem
    from ats.soil import SoilProfile
    from ats.complexes import ComplexProject
    from ats.self_solar import _polygon
    from ats.site_ground import point_in_polygon
    r=qattara_dem();land=Landscape(r,soil_profile=SoilProfile.qattara())
    points=[list(map(float,r.to_metres(*p))) for p in [(28.3,31.2),(27.5,29.5)]]
    c=ChannelProject.create(r,points,side_m=512).configuration(land,seasonal_config)
    s=Simulation(c);project=ComplexProject(s,land) if shared else None
    if shared:project.enable_shared()
    assert s.base.layout_spec['platform_count']==45 and len(s.drones)==4
    home=[d.position.copy() for d in s.drones];ids=set(s.logistics.inventory.items)
    array=s.logistics.solar_array;d=s.drones[3]
    # Drone 4 starts in the empty corner of PV 20's enclosing world box,
    # outside its actual rotated sweep. It must be allowed to take off there.
    section=next(q for q in array.sections if q.platform_id==20)
    polygon=_polygon(s.base.layout_spec,array.module(s.logistics,section),s.base.position)
    x,y,w,h=polygon_bounds(polygon)
    assert x<d.position[0]<x+w and y<d.position[1]<y+h
    assert not point_in_polygon(d.position,polygon)
    for drone in s.drones:
        assert array.obstacle_height(s.logistics,drone.position)<=drone.position[2]
        if shared:assert project.shared_site.obstacle_height(drone.position)<=drone.position[2]
    step=project.step if shared else s.step
    for _ in range(5):
        step();assert s.state.value=='RUNNING',s.reason
    assert all(not d.grounded for d in s.drones)
    path=tmp_path/'coastal-first-flight.atsstate'
    save_state(path,s,landscape=land,complex_project=project)
    loaded=load_state(path);other=loaded.simulation
    other_step=loaded.complex_project.step if shared else other.step
    for _ in range(150):
        step();other_step()
        assert s.state.value==other.state.value=='RUNNING',(s.reason,other.reason)
    assert all(d.trips>0 for d in s.drones) and s.report()==other.report()
    s.request_landing();other.request_landing()
    for _ in range(300):
        step();other_step()
        assert s.state.value==other.state.value=='RUNNING',(s.reason,other.reason)
        if all(d.grounded and d.cargo<1e-7 for d in s.drones):break
    assert all(d.docked and d.cargo==0 for d in s.drones)
    # Returning aircraft may receive another free charging pad; all four must
    # land on distinct real pads, including the initially affected fourth pad.
    assert len({d.slot for d in s.drones})==4
    for drone in s.drones:np.testing.assert_array_equal(drone.position,s.slot_position(drone.slot))
    assert sorted(tuple(d.position) for d in s.drones)==sorted(tuple(p) for p in home)
    assert s.report()==other.report() and set(s.logistics.inventory.items)==ids
    report=project.report() if shared else s.report()
    assert abs(report['mass_error_kg'])<1e-7 and abs(report['energy_error_j'])<.002
    if shared:assert abs(report['equipment_mass_error_kg'])<1e-7


@pytest.mark.parametrize('shared',[False,True])
def test_oriented_flight_obstacles_retain_real_bodies_wings_and_old_save_support(tmp_path,shared):
    from ats.complexes import ComplexProject
    from ats.site_ground import point_in_polygon
    from ats.self_solar import _polygon
    s=rotated(35.);w=s.logistics;a=w.solar_array
    project=ComplexProject(s) if shared else None
    if shared:project.enable_shared()
    def height(point):return project.shared_site.obstacle_height(point) if shared else a.obstacle_height(w,point)
    q=a.sections[0];module=a.module(w,q)
    polygon=_polygon(s.base.layout_spec,module,s.base.position)
    centre=np.mean(polygon,axis=0)
    assert height(centre)>=s.terrain.sample(*centre)+3.2
    # Keep the real swept wing and all its boundaries solid during unfolding.
    for point in [*polygon,(polygon[0]+polygon[1])/2]:
        assert point_in_polygon(point,polygon) and height(point)>s.terrain.sample(*point)
    for vertices in (polygon,polygon[::-1]):
        assert point_in_polygon(centre,vertices)
        assert not point_in_polygon(centre+[100,100],vertices)
    # Pre-fix saves may carry only rectangular acceleration structures. They
    # are derived caches, so loading must rebuild them without touching hardware.
    if shared:
        del project.shared_site._ground_polygons
    else:
        a._obstacle_key=(s.time,tuple(s.base.position));a._obstacle_grid={}
    path=tmp_path/'old-flight-cache.atsstate'
    save_state(path,s,complex_project=project);loaded=load_state(path)
    other=loaded.simulation
    actual=loaded.complex_project.shared_site.obstacle_height(centre) if shared else other.logistics.solar_array.obstacle_height(other.logistics,centre)
    assert actual==height(centre) and other.report()==s.report()
    assert set(other.logistics.inventory.items)==set(w.inventory.items)
    # Refining empty corners must not disable collision rejection for a real
    # neighbouring carrier. A descent into its solid footprint is still stopped.
    from ats.drone import DroneState
    d=s.drone;top=height(centre);d.position[:]=[*centre,top+1.]
    d.transition(DroneState.TAKEOFF);previous=d.position.copy()
    s._starts={d.identifier:d.position.copy() for d in s.drones}
    if shared:project.shared_site.start_tick()
    s._move(d,np.r_[centre,top-.1],DroneState.TO_DIG,1.)
    assert s.state.value=='BLOCKED' and 'Траектория' in s.reason
    np.testing.assert_array_equal(d.position,previous)


def test_actual_default_gui_coastal_start_and_rotated_corridor_drawing(seasonal_config,tmp_path):
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.landscape_ui import LandscapeDialog
    from ats.winter_power import ready_design
    from ats.compact_layout import active_polygons
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    initial,_=ready_design(deepcopy(seasonal_config),seasonal_config.energy_design)
    w=MainWindow(initial.config,session=Session(initial,ui={}));w.timer.stop()
    dialog=LandscapeDialog(w)
    try:
        dialog.example_corridor();assert dialog.project is not None
        dialog.start_button.click();w.timer.stop()
        assert w.running and w.sim.state.value=='RUNNING',w.sim.reason
        assert w.config.georeference['channel']['section']==0
        assert w.sim.base.layout_spec['platform_count']==45
        for _ in range(5):
            w.sim.step();assert w.sim.state.value=='RUNNING',w.sim.reason
        assert all(not d.grounded for d in w.sim.drones)
        w.running=False;w.show();w.refresh(True);app.processEvents()
        expected=np.array([p for polygon in active_polygons(w.sim) for p in [*polygon,polygon[0],(np.nan,np.nan)]])
        x,y=w.site_access.getData();np.testing.assert_allclose(np.column_stack((x,y)),expected,equal_nan=True)
        assert w.grab().save(str(tmp_path/'coastal-start.png'))
    finally:
        w.running=False;w.timer.stop();dialog.close();w.close();app.processEvents()
