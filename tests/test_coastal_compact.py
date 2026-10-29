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
        w.running=False;w.show();w.refresh(True);app.processEvents()
        expected=np.array([p for polygon in active_polygons(w.sim) for p in [*polygon,polygon[0],(np.nan,np.nan)]])
        x,y=w.site_access.getData();np.testing.assert_allclose(np.column_stack((x,y)),expected,equal_nan=True)
        assert w.grab().save(str(tmp_path/'coastal-start.png'))
    finally:
        w.running=False;w.timer.stop();dialog.close();w.close();app.processEvents()
