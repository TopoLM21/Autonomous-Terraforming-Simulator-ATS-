"""Coast-to-depression routing, prepared SI excavation and sparse large windows."""
import json
import subprocess
import sys
import numpy as np
import pytest
from ats.config import Config
from ats.channel import ChannelProject,coastal_alignment,is_ocean,latitude_at
from ats.landscape import ElevationRaster,Landscape,qattara_dem,workspace_config
from ats.maps import with_map_size
from ats.simulation import Simulation


def coastal_dem(gap=15):
    # Exact synthetic geography: western sea, flat dry land, enclosed depression.
    y,x=np.indices((301,301),dtype=float)
    h=np.full(x.shape,.02);h[x<80]=-.02
    h[(x-(80+gap+55))**2+(y-150)**2<56**2]=-.02
    return ElevationRaster(h,dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=1.,dy=1.))


def test_trimmed_route_starts_ten_metres_on_land_and_ends_at_first_dry_negative_height():
    r=coastal_dem();p=ChannelProject.create(r,[[50,150],[150,150]],width_m=1,depth_m=.05,side_m=32)
    assert p.coast==pytest.approx([79.5,150],abs=1e-7)
    assert p.points[0]==pytest.approx([89.5,150],abs=1e-7)
    assert 94.5<p.points[-1][0]<94.6
    assert r.sample(*p.points[0])>0>r.sample(*p.points[-1])
    assert not is_ocean(r,*p.points[-1])
    land_start,_=coastal_alignment(r,[[82,150],[150,150]],10)
    np.testing.assert_allclose(land_start,p.points,atol=1e-7)


def test_user_bends_are_preserved_and_route_rejects_marine_or_unfinished_end():
    r=coastal_dem()
    route,_=coastal_alignment(r,[[50,150],[85,150],[90,152],[150,150]])
    assert any(np.linalg.norm(v-[90,152])<1e-8 for v in route)
    with pytest.raises(ValueError,match='Продлите'):coastal_alignment(r,[[50,150],[88,150]])
    with pytest.raises(ValueError,match='сухой впадине'):coastal_alignment(r,[[150,150],[170,150]])
    with pytest.raises(ValueError,match='отступ'):coastal_alignment(r,[[50,150],[150,150]],100)


def test_qattara_example_trims_to_basin_rim_and_allocates_no_fine_storage():
    r=qattara_dem();land=Landscape(r)
    raw=[[float(v) for v in r.to_metres(lon,lat)] for lon,lat in [(28.3,31.2),(27.5,29.5)]]
    p=ChannelProject.create(r,raw)
    assert 80000<p.length<90000 and p.section_count>100
    assert float(r.sample(*p.points[0]))>0>float(r.sample(*p.points[-1]))
    assert not land.tiles
    p.profile(r);assert not land.tiles


def test_end_uses_the_selected_basin_instead_of_an_earlier_small_hollow():
    r=coastal_dem(gap=80);h=r.height.copy();h[149:152,100:103]=-.02
    r=ElevationRaster(h,r.metadata)
    p=ChannelProject.create(r,[[50,150],[215,150]])
    assert 159.5<p.points[-1][0]<159.6


def test_local_profiles_clip_regional_edges_to_the_active_map():
    from ats.plan import PlanConfig,PlanSegment
    from ats.profiles import profiles,segments_for
    c=Config();c.plan=PlanConfig([PlanSegment([[-100,10],[100,10]],height=-.05)],channel=dict(start=[-100,10],tangent=[1,0]))
    s=Simulation(c);segments=segments_for(c)
    np.testing.assert_allclose(segments[0].points,[[0,10],[32,10]])
    cross=profiles(s.terrain,segments,direction='cross')[0]
    assert len(cross.current)>0 and np.min(cross.target)==pytest.approx(-.05)


@pytest.mark.parametrize('mode',['legacy','logistics'])
def test_prepared_channel_excavates_without_brush_and_saves_mass_and_project(tmp_path,mode):
    r=coastal_dem();land=Landscape(r)
    p=ChannelProject.create(r,[[50,150],[150,150]],width_m=1,depth_m=.05,side_m=32)
    config=Config()
    if mode=='logistics':
        from ats.experiments import experiment_config
        config=experiment_config(config,drones=4)
    c=p.configuration(land,config);s=Simulation(c)
    assert s.state.value=='RUNNING',s.reason
    assert c.plan.segments and c.plan.channel and s.terrain.remaining>0
    assert all(seg.height==-.05 and seg.width==1 for seg in c.plan.segments)
    xx=s.terrain.x+c.georeference['origin_m'][0]
    yy=s.terrain.y+c.georeference['origin_m'][1]
    marine=is_ocean(r,xx,yy)
    assert not np.any(s.terrain.excavation&marine)
    # Opening a plan creates no fine stored tiles and moves no sand.
    assert not land.tiles and s.terrain.mass_change==0
    result=s.run(20000)
    assert result['state']=='COMPLETED',result['reason']
    assert result['excavated_m3']>0
    assert abs(result['mass_error_kg'])<1e-7 and abs(result['energy_error_j'])<.002
    land.commit_window(c.georeference['origin_m'],s.terrain.height,cell_size=s.terrain.cell)
    # Completed canal cells still exclude base/spoil placement on a reopened site.
    reopened=p.configuration(land,Config())
    assert reopened.terrain.dump_rect==c.terrain.dump_rect
    p.record(result);p.record(result)
    assert p.completed==[0] and len(p.history)==1
    assert abs(land.report()['soil_delta_m3'])<1e-9
    land.channel_project=p.to_data();path=tmp_path/'channel.atsregion';land.save(path)
    loaded=Landscape.load(path);q=ChannelProject.from_data(loaded.channel_project)
    assert q.identity==p.identity and q.completed==[0] and q.history==p.history
    assert loaded.report()['channel']['target_msl']==-.05


def test_bending_back_along_the_coast_does_not_clip_later_route_legs():
    from types import SimpleNamespace
    from ats.plan import target_surface
    r=coastal_dem(gap=80)
    p=ChannelProject.create(r,[[50,150],[110,150],[110,220],[85,220],[215,150]],width_m=1,depth_m=.05)
    x,y=np.meshgrid([85.5,86.,86.5],[219.5,220.,220.5])
    h=r.sample(x,y)
    terrain=SimpleNamespace(height=h,reference=h.copy(),x=x,y=y,slope=np.tan(np.deg2rad(35)))
    target,_=target_surface(terrain,p.plan([0,0]))
    assert target[1,1]==pytest.approx(-.05)  # this leg is behind the first tangent


def test_oblique_wide_canal_preserves_sea_ground_with_stable_shore_banks():
    from types import SimpleNamespace
    from ats.channel import marine_window
    from ats.plan import target_surface
    r=coastal_dem();p=ChannelProject.create(r,[[75,80],[85,200],[150,150]],width_m=10,depth_m=1,side_m=64)
    origin=[60.,130.];axis=(np.arange(128)+.5)*.5;x,y=np.meshgrid(axis,axis)
    h=r.sample(x+origin[0],y+origin[1]);plan=p.plan(origin)
    plan.channel.update(marine=marine_window(r,origin,64),cell_size=.5)
    target,_=target_surface(SimpleNamespace(height=h,reference=h.copy(),x=x,y=y,slope=np.tan(np.deg2rad(35))),plan)
    marine=is_ocean(r,x+origin[0],y+origin[1])
    assert np.any(marine) and np.any(target<h-1e-9)
    np.testing.assert_array_equal(target[marine],h[marine])
    assert np.max(np.abs(np.diff(target,axis=0)))<=.5*np.tan(np.deg2rad(35))+1e-6
    assert np.max(np.abs(np.diff(target,axis=1)))<=.5*np.tan(np.deg2rad(35))+1e-6


@pytest.mark.parametrize('factor',[2,4])
def test_coarser_active_window_preserves_fine_edits_and_global_volume(factor):
    r=coastal_dem();land=Landscape(r);origin=[100.25,100.25]
    baseline,fine=land.read_window(origin,64);fine[0,0]-=1
    land.commit_window(origin,fine)
    coarse_base,coarse=land.read_window(origin,32,cell_size=.25*factor)
    before=land.report()['soil_delta_m3']
    coarse[0,0]-=.125;coarse[10,10]+=.125
    land.commit_window(origin,coarse,cell_size=.25*factor)
    assert land.report()['soil_delta_m3']==pytest.approx(before,abs=1e-12)
    _,again=land.read_window(origin,32,cell_size=.25*factor)
    np.testing.assert_allclose(again,coarse,atol=1e-13,rtol=0)
    # Idempotent save, including fine patterns hidden by the coarse average.
    land.commit_window(origin,again,cell_size=.25*factor)
    assert land.report()['soil_delta_m3']==pytest.approx(before,abs=1e-12)
    _,detail=land.read_window(origin,64)
    assert detail[0,0]-baseline[0,0]==pytest.approx(-1.125)
    assert detail[0,1]-baseline[0,1]==pytest.approx(-.125)


@pytest.mark.parametrize('side,cell',[(512,.5),(1024,1.)])
def test_larger_maps_bound_array_count_and_preserve_physical_bucket(side,cell):
    from ats.equipment import resolve_config
    original=resolve_config(Config());radius=original.fleet.models[0].grab_radius
    c=with_map_size(original,side);c.validate()
    assert c.terrain.n==1024 and c.terrain.cell_size==cell
    assert c.fleet.models[0].grab_radius==radius and c.fleet.models[0].payload==60


def test_same_cell_count_cannot_silently_rescale_saved_elevations():
    c=Config();c.terrain.n=1024;c.initial_height=np.zeros((1024,1024)).tolist()
    with pytest.raises(ValueError,match='рельеф'):with_map_size(c,512)


def test_latitude_comes_from_actual_base_and_changes_real_generated_energy():
    energies=[]
    for south in (30.,55.):
        r=ElevationRaster(np.full((3,3),.1),dict(crs='EPSG:4326',units='degrees',vertical_datum='MSL',x0=27.,y0=south,dx=.02,dy=.02))
        land=Landscape(r);c=workspace_config(land,Config(),[64.,64.],32.)
        from ats.plan import PlanConfig,PlanSegment
        c.plan=PlanConfig([PlanSegment([[15,16],[17,16]],height=-.05)])
        c.base.start_day_of_year=355;c.start_hour=12.
        s=Simulation(c)
        expected=latitude_at(c.georeference,*s.base.position[:2])
        s.config.base.latitude_deg=0  # cached/manual defaults may not replace location
        s.run(30)
        assert s.report()['latitude_deg']==pytest.approx(expected)
        assert abs(s.report()['energy_error_j'])<1e-4
        energies.append(s.base.generated)
    assert energies[0]>energies[1]*2


def test_headless_route_json_creates_prepared_plan_without_internet(tmp_path):
    dem=tmp_path/'dem.npz';coastal_dem().save(dem)
    route=tmp_path/'route.json';route.write_text(json.dumps(dict(coordinates='metres',points=[[50,150],[150,150]],width_m=1,depth_m=.05,side_m=32)))
    out=tmp_path/'report.json'
    result=subprocess.run([sys.executable,'-m','ats','--headless','--mode','legacy','--dem',str(dem),'--channel-route',str(route),'--steps','20000','--json',str(out)],capture_output=True,timeout=90)
    assert result.returncode==0,result.stderr.decode(errors='replace')
    r=json.loads(out.read_text(encoding='utf-8'))
    assert r['state']=='COMPLETED' and r['georeference']['channel']['depth_m']==.05
    assert r['plan_segments']>0 and r['excavated_m3']>0 and abs(r['mass_error_kg'])<1e-7
