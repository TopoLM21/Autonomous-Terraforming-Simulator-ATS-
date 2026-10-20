"""Offline georeferencing, lazy patches and MSL physics regressions."""
import json
import subprocess
import sys
import numpy as np
import pytest
from ats.config import Config
from ats.landscape import ElevationRaster,Landscape,load_dem,qattara_dem,workspace_config,corridor_profile
from ats.plan import PlanConfig,PlanSegment,save_plan,load_plan
from ats.simulation import Simulation


def raster(level=10.,spacing=100.,size=5):
    return ElevationRaster(np.full((size,size),level),dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=spacing,dy=spacing,source='synthetic test plane'))


def test_bundled_real_data_has_provenance_and_qattara_below_sea():
    r=qattara_dem()
    assert r.height.shape==(241,301)
    assert r.metadata['vertical_datum']=='MSL'
    assert 'NOAA' in r.metadata['source'] and len(r.metadata['raw_csv_sha256'])==64
    x,y=r.to_metres(27.5,29.5)
    assert r.sample(x,y)<-80
    x,y=r.to_metres(28.,31.8)
    assert r.sample(x,y)<0  # Mediterranean bathymetry, not interpreted as water
    assert 1500<r.resolution_m[0]<1900
    np.testing.assert_allclose(r.to_source(*r.to_metres(28.5,30.5)),[28.5,30.5])
    with pytest.raises(ValueError):r.height[0,0]=0


def test_npz_and_asc_coordinate_conventions_nodata_and_datum(tmp_path):
    r=raster();p=tmp_path/'plane.npz';r.save(p)
    clone=load_dem(p);np.testing.assert_array_equal(clone.height,r.height)
    asc=tmp_path/'sample.asc'
    asc.write_text('ncols 3\nnrows 2\nxllcorner 100\nyllcorner 200\ncellsize 10\nNODATA_value -9999\n30 40 -9999\n10 20 30\n')
    with pytest.raises(ValueError,match='CRS'):load_dem(asc)
    with pytest.raises(ValueError,match='MSL'):load_dem(asc,crs='LOCAL_METRES')
    a=load_dem(asc,crs='LOCAL_METRES',vertical_datum='MSL')
    assert a.metadata['x0']==105 and a.metadata['y0']==205
    assert a.sample(0,0)==10 and a.sample(5,5)==25
    assert a.sample(20,0)==30  # zero-weight NoData neighbour must not poison a node
    with pytest.raises(ValueError,match='NoData'):a.sample(20,10)
    with pytest.raises(ValueError,match='границы'):a.sample(-1,0)
    asc.with_suffix('.json').write_text(json.dumps(dict(crs='LOCAL_METRES',vertical_datum='MSL')))
    assert load_dem(asc).sample(0,0)==10
    with pytest.raises(ValueError,match='MSL'):ElevationRaster(r.height,{**r.metadata,'vertical_datum':'ellipsoid'})


def test_sparse_storage_does_not_allocate_distant_or_unchanged_cells(tmp_path):
    # More than a trillion hypothetical fine cells; only 25 coarse DEM samples.
    land=Landscape(raster(spacing=100_000.))
    base,current=land.read_window([64.,64.],128)
    assert not land.tiles
    land.commit_window([64.,64.],current)
    assert not land.tiles
    current[4,4]-=.25;current[90,100]+=.25
    land.commit_window([64.,64.],current)
    assert len(land.tiles)==2 and land.report()['fine_storage_bytes']==2*64*64*8
    assert land.report()['soil_delta_m3']==pytest.approx(0,abs=1e-14)
    _,again=land.read_window([65.,65.],128)
    assert again[0,0]==9.75 and again[86,96]==10.25
    assert len(land.tiles)==2
    path=tmp_path/'world.atsregion';land.save(path)
    restored=Landscape.load(path)
    np.testing.assert_array_equal(restored.read_window([64.,64.],128)[1],current)
    np.testing.assert_array_equal(restored.read_window([300_000.,300_000.],128)[1],base)
    land.commit_window([64.,64.],base);assert not land.tiles


def test_geotiff_optional_import_has_msl_metres_and_correct_orientation(tmp_path):
    rio=pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    path=tmp_path/'plane.tif'
    with rio.open(path,'w',driver='GTiff',width=3,height=2,count=1,dtype='float32',crs='EPSG:32635',transform=from_origin(500000,3400000,10,10),nodata=-9999) as dst:
        dst.write(np.array([[30,40,-9999],[10,20,30]],dtype=np.float32),1)
    with pytest.raises(ValueError,match='MSL'):load_dem(path)
    r=load_dem(path,vertical_datum='MSL')
    assert r.sample(0,0)==10 and r.sample(5,5)==25
    assert r.metadata['x0']==500005 and r.metadata['y0']==3399985
    assert 29<r.metadata['latitude_deg']<32
    with pytest.raises(ValueError,match='NoData'):r.sample(20,10)


def test_sparse_overlapping_commit_preserves_other_edits_and_counts_remote_spoil():
    land=Landscape(raster(spacing=1000))
    _,a=land.read_window([0.,0.],64);a[0,0]-=1;land.commit_window([0.,0.],a)
    _,b=land.read_window([4.,4.],64);b[0,0]+=1;land.commit_window([4.,4.],b)
    assert land.read_window([0.,0.],64)[1][0,0]==9
    assert land.report()['soil_delta_m3']==0
    _,c=land.read_window([2000.,2000.],64);c[0,0]+=2;land.commit_window([2000.,2000.],c)
    assert land.report()['soil_delta_m3']==.125
    with pytest.raises(ValueError,match='кратно'):land.read_window([.1,0.],64)
    with pytest.raises(ValueError,match='границы'):land.read_window([3999.,3999.],64)


@pytest.mark.parametrize('level,mode',[(120.,'legacy'),(-120.,'legacy'),(120.,'logistics'),(-120.,'logistics')])
def test_absolute_elevation_excavation_has_local_flight_height_and_closed_ledgers(level,mode):
    land=Landscape(raster(level))
    c=workspace_config(land,Config(),[64.,64.],32.)
    c.plan=PlanConfig([PlanSegment([[15,16],[17,16]],height=level-.25)])
    if mode=='logistics':
        from ats.experiments import experiment_config
        c=experiment_config(c,drones=4)
    s=Simulation(c)
    assert s.state.value=='RUNNING',s.reason
    assert level+7<s.cruise(s.drone)<level+15
    assert s.terrain.sample(5,5)==level
    result=s.run(20000)
    assert result['state']=='COMPLETED',result['reason']
    assert abs(result['mass_error_kg'])<1e-7
    assert abs(result['energy_error_j'])<.002
    assert np.min(s.terrain.height)>=level-.25-1e-7
    assert result['excavated_mass_kg']>0
    land.commit_window(c.georeference['origin_m'],s.terrain.height)
    assert abs(land.report()['soil_delta_m3'])<1e-9


def test_natural_cliffs_do_not_avalanche_without_excavation():
    r=raster();h=r.height.copy();h[:,2:]+=100
    land=Landscape(ElevationRaster(h,r.metadata))
    c=workspace_config(land,Config(),[180.,64.],32.)
    s=Simulation(c);before=s.terrain.height.copy()
    s.terrain.relax(100)
    np.testing.assert_array_equal(s.terrain.height,before)


def test_plan_and_configuration_roundtrip_preserves_msl_and_dump(tmp_path):
    from ats.experiments import save_configuration,load_configuration
    c=workspace_config(Landscape(raster(80)),Config(),[64.,64.],32.)
    c.plan=PlanConfig([PlanSegment([[15,16],[17,16]],height=79.75)])
    c.validate()
    path=tmp_path/'plan.json';save_plan(path,c)
    clone=load_plan(path,Config())
    assert clone.georeference==c.georeference and clone.reference_height==c.reference_height
    np.testing.assert_array_equal(Simulation(c).terrain.target,Simulation(clone).terrain.target)
    path=tmp_path/'config.json';save_configuration(path,c)
    loaded=load_configuration(path)
    assert loaded.georeference==c.georeference
    assert loaded.drone.dig_rate_kg_s==6


def test_regional_corridor_is_absolute_coarse_and_does_not_create_fine_tiles():
    land=Landscape(raster(10.))
    p=corridor_profile(land.raster,[[10,10],[110,10],[110,110]],target_msl=-5,width_m=20,spacing_m=10)
    expected_area=20*15+15**2/np.tan(np.deg2rad(35))
    assert p['estimated_cut_m3']==pytest.approx(200*expected_area)
    assert p['ground_msl']==[10]*21 and not land.tiles
    with pytest.raises(ValueError,match='границы'):corridor_profile(land.raster,[[10,10],[500,500]])
    with pytest.raises(ValueError,match='положительной'):corridor_profile(land.raster,[[10,10],[10,10]])


def test_offline_cli_import_and_save_sparse_region(tmp_path):
    dem=tmp_path/'dem.npz';raster(10.).save(dem)
    output=tmp_path/'report.json';world=tmp_path/'world.atsregion'
    command=[sys.executable,'-m','ats','--headless','--mode','legacy','--dem',str(dem),'--work-origin','64','64','--map-size','32','--goal-msl','9.75','--steps','20000','--save-landscape',str(world),'--json',str(output)]
    result=subprocess.run(command,capture_output=True,timeout=90)
    assert result.returncode==0,result.stderr.decode(errors='replace')
    report=json.loads(output.read_text(encoding='utf-8'))
    assert report['state']=='COMPLETED' and report['georeference']['vertical_datum']=='MSL'
    assert report['landscape']['modified_tiles']>0
    assert abs(Landscape.load(world).report()['soil_delta_m3'])<1e-9
