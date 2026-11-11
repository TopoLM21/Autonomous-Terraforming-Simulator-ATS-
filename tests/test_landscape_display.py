"""Read-only DEM views use metre coordinates, MSL heights and real sea connectivity."""
import numpy as np
import pytest
from ats.landscape import ElevationRaster,qattara_dem
from ats.channel import ocean_nodes,is_ocean
from ats.landscape_display import (samples,overview,grid_key,marine_samples,colors,
    level_segments,profile,SEA_COLOR,SELECTED_COLOR,NODATA_COLOR)


def raster(height,dx=3.,dy=2.):
    return ElevationRaster(np.asarray(height),dict(crs='LOCAL_METRES',units='metres',
        vertical_datum='MSL',x0=100.,y0=200.,dx=dx,dy=dy))


def test_exact_zero_contour_uses_node_coordinates_and_distinct_axis_spacing():
    r=raster([[-1,1,3],[-1,1,3],[-1,1,3]])
    points=level_segments(r.height,0,(10,20,3,2))
    assert len(points)==4
    np.testing.assert_array_equal(points[:,0],11.5)
    np.testing.assert_array_equal(np.sort(points[:,1]),[20,22,22,24])


def test_contours_do_not_bridge_nodata_or_invent_zero_for_empty_cells():
    r=raster([[-1,1,1],[-1,1,np.nan],[-1,1,1]])
    points=level_segments(r.height,0,(0,0,3,2))
    assert len(points)==4 and np.all(points[:,0]==1.5)
    assert level_segments(np.full((5,7),np.nan),0,(0,0,3,2)).shape==(0,2)
    assert not len(level_segments(r.height,10,(0,0,3,2)))


def test_samples_keep_bilinear_interpolation_but_leave_missing_support_as_gaps():
    r=raster([[0,3,6],[4,np.nan,10],[8,11,14]])
    x=np.array([0.,6.,0.,6.,3.,1.,-1.,np.nan]);y=np.array([0.,4.,4.,0.,2.,1.,0.,0.])
    z=samples(r,x,y)
    np.testing.assert_allclose(z[:4],r.sample(x[:4],y[:4]))
    assert np.isnan(z[4:]).all()
    assert float(samples(r,0.,0.))==0


def test_overview_is_bounded_and_zoom_restores_native_detail_without_changing_dem():
    y,x=np.mgrid[:401,:601];r=raster(x+2*y)
    original=r.height.copy();data,frame=overview(r,max_side=32)
    assert data.shape==(32,32)
    xs=frame[0]+np.arange(32)*frame[2];ys=frame[1]+np.arange(32)*frame[3]
    np.testing.assert_allclose(data,r.sample(xs[None,:],ys[:,None]))
    key=grid_key(r,[[30,45],[40,50]]);detail,frame=overview(r,key,max_side=32)
    x0,x1,y0,y1=key
    np.testing.assert_array_equal(detail,r.height[y0:y1+1,x0:x1+1])
    assert frame==(x0*3.,y0*2.,3.,2.)
    np.testing.assert_array_equal(r.height,original);assert not r.height.flags.writeable


def test_geographic_overview_has_no_artificial_nodata_at_northern_eastern_edges():
    small=ElevationRaster(np.arange(35.).reshape(5,7),dict(crs='EPSG:4326',units='degrees',
        vertical_datum='MSL',x0=27.,y0=30.,dx=.01,dy=.02))
    assert small.extent_m[0]<6*small.resolution_m[0]  # actual cancellation case
    for r,size in ((qattara_dem(),37),(small,3)):
        data,frame=overview(r,max_side=size)
        assert data.shape==(size,size) and np.isfinite(data).all()
        np.testing.assert_allclose(data[[0,-1]][:,[0,-1]],r.height[[0,-1]][:,[0,-1]],atol=1e-10)


@pytest.mark.parametrize('height',[-30.,0.,40.])
def test_flat_dem_and_extreme_pan_are_displayable(height):
    r=raster(np.full((4,7),height))
    key=grid_key(r,[[1e9,2e9],[-2e9,-1e9]])
    data,frame=overview(r,key)
    assert min(data.shape)>=2 and np.isfinite(frame).all()
    assert colors(data).shape==(*data.shape,3)
    assert not len(level_segments(data,0,frame))


def test_marine_colors_do_not_confuse_closed_dry_basin_with_sea_at_same_height():
    h=np.full((9,13),10.);h[:,:2]=-30.;h[3:6,6:9]=-30.
    r=raster(h);data,frame=overview(r);marine=marine_samples(r,ocean_nodes(r),data,frame)
    assert marine[4,0] and not marine[4,7]
    for iy,ix in ((4,0),(4,7),(8,12),(4,2)):
        assert bool(marine[iy,ix])==bool(is_ocean(r,ix*3.,iy*2.))
    rgb=colors(data,marine)
    np.testing.assert_array_equal(rgb[4,0],SEA_COLOR)
    assert not np.array_equal(rgb[4,7],SEA_COLOR)
    assert not np.array_equal(rgb[4,7],rgb[8,12])
    for mode in ('below','band'):
        rgb=colors(data,marine,mode=mode,level=-30.,thickness=2.)
        np.testing.assert_array_equal(rgb[4,0],SEA_COLOR)
        np.testing.assert_array_equal(rgb[4,7],SELECTED_COLOR)


def test_horizontal_selection_is_in_msl_with_explicit_full_band_thickness():
    h=np.array([[-22.,-21.,-20.,-19.,-18.,0.,np.nan]])
    rgb=colors(h,mode='below',level=-20.)
    assert np.all(rgb[0,:3]==SELECTED_COLOR) and not np.array_equal(rgb[0,3],SELECTED_COLOR)
    rgb=colors(h,mode='band',level=-20.,thickness=2.)
    assert np.all(rgb[0,1:4]==SELECTED_COLOR)
    assert not np.array_equal(rgb[0,0],SELECTED_COLOR)
    np.testing.assert_array_equal(rgb[0,-1],NODATA_COLOR)
    np.testing.assert_array_equal(colors(np.full((2,3),np.nan)),np.broadcast_to(NODATA_COLOR,(2,3,3)))


def test_vertical_profiles_follow_east_north_and_polyline_distance_in_metres():
    y,x=np.mgrid[:7,:9];r=raster(3*x+4*y-10)
    axis,z=profile(r,'x',position=3.)
    np.testing.assert_allclose(z,axis-4.)
    axis,z=profile(r,'y',position=6.)
    np.testing.assert_allclose(z,2*axis-4.)
    axis,z=profile(r,'route',points=[[0.,0.],[12.,0.],[12.,12.]])
    assert axis[-1]==24
    expected=np.where(axis<=12,axis-10,2+2*(axis-12))
    np.testing.assert_allclose(z,expected,atol=1e-12)


def test_profiles_have_nodata_gaps_and_do_not_mutate_a_route():
    r=raster([[0,1,2],[0,np.nan,2],[0,1,2]])
    axis,z=profile(r,'x',position=2.)
    np.testing.assert_allclose(z,[0,np.nan,2],equal_nan=True)
    points=[[0.,0.],[6.,4.]];before=[p.copy() for p in points]
    axis,z=profile(r,'route',points=points,max_points=5)
    assert len(axis)<=5 and np.isnan(z).any() and points==before
    assert not len(profile(r,'route',points=[])[0])
    assert not len(profile(r,'route',points=[[0,0],[0,0]])[0])
