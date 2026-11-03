"""Regional reservoir/rating physics, intentionally separate from soil work."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import replace
import math
import numpy as np
import pytest
from ats.hydrology import BasinCurve,FloodConfig,Flooding,regional_bed
from ats.landscape import Landscape
from tests.test_channel import coastal_dem


def flat(volume=0.,area=100.,bottom=-10.,evap=0.,flow=1.):
    curve=BasinCurve.create((3,3),[4],[bottom],[area])
    c=FloodConfig(10.,2.,1000.,flow_limit_m3_s=flow,evaporation_mm_day=evap,season_amplitude=0.)
    return Flooding(curve,c,'test',volume_m3=volume,initial_volume_m3=volume,channel_open=True)


def test_hypsometry_level_volume_area_are_inverse_and_use_partial_columns():
    curve=BasinCurve.create((3,3),[0,1,2],[-10.,-5.,-2.],[2.,3.,4.])
    assert curve.volume_at(-5)==10 and curve.volume_at(0)==43
    assert curve.area_at(-5)==5
    for h in np.linspace(-10,0,101):assert curve.level_at(curve.volume_at(h))==pytest.approx(h,abs=1e-12)
    assert curve.water_depth(0).sum()==0
    assert curve.water_depth(43).ravel()[2]==2


def test_constant_inflow_matches_analytic_storage_and_mass_balance():
    m=flat();m.advance(100)
    assert m.volume_m3==pytest.approx(100,abs=1e-8) and m.level==pytest.approx(-9)
    assert m.inflow_m3==pytest.approx(100) and m.evaporated_m3==0 and abs(m.balance_error)<1e-10
    m.validate()


def test_constant_evaporation_matches_level_and_cannot_remove_absent_water():
    m=flat(volume=500,evap=100.,flow=0.)
    m.advance(86400)
    assert m.level==pytest.approx(-5.1,abs=1e-9) and m.evaporated_m3==pytest.approx(10.)
    m.advance(100*86400)
    assert m.volume_m3==0 and m.evaporated_m3==pytest.approx(500)
    assert m.balance_error==pytest.approx(0,abs=1e-7)
    m.validate()


def test_channel_rating_is_bounded_by_outlet_and_stops_at_sea_level():
    c=FloodConfig(20.,5.,200_000.)
    assert c.flow(-120)==c.flow(-5)>0
    assert c.flow(-2)<c.flow(-5) and c.flow(0)==0
    assert replace(c,opening=0).flow(-120)==0
    assert replace(c,width_m=40).flow(-120)>c.flow(-120)
    assert replace(c,roughness=.06).flow(-120)==pytest.approx(c.flow(-120)/2)
    assert replace(c,length_m=800_000.).flow(-120)==pytest.approx(c.flow(-120)/2)
    assert replace(c,length_m=.001).flow(-120)<=20*math.sqrt(9.81)*(10/3)**1.5


def test_fill_approaches_sea_without_overshoot_or_creating_water():
    m=flat();m.advance(10000)
    assert -1e-5<m.level<=0 and m.volume_m3<=m.curve.capacity
    assert m.spilled_m3<1e-8
    assert abs(m.balance_error)<1e-7
    m.validate()


def test_no_flow_or_evaporation_skips_years_and_season_can_wake_a_balanced_lake():
    m=flat(volume=400,flow=0.);m.advance(365*86400.)
    assert m.calculations==0 and m.volume_m3==400 and m.skipped_seconds==365*86400
    m=flat(volume=400,evap=10.,flow=1.)
    m.config.season_amplitude=.5
    summer=m.config.evaporation(0);winter=m.config.evaporation(182.5*86400)
    assert summer>winter
    m.advance(86400);assert m.calculations>0


def test_closed_gate_lake_still_evaporates():
    m=flat(volume=500,evap=10);m.channel_open=False;m.advance(86400)
    assert m.volume_m3==pytest.approx(499,abs=1e-8) and m.inflow_m3==0


def test_dry_basin_with_insufficient_inflow_clamps_real_evaporation():
    m=flat(area=1000,evap=100,flow=.0001);m.advance(86400)
    assert m.volume_m3==0
    assert m.inflow_m3==pytest.approx(8.64) and m.evaporated_m3==pytest.approx(8.64)
    assert abs(m.balance_error)<1e-10


def test_coarse_bed_preserves_fine_edit_volume_without_mutating_landscape():
    land=Landscape(coastal_dem(gap=100));a=np.zeros((64,64));a[0,0]=-.4;land.tiles[8,8]=a
    # A separate finite address within this 300m raster, not the fine sea mask.
    original=land.raster.height.copy();bed,areas=regional_bed(land)
    assert float(((bed-original)*areas).sum())==pytest.approx(-.4*.25**2,abs=1e-12)
    np.testing.assert_array_equal(land.raster.height,original)


def test_basin_selection_excludes_the_sea_and_reads_current_height_tiles():
    land=Landscape(coastal_dem(gap=100));curve=BasinCurve.from_landscape(land,[230.,150.])
    cols=land.raster.height.shape[1];assert np.all(curve.indices%cols>80)
    assert curve.capacity>0 and np.all(curve.bottom<0)
    with pytest.raises(ValueError,match='сухой впадине'):BasinCurve.from_landscape(land,[50.,150.])


def test_zero_evaporation_at_seasonal_trough_does_not_freeze_future_weather():
    m=flat(volume=400,flow=0.,evap=10.)
    m.config.start_day=354.5;m.config.season_amplitude=1.
    assert m.config.evaporation(0)==0
    m.advance(30*86400.)
    assert m.volume_m3<400 and m.evaporated_m3>0 and m.calculations>0


def test_region_and_full_state_preserve_flooding_without_touching_earthworks(tmp_path):
    from ats.checkpoint import save_state,load_state
    from tests.test_regional import region_group
    p=region_group(count=1,soil=True);land=p.landscape
    land.flooding=flat();land.flooding.advance(123.)
    land.flooding.curve.shape=land.raster.height.shape
    path=tmp_path/'water.atsregion';land.save(path);q=Landscape.load(path)
    assert q.flooding.report()==land.flooding.report()
    assert q.flooding.config==land.flooding.config
    source=p.report();height=p.shared_site.core.height.copy();ids=p.inventory_ids()
    path=tmp_path/'mission.atsstate';save_state(path,p.active.simulation,land,complex_project=p)
    q=load_state(path).complex_project
    assert q.landscape.flooding.report()==land.flooding.report() and q.report()==source
    assert q.inventory_ids()==ids
    np.testing.assert_array_equal(q.shared_site.core.height,height)
    q.landscape.flooding.advance(100.);land.flooding.advance(100.)
    assert q.landscape.flooding.report()==land.flooding.report()
    assert q.report()==source and p.report()==source


def test_ui_instant_channel_advance_and_close_leave_earthworks_untouched():
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.config import Config
    from ats.landscape_ui import LandscapeDialog
    from ats.hydrology_ui import FloodingDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(Config());w.timer.stop();w.running=True
    owner=LandscapeDialog(w);owner.region=Landscape(coastal_dem(gap=100));owner.show_region()
    owner.points=[[50,150],[250,150]];owner.width.setValue(2);owner.depth.setValue(1);owner.project_side.setCurrentIndex(owner.project_side.findData(128))
    assert owner.prepare_channel()
    before=w.sim.report();water=FloodingDialog(owner);water.timer.stop();water.show();app.processEvents()
    try:
        assert not w.running and not water.model.channel_open
        water.inputs['evaporation_mm_day'].setValue(0)
        water.instant.click();assert water.model.channel_open
        water.day.click();assert water.model.volume_m3>0 and water.model.elapsed_seconds==86400
        assert water.water_image.image is not None
        assert w.sim.report()==before and not owner.region.tiles and not owner.region.soil_tiles
        water.close_gate.click();assert not water.model.channel_open
    finally:
        water.close();owner.close();w.timer.stop()
        assert w.running
        w.running=False;w.close();app.processEvents()


def test_constant_equilibrium_tracks_both_fluxes_without_numeric_recalculation():
    m=flat(volume=400,evap=10.)
    q=m.config.evaporation(0)*100;m.config.flow_limit_m3_s=q
    m.advance(365*86400.)
    assert m.calculations==0 and m.volume_m3==400
    assert m.inflow_m3==pytest.approx(q*365*86400) and m.evaporated_m3==m.inflow_m3
    m.config.opening=0;m.advance(86400)
    assert m.calculations>0 and m.volume_m3<400


def test_coarse_rebuild_uses_metadata_for_interior_fine_tiles(monkeypatch):
    from tests.test_landscape import raster
    from ats.tile_cache import TileVersion
    land=Landscape(raster(10.,1000.));a=np.full((64,64),-.1);land.tiles[8,8]=a
    def fail(*a,**k):raise AssertionError('unchanged block should not be read from disk')
    with monkeypatch.context() as m:
        m.setattr(TileVersion,'read',fail)
        bed,areas=regional_bed(land)
    assert float(((bed-land.raster.height)*areas).sum())==pytest.approx(-.1*64*64*.25**2,abs=1e-8)


def test_cached_curves_and_idle_steps_do_not_read_landscape(monkeypatch):
    m=flat(volume=100)
    def fail(*a,**k):raise AssertionError('time integration must not rebuild terrain')
    monkeypatch.setattr(BasinCurve,'from_landscape',fail)
    m.advance(100.)
    assert m.volume_m3>100


def test_numeric_step_refinement_and_resume_agree():
    curve=BasinCurve.create((3,3),[0,1,2],[-10.,-6.,-2.],[1000.,2000.,3000.])
    c=FloodConfig(10.,2.,1000.,flow_limit_m3_s=.01,evaporation_mm_day=5.,season_amplitude=.5)
    a=Flooding(curve,c,'test',channel_open=True)
    b=Flooding.from_data(a.to_data());d=Flooding.from_data(a.to_data())
    a.advance(100*86400.);b.advance(100*86400.,max_step=1800.)
    d.advance(50*86400.);d=Flooding.from_data(d.to_data());d.advance(50*86400.)
    assert d.report()==a.report()
    assert abs(a.volume_m3-b.volume_m3)/max(1,b.volume_m3)<.01
    assert abs(a.balance_error)<1e-7 and abs(b.balance_error)<1e-7


def test_shoreline_equilibrium_between_two_dem_terraces_does_not_chatter():
    curve=BasinCurve.create((3,3),[0,1],[-10.,-5.],[100.,100.])
    e=10*.001/86400
    c=FloodConfig(10.,2.,1000.,flow_limit_m3_s=150*e,evaporation_mm_day=10.,season_amplitude=0.)
    m=Flooding(curve,c,'test',volume_m3=490.,initial_volume_m3=490.,channel_open=True)
    m.advance(100*86400.)
    assert m.volume_m3==pytest.approx(500,abs=1e-8) and m.level==-5
    solves=m.calculations;m.advance(100*86400.)
    assert m.calculations==solves and m.volume_m3==pytest.approx(500,abs=1e-8)
    assert m.current_rates()[0]==m.current_rates()[1]
    assert abs(m.balance_error)<1e-7


def test_tiny_dry_pool_does_not_take_microscopic_turnover_steps(monkeypatch):
    m=flat(area=100,bottom=-1e-20,evap=5.,flow=1e-12)
    m.config.season_amplitude=1.;original=m.rates;calls=[0]
    def counted(*a):
        calls[0]+=1
        if calls[0]>10000:raise AssertionError('dry throughput must not force microscopic steps')
        return original(*a)
    monkeypatch.setattr(m,'rates',counted)
    m.advance(86400.)
    assert m.volume_m3==0 and m.calculations<=4
    assert m.evaporated_m3==pytest.approx(m.inflow_m3) and m.balance_error==pytest.approx(0,abs=1e-18)


@pytest.mark.parametrize('flow',[1e-13,1e-20])
def test_tiny_positive_flow_is_not_frozen_or_relabelled_as_evaporation(flow):
    m=flat(flow=flow)
    m.advance(86400.)
    assert m.volume_m3==pytest.approx(flow*86400,rel=1e-12,abs=1e-30)
    assert m.evaporated_m3==0 and abs(m.balance_error)<1e-25
