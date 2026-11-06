"""Submerged two-layer boundary: conservative salt, unchanged net water."""
from copy import deepcopy
import math
import pytest
from ats.hydrology import Flooding
from tests.test_hydrology import flat


def lake(level=0.,salinity=80.,area=100.):
    m=flat(volume=(10+level)*area,area=area,flow=1e6)
    m.initialize_salt(salinity)
    return m


@pytest.mark.parametrize('level',[-9.,-2.,-1.9,-.5])
def test_no_exchange_through_dry_outlet_or_against_dominant_head(level):
    m=lake(level)
    assert m.exchange_rate==0


@pytest.mark.parametrize('salinity',[0.,80.])
def test_submerged_equal_level_exchange_approaches_sea_without_crossing_it(salinity):
    m=lake(salinity=salinity);volume=m.volume_m3
    before=abs(salinity-m.config.inflow_salinity_g_l)
    for _ in range(20):
        m.advance(21600.)
        assert m.volume_m3==volume and m.calculations==0
        assert abs(m.salinity_g_l-38.)<=before+1e-10
        assert min(salinity,38.)-1e-10<=m.salinity_g_l<=max(salinity,38.)+1e-10
        before=abs(m.salinity_g_l-38.)
    assert before<.01 and m.exchanged_m3>0
    assert m.imported_salt_kg>0 and m.exported_salt_kg>0
    assert abs(m.salt_balance_error)<1e-7
    m.validate()


def test_gate_and_explicit_disabled_exchange_preserve_salt_exactly():
    for setting in ('channel','factor','opening'):
        m=lake();mass=m.salt_mass_kg
        if setting=='channel':m.channel_open=False
        if setting=='factor':m.config.exchange_factor=0.
        if setting=='opening':m.config.opening=0.
        m.advance(365*86400.)
        assert m.salt_mass_kg==mass and m.exchanged_m3==0 and m.exchange_calculations==0


def test_gross_counterflows_share_the_channel_limit_and_solid_salt_is_retained():
    m=lake(salinity=300.);m.config.solubility_g_l=100.;m.config.flow_limit_m3_s=.01
    assert m.config.flow(m.level)+2*m.exchange_rate<=.01+1e-12
    initial=m.salt_mass_kg;solid=m.precipitated_salt_kg
    m.advance(3600.)
    assert m.salt_mass_kg<initial and m.precipitated_salt_kg<solid
    assert m.exported_salt_kg==pytest.approx(100*m.exchanged_m3)
    assert m.salt_mass_kg==pytest.approx(initial+38*m.exchanged_m3-m.exported_salt_kg)
    assert abs(m.salt_balance_error)<1e-7;m.validate()


def test_saline_source_above_effective_solubility_deposits_residue_conservatively():
    m=lake(salinity=100.);m.config.inflow_salinity_g_l=400.;m.config.solubility_g_l=150.
    m.advance(86400.)
    assert m.salinity_g_l<=150.+1e-10 and m.precipitated_salt_kg>0
    assert m.exported_salt_kg<=150*m.exchanged_m3+1e-7
    assert abs(m.salt_balance_error)<1e-5;m.validate()


def test_evaporating_lake_has_a_salt_equilibrium_above_sea_concentration():
    m=lake(level=-.0001,area=1000000.);q=m.config.flow(m.level)
    m.config.evaporation_mm_day=q/1000000*86400*1000
    assert m.config.evaporation_mm_day<=100
    geometry=m._exchange_geometry();low,high=38.,300.
    for _ in range(80):
        concentration=(low+high)/2
        net=38*q+(38-concentration)*m._exchange_rate(concentration,geometry)
        if net>0:low=concentration
        else:high=concentration
    equilibrium=(low+high)/2;m.initialize_salt(equilibrium)
    old_volume=m.volume_m3;m.advance(30*86400.)
    assert 38.<equilibrium<300.
    assert m.salinity_g_l==pytest.approx(equilibrium,abs=1e-6)
    assert m.volume_m3==pytest.approx(old_volume,abs=1e-6)
    assert m.inflow_m3>0 and m.exchanged_m3>0 and m.evaporated_m3>0
    assert abs(m.salt_balance_error)<1e-5;m.validate()


def test_old_v2_boundary_remains_one_way_and_unknown_composition_cannot_drive_exchange():
    m=lake();data=m.to_data();data['version']=data['state']['version']=2
    for key in ('exchanged_m3','exchange_calculations'):data['state'].pop(key)
    data['config'].pop('exchange_factor')
    q=Flooding.from_data(data)
    assert q.config.exchange_factor==0 and q.salt_mass_kg==m.salt_mass_kg
    q.config.exchange_factor=.25;q.salt_known=False;q.salt_origin='legacy_unknown'
    assert q.exchange_rate==0


def test_exchange_resume_is_exact_and_step_refinement_is_bounded():
    a=lake(level=0.,area=10000.);b=deepcopy(a);fine=deepcopy(a)
    a.advance(86400.);b.advance(43200.);b=Flooding.from_data(b.to_data());b.advance(43200.)
    assert a.to_data()==b.to_data()
    fine.advance(86400.,max_step=5400.)
    assert abs(a.salinity_g_l-fine.salinity_g_l)<1.
    assert abs(a.salt_balance_error)<1e-6 and abs(fine.salt_balance_error)<1e-6


def test_ui_exchange_setting_changes_only_the_boundary_and_survives_cancel():
    from PySide6 import QtWidgets
    from ats.hydrology_ui import SaltDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    owner=QtWidgets.QDialog();owner.model=lake();owner.refresh=lambda:None
    dialog=SaltDialog(owner);mass=owner.model.salt_mass_kg;volume=owner.model.volume_m3
    dialog.exchange.setValue(.5);dialog.apply()
    assert owner.model.config.exchange_factor==.5 and owner.model.salt_mass_kg==mass and owner.model.volume_m3==volume
    dialog=SaltDialog(owner);dialog.exchange.setValue(0.);dialog.reject()
    assert owner.model.config.exchange_factor==.5
    dialog.deleteLater();owner.close();app.processEvents()


def test_active_exchange_region_and_mission_snapshots_resume_exactly(tmp_path):
    import numpy as np
    from ats.landscape import Landscape,ElevationRaster
    from ats.config import Config
    from ats.simulation import Simulation
    from ats.checkpoint import save_state,load_state
    height=np.ones((3,3));height[1,1]=-10.
    land=Landscape(ElevationRaster(height,dict(crs='LOCAL_METRES',units='metres',vertical_datum='MSL',x0=0.,y0=0.,dx=10.,dy=10.)))
    land.flooding=m=lake();m.advance(21600.);assert m.exchanged_m3>0
    simulation=Simulation(Config());before=simulation.report()
    state=tmp_path/'mixing.atsstate';region=tmp_path/'mixing.atsregion'
    save_state(state,simulation,land);land.save(region)
    a=load_state(state);b=Landscape.load(region)
    assert a.simulation.report()==before
    for model in (m,a.landscape.flooding,b.flooding):model.advance(86400.)
    assert m.to_data()==a.landscape.flooding.to_data()==b.flooding.to_data()


def test_a_microscopically_shallow_lake_cannot_feed_a_full_depth_return_layer():
    m=flat(volume=1e-10,area=100.,bottom=-1e-12,flow=1e6);m.initialize_salt(80.)
    assert m.exchange_rate<1e-20
    m.advance(86400.)
    assert m.volume_m3==1e-10 and m.exchanged_m3<1e-15
    assert abs(m.salt_balance_error)<1e-20
