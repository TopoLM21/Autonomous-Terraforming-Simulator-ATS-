"""Stationary equipment reads local live ground, retaining the sector forecast."""
from copy import deepcopy
import numpy as np
import pytest
from ats.checkpoint import save_state,load_state
from ats.simulation import Simulation
from ats.terrain import Terrain
from ats.channel_site import equipment_spec
from ats.canal_station import candidates,forecast_exclusion,work_exclusion
from tests.test_axis_stations import axis_config


def test_distant_height_changes_reuse_forecast_but_keep_exact_live_cuts(monkeypatch):
    import ats.canal_station as station
    s=Simulation(axis_config());t=s.terrain;t.__dict__.pop('_axis_station_cache',None)
    calls=[];original=station.slope_run
    def counted(terrain):calls.append(1);return original(terrain)
    monkeypatch.setattr(station,'slope_run',counted)
    forecast=forecast_exclusion(t);before=forecast.copy()
    for row,col in ((4,4),(500,500),(256,300)):
        t.height[row,col]-=.01;t.revision+=1
        assert forecast_exclusion(t) is forecast
        expected=before|(t.height<t.reference-1e-8)
        np.testing.assert_array_equal(work_exclusion(t),expected)
        patch=np.s_[row-2:row+3,col-2:col+3]
        np.testing.assert_array_equal(work_exclusion(t,patch=patch),expected[patch])
    assert len(calls)==1
    np.testing.assert_array_equal(forecast,before)
    assert not forecast.flags.writeable


def test_local_refill_and_untracked_revision_do_not_leave_a_stale_cut():
    s=Simulation(axis_config());t=s.terrain;patch=np.s_[490:502,490:502]
    forecast=forecast_exclusion(t)
    assert not np.any(forecast[patch])
    t.height[patch]-=.01;t.revision+=1
    assert np.all(work_exclusion(t,patch=patch))
    assert np.all(work_exclusion(t)[patch])
    t.height[patch]=t.reference[patch];t.revision+=1
    assert not np.any(work_exclusion(t,patch=patch))
    assert not np.any(work_exclusion(t)[patch])
    assert forecast_exclusion(t) is forecast


def test_old_and_next_faces_have_distinct_bounded_forecasts():
    s=Simulation(axis_config());t=s.terrain;f=t.front
    first=forecast_exclusion(t);f._set_sector(200.);f.support_guard_end=200.
    assert forecast_exclusion(t) is first
    second=forecast_exclusion(t,future=True)
    assert second is not first
    assert not first[256,300] and second[256,300]
    # Two queries for the same guard share a dynamic map, regardless of the
    # request's spelling; a new sector without a soil edit cannot reuse it.
    old=work_exclusion(t);future=work_exclusion(t,future=True)
    assert not old[256,300] and future[256,300]
    f.settings['sector_length_m']=1.
    for along in range(40,48):
        f._set_sector(float(along));f.support_guard_end=float(along)
        forecast_exclusion(t);forecast_exclusion(t,future=True)
        assert len(t._axis_station_cache['masks'])<=2


@pytest.mark.parametrize('change',['angle','reference','target','plan','project_end'])
def test_changed_forecast_inputs_update_protection(change):
    c=axis_config();c.plan.segments[0].height=-1.
    t=Terrain(c);t.front._set_sector(400.);t.front.support_guard_end=t.front.global_sector_end
    old=forecast_exclusion(t)
    if change=='angle':t.config.terrain.repose_deg=25.
    elif change=='reference':t.reference=t.reference+1.
    elif change in ('target','plan'):
        plan=deepcopy(t.config.plan)
        if change=='target':plan.segments[0].height=-1.5
        else:plan.segments[0].width+=20.
        t.set_plan(plan);t.config.plan=plan
    else:t.front.project_end+=1.
    assert np.any(forecast_exclusion(t)!=old)


def test_cache_is_derived_on_load_and_does_not_change_physical_reports(tmp_path):
    s=Simulation(axis_config());s.run(40);t=s.terrain
    expected=work_exclusion(t).copy();path=tmp_path/'cache.atsstate'
    save_state(path,s);q=load_state(path).simulation
    np.testing.assert_array_equal(work_exclusion(q.terrain),expected)
    for _ in range(20):s.step();q.step()
    assert s.report()==q.report()
    assert abs(s.report()['mass_error_kg'])<1e-7 and abs(s.report()['energy_error_j'])<.002


def test_stationary_pv_and_battery_audits_read_local_patches(monkeypatch):
    import ats.canal_station as station
    from ats.site_ground import rectangle_patch
    c=axis_config();c.logistics.self_solar={};c.base.equipment.panels[0].quantity=452
    t=Terrain(c);spec=equipment_spec(c);_,c.base.x,c.base.y=candidates(t,spec,200.)[0]
    s=Simulation(c);w=s.logistics;a=w.solar_array;t=s.terrain
    assert s.state.value=='RUNNING',s.reason
    a.audit(w);w.battery_platforms.audit_site()
    calls=[];original=station.work_exclusion
    def local(terrain,*,future=False,patch=None):
        calls.append(patch)
        assert patch is not None,'Stationary audit requested the whole dynamic map'
        return original(terrain,future=future,patch=patch)
    monkeypatch.setattr(station,'work_exclusion',local)
    t.height[-3,-3]-=.01;t.revision+=1
    a.audit(w);w.battery_platforms.audit_site()
    assert calls and not a.invalid_sections and not w.battery_platforms.site_issues
    q=a.sections[0];patch=rectangle_patch(t,q.rect)
    t.height[patch]-=.01;t.revision+=1
    a.audit(w)
    assert q.id in a.invalid_sections
