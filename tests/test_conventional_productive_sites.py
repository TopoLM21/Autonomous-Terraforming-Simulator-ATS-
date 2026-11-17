"""A useful stance must beat a tiny road lip without erasing that lip."""
import numpy as np
import pytest
from ats.checkpoint import save_state,load_state
from tests.test_conventional import fuel_tanks,balanced
from tests.test_conventional_siting import channel_site


def prepared_lip(clear_front_only=False):
    """Conservative prepared-ramp geometry, with one twelve-kilogram nibble.

    This is a constructed intermediate cut, not a replay of a user's save.
    All removed material is placed in the existing dump and recorded once.
    """
    s,c=channel_site(depth=1.,side=128.)
    c.truck_count=1;c.refuel_l_min=3000.
    f=s.install_conventional(c);fuel_tanks(f);t=s.terrain;e=f.excavator
    original=t.height.copy();entry=np.asarray(f.ramp['entry_m'])
    t.height[f.access_mask&(t.x<=entry[0]+5.)]=f.ramp_surface[f.access_mask&(t.x<=entry[0]+5.)]
    e.position[:2]=entry+[1.5,.5];e.position[2]=t.sample(*e.position[:2])
    patch=t.patch(*e.position[:2],f.config.excavator_reach_m)
    floor,allowed=f._bucket_geometry(e,patch)
    distance=(t.x[patch]-e.position[0])**2+(t.y[patch]-e.position[1])**2
    active=allowed&(distance<=f.config.excavator_reach_m**2)
    if clear_front_only:active&=t.x[patch]<=entry[0]+7.
    t.height[patch][active]=np.minimum(t.height[patch][active],floor[active])
    x,y=entry+[5.5,2.25];row,col=int(y/t.cell),int(x/t.cell)
    t.height[row,col]+=.03
    removed=float((original-t.height).sum()*t.area)
    mass=removed*t.config.terrain.density
    point=t.dump_location(removed)
    assert point is not None
    t.deposit(*point,mass)
    f.excavated_m3=f.delivered_m3=removed
    f.excavated_mass_kg=f.delivered_mass_kg=mass;e.excavated_m3=removed
    t.revision+=1
    e.target=f._available_target(e);e.state='WAIT_TRUCK'
    balanced(s,f)
    return s,f


def test_support_sized_road_band_uses_bulk_soil_without_moving_the_machine():
    s,f=prepared_lip(clear_front_only=True);e=f.excavator;t=s.terrain
    before=t.height.copy();position=e.position.copy();p=f._work_plan()
    ids,_=f._target_indices(e)
    along=p['along'][np.searchsorted(p['ids'],ids)]
    narrow=ids[along<=float(along.min())+1.]
    small=max(f.target_yield(e,(float(t.x.ravel()[i]),float(t.y.ravel()[i]),0.)) for i in narrow)
    assert small<=.02
    assert f.target_yield(e)>=.5
    np.testing.assert_array_equal(e.position,position)
    np.testing.assert_array_equal(t.height,before)
    balanced(s,f)


def test_productive_site_prefers_real_bucket_gain_and_drives_a_safe_route():
    s,f=prepared_lip();e=f.excavator;t=s.terrain
    initial=e.position.copy();height=t.height.copy();before=f.excavated_m3
    small=f.target_yield(e)
    assert small==pytest.approx(.0075,abs=1e-9)
    assert f.available_work_at_stance(e)>0.
    calculations=f.route_calculations
    assert f._choose_excavator_site(e,productive_only=True),e.reason
    assert e.state=='DRIVING' and f.route_calculations-calculations<=8
    assert all(f.segment_safe(a,b,e.radius,e) for a,b in zip([e.position[:2],*e.path[:-1]],e.path))
    np.testing.assert_array_equal(t.height,height)
    assert f.excavated_m3==before and not f._work_complete
    for _ in range(100):
        s.step()
        if e.state!='DRIVING':break
    assert e.state=='WAIT_TRUCK'
    assert np.linalg.norm(e.position-initial)>.5
    assert f.target_yield(e)>=max(2*small,small+.1*f.config.bucket_m3)-1e-9
    assert e.fuel_used_l>0.
    balanced(s,f)


def test_tiny_required_road_cut_remains_available_when_no_better_move_exists(monkeypatch):
    s,f=prepared_lip();e=f.excavator;truck=f.machines[1]
    point=e.position.copy();before=s.terrain.height.copy();small=f.target_yield(e)
    monkeypatch.setattr(f,'passable',lambda *a,**kw:False)
    assert not f._choose_excavator_site(e,productive_only=True)
    np.testing.assert_array_equal(e.position,point)
    np.testing.assert_array_equal(s.terrain.height,before)
    assert not f._work_complete and f._available_target(e) is not None
    old=f.excavated_m3
    assert f._cut(e,truck)
    assert f.excavated_m3-old==pytest.approx(small,abs=1e-9)
    assert truck.cargo_kg==pytest.approx(12.,abs=1e-7)
    balanced(s,f)


def test_productive_route_and_optional_search_cursor_resume_exactly(tmp_path):
    s,f=prepared_lip();e=f.excavator
    assert f._choose_excavator_site(e,productive_only=True)
    path=tmp_path/'productive-route.atsstate';save_state(path,s)
    q=load_state(path).simulation
    assert q.report()==s.report()
    for _ in range(15):s.step();q.step()
    assert q.report()==s.report()
    np.testing.assert_array_equal(q.terrain.height,s.terrain.height)
    balanced(s,f);balanced(q,q.conventional)


def test_unproductive_query_does_not_clear_a_loading_bay(monkeypatch):
    s,f=prepared_lip();e=f.excavator;calls=[];position=e.position.copy()
    monkeypatch.setattr(f,'_home_route',lambda *a,**kw:None)
    monkeypatch.setattr(f,'passable',lambda *a,**kw:False)
    monkeypatch.setattr(f,'_release_loading_bay',lambda *a:calls.append('release') or True)
    monkeypatch.setattr(f,'_prepare_productive_relocation',lambda *a,**kw:calls.append('prepare') or True,raising=False)
    assert not f._choose_excavator_site(e,productive_only=True)
    assert not calls
    np.testing.assert_array_equal(e.position,position)
    assert not f._work_complete
    balanced(s,f)


def test_cooperative_yield_does_not_start_excavator_motion(monkeypatch):
    s,f=prepared_lip();e=f.excavator;calls=[];position=e.position.copy();cargo=f.carried_mass_kg
    monkeypatch.setattr(f,'_home_route',lambda *a,**kw:None)
    monkeypatch.setattr(f,'_prepare_productive_relocation',lambda *a,**kw:calls.append(a[1]) or True,raising=False)
    assert f._choose_excavator_site(e,productive_only=True)
    assert calls and e.state!='DRIVING' and not e.path
    np.testing.assert_array_equal(e.position,position)
    assert f.carried_mass_kg==cargo
    balanced(s,f)
