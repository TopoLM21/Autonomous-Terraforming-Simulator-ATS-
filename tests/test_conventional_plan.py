"""Executable static ordering with live geometry, stock and route guards."""
import json
import zipfile
import numpy as np
import pytest
from ats.checkpoint import save_state,load_state
from ats.conventional import ConventionalConfig
from ats.conventional_plan import far_dump_candidates
from tests.test_conventional import world,balanced
from tests.test_conventional_routes import loaded_truck


def test_new_policy_and_validation():
    assert ConventionalConfig().dump_order=='far_to_near'
    with pytest.raises(ValueError,match='порядок'):ConventionalConfig(dump_order='random').validate()


def test_work_geometry_is_built_once_but_remaining_follows_live_cut_and_plan():
    s,f=world(trucks=1);t=s.terrain;p=f._work_plan()
    np.testing.assert_array_equal(f._work_remaining(),f._remaining().ravel()[p['ids']])
    for _ in range(8):f._available_target(f.excavator);f.plan_report()
    assert f._work_plan_builds==1 and f._dump_plan_builds==1
    index=p['ids'][int(np.argmax(f._work_remaining()))]
    t.height.ravel()[index]-=.1
    np.testing.assert_array_equal(f._work_remaining(),f._remaining().ravel()[p['ids']])
    t.dig_mask.ravel()[index]=False
    np.testing.assert_array_equal(f._work_remaining(),f._remaining().ravel()[p['ids']])
    assert f._work_plan() is p and f._work_plan_builds==1


def test_dump_order_is_fixed_to_camp_and_skips_filled_cells_live():
    s,f=world(trucks=1);t=s.terrain;truck=f.machines[1];p=f._dump_plan();ids=p['ids']
    distance=(t.x.ravel()[ids]-f.base_position[0])**2+(t.y.ravel()[ids]-f.base_position[1])**2
    assert np.all(np.diff(distance)<=1e-8)
    before=ids.copy();truck.position[:2]=(118.,118.)
    assert f._dump_plan() is p
    np.testing.assert_array_equal(p['ids'],before)
    free=(t.dump_cap.ravel()[ids]-t.height.ravel()[ids])>1e-8
    first=next(far_dump_candidates(f,free,0));assert first[0]==0
    free[:12]=False
    next_point=next(far_dump_candidates(f,free,0));assert next_point[0]==48
    # A direction-level cursor survives capacity changes before its location.
    candidates=list(far_dump_candidates(f,free,53))
    assert candidates[0][0]==53 and len({item[0] for item in candidates})==len(candidates)
    assert len(candidates)==4*np.count_nonzero(free)


def test_live_policy_change_preserves_committed_cargo_route_and_tip():
    s,f,truck=loaded_truck(True);truck.state='DRIVING';truck.after_drive='UNLOADING'
    truck.path=[(94.,100.),(95.,100.)];truck.destination=(95.,100.)
    old=(truck.cargo_kg,truck.target,list(truck.path),truck.destination,truck.soil.masses.copy())
    f.set_dump_order('nearest')
    assert (truck.cargo_kg,truck.target,truck.path,truck.destination)==old[:4]
    np.testing.assert_array_equal(truck.soil.masses,old[4])
    assert truck.state=='DRIVING' and f.config.dump_order=='nearest'
    balanced(s,f)


def test_unload_revalidates_return_path_without_a_second_astar():
    s,f,truck=loaded_truck(True)
    assert f._return_budget(truck) is not None
    calculations=f.route_calculations;assert calculations==1
    assert f._unload(truck)
    assert f.route_calculations==calculations
    assert f.delivered_m3>0.;balanced(s,f)


def test_unload_does_not_trust_a_cached_unsafe_return(monkeypatch):
    s,f,truck=loaded_truck(True);assert f._home_route(truck) is not None
    cargo=truck.cargo_kg;before=s.terrain.height.copy();calls=[]
    monkeypatch.setattr(f,'segment_safe',lambda *args:False)
    monkeypatch.setattr(f,'route',lambda *args:calls.append(args) or None)
    assert not f._unload(truck) and len(calls)==1
    assert truck.cargo_kg==cargo and 'выезда' in truck.reason
    np.testing.assert_array_equal(s.terrain.height,before);balanced(s,f)


def test_far_plan_respects_an_existing_truck_tip_reservation(monkeypatch):
    s,f=world(trucks=2);t=s.terrain;truck,other=f.machines[1:]
    truck.cargo_kg=500.;p=f._dump_plan();ids=p['ids'];first=ids[0]
    other.cargo_kg=500.;other.state='DRIVING';other.after_drive='UNLOADING'
    other.target=(float(t.x.ravel()[first]),float(t.y.ravel()[first]));other.destination=other.target
    chosen=[]
    monkeypatch.setattr(f,'passable',lambda *args:True)
    monkeypatch.setattr(f,'_dump_capacity',lambda *args:(None,np.array([100.])))
    monkeypatch.setattr(f,'_dump_obstacles',lambda:False)
    monkeypatch.setattr(f,'_escape_from',lambda *args:[])
    def assign(machine,point,*args,**kwargs):chosen.append(point);return True
    monkeypatch.setattr(f,'_assign_route',assign)
    # Search is bounded; resume until it moves beyond the other tip footprint.
    for _ in range(10):
        if f._dump_point(truck) is not None:break
        s.time=truck.retry_route_at
    assert chosen and np.linalg.norm(np.asarray(truck.target)-other.target)>=4.
    assert truck.cargo_kg==500. and other.cargo_kg==500.


def test_new_far_plan_checkpoint_rebuilds_only_derived_geometry(tmp_path):
    s,f=world(trucks=1,soil=True);f.plan_report()
    for _ in range(35):s.step()
    path=tmp_path/'far.atsstate';save_state(path,s)
    with zipfile.ZipFile(path) as archive:
        manifest=json.loads(archive.read('manifest.json'))
    encoded=json.dumps(manifest)
    assert '_work_plan_cache' not in encoded and '_dump_plan_cache' not in encoded
    restored=load_state(path).simulation;g=restored.conventional
    assert g.config.dump_order=='far_to_near' and restored.report()==s.report()
    g.plan_report()
    assert g._work_plan_cache['key'][0]==id(g.terrain)
    assert g._work_plan_builds==f._work_plan_builds+1
    for _ in range(30):s.step();restored.step()
    assert restored.report()==s.report();balanced(s,f);balanced(restored,g)


def test_old_config_missing_policy_keeps_legacy_nearest_on_load(tmp_path):
    s,f=world(trucks=1);vars(f.config).pop('dump_order')
    path=tmp_path/'old.atsstate';save_state(path,s)
    restored=load_state(path).simulation
    assert restored.conventional.config.dump_order=='nearest'
    assert restored.conventional.plan_report()['policy']=='nearest'
    balanced(restored,restored.conventional)


def test_rebind_rebuilds_geometry_and_keeps_live_residual():
    s,f=world(trucks=1);f.plan_report();before=f._work_remaining().copy()
    f.rebind_terrain(s.terrain)
    assert f._work_plan_cache is None and f._dump_plan_cache is None
    f.plan_report()
    assert f._work_plan_builds==2 and f._dump_plan_builds==2
    np.testing.assert_allclose(f._work_remaining(),before,atol=1e-12)


def test_filled_dump_keeps_cargo_and_does_not_search_routes():
    s,f,truck=loaded_truck(True);t=s.terrain
    t.dump_cap[t.dump_mask]=t.height[t.dump_mask]
    cargo=truck.cargo_kg;routes=f.route_calculations
    assert f._dump_point(truck) is None
    assert 'заполнен' in truck.reason and truck.cargo_kg==cargo
    assert f.route_calculations==routes
    balanced(s,f)
