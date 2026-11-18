"""A checked ground route must remain executable at the model's driving steps."""
import numpy as np

from ats.checkpoint import save_state,load_state
from ats.conventional import ConventionalConfig,ConventionalFleet
from ats.simulation import Simulation
from tests.test_conventional import world,fuel_tanks,balanced
from tests.test_conventional_workspace import workspace_config


def cut_edge():
    config=workspace_config()
    axis=(np.arange(config.terrain.n)+.5)*config.terrain.cell_size
    x,y=np.meshgrid(axis,axis)
    # An imported shallow cut, with no runtime loss of ground in the test.
    height=np.where((x>=40.25)&(x<42.25)&(y>=28.75),-.3,0.)
    config.initial_height=height;config.reference_height=height;config.local_reference=True
    s=Simulation.conventional_only(config)
    f=ConventionalFleet(s,ConventionalConfig(base_x=12.,base_y=12.,work_rect=(35.,40.,60.,20.),truck_count=1))
    s.conventional=f;fuel_tanks(f);e,truck=f.machines
    truck.position[:]=(100.,25.,0.)
    line=28.75-e.radius-s.terrain.cell*.71+.001
    e.position[:]=(30.1,line,0.);e.home[:]=e.position
    goal=(70.1,line)
    assert f.passable(e.position,e.radius,e) and f.passable(goal,e.radius,e)
    return s,f,e,goal


def finish_drive(s,m,limit=150):
    f=s.conventional
    for _ in range(limit):
        if m.state!='DRIVING':break
        s.time+=1;f._move(m)
    assert m.state!='DRIVING',(m.position,m.reason,m.path)
    np.testing.assert_allclose(m.position[:2],m.destination,atol=1e-10)


def test_shallow_cut_edge_is_detoured_and_reverse_escape_really_drives():
    s,f,e,goal=cut_edge();start=e.position[:2].copy();before=s.terrain.height.copy()
    assert not f.segment_safe(start,goal,e.radius,e)
    assert f._assign_route(e,goal,'PARKED')
    outbound=list(e.path)
    assert any(abs(p[1]-goal[1])>.1 for p in outbound)
    finish_drive(s,e)
    # Site selection uses this same reversed-outbound construction as an exit.
    reverse=[*reversed(outbound[:-1]),tuple(start)]
    assert f._assign_route(e,start,'PARKED',outbound_path=reverse)
    finish_drive(s,e)
    assert e.distance_m>80. and not e.reason
    np.testing.assert_array_equal(s.terrain.height,before)
    balanced(s,f)


def test_loaded_legacy_route_at_cut_edge_recovers_and_resumes_exactly(tmp_path):
    s,f,e,goal=cut_edge()
    # Recreate the accepted direct route in a pre-fix v23 save. With that
    # sampler, repeated replans left this pose unchanged for hundreds of steps.
    e.position[0]=39.9;e.destination=goal;e.after_drive='PARKED';e.state='DRIVING'
    f._set_drive_path(e,[goal]);e.reason='Проезд занят или уклон изменился'
    path=tmp_path/'old-edge-route.atsstate';save_state(path,s)
    other=load_state(path).simulation;oe=other.conventional.excavator
    before=s.terrain.height.copy()
    for _ in range(150):
        for owner,machine in ((s,e),(other,oe)):
            owner.time+=1
            if machine.state=='DRIVING':owner.conventional._move(machine)
        assert other.report()==s.report()
        np.testing.assert_array_equal(oe.position,e.position)
        if e.state!='DRIVING':break
    assert e.state=='PARKED' and not e.reason
    np.testing.assert_allclose(e.position[:2],goal,atol=1e-10)
    np.testing.assert_array_equal(s.terrain.height,before)
    balanced(s,f);balanced(other,other.conventional)


def test_grazing_other_vehicle_cannot_make_an_unexecutable_direct_route():
    s,f=world(trucks=1);fuel_tanks(f);e,truck=f.machines
    e.position[:]=(30.,25.,0.);e.home[:]=e.position
    clearance=e.radius+truck.radius+.25
    truck.position[:]=(40.12,25.+clearance-.0001,0.)
    goal=(70.,25.)
    assert not f.segment_safe(e.position,goal,e.radius,e)
    assert f._assign_route(e,goal,'PARKED')
    finish_drive(s,e)
    assert np.linalg.norm(truck.position[:2]-e.position[:2])>=clearance
    balanced(s,f)


def test_warning_clears_on_first_successful_step_and_real_block_keeps_fuel():
    s,f=world(trucks=1);fuel_tanks(f);e,truck=f.machines
    e.position[:]=(30.,25.,0.);e.home[:]=e.position
    truck.position[:]=(100.,25.,0.);goal=(70.,25.)
    assert f._assign_route(e,goal,'PARKED')
    truck.position[:]=(38.,25.,0.)
    # Actually close the next swept support footprint after route assignment.
    truck.position[0]=e.position[0]+e.radius+truck.radius+.25+.1
    position=e.position.copy();fuel=e.fuel_l
    s.time+=1;f._move(e)
    assert 'Проезд занят' in e.reason and e.state=='DRIVING'
    np.testing.assert_array_equal(e.position,position)
    assert e.fuel_l==fuel
    truck.position[:]=(100.,25.,0.)
    s.time+=1;f._move(e)
    assert np.linalg.norm(e.position[:2]-position[:2])>0 and not e.reason
    balanced(s,f)


def test_replan_watches_the_new_detour_ground_corridor():
    s,f=world(trucks=1);fuel_tanks(f);e,truck=f.machines
    e.position[:]=(30.,25.,0.);e.home[:]=e.position
    truck.position[:]=(100.,25.,0.);goal=(70.,25.)
    assert f._assign_route(e,goal,'PARKED');old_bounds=e.route_bounds
    # A real intervening vehicle forces a new corridor outside the old bounds.
    truck.position[:]=(40.,25.,0.)
    s.time+=1;f._move(e)
    # Continue until the near-step guard encounters the occupied corridor.
    for _ in range(30):
        if e.route_bounds!=old_bounds:break
        s.time+=1;f._move(e)
    assert e.route_bounds!=old_bounds
    points=np.array([e.position[:2],*e.path])
    assert np.all(points.min(axis=0)-e.radius>=np.array(e.route_bounds[:2])-1e-10)
    assert np.all(points.max(axis=0)+e.radius<=np.array(e.route_bounds[2:])+1e-10)
    # Relevant edits to the detour must invalidate its own route token.
    x=min(e.path,key=lambda p:p[1])[0];y=e.route_bounds[1]+.1
    assert y<old_bounds[1]
    t=s.terrain;old_token=t.route_token(old_bounds);row=int(y/t.cell);col=int(x/t.cell)
    t.height[row,col]+=.01;t.revision+=1;t._touch_route_cells(np.array([row]),np.array([col]))
    assert e.route_token!=t.route_token(e.route_bounds)
    assert old_token==t.route_token(old_bounds)
