"""Intact-ground canal-axis stations. Scheduling geometry, never a soil ledger.

The whole formation stays ahead of the active face, including the conservative
end-slope run. A shared fleet advances its common face only after every team
has drained its loads, and resumes digging only after every base has arrived.
"""
import math
import numpy as np


def enabled(terrain):
    plan=terrain.config.plan
    return bool(plan and plan.channel and plan.channel.get('operations',{}).get('base_strategy')=='axis_ahead')


def core(terrain):return getattr(terrain,'core',terrain)


def slope_run(terrain):
    """Weakest native/loose angle; no claim of rock-wall strength or cohesion."""
    t=core(terrain)
    from .slopes import minimum_slope
    slope=min(minimum_slope(t.config),minimum_slope(t.config,loose=True))
    height=max(float(t.reference.max()),float(t.target.max()))-float(t.target.min())
    return max(0.,height)/max(slope,1e-8)+2*t.cell


def work_exclusion(terrain,*,future=False):
    """Keep old forecasts in bank mode; allow only unworked ground in axis mode.

    While moving, the old face remains authoritative for route/folding checks.
    The destination additionally passes the next face's full deployment check.
    No frozen cells or protected support columns are added to soil relaxation.
    """
    if not enabled(terrain):return terrain.design_work
    t=core(terrain);f=t.front
    if f is None:return terrain.design_work
    guard=f.global_sector_end if future else getattr(f,'support_guard_end',f.global_sector_end)
    cache=t.__dict__.setdefault('_axis_station_cache',{})
    profile=t.config.soil_profile
    angles=tuple((m.native_angle_deg,m.loose_angle_deg) for m in profile.materials) if profile else ()
    key=(id(t.height),id(t.reference),id(t.target),t.revision,t.config.slope_model,
         t.config.terrain.repose_deg,t.config.trench.angle_deg,angles)
    if cache.get('key')!=key:cache.clear();cache.update(key=key,masks={})
    if float(guard) in cache['masks']:return cache['masks'][float(guard)]
    run=slope_run(t);blocked=t.height<t.reference-1e-8
    starts=t.config.plan.channel.get('chainage_starts',[]);cursor=0.
    for i,segment in enumerate(t.config.plan.segments):
        cursor=starts[i] if starts else cursor
        for a,b in zip(segment.points,segment.points[1:]):
            a,b=np.asarray(a),np.asarray(b);v=b-a;length=float(np.linalg.norm(v))
            if length<1e-12:continue
            if cursor<guard:
                stop=min(length,max(0.,guard-cursor));direction=v/length
                along=(t.x-a[0])*direction[0]+(t.y-a[1])*direction[1]
                across=(t.y-a[1])*direction[0]-(t.x-a[0])*direction[1]
                # A sector boundary is a flat face across the channel. Its
                # longitudinal setback is the slope run, not width/2 + run.
                # Preserve real rounded caps at earlier vertices and at the
                # true end of the project, as in the plan's bottom mask.
                dx=np.maximum(np.maximum(-along,along-stop),0.)
                dy=np.maximum(np.abs(across)-segment.width/2,0.)
                blocked|=np.hypot(dx,dy)<run
                if cursor>0:blocked|=np.hypot(t.x-a[0],t.y-a[1])<segment.width/2+run
                if cursor+length<guard+1e-9 and (cursor+length<guard-1e-9 or guard>=f.project_end-1e-8):
                    blocked|=np.hypot(t.x-b[0],t.y-b[1])<segment.width/2+run
            cursor+=length
    cache['masks'][float(guard)]=blocked
    return blocked


def axis_point(plan,station):
    """Continue the final tangent onto a terminal staging apron if necessary."""
    starts=plan.channel.get('chainage_starts',[]);cursor=0.;last=None
    for i,segment in enumerate(plan.segments):
        cursor=starts[i] if starts else cursor
        for a,b in zip(segment.points,segment.points[1:]):
            a,b=np.asarray(a),np.asarray(b);length=float(np.linalg.norm(b-a))
            if length<1e-12:continue
            tangent=(b-a)/length;last=(b,tangent,cursor+length,segment.width)
            if station<=cursor+length:return a+(station-cursor)*tangent,tangent,segment.width
            cursor+=length
    if last is None:raise ValueError('У канала отсутствует ось')
    b,tangent,end,width=last
    return b+(station-end)*tangent,tangent,width


def candidates(terrain,spec,sector_end,*,limit=240):
    """Anchors on/near the future axis, fitted by the full deployed footprint."""
    t=core(terrain);plan=t.config.plan
    _,tangent,width=axis_point(plan,sector_end)
    from .base_layout import layout_footprints,rotate_offset
    from .self_solar import deployment_rectangles
    from .battery_platforms import deployment_rectangles as batteries
    from .compact_layout import rectangles
    boxes=[]
    if spec:
        boxes=[*layout_footprints(spec,[0.,0.]),*deployment_rectangles(spec,[0.,0.]),
            *batteries(spec,[0.,0.]),*rectangles(spec,[0.,0.])]
    else:
        r=t.config.base.radius;boxes=[(-r,-r,2*r,2*r)]
    f=t.config.fleet
    for i in range(max(f.count,f.charge_slots)):
        offsets=(spec or {}).get('dock_offsets',[])
        xy=offsets[i] if i<len(offsets) else [0.,((i+1)//2)*(1 if i%2 else -1)*f.pad_spacing]
        distance=max(f.waiting_offset,(spec or {}).get('hub_radius_m',t.config.base.radius)+3+f.separation) if offsets else f.waiting_offset
        for dx in (0.,-distance):
            x,y=rotate_offset((spec or {}).get('parameters',t.config.base.layout),[xy[0]+dx,xy[1]])
            boxes.append((x-f.separation,y-f.separation,2*f.separation,2*f.separation))
    corners=np.array([(x+dx*w,y+dy*h) for x,y,w,h in boxes for dx in (0,1) for dy in (0,1)])
    behind=max(0.,-float(np.min(corners@tangent)))
    from .sector_progress import project_length
    cap=width/2 if sector_end>=project_length(t)-1e-8 else 0.
    first=sector_end+slope_run(t)+behind+max(3.,2*t.cell)+cap
    side=t.height.shape[0]*t.cell
    # Prefer the centre line. Additional rows let several complexes share a
    # wide future channel without assigning them different excavation fronts.
    shifts=[0.,*[sign*v for v in np.arange(12.,width/2+1.,12.) for sign in (-1,1)]]
    result=[]
    for station in np.arange(first,first+side,max(4.,t.cell*2)):
        point,direction,_=axis_point(plan,float(station));n=np.array([-direction[1],direction[0]])
        for shift in shifts:
            xy=point+n*shift
            if np.all(xy>t.cell)&np.all(xy<side-t.cell):result.append((float(station),*map(float,xy)))
            if len(result)>=limit:return result
    return result


def cohort(sim):
    site=sim.shared_site
    return [u.simulation for u in site.project.units] if site else [sim]


def ready(terrain):
    if not enabled(terrain):return True
    return bool(core(terrain).front.stationed)


def align(front,leader):
    for key in ('sector_id','project_end','global_sector_start','global_sector_end','sector_start','sector_end','sector_centre'):
        setattr(front,key,getattr(leader,key))
    front.revision=-1


def update(sim,front):
    """Return True when ordinary per-base arrival/placement should run."""
    t=core(sim.terrain);leader=t.front;teams=cohort(sim)
    if sim.shared_site:
        if leader.stationed:
            if not all(s.terrain.front.complete for s in teams):
                sim.base_move_wait='WAITING_SECTOR_TEAMS' if front.complete else ''
                return False
            if not all(s._sector_drained() for s in teams):
                sim.base_move_wait='DRAINING_SECTOR';return False
            from .sector_progress import record,record_visible,status
            record(t,leader.global_sector_start,leader.global_sector_end)
            if t.remaining<=sim.shared_site.completion_volume:record_visible(t)
            if status(leader)['needs_next_window']:
                for s in teams:s.base_move_wait='NEEDS_NEXT_WINDOW'
                return False
            old_end=leader.global_sector_end
            if not leader.advance_sector():return False
            leader.support_guard_end=old_end
            for s in teams:
                f=s.terrain.front;align(f,leader)
                f.stationed=False;f.moving=False;f.followers_launched=False;f.legacy_pickups=[]
                s.base_move_wait='POSITIONING_SECTOR';s._relocation_attempt_time=-math.inf
                if s.state.value=='COMPLETED':
                    from .simulation import MissionState
                    s.state=MissionState.RUNNING;s.reason=''
            sim.event(f'Вся группа завершила сектор; переезд вперёд по оси к {leader.sector_start:g}–{leader.sector_end:g} м')
        if front.stationed:
            if all(s.terrain.front.stationed and s.base.destination is None and s.relocation_pending is None for s in teams):
                leader.stationed=True;leader.support_guard_end=leader.global_sector_end
                for s in teams:s.terrain.front.revision=-1;s.base_move_wait=''
            else:sim.base_move_wait='WAITING_SECTOR_TEAMS'
            return False
        # A convoy cannot drive through a still parked base ahead of it. Move
        # the leading formation first, then release the next one; no disabling
        # of inter-complex collisions or overlapping swept footprints.
        _,direction,_=axis_point(t.config.plan,leader.global_sector_end)
        key=lambda s:(float(s.base.position[:2]@direction),-teams.index(s))
        if any(not s.terrain.front.stationed and key(s)>key(sim) for s in teams):
            sim.base_move_wait='WAITING_SECTOR_TEAMS';return False
        return True
    return True
