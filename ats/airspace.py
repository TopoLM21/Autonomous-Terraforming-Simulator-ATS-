"""Conservative cruise envelope, independent of individual obstacle polygons.

Soil transfers cannot create a height above both donor and receiver. Controlled
deposits are capped, and Terrain tracks new plans/edits/window bounds. Hardware
is bounded in its highest transport/deployment state, not its current pose.
Only an entire segment above this envelope may bypass detailed ground tests.
Aircraft and suspended loads still use swept collision checks.
"""
import math


def units(simulation):
    site=getattr(simulation,'shared_site',None)
    return [u.simulation for u in site.project.units] if site is not None else [simulation]


def unit_ceiling(s,ground):
    b=s.base;w=s.logistics;original_spec=getattr(b,'layout_spec',None);spec=original_spec or {}
    solar=getattr(w,'solar_array',None);settings=getattr(solar,'settings',{})
    e=getattr(w,'excavator',None)
    excavator_z=e.position[2] if e is not None and e.state.value!='AIR_TRANSPORT' else -math.inf
    key=(s.time,ground,id(original_spec),len(spec.get('platforms',())),b.config.roof_height,float(b.position[2]),
         id(solar),settings.get('hinge_height_m',0.),settings.get('leaf_width_m',0.),float(excavator_z))
    cached=getattr(s,'_flight_ceiling_cache',None)
    if cached is not None and cached[0]==key:return cached[1]
    hardware=max(b.config.roof_height,1.4,.5,
        max((a['dimensions_m'][2] for a in spec.get('platforms',())),default=0.),
        settings.get('hinge_height_m',0.)+settings.get('leaf_width_m',0.))
    result=max(ground+hardware,float(b.position[2])+hardware,float(excavator_z)+1.)
    s._flight_ceiling_cache=(key,result)
    return result


def ground_ceiling(s):
    terrain=getattr(s.terrain,'core',s.terrain)
    ground=terrain.clearance_surface
    return max(unit_ceiling(u,ground) for u in units(s))


def above_ground(s,start,end):
    # A straight segment's minimum Z occurs at an endpoint. Do not infer
    # anything about unobserved ground beyond the active physical window.
    side=s.terrain.cell*s.terrain.height.shape[0]
    if any(not 0<=p[i]<side for p in (start,end) for i in (0,1)):return False
    return min(start[2],end[2])>=ground_ceiling(s)+s.config.fleet.ground_clearance


def cruise(s,drone):
    group=units(s)
    spacing=max(max(u.config.fleet.altitude_spacing,u.config.fleet.separation+u.config.fleet.ground_clearance) for u in group)
    clearance=max(max(d.config.work_height for d in u.drones) for u in group)
    # Four common levels, reused between distant teams. Offset neighbouring
    # complexes so equally numbered aircraft do not always share one level.
    number=next((i for i,u in enumerate(group) if u is s),0)
    lane=(drone.identifier+number)%4
    return ground_ceiling(s)+max(clearance,s.config.fleet.ground_clearance)+lane*spacing
