"""New-project placement preflight, using the existing design and equipment.

Only a new mission may choose a larger window. A physical regional move keeps
its current grid, coordinate frame and overlap requirements.
"""
import math
import numpy as np


class SiteError(ValueError):
    """Geometry/capacity that another workspace may resolve."""


def slope_runs(config,profile,depth):
    depth=np.maximum(depth,0.)
    if config.trench.angle_deg is not None:return depth/math.tan(math.radians(config.trench.angle_deg))
    from .slopes import minimum_slope,legacy
    if legacy(config) or profile is None:return depth/minimum_slope(config)
    if config.terrain.cut_policy=='natural':
        slope=min(math.tan(math.radians(getattr(m,k))) for m in profile.materials for k in ('native_angle_deg','loose_angle_deg'))
        return depth/slope
    result=np.zeros_like(depth);lo=0.
    for hi,m in zip([*profile.boundaries,math.inf],profile.materials):
        result+=np.maximum(0.,np.minimum(depth,hi)-lo)/math.tan(math.radians(m.native_angle_deg));lo=hi
    return result


def cross_section(project,landscape,config,origin,side,station):
    """Probe the original regional cross section beyond the active window.

    This is a conservative sizing check, not a second target or soil field.
    The normal target_surface/shore law remains authoritative inside the grid.
    """
    from .channel import _geometry,_point,is_ocean
    points,axis=_geometry(project.points);point=_point(points,axis,station)
    i=min(len(points)-2,max(0,int(np.searchsorted(axis,station,side='right')-1)))
    tangent=(points[i+1]-points[i])/max(1e-12,axis[i+1]-axis[i]);normal=np.array([-tangent[1],tangent[0]])
    raster=landscape.raster
    highest=float(np.nanmax(raster.height))+project.depth_m+1.
    maximum=project.width_m/2+float(slope_runs(config,landscape.soil_profile,np.array(highest)))
    offset=np.unique(np.r_[np.linspace(-maximum,maximum,4097),0.])
    sampling_pad=maximum/4096+config.terrain.cell_size
    xy=point+offset[:,None]*normal
    valid=np.all((xy>=0)&(xy<=raster.extent_m),axis=1)
    xy=xy[valid];offset=offset[valid]
    heights=raster.sample(xy[:,0],xy[:,1]);depth=heights+project.depth_m
    amplitude=(landscape.generation or {}).get('amplitude_m',0.)
    runs=slope_runs(config,landscape.soil_profile,depth+amplitude)
    cutting=(depth>0)&(np.abs(offset)<project.width_m/2+runs-1e-8)&~is_ocean(raster,xy[:,0],xy[:,1])
    if not np.any(cutting):return dict(width_m=0.,fits=True)
    cut=xy[cutting];lo=cut.min(axis=0);hi=cut.max(axis=0)
    dem_limited=bool((not valid[0] and cutting[0]) or (not valid[-1] and cutting[-1]))
    fits=bool(not dem_limited and np.all(lo>=np.asarray(origin)+sampling_pad)&np.all(hi<=np.asarray(origin)+side-sampling_pad))
    return dict(width_m=float(offset[cutting].max()-offset[cutting].min()),fits=fits,
        bounds_world_m=[*map(float,lo),*map(float,hi)],station_m=float(station),dem_limited=dem_limited)


def equipment_spec(config):
    from .equipment import resolve_config
    from .base_layout import configure_layout
    resolve_config(config)
    return configure_layout(config)


def equipment_fits(config,spec,point,terrain):
    if spec is None:return True
    from .site_ground import footprint_safe
    from .base_layout import layout_footprints,rotate_offset
    from .self_solar import deployment_rectangles as solar
    from .battery_platforms import deployment_rectangles as batteries
    from .compact_layout import rectangles
    blocked=terrain.design_work|terrain.dump_mask
    if terrain.protected is not None:blocked=blocked|terrain.protected
    for rect in [*layout_footprints(spec,point),*rectangles(spec,point)]:
        if not footprint_safe(terrain,rect,blocked,grade_deg=math.degrees(math.atan(config.base.max_slope)),relief_m=float('inf')):return False
    for rect in [*solar(spec,point),*batteries(spec,point)]:
        if not footprint_safe(terrain,rect,blocked,grade_deg=8.,relief_m=.6):return False
    f=config.fleet
    for i in range(max(f.count,f.charge_slots)):
        offsets=spec.get('dock_offsets',())
        xy=offsets[i] if i<len(offsets) else [0.,((i+1)//2)*(1 if i%2 else -1)*f.pad_spacing]
        distance=max(f.waiting_offset,spec.get('hub_radius_m',config.base.radius)+3+f.separation) if offsets else f.waiting_offset
        for shift in (0.,-distance):
            x,y=np.asarray(point)+rotate_offset(spec['parameters'],[xy[0]+shift,xy[1]])
            if not footprint_safe(terrain,(x-f.separation,y-f.separation,2*f.separation,2*f.separation),blocked,grade_deg=8.,relief_m=float('inf')):return False
    return True
