"""Growing, volume-sized spoil reservations. Occupied/reserved ground never shrinks."""
import math
import numpy as np


def inside_distance(mask):
    """Exact chessboard distance, matching repeated 8-neighbour erosion, O(n²)."""
    n,m=mask.shape;d=np.pad(np.where(mask,n+m+1,0),1);x=np.arange(m+2)
    for y in range(1,n+1):
        d[y,1:-1]=np.minimum(d[y,1:-1],1+np.minimum.reduce((d[y-1,:-2],d[y-1,1:-1],d[y-1,2:])))
        d[y]=np.minimum.accumulate(d[y]-x)+x
    for y in range(n,0,-1):
        d[y,1:-1]=np.minimum(d[y,1:-1],1+np.minimum.reduce((d[y+1,:-2],d[y+1,1:-1],d[y+1,2:])))
        d[y]=np.minimum.accumulate((d[y]+x)[::-1])[::-1]-x
    return np.maximum(d[1:-1,1:-1]-1,0)


def reserve(terrain,allowed,along,signed,opening,settings,*,old_mask=None,old_cap=None,blocked=None,fronts=None,carried_volume=0.):
    c=terrain.config;t=c.terrain;cell=terrain.cell
    old_mask=np.zeros_like(allowed) if old_mask is None else old_mask
    old_cap=terrain.reference if old_cap is None else old_cap
    work=terrain.design_work if getattr(terrain,'design_work',None) is not None else terrain.work_mask
    excess=np.where(work,np.maximum(terrain.height-terrain.target,0),0.)
    pending=excess>1e-10
    start=float(along[pending].min()) if np.any(pending) else float(along[work].max()) if np.any(work) else 0.
    front=settings['front_length_m'];starts=list(fronts) if fronts else [start]
    # A shared site can have several independent advancing work fronts.
    # Reserve the union, rather than multiplying overlapping requirements.
    end=max(starts)+front*t.spoil_front_sections;start=min(starts)
    active=np.zeros_like(allowed);source=np.zeros_like(allowed)
    for position in starts:
        stop=position+front*t.spoil_front_sections
        active |= (along>=position-front*.5)&(along<=stop+front*.5)
        source |= (along>=position-front*.5)&(along<=stop)
    required=(float(excess[source].sum()*terrain.area)+carried_volume)*(1+t.spoil_reserve_margin)
    # Existing piles from earlier windows remain unavailable to equipment even
    # when the new compact policy would not have chosen their position.
    occupied=~work&(terrain.height>terrain.reference+1e-10)
    retained=old_mask|occupied
    allowed=allowed.copy()
    if blocked is not None:allowed &= ~blocked
    # The total surface slope is ground slope + heap slope. A reduced tent
    # on sloping ground remains below the loose-material stability limit.
    gy,gx=np.gradient(terrain.reference,cell)
    grade=np.hypot(gx,gy);allowed &= grade<terrain.dump_slope*.95
    used_grade=float(grade[allowed|retained].max()) if np.any(allowed|retained) else 0.
    tent_slope=max(0.,min(terrain.dump_slope*t.dump_slope_factor,terrain.dump_slope-used_grade))
    def candidate(width,window):
        mask=retained|(allowed&window&(np.abs(signed)<opening+settings['setback_m']+width))
        depth=inside_distance(mask)*cell
        cap=terrain.reference+np.minimum(t.dump_max_height,depth*tent_slope)
        # Monotone extension keeps every existing unloading allocation valid.
        cap=np.maximum(old_cap,cap)
        cap=np.maximum(cap,np.where(occupied,terrain.height,terrain.reference))
        free=float(np.maximum(cap-terrain.height,0)[mask&window].sum()*terrain.area)
        return mask,cap,free
    maximum=settings['bank_width_m'];chosen=None
    for reach in (1.,2.,float('inf')):
        window=active if reach==1 else ((along>=start-front*reach)&(along<=end+front*reach)) if math.isfinite(reach) else np.ones_like(active)
        maximum_case=candidate(maximum,window)
        if maximum_case[2]+1e-9<required:
            chosen=maximum_case;continue
        low,high=0.,maximum;chosen=maximum_case
        for _ in range(8):
            mid=(low+high)/2;trial=candidate(mid,window)
            if trial[2]+1e-9>=required:high=mid;chosen=trial
            else:low=mid
        if chosen is None:chosen=maximum_case
        break
    mask,cap,free=chosen
    return mask,cap,dict(required_free_m3=required,available_free_m3=free,
        margin=t.spoil_reserve_margin,front_start_m=start,front_end_m=end,
        sufficient=free+1e-9>=required,allowed_area_m2=float(allowed.sum()*terrain.area),
        reserved_area_m2=float(mask.sum()*terrain.area),occupied_area_m2=float(occupied.sum()*terrain.area))
