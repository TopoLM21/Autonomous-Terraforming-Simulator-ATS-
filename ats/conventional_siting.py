"""Deterministic, read-only allocation of a conventional canal work patch.

An allocated patch is a finite part of the loaded physical window, not a new
terrain or a claim that the entire regional canal sector has been completed.
"""
from copy import deepcopy
import math
import numpy as np


def _blocked(sim,terrain,owners):
    from .site_ground import equipment_mask,paint_rect
    result=equipment_mask(terrain,owners)
    for owner in owners:
        for drone in owner.drones:
            if drone.trip is not None:
                radius=max(owner.config.fleet.separation,drone.config.grab_radius or terrain.cell)
                for point in (drone.trip.dig,drone.trip.dump):
                    paint_rect(terrain,result,(*np.asarray(point)[:2]-radius,2*radius,2*radius))
        if owner.logistics:
            for job in owner.logistics.jobs:
                if job.status in ('QUEUED','ACTIVE'):
                    radius=max(terrain.cell,owner.logistics.config.reach_m)
                    for point in (job.origin,job.destination):
                        paint_rect(terrain,result,(*np.asarray(point)[:2]-radius,2*radius,2*radius))
    return result


def _camp_candidates(terrain,rect,blocked):
    """Full camp footprint first; fleet construction checks every parking bay."""
    from .site_ground import rectangle_patch
    x,y,w,h=rect;side=terrain.cell*terrain.height.shape[0]
    cx,cy=x+w/2,y+h/2
    candidates=[(x-22,cy),(x+w+22,cy),(cx,y-22),(cx,y+h+22),
                (x-22,y-22),(x+w+22,y-22),(x-22,y+h+22),(x+w+22,y+h+22)]
    axis=np.linspace(6.,side-6.,max(3,min(17,math.ceil(side/16))))
    candidates.extend((float(a),float(b)) for a in axis for b in axis)
    seen=set();valid=[]
    for point in sorted(candidates,key=lambda p:((p[0]-cx)**2+(p[1]-cy)**2,p)):
        key=tuple(round(v,6) for v in point)
        if key in seen:continue
        seen.add(key);patch=rectangle_patch(terrain,(point[0]-4,point[1]-4,8,8))
        if patch is None or np.any(blocked[patch]):continue
        valid.append(point)
        if len(valid)==16:break
    return valid


def _channel_edges(plan):
    """Straight physical ramp axes; do not run one ramp around a canal bend."""
    starts=plan.channel.get('chainage_starts',[]);offset=0.
    for number,segment in enumerate(plan.segments):
        local=0.
        for a,b in zip(segment.points,segment.points[1:]):
            a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
            length=float(np.linalg.norm(b-a))
            if length<=1e-8:continue
            start=(starts[number] if starts else offset)+local
            yield a,b,start,start+length
            local+=length
        offset+=local


def prepare(sim,config):
    """Resolve automatic canal allocation without moving, buying or digging.

    Explicit manual configurations are returned untouched. An automatic result
    is a separate, resolved configuration: loading it cannot rerun allocation.
    """
    if not getattr(config,'auto_work_area',False):return config
    from .conventional import ConventionalFleet
    from .conventional_integration import simulations_for,reservation_mask
    from .site_ground import marine_mask,rectangle_patch
    from .workfront import coordinates
    terrain=getattr(sim.terrain,'core',sim.terrain);plan=terrain.config.plan
    if plan is None or not plan.channel:
        raise ValueError('Автоматическая захватка требует проекта канала; для кисти задайте участок вручную')
    owners=simulations_for(sim);site=getattr(sim,'shared_site',None)
    marine=marine_mask(terrain);blocked=_blocked(sim,terrain,owners)
    design=terrain.excavation
    pending=terrain.dig_mask&(terrain.height>terrain.target+1e-6)&~marine
    if not np.any(pending):raise ValueError('В загруженном окне нет оставшейся выемки канала')
    _,along,_=coordinates(plan,terrain.x,terrain.y)
    # Try successive parts of real straight channel edges. An AABB diagonal is
    # not a substitute for the available physical length of a ramp.
    candidates=[]
    for a,b,start,end in _channel_edges(plan):
        useful=pending&(along>=start-1e-8)&(along<=end+1e-8)
        if not np.any(useful):continue
        first,last=float(along[useful].min()),float(along[useful].max())
        direction=(b-a)/(end-start)
        candidates.extend((a,direction,start,end,float(anchor)) for anchor in np.unique(np.linspace(first,last,8)))
    excavator=getattr(config,'excavator_dimensions_m',(6.,3.,3.))
    truck=getattr(config,'truck_dimensions_m',(7.,2.8,3.2))
    radius=max(math.hypot(*excavator[:2]),math.hypot(*truck[:2]))/2
    camp_blocked=blocked|marine|terrain.work_mask|terrain.dump_mask
    if terrain.protected is not None:camp_blocked|=terrain.protected
    dump=terrain.dump_mask&~marine&(terrain.dump_cap>terrain.height+1e-8)
    if not np.any(dump):raise ValueError('Нет свободного сухого отвала для обычной техники')
    reason='Для захватки и стоянки нет безопасного места'
    for edge_start,direction,start,end,anchor in candidates:
        length=max(32.,2*config.excavator_reach_m+2*radius)
        selected=None;depth=0.
        for _ in range(4):
            lo=max(start,min(anchor,end-length));hi=min(end,lo+length)
            selected=design&(along>=lo-terrain.cell)&(along<=hi)
            if not np.any(selected&pending):break
            depth=float(np.maximum(terrain.height-terrain.target,0.)[selected].max())
            needed=depth/config.ramp_gradient+2*radius+2*terrain.cell
            if needed<=length+1e-6:break
            length=needed
        if selected is None or not np.any(selected&pending):continue
        iy,ix=np.nonzero(selected);padding=max(1.,terrain.cell)
        x,y=float(ix.min()*terrain.cell-padding),float(iy.min()*terrain.cell-padding)
        w,h=float((ix.max()-ix.min()+1)*terrain.cell+2*padding),float((iy.max()-iy.min()+1)*terrain.cell+2*padding)
        rect=(x,y,w,h);patch=rectangle_patch(terrain,rect)
        if patch is None:
            reason='Полный поперечник канала с откосами выходит за окно; увеличьте локальную карту'
            continue
        if hi-lo+1e-6<depth/config.ramp_gradient+2*radius:
            reason=f'Рампе требуется не менее {depth/config.ramp_gradient+2*radius:.1f} м; выберите более длинный участок или увеличьте карту'
            continue
        if np.any(marine[patch]) or np.any(blocked[patch]):
            reason='Захватка пересекает море, установленную технику или действующее задание'
            continue
        candidate=deepcopy(config);candidate.work_rect=rect;candidate.auto_work_area=False
        candidate.ramp_entry_m=tuple(map(float,edge_start+direction*(lo-start)))
        candidate.ramp_end_m=tuple(map(float,edge_start+direction*(hi-start)))
        side=terrain.cell*terrain.height.shape[0]
        ends=np.array([candidate.ramp_entry_m,candidate.ramp_end_m])
        clearance=candidate.ramp_width_m/2+terrain.cell
        if np.any(ends<clearance) or np.any(ends>side-clearance):
            reason='Рампа с полным габаритом выходит за загруженное окно; увеличьте локальную карту'
            continue
        reserved=reservation_mask(terrain,candidate,owners)
        if np.any(reserved&blocked):
            reason='Захватка с защитным отступом пересекает установленную технику или действующее задание'
            continue
        if site is not None and any(owner.drones for owner in owners):
            free=terrain.work_mask&~reserved
            if float(np.abs(terrain.height-terrain.target)[free].sum()*terrain.area)<=site.completion_volume:
                reason='После захватки не остаётся отдельной работы для дронов; расширьте план или задайте участок вручную'
                continue
        for point in _camp_candidates(terrain,rect,camp_blocked):
            candidate.base_x,candidate.base_y=point
            try:
                fleet=ConventionalFleet(sim,candidate)
                # Actual bodies and static ground route, not just point centres.
                entry=np.asarray(fleet.ramp['entry_m'],dtype=float)
                end=np.asarray(fleet.ramp['end_m'],dtype=float);delta=end-entry
                norm=float(np.linalg.norm(delta));direction=delta/norm if norm>1e-9 else np.array([1.,0.])
                approach=entry-direction*(fleet.excavator.radius+terrain.cell)
                if fleet.route(fleet.excavator,approach) is None:
                    reason='Не найден безопасный подъезд от стоянки к началу рампы'
                    continue
                return candidate
            except ValueError as error:reason=str(error)
    raise ValueError('Не удалось автоматически выделить участок канала: '+reason)
