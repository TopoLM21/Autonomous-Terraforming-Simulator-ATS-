"""Conservative full-cross-section progress across overlapping terrain windows.

Only proved, excavated spans are stored. A cut touching a window edge does not
prove its whole cross section; unknown terrain is never counted as finished.
These intervals describe progress, not a second terrain or material ledger.
"""
import math
import numpy as np


def merge(spans):
    result=[]
    for a,b in sorted(spans):
        if b<=a:continue
        if result and a<=result[-1][1]+1e-8:result[-1][1]=max(result[-1][1],float(b))
        else:result.append([float(a),float(b)])
    return result


def subtract(spans,removed):
    result=merge(spans)
    for lo,hi in merge(removed):
        result=[[a,b] for x,y in result for a,b in ((x,min(y,lo)),(max(x,hi),y)) if b>a]
    return result


def project_length(terrain):
    channel=terrain.config.plan.channel
    length=channel.get('project_length_m',(terrain.config.georeference or {}).get('channel',{}).get('length_m'))
    if length is not None:return float(length)
    starts=channel.get('chainage_starts',[]);cursor=0.
    for i,segment in enumerate(terrain.config.plan.segments):
        cursor=(starts[i] if starts else cursor)+sum(float(np.linalg.norm(np.asarray(b)-a)) for a,b in zip(segment.points,segment.points[1:]))
    return cursor


def context(terrain):
    core=getattr(terrain,'core',terrain);owner=getattr(core,'_site_owner',None)
    site=getattr(terrain,'shared_site',None) or getattr(owner,'shared_site',None)
    sims=[u.simulation for u in site.project.units] if site else [owner] if owner else []
    return core,site,sims


def cargo_clear(terrain):
    _,_,sims=context(terrain)
    for s in sims:
        if s._terrain_active or any(not d.grounded or d.trip is not None or d.cargo>1e-8 for d in s.drones):return False
        w=s.logistics
        if w and (w.active or w.direct_ids or w.group or any(c.soil_kg>1e-8 for c in w.containers)
                or w.excavator and (w.excavator.bucket_kg>1e-8 or w.excavator.active_container)):
            return False
    return True


def geometry(terrain,lo,hi):
    core,site,_=context(terrain);plan=core.config.plan;front=core.front or terrain.front
    cache=core.__dict__.setdefault('_sector_progress_cache',{})
    key=(id(core.target),id(core.design_work),float(lo),float(hi),core.cell)
    if key in cache:return cache[key]
    step=core.cell;count=max(1,math.ceil((hi-lo)/step))
    edges=np.minimum(hi,lo+np.arange(count+1)*step);centres=(edges[1:]+edges[:-1])/2
    points=np.full((count,2),np.nan);starts=plan.channel.get('chainage_starts',[]);cursor=0.
    for i,segment in enumerate(plan.segments):
        cursor=starts[i] if starts else cursor
        for a,b in zip(segment.points,segment.points[1:]):
            a,b=np.asarray(a),np.asarray(b);length=float(np.linalg.norm(b-a))
            if length<1e-12:continue
            mask=(centres>=cursor-1e-9)&(centres<=cursor+length+1e-9)
            points[mask]=a+(centres[mask,None]-cursor)*(b-a)/length;cursor+=length
    side=core.height.shape[0]*step
    loaded=np.all(np.isfinite(points)&(points>=step)&(points<=side-step),axis=1)
    work=core.design_work if core.design_work is not None else core.work_mask
    along=front.chainage
    # Include a small longitudinal halo for the projected footprint of an edge
    # cell. This also detects a route leaving/re-entering the window at bends.
    edge_along=np.r_[along[0,work[0]],along[-1,work[-1]],along[work[:,0],0],along[work[:,-1],-1]]
    bins=np.floor((edge_along-lo)/step).astype(int)
    for offset in range(-2,3):
        bad=bins+offset;bad=bad[(bad>=0)&(bad<count)];loaded[bad]=False
    ids=np.flatnonzero(work&(along>=lo-1e-9)&(along<hi+1e-9))
    indices=np.clip(np.floor((along.ravel()[ids]-lo)/step).astype(int),0,count-1)
    result=dict(edges=edges,loaded=loaded,ids=ids,indices=indices,revision=None)
    cache[key]=result
    return result


def ready_bins(terrain,lo,hi):
    core,site,_=context(terrain);g=geometry(terrain,lo,hi)
    tolerance=site.completion_volume if site else core.config.terrain.completion_volume
    stamp=(core.revision,tolerance)
    if g['revision']!=stamp:
        residual=np.abs(core.height.ravel()[g['ids']]-core.target.ravel()[g['ids']])*core.area
        volumes=np.bincount(g['indices'],weights=residual,minlength=len(g['loaded']))
        ready=volumes<=tolerance*.1*np.diff(g['edges'])/max(hi-lo,core.cell)
        # Keep the engine's existing per-window completion tolerance. A field
        # it legitimately finishes must not become an unfinishable certificate
        # because the display used a stricter, unrelated threshold.
        if float(volumes.sum())<=tolerance*.1 or core.remaining<=tolerance:ready[:]=True
        g.update(revision=stamp,ready=ready&g['loaded'])
    return g


def intervals(edges,mask):
    changes=np.diff(np.r_[False,mask,False].astype(np.int8))
    return [[float(edges[a]),float(edges[b])] for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]


def ledger(terrain):
    core,_,_=context(terrain)
    return core.config.plan.channel.get('finished_spans',[])


def status(front):
    lo,hi=front.global_sector_start,front.global_sector_end
    g=ready_bins(front.terrain,lo,hi);edges=g['edges'];covered=np.zeros(len(g['loaded']),dtype=bool)
    for a,b in ledger(front.terrain):covered|=(edges[:-1]>=a-1e-8)&(edges[1:]<=b+1e-8)
    covered[g['loaded']]=g['ready'][g['loaded']]
    spans=merge([*subtract(ledger(front.terrain),intervals(edges,g['loaded'])),*intervals(edges,g['ready'])])
    project_done=bool(spans and spans[0][0]<=1e-8 and spans[0][1]>=front.project_end-1e-8)
    core,site,_=context(front.terrain)
    empty_window=core.remaining<=(site.completion_volume if site else core.config.terrain.completion_volume)
    return dict(global_sector_start_m=lo,global_sector_end_m=hi,
        visible_part_complete=front.complete or empty_window,window_limited=bool(np.any(~g['loaded'])),
        needs_next_window=bool(np.any(~g['loaded']&~covered)) or bool(empty_window and not project_done),
        global_profile_complete=bool(np.all(covered)),
        global_sector_complete=bool(np.all(covered)) and cargo_clear(front.terrain),
        project_profile_complete=project_done,
        at_project_end=hi>=front.project_end-1e-8)


def record(terrain,lo,hi):
    core,site,sims=context(terrain);g=ready_bins(terrain,lo,hi)
    old=ledger(terrain)
    # A new slump or edit inside an observed cross section invalidates its old
    # certificate. Never erase certificates for terrain outside this window.
    spans=subtract(old,intervals(g['edges'],g['loaded']))
    if cargo_clear(terrain):spans=merge([*spans,*intervals(g['edges'],g['ready'])])
    land=site.project.landscape if site else None
    external=land.channel_project if land and land.channel_project is not None else None
    if spans==old and (external is None or external.get('finished_spans',[])==spans) and all(s.config.plan.channel.get('finished_spans',[])==spans for s in sims):return
    core.config.plan.channel['finished_spans']=spans
    for s in sims:s.config.plan.channel['finished_spans']=spans
    if external is not None:external['finished_spans']=spans


def record_visible(terrain):
    t,_,_=context(terrain);f=t.front
    if f is None or not f.sectors:return
    cache=t.__dict__.setdefault('_sector_progress_cache',{})
    key=(id(t.target),t.revision,cargo_clear(t),tuple(map(tuple,ledger(t))))
    if cache.get('recorded_window')==key:return
    work=t.design_work if t.design_work is not None else t.work_mask
    length=f.settings['sector_length_m'];end=project_length(t)
    for number in np.unique(np.floor((f.chainage[work]+1e-9)/length).astype(int)):
        lo=number*length;hi=min(end,lo+length)
        if hi>lo:record(t,float(lo),float(hi))
    cache['recorded_window']=(id(t.target),t.revision,cargo_clear(t),tuple(map(tuple,ledger(t))))


def record_window(project):
    if project.shared_site is not None:record_visible(project.shared_site.core)
