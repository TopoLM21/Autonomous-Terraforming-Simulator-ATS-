"""Deterministic column-face stability, without cohesion or overhangs.

Horizontal support run is integral(dz / tan(angle)) through the exposed face.
Native layers keep their specified angle; transported soil uses a loose,
volume-weighted friction coefficient. These are scenario parameters, not a
rock-mechanics calculation. Legacy snapshots retain their original law.
"""
import math
import numpy as np


def legacy(config):return config.slope_model=='legacy'


def minimum_slope(config,*,loose=False):
    if legacy(config):return math.tan(math.radians(config.trench.angle_deg or 35.))
    profile=config.soil_profile
    angle=min(getattr(m,'loose_angle_deg' if loose else 'native_angle_deg') for m in profile.materials) if profile else config.terrain.repose_deg
    return math.tan(math.radians(angle))


def _columns(terrain,indices,heights=None):
    soil=terrain.soil;profile=soil.profile
    xx,yy=soil._coordinates(indices)
    values=soil._read(xx,yy).reshape(len(profile.materials)+1,-1,soil.factor**2).mean(axis=2)
    if heights is not None:
        change=terrain.height.ravel()[indices]-heights
        values,_=soil._removed(values,np.maximum(change,0.))
        weakest=min(range(len(profile.materials)),key=lambda k:profile.materials[k].loose_angle_deg)
        values[weakest+1]+=np.maximum(-change,0.)
    loose=values[1:].sum(axis=0)
    friction=np.array([math.tan(math.radians(m.loose_angle_deg)) for m in profile.materials])
    mixed=np.divide((values[1:]*friction[:,None]).sum(axis=0),loose,
                    out=np.full_like(loose,float(friction.min())),where=loose>1e-12)
    boundaries=np.r_[0.,profile.boundaries,np.inf]
    thickness=np.maximum(0.,boundaries[1:,None]-np.maximum(values[0],boundaries[:-1,None]))
    angles=np.array([math.tan(math.radians(m.native_angle_deg)) for m in profile.materials])
    return np.vstack((loose,thickness)),np.vstack((mixed,np.broadcast_to(angles[:,None],thickness.shape)))


def _run(depth,data):
    remaining=np.maximum(depth,0.);result=np.zeros_like(remaining)
    for thickness,slope in zip(*data):
        taken=np.minimum(remaining,thickness);result+=taken/slope;remaining=np.maximum(0.,remaining-taken)
    return result


def _depth(run,data):
    remaining=np.maximum(run,0.);result=np.zeros_like(remaining)
    for thickness,slope in zip(*data):
        taken=np.minimum(remaining*slope,thickness);result+=taken;remaining=np.maximum(0.,remaining-taken/slope)
    return result


def pair_transfer(terrain,a,b,distance,*,height=None):
    """Equal/opposite height transfer required to make each face stable."""
    h=(terrain.height if height is None else height).ravel()
    difference=np.abs(h[a]-h[b]);c=terrain.config
    if legacy(c) or terrain.soil is None:
        return np.maximum(0.,difference-minimum_slope(c)*distance-c.terrain.avalanche_epsilon)/2
    # A cheap bound avoids material reads for already shallow neighbouring cells.
    weakest=min(minimum_slope(c),minimum_slope(c,loose=True))
    result=np.zeros_like(difference);possible=difference>weakest*distance+c.terrain.avalanche_epsilon
    if not np.any(possible):return result
    ids=np.flatnonzero(possible);source=np.where(h[a[ids]]>=h[b[ids]],a[ids],b[ids])
    data=_columns(terrain,source,h[source] if height is not None else None)
    d=difference[ids];required=_run(d,data)
    active=required>distance+c.terrain.avalanche_epsilon/weakest
    if not np.any(active):return result
    ids=ids[active];d=d[active];data=tuple(v[:,active] for v in data)
    # The law is linear between layer contacts. Either end of the exposed
    # interval can cross a contact as equal soil heights are transferred.
    contacts=np.cumsum(data[0],axis=0)[:-1]
    events=np.sort(np.clip(np.vstack((np.zeros_like(d),d/2,contacts,d-contacts)),0,d/2),axis=0)
    previous=events[0];previous_run=_run(d,data);solved=np.zeros(len(d),dtype=bool);root=d/2
    threshold=distance+c.terrain.avalanche_epsilon/weakest
    for point in events[1:]:
        run=_run(d-point,data)-_run(point,data)
        crossing=~solved & (run<=threshold)
        fraction=np.divide(previous_run-threshold,previous_run-run,
                           out=np.zeros_like(run),where=previous_run>run)
        root=np.where(crossing,previous+(point-previous)*np.clip(fraction,0,1),root)
        solved|=crossing;previous=point;previous_run=run
    result[ids]=root
    return result


def envelope(terrain,offset,height,mode):
    """Explicit project angle, or least-width stable layered excavation."""
    if not hasattr(terrain,'config'):
        return height+offset*terrain.slope if mode=='dig' else height-offset*terrain.slope
    c=terrain.config;baseline=terrain.height
    if c.trench.angle_deg is not None:
        ramp=offset*math.tan(math.radians(c.trench.angle_deg))
        return height+ramp if mode=='dig' else height-ramp
    if mode=='fill':return height-offset*minimum_slope(c,loose=True)
    if terrain.soil is None:return height+offset*minimum_slope(c)
    result=baseline.copy();depth=np.maximum(baseline-height,0.)
    weakest=min(minimum_slope(c),minimum_slope(c,loose=True))
    ids=np.flatnonzero((depth>0)&(offset<depth/weakest+1e-10))
    for start in range(0,len(ids),65536):
        part=ids[start:start+65536];data=_columns(terrain,part)
        run=np.maximum(0.,_run(depth.ravel()[part],data)-offset.ravel()[part])
        result.ravel()[part]=baseline.ravel()[part]-_depth(run,data)
    return result


def validate_target(terrain,target,work):
    if legacy(terrain.config):return
    n=target.shape[0];grid=np.arange(n*n).reshape(n,n)
    for oy,ox in ((0,1),(1,0),(1,1),(1,-1)):
        ya,yb=slice(0,n-oy),slice(oy,n)
        xa,xb=slice(max(0,-ox),min(n,n-ox)),slice(max(0,ox),min(n,n+ox))
        active=work[ya,xa]|work[yb,xb]
        a,b=grid[ya,xa][active],grid[yb,xb][active]
        for start in range(0,len(a),65536):
            excess=pair_transfer(terrain,a[start:start+65536],b[start:start+65536],terrain.cell*math.hypot(oy,ox),height=target)
            if np.any(excess>1e-6):
                raise ValueError('Проектный профиль круче угла откоса раскрытых материалов: уменьшите угол или выберите «Авто · минимальная ширина»; проверьте переход между мазками')
