"""Initial square-like formation, measured with fully deployed PV and service space.

Only a new delivery is arranged. Live carriers keep their relative coordinates.
The rectangular cell dimensions follow the real wing sweep, not chassis area.
"""
from copy import deepcopy
import math
import numpy as np

MODE='compact_grid_v1'


def move(module,offset):
    old=np.asarray(module['offset_m']);delta=np.asarray(offset)-old
    module['offset_m']=list(map(float,offset))
    cg=module['centre_of_gravity_m'];module['centre_of_gravity_m']=[cg[0]+float(delta[0]),cg[1]+float(delta[1]),cg[2]]


def dock_offsets(spec):
    if 'dock_offsets' in spec:
        length=spec['parameters']['platform_length_m']
        spec['dock_offsets']=[[a['offset_m'][0],a['offset_m'][1]+length/4] for a in spec['platforms'] if a.get('role')=='SERVICE'][:len(spec['dock_offsets'])]


def polygons(spec,point):
    from .base_layout import rotate_offset
    result=[]
    for x,y,w,h in spec.get('access_lanes_local',()):
        corners=np.array([rotate_offset(spec['parameters'],v) for v in ((x,y),(x+w,y),(x+w,y+h),(x,y+h))])+np.asarray(point[:2])
        result.append(corners)
    return result


def rectangles(spec,point):
    from .site_ground import polygon_bounds
    return [polygon_bounds(p) for p in polygons(spec,point)]


def active_polygons(simulation):
    w=simulation.logistics
    if w is None or simulation.base.destination is not None or getattr(w,'regional_folded',False) or w.field_rect is None:return []
    a=getattr(w,'solar_array',None);b=getattr(w,'battery_platforms',None)
    if a is not None and a.stowed and a.mode=='FOLD' and (b is None or b.stowed):return []
    return polygons(getattr(simulation.base,'layout_spec',{}),simulation.base.position)


def active_rectangles(simulation):
    from .site_ground import polygon_bounds
    return [polygon_bounds(p) for p in active_polygons(simulation)]


def safe(spec,point,terrain):
    from .site_ground import footprint_safe
    blocked=terrain.design_work|terrain.dump_mask
    if terrain.protected is not None:blocked=blocked|terrain.protected
    site=getattr(terrain,'shared_site',None)
    return all(footprint_safe(terrain,r,blocked,grade_deg=8.,relief_m=float('inf')) and
        (site is None or not site.ground_occupied(getattr(terrain,'owner_id',None),r)) for r in rectangles(spec,point))


def arrange(config,spec,world=None):
    from .self_solar import _rect
    from .battery_platforms import deployment_rectangles
    from .base_layout import layout_footprints
    previous=getattr(getattr(getattr(world,'sim',None),'base',None),'layout_spec',{})
    if world is not None:
        if previous.get('formation')!=MODE:return
        for key in ('formation','access_lanes_local','grid','hub_slots_local','service_header_y','service_aisles_x'):
            spec[key]=deepcopy(previous[key])
        old={a['id']:a for a in previous['platforms']};used={tuple(a['offset_m']) for a in previous['platforms'] if a.get('role')!='PV'}
        for a in spec['platforms']:
            if a['id'] in old:move(a,old[a['id']]['offset_m'])
            elif a.get('role')!='PV':
                slot=next((xy for xy in spec['hub_slots_local'] if tuple(xy) not in used),None)
                if slot is None:raise ValueError('Компактный сервисный блок заполнен: новый носитель требует нового шаблона; существующие платформы не переставлены')
                move(a,slot);used.add(tuple(slot))
        dock_offsets(spec);return
    p=spec['parameters'];sp=spec['self_solar'];core=[a for a in spec['platforms'] if a.get('role')!='PV'];pv=[a for a in spec['platforms'] if a.get('role')=='PV']
    services=[a for a in core if a.get('role')=='SERVICE'];other=[a for a in core if a.get('role')!='SERVICE']
    width=p['platform_width_m'];length=p['platform_length_m'];bp=spec.get('battery_platforms',{})
    apron=max(bp.get('apron_width_m',0.),bp.get('support_extension_m',0.))
    px=width+2*apron+3.;py=length+p['platform_gap_m']
    rows=max(len(services),math.ceil(math.sqrt(max(1,len(core))*px/py)))
    raw={}
    if services:
        for i,a in enumerate(services):raw[a['id']]=[0.,i*py]
        for i,a in enumerate(other):raw[a['id']]=[(1+i//rows)*px,(i%rows)*py]
    else:
        cols=max(1,math.ceil(len(core)/rows))
        for i,a in enumerate(other):raw[a['id']]=[(i//rows)*px,(i%rows)*py]
    anchor=np.asarray(raw[core[0]['id']]);slots=[]
    cols=max(1,1+math.ceil(len(other)/rows)) if services else max(1,math.ceil(len(core)/rows))
    for col in range(cols):
        for row in range(rows):slots.append((np.array([col*px,row*py])-anchor).tolist())
    for a in core:move(a,np.asarray(raw[a['id']])-anchor)
    dock_offsets(spec)
    local={**spec,'parameters':{**p,'heading_deg':0.},'platforms':core}
    boxes=[*layout_footprints(local,[0.,0.]),*deployment_rectangles(local,[0.,0.])]
    f=config.fleet
    for xy in spec.get('dock_offsets',[]):
        for shift in (0.,-max(f.waiting_offset,spec['hub_radius_m']+3+f.separation)):
            boxes.append((xy[0]+shift-f.separation,xy[1]-f.separation,2*f.separation,2*f.separation))
    header=min(y for x,y,w,h in boxes)-2.
    aisles=[(col+.5)*px-anchor[0] for col in range(cols-1)]
    hub_lanes=[(x-1.5,header-1.5,3.,(rows-1)*py+(length+p['platform_gap_m'])/2-header+1.5) for x in aisles]
    if aisles:hub_lanes.append((min(aisles)-1.5,header-1.5,max(3.,max(aisles)-min(aisles)+3.),3.))
    boxes+=hub_lanes
    x0=min(x for x,y,w,h in boxes);y0=min(y for x,y,w,h in boxes)
    hubw=max(x+w for x,y,w,h in boxes)-x0;hubh=max(y+h for x,y,w,h in boxes)-y0
    local={**spec,'parameters':{**p,'heading_deg':0.}}
    sweep=_rect(local,pv[0],[0.,0.]);cw=max(hubw,sweep[2]);ch=sweep[3]
    gx,gy=sp['column_access_m'],sp['row_access_m'];hubrows=math.ceil((hubh+gy)/(ch+gy))
    candidates=[]
    for columns in range(1,len(pv)+hubrows+1):
        nrows=max(hubrows,math.ceil((len(pv)+hubrows)/columns))
        w=columns*cw+(columns-1)*gx;h=nrows*ch+(nrows-1)*gy
        candidates.append(((max(w,h),w*h,abs(w-h),columns),columns,nrows,w,h))
    _,columns,nrows,w,h=min(candidates)
    cells=[(col,row) for row in range(nrows) for col in range(columns) if col!=0 or row>=hubrows]
    for a,(col,row) in zip(pv,cells):
        move(a,[x0+col*(cw+gx)+cw/2,y0+row*(ch+gy)+ch/2]);a['pv_fixed']=True
    lanes=list(hub_lanes)
    for col in range(columns-1):lanes.append((x0+(col+1)*cw+col*gx,y0,gx,h))
    for row in range(nrows-1):
        start=x0+cw+gx if row<hubrows-1 else x0
        if x0+w>start:lanes.append((start,y0+(row+1)*ch+row*gy,x0+w-start,gy))
    spec.update(formation=MODE,access_lanes_local=lanes,hub_slots_local=slots,
        service_header_y=header,service_aisles_x=aisles,
        grid=dict(columns=columns,rows=nrows,hub_rows=hubrows,cell_width_m=cw,cell_length_m=ch,
            deployed_dimensions_local_m=[w,h],deployed_bounds_local_m=[x0,y0,w,h],column_access_m=gx,row_access_m=gy))
