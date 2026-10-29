"""Shared ground checks for equipment: complete footprints, not just centres."""
import base64
import math
import numpy as np


def marine_mask(terrain):
    c=terrain.config;geo=c.georeference or {};channel=(c.plan.channel or {}) if c.plan else {}
    marine=channel.get('marine') or geo.get('marine')
    source=(marine['bits'],tuple(marine['origin']),tuple(marine['spacing']),tuple(marine['shape'])) if marine else None
    key=(source,id(terrain.reference),terrain.height.shape)
    if getattr(terrain,'_marine_key',None)==key:return terrain._marine_mask
    if marine is None:
        # Elevation alone cannot distinguish a dry basin from the sea. New DEM
        # windows always carry connectivity; historical regional saves recover
        # it from their embedded raster on load. Local synthetic maps have none.
        result=np.zeros_like(terrain.height,dtype=bool)
    else:
        rows,cols=marine['shape'];raw=base64.b64decode(marine['bits'],validate=True)
        nodes=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),count=rows*cols).reshape(rows,cols).astype(bool)
        rx,ry=marine['spacing'];ox,oy=marine['origin']
        ix=np.clip(np.floor((terrain.x-ox)/rx).astype(int),0,cols-2)
        iy=np.clip(np.floor((terrain.y-oy)/ry).astype(int),0,rows-2)
        result=(terrain.reference<0)&(nodes[iy,ix]|nodes[iy+1,ix]|nodes[iy,ix+1]|nodes[iy+1,ix+1])
    terrain._marine_key=key;terrain._marine_mask=result
    return result


def rectangle_patch(terrain,rect,margin=0.):
    x,y,w,h=rect;cell=terrain.cell;n=terrain.height.shape[0];side=n*cell
    if min(x-margin,y-margin)<0 or max(x+w+margin,y+h+margin)>side:return None
    return (slice(max(0,math.floor((y-margin)/cell)),min(n,math.ceil((y+h+margin)/cell))),
            slice(max(0,math.floor((x-margin)/cell)),min(n,math.ceil((x+w+margin)/cell))))


def paint_rect(terrain,mask,rect,margin=0.):
    x,y,w,h=rect;cell=terrain.cell;n=terrain.height.shape[0]
    y0,y1=max(0,math.floor((y-margin)/cell)),min(n,math.ceil((y+h+margin)/cell))
    x0,x1=max(0,math.floor((x-margin)/cell)),min(n,math.ceil((x+w+margin)/cell))
    if y0<y1 and x0<x1:mask[y0:y1,x0:x1]=True


def footprint_safe(terrain,rect,blocked=None,*,grade_deg=8.,relief_m=.6,margin=0.):
    patch=rectangle_patch(terrain,rect,margin)
    if patch is None:return False
    if np.any(marine_mask(terrain)[patch]):return False
    if blocked is not None and np.any(blocked[patch]):return False
    h=terrain.height[patch]
    if h.size==0 or not np.all(np.isfinite(h)) or np.ptp(h)>relief_m+1e-9:return False
    slope=math.tan(math.radians(grade_deg))*terrain.cell
    dy,dx=np.abs(np.diff(h,axis=0)),np.abs(np.diff(h,axis=1))
    if np.any(dy>slope+1e-9) or np.any(dx>slope+1e-9):return False
    if min(h.shape)>1:
        # The vector grade matters: two individually admissible 7° axes form
        # a diagonal slope steeper than the installer's 8° rating.
        norm=np.hypot(np.maximum(dx[:-1,:],dx[1:,:]),np.maximum(dy[:,:-1],dy[:,1:]))
        if np.any(norm>slope+1e-9):return False
    return True


def equipment_mask(terrain,simulations=()):
    from .base_layout import layout_footprints
    mask=np.zeros_like(terrain.height,dtype=bool)
    for s in simulations:
        b=s.base
        boxes=list(layout_footprints(b.layout_spec,b.position)) if b.config.layout is not None else [(b.position[0]-b.config.radius,b.position[1]-b.config.radius,2*b.config.radius,2*b.config.radius)]
        for box in boxes:paint_rect(terrain,mask,box,1.)
        for i in range(max(len(s.slots),len(s.drones))):
            points=[s.wait_position(i)]
            if s.slots:points.append(s.slot_position(min(i,len(s.slots)-1)))
            for p in points:
                paint_rect(terrain,mask,(p[0]-.5,p[1]-.5,1.,1.),s.config.fleet.separation)
        w=s.logistics
        if w:
            from .compact_layout import active_rectangles
            for rect in active_rectangles(s):paint_rect(terrain,mask,rect)
            if w.battery_platforms:
                for rect in w.battery_platforms.rectangles():paint_rect(terrain,mask,rect)
            array=getattr(w,'solar_array',None)
            if array is not None:
                for rect in w.solar_ground_rects():paint_rect(terrain,mask,rect)
            elif w.field_rect:paint_rect(terrain,mask,w.field_rect)
            if w.excavator:
                p=w.excavator.position;r=w.config.support_half_width_m+1.
                paint_rect(terrain,mask,(p[0]-r,p[1]-r,2*r,2*r))
        destination=b.destination if b.destination is not None else s.relocation_pending
        if destination is not None:
            delta=np.asarray(destination[:2])-b.position[:2]
            for x,y,width,height in boxes:
                paint_rect(terrain,mask,(x+min(0,delta[0]),y+min(0,delta[1]),width+abs(delta[0]),height+abs(delta[1])),1.)
    return mask
