"""Derived work geometry, never a second terrain or material ledger.

Only coordinates and ordering are cached. Heights, project floors, access,
capacity, swept routes and support guards are always read from live terrain.
"""
import numpy as np


def work_geometry(fleet):
    t=fleet.terrain
    entry=np.asarray(fleet.ramp['entry_m']);end=np.asarray(fleet.ramp['end_m'])
    delta=end-entry;length=float(np.linalg.norm(delta));unit=delta/max(length,1e-9)
    ids=np.flatnonzero(fleet.work_mask)
    x=t.x.ravel()[ids];y=t.y.ravel()[ids]
    along=(x-entry[0])*unit[0]+(y-entry[1])*unit[1]
    lateral=np.abs((x-entry[0])*unit[1]-(y-entry[1])*unit[0])
    radius=fleet.excavator.radius
    stations=np.r_[np.arange(-radius-1.,0.,t.cell),np.arange(0.,length+1e-8,t.cell)]
    return dict(ids=ids,x=x,y=y,along=along,lateral=lateral,
                stations=stations,points=entry+stations[:,None]*unit,unit=unit,
                side=np.array([-unit[1],unit[0]])*(fleet.config.ramp_width_m/2+radius+1.))


def dump_geometry(fleet):
    t=fleet.terrain;ids=np.asarray(t._dump_indices)
    x=t.x.ravel()[ids];y=t.y.ravel()[ids];anchor=fleet.base_position[:2]
    distance=(x-anchor[0])**2+(y-anchor[1])**2
    # Ties retain the terrain's stable row/column order. The fixed camp anchor
    # cannot change priority merely because a truck completed a trip elsewhere.
    order=np.argsort(-distance,kind='stable')
    return dict(ids=ids[order],anchor=tuple(anchor))


def shelf_candidates(terrain, unfinished, reach, start):
    """Yield possible track centres near the remaining work, coarse grid first.

    The index can be saved between planning ticks. A large work area needs no
    full allocation of candidate points; each finer grid skips previous cells.
    Actual support, bucket work, outbound and return routes are checked live.
    """
    rows,cols=np.unravel_index(unfinished,terrain.height.shape)
    margin=int(np.ceil(reach/terrain.cell))
    left=max(0,int(cols.min())-margin)
    top=max(0,int(rows.min())-margin)
    right=min(terrain.height.shape[1]-1,int(cols.max())+margin)
    bottom=min(terrain.height.shape[0]-1,int(rows.max())+margin)
    stride=1
    while 2*stride*terrain.cell<=reach/2:
        stride*=2
    first_stride=stride
    offset=0
    while stride:
        width=(right-left)//stride+1
        count=width*((bottom-top)//stride+1)
        for local in range(max(0,start-offset),count):
            row,col=divmod(local,width)
            if stride!=first_stride and row%2==0 and col%2==0:
                continue
            yield offset+local,np.array([(left+col*stride+.5)*terrain.cell,
                                         (top+row*stride+.5)*terrain.cell])
        offset+=count
        stride//=2


def far_dump_candidates(fleet,available,cursor):
    """Resume a bounded search without forgetting blocked or partly filled cells.

    The cursor is an index into immutable geometry, not into the changing list
    of free cells. No failed transient occupancy permanently removes a place.
    """
    t=fleet.terrain;ids=fleet._dump_plan()['ids'];total=len(ids)*4
    if not total:return
    cursor=int(cursor)%total;positions=np.flatnonzero(available)
    split=int(np.searchsorted(positions,cursor//4))
    directions=((-1,0),(0,-1),(1,0),(0,1))
    first=None
    for group in (positions[split:],positions[:split]):
        for position in group:
            start=cursor%4 if position==cursor//4 else 0
            if start:first=position
            index=ids[position];x=float(t.x.ravel()[index]);y=float(t.y.ravel()[index])
            for direction in range(start,4):
                yield int(position)*4+direction,(x,y,*directions[direction])
    if first is not None:
        index=ids[first];x=float(t.x.ravel()[index]);y=float(t.y.ravel()[index])
        for direction in range(cursor%4):yield int(first)*4+direction,(x,y,*directions[direction])
