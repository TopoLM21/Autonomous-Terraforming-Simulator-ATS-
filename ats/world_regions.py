"""Route-independent, globally addressed interest windows for one physical group."""
import numpy as np


def interest_windows(region,simulation,limit=4):
    c=simulation.config;geo=c.georeference
    if not geo:return []
    n=c.terrain.n;cell=c.terrain.cell_size;side=n*cell;stride=side/2
    extent=np.asarray(region.raster.extent_m);maximum=extent-side
    if np.any(maximum<0):return []
    origin=np.asarray(geo['origin_m']);phase=np.mod(origin,cell)
    site=simulation.shared_site
    sims=[u.simulation for u in site.project.units] if site else [simulation]
    points=[]
    for s in sims:
        points.append(origin+s.base.position[:2])
        if s.base.destination is not None:points.append(origin+s.base.destination[:2])
        for d in s.drones:
            points.append(origin+d.position[:2])
            if d.destination is not None:points.append(origin+d.destination[:2])
            if d.trip:
                points.extend([origin+np.asarray(d.trip.dig),origin+np.asarray(d.trip.dump)])
        if s.logistics and s.logistics.excavator:
            e=s.logistics.excavator;points.append(origin+e.position[:2])
            if e.destination is not None:points.append(origin+e.destination[:2])
    points=np.asarray(points);centre=points.mean(axis=0)
    # Vectorise the nine neighbours for all actors. Keep the original scores
    # (including repeated positions); avoid thousands of tiny NumPy calls.
    offsets=np.array(((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)))
    anchors=np.floor((points-side/2-phase)/stride+.5)*stride+phase
    raw=anchors[:,None,:]+offsets[None,:,:]*stride
    steps=np.clip(np.rint((raw-origin)/cell),np.ceil(-origin/cell),np.floor((maximum-origin)/cell))
    xy=origin+steps*cell
    scores=np.sum((xy+side/2-points[:,None,:])**2,axis=2)+.05*np.sum((xy+side/2-centre)**2,axis=2)
    coords,inverse=np.unique(xy.reshape(-1,2),axis=0,return_inverse=True)
    best=np.full(len(coords),np.inf);np.minimum.at(best,inverse,scores.ravel())
    rows=[(float(score),float(x),float(y)) for (x,y),score in zip(coords,best) if not np.array_equal((x,y),origin)]
    return [[x,y,n,cell] for _,x,y in sorted(rows)[:limit]]
