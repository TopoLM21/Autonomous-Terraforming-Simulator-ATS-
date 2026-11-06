"""Swept separation checks; deterministic local detours, no impact dynamics."""
import numpy as np


def ground_path_clear(simulation,target):
    """A base cannot drive through its own aircraft parked beside the docks.

    Docked aircraft travel on the base. Other landed aircraft retain their real
    positions; shifting a nominal waiting point never moves the parked machine.
    Use a continuous swept rectangle check, not just the destination footprint.
    """
    s=simulation;parked=[d.position[:2] for d in s.drones if d.grounded and not d.docked]
    if not parked:return True
    b=s.base;r=s.config.fleet.separation/2
    if b.config.layout is not None:
        from .base_layout import layout_footprints
        boxes=np.asarray(list(layout_footprints(b.layout_spec,[0.,0.])))
    else:boxes=np.array([[-b.config.radius,-b.config.radius,2*b.config.radius,2*b.config.radius]])
    positions=np.asarray(parked)[:,None,:]
    lo=positions-(boxes[:,:2]+boxes[:,2:])-r;hi=positions-boxes[:,:2]+r
    start=b.position[:2];delta=np.asarray(target)[:2]-start
    enter=np.zeros(lo.shape[:2]);leave=np.ones_like(enter);possible=np.ones_like(enter,dtype=bool)
    for axis in (0,1):
        if abs(delta[axis])<1e-12:possible &= (start[axis]>=lo[:,:,axis])&(start[axis]<=hi[:,:,axis])
        else:
            a=(lo[:,:,axis]-start[axis])/delta[axis];z=(hi[:,:,axis]-start[axis])/delta[axis]
            enter=np.maximum(enter,np.minimum(a,z));leave=np.minimum(leave,np.maximum(a,z))
    return not bool(np.any(possible&(enter<=leave)))


def closest_separation(a0, a1, b0, b1):
    relative = np.asarray(a0)-b0
    velocity = (np.asarray(a1)-a0)-(np.asarray(b1)-b0)
    norm = float(np.dot(velocity, velocity))
    time = np.clip(-np.dot(relative, velocity)/norm, 0, 1) if norm > 1e-18 else 0
    return float(np.linalg.norm(relative+velocity*time))


def conflicts(drone, old, new, others, starts, separation):
    for other in others:
        if other is drone:
            continue
        before=starts[other.identifier]
        if (min(old[2],new[2])-max(before[2],other.position[2])>=separation or
                min(before[2],other.position[2])-max(old[2],new[2])>=separation):continue
        if closest_separation(old, new, before, other.position) < separation-1e-9:
            return other
    return None
