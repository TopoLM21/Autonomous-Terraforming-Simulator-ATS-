"""Numeric cross sections and joined longitudinal profiles of plan segments."""
from dataclasses import dataclass
import numpy as np
from .plan import PlanSegment, sample_line, cross_section


@dataclass
class Profile:
    name: str
    axis: np.ndarray
    current: np.ndarray
    target: np.ndarray
    reference: np.ndarray
    points: np.ndarray
    boundaries: tuple = ()


def segments_for(config):
    if config.plan is not None:
        return config.plan.segments
    q = config.trench
    return [PlanSegment([[q.x-q.length/2,q.y],[q.x+q.length/2,q.y]],"dig",q.width,-q.depth,"Канава")]


def sample_surface(surface, points, cell):
    indices = np.floor(points/cell).astype(int)
    indices = indices.clip(0,surface.shape[0]-1)
    return surface[indices[:,1],indices[:,0]].copy()


def profiles(terrain, segments, selection=None, direction="cross", fraction=.5):
    chosen = segments if selection is None else [segments[selection]]
    result = []
    side = terrain.cell*terrain.height.shape[0]
    for segment in chosen:
        if direction == "cross":
            depth = max(abs(float(terrain.height.min())-segment.height),abs(float(terrain.height.max())-segment.height))
            extent = min(side/2,segment.width/2+depth/terrain.slope+1)
            points, axis = cross_section(segment,fraction,extent,terrain.cell/2)
        else:
            points, axis = sample_line(segment.points,terrain.cell/2)
        valid = ((points>=0)&(points<side)).all(axis=1)
        points, axis = points[valid],axis[valid]
        samples = [sample_surface(surface,points,terrain.cell) for surface in (terrain.height,terrain.target,terrain.reference)]
        result.append(Profile(segment.name,axis,*samples,points))
    if direction == "along" and selection is None and len(result)>1:
        axes, current, target, reference, points, boundaries = [],[],[],[],[],[]
        offset = 0.
        for profile in result:
            if axes:
                boundaries.append(offset)
                for array in (axes,current,target,reference):array.append(np.array([np.nan]))
                points.append(np.array([[np.nan,np.nan]]))
            axes.append(profile.axis+offset)
            current.append(profile.current);target.append(profile.target);reference.append(profile.reference);points.append(profile.points)
            offset += float(profile.axis[-1]) if len(profile.axis) else 0.
        result = [Profile("Сквозной по всем сегментам",np.concatenate(axes),np.concatenate(current),np.concatenate(target),np.concatenate(reference),np.concatenate(points),tuple(boundaries))]
    return result
