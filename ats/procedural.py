"""Versioned, coordinate-seeded detail over an immutable coarse DEM.

This adds conditional small-scale relief, not measured elevation accuracy.
No tile-local RNG, Python hash, water or material edits are involved.
"""
import math
import numpy as np


def default_generation(seed=42,amplitude=.1,scale=16.):
    result=dict(version=1,seed=seed,amplitude_m=amplitude,scale_m=scale)
    validate_generation(result);return result


def validate_generation(spec):
    if spec is None:return
    if (not isinstance(spec,dict) or set(spec)!={'version','seed','amplitude_m','scale_m'} or
            type(spec['version']) is not int or spec['version']!=1 or
            type(spec['seed']) is not int or not 0<=spec['seed']<2**32 or
            any(type(spec[k]) not in (int,float) or not math.isfinite(spec[k]) for k in ('amplitude_m','scale_m')) or
            not 0<=spec['amplitude_m']<=1 or not 4<=spec['scale_m']<=1024):
        raise ValueError('Генератор: версия 1, seed 0…2³²−1, неровности 0…1 м, масштаб 4…1024 м')


def signature(spec):
    return None if spec is None else (spec['version'],spec['seed'],float(spec['amplitude_m']),float(spec['scale_m']))


def _hash(x,y,seed):
    with np.errstate(over='ignore'):
        z=(x.astype(np.uint64)*np.uint64(0x9E3779B185EBCA87)^y.astype(np.uint64)*np.uint64(0xC2B2AE3D27D4EB4F)^np.uint64(seed))
        z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        z=z^(z>>np.uint64(31))
    return ((z>>np.uint64(11)).astype(np.float64)/2**53)*2-1


def _noise(x,y,scale,seed):
    fx,fy=x/scale,y/scale;ix,iy=np.floor(fx).astype(np.int64),np.floor(fy).astype(np.int64)
    u,v=fx-ix,fy-iy
    u=u*u*u*(u*(u*6-15)+10);v=v*v*v*(v*(v*6-15)+10)
    a=_hash(ix,iy,seed);b=_hash(ix+1,iy,seed)
    c=_hash(ix,iy+1,seed);d=_hash(ix+1,iy+1,seed)
    return (a+u*(b-a))*(1-v)+(c+u*(d-c))*v


def sample(raster,x,y,spec):
    baseline=raster.sample(x,y)
    if spec is None or spec['amplitude_m']==0:return baseline
    sx,sy=raster.to_source(x,y)
    if raster.metadata['crs']=='EPSG:4326':
        radius=6_371_008.8
        sx=radius*np.deg2rad(sx)
        sy=radius*np.log(np.tan(np.pi/4+np.deg2rad(sy)/2))
    noise=sum(weight*_noise(sx,sy,spec['scale_m']/divisor,spec['seed']+octave)
        for octave,(divisor,weight) in enumerate(((1,.6),(4,.3),(16,.1))))
    rx,ry=raster.resolution_m;u=np.mod(x/rx,1.);v=np.mod(y/ry,1.)
    # Coarse DEM nodes and cell edges remain anchors. The envelope and the
    # coordinate noise are shared across all fine-tile boundaries.
    envelope=256*u*u*(1-u)**2*v*v*(1-v)**2
    detail=spec['amplitude_m']*noise*envelope
    detail=np.clip(detail,-np.abs(baseline)*.25,np.abs(baseline)*.25)
    return baseline+detail
