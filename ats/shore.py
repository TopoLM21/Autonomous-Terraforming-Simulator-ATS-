"""Conservative dry crest setback on the active DEM grid.

Eight-neighbour distance underestimates Euclidean distance, so a protected
chessboard buffer is conservative at diagonal coasts. A max-plus distance
transform anchors the retaining slope to the actual dry crest elevations.
This changes a new design, never the terrain or a material ledger.
"""
import math
import base64
import zlib
import numpy as np


def spread(values,drop):
    """Max(source - drop * chessboard_distance), O(rows * columns)."""
    rows,cols=values.shape
    a=np.pad(np.asarray(values,dtype=float),1,constant_values=-np.inf)
    x=np.arange(cols+2)*drop
    for order,previous in ((range(1,rows+1),-1),(range(rows,0,-1),1)):
        for y in order:
            a[y,1:-1]=np.maximum(a[y,1:-1],np.maximum.reduce((a[y+previous,:-2],a[y+previous,1:-1],a[y+previous,2:]))-drop)
            a[y]=np.maximum.accumulate(a[y]+x)-x
            a[y]=np.maximum.accumulate((a[y]-x)[::-1])[::-1]+x
    return a[1:-1,1:-1]


def crest_bank(terrain,marine,depth,offset,cell,*,valid=None):
    distance=-spread(np.where(marine,0.,-np.inf),1.)
    # Two grid cells cover the sampled shoreline and the footprint of an edge
    # cell; no sub-cell coastline accuracy is claimed from an interpolated DEM.
    protected=distance<=math.ceil(offset/cell)+2
    if valid is not None:protected &= valid
    reference=getattr(terrain,'reference',terrain.height)
    seeds=np.where(protected,np.maximum(reference,0.),-depth)
    cap=np.maximum(-depth,spread(seeds,cell*terrain.slope*.8))
    return protected,cap


def prepare(landscape,config,origin,side):
    """Prepare immutable design constraints with a sufficient world-space halo.

    The elevation Lipschitz bound limits dry buffer crest height; beyond the
    halo even its tallest source cannot raise the cap above the canal bottom.
    A bounded grid and strip sampling avoid unbounded allocation for extreme
    parameters. This is a design field, not a second mutable terrain.
    """
    from types import SimpleNamespace
    from .slopes import minimum_slope
    from .procedural import sample
    from .channel import ocean_nodes
    from .channel_site import SiteError
    c=config;plan=c.plan;raster=landscape.raster;n=c.terrain.n;cell=c.terrain.cell_size
    slope=minimum_slope(c);depth=max(-s.height for s in plan.segments) if plan.segments else 0.
    offset=plan.channel['shore_offset_m'];amplitude=(landscape.generation or {}).get('amplitude_m',0.)
    gradient=getattr(raster,'_shore_gradient',None)
    if gradient is None:
        rx,ry=raster.resolution_m
        gx=float(np.nanmax(np.abs(np.diff(raster.height,axis=1))))/rx
        gy=float(np.nanmax(np.abs(np.diff(raster.height,axis=0))))/ry
        gradient=math.hypot(gx,gy);raster._shore_gradient=gradient
    buffer=(math.ceil(offset/cell)+2)*cell
    highest=math.sqrt(2)*gradient*(buffer+2*cell)+2*amplitude
    halo=math.ceil((2*buffer+(highest+depth)/(slope*.8))/cell)+3
    count=n+2*halo
    rx,ry=raster.resolution_m;nodes=ocean_nodes(raster)
    lower=np.floor((np.asarray(origin)-halo*cell)/[rx,ry]).astype(int)-1
    upper=np.ceil((np.asarray(origin)+side+halo*cell)/[rx,ry]).astype(int)+2
    lo=np.clip(lower,[0,0],np.array(nodes.shape[::-1])-1);hi=np.clip(upper,[0,0],nodes.shape[::-1])
    def encode(a):return base64.b64encode(zlib.compress(a.tobytes(),6)).decode('ascii')
    if not np.any(nodes[lo[1]:hi[1],lo[0]:hi[0]]):
        plan.channel['shore_data']=dict(shape=[n,n],origin_m=list(origin),cell_m=cell,
            protected=encode(np.zeros((n*n+7)//8,dtype=np.uint8)),cap=encode(np.full((n,n),-depth,dtype='<f8')))
        return
    if count>2048:
        raise SiteError(f'Береговой профиль требует сетку с окружением {count}×{count}; предел подготовки 2048×2048. Увеличьте рабочее окно с более крупной ячейкой либо уменьшите геометрию проекта')
    start=np.asarray(origin)-halo*cell
    axis=(np.arange(count)+.5)*cell
    reference=np.empty((count,count),dtype=float)
    factor=round(cell/landscape.cell_size)
    generated=landscape.generation is not None and amplitude>0
    # The mean uses exactly the permanent fine columns used by read_window.
    fine=landscape.cell_size if generated else cell;mult=factor if generated else 1
    fine_axis=(np.arange(count*mult)+.5)*fine
    width,height=raster.extent_m
    for row in range(0,count*mult,64):
        stop=min(row+64,count*mult)
        x=np.clip(start[0]+fine_axis,0,width)[None,:]
        y=np.clip(start[1]+fine_axis[row:stop],0,height)[:,None]
        h=sample(raster,x,y,landscape.generation)
        reference[row//mult:stop//mult]=h.reshape((stop-row)//mult,mult,count,mult).mean(axis=(1,3))
    x=start[0]+axis[None,:];y=start[1]+axis[:,None]
    fake=SimpleNamespace(height=reference,reference=reference,x=x-start[0],y=y-start[1],slope=slope)
    ix=np.clip(np.floor(x/rx).astype(int),0,nodes.shape[1]-2)
    iy=np.clip(np.floor(y/ry).astype(int),0,nodes.shape[0]-2)
    water=(reference<0)&(nodes[iy,ix]|nodes[iy+1,ix]|nodes[iy,ix+1]|nodes[iy+1,ix+1])
    valid=(x>=0)&(x<=width)&(y>=0)&(y<=height)
    water &= valid
    protected,cap=crest_bank(fake,water,depth,offset,cell,valid=valid)
    patch=(slice(halo,halo+n),slice(halo,halo+n))
    plan.channel['shore_data']=dict(shape=[n,n],origin_m=list(origin),cell_m=cell,
        protected=encode(np.packbits(protected[patch])),cap=encode(cap[patch].astype('<f8')))


def decode(terrain,data):
    shape=tuple(data['shape']);size=math.prod(shape)
    if shape!=terrain.height.shape or size>1024**2:raise ValueError('Неверная сетка береговой защиты')
    config=getattr(terrain,'config',None)
    if config is not None:
        if data.get('origin_m')!=(config.georeference or {}).get('origin_m') or data.get('cell_m')!=config.terrain.cell_size:
            raise ValueError('Береговая защита принадлежит другому рабочему окну')
    cached=getattr(terrain,'_shore_cache',None)
    if cached is not None and cached[0] is data:return cached[1:]
    def unpack(value,limit):
        stream=zlib.decompressobj();raw=stream.decompress(base64.b64decode(value,validate=True),limit+1)
        if len(raw)>limit or not stream.eof or stream.unused_data:raise ValueError('Повреждена береговая защита')
        return raw
    raw=unpack(data['protected'],(size+7)//8)
    if len(raw)!=(size+7)//8:raise ValueError('Повреждена береговая маска')
    protected=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),count=size).reshape(shape).astype(bool)
    raw=unpack(data['cap'],size*8)
    if len(raw)!=size*8:raise ValueError('Повреждены высоты береговой защиты')
    cap=np.frombuffer(raw,dtype='<f8').reshape(shape)
    if not np.all(np.isfinite(cap)):raise ValueError('Неверные высоты береговой защиты')
    terrain._shore_cache=(data,protected,cap)
    return protected,cap
