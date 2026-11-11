"""Read-only regional elevation display. Coordinates and heights are SI/MSL."""
import math
import numpy as np


def samples(raster,x,y):
    """Use the DEM interpolator on valid points and leave NoData as gaps."""
    x,y=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    width,height=raster.extent_m;rx,ry=raster.resolution_m
    valid=np.isfinite(x)&np.isfinite(y)&(x>=0)&(y>=0)&(x<=width)&(y<=height)
    fx=np.where(valid,x,0.)/rx;fy=np.where(valid,y,0.)/ry
    ix=np.minimum(fx.astype(int),raster.height.shape[1]-2);iy=np.minimum(fy.astype(int),raster.height.shape[0]-2)
    ux,uy=fx-ix,fy-iy
    for ox,oy,weight in ((0,0,(1-ux)*(1-uy)),(1,0,ux*(1-uy)),(0,1,(1-ux)*uy),(1,1,ux*uy)):
        valid &= (weight<=1e-12)|np.isfinite(raster.height[iy+oy,ix+ox])
    result=np.full(x.shape,np.nan)
    if valid.any():result[valid]=raster.sample(x[valid],y[valid])
    return result


def grid_key(raster,bounds=None):
    rows,cols=raster.height.shape;rx,ry=raster.resolution_m
    if bounds is None:return 0,cols-1,0,rows-1
    (left,right),(bottom,top)=bounds
    x0=int(np.clip(math.floor(left/rx)-1,0,cols-2));x1=int(np.clip(math.ceil(right/rx)+1,x0+1,cols-1))
    y0=int(np.clip(math.floor(bottom/ry)-1,0,rows-2));y1=int(np.clip(math.ceil(top/ry)+1,y0+1,rows-1))
    return x0,x1,y0,y1


def overview(raster,key=None,max_side=768):
    """Uniform display nodes, bounded per axis; zooming recovers native detail."""
    x0,x1,y0,y1=grid_key(raster) if key is None else key;rx,ry=raster.resolution_m
    nx=min(max_side,x1-x0+1);ny=min(max_side,y1-y0+1)
    # Geographic extent uses subtraction of longitudes/latitudes. Its last
    # coordinate can be a few ulps inside (count-1)*resolution; do not create
    # a false NoData border when calling the strict physical interpolator.
    width,height=raster.extent_m
    x=np.linspace(min(x0*rx,width),min(x1*rx,width),nx)
    y=np.linspace(min(y0*ry,height),min(y1*ry,height),ny)
    if nx==x1-x0+1 and ny==y1-y0+1:data=raster.height[y0:y1+1,x0:x1+1]
    else:
        data=np.empty((ny,nx))
        for start in range(0,ny,64):data[start:start+64]=samples(raster,x[None,:],y[start:start+64,None])
    frame=(x[0],y[0],(x[-1]-x[0])/(nx-1),(y[-1]-y[0])/(ny-1))
    return data,frame


def marine_samples(raster,ocean,data,frame):
    x0,y0,dx,dy=frame;rx,ry=raster.resolution_m
    x=(x0+np.arange(data.shape[1])*dx)/rx;y=(y0+np.arange(data.shape[0])*dy)/ry
    ix=np.clip(x.astype(int),0,ocean.shape[1]-2);iy=np.clip(y.astype(int),0,ocean.shape[0]-2)
    near=ocean[iy[:,None],ix]|ocean[iy[:,None]+1,ix]|ocean[iy[:,None],ix+1]|ocean[iy[:,None]+1,ix+1]
    return near&np.isfinite(data)&(data<0)


LAND_STOPS=np.array([-11000.,-500.,-150.,-50.,0.,50.,150.,500.,1500.,9000.])
LAND_COLORS=np.array([[45,24,81],[66,36,107],[102,65,150],[160,130,185],[237,228,202],
                      [221,195,137],[188,153,96],[147,116,77],[110,97,81],[243,239,227]])
SEA_COLOR=np.array([31,100,154],dtype=np.uint8)
SELECTED_COLOR=np.array([98,200,133],dtype=np.uint8)
NODATA_COLOR=np.array([77,81,89],dtype=np.uint8)


def colors(height,marine=None,*,mode='msl',level=0.,thickness=10.):
    if mode not in ('height','msl','below','band') or not math.isfinite(level) or not math.isfinite(thickness) or thickness<=0:
        raise ValueError('Неверный режим отображения высот')
    finite=np.isfinite(height);safe=np.where(finite,height,0.)
    if not finite.any():return np.broadcast_to(NODATA_COLOR,(*height.shape,3)).copy()
    if mode=='height':
        lo,hi=float(height[finite].min()),float(height[finite].max())
        if hi-lo<1e-9:hi=lo+1.
        stops=[lo,(lo+hi)/2,hi];palette=np.array([[48,76,107],[169,171,123],[247,233,189]])
    else:stops,palette=LAND_STOPS,LAND_COLORS
    rgb=np.stack([np.interp(safe,stops,palette[:,i]) for i in range(3)],axis=-1)
    sea=np.zeros(height.shape,dtype=bool) if marine is None else marine
    if mode in ('below','band'):
        selected=(safe<=level) if mode=='below' else (np.abs(safe-level)<=thickness/2)
        rgb=rgb*.38+np.array([104,108,115])*.62;rgb[selected&~sea]=SELECTED_COLOR
    if mode!='height':rgb[sea]=SEA_COLOR
    rgb[~finite]=NODATA_COLOR
    return rgb.clip(0,255).astype(np.uint8)


# Marching squares, in node coordinates. Exclude every cell touching NoData.
_EDGES=(((0,0),(0,1)),((0,0),(1,0)),((1,0),(1,1)),((0,1),(1,1)))
_SIDES=((),(0,1),(1,2),(0,2),(0,3),(1,3),(0,1,2,3),(2,3),
        (2,3),(0,1,2,3),(1,3),(0,3),(0,2),(1,2),(0,1),())


def level_segments(height,level,frame):
    """Vectorized isolines, without a Python loop over all DEM cells."""
    a,b,c,d=height[:-1,:-1],height[:-1,1:],height[1:,:-1],height[1:,1:]
    valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(c)&np.isfinite(d)
    index=(a<level).astype(np.uint8)+2*(b<level)+4*(c<level)+8*(d<level)
    yy,xx=np.nonzero(valid&(index>0)&(index<15));cases=index[yy,xx];parts=[]
    def endpoint(x,y,edge):
        (px,py),(qx,qy)=_EDGES[edge]
        start=height[y+py,x+px];stop=height[y+qy,x+qx]
        fraction=(level-start)/(stop-start)
        return np.column_stack((x+px+fraction*(qx-px),y+py+fraction*(qy-py)))
    for case,sides in enumerate(_SIDES):
        chosen=cases==case
        if not sides or not chosen.any():continue
        x,y=xx[chosen],yy[chosen]
        for pair in range(0,len(sides),2):
            parts.append(np.stack((endpoint(x,y,sides[pair]),endpoint(x,y,sides[pair+1])),axis=1))
    result=np.concatenate(parts) if parts else np.empty((0,2,2))
    x0,y0,dx,dy=frame;result*=np.array([dx,dy]);result+=np.array([x0,y0])
    return result.reshape(-1,2)


def profile(raster,mode,position=0.,points=None,max_points=2048):
    width,height=raster.extent_m;rx,ry=raster.resolution_m
    if mode=='x':
        axis=np.linspace(0,width,min(max_points,raster.height.shape[1]));x=axis;y=np.full_like(x,position)
    elif mode=='y':
        axis=np.linspace(0,height,min(max_points,raster.height.shape[0]));y=axis;x=np.full_like(y,position)
    elif mode=='route':
        p=np.asarray(points,dtype=float)
        if p.ndim!=2 or len(p)<2:return np.array([]),np.array([])
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        if distance[-1]<=0:return np.array([]),np.array([])
        axis=np.linspace(0,distance[-1],min(max_points,max(2,int(distance[-1]/min(rx,ry))+1)))
        x=np.interp(axis,distance,p[:,0]);y=np.interp(axis,distance,p[:,1])
    else:raise ValueError('Неизвестное направление сечения')
    return axis,samples(raster,x,y)
