"""User drawn canal alignment, coastline trimming and executable local plans.

The regional design is implicit: only the current work window is rasterized.
Ocean classification is connectivity of negative DEM nodes, not a water model.
"""
from collections import deque
from dataclasses import dataclass,field,asdict
from types import SimpleNamespace
import hashlib
import json
import math
import base64
import numpy as np
from .plan import PlanConfig,PlanSegment,sample_line,target_surface,channel_shore
from .landscape import workspace_config,corridor_profile


def _geometry(points):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or not 2<=len(p)<=4096 or not np.all(np.isfinite(p)):
        raise ValueError('Нарисуйте маршрут как минимум из двух точек')
    p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-8]]
    if len(p)<2:raise ValueError('Маршрут должен иметь положительную длину')
    return p,np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]


def _point(points,axis,distance):
    return np.array([np.interp(distance,axis,points[:,i]) for i in (0,1)])


def _trim(points,axis,lo,hi):
    return np.vstack([_point(points,axis,lo),points[(axis>lo+1e-8)&(axis<hi-1e-8)],_point(points,axis,hi)])


def _clip_edge(a,b,lo,hi):
    start,stop=0.,1.;direction=b-a
    for k in (0,1):
        if abs(direction[k])<1e-12:
            if not lo<=a[k]<=hi:return None
        else:
            t,u=(lo-a[k])/direction[k],(hi-a[k])/direction[k]
            start=max(start,min(t,u));stop=min(stop,max(t,u))
    return (a+start*direction,a+stop*direction) if start<stop else None


def ocean_nodes(raster):
    cached=getattr(raster,'_ocean_nodes',None)
    if cached is not None:return cached
    negative=np.isfinite(raster.height)&(raster.height<0)
    ocean=np.zeros(negative.shape,dtype=bool)
    edge=np.zeros_like(ocean);edge[[0,-1],:]=True;edge[:,[0,-1]]=True
    seeds=np.argwhere(edge&negative);queue=deque(map(tuple,seeds));ocean[edge&negative]=True
    rows,cols=ocean.shape
    while queue:
        y,x=queue.popleft()
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<rows and 0<=xx<cols and negative[yy,xx] and not ocean[yy,xx]:
                ocean[yy,xx]=True;queue.append((yy,xx))
    raster._ocean_nodes=ocean
    return ocean


def is_ocean(raster,x,y):
    z=raster.sample(x,y);rx,ry=raster.resolution_m
    ix=np.clip((np.asarray(x)/rx).astype(int),0,raster.height.shape[1]-2)
    iy=np.clip((np.asarray(y)/ry).astype(int),0,raster.height.shape[0]-2)
    ocean=ocean_nodes(raster)
    return (z<0)&(ocean[iy,ix]|ocean[iy+1,ix]|ocean[iy,ix+1]|ocean[iy+1,ix+1])


def destination_basin(raster,point):
    """Identify the user's destination depression, ignoring earlier dry hollows."""
    if float(raster.sample(*point))>=0:
        raise ValueError('Продлите конец маршрута внутрь нужной сухой впадины ниже 0 м MSL')
    if bool(is_ocean(raster,*point)):
        raise ValueError('Конец маршрута находится в море; выберите сухую впадину')
    rx,ry=raster.resolution_m;ix=int(point[0]/rx);iy=int(point[1]/ry)
    rows,cols=raster.height.shape;ix=min(ix,cols-2);iy=min(iy,rows-2)
    seeds=[(y,x) for y in (iy,iy+1) for x in (ix,ix+1) if raster.height[y,x]<0]
    seed=min(seeds,key=lambda p:(p[1]*rx-point[0])**2+(p[0]*ry-point[1])**2)
    basin=np.zeros((rows,cols),dtype=bool);queue=deque([seed]);basin[seed]=True
    while queue:
        y,x=queue.popleft()
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<rows and 0<=xx<cols and raster.height[yy,xx]<0 and not basin[yy,xx]:
                basin[yy,xx]=True;queue.append((yy,xx))
    return basin


def _inside_component(raster,point,mask):
    rx,ry=raster.resolution_m;ix=min(int(point[0]/rx),mask.shape[1]-2);iy=min(int(point[1]/ry),mask.shape[0]-2)
    return bool(mask[iy:iy+2,ix:ix+2].any())


def marine_window(raster,origin,side):
    rx,ry=raster.resolution_m
    col=max(0,int(origin[0]/rx));row=max(0,int(origin[1]/ry))
    right=min(raster.height.shape[1],int(math.ceil((origin[0]+side)/rx))+2)
    top=min(raster.height.shape[0],int(math.ceil((origin[1]+side)/ry))+2)
    mask=ocean_nodes(raster)[row:top,col:right]
    return dict(shape=list(mask.shape),bits=base64.b64encode(np.packbits(mask).tobytes()).decode(),
                spacing=[rx,ry],origin=[col*rx-origin[0],row*ry-origin[1]])


def _zero(raster,a,b):
    za=float(raster.sample(*a))
    for _ in range(42):
        middle=(a+b)/2;z=float(raster.sample(*middle))
        if (z>=0)==(za>=0):a=middle;za=z
        else:b=middle
    return (a+b)/2


def coastal_alignment(raster,raw_points,offset=10.):
    """Keep the inland route from the sea + offset to its first dry depression."""
    if not math.isfinite(offset) or offset<.01 or offset>10000:
        raise ValueError('Береговой отступ: 0,01…10000 м')
    p,axis=_geometry(raw_points);raster.sample(p[:,0],p[:,1])
    if float(raster.sample(*p[0]))>=0:
        direction=(p[1]-p[0])/np.linalg.norm(p[1]-p[0])
        step=max(.1,min(raster.resolution_m)/8)
        previous=p[0]
        for distance in np.arange(step,50000+step,step):
            candidate=p[0]-direction*distance
            if np.any(candidate<0) or np.any(candidate>raster.extent_m):break
            if bool(is_ocean(raster,*candidate)):
                coast=_zero(raster,candidate,previous)
                p=np.vstack([candidate,p]);p,axis=_geometry(p);break
            previous=candidate
        else:raise ValueError('Не найден морской берег перед началом: начните линию ближе к морю')
        if not bool(is_ocean(raster,*p[0])):
            raise ValueError('Начало маршрута не связано с морским берегом')
    elif not bool(is_ocean(raster,*p[0])):
        raise ValueError('Начало находится в сухой впадине. Рисуйте от морского берега к Каттаре')
    basin=destination_basin(raster,p[-1])
    spacing=max(.05,min(raster.resolution_m)/8)
    if axis[-1]/spacing>100000:raise ValueError('Слишком длинный маршрут для разрешения DEM')
    line,dist=sample_line(p,spacing);z=raster.sample(line[:,0],line[:,1])
    entering=np.flatnonzero((z[:-1]<0)&(z[1:]>=0))
    if not len(entering):raise ValueError('Маршрут не пересекает морской берег и сушу')
    i=int(entering[0]);coast=_zero(raster,line[i],line[i+1])
    coast_s=float(dist[i]+np.linalg.norm(coast-line[i]))
    exiting=np.flatnonzero((z[:-1]>=0)&(z[1:]<0)&(dist[:-1]>coast_s))
    if not len(exiting):raise ValueError('Продлите маршрут до сухой впадины с отметкой ниже 0 м MSL')
    j=None
    for crossing in exiting:
        crossing=int(crossing)
        if bool(is_ocean(raster,*line[crossing+1])):
            raise ValueError('Маршрут возвращается в море. Конец должен входить в сухую впадину')
        if _inside_component(raster,line[crossing+1],basin):j=crossing;break
    if j is None:raise ValueError('Маршрут не пересекает край выбранной сухой впадины')
    rim=_zero(raster,line[j],line[j+1]);rim_s=float(dist[j]+np.linalg.norm(rim-line[j]))
    end_s=min(float(dist[j+1]),rim_s+.05)
    start_s=coast_s+offset
    if end_s<=start_s+.1:raise ValueError('Береговой отступ больше сухопутной части маршрута')
    result=_trim(p,axis,start_s,end_s)
    if float(raster.sample(*result[0]))<=0:
        raise ValueError('После отступа начало всё ещё ниже моря; измените начальный участок линии')
    return result,coast


@dataclass
class ChannelProject:
    raw_points:list
    points:list
    coast:list
    width_m:float=20.
    depth_m:float=5.
    shore_offset_m:float=10.
    angle_deg:float=35.
    side_m:float=512.
    spoil_height_m:float=20.
    active_section:int=0
    completed:list=field(default_factory=list)
    history:list=field(default_factory=list)
    version:int=1

    def validate(self):
        _geometry(self.raw_points);_geometry(self.points)
        if self.version!=1 or not all(math.isfinite(v) for v in (self.width_m,self.depth_m,self.shore_offset_m,self.angle_deg,self.side_m,self.spoil_height_m)):
            raise ValueError('Неверный проект канала')
        if not (1<=self.width_m<=300 and .01<=self.depth_m<=500 and .01<=self.shore_offset_m<=10000 and 5<=self.angle_deg<=55 and 32<=self.side_m<=1024 and .1<=self.spoil_height_m<=200):
            raise ValueError('Проверьте ширину, глубину канала, отступ, откосы и размер участка')
        if not isinstance(self.active_section,int) or not 0<=self.active_section<self.section_count:
            raise ValueError('Неверный номер участка канала')

    @classmethod
    def create(cls,raster,points,**parameters):
        route,coast=coastal_alignment(raster,points,parameters.get('shore_offset_m',10.))
        result=cls(np.asarray(points,dtype=float).tolist(),route.tolist(),coast.tolist(),**parameters)
        result.validate();return result

    @classmethod
    def from_data(cls,data):
        result=cls(**data);result.validate();return result

    def to_data(self):return asdict(self)

    @property
    def length(self):return float(_geometry(self.points)[1][-1])

    @property
    def section_count(self):return max(1,int(math.ceil(self.length/(self.side_m/2))))

    @property
    def identity(self):
        data={k:v for k,v in self.to_data().items() if k not in ('active_section','completed','history')}
        for key in ('width_m','depth_m','shore_offset_m','angle_deg','side_m','spoil_height_m'):data[key]=float(data[key])
        return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:16]

    def origin(self,raster,index=None):
        index=self.active_section if index is None else index
        if not 0<=index<self.section_count:raise ValueError('Номер участка вне проекта')
        p,axis=_geometry(self.points);center=_point(p,axis,min(axis[-1],(index+.5)*self.side_m/2))
        extent=np.asarray(raster.extent_m)
        if np.any(extent<self.side_m):raise ValueError('Локальная карта больше исходного DEM')
        # Quarter-metre alignment preserves the landscape's permanent fine grid.
        return (np.floor(np.clip(center-self.side_m/2,0,extent-self.side_m)/.25)*.25).tolist()

    def plan(self,origin):
        p=np.asarray(self.points)-origin
        tangent=(p[1]-p[0])/np.linalg.norm(p[1]-p[0])
        # Only nearby edges are evaluated on the fine grid; no regional fine map.
        reach=self.width_m/2+1200/math.tan(math.radians(self.angle_deg))
        segments=[];starts=[];first=[];distance=0.
        for index,(a,b) in enumerate(zip(p,p[1:])):
            edge_start=distance;distance+=float(np.linalg.norm(b-a))
            if np.any(np.maximum(a,b)<-reach) or np.any(np.minimum(a,b)>self.side_m+reach):continue
            clipped=_clip_edge(a,b,-reach,self.side_m+reach)
            if clipped is None:continue
            start,end=clipped
            starts.append(edge_start+float(np.linalg.norm(start-a)));first.append(index==0)
            segments.append(PlanSegment([start.tolist(),end.tolist()],'dig',self.width_m,-self.depth_m,f'Канал · линия {index+1}'))
        return PlanConfig(segments,channel=dict(start=p[0].tolist(),tangent=tangent.tolist(),project_id=self.identity,
            chainage_starts=starts,first_edges=first))

    def configuration(self,landscape,config,index=None):
        index=self.active_section if index is None else index
        self.validate();origin=self.origin(landscape.raster,index)
        c=workspace_config(landscape,config,origin,self.side_m)
        c.trench.angle_deg=self.angle_deg;c.terrain.dump_max_height=self.spoil_height_m
        c.plan=self.plan(origin)
        c.plan.channel.update(marine=marine_window(landscape.raster,origin,self.side_m),cell_size=c.terrain.cell_size)
        n,cell=c.terrain.n,c.terrain.cell_size
        axis=(np.arange(n)+.5)*cell;x,y=np.meshgrid(axis,axis,sparse=True)
        current=np.asarray(c.initial_height)
        terrain=SimpleNamespace(height=current,reference=np.asarray(c.reference_height),x=x,y=y,
            slope=math.tan(math.radians(self.angle_deg)),cell=cell,config=c)
        target,labels=target_surface(terrain,c.plan);work=labels>0
        if np.any(current!=terrain.reference):
            original=SimpleNamespace(height=terrain.reference,reference=terrain.reference,x=x,y=y,slope=terrain.slope)
            _,design_labels=target_surface(original,c.plan)
            work=design_labels>0
        protected,_=channel_shore(terrain,c.plan)
        margin=max(3.,3*cell)
        from .workfront import bank_layout
        pad_extent=math.ceil((max(c.fleet.count,c.fleet.charge_slots)-1)/2)*c.fleet.pad_spacing
        service_width=2*(pad_extent+c.fleet.waiting_offset+c.fleet.separation+cell)
        c.plan.channel['operations']=dict(front_length_m=min(32.,self.side_m/3),
            setback_m=max(6.,min(24.,service_width,self.side_m*.23)),bank_width_m=max(4.,self.side_m*.22))
        dump,cap,chainage,signed=bank_layout(terrain,c.plan,work,protected)
        if not np.any(dump & (cap>terrain.reference)):
            raise ValueError('Канал с откосами не оставляет места для боковых отвалов. Увеличьте локальную карту')
        iy,ix=np.nonzero(dump)
        # Compatibility bounding box; physical capacity is the two bank masks.
        c.terrain.dump_rect=(float(ix.min()*cell),float(iy.min()*cell),
                            float((ix.max()-ix.min()+1)*cell),float((iy.max()-iy.min()+1)*cell))
        pads=[(0.,0.)]+[(0.,(i+1)//2*(1 if i%2 else -1)*c.fleet.pad_spacing) for i in range(1,max(c.fleet.count,c.fleet.charge_slots))]
        pad_margin=margin+c.fleet.waiting_offset+2
        sites=[]
        candidates=[(bx,by) for bx in np.linspace(pad_margin,self.side_m-pad_margin,13)
                    for by in np.linspace(pad_margin,self.side_m-pad_margin,13)]
        remaining=work & (current-target>1e-10)
        first=float(chainage[remaining].min()) if np.any(remaining) else float(chainage[work].min()) if np.any(work) else 0.
        work_along,work_side=chainage[work],signed[work]
        # Include regular positions in the service lanes, on both banks.
        for segment in c.plan.segments:
            a,b=np.asarray(segment.points[0]),np.asarray(segment.points[-1]);v=b-a
            length=float(np.linalg.norm(v));normal=np.array([-v[1],v[0]])/max(length,1e-12)
            for u in np.linspace(0,1,max(2,math.ceil(length/max(3.,self.side_m/96)))):
                p=a+u*v
                if np.any(p<0) or np.any(p>=self.side_m):continue
                local=np.abs(work_along-chainage[min(n-1,int(p[1]/cell)),min(n-1,int(p[0]/cell))])<max(3.,cell)
                opening=max(segment.width/2,float(np.max(np.abs(work_side[local]))) if np.any(local) else 0.)
                for sign in (-1,1):
                    candidates.append(tuple(p+sign*normal*(opening+c.plan.channel['operations']['setback_m']/2)))
        for bx,by in candidates:
            if not (pad_margin<=bx<=self.side_m-pad_margin and margin<=by<=self.side_m-margin):continue
            safe=True
            for px,py in pads+[(px-c.fleet.waiting_offset,py) for px,py in pads]:
                xx,yy=bx+px,by+py
                ix,iy=int(xx/cell),int(yy/cell);radius=math.ceil(c.fleet.separation/cell)+1
                row=slice(max(0,iy-radius),min(n,iy+radius+1));col=slice(max(0,ix-radius),min(n,ix+radius+1))
                near=(axis[col][None,:]-xx)**2+(axis[row][:,None]-yy)**2<(c.fleet.separation+cell)**2
                blocked=work[row,col]|dump[row,col]
                if protected is not None:blocked |= protected[row,col]
                if np.any(near&blocked) or bool(is_ocean(landscape.raster,origin[0]+xx,origin[1]+yy)):
                    safe=False;break
                region=current[max(0,iy-1):iy+2,max(0,ix-1):ix+2]
                if float(np.ptp(region))>c.base.max_slope*2*cell:safe=False;break
            if safe:
                along=float(chainage[int(by/cell),int(bx/cell)])
                sites.append((along,float(bx),float(by)))
        if not sites:raise ValueError('Не найдена сухая устойчивая площадка базы вне канала и отвала. Измените размер карты/маршрут')
        first_cells=np.where(work & (np.abs(chainage-first)<=cell*2))
        start=np.array([float(np.mean(np.broadcast_to(x,current.shape)[first_cells])),
                        float(np.mean(np.broadcast_to(y,current.shape)[first_cells]))]) if np.any(work) else np.array([self.side_m/2]*2)
        _,c.base.x,c.base.y=min(sites,key=lambda p:math.hypot(p[1]-start[0],p[2]-start[1]))
        c.plan.channel['operations'].update(base_sites=[list(p) for p in sites],base_chainage_m=first)
        excess=current-target
        available=np.flatnonzero((excess>1e-10)&work)
        cell_index=int(available[np.argmin(chainage.ravel()[available])]) if len(available) else int(np.argmax(excess))
        iy,ix=np.unravel_index(cell_index,excess.shape)
        c.trench.x=float(np.clip(axis[ix],4+cell,self.side_m-4-cell))
        c.trench.y=float(np.clip(axis[iy],4+cell,self.side_m-4-cell))
        c.georeference['channel']=dict(project_id=self.identity,section=index,sections=self.section_count,
            length_m=self.length,width_m=self.width_m,depth_m=self.depth_m,target_msl=-self.depth_m,shore_offset_m=self.shore_offset_m)
        c.base.latitude_deg=latitude_at(c.georeference,c.base.x,c.base.y,c.base.latitude_deg)
        c.validate();return c

    def profile(self,raster):
        return corridor_profile(raster,self.points,-self.depth_m,self.width_m,angle_deg=self.angle_deg)

    def record(self,report):
        info=report.get('georeference',{}).get('channel',{})
        if info.get('project_id')!=self.identity:return
        index=info['section']
        if report['state']=='COMPLETED' and index not in self.completed:
            self.completed.append(index)
            self.history.append(dict(section=index,**{k:report[k] for k in ('time_seconds','excavated_m3','consumed_kwh','solar_generated_kwh','mass_error_kg','energy_error_j')}))
            self.history[-1]['initial_energy_kwh']=report['initial_energy_kwh']


def latitude_at(georeference,x,y,fallback=55.):
    """Latitude of the actual base, with SI world coordinates preserved."""
    if not georeference:return fallback
    source=georeference['source'];origin=georeference['origin_m']
    if source['crs']=='EPSG:4326':
        from .landscape import EARTH_RADIUS
        return source['y0']+(origin[1]+y)/(EARTH_RADIUS*math.pi/180)
    if 'latitude_deg' in source:
        try:
            from rasterio.warp import transform
            _,latitude=transform(source['crs'],'EPSG:4326',[source['x0']+origin[0]+x],[source['y0']+origin[1]+y])
            return latitude[0]
        except ImportError:return source['latitude_deg']
    return fallback
