"""Closed-boundary, float64 heightfield with conservative sand transfers."""
import math
import numpy as np
from .soil import SoilField,SoilLoad,cargo_volume


class Terrain:
    def __init__(self, config):
        config.validate()
        self.config = config
        t, q = config.terrain, config.trench
        self.cell = t.cell_size
        self.area = self.cell**2
        self.axis = (np.arange(t.n) + 0.5) * self.cell
        self.x, self.y = np.meshgrid(self.axis, self.axis)
        rng = np.random.default_rng(config.seed)
        self.height = rng.uniform(-t.roughness, t.roughness, (t.n, t.n)).astype(np.float64)
        self.reference = self.height.copy()
        if config.reference_height is not None:
            self.height = np.asarray(config.reference_height, dtype=np.float64).copy()
            self.reference = self.height.copy()
        if config.initial_height is not None:
            self.height = np.asarray(config.initial_height, dtype=np.float64).copy()
        self._initialize_soil()
        from .slopes import minimum_slope
        self.slope = minimum_slope(config)
        self.dump_slope = minimum_slope(config,loose=True)
        dx = np.maximum(np.abs(self.x-q.x)-q.length/2, 0)
        dy = np.maximum(np.abs(self.y-q.y)-q.width/2, 0)
        self.bottom = (dx == 0) & (dy == 0)
        from .slopes import envelope
        self.target = np.minimum(self.reference, envelope(self,np.hypot(dx,dy),-q.depth,'dig'))
        if config.reference_height is not None:
            self.target = self.reference.copy()  # imported terrain starts without an implicit cut to zero
        from .slopes import validate_target
        if config.reference_height is None:validate_target(self,self.target,self.target<self.reference-1e-12)
        self.excavation = self.target < self.reference - 1e-12
        self.fill_mask = np.zeros(self.height.shape, dtype=bool)
        self.segment_labels = np.zeros(self.height.shape, dtype=np.int16)
        x, y, w, h = t.dump_rect
        self.dump_mask = (self.x >= x) & (self.x < x+w) & (self.y >= y) & (self.y < y+h)
        if np.any(self.dump_mask & self.excavation):
            raise ValueError("Отвал пересекает канаву с откосами")
        # The cap is a stable tent with a one-cell setback. It reserves space for
        # avalanches and prevents sand leaving the designated dump rectangle.
        distance = np.maximum(0, np.minimum.reduce([self.x-x, x+w-self.x, self.y-y, y+h-self.y])-self.cell)
        self.dump_cap = self.reference + np.where(self.dump_mask, np.minimum(t.dump_max_height, distance*self.dump_slope*t.dump_slope_factor), 0)
        self.revision = 0
        self._dump_content_revision=0
        self.front = None
        self.protected = None
        self.design_work=None
        if config.plan is not None and config.plan.channel and config.plan.channel.get('operations'):
            from types import SimpleNamespace
            from .plan import target_surface,channel_shore
            from .workfront import bank_layout
            original=SimpleNamespace(height=self.reference,reference=self.reference,x=self.x,y=self.y,slope=self.slope,config=config,
                soil=SoilField(config.soil_profile,t.n,self.cell) if config.soil_profile else None)
            _,labels=target_surface(original,config.plan)
            protected,_=channel_shore(original,config.plan)
            self.protected=protected
            self.design_work=labels>0
            self.dump_mask,self.dump_cap,_,_=bank_layout(self,config.plan,labels>0,protected)
        if config.plan is not None:
            self.set_plan(config.plan)
        self._cut_indices=np.flatnonzero(self.excavation)
        self._fill_indices=np.flatnonzero(self.fill_mask)
        self._dump_indices=np.flatnonzero(self.dump_mask & (self.dump_cap>self.reference))
        self._clearance_surface=max(float(self.reference.max()),float(self.dump_cap.max()),
                                    float(self.height.max()),float(self.target.max()))
        self.minimum_surface=float(min(self.reference.min(),self.target.min()))
        if config.plan is not None and config.plan.channel and config.plan.channel.get('operations'):
            from .workfront import Workfront
            self.front=Workfront(self)
        if self.design_work is None:self.design_work=self.work_mask
        self.initial_volume = self.volume
        self._initial_surface = self.height.copy()
        self._relax_dirty=np.zeros(self.height.shape,dtype=bool)
        self._dirty_bounds=None
        if self.soil is not None and t.roughness>0:
            self._relax_dirty[:]=True;self._dirty_bounds=(0,t.n,0,t.n)
        self.initial_excavation = self.remaining
        self._pairs = []
        grid = np.arange(t.n*t.n).reshape(t.n, t.n)
        # Eight-neighbour slopes, split into disjoint matchings. Sequential
        # equal-and-opposite pair updates cannot overshoot or double-spend sand.
        for oy, ox in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            ya = slice(0, t.n-oy)
            xa = slice(max(0, -ox), min(t.n, t.n-ox))
            yb = slice(oy, t.n)
            xb = slice(max(0, ox), min(t.n, t.n+ox))
            a, b = grid[ya, xa], grid[yb, xb]
            colors = np.indices(a.shape)[1 if oy == 0 else 0] % 2
            for color in [0, 1]:
                self._pairs.append((a[colors == color], b[colors == color], self.slope*self.cell*math.hypot(oy, ox)))

    def _initialize_soil(self):
        t=self.config.terrain;config=self.config
        self.soil=None
        if config.soil_profile is not None:
            self.soil=SoilField(config.soil_profile,t.n,self.cell,config.initial_soil)
            if config.initial_soil is None:
                delta=self.height-self.reference;ids=np.arange(t.n*t.n)
                self.soil.extract(ids,np.maximum(-delta,0).ravel(),commit=True)
                if np.any(delta>0):
                    load=SoilLoad.empty(config.soil_profile)
                    load.masses[0]=float(np.maximum(delta,0).sum()*self.area)*config.soil_profile.materials[0].density
                    self.soil.add(ids,np.maximum(delta,0).ravel(),load)
            else:
                for y in range(0,t.n,16):
                    yy,xx=np.mgrid[y:min(t.n,y+16),0:t.n]
                    change=self.soil.height_delta((yy*t.n+xx).ravel()).reshape(yy.shape)
                    if not np.allclose(change,(self.height-self.reference)[y:y+16],rtol=0,atol=1e-7):
                        raise ValueError('Состав грунта не совпадает с загруженным рельефом')
            self._initial_soil_mass=self.soil.mass_delta
            self._initial_soil_volume=self.soil.volume_delta_by_material
            bounds=np.r_[0.,config.soil_profile.boundaries,np.inf]
            depth=np.maximum(self.reference-t.floor,0)
            self._native_mass=sum(float(np.maximum(0.,np.minimum(depth,b)-a).sum()*self.area)*m.density
                                  for a,b,m in zip(bounds[:-1],bounds[1:],config.soil_profile.materials))

    @property
    def volume(self):
        return float(np.sum(self.height-self.config.terrain.floor, dtype=np.float64)*self.area)

    @property
    def mass(self):
        if self.soil is not None:return self._native_mass+self.soil.mass_delta
        return self.volume*self.config.terrain.density

    @property
    def mass_change(self):
        if self.soil is not None:return self.soil.mass_delta-self._initial_soil_mass
        # Difference first avoids subtracting huge bedrock columns at MSL.
        return float(np.sum(self.height-self._initial_surface,dtype=np.float64)*self.area*self.config.terrain.density)

    @property
    def clearance_surface(self):
        # Dump height is relative to the DEM, whereas flight Z is absolute.
        return self._clearance_surface

    @property
    def remaining(self):
        revision=getattr(self,"revision",0)
        if getattr(self,"_remaining_revision",None)!=revision:
            h,target=self.height.ravel(),self.target.ravel()
            self._cut_cache=float(np.maximum(h[self._cut_indices]-target[self._cut_indices],0).sum()*self.area)
            self._fill_cache=float(np.maximum(target[self._fill_indices]-h[self._fill_indices],0).sum()*self.area)
            self._remaining_cache=self._cut_cache+self._fill_cache
            self._remaining_revision=revision
        return self._remaining_cache

    @property
    def remaining_cut(self):
        _ = self.remaining
        return self._cut_cache

    @property
    def remaining_fill(self):
        _ = self.remaining
        return self._fill_cache

    @property
    def work_mask(self):
        return self.excavation | self.fill_mask

    @property
    def stockpile_volume(self):
        if getattr(self,'_stockpile_revision',None)!=self.revision:
            indices=np.flatnonzero(self.dump_mask)
            self._stockpile_cache=float(np.maximum(self.height.ravel()[indices]-self.reference.ravel()[indices],0).sum()*self.area)
            self._stockpile_revision=self.revision
        return self._stockpile_cache

    def set_plan(self, plan):
        from .plan import target_surface
        plan.validate(self.config)
        target, labels = target_surface(self, plan)
        cutting = np.zeros(self.height.shape,dtype=bool)
        filling = np.zeros(self.height.shape,dtype=bool)
        for number,segment in enumerate(plan.segments,1):
            if segment.mode=='dig':cutting |= labels==number
            else:filling |= labels==number
        work = cutting | filling
        if np.any(work & self.dump_mask):
            raise ValueError("План пересекает отвал: он используется как запас грунта")
        if plan.channel is None and (np.any(work[[0,-1], :]) or np.any(work[:, [0,-1]])):
            raise ValueError("План с откосами касается границы карты")
        from .slopes import legacy,validate_target
        validate_target(self,target,work)
        for oy, ox in ([(0,1),(1,0),(1,1),(1,-1)] if legacy(self.config) else []):
            ya, yb = slice(0,target.shape[0]-oy),slice(oy,target.shape[0])
            xa, xb = slice(max(0,-ox),min(target.shape[1],target.shape[1]-ox)),slice(max(0,ox),min(target.shape[1],target.shape[1]+ox))
            active = work[ya,xa] | work[yb,xb]
            steep = np.abs(target[ya,xa]-target[yb,xb]) > self.slope*self.cell*math.hypot(oy,ox)+1e-6
            if np.any(active & steep):
                raise ValueError("Переход между мазками круче угла откоса: увеличьте расстояние, измените высоты или удалите перекрывающийся мазок")
        self.target, self.segment_labels = target, labels
        self.excavation, self.fill_mask = cutting, filling
        self._cut_indices=np.flatnonzero(cutting);self._fill_indices=np.flatnonzero(filling)
        self.bottom = labels > 0
        self.revision += 1

    @property
    def dump_space(self):
        return float(np.maximum(self.dump_cap-self.height, 0)[self.dump_mask].sum()*self.area)

    def sample(self, x, y):
        # Nearest-cell sampling uses exactly the same cell convention as edits.
        ix = min(max(int(x/self.cell), 0), self.height.shape[1]-1)
        iy = min(max(int(y/self.cell), 0), self.height.shape[0]-1)
        return float(self.height[iy, ix])

    def sample_many(self, points):
        """The scalar nearest-cell convention, batched without Python cell reads."""
        height,cell=self.height,self.cell
        indices=(np.asarray(points)[...,:2]/cell).astype(np.intp)
        ix=np.clip(indices[...,0],0,height.shape[1]-1)
        iy=np.clip(indices[...,1],0,height.shape[0]-1)
        return height[iy,ix]

    def route_token(self,bounds):
        """Versions of only the terrain blocks touched by a ground route."""
        versions=getattr(self,'_route_versions',None)
        shape=((self.height.shape[0]+31)//32,(self.height.shape[1]+31)//32)
        if versions is None or versions.shape!=shape:
            self._route_versions=versions=np.zeros(shape,dtype=np.uint64)
        x0,y0,x1,y1=bounds;cell=self.cell
        a=max(0,min(shape[0]-1,int(y0/cell)//32));b=max(a,min(shape[0]-1,int(y1/cell)//32))
        c=max(0,min(shape[1]-1,int(x0/cell)//32));d=max(c,min(shape[1]-1,int(x1/cell)//32))
        return tuple(map(int,versions[a:b+1,c:d+1].ravel()))

    def _touch_route_cells(self,iy,ix):
        versions=getattr(self,'_route_versions',None)
        if versions is not None:
            ids=np.unique((np.asarray(iy)//32)*versions.shape[1]+np.asarray(ix)//32)
            versions.ravel()[ids]+=1

    def source_excess(self, source="cut"):
        mask, surface = (self.dump_mask, self.reference) if source == "stockpile" else (self.excavation, self.target)
        return np.where(mask, np.maximum(self.height-surface, 0), 0)

    def source_indices(self, source='cut'):
        if source=='cut' and self.front is not None:
            self.front.update()
            return self.front.active_indices
        return np.flatnonzero(self.source_excess(source)>1e-12)

    def patch(self,x,y,radius):
        n=self.height.shape[0]
        return (slice(max(0,math.floor((y-radius)/self.cell)),min(n,math.floor((y+radius)/self.cell)+1)),
                slice(max(0,math.floor((x-radius)/self.cell)),min(n,math.floor((x+radius)/self.cell)+1)))

    def local_available(self, x, y, source="cut", radius=None):
        excess = self.source_excess(source)
        radius=self.config.terrain.grab_radius if radius is None else radius
        mask = (excess > 0) & ((self.x-x)**2+(self.y-y)**2 <= radius**2)
        return mask, np.where(mask, excess, 0)

    def _available_patch(self,x,y,source,radius):
        radius=self.config.terrain.grab_radius if radius is None else radius
        n=self.height.shape[0]
        left=max(0,math.floor((x-radius)/self.cell));right=min(n,math.floor((x+radius)/self.cell)+1)
        bottom=max(0,math.floor((y-radius)/self.cell));top=min(n,math.floor((y+radius)/self.cell)+1)
        patch=(slice(bottom,top),slice(left,right))
        mask,surface=(self.dump_mask,self.reference) if source=='stockpile' else (self.excavation,self.target)
        footprint=(self.x[patch]-x)**2+(self.y[patch]-y)**2<=radius**2
        if source=='cut' and self.front is not None and self.front.sectors:
            allowed=self.front.cut_mask(patch)
            # A v1–10 flight committed across a future sector boundary may
            # finish its original grab; no new job receives this exception.
            for px,py,r in getattr(self.front,'legacy_pickups',()):
                if abs(x-px)<1e-8 and abs(y-py)<1e-8:
                    allowed |= (self.x[patch]-px)**2+(self.y[patch]-py)**2<=r*r
            footprint &= allowed
        return patch,np.where(mask[patch]&footprint,np.maximum(self.height[patch]-surface[patch],0),0)

    def available_mass(self,x,y,source='cut',radius=None):
        if self.soil is not None:return self.preview_load(x,y,math.inf,source,radius).mass
        _,excess=self._available_patch(x,y,source,radius)
        return float(excess.sum()*self.area*self.config.terrain.density)

    def _soil_cut(self,x,y,max_mass,source,radius,max_volume,allowed_mask,minimum_height,commit):
        if math.isnan(max_mass) or max_mass<0 or math.isnan(max_volume) or max_volume<0:
            raise ValueError('Неверная вместимость захвата')
        if max_mass==0 or max_volume==0:return SoilLoad.empty(self.soil.profile)
        patch,excess=self._available_patch(x,y,source,radius)
        if minimum_height is not None:excess=np.minimum(excess,np.maximum(self.height[patch]-minimum_height,0.))
        if allowed_mask is not None:excess=np.where(allowed_mask[patch],excess,0.)
        yy,xx=np.mgrid[patch[0],patch[1]];ids=(yy*self.height.shape[0]+xx).ravel()
        removed,load=self.soil.extract(ids,excess.ravel(),max_mass,max_volume,commit)
        if commit and load.mass>0:
            change=removed.reshape(excess.shape);self.height[patch]-=change
            self._relax_dirty[patch]|=change>0;self._mark_dirty(patch,change>0);self.revision+=1
        return load

    def preview_load(self,x,y,max_mass,source='cut',radius=None,max_volume=math.inf,
                     allowed_mask=None,minimum_height=None):
        return self._soil_cut(x,y,max_mass,source,radius,max_volume,allowed_mask,minimum_height,False)

    def scoop_load(self,x,y,max_mass,source='cut',radius=None,max_volume=math.inf,
                   allowed_mask=None,minimum_height=None):
        return self._soil_cut(x,y,max_mass,source,radius,max_volume,allowed_mask,minimum_height,True)

    def scoop(self, x, y, max_mass, source="cut", radius=None, allowed_mask=None, minimum_height=None):
        if not math.isfinite(max_mass) or max_mass < 0:
            raise ValueError("Неверная масса захвата")
        if self.soil is not None:
            raise ValueError('Для слоистого грунта используйте scoop_load: груз должен сохранять состав')
        patch, excess = self._available_patch(x, y, source, radius)
        if minimum_height is not None:
            excess=np.minimum(excess,np.maximum(self.height[patch]-minimum_height,0.))
        if allowed_mask is not None:
            excess=np.where(allowed_mask[patch],excess,0.)
        available = float(excess.sum()*self.area)
        volume = min(max_mass/self.config.terrain.density, available)
        if available <= 0:
            return 0.0
        # Proportional removal never digs below the design and changes many cells.
        self.height[patch] -= excess*(volume/available)
        self._relax_dirty[patch] |= excess>0
        self._mark_dirty(patch,excess>0)
        self.revision += 1
        return volume*self.config.terrain.density

    def _dump_geometry(self):
        key=(id(self.dump_mask),self.height.shape,self.cell)
        if getattr(self,'_dump_geometry_key',None)!=key:
            self._dump_geometry_key=key;n=self.height.shape[1]
            indices=self._dump_indices
            self._dump_rows=self.axis[np.unique(indices//n)]
            self._dump_columns=self.axis[np.unique(indices%n)]

    def _dump_available(self):
        self.refresh_dump_reserve()
        dump_revision=getattr(self,'_dump_content_revision',0)
        if getattr(self,'_dump_revision',None)!=dump_revision:
            indices=self._dump_indices
            capacity=np.maximum(self.dump_cap.ravel()[indices]-self.height.ravel()[indices],0)
            self._dump_free=float(capacity.sum()*self.area)
            if self.front is not None:
                signed=self.front.signed.ravel()[indices]
                heights=np.maximum(self.height.ravel()[indices]-self.reference.ravel()[indices],0)
                self._lighter_bank=1 if heights[signed>0].sum()<=heights[signed<0].sum() else -1
            self._dump_revision=dump_revision
        return self._dump_free

    def refresh_dump_reserve(self,*,force=False,repack=False):
        geometry=getattr(self,'_spoil_geometry',None)
        if not self.config.terrain.compact_spoil or geometry is None:return False
        owner=getattr(self,'_site_owner',None);site=getattr(owner,'shared_site',None) if owner else None
        simulations=[u.simulation for u in site.project.units] if site else [owner] if owner else []
        front=getattr(self,'front',None)
        if front is not None:front.update()
        fronts=[]
        for s in simulations:
            f=s.terrain.front
            if f is not None and s.terrain.remaining>s.config.terrain.completion_volume:
                f.update();fronts.append(f.sector_start if f.sectors else f.position)
        if not fronts and front is not None:fronts=[front.position]
        stride=geometry['settings'].get('sector_length_m',geometry['settings']['front_length_m']/2)
        key=tuple(int(p/max(4.,stride)) for p in fronts)
        if not force and getattr(self,'_spoil_front_key',None)==key:return False
        if repack and any(s._reserved_dump(None) for s in simulations):
            raise ValueError('Для сокращения резерва завершите назначенные выгрузки')
        from .site_ground import equipment_mask
        from .spoil import reserve
        carried=0.
        for s in simulations:
            w=s.logistics
            carried+=sum(cargo_volume(d.cargo,d.soil,self.config.terrain.density) for d in s.drones if not w or d.identifier in w.direct_ids)
            if w:
                carried+=sum(cargo_volume(c.soil_kg,c.soil,self.config.terrain.density) for c in w.containers)
                if w.excavator:carried+=cargo_volume(w.excavator.bucket_kg,w.excavator.soil,self.config.terrain.density)
        old_mask=np.zeros_like(self.dump_mask) if repack else self.dump_mask
        old_cap=self.reference if repack else self.dump_cap
        mask,cap,info=reserve(self,**geometry,old_mask=old_mask,old_cap=old_cap,
                             blocked=equipment_mask(self,simulations),fronts=fronts,carried_volume=carried)
        changed=not np.array_equal(mask,self.dump_mask) or not np.array_equal(cap,self.dump_cap)
        self._spoil_front_key=key;self.dump_reserve_info=info
        if changed:
            self._spoil_reserve_revision=getattr(self,'_spoil_reserve_revision',0)+1
            delta=mask^self.dump_mask
            yy,xx=np.nonzero(delta)
            if len(xx):self._touch_route_cells(yy,xx)
            self.dump_mask,self.dump_cap=mask,cap
            self._dump_indices=np.flatnonzero(mask&(cap>self.reference))
            self._dump_content_revision=getattr(self,'_dump_content_revision',0)+1
            self._clearance_surface=max(self._clearance_surface,float(cap.max()))
            self.revision+=1
            if site:site.invalidate_geometry()
        return changed

    def _exclude_dump_reservations(self,useful,patch,reserved,separation):
        if len(reserved)==0 or separation<=0:return
        ys,xs=patch;axis=self.axis
        for point in reserved:
            px,py=point[:2]
            if px+separation<axis[xs.start] or px-separation>axis[xs.stop-1] or py+separation<axis[ys.start] or py-separation>axis[ys.stop-1]:continue
            ry,rx=self.patch(px,py,separation)
            y0,y1=max(ys.start,ry.start),min(ys.stop,ry.stop)
            x0,x1=max(xs.start,rx.start),min(xs.stop,rx.stop)
            if y0>=y1 or x0>=x1:continue
            sub=np.s_[y0:y1,x0:x1]
            useful[y0-ys.start:y1-ys.start,x0-xs.start:x1-xs.start]&=(self.x[sub]-px)**2+(self.y[sub]-py)**2>=separation**2

    def _local_dump_capacities(self):
        """Disk capacity at every cell centre, without a Python search per cell.

        Only used when the first nearby site cannot fit the load. A fragmented
        bank can have plenty of total room and no usable unloading footprint.
        Row prefix sums give the same circular footprint as local_dump_space.
        """
        radius=max(2.,self.config.terrain.grab_radius)
        key=(getattr(self,'_dump_content_revision',0),id(self.dump_cap),id(self.dump_mask),
             self.height.shape,self.cell,radius)
        if getattr(self,'_local_dump_capacity_key',None)==key:return self._local_dump_capacity
        n=self.height.shape[0]
        cap=np.where(self.dump_mask,np.maximum(self.dump_cap-self.height,0),0.)
        prefix=np.pad(np.cumsum(cap,axis=1),((0,0),(1,0)))
        result=np.zeros_like(cap);columns=np.arange(n)
        for dy in range(-int(radius/self.cell),int(radius/self.cell)+1):
            rx=int(math.floor(math.sqrt(max(0.,radius**2-(dy*self.cell)**2))/self.cell+1e-12))
            left=np.maximum(0,columns-rx);right=np.minimum(n,columns+rx+1)
            y0,y1=max(0,-dy),min(n,n-dy)
            result[y0:y1]+=prefix[y0+dy:y1+dy,right]-prefix[y0+dy:y1+dy,left]
        self._local_dump_capacity=result*self.area
        self._local_dump_capacity_key=key
        return self._local_dump_capacity

    def dump_location(self, volume, reserved=(), separation=0.0, near=None):
        if near is not None:
            if self._dump_available()<volume-1e-12:
                # An exhausted reservation is extended once per terrain change,
                # never retried by every drone on every unchanged second.
                key=(self.revision,getattr(self,'_dump_content_revision',0))
                if getattr(self,'_spoil_exhaustion_key',None)!=key:
                    self.refresh_dump_reserve(force=True)
                    self._spoil_exhaustion_key=(self.revision,getattr(self,'_dump_content_revision',0))
                if self._dump_available()<volume-1e-12:return None
            self._dump_geometry()
            if not len(self._dump_rows):return None
            # Search increasing local windows; no repeated sort of the full map.
            radius=max(4.,self.config.fleet.separation*2)
            # Independent projections give a conservative lower distance bound
            # for arbitrary masks, including bends and disconnected banks.
            lower=math.hypot(float(np.abs(self._dump_columns-near[0]).min()),float(np.abs(self._dump_rows-near[1]).min()))
            while radius<lower-1e-12:radius*=2
            side=self.height.shape[0]*self.cell
            while radius<=side*2:
                patch=self.patch(*near[:2],radius)
                cap=np.maximum(self.dump_cap[patch]-self.height[patch],0)
                useful=self.dump_mask[patch] & (cap>1e-12)
                xx,yy=self.x[patch],self.y[patch]
                distance=(xx-near[0])**2+(yy-near[1])**2
                useful &= distance<=radius**2
                self._exclude_dump_reservations(useful,patch,reserved,separation)
                if np.any(useful):
                    score=np.where(useful,distance+np.maximum(self.height[patch]-self.reference[patch],0)**2,np.inf)
                    if self.front is not None:
                        # Balance equally short hauls; a remote bank never wins
                        # merely because it currently contains less soil.
                        score+=np.where(self.front.signed[patch]*self._lighter_bank<0,
                                        float(distance[useful].min())*.1,0.)
                    candidates=np.flatnonzero(useful)
                    first=int(candidates[int(np.argmin(score.ravel()[candidates]))])
                    i=np.unravel_index(first,score.shape);point=float(xx[i]),float(yy[i])
                    if self.front is None or self.local_dump_space(*point)>=volume-1e-12:return point
                    # Filter all infeasible footprints at once. In late runs the
                    # old loop could perform hundreds of thousands of identical
                    # small capacity checks, per drone and per search radius.
                    room=self._local_dump_capacities()[patch].ravel()
                    candidates=candidates[room[candidates]>=volume-1e-9]
                    for k in candidates[np.argsort(score.ravel()[candidates],kind='stable')]:
                        if int(k)==first:continue
                        i=np.unravel_index(int(k),score.shape)
                        point=float(xx[i]),float(yy[i])
                        if self.front is None or self.local_dump_space(*point)>=volume-1e-12:return point
                radius*=2
            return None
        capacity = np.where(self.dump_mask, np.maximum(self.dump_cap-self.height, 0), 0)
        if capacity.sum()*self.area < volume-1e-12:
            return None
        candidates = np.flatnonzero(capacity.ravel() > 1e-12)
        for point in reserved:
            candidates=candidates[(self.x.ravel()[candidates]-point[0])**2+(self.y.ravel()[candidates]-point[1])**2 >= separation**2]
        if not len(candidates):
            return None
        # Lowest available site produces distributed heaps, not a single spire.
        score = self.height.ravel()[candidates] + 0.02*np.hypot(self.x.ravel()[candidates]-self.x[self.dump_mask].mean(), self.y.ravel()[candidates]-self.y[self.dump_mask].mean())
        i = candidates[int(np.argmin(score))]
        return (float(self.x.ravel()[i]), float(self.y.ravel()[i]))

    def local_dump_space(self,x,y):
        radius=max(2.,self.config.terrain.grab_radius)
        patch=self.patch(x,y,radius)
        near=(self.x[patch]-x)**2+(self.y[patch]-y)**2<=radius**2
        cap=np.where(self.dump_mask[patch]&near,np.maximum(self.dump_cap[patch]-self.height[patch],0),0)
        return float(cap.sum()*self.area)

    def deposit(self, x, y, mass, radius=None,load=None):
        if not math.isfinite(mass) or mass < 0:
            raise ValueError("Неверная масса выгрузки")
        if self.soil is not None and mass>0 and (load is None or load.profile!=self.soil.profile):
            raise ValueError('Выгрузка слоистого грунта требует состава груза')
        if load is not None:load.validate(mass)
        volume = cargo_volume(mass,load,self.config.terrain.density)
        if self.front is not None:
            if self._dump_available()<volume-1e-12:raise ValueError('Отвал заполнен: груз не выгружен')
            reach=max(2.,self.config.terrain.grab_radius);bounds=self.patch(x,y,reach)
            near=(self.x[bounds]-x)**2+(self.y[bounds]-y)**2<=reach**2
            capacity=np.where(self.dump_mask[bounds]&near,np.maximum(self.dump_cap[bounds]-self.height[bounds],0),0.)
            if float(capacity.sum()*self.area)<volume-1e-12:
                raise ValueError('Локальная площадка отвала заполнена: нужен новый заход с грузом')
            self._deposit_capacity(x,y,volume,capacity,radius,load,bounds)
            return
        capacity = np.where(self.dump_mask, np.maximum(self.dump_cap-self.height, 0), 0)
        if capacity.sum()*self.area < volume-1e-12:
            raise ValueError("Отвал заполнен: груз не выгружен")
        if self.front is not None:
            if self.local_dump_space(x,y)<volume-1e-12:
                raise ValueError('Локальная площадка отвала заполнена: нужен новый заход с грузом')
            reach=max(2.,self.config.terrain.grab_radius)
            capacity=np.where((self.x-x)**2+(self.y-y)**2<=reach**2,capacity,0.)
        self._deposit_capacity(x, y, volume, capacity, radius,load)

    def fill_location(self, reserved=(), separation=0.0):
        capacity = np.where(self.fill_mask, np.maximum(self.target-self.height, 0), 0)
        for point in reserved:
            capacity[(self.x-point[0])**2+(self.y-point[1])**2 < separation**2] = 0
        if not np.any(capacity > 1e-12):
            return None
        index = int(np.argmax(capacity))
        return float(self.x.ravel()[index]), float(self.y.ravel()[index])

    def deposit_fill(self, x, y, mass, radius=None,load=None):
        if not math.isfinite(mass) or mass < 0:
            raise ValueError("Неверная масса засыпки")
        if self.soil is not None and mass>0 and (load is None or load.profile!=self.soil.profile):
            raise ValueError('Засыпка слоистым грунтом требует состава груза')
        capacity = np.where(self.fill_mask, np.maximum(self.target-self.height, 0), 0)
        if load is not None:load.validate(mass)
        available=cargo_volume(mass,load,self.config.terrain.density)
        volume = min(available, float(capacity.sum()*self.area))
        fraction=volume/available if available>0 else 0.
        self._deposit_capacity(x, y, volume, capacity, radius,load.scaled(fraction) if load is not None else None)
        return mass*fraction if load is not None else volume*self.config.terrain.density

    def _deposit_capacity(self, x, y, volume, capacity, radius=None,load=None,bounds=None):
        if volume <= 0:
            return
        radius=self.config.terrain.grab_radius if radius is None else radius
        expanded=False
        while True:
            patch=self.patch(x,y,radius)
            if bounds is not None:
                patch=tuple(slice(max(a.start,b.start),min(a.stop,b.stop)) for a,b in zip(patch,bounds))
            distance=(self.x[patch]-x)**2+(self.y[patch]-y)**2
            cap_patch=(tuple(slice(a.start-b.start,a.stop-b.start) for a,b in zip(patch,bounds)) if bounds is not None else patch)
            local_cap=np.where(distance<=radius**2,capacity[cap_patch],0)
            if local_cap.sum()*self.area>=volume-1e-12:break
            radius*=2;expanded=True
            if radius>self.height.shape[0]*self.cell*4:raise ValueError('Нет места для выгрузки')
        if expanded:
            # Stable nearest-first expansion, limited to this small patch.
            order=np.argsort(distance.ravel(),kind='stable')
            cap=local_cap.ravel()[order]*self.area
            needed=min(len(order),int(np.searchsorted(np.cumsum(cap),volume))+1)
            local=np.zeros(local_cap.size,dtype=bool);local[order[:needed]]=True
            local_cap=np.where(local.reshape(local_cap.shape),local_cap,0)
        total = float(local_cap.sum()*self.area)
        if total<=0:raise ValueError('Нет места для выгрузки: груз сохранён')
        if volume > 0:
            change=local_cap*(volume/total)
            if self.soil is not None:
                if load is None:
                    load=SoilLoad.empty(self.soil.profile);load.masses[0]=volume*self.soil.profile.materials[0].density
                yy,xx=np.mgrid[patch[0],patch[1]]
                self.soil.add((yy*self.height.shape[0]+xx).ravel(),change.ravel(),load)
            self.height[patch] += change
            self._relax_dirty[patch] |= local_cap>0
            self._mark_dirty(patch,local_cap>0)
            self.revision += 1

    def _mark_dirty(self,patch,mask):
        yy,xx=np.nonzero(mask)
        if not len(xx):return
        self._touch_route_cells(yy+patch[0].start,xx+patch[1].start)
        if np.any(self.dump_mask[patch]&mask):self._dump_content_revision=getattr(self,'_dump_content_revision',0)+1
        self._grow_bounds(int(yy.min())+patch[0].start,int(yy.max())+patch[0].start+1,
                          int(xx.min())+patch[1].start,int(xx.max())+patch[1].start+1)

    def _grow_bounds(self,y0,y1,x0,x1):
        if self._dirty_bounds is not None:
            a,b,c,d=self._dirty_bounds;y0,y1,x0,x1=min(a,y0),max(b,y1),min(c,x0),max(d,x1)
        self._dirty_bounds=(y0,y1,x0,x1)

    def _local_pairs(self,sweeps):
        if self._dirty_bounds is None:return []
        n=self.height.shape[0];y0,y1,x0,x1=self._dirty_bounds
        pad=sweeps*8+1
        y0,y1,x0,x1=max(0,y0-pad),min(n,y1+pad),max(0,x0-pad),min(n,x1+pad)
        yy,xx=np.mgrid[y0:y1,x0:x1]
        pairs=[]
        for oy,ox in [(0,1),(1,0),(1,1),(1,-1)]:
            valid=(yy+oy<n)&(xx+ox>=0)&(xx+ox<n)
            for color in (0,1):
                mask=valid & ((xx if oy==0 else yy)%2==color)
                a=(yy[mask]*n+xx[mask]).ravel()
                pairs.append((a,a+oy*n+ox,self.slope*self.cell*math.hypot(oy,ox)))
        return pairs

    def relax(self, sweeps=None):
        t = self.config.terrain
        height = self.height.ravel()
        moved = 0.0
        count=t.avalanche_sweeps if sweeps is None else sweeps
        pairs=self._local_pairs(count) if self.config.reference_height is not None or self.soil is not None else self._pairs
        for _ in range(count):
            sweep_moved = 0.0
            for a, b, limit in pairs:
                if self.config.reference_height is not None:
                    touched=self._relax_dirty.ravel()
                    near=touched[a]|touched[b]
                    a,b=a[near],b[near]
                    if not len(a):continue
                difference = height[a]-height[b]
                from .slopes import pair_transfer,legacy
                if legacy(self.config):required=np.maximum(0.,np.abs(difference)-limit-t.avalanche_epsilon)/2
                else:required=pair_transfer(self,a,b,limit/self.slope)
                active = required>0
                if self.config.reference_height is not None:
                    # Natural cliffs in the DEM are not freshly disturbed sand.
                    # Only pairs touched by an edit/avalanche may relax.
                    original = self.reference.ravel()
                    active &= (np.abs(height[a]-original[a]) > 1e-10) | (np.abs(height[b]-original[b]) > 1e-10)
                if not np.any(active):
                    continue
                a, b, difference = a[active], b[active], difference[active]
                transfer = np.sign(difference)*required[active]*(2*t.relaxation)
                if self.soil is not None:
                    source=np.where(transfer>0,a,b);destination=np.where(transfer>0,b,a)
                    amount=np.minimum(np.abs(transfer),np.maximum(height[source]-t.floor,0.))
                    self.soil.avalanche(source,destination,amount)
                    transfer=np.sign(transfer)*amount
                height[a] -= transfer
                height[b] += transfer
                n=self.height.shape[0]
                if getattr(self,'_route_versions',None) is not None:
                    self._touch_route_cells(np.r_[a//n,b//n],np.r_[a%n,b%n])
                if np.any(self.dump_mask.ravel()[a]|self.dump_mask.ravel()[b]):
                    self._dump_content_revision=getattr(self,'_dump_content_revision',0)+1
                self._relax_dirty.ravel()[a]=True;self._relax_dirty.ravel()[b]=True
                n=self.height.shape[0]
                self._grow_bounds(int(min(a.min()//n,b.min()//n)),int(max(a.max()//n,b.max()//n))+1,
                                  int(min((a%n).min(),(b%n).min())),int(max((a%n).max(),(b%n).max()))+1)
                sweep_moved += float(np.abs(transfer).sum())*self.area
            moved += sweep_moved
            if sweep_moved == 0:
                self._relax_dirty[:]=False
                self._dirty_bounds=None
                break
        if moved > 0:
            self.revision += 1
        return moved

    def max_slope_excess(self):
        h = self.height.ravel()
        from .slopes import legacy,pair_transfer
        if not legacy(self.config):
            return max(float(pair_transfer(self,a,b,limit/self.slope).max(initial=0))*2 for a,b,limit in self._pairs)
        return max(float(np.maximum(np.abs(h[a]-h[b])-limit, 0).max()) for a, b, limit in self._pairs)
