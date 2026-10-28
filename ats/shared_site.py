"""One physical heightfield and material ledger, shared by up to twenty bases.

Work masks belong to the dispatcher; height, deposits and materials never have
per-complex copies. Existing autonomous controllers operate on Terrain views.
"""
from copy import deepcopy
import math
import numpy as np
from .terrain import Terrain


class WorkTerrain(Terrain):
    _local={'core','config','shared_site','owner_id','excavation','fill_mask','bottom',
        '_cut_indices','_fill_indices','front','initial_excavation','_remaining_revision',
        '_remaining_cache','_cut_cache','_fill_cache'}

    def __init__(self,core,config,site,owner_id):
        self.__dict__.update(core=core,config=config,shared_site=site,owner_id=owner_id)
        self.assign(np.zeros(core.height.shape,dtype=bool))

    def __getattr__(self,name):
        core=self.__dict__.get('core')
        if core is None:raise AttributeError(name)
        return getattr(core,name)

    def __setattr__(self,name,value):
        if name in self._local or 'core' not in self.__dict__:self.__dict__[name]=value
        else:setattr(self.core,name,value)

    def assign(self,mask):
        self.excavation=self.core.excavation&mask;self.fill_mask=self.core.fill_mask&mask
        self.bottom=self.core.bottom&mask
        self._cut_indices=np.flatnonzero(self.excavation);self._fill_indices=np.flatnonzero(self.fill_mask)
        self._remaining_revision=-1;self.front=None
        if self.config.plan and self.config.plan.channel and self.config.plan.channel.get('operations'):
            from .workfront import Workfront
            self.front=Workfront(self);self.front.settings=deepcopy(self.front.settings)
            self.front.settings['base_chainage_m']=self.front.start
        self.initial_excavation=self.remaining

    def relax(self):return self.core.relax()

    @property
    def clearance_surface(self):
        return max(self.core.clearance_surface,self.shared_site.obstacle_top())

    def source_indices(self,source='cut'):
        if source=='cut' and self.front is None:
            ids=self._cut_indices
            return ids[self.height.ravel()[ids]-self.target.ravel()[ids]>1e-12]
        return super().source_indices(source)


class SharedSite:
    def __init__(self,project):
        self.project=project;self.core=project.units[0].simulation.terrain
        self.completion_volume=self.core.config.terrain.completion_volume
        self.owners=np.ones(self.core.height.shape,dtype=np.int16)
        self.assignments=0;self._starts={};self._airspace={}
        self._install(project.units[0]);self.partition()

    def _install(self,unit):
        s=unit.simulation
        if not isinstance(s.terrain,WorkTerrain):s.terrain=WorkTerrain(self.core,s.config,self,unit.id)
        s.shared_site=self

    def reserved_mask(self):
        t=self.core;mask=np.zeros(t.height.shape,dtype=bool)
        for unit in self.project.units:
            s=unit.simulation
            for d in s.drones:
                if d.trip is not None:
                    x,y=d.trip.dig;r=max(s.config.fleet.separation,d.config.grab_radius)
                    patch=t.patch(x,y,r);mask[patch]|=(t.x[patch]-x)**2+(t.y[patch]-y)**2<=r*r
            if s.logistics and s.logistics.excavator:
                e=s.logistics.excavator;r=e.config.reach_m
                if e.bucket_kg or e.state.value in ('DIGGING','FILLING_CONTAINER'):
                    x,y=e.position[:2];patch=t.patch(x,y,r)
                    mask[patch]|=(t.x[patch]-x)**2+(t.y[patch]-y)**2<=r*r
        return mask

    def partition(self):
        t=self.core;indices=np.flatnonzero(t.work_mask)
        if not len(indices):self._refresh();return
        fixed=self.reserved_mask().ravel()[indices]
        free=indices[~fixed];units=self.project.units
        if t.front is not None:
            along=t.front.chainage.ravel()[free]
            ordered=sorted(units,key=lambda u:float(t.front.chainage[min(t.height.shape[0]-1,max(0,int(u.simulation.base.position[1]/t.cell))),
                min(t.height.shape[1]-1,max(0,int(u.simulation.base.position[0]/t.cell)))]))
        else:
            coords=np.c_[t.x.ravel()[free],t.y.ravel()[free]]
            if len(coords)>1:
                _,_,axes=np.linalg.svd(coords-coords.mean(axis=0),full_matrices=False);axis=axes[0]
                if axis[np.argmax(abs(axis))]<0:axis=-axis
            else:axis=np.array([1.,0.])
            along=coords@axis;ordered=sorted(units,key=lambda u:float(u.simulation.base.position[:2]@axis))
        order=np.argsort(along,kind='stable');free=free[order]
        volume=np.abs(t.height.ravel()[free]-t.target.ravel()[free])*t.area
        cumulative=np.cumsum(volume);total=float(cumulative[-1]) if len(cumulative) else 0.
        if total>0:
            bins=np.minimum(len(units)-1,np.floor((cumulative-volume/2)/total*len(units)).astype(int))
            for i,unit in enumerate(ordered):self.owners.ravel()[free[bins==i]]=int(unit.id[8:])
        self.assignments+=1;self._refresh()

    def _refresh(self):
        from .simulation import MissionState
        for unit in self.project.units:
            self._install(unit);s=unit.simulation;t=s.terrain
            # The one global tolerance is divided, rather than allowing twenty
            # independent residuals to accumulate into a large unfinished cut.
            s.config.terrain.completion_volume=self.completion_volume/len(self.project.units)
            t.assign(self.owners==int(unit.id[8:]));s._capability_key=None
            if s.time==0 and t.front is not None:s._base_front_anchor=t.front.start
            if t.remaining>s.config.terrain.completion_volume and s.state==MissionState.COMPLETED:
                s.state=MissionState.RUNNING;s.reason='';unit.committed=False

    def advance_tasks(self):
        """Give a finished team the next unreserved part of the same map."""
        t=self.core
        if t.remaining<=self.completion_volume:return
        idle=[u for u in self.project.units if u.simulation.state.value=='COMPLETED']
        if not idle:return
        fixed=self.reserved_mask();available=t.work_mask&~fixed&(np.abs(t.height-t.target)>1e-10)
        changed=False
        for unit in idle:
            candidates=np.flatnonzero(available)
            if not len(candidates):break
            donor=max(self.project.units,key=lambda u:u.simulation.terrain.remaining)
            candidates=candidates[self.owners.ravel()[candidates]==int(donor.id[8:])]
            if not len(candidates):continue
            p=unit.simulation.base.position
            order=np.argsort((t.x.ravel()[candidates]-p[0])**2+(t.y.ravel()[candidates]-p[1])**2,kind='stable')
            take=candidates[order[:max(1,len(candidates)//2)]]
            self.owners.ravel()[take]=int(unit.id[8:]);available.ravel()[take]=False;changed=True
        if changed:self.assignments+=1;self._refresh()

    def apply_plan(self,plan):
        if self.project.transition is not None:raise ValueError('Сначала завершите или отмените региональный переезд')
        from copy import copy
        from .simulation import MissionState
        for u in self.project.units:
            s=u.simulation;w=s.logistics
            if s.base.destination is not None or not all(d.grounded and abs(d.cargo)<=1e-8 for d in s.drones):
                raise ValueError('Перед изменением общего плана посадите все комплексы')
            if w and (w.active or w.group or any(c.soil_kg for c in w.containers) or w.excavator and w.excavator.bucket_kg):
                raise ValueError('Завершите перевозки и выгрузку всего общего участка')
        t=self.core;candidate=copy(t);candidate.set_plan(plan)
        for u in self.project.units:
            s=u.simulation
            points=[s.slot_position(i) for i in range(len(s.slots))]+[s.wait_position(i) for i in range(len(s.drones))]
            for point in points:
                patch=t.patch(*point[:2],s.config.fleet.separation)
                near=(t.x[patch]-point[0])**2+(t.y[patch]-point[1])**2<s.config.fleet.separation**2
                if np.any(candidate.work_mask[patch]&near):raise ValueError('Общий план пересекает посадочную зону комплекса')
            if s.logistics:
                for rect in s.logistics.solar_ground_rects():
                    from .site_ground import rectangle_patch
                    patch=rectangle_patch(t,rect)
                    if patch is None or np.any(candidate.work_mask[patch]):raise ValueError('Общий план пересекает солнечное поле комплекса')
        for key in ('target','segment_labels','excavation','fill_mask','bottom','_cut_indices','_fill_indices'):setattr(t,key,getattr(candidate,key))
        t.revision+=1;t.initial_excavation=t.remaining;t.design_work=t.work_mask
        t.config.plan=deepcopy(plan)
        t.minimum_surface=float(min(t.reference.min(),t.target.min()));t._clearance_surface=max(float(t.reference.max()),float(t.dump_cap.max()),float(t.height.max()),float(t.target.max()))
        initial=t.height.tolist()
        for u in self.project.units:
            s=u.simulation;s.config.plan=deepcopy(plan);s.config.initial_height=initial
            s.state=MissionState.RUNNING;s.reason='';s.landing_requested=False;s._terrain_active=True
            s.relocation_pending=None;s.base_move_wait='';u.committed=False
            if s.logistics:
                for job in s.logistics.jobs:
                    if job.status=='QUEUED':s.logistics.cancel_job(job.id)
                s.logistics.progress_at=s.time
            s.event(f'Применён общий план: {len(plan.segments)} сегментов')
        if (plan.channel or {}).get('operations'):
            from .workfront import Workfront
            t.front=Workfront(t)
        else:t.front=None
        self.partition()

    def enlarge(self,side_m):
        """Extend a live map, conserving edited fine material and all machinery."""
        if self.project.transition is not None:raise ValueError('Сначала завершите или отмените региональный переезд')
        from .maps import workspace_cell
        from .power_design import equipment_copy
        from .plan import PlanConfig,PlanSegment
        old=self.core;old_side=old.height.shape[0]*old.cell;side=float(side_m)
        if not math.isfinite(side) or not old_side<=side<=1024:raise ValueError('Карту можно увеличивать до 1024 м')
        if side==old_side:return
        cell=workspace_cell(side,old.cell);n=round(side/cell)
        if abs(n*cell-side)>1e-8:raise ValueError('Размер должен быть кратен рабочей ячейке')
        if cell!=old.cell and any(not d.grounded for u in self.project.units for d in u.simulation.drones):
            raise ValueError('Для изменения разрешения посадите все комплексы; время и заряды сохранятся')
        c=equipment_copy(old.config)
        for key in ('terrain','trench','plan','georeference'):setattr(c,key,deepcopy(getattr(old.config,key)))
        c.terrain.n=n;c.terrain.cell_size=cell;c.terrain.grab_radius=max(cell,c.terrain.grab_radius)
        landscape=self.project.landscape
        origin=(c.georeference or {}).get('origin_m',[0.,0.])
        initial_soil=old.soil.export() if old.soil is not None else None
        if landscape is not None:
            reference,current=landscape.read_window(origin,n,cell_size=cell)
            if old.soil is not None:
                outside=landscape.soil_window(origin,n,cell)
                merged={(x,y):a for x,y,a in outside['tiles']}
                merged.update({(x,y):a for x,y,a in initial_soil['tiles']})
                initial_soil={**outside,'tiles':[[x,y,a] for (x,y),a in merged.items()]}
        else:
            reference=np.zeros((n,n),dtype=float);current=reference.copy()
        factor=round(cell/old.cell)
        def reduced(array):
            if factor==1:return array
            size=array.shape[0]//factor
            return array.reshape(size,factor,size,factor).mean(axis=(1,3))
        extent=old.height.shape[0]//factor
        reference[:extent,:extent]=reduced(old.reference);current[:extent,:extent]=reduced(old.height)
        baseline=current.copy();baseline[:extent,:extent]=reduced(old._initial_surface)
        c.reference_height=reference.tolist();c.initial_height=current.tolist();c.initial_soil=initial_soil
        c.local_reference=c.georeference is None
        if c.plan is None:
            q=c.trench;c.plan=PlanConfig([PlanSegment([[q.x-q.length/2,q.y],[q.x+q.length/2,q.y]],'dig',q.width,-q.depth)])
        if c.plan.channel is not None:
            c.plan.channel['cell_size']=cell
            if landscape is not None:
                from .channel import marine_window
                c.plan.channel['marine']=marine_window(landscape.raster,origin,side)
        c.terrain.floor=min(c.terrain.floor,float(current.min())-1.)
        replacement=Terrain(c);replacement._initial_surface=baseline;replacement.initial_excavation=old.initial_excavation
        if replacement.soil is not None:
            replacement._initial_soil_mass=replacement.soil.mass_delta-(old.soil.mass_delta-old._initial_soil_mass)
            replacement._initial_soil_volume=replacement.soil.volume_delta_by_material-(old.soil.volume_delta_by_material-old._initial_soil_volume)
        ownership=np.ones((n,n),dtype=np.int16)
        ownership[:extent,:extent]=self.owners[::factor,::factor][:extent,:extent]
        self.core=replacement;self.owners=ownership;self._mass_revision=-1
        for u in self.project.units:
            s=u.simulation;s.terrain.core=replacement
            if s.logistics is not None and s.logistics.excavator is not None:s.logistics.excavator._drive_cache=None
            s.config.terrain=deepcopy(c.terrain);s.config.plan=deepcopy(c.plan)
            s.config.initial_height=c.initial_height;s.config.reference_height=c.reference_height;s.config.initial_soil=c.initial_soil
            s.config.local_reference=c.local_reference
            s.config.georeference=deepcopy(c.georeference);s._terrain_active=True
            # Dirty display and capability caches refer to the new shared grid.
            s._capability_key=None;s._latitude_position=None
        self.partition()
        self.project.active.simulation.event(f'Общая карта увеличена до {side:g} м; грунт, техника и заряды сохранены')

    def start_tick(self):
        self._build_geometry()
        self._actors=[];self._actor_groups=[];self._actor_returns=[];owners=[];starts=[];radii=[]
        self._simulation_numbers={u.simulation:int(u.id[8:]) for u in self.project.units}
        separation=max(u.simulation.config.fleet.separation for u in self.project.units)
        for unit in self.project.units:
            number=int(unit.id[8:])
            w=unit.simulation.logistics
            for drone in unit.simulation.drones:
                self._actors.append(drone);self._actor_groups.append(False);self._actor_returns.append(drone)
                owners.append(number);starts.append(drone.position.copy());radii.append(separation)
            if w and w.group:
                self._actors.append(w.group);self._actor_groups.append(True);self._actor_returns.append(unit.simulation.drone)
                owners.append(number);starts.append(w.group.cargo_position.copy());radii.append(separation+w.config.support_half_width_m)
        self._actor_owners=np.array(owners);self._actor_starts=np.array(starts);self._actor_radii=np.array(radii)

    def collision(self,simulation,drone,old,new):
        own=self._simulation_numbers[simulation]
        ids=np.flatnonzero(self._actor_owners!=own)
        ids=np.array([i for i in ids if not self._actor_groups[i] or self._actors[i].phase in ('LIFT','FLY','LOWER')],dtype=int)
        if not len(ids):return None
        current=np.array([self._actors[i].cargo_position if self._actor_groups[i] else self._actors[i].position for i in ids])
        relative=np.asarray(old)-self._actor_starts[ids]
        velocity=(np.asarray(new)-old)-(current-self._actor_starts[ids])
        norm=np.einsum('ij,ij->i',velocity,velocity)
        times=np.divide(-np.einsum('ij,ij->i',relative,velocity),norm,out=np.zeros(len(ids)),where=norm>1e-18).clip(0,1)
        delta=relative+times[:,None]*velocity
        hits=np.flatnonzero(np.einsum('ij,ij->i',delta,delta)<np.maximum(0.,self._actor_radii[ids]-1e-9)**2)
        return self._actor_returns[ids[hits[0]]] if len(hits) else None

    def obstacle_top(self):
        self._ensure_geometry();return self._obstacle_top

    def obstacle_height(self,point):
        self._ensure_geometry()
        height=-math.inf
        for row in self._ground_grid.get((math.floor(point[0]/16),math.floor(point[1]/16)),()):
            _,x,y,w,h,top=row
            if x<=point[0]<=x+w and y<=point[1]<=y+h:height=max(height,top)
        return height

    def reservations(self,simulation,kind):
        points=[]
        for unit in self.project.units:
            s=unit.simulation
            if s is simulation:continue
            for d in s.drones:
                if d.trip is not None:points.append(d.trip.dig if kind=='dig' else d.trip.dump)
            if kind=='dig' and s.logistics and s.logistics.excavator:points.append(s.logistics.excavator.position[:2])
            if kind=='dump' and s.logistics:
                points.extend(j.destination[:2] for j in s.logistics.jobs if j.kind=='PICKUP_FULL_CONTAINER' and j.status in ('ACTIVE','QUEUED'))
        return points

    def ground_occupied(self,owner,rect):
        self._ensure_geometry()
        x,y,w,h=rect;seen=set()
        for gx in range(math.floor((x-2)/16),math.floor((x+w+2)/16)+1):
            for gy in range(math.floor((y-2)/16),math.floor((y+h+2)/16)+1):
                for row in self._ground_grid.get((gx,gy),()):
                    if row[0]==owner or row in seen:continue
                    seen.add(row);_,xx,yy,ww,hh,_=row
                    if x<xx+ww+2 and x+w+2>xx and y<yy+hh+2 and y+h+2>yy:return True
        return False

    def ground_token(self,owner,bounds):
        """Actual nearby obstacle geometry; distant moving units do not invalidate routes."""
        self._ensure_geometry();x,y,x1,y1=bounds;rows=set()
        for gx in range(math.floor((x-2)/16),math.floor((x1+2)/16)+1):
            for gy in range(math.floor((y-2)/16),math.floor((y1+2)/16)+1):
                for row in self._ground_grid.get((gx,gy),()):
                    other,xx,yy,w,h,_=row
                    if other!=owner and x<xx+w+2 and x1+2>xx and y<yy+h+2 and y1+2>yy:rows.add(row)
        return tuple(sorted(rows))

    def invalidate_geometry(self):self._geometry_time=None

    def placement_goal(self):
        t=self.core;ids=np.flatnonzero(t.work_mask&(np.abs(t.height-t.target)>1e-10))
        if not len(ids):ids=np.flatnonzero(t.work_mask)
        if not len(ids):return np.array(t.height.shape)*t.cell/2
        if t.front is not None:
            along=t.front.chainage.ravel()[ids]
            values=[t.front.chainage[min(t.height.shape[0]-1,max(0,int(u.simulation.base.position[1]/t.cell))),min(t.height.shape[1]-1,max(0,int(u.simulation.base.position[0]/t.cell)))] for u in self.project.units]
        else:
            axis=np.array([1.,0.])
            if t.config.plan and t.config.plan.segments:
                segment=max(t.config.plan.segments,key=lambda s:np.linalg.norm(np.asarray(s.points[-1])-s.points[0]))
                delta=np.asarray(segment.points[-1])-segment.points[0]
                if np.linalg.norm(delta)>1e-9:axis=delta/np.linalg.norm(delta)
            along=t.x.ravel()[ids]*axis[0]+t.y.ravel()[ids]*axis[1]
            values=[u.simulation.base.position[:2]@axis for u in self.project.units]
        lo,hi=float(along.min()),float(along.max());bounds=np.sort(np.r_[lo,np.clip(values,lo,hi),hi])
        k=int(np.argmax(np.diff(bounds)));wanted=(bounds[k]+bounds[k+1])/2
        index=ids[int(np.argmin(abs(along-wanted)))];return np.array([t.x.ravel()[index],t.y.ravel()[index]])

    def pads_bounds(self,config,point,spec=None):
        from .equipment import station_offset
        from .base_layout import rotate_offset
        f=config.fleet
        offsets=(spec or {}).get('dock_offsets',[])
        distance=max(f.waiting_offset,config.base.radius+3.+f.separation) if offsets else f.waiting_offset
        points=np.array([np.asarray(point)+rotate_offset(config.base.layout,[dx,offsets[i][1] if i<len(offsets) else station_offset(i,f.pad_spacing)])
            for i in range(max(f.count,f.charge_slots)) for dx in (0.,-distance)])
        lower=points.min(axis=0)-f.separation;upper=points.max(axis=0)+f.separation
        return (*lower,*(upper-lower))

    def _ensure_geometry(self):
        if getattr(self,'_geometry_time',None)!=self.project.time:self._build_geometry()

    def _build_geometry(self):
        from .base_layout import layout_footprints
        self._ground_grid={};self._obstacle_top=-math.inf;self._geometry_time=self.project.time
        for unit in self.project.units:
            s=unit.simulation;b=s.base;r=b.config.radius
            boxes=list(layout_footprints(b.layout_spec,b.position)) if b.config.layout is not None else [(b.position[0]-r,b.position[1]-r,2*r,2*r)]
            top=float(b.position[2]+b.config.roof_height);self._obstacle_top=max(self._obstacle_top,top)
            boxes=[(*box,top) for box in boxes]
            boxes.append((*self.pads_bounds(s.config,b.position[:2],getattr(b,'layout_spec',None)),float(b.position[2])))
            if s.logistics and s.logistics.excavator and s.logistics.excavator.state.value!='AIR_TRANSPORT':
                e=s.logistics.excavator;q=e.config.support_half_width_m
                boxes.append((e.position[0]-q,e.position[1]-q,2*q,2*q,float(e.position[2]+1.)))
            if s.logistics:
                for rect in s.logistics.solar_ground_rects():boxes.append((*rect,float(b.position[2]+1.4)))
                if s.logistics.battery_platforms:
                    for rect in s.logistics.battery_platforms.rectangles():boxes.append((*rect,float(b.position[2]+.5)))
            for xx,yy,ww,hh,z in boxes:
                margin=b.config.speed if b.destination is not None else 0.
                row=(unit.id,float(xx-margin),float(yy-margin),float(ww+2*margin),float(hh+2*margin),z)
                for gx in range(math.floor(row[1]/16),math.floor((row[1]+row[3])/16)+1):
                    for gy in range(math.floor(row[2]/16),math.floor((row[2]+row[4])/16)+1):self._ground_grid.setdefault((gx,gy),[]).append(row)

    def pads_clear(self,config,point,spec=None):
        from .equipment import station_offset
        from .base_layout import rotate_offset
        t=self.core;f=config.fleet;n=t.height.shape[0];side=n*t.cell
        offsets=(spec or {}).get('dock_offsets',[])
        distance=max(f.waiting_offset,config.base.radius+3.+f.separation) if offsets else f.waiting_offset
        for i in range(max(f.count,f.charge_slots)):
            for dx in (0.,-distance):
                x,y=point+rotate_offset(config.base.layout,[dx,offsets[i][1] if i<len(offsets) else station_offset(i,f.pad_spacing)])
                if not t.cell<x<side-t.cell or not t.cell<y<side-t.cell:return False
                patch=t.patch(x,y,f.separation)
                mask=(t.x[patch]-x)**2+(t.y[patch]-y)**2<f.separation**2
                blocked=t.design_work[patch]|t.dump_mask[patch]
                if t.protected is not None:blocked=blocked|t.protected[patch]
                from .site_ground import marine_mask
                blocked=blocked|marine_mask(t)[patch]
                if np.any(mask&blocked):return False
        return True

    def loads(self):
        for u in self.project.units:
            s=u.simulation
            for d in s.drones:
                if d.soil is not None:yield d.soil
            if s.logistics:
                for c in s.logistics.containers:
                    if c.soil is not None:yield c.soil
                e=s.logistics.excavator
                if e and e.soil is not None:yield e.soil

    def mass_error(self):
        soil=0.
        for u in self.project.units:
            s=u.simulation;w=s.logistics
            soil+=sum(d.cargo for d in s.drones if not w or d.identifier in w.direct_ids)
            if w:
                soil+=sum(c.soil_kg for c in w.containers)
                if w.excavator:soil+=w.excavator.bucket_kg
        if getattr(self,'_mass_revision',None)!=self.core.revision:
            self._ground_mass=self.core.mass_change;self._mass_revision=self.core.revision
        return self._ground_mass+soil+getattr(self,'archived_mass_error',0.)

    def material_errors(self):
        t=self.core
        if t.soil is None:return None
        error=(t.soil.volume_delta_by_material-t._initial_soil_volume)*t.soil.profile.densities
        archived=getattr(self,'archived_material_errors',None)
        if archived is not None:error+=archived
        for load in self.loads():error+=load.masses
        return error

    def validate(self):
        if self.owners.shape!=self.core.height.shape or self.owners.dtype.kind not in 'iu':raise ValueError('Повреждён диспетчер общего участка')
        ids={int(u.id[8:]) for u in self.project.units}
        if not set(np.unique(self.owners[self.core.work_mask])).issubset(ids):raise ValueError('Участок принадлежит отсутствующему комплексу')
        for u in self.project.units:
            t=u.simulation.terrain
            if not isinstance(t,WorkTerrain) or t.core is not self.core or t.shared_site is not self or t.owner_id!=u.id:
                raise ValueError('Потеряна связь с общим грунтом')
            expected=self.owners==int(u.id[8:])
            if not np.array_equal(t.excavation,self.core.excavation&expected) or not np.array_equal(t.fill_mask,self.core.fill_mask&expected):
                raise ValueError('Пересекающиеся или потерянные задания комплексов')
