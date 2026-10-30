"""A physical group transfer through the overlap of two bounded DEM windows.

The old window remains live until every actor fits the overlap. Only then is
terrain committed and the coordinate frame changed; changing that frame never
moves a physical actor. Inactive ground is kept in Landscape's sparse tiles.
"""
from copy import copy,deepcopy
import heapq
import math
import numpy as np
from .inventory import Location as L
from .drone import DroneState as DS
from .excavator import ExcavatorState as ES
from .simulation import MissionState
from .landscape import workspace_config
from .terrain import Terrain


GEOMETRY=('terrain','trench','plan','georeference','local_reference')


def window_key(origin):return tuple(map(float,origin))


def target_configuration(project,origin,section=None,landscape=None):
    land=landscape or project.landscape;source=project.units[0].simulation.config
    side=source.terrain.n*source.terrain.cell_size
    old=np.asarray(source.georeference['origin_m']);cell=source.terrain.cell_size
    def aligned(point):
        point=np.asarray(point,dtype=float)
        if point.shape!=(2,) or not np.all(np.isfinite(point)):raise ValueError('Нужны конечные координаты начала окна X/Y')
        limit=np.asarray(land.raster.extent_m)-side
        if np.any(point<0) or np.any(point>limit+1e-8):raise ValueError('Участок выходит за границы DEM')
        steps=np.clip(np.rint((point-old)/cell),np.ceil(-old/cell),np.floor((limit-old)/cell))
        return (old+steps*cell).tolist()
    if section is not None:
        from .channel import ChannelProject
        if land.channel_project is None:raise ValueError('Нет регионального проекта канала')
        channel=ChannelProject.from_data(land.channel_project)
        if type(section) is not int or not 0<=section<channel.section_count:raise ValueError('Участок вне проекта канала')
        if channel.side_m!=side:raise ValueError('Для переезда сохраняйте размер рабочего окна')
        return channel.configuration(land,source,section,origin=aligned(channel.origin(land.raster,section)))
    origin=aligned(origin)
    c=workspace_config(land,source,origin,side)
    stored=getattr(project,'window_plans',{}).get(window_key(origin))
    if stored:
        for key in GEOMETRY:setattr(c,key,deepcopy(stored[key]))
    elif source.plan is not None and source.plan.channel is not None:
        raise ValueError('Для нового участка канала укажите номер участка проекта')
    c.validate();return c


def ground_route(actor,terrain,destination,*,base=False):
    """Direct segment first, then bounded deterministic eight-neighbour A*."""
    destination=np.asarray(destination,dtype=float)[:2];start=actor.position[:2].copy()
    probe=copy(actor);probe.position=actor.position.copy()
    def clear(a,b):
        probe.position=np.r_[a,terrain.sample(*a)]
        if not own_ground_clear(actor,terrain,a,b,base=base):return False
        return probe.safe_path(b,terrain) if base else probe.can_drive(np.r_[b,terrain.sample(*b)],terrain)
    if clear(start,destination):return [destination.tolist()]
    side=terrain.cell*terrain.height.shape[0];spacing=max(2.,side/40)
    nodes=[start,destination]
    axis=np.arange(spacing,side,spacing)
    nodes.extend(np.array([x,y]) for y in axis for x in axis)
    n=len(axis);start_links=sorted(range(2,len(nodes)),key=lambda i:float(np.sum((nodes[i]-start)**2)))[:12]
    goal_links=set(sorted(range(2,len(nodes)),key=lambda i:float(np.sum((nodes[i]-destination)**2)))[:12])
    parents={};costs={0:0.};queue=[(float(np.linalg.norm(start-destination)),0.,0)];visited=set();checks=0
    while queue and checks<8000:
        _,cost,index=heapq.heappop(queue)
        if index in visited:continue
        if index==1:
            path=[]
            while index: path.append(nodes[index].tolist());index=parents[index]
            return path[::-1]
        visited.add(index)
        if index==0:links=start_links
        else:
            y,x=divmod(index-2,n)
            links=[2+(y+dy)*n+x+dx for dy,dx in ((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1))
                   if 0<=y+dy<n and 0<=x+dx<n]
            if index in goal_links:links=[1,*links]
        for other in links:
            if other in visited:continue
            trial=cost+float(np.linalg.norm(nodes[index]-nodes[other]))
            if trial>=costs.get(other,math.inf):continue
            checks+=1
            if not clear(nodes[index],nodes[other]):continue
            parents[other]=index;costs[other]=trial
            heapq.heappush(queue,(trial+float(np.linalg.norm(nodes[other]-destination)),trial,other))
    return None


def own_ground_clear(actor,terrain,a,b,*,base=False):
    """The shared dispatcher excludes the owner; its own base/robot still collide."""
    site=getattr(terrain,'shared_site',None)
    if site is None:return True
    from .base_layout import layout_footprints
    s=next(u.simulation for u in site.project.units if u.id==terrain.owner_id)
    points=np.linspace(a,b,max(2,math.ceil(float(np.linalg.norm(np.asarray(a)-b))/max(.25,terrain.cell))+1))
    if base:
        e=s.logistics.excavator
        if e is None:return True
        r=actor.config.radius;margin=e.config.support_half_width_m+2
        boxes=list(layout_footprints(actor.layout_spec,[0.,0.])) if actor.config.layout is not None else [(-r,-r,2*r,2*r)]
        for x,y,w,h in boxes:
            if np.any((points[:,0]+x<e.position[0]+margin)&(points[:,0]+x+w>e.position[0]-margin)&
                      (points[:,1]+y<e.position[1]+margin)&(points[:,1]+y+h>e.position[1]-margin)):return False
        return True
    own=s.base;r=own.config.radius;margin=actor.config.support_half_width_m+.2
    boxes=list(layout_footprints(own.layout_spec,own.position)) if own.config.layout is not None else [(own.position[0]-r,own.position[1]-r,2*r,2*r)]
    if s.logistics.battery_platforms:boxes.extend(s.logistics.battery_platforms.rectangles())
    return not any(np.any((points[:,0]>x-margin)&(points[:,0]<x+w+margin)&(points[:,1]>y-margin)&(points[:,1]<y+h+margin)) for x,y,w,h in boxes)


class RegionalTransition:
    @classmethod
    def prepare(cls,project,origin,section):
        if project.landscape is None:raise ValueError('Переезд требует открытого регионального DEM')
        if project.blocked:raise ValueError('Сначала устраните блокировку комплекса')
        if project.shared_site is None and len(project.units)!=1:
            raise ValueError('Переезд группы доступен для комплексов на одной общей карте')
        for u in project.units:
            s=u.simulation
            if s.logistics is None:raise ValueError('Переезд группы требует ATS 2.0 с физическим инвентарём')
            if s.pending_equipment is not None:raise ValueError('Завершите изменение оборудования перед переездом')
        s=project.units[0].simulation;geo=s.config.georeference
        if not geo:raise ValueError('У действующей карты нет региональной привязки')
        old=np.asarray(geo['origin_m'],dtype=float);side=s.terrain.height.shape[0]*s.terrain.cell
        # A preview receives the live overlay without modifying the user's region.
        preview=copy(project.landscape);preview.tiles=dict(preview.tiles);preview.soil_tiles=dict(preview.soil_tiles)
        preview.commit_window(old,s.terrain.height,cell_size=s.terrain.cell,soil=s.terrain.soil)
        c=target_configuration(project,origin,section,preview);new=np.asarray(c.georeference['origin_m'])
        if np.array_equal(old,new):raise ValueError('Выбрано текущее рабочее окно')
        lo=np.maximum(old,new);hi=np.minimum(old+side,new+side)
        if np.any(hi-lo<=4*s.terrain.cell):raise ValueError('Нужны соседние перекрывающиеся окна; выберите более близкий участок')
        if c.terrain.cell_size!=s.terrain.cell:raise ValueError('Переезд не меняет разрешение активной сетки')
        self=cls();self.project=project;self.old_origin=old.tolist();self.new_origin=new.tolist()
        self.section=section;self.target=c;self.preview=Terrain(c);self.started_at=project.time
        self.phase='DRAIN';self.reason='Завершение рейсов и выгрузка грунта';self.routes={};self.goals={}
        self.actor_index=0;self.actors=[];self.followers={};self.previous_landing={u.id:u.simulation.landing_requested for u in project.units}
        self.overlap=(lo-old).tolist()+(hi-old).tolist();self.retry_at=0.;self.route_revision=None
        self._choose_goals()
        self.preview=None  # Keep the preloaded target data, not a second live physics grid.
        return self

    def _choose_goals(self):
        p=self.project;old=np.array(self.old_origin);new=np.array(self.new_origin);offset=old-new
        lo=np.array(self.overlap[:2]);hi=np.array(self.overlap[2:]);placed=[]
        for u in p.units:
            s=u.simulation;b=s.base;r=b.config.radius
            pad=(max(len(s.slots),len(s.drones))//2)*s.config.fleet.pad_spacing+s.config.fleet.waiting_offset+s.config.fleet.separation+2
            from .base_layout import layout_footprints
            def footprint(point):
                boxes=list(layout_footprints(b.layout_spec,point)) if b.config.layout is not None else [(point[0]-r,point[1]-r,2*r,2*r)]
                if p.shared_site:boxes.append(p.shared_site.pads_bounds(s.config,point,getattr(b,'layout_spec',None)))
                if s.logistics.battery_platforms:
                    from .battery_platforms import deployment_rectangles
                    boxes.extend(deployment_rectangles(b.layout_spec,point))
                if b.config.layout is not None:
                    from .self_solar import deployment_rectangles as solar_rectangles
                    boxes.extend(solar_rectangles(b.layout_spec,point))
                    from .compact_layout import rectangles as access_rectangles
                    boxes.extend(access_rectangles(b.layout_spec,point))
                return (min(x for x,y,w,h in boxes),min(y for x,y,w,h in boxes),
                        max(x+w for x,y,w,h in boxes)-min(x for x,y,w,h in boxes),
                        max(y+h for x,y,w,h in boxes)-min(y for x,y,w,h in boxes))
            box=footprint(np.zeros(2));extent=max(abs(box[0]),abs(box[1]),abs(box[0]+box[2]),abs(box[1]+box[3]))
            margin=max(r+2,pad,extent+2);lower=lo+margin;upper=hi-margin
            if np.any(lower>upper):raise ValueError(f'{u.name}: техника и посадочные зоны не помещаются в перекрытии окон; увеличьте перекрытие')
            ideal=np.clip(b.position[:2]+(new-old)/2,lower,upper)
            candidates=[ideal,*[np.array([x,y]) for y in np.linspace(lower[1],upper[1],17) for x in np.linspace(lower[0],upper[0],17)]]
            candidates.sort(key=lambda a:float(np.sum((a-ideal)**2)))
            chosen=None
            for point in candidates:
                rect=footprint(point)
                x,y,w,h=rect
                if any(x<xx+ww+3 and x+w+3>xx and y<yy+hh+3 and y+h+3>yy for xx,yy,ww,hh in placed):continue
                probe=copy(b);probe.position=np.r_[point,s.terrain.sample(*point)]
                if not probe.safe_path(point,s.terrain):continue
                probe.position=np.r_[point+offset,self.preview.sample(*(point+offset))]
                if not probe.safe_path(point+offset,self.preview):continue
                from .battery_platforms import deployment_safe
                if b.config.layout is not None and not deployment_safe(b.layout_spec,point,s.terrain):continue
                if p.shared_site and not p.shared_site.pads_clear(s.config,point,getattr(b,'layout_spec',None)):continue
                if p.shared_site:
                    target_site=copy(p.shared_site);target_site.core=self.preview
                    if not target_site.pads_clear(s.config,point+offset,getattr(b,'layout_spec',None)):continue
                if b.config.layout is not None and not deployment_safe(b.layout_spec,point+offset,self.preview):continue
                if b.config.layout is not None:
                    from .self_solar import deployment_safe as solar_safe
                    if not solar_safe(b.layout_spec,point,s.terrain) or not solar_safe(b.layout_spec,point+offset,self.preview):continue
                e=s.logistics.excavator;eg=None
                if e:
                    support=e.config.support_half_width_m
                    offsets=[(rect[0]+rect[2]-point[0]+support+3,0),(rect[0]-point[0]-support-3,0),
                             (0,rect[1]+rect[3]-point[1]+support+3),(0,rect[1]-point[1]-support-3)]
                    offsets.sort(key=lambda delta:-float(np.sum((point+delta-b.position[:2])**2)))
                    for dx,dy in offsets:
                        test=point+[dx,dy]
                        if np.any(test<lo+2) or np.any(test>hi-2):continue
                        if any(xx-support-2<test[0]<xx+ww+support+2 and yy-support-2<test[1]<yy+hh+support+2 for xx,yy,ww,hh in placed):continue
                        patch=self.preview.patch(*(test+offset),support+self.preview.cell)
                        blocked=self.preview.design_work[patch]|self.preview.dump_mask[patch]
                        if self.preview.protected is not None:blocked|=self.preview.protected[patch]
                        if np.any(blocked):continue
                        if e.stable(np.r_[test,s.terrain.sample(*test)],s.terrain) and e.stable(np.r_[test+offset,self.preview.sample(*(test+offset))],self.preview):eg=test.tolist();break
                    if eg is None:continue
                chosen=point.tolist();placed.append(rect)
                if eg:placed.append((eg[0]-support,eg[1]-support,2*support,2*support))
                self.goals[u.id]=dict(base=chosen,excavator=eg);break
            if chosen is None:raise ValueError(f'{u.name}: нет безопасного места в перекрытии для всей группы')

    def start(self):
        p=self.project;p.enable_shared()
        if not hasattr(p,'window_plans'):p.window_plans={}
        if not hasattr(p,'window_history'):p.window_history=[]
        self.start_consumed_kwh=p.report()['consumed_kwh']
        for u in p.units:
            s=u.simulation;s.regional_stage='DRAIN';s.request_landing();s.state=MissionState.RUNNING;s.reason=''
            for job in s.logistics.jobs:
                if job.status=='QUEUED' and job.kind!='PICKUP_FULL_CONTAINER':s.logistics.cancel_job(job.id)
            s.event('Переезд группы: завершаем рейсы и собираем оборудование')

    def _stage(self,phase,reason):
        self.phase=phase;self.reason=reason
        for u in self.project.units:u.simulation.regional_stage=phase

    def _settled(self):
        for u in self.project.units:
            s=u.simulation;w=s.logistics;e=w.excavator
            if self.phase=='ARRIVE' and s._terrain_active:return False
            if (s.pending_equipment is not None or w.active or w.group or w.direct_ids or w.tool_changes or w.tool_targets or w.battery_service.robot_swap or
                    w.battery_service.drone_swaps or w.battery_platforms and w.battery_platforms.returns or s.base.destination is not None or
                    any(not d.grounded or abs(d.cargo)>1e-8 for d in s.drones) or
                    any(c.soil_kg>1e-8 for c in w.containers) or e and (e.bucket_kg>1e-8 or e.active_container)):
                return False
        return True

    def _plan_routes(self):
        # Fields will be folded before any ground actor is allowed to move.
        # Omit them only from this future-route preview, never live collisions.
        fields=[u.simulation.logistics.field_rect for u in self.project.units]
        try:
            for u in self.project.units:
                w=u.simulation.logistics;w.field_rect=None
                if w.battery_platforms:w.battery_platforms.route_preview=True
            self.project.shared_site.invalidate_geometry()
            result=self._routes_without_fields()
        finally:
            for u,field in zip(self.project.units,fields):
                w=u.simulation.logistics;w.field_rect=field
                if w.battery_platforms:w.battery_platforms.route_preview=False
            self.project.shared_site.invalidate_geometry()
        self.last_failed_route=None if result else self._route_environment()
        return result

    def _routes_without_fields(self):
        self.route_generation=getattr(self,'route_generation',0)+1
        self.routes={};self.actors=[]
        poses=[(actor,actor.position.copy()) for u in self.project.units
               for actor in (u.simulation.base,u.simulation.logistics.excavator) if actor is not None]
        try:
            # Preview the actual sequence: each robot clears its old base, then
            # that base departs. No physical tick or power integration occurs
            # inside this preview; every pose is restored before returning.
            for u in self.project.units:
                s=u.simulation;w=s.logistics
                for kind,actor in (('excavator',w.excavator),('base',s.base)):
                    if actor is None:continue
                    goal=self.goals[u.id][kind];path=ground_route(actor,s.terrain,goal,base=kind=='base')
                    if path is None:
                        self.reason=f'{u.name}: нет безопасного пути для '+('базы' if kind=='base' else 'Крота');return False
                    self.routes[u.id,kind]=path;self.actors.append((u.id,kind))
                    actor.position[:]=[*goal,s.terrain.sample(*goal)]
                    self.project.shared_site.invalidate_geometry()
            return True
        finally:
            for actor,pose in poses:actor.position[:]=pose
            self.project.shared_site.invalidate_geometry()

    def _route_environment(self):
        return (self.project.shared_site.core.revision,tuple((u.id,tuple(u.simulation.base.position),
            tuple(u.simulation.logistics.excavator.position) if u.simulation.logistics.excavator else None,
            u.simulation.base.config.mass,repr(u.simulation.base.config.layout)) for u in self.project.units))

    def _movement_budget(self):
        p=self.project
        key=(id(self.routes),getattr(self,'route_generation',0),p.shared_site.core.revision,tuple(
            (u.id,u.simulation.base.config.mass,u.simulation.base.config.battery,u.simulation.base.config.speed,
             u.simulation.base.config.move_power,u.simulation.base.config.idle_power,repr(u.simulation.base.config.layout),
             tuple(u.simulation.base.position),
             (tuple(u.simulation.logistics.excavator.position),u.simulation.logistics.excavator.capacity(u.simulation.logistics.inventory),
              u.simulation.logistics.config.moving_power_w,u.simulation.logistics.config.excavator_speed,
              u.simulation.logistics.config.sleep_power_w,u.simulation.logistics.config.idle_power_w) if u.simulation.logistics.excavator else None,
             tuple((d.config.empty_mass,d.cargo,d.docked) for d in u.simulation.drones),
             tuple((id,u.simulation.logistics.inventory.items[id].max_discharge_w) for id in u.simulation.logistics.base_battery_ids),
             u.simulation.logistics.config.inverter_power_w,u.simulation.logistics.field_setup_seconds())
            for u in p.units))
        if getattr(self,'_budget_key',None)==key:return self._budget
        duration=0.
        for (owner,kind),path in self.routes.items():
            s=next(u.simulation for u in self.project.units if u.id==owner)
            actor=s.base if kind=='base' else s.logistics.excavator;previous=actor.position[:2]
            length=sum(float(np.linalg.norm(np.array(point)-(previous if i==0 else np.array(path[i-1])))) for i,point in enumerate(path))
            duration+=length/(actor.config.speed if kind=='base' else actor.config.excavator_speed)
        budget={}
        for u in self.project.units:
            s=u.simulation;w=s.logistics;b=s.base;probe=copy(b);probe.position=b.position.copy();cost=0.
            for point in self.routes[u.id,'base']:
                cost+=probe.route_energy(point,s.terrain,s.drones);probe.position=np.r_[point,s.terrain.sample(*point)]
            array=getattr(w,'solar_array',None)
            handling=2*array.energy_budget() if array else 400*w.field_setup_seconds()
            required=cost*1.3+b.config.idle_power*(duration+2*w.field_setup_seconds())+handling
            if w.battery_platforms:
                required+=2*w.battery_platforms.energy_budget()+2*b.config.idle_power*sum(w.battery_platforms.settings[k] for k in ('support_seconds','apron_seconds','connect_seconds'))
            budget[u.id]=[required,0.,b.motion_power(s.drones,s.terrain)]
            e=w.excavator
            if e:
                prev=e.position[:2];cost=0.
                for point in self.routes[u.id,'excavator']:
                    delta=float(np.linalg.norm(np.array(point)-prev));grade=abs(s.terrain.sample(*point)-s.terrain.sample(*prev))/max(delta,s.terrain.cell)
                    cost+=(e.config.moving_power_w*(1.2+5*grade)+e.config.idle_power_w)*math.ceil(delta/e.config.excavator_speed);prev=np.array(point)
                budget[u.id][1]=cost*1.3+e.config.sleep_power_w*duration+e.config.idle_power_w*900
        self._budget_key=key;self._budget=budget
        return budget

    def _energy_ready(self):
        budget=self._movement_budget();ready=True
        for u in self.project.units:
            s=u.simulation;w=s.logistics;b=s.base
            required,robot_required,drive_power=budget[u.id]
            if required>b.config.battery:self.reason=f'{u.name}: ёмкости базы недостаточно для переезда; добавьте аккумуляторы';ready=False
            elif b.battery<required:self.reason=f'{u.name}: ждём заряд базы для переезда';ready=False
            if drive_power>min(w.config.inverter_power_w,w.storage_power('discharge')):
                self.reason=f'{u.name}: недостаточная мощность привода/инвертора';ready=False
            blocked=getattr(self,'blocked_power',None)
            if blocked and blocked[0]==u.id and blocked[2]==self.project.shared_site.core.revision:
                if min(w.config.inverter_power_w,w.storage_power('discharge'))<blocked[1]:
                    self.reason=f'{u.name}: недостаточная мощность для подъёма; требуется {blocked[1]/1000:.1f} кВт';ready=False
            e=w.excavator
            if e:
                required=robot_required
                e.regional_energy_required=min(required,e.capacity(w.inventory)*.99)
                if required>e.capacity(w.inventory):self.reason=f'{u.name}: батарей Крота недостаточно для пути';ready=False
                elif e.energy(w.inventory)<required:self.reason=f'{u.name}: ждём доставки заряженных батарей Кроту';ready=False
            for d in s.drones:
                if d.docked:continue
                if d.charge_requested and d.battery<d.config.battery-1e-7:
                    s._enqueue(d);ready=False;self.reason=f'{u.name}: дрон завершает полную зарядку перед перелётом';continue
                distance=float(np.linalg.norm(d.position[:2]-self.goals[u.id]['base']))+s.config.fleet.waiting_offset
                seconds=distance/d.config.empty_speed+2*max(0.,s.cruise(d)-d.position[2])/d.config.vertical_speed+90
                required=d.power(True)*seconds*d.config.budget_margin+d.config.battery*d.config.reserve
                if d.battery<required:s._request_charge(d,'перелёт вслед за базой');s._enqueue(d);ready=False;self.reason=f'{u.name}: заряд дрона для перелёта'
        return ready

    def before_step(self):
        p=self.project
        if self.phase=='DEPLOY':
            if p.time>=self.retry_at:
                self.retry_at=p.time+30;self._deploy_fields()
            return
        if self.phase in ('DRAIN','COLLECT','ENERGY'):
            for u in p.units:
                s=u.simulation
                # Interrupted tool requests must not prevent preparation forever.
                for identifier in list(s.logistics.tool_targets):
                    if s.logistics.tool_targets[identifier]!='CONTAINER_HOOK':s.logistics.request_tool(s.drones[identifier],s.drones[identifier].attachment)
            return
        if self.phase=='DRIVE' and self.actor_index<len(self.actors):
            id,kind=self.actors[self.actor_index];s=next(u.simulation for u in p.units if u.id==id)
            actor=s.base if kind=='base' else s.logistics.excavator;path=self.routes[id,kind]
            if not path:return
            goal=np.asarray(path[0]);distance=float(np.linalg.norm(actor.position[:2]-goal))
            if distance<1e-7:
                path.pop(0)
                if not path:self.actor_index+=1
                self.retry_at=0.
                return
            if actor.destination is not None and (kind=='base' or actor.state==ES.MOVING_LOCAL):
                if kind=='base':self._check_base_step(id,s,goal,distance)
                return
            if p.time<self.retry_at:return
            self.retry_at=p.time+30
            valid=(actor.safe_path(goal,s.terrain) and own_ground_clear(actor,s.terrain,actor.position[:2],goal,base=True)
                   if kind=='base' else actor.can_drive(np.r_[goal,s.terrain.sample(*goal)],s.terrain))
            if not valid:
                replacement=ground_route(actor,s.terrain,self.goals[id][kind],base=kind=='base')
                if replacement is None:
                    self.routes={};self._unfold_for_wait(f'{id}: путь перекрыт; ожидание, можно отменить переезд');return
                self.routes[id,kind]=replacement;return
            actor.destination=goal.copy() if kind=='base' else np.r_[goal,s.terrain.sample(*goal)]
            if kind=='excavator':actor.state=ES.MOVING_LOCAL;actor.supports=False
            self.reason=f'{id}: '+('база движется' if kind=='base' else 'Крот движется')
            if kind=='base':self._check_base_step(id,s,goal,distance)

    def _check_base_step(self,id,s,goal,distance):
        actor=s.base;delta=goal-actor.position[:2]
        point=actor.position[:2]+delta*min(1.,actor.config.speed/max(distance,1e-9))
        reason=''
        if not actor.safe_path(point,s.terrain) or not own_ground_clear(actor,s.terrain,actor.position[:2],point,base=True):reason=f'{id}: новое препятствие, перепланирование пути'
        else:
            mass=(actor.config.mass+sum(d.config.empty_mass+d.cargo for d in s.drones if d.docked)
                  if actor.config.layout is not None else actor.motion_power(s.drones)/actor.config.move_power*actor.config.reference_total_mass)
            efficiency=actor.layout_spec['parameters']['drive_efficiency'] if actor.config.layout is not None else .8
            needed=actor.motion_power(s.drones,s.terrain)*min(1.,distance/actor.config.speed)
            needed+=mass*9.80665*max(0.,s.terrain.sample(*point)-actor.position[2])/efficiency
            w=s.logistics
            if needed>min(w.config.inverter_power_w,w.storage_power('discharge')):
                self.blocked_power=(id,needed,self.project.shared_site.core.revision)
                reason=f'{id}: недостаточно мощности для следующего шага'
        if reason:
            actor.destination=None;self.routes={};self._unfold_for_wait(reason)

    def after_step(self):
        p=self.project
        if self.phase=='DEPLOY':
            if all(u.simulation.logistics.field_ready and not getattr(u.simulation.logistics,'regional_folded',False) and
                   (u.simulation.logistics.battery_platforms is None or u.simulation.logistics.battery_platforms.ready) for u in p.units):self._release('COMPLETED')
            return
        if self.phase=='DRAIN' and self._settled():
            if not self._plan_routes():self.retry_at=p.time+30;self._stage('ENERGY',self.reason);return
            self._stage('ENERGY','Проверка энергетического резерва');return
        if self.phase=='ENERGY':
            if not self.routes or len(self.routes)<sum(1+(u.simulation.logistics.excavator is not None) for u in p.units):
                if p.time<self.retry_at:return
                if getattr(self,'last_failed_route',None)==self._route_environment():return
                self.retry_at=p.time+30
                if not self._plan_routes():return
            if self._energy_ready() and self._settled():
                self._stage('COLLECT','Перевозка оборудования на базы')
            return
        if self.phase=='COLLECT':
            loose=any(item.location in (L.WORKSITE_RACK,L.FILLED_QUEUE,L.DUMP_SITE,L.GROUND) and item.kind!='HOIST'
                      for u in p.units for item in u.simulation.logistics.inventory.mutable_items())
            if not loose and self._settled():
                # Routes are rechecked after the last flight/soil relaxation.
                if not self._plan_routes():self.retry_at=p.time+30;self._stage('ENERGY',self.reason);return
                if not self._energy_ready():self._stage('ENERGY',self.reason);return
                for u in p.units:
                    s=u.simulation;w=s.logistics
                    for d in s.drones:
                        hoist=w.inventory.items[f'hoist_{d.serial}']
                        # Personal ground equipment boards with its aircraft;
                        # handling is included in the powered packing phase.
                        if hasattr(d,'regional_mass'):continue
                        d.regional_mass=(d.config.empty_mass,d.config.hover_power)
                        tool=w.inventory.items[f'grabber_{d.serial}' if d.attachment=='GRABBER' else f'hook_{d.serial}']
                        w._tool_mass(d,tool.mass_kg+hoist.mass_kg)
                        w.inventory.move(hoist.id,L.DRONE_MOUNT,f'drone_{d.serial}',d.position,'INSTALLED')
                    if w.battery_platforms:w.battery_platforms.request('PACK')
                self._stage('PACK','Сборка солнечных и батарейных платформ');return
        if self.phase=='PACK':
            for u in p.units:
                w=u.simulation.logistics
                if w.base_pack_elapsed>=w.field_setup_seconds():
                    w.regional_folded=True;w.field_rect=None;w.field_ready=False;w.field_elapsed=0.;w.base_pack_elapsed=0.
            if all(getattr(u.simulation.logistics,'regional_folded',False) and
                   (u.simulation.logistics.battery_platforms is None or u.simulation.logistics.battery_platforms.stowed) for u in p.units):
                p.shared_site.invalidate_geometry();self.retry_at=0.;self._stage('DRIVE','Движение всей группы')
            return
        if self.phase=='DRIVE' and self.actor_index>=len(self.actors):
            for u in p.units:
                s=u.simulation
                for d in s.drones:
                    if not d.docked:
                        s._clear_slot(d);s._reserve_return(d);d.transition(DS.RETURN);d.regional_launch=True
                self.followers[u.id]=True
            self._stage('ARRIVE','Перелёт оставшихся дронов к базам');return
        if self.phase=='DRIVE' and self.actor_index<len(self.actors):
            id,kind=self.actors[self.actor_index];s=next(u.simulation for u in p.units if u.id==id)
            actor=s.base if kind=='base' else s.logistics.excavator
            if ((kind=='base' and actor.destination is None and actor.battery<actor.motion_power(s.drones,s.terrain)*2)
                    or (kind=='excavator' and actor.state==ES.LOW_BATTERY)):
                self.routes={};self._unfold_for_wait(f'{id}: остановка для пополнения энергии')
        if self.phase=='ARRIVE' and self._settled():self._finish()

    def _unfold_for_wait(self,reason):
        self.actor_index=0;self.retry_at=self.project.time+30
        for u in self.project.units:
            s=u.simulation;w=s.logistics;s.base.destination=None
            if w.battery_platforms:w.battery_platforms.request('DEPLOY')
            if w.excavator:
                w.excavator.destination=None
                if w.excavator.state==ES.MOVING_LOCAL:w.excavator.state=ES.IDLE
            if getattr(w,'regional_folded',False):
                w.regional_folded=False;w.field_ready=False;w.field_elapsed=0.;w.base_pack_elapsed=0.
                if s.base.config.panel_area>20:
                    try:w.plan_solar_field()
                    except ValueError:w.regional_folded=True
        self._stage('ENERGY',reason)
        self.project.shared_site.invalidate_geometry()

    def _finish(self):
        p=self.project;site=p.shared_site;land=p.landscape;old=site.core
        source=p.units[0].simulation.config
        preview=copy(land);preview.tiles=dict(land.tiles);preview.soil_tiles=dict(land.soil_tiles)
        preview.commit_window(self.old_origin,old.height,cell_size=old.cell,soil=old.soil)
        side=old.height.shape[0]*old.cell
        target=workspace_config(preview,source,self.new_origin,side)
        for key in GEOMETRY:setattr(target,key,deepcopy(getattr(self.target,key)))
        replacement=Terrain(target)
        shift=np.asarray(self.old_origin)-self.new_origin
        for u in p.units:
            s=u.simulation;probe=copy(s.base);point=s.base.position[:2]+shift
            probe.position=np.r_[point,replacement.sample(*point)]
            if not probe.safe_path(point,replacement):
                self.routes={};self._unfold_for_wait(f'{u.name}: площадка нового окна изменилась; требуется другой маршрут');return
        if land.channel_project is not None and old.remaining<=site.completion_volume:
            from .channel import ChannelProject
            channel=ChannelProject.from_data(land.channel_project);report=p.report()
            report.update(state='COMPLETED',georeference=source.georeference,
                          solar_generated_kwh=sum(u.simulation.base.generated for u in p.units)/3_600_000)
            channel.record(report);land.channel_project=channel.to_data()
        land.commit_window(self.old_origin,old.height,cell_size=old.cell,soil=old.soil)
        p.window_plans[window_key(self.old_origin)]={key:deepcopy(getattr(source,key)) for key in GEOMETRY}
        mass_offset=getattr(site,'archived_mass_error',0.)+old.mass_change
        materials=(site.material_errors().copy() if old.soil is not None else None)
        # No jobs carry resources at this point. History keeps its original frame.
        for u in p.units:
            s=u.simulation;w=s.logistics;c=s.config
            for job in [*(w.archived_jobs or []),*w.jobs]:
                if not hasattr(job,'region_origin_m'):job.region_origin_m=list(self.old_origin)
                if job.status=='QUEUED':w.cancel_job(job.id)
            for key in GEOMETRY:setattr(c,key,deepcopy(getattr(target,key)))
            c.initial_height=target.initial_height;c.reference_height=target.reference_height;c.initial_soil=target.initial_soil
            s.base.position[:2]+=shift;c.base.x,c.base.y=map(float,s.base.position[:2])
            for d in s.drones:
                d.position[:2]+=shift;d.destination=None;d.detour=None;d.trip=None
            for item in w.inventory.items.values():item.position=(item.position[0]+shift[0],item.position[1]+shift[1],item.position[2])
            if w.excavator:
                e=w.excavator;e.position[:2]+=shift;e.destination=None;e.target=None;e._drive_cache=None
                e.state=ES.DEPLOYING_SUPPORTS;e.elapsed=0.;e.regional_energy_required=0.
            w.rack=w.excavator.position.copy() if w.excavator else s.base.position.copy()
            u.section=p._section(s);u.committed=False;s._capability_key=None;s._latitude_position=None
            s._terrain_active=True;s._base_front_anchor=0.;s.relocation_pending=None
        replacement.config=p.units[0].simulation.config;replacement._site_owner=p.units[0].simulation;site.core=replacement
        for u in p.units:u.simulation.terrain.core=replacement
        site.owners=np.ones(replacement.height.shape,dtype=np.int16);site.archived_mass_error=mass_offset
        site.archived_material_errors=materials;site._mass_revision=-1;site.invalidate_geometry();site.partition()
        if land.channel_project is not None and self.section is not None:land.channel_project['active_section']=self.section
        for u in p.units:
            s=u.simulation
            if s.terrain.front is not None:
                s._base_front_anchor=s.terrain.front.start
                if s.terrain.front.sectors:
                    # A user-requested regional move has already paid for a
                    # safe arrival/deployment. Adopt that real position for
                    # its current sector, without another short fold/drive.
                    f=s.terrain.front;f.stationed=True;f.moving=False;f.revision=-1
            if s.logistics.battery_platforms:s.logistics.battery_platforms.request('DEPLOY')
        self._stage('DEPLOY','Развёртывание солнечных полей на новом участке');self.retry_at=0.
        self._deploy_fields()

    def _deploy_fields(self):
        p=self.project
        for u in p.units:
            s=u.simulation;w=s.logistics
            if not getattr(w,'regional_folded',False):continue
            if s.base.config.panel_area>20:
                try:w.plan_solar_field()
                except ValueError:
                    self.reason=f'{u.name}: нет места для солнечного поля; уменьшите панели или отмените переход';continue
            w.regional_folded=False;w.field_ready=False;w.field_elapsed=0.;w.base_pack_elapsed=0.
            p.shared_site.invalidate_geometry()

    def _release(self,status):
        p=self.project
        for u in p.units:
            s=u.simulation;w=s.logistics
            s.base.destination=None;s.regional_stage=None;s.landing_requested=self.previous_landing[u.id]
            if w.battery_platforms:w.battery_platforms.request('DEPLOY')
            for job in w.jobs:
                if job.kind=='PACK_FOR_RELOCATION' and job.status=='QUEUED':w.cancel_job(job.id)
            if w.excavator:
                e=w.excavator;e.destination=None;e._drive_cache=None;e.regional_energy_required=0.
                if e.state==ES.MOVING_LOCAL:e.state=ES.IDLE
            for d in s.drones:
                if d.grounded and hasattr(d,'regional_mass'):
                    d.config.empty_mass,d.config.hover_power=d.regional_mass;del d.regional_mass
                    hoist=w.inventory.items[f'hoist_{d.serial}']
                    w.inventory.move(hoist.id,L.GROUND,f'drone_{d.serial}',d.position,'STORED_READY')
                elif hasattr(d,'regional_mass'):w.regional_cleanup_pending=True
            if getattr(w,'solar_array',None) and w.solar_array.mode=='FOLD' and not getattr(w,'regional_folded',False):w.solar_array.request_deploy()
            if getattr(w,'regional_folded',False) or w.base_pack_elapsed:
                progress=1. if getattr(w,'regional_folded',False) else w.base_pack_elapsed/max(1.,w.field_setup_seconds())
                w.regional_folded=False;w.field_ready=False;w.field_elapsed=(1-progress)*w.field_setup_seconds();w.base_pack_elapsed=0.
                array=getattr(w,'solar_array',None)
                if array and not array.stowed:array.request_deploy()
                elif s.base.config.panel_area>20:
                    try:w.plan_solar_field()
                    except ValueError:
                        w.regional_folded=True;s.landing_requested=True;s.reason='Нет места для развёртывания панелей; измените оборудование или продолжите переезд'
                p.shared_site.invalidate_geometry()
            w.progress_at=s.time;s.event('Переезд группы завершён' if status=='COMPLETED' else 'Переезд отменён; техника остаётся на достигнутых позициях')
        p.window_history.append(dict(status=status,from_origin_m=self.old_origin,to_origin_m=self.new_origin,
            started_at=self.started_at,finished_at=p.time,complex_ids=[u.id for u in p.units],
            consumed_kwh=p.report()['consumed_kwh']-self.start_consumed_kwh))
        p.transition=None

    def cancel(self):self._release('CANCELLED')

    def report(self):
        return dict(phase=self.phase,reason=self.reason,from_origin_m=self.old_origin,to_origin_m=self.new_origin,
                    started_at=self.started_at,elapsed_seconds=self.project.time-self.started_at,
                    actors_done=self.actor_index,actor_count=len(self.actors))

    def validate(self):
        p=self.project
        if self.phase not in ('DRAIN','ENERGY','COLLECT','PACK','DRIVE','ARRIVE','DEPLOY') or p.shared_site is None:
            raise ValueError('Повреждено состояние регионального переезда')
        if not 0<=self.started_at<=p.time or not 0<=self.actor_index<=len(self.actors):raise ValueError('Повреждён прогресс переезда')
        if any(s.simulation.regional_stage!=self.phase for s in p.units):raise ValueError('Потеряна группа переезда')
        origin=self.new_origin if self.phase=='DEPLOY' else self.old_origin
        if any(list(u.simulation.config.georeference['origin_m'])!=origin for u in p.units):raise ValueError('Рабочее окно не соответствует фазе переезда')
        self.target.validate()
        if list(self.target.georeference['origin_m'])!=self.new_origin:raise ValueError('Потеряно целевое окно переезда')
        side=p.shared_site.core.height.shape[0]*p.shared_site.core.cell
        if (set(self.goals)!={u.id for u in p.units} or set(self.previous_landing)!=set(self.goals)
                or not math.isfinite(self.start_consumed_kwh) or self.start_consumed_kwh<0):raise ValueError('Повреждён состав переезда')
        for goal in self.goals.values():
            for point in goal.values():
                if point is not None and (len(point)!=2 or not np.all(np.isfinite(point)) or min(point)<0 or max(point)>side):
                    raise ValueError('Площадка переезда вне рабочего окна')
        for path in self.routes.values():
            if any(len(point)!=2 or not np.all(np.isfinite(point)) or min(point)<0 or max(point)>side for point in path):raise ValueError('Повреждён маршрут переезда')
