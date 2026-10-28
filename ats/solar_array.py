"""Physical solar cassettes and powered ground installation on disjoint lanes.

Scenario assumptions, not a manufacturer's design. Installers belong to carrier
platforms. A multi-source lane tree assigns disjoint work areas to installers;
each cassette follows its installer's route and returns before base movement.
"""
from dataclasses import dataclass,field
from copy import deepcopy
import heapq
import math
import numpy as np
from .site_ground import footprint_safe,marine_mask,paint_rect,equipment_mask


DEFAULTS=dict(section_area_m2=24.,section_width_m=6.,frame_mass_kg=24.,
    folded_length_m=2.,folded_width_m=1.,folded_height_m=.5,support_height_m=1.4,
    column_lane_m=3.,row_lane_m=4.,installer_area_m2=512.,installer_count=0,
    installer_mass_kg=25.,installer_payload_kg=200.,speed_m_s=.8,
    install_seconds=45.,fold_seconds=45.,mechanism_power_w=200.,
    rolling_resistance=.1,drive_efficiency=.8,max_grade_deg=8.,support_travel_m=.6)


def parameters(values=None):
    p=dict(DEFAULTS);p.update(values or {})
    if set(p)!=set(DEFAULTS) or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in p.values()):raise ValueError('Неверные параметры солнечных секций')
    if any(v<=0 for k,v in p.items() if k!='installer_count') or type(p['installer_count']) is not int or not 0<=p['installer_count']<=64:raise ValueError('Неверные размеры или количество монтажников')
    if p['column_lane_m']<2.5 or p['row_lane_m']<3 or not 0<p['drive_efficiency']<=1 or not 1<=p['max_grade_deg']<=20:raise ValueError('Недостаточные проезды или неверный привод солнечных секций')
    return p


def roof_counts(panels):
    """Retain up to 20 m² of whole panels on the existing integrated roof."""
    remaining=20.;counts=[]
    for panel in panels:
        count=min(panel.quantity,int((remaining+1e-9)/panel.area_m2))
        counts.append(count);remaining-=count*panel.area_m2
    return counts


def recipe(panels,settings):
    p=parameters(settings);groups=[]
    roof=roof_counts(panels)
    for panel,mounted in zip(panels,roof):
        per=max(1,int(p['section_area_m2']/panel.area_m2))
        left=panel.quantity-mounted
        while left:
            count=min(per,left);area=count*panel.area_m2
            groups.append((count,area,count*panel.mass_kg));left-=count
    crews=p['installer_count'] or max(1,min(64,math.ceil(sum(g[1] for g in groups)/p['installer_area_m2'])))
    frame_mass=sum(max(1.,a/p['section_area_m2'])*p['frame_mass_kg'] for _,a,_ in groups)
    volume=sum(max(1.,a/p['section_area_m2'])*p['folded_length_m']*p['folded_width_m']*p['folded_height_m'] for _,a,_ in groups)+sum(q.area_m2*n*.04 for q,n in zip(panels,roof))
    return groups,crews,frame_mass,volume


@dataclass
class SolarSection:
    id: str
    panel_ids: list
    area_m2: float
    frame_mass_kg: float
    panel_mass_kg: float = 0.
    rect: tuple = ()
    carrier: int = 0
    state: str = 'STOWED'
    elapsed: float = 0.
    route: list = field(default_factory=list)
    open_fraction: float = 0.
    work_start_fraction: float = 0.


@dataclass
class SolarInstaller:
    id: str
    home: np.ndarray
    position: np.ndarray
    sections: list = field(default_factory=list)
    active: int | None = None
    phase: str = 'IDLE'
    path: list = field(default_factory=list)
    cursor: int = 0
    mount: np.ndarray = field(default_factory=lambda:np.zeros(3))
    elapsed: float = 0.
    lift_origin: np.ndarray = field(default_factory=lambda:np.zeros(3))
    carrying: bool = False
    payload_origins: dict = field(default_factory=dict)
    lift_duration: float = 45.


def _segment_clear(t,a,b,blocked,p):
    length=float(np.linalg.norm(np.asarray(b)-a));steps=max(2,math.ceil(length/max(.5,t.cell))+1)
    for point in np.linspace(a,b,steps):
        if not footprint_safe(t,(point[0]-.65,point[1]-.65,1.3,1.3),blocked,
                              grade_deg=p['max_grade_deg'],relief_m=p['support_travel_m']):return False
    return True


class SolarArray:
    def __init__(self,world):
        self.settings=parameters(world.config.solar);world.config.solar=dict(self.settings)
        p=self.settings;groups,count,_,_=recipe(world.sim.base.config.equipment.panels,p)
        all_panels=[o for o in world.inventory.items.values() if o.kind=='SOLAR_PANEL'];panels=[];self.roof_panel_ids=[];offset=0;self.roof_area_m2=0.
        for panel,mounted in zip(world.sim.base.config.equipment.panels,roof_counts(world.sim.base.config.equipment.panels)):
            self.roof_panel_ids.extend(o.id for o in all_panels[offset:offset+mounted]);self.roof_area_m2+=mounted*panel.area_m2
            panels.extend(all_panels[offset+mounted:offset+panel.quantity]);offset+=panel.quantity
        self.sections=[];offset=0
        for i,(number,area,mass) in enumerate(groups):
            frame=max(1.,area/p['section_area_m2'])*p['frame_mass_kg']
            if mass+frame>p['installer_payload_kg']:raise ValueError('Солнечная кассета тяжелее грузоподъёмности монтажника')
            self.sections.append(SolarSection(f'solar_section_{i+1:04}',[o.id for o in panels[offset:offset+number]],area,frame,mass));offset+=number
        self.crews=[];self.mode='DEPLOY';self.issue='';self.reserved_rects=[]
        self.available_installers=count
        self.plan(world,count)

    def plan(self,world,count=None):
        from .base_layout import platform_polygons
        s=world.sim;t=s.terrain;p=self.settings;self._cycle_seconds=None
        count=len(self.crews) if count is None else count
        others=[s if u.id==getattr(s.terrain,'owner_id',None) else u.simulation for u in s.shared_site.project.units] if s.shared_site else [s]
        if not any(actor is s for actor in others):others.append(s)
        blocked=equipment_mask(t,others)|t.design_work|t.dump_mask|marine_mask(t)
        # The existing own layout is stowed when replanning. Its old field
        # reserve is removed only after all equipment has physically returned.
        if world.excavator:
            e=world.excavator;r=world.config.support_half_width_m+1.
            paint_rect(t,blocked,(e.position[0]-r,e.position[1]-r,2*r,2*r))
        scale=max(1.,max((q.area_m2 for q in self.sections),default=24.)/p['section_area_m2'])
        width=p['section_width_m']*math.sqrt(scale);height=p['section_area_m2']/p['section_width_m']*math.sqrt(scale)
        pitch_x=width+p['column_lane_m'];pitch_y=height+p['row_lane_m']
        side=t.height.shape[0]*t.cell;margin=max(.5,t.cell)
        nx=int((side-2*margin)/pitch_x);ny=int((side-2*margin)/pitch_y)
        tiles={};nodes={}
        for iy in range(ny):
            for ix in range(nx):
                x,y=margin+ix*pitch_x,margin+iy*pitch_y
                rect=(x,y,pitch_x,pitch_y)
                frame=(x+p['column_lane_m'],y+p['row_lane_m'],width,height)
                if footprint_safe(t,rect,blocked,grade_deg=p['max_grade_deg'],relief_m=max(p['support_travel_m'],max(pitch_x,pitch_y)*math.tan(math.radians(p['max_grade_deg'])))) and footprint_safe(t,frame,blocked,grade_deg=p['max_grade_deg'],relief_m=p['support_travel_m']):
                    index=iy*nx+ix;tiles[index]=rect;nodes[index]=np.array([x+p['column_lane_m']/2,y+p['row_lane_m']/2])
        if len(tiles)<len(self.sections):raise ValueError(f'BLOCKED_SOLAR_SITE: на суше помещается {len(tiles)} из {len(self.sections)} секций с проездами')
        carriers=list(platform_polygons(s.base.layout_spec,s.base.position)) if s.base.config.layout is not None else []
        homes=[];mounts=[];seeds=[];ids=list(nodes);positions=np.array([nodes[k] for k in ids]);used=[]
        connectors=np.zeros_like(blocked)
        for i in range(count):
            if carriers:
                poly=carriers[i%len(carriers)];lo,hi=poly.min(axis=0),poly.max(axis=0);centre=(lo+hi)/2
                choices=[point for gap in (max(2.5,1+2*t.cell),max(4.,1+4*t.cell)) for point in ((lo[0]-gap,centre[1]),(hi[0]+gap,centre[1]),(centre[0],lo[1]-gap),(centre[0],hi[1]+gap))]
            else:
                x,y=s.base.position[:2];radius=max(s.base.config.radius+2.,s.config.fleet.separation+1.5)
                choices=[point for r in (radius,radius+2*t.cell,radius+4.) for point in ((x+r,y),(x-r,y),(x,y+r),(x,y-r))]
            best=None
            for home in map(np.asarray,choices):
                if any(np.linalg.norm(home-other)<1.5 for other in used):continue
                order=np.argsort(np.sum((positions-home)**2,axis=1),kind='stable')[:16]
                for k in order:
                    index=ids[int(k)]
                    if _segment_clear(t,home,nodes[index],blocked|connectors,p):best=(home,index);break
                if best:break
            if best:
                home,index=best;used.append(home);homes.append(home)
                mount=centre if carriers else s.base.position[:2]
                mounts.append(np.r_[mount,t.sample(*mount)+s.base.config.roof_height])
                seeds.append((float(np.linalg.norm(nodes[index]-home)),index,len(homes)-1))
                for point in np.linspace(home,nodes[index],max(2,math.ceil(float(np.linalg.norm(nodes[index]-home))/max(.5,t.cell))+1)):
                    paint_rect(t,connectors,(point[0]-.75,point[1]-.75,1.5,1.5))
        if not homes:raise ValueError('BLOCKED_SOLAR_ACCESS: нет безопасного наземного прохода от носителя к солнечным секциям')
        parents={};roots={};dist={};queue=[]
        for cost,index,root in seeds:
            if cost<dist.get(index,float('inf')):dist[index]=cost;roots[index]=root;parents[index]=None;heapq.heappush(queue,(cost,index))
        selected=[];visited=set()
        while queue and len(selected)<len(self.sections):
            cost,index=heapq.heappop(queue)
            if index in visited:continue
            visited.add(index);selected.append(index);iy,ix=divmod(index,nx)
            for dy,dx in ((0,1),(1,0),(0,-1),(-1,0)):
                yy,xx=iy+dy,ix+dx;other=yy*nx+xx
                if not 0<=xx<nx or not 0<=yy<ny or other not in tiles:continue
                trial=cost+(pitch_x if dx else pitch_y)
                if trial<dist.get(other,float('inf')):
                    dist[other]=trial;parents[other]=index;roots[other]=roots[index];heapq.heappush(queue,(trial,other))
        if len(selected)<len(self.sections):raise ValueError('BLOCKED_SOLAR_ACCESS: нужное количество секций не связано безопасными проездами')
        self.crews=[SolarInstaller(f'solar_installer_{i+1:03}',np.r_[home,t.sample(*home)],mounts[i].copy(),mount=mounts[i]) for i,home in enumerate(homes)]
        self.reserved_rects=[]
        for i,(section,index) in enumerate(zip(self.sections,selected)):
            x,y,_,_=tiles[index];section.rect=(x+p['column_lane_m'],y+p['row_lane_m'],width,height)
            section.carrier=roots[index];path=[];k=index
            while k is not None:path.append(nodes[k].tolist());k=parents[k]
            path.reverse();home=self.crews[section.carrier].home[:2].tolist()
            centre=[section.rect[0]+width/2,section.rect[1]+height/2]
            section.route=[home,*path,centre];self.crews[section.carrier].sections.append(i)
            self.reserved_rects.append(tiles[index])
        for crew in self.crews:
            if not crew.sections:continue
            first=self.sections[crew.sections[0]].route
            a,b=np.array(first[0]),np.array(first[1]);lo=np.minimum(a,b)-1.;hi=np.maximum(a,b)+1.
            self.reserved_rects.append((*lo,*(hi-lo)))
        self.issue=''

    def add_inventory(self,world):
        from .inventory import Item,Location as L
        items=[Item(q.id,q.frame_mass_kg,L.BASE_STORAGE,holder_id='base',position=tuple(world.sim.base.position),kind='SOLAR_FRAME') for q in self.sections]
        items += [Item(c.id,self.settings['installer_mass_kg'],L.BASE_STORAGE,holder_id='base',position=tuple(world.sim.base.position),kind='SOLAR_INSTALLER') for c in self.crews]
        items += [Item(f'solar_installer_spare_{i:03}',self.settings['installer_mass_kg'],L.BASE_STORAGE,holder_id='base',position=tuple(world.sim.base.position),kind='SOLAR_INSTALLER') for i in range(self.available_installers-len(self.crews))]
        world.inventory.add_many(items)
        self.support_ids=[o.id for o in items]
        for q in self.sections:
            for id in [q.id,*q.panel_ids]:world.inventory.items[id].position=tuple(self.crews[q.carrier].mount)
        for crew in self.crews:world.inventory.items[crew.id].position=tuple(crew.mount)

    @property
    def installed_area(self):return self.roof_area_m2+sum(q.area_m2 for q in self.sections if q.state=='DEPLOYED' and q.id not in getattr(self,'invalid_sections',()))

    @property
    def stowed(self):return all(q.state=='STOWED' for q in self.sections) and all(c.active is None for c in self.crews)

    @property
    def ready(self):return all(q.state=='DEPLOYED' for q in self.sections) and all(c.active is None for c in self.crews)

    def duration(self):
        if getattr(self,'_cycle_seconds',None) is not None:return self._cycle_seconds
        p=self.settings
        self._cycle_seconds=max((sum(2*sum(math.ceil(float(np.linalg.norm(np.array(b)-a))/p['speed_m_s']) for a,b in zip(self.sections[i].route,self.sections[i].route[1:]))+max(p['install_seconds'],p['fold_seconds'])+90 for i in c.sections) for c in self.crews),default=0.)+1.
        return self._cycle_seconds

    def energy_budget(self):
        p=self.settings;mass=p['installer_mass_kg']+p['installer_payload_kg']
        watts=p['mechanism_power_w']+mass*9.80665*(p['rolling_resistance']+math.tan(math.radians(p['max_grade_deg'])))*p['speed_m_s']/p['drive_efficiency']
        return self.duration()*len(self.crews)*watts

    def crane_supported(self):
        """Loads on a carrier's handling mechanism still press on its tracks."""
        ids=set()
        for c in self.crews:
            if c.active is not None and c.phase in ('LOAD_OUT','LEAVE_BASE','STORE'):
                ids.add(c.id)
                if c.carrying:
                    q=self.sections[c.active];ids.update([q.id,*q.panel_ids])
        return ids

    def offboard_panel_mass(self):
        crane=self.crane_supported()
        return sum(q.panel_mass_kg for q in self.sections if q.state!='STOWED' and q.id not in crane)

    def validate(self,world):
        from .inventory import Location as L
        parameters(self.settings)
        if self.mode not in ('FOLD','DEPLOY') or not self.crews or len(self.crews)>self.available_installers:raise ValueError('Неверное состояние солнечной установки')
        panels=[*self.roof_panel_ids,*(id for q in self.sections for id in q.panel_ids)]
        actual=[o.id for o in world.inventory.items.values() if o.kind=='SOLAR_PANEL']
        if len(panels)!=len(set(panels)) or set(panels)!=set(actual):raise ValueError('Потеряна связь панелей с солнечными секциями')
        owners=[]
        for i,crew in enumerate(self.crews):
            if crew.id not in world.inventory.items or world.inventory.items[crew.id].kind!='SOLAR_INSTALLER':raise ValueError('Потерян монтажник солнечных секций')
            if not np.all(np.isfinite(crew.position)) or not np.all(np.isfinite(crew.mount)):raise ValueError('Неверное положение монтажника')
            if crew.active is not None and (crew.active not in crew.sections or not 0<=crew.cursor<=len(crew.path)):raise ValueError('Неверный рейс монтажника')
            for index in crew.sections:
                if not 0<=index<len(self.sections) or self.sections[index].carrier!=i:raise ValueError('Неверный носитель солнечной секции')
                owners.append(index)
        if sorted(owners)!=list(range(len(self.sections))):raise ValueError('Секция назначена нескольким монтажникам')
        for index,q in enumerate(self.sections):
            if q.state not in ('STOWED','IN_TRANSIT','ASSEMBLING','FOLDING','DEPLOYED') or not 0<=q.open_fraction<=1:raise ValueError('Неверная стадия солнечной секции')
            if q.id not in world.inventory.items or world.inventory.items[q.id].kind!='SOLAR_FRAME':raise ValueError('Потеряна опора солнечной секции')
            if len(q.rect)!=4 or len(q.route)<2 or not np.all(np.isfinite(q.rect)) or not np.all(np.isfinite(q.route)):raise ValueError('Повреждена геометрия солнечной секции')
            if q.state=='DEPLOYED' and any(world.inventory.items[id].location!=L.GROUND for id in [q.id,*q.panel_ids]):raise ValueError('Развёрнутая секция потеряла панели или опору')
            if q.state=='STOWED' and any(world.inventory.items[id].holder_id!='base' for id in [q.id,*q.panel_ids]):raise ValueError('Сложенная секция отсутствует на носителе')
            if q.state not in ('STOWED','DEPLOYED') and self.crews[q.carrier].active!=index:raise ValueError('Перевозимая секция потеряла монтажника')
        ids=getattr(self,'support_ids',[])
        if len(ids)!=len(set(ids)) or any(id not in world.inventory.items for id in ids):raise ValueError('Повреждён инвентарь солнечной установки')
        if set(ids)!={o.id for o in world.inventory.items.values() if o.kind in ('SOLAR_FRAME','SOLAR_INSTALLER')}:raise ValueError('Солнечные опоры или монтажники отсутствуют в едином инвентаре')

    def request_fold(self):
        self.mode='FOLD'
        for crew in self.crews:
            if crew.phase=='ASSEMBLE':
                q=self.sections[crew.active];crew.phase='FOLD';q.elapsed=0.;q.state='FOLDING';q.work_start_fraction=q.open_fraction

    def request_deploy(self):
        self.mode='DEPLOY'
        for crew in self.crews:
            if crew.phase=='FOLD':
                q=self.sections[crew.active];crew.phase='ASSEMBLE';q.elapsed=0.;q.work_start_fraction=q.open_fraction

    def _lift_start(self,world,crew,q,target):
        crew.elapsed=0.;crew.lift_origin=crew.position.copy()
        crew.payload_origins={id:np.asarray(world.inventory.items[id].position,dtype=float) for id in [q.id,*q.panel_ids]} if crew.carrying else {}
        lengths=[float(np.linalg.norm(target-crew.position))]+[float(np.linalg.norm(target-v)) for v in crew.payload_origins.values()]
        crew.lift_duration=max(45.,max(lengths)/self.settings['speed_m_s'])

    def _items(self,world,section,crew,location,state):
        from .inventory import Location as L
        for id in [section.id,*section.panel_ids]:
            holder='base' if location in (L.BASE_STORAGE,L.BASE_MOUNT) else section.id if location==L.GROUND else crew.id
            position=crew.position.copy()
            if location==L.GROUND and id!=section.id:position[2]+=self.settings['support_height_m']
            world.inventory.move(id,location,holder,position,state)

    def step(self,world):
        from .inventory import Location as L
        p=self.settings;s=world.sim;t=s.terrain
        if self.issue:return
        for crew in self.crews:
            if crew.active is None:
                wanted='DEPLOYED' if self.mode=='FOLD' else 'STOWED'
                choices=reversed(crew.sections) if self.mode=='FOLD' else iter(crew.sections)
                index=next((i for i in choices if self.sections[i].state==wanted),None)
                if index is None:continue
                q=self.sections[index];crew.active=index;crew.cursor=1;crew.path=deepcopy(q.route)
                crew.position=np.asarray(world.inventory.items[crew.id].position,dtype=float);crew.phase='LEAVE_BASE' if self.mode=='FOLD' else 'LOAD_OUT'
                crew.elapsed=0.;crew.lift_origin=crew.position.copy();crew.carrying=self.mode=='DEPLOY'
                self._lift_start(world,crew,q,crew.home)
                if crew.carrying:q.state='IN_TRANSIT'
                world.inventory.move(crew.id,L.IN_TRANSFER,crew.id,crew.position,'SOLAR_SERVICE')
            q=self.sections[crew.active]
            if self.mode=='FOLD' and crew.phase=='LOAD_OUT':
                crew.phase='STORE';self._lift_start(world,crew,q,crew.mount)
            if crew.phase in ('LOAD_OUT','LEAVE_BASE','STORE'):
                target=crew.mount if crew.phase=='STORE' else crew.home
                joules=sum(world.inventory.items[id].mass_kg*9.80665*(max(0.,target[2]-origin[2])+p['rolling_resistance']*float(np.linalg.norm(target[:2]-origin[:2])))/p['drive_efficiency'] for id,origin in crew.payload_origins.items())
                joules+=p['installer_mass_kg']*9.80665*(max(0.,target[2]-crew.lift_origin[2])+p['rolling_resistance']*float(np.linalg.norm(target[:2]-crew.lift_origin[:2])))/p['drive_efficiency']
                if not world.use_station(p['mechanism_power_w']+joules/crew.lift_duration):continue
                crew.elapsed+=1.;fraction=min(1.,crew.elapsed/crew.lift_duration)
                crew.position=crew.lift_origin+(target-crew.lift_origin)*fraction
                world.inventory.items[crew.id].position=tuple(crew.position)
                for id,origin in crew.payload_origins.items():
                    world.inventory.move(id,L.IN_TRANSFER,crew.id,origin+(target-origin)*fraction,'SOLAR_HANDLING')
                if crew.elapsed>=crew.lift_duration:
                    if crew.phase=='STORE':
                        if crew.carrying:q.state='STOWED';self._items(world,q,crew,L.BASE_MOUNT,'INSTALLED')
                        world.inventory.move(crew.id,L.BASE_STORAGE,'base',crew.mount,'STORED_READY')
                        crew.active=None;crew.phase='IDLE';crew.carrying=False
                    else:crew.phase='OUT' if crew.carrying else 'FETCH'
                continue
            # A cancellation never teleports an outgoing loaded cassette home.
            if self.mode=='FOLD' and crew.phase=='OUT':
                crew.path=[crew.position[:2].tolist(),*reversed(crew.path[:crew.cursor])];crew.cursor=1;crew.phase='RETURN_LOAD'
            if self.mode=='DEPLOY' and crew.phase=='FETCH':
                crew.path=[crew.position[:2].tolist(),*reversed(crew.path[:crew.cursor])];crew.cursor=1;crew.phase='BACK'
            if crew.phase in ('OUT','FETCH','BACK','RETURN_LOAD'):
                target=np.asarray(crew.path[crew.cursor]);delta=target-crew.position[:2];distance=float(np.linalg.norm(delta));travel=min(p['speed_m_s'],distance)
                next_xy=crew.position[:2]+delta*travel/max(distance,1e-12)
                if not footprint_safe(t,(next_xy[0]-.55,next_xy[1]-.55,1.1,1.1),t.design_work|t.dump_mask,grade_deg=p['max_grade_deg'],relief_m=p['support_travel_m']):
                    self.issue='SOLAR_ROUTE_CHANGED: монтажник остановлен с сохранённым грузом';continue
                load=sum(world.inventory.items[id].mass_kg for id in [q.id,*q.panel_ids]) if crew.phase in ('OUT','RETURN_LOAD') else 0.
                rise=max(0.,t.sample(*next_xy)-crew.position[2])
                energy=p['mechanism_power_w']+(p['installer_mass_kg']+load)*9.80665*(p['rolling_resistance']*travel+rise)/p['drive_efficiency']
                if not world.use_station(energy):continue
                crew.position=np.r_[next_xy,t.sample(*next_xy)]
                world.inventory.items[crew.id].position=tuple(crew.position)
                if load:self._items(world,q,crew,L.IN_TRANSFER,'SOLAR_DELIVERY')
                if distance<=p['speed_m_s']+1e-9:
                    crew.cursor+=1
                    if crew.cursor>=len(crew.path):
                        if crew.phase=='OUT':crew.phase='ASSEMBLE';q.state='ASSEMBLING';q.elapsed=0.;q.work_start_fraction=q.open_fraction
                        elif crew.phase=='FETCH':crew.phase='FOLD';q.state='FOLDING';q.elapsed=0.;q.work_start_fraction=q.open_fraction
                        else:
                            crew.carrying=crew.phase=='RETURN_LOAD';crew.phase='STORE';self._lift_start(world,crew,q,crew.mount)
            elif crew.phase in ('ASSEMBLE','FOLD'):
                folding=crew.phase=='FOLD' or self.mode=='FOLD'
                if not folding and not footprint_safe(t,q.rect,t.design_work|t.dump_mask,grade_deg=p['max_grade_deg'],relief_m=p['support_travel_m']):
                    self.issue='SOLAR_ROUTE_CHANGED: площадка секции изменилась; груз сохранён';continue
                if not world.use_station(p['mechanism_power_w']):continue
                q.elapsed+=1.
                fraction=min(1.,q.elapsed/(p['fold_seconds'] if folding else p['install_seconds']))
                q.open_fraction=q.work_start_fraction*(1-fraction) if folding else q.work_start_fraction+(1-q.work_start_fraction)*fraction
                for id in q.panel_ids:
                    item=world.inventory.items[id];position=crew.position.copy()
                    position[2]+=p['support_height_m']*q.open_fraction
                    item.position=tuple(position);item.location=L.IN_TRANSFER;item.holder_id=crew.id
                if q.elapsed>=(p['fold_seconds'] if folding else p['install_seconds']):
                    crew.path=[crew.position[:2].tolist(),*reversed(q.route[:-1])];crew.cursor=1
                    if folding:q.state='IN_TRANSIT';crew.phase='RETURN_LOAD';crew.carrying=True;self._items(world,q,crew,L.IN_TRANSFER,'SOLAR_RETURN')
                    else:q.state='DEPLOYED';crew.phase='BACK';self._items(world,q,crew,L.GROUND,'SOLAR_DEPLOYED')
        count=max(1,len(self.sections));duration=max(1.,self.duration())
        world.field_ready=self.ready;world.field_elapsed=duration*sum(q.state=='DEPLOYED' for q in self.sections)/count
        world.base_pack_elapsed=(duration*(sum(q.state=='STOWED' for q in self.sections)/count) if self.mode=='FOLD' else 0.)
        if self.mode=='FOLD' and self.stowed:world.base_pack_elapsed=duration
        elif self.mode=='DEPLOY' and self.ready:world.field_elapsed=duration

    def report(self):
        return dict(section_count=len(self.sections),installed_area_m2=self.installed_area,
            roof_area_m2=self.roof_area_m2,
            deployed_sections=sum(q.state=='DEPLOYED' for q in self.sections),installer_count=len(self.crews),
            mode=self.mode,issue=self.issue,reserved_area_m2=sum(r[2]*r[3] for r in self.reserved_rects),
            setup_seconds_estimate=self.duration(),settings=dict(self.settings))
