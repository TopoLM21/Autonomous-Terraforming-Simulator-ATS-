"""Diesel earthmoving on the existing height/material field.

Model equipment (not manufacturer specifications): 22 t excavator, 18 t payload
truck, conservative circular swept footprints, a retained 12% access ramp.
Planning is event-driven; fuel, staff time and physical motion integrate at 1 s.
"""
from dataclasses import dataclass,field,asdict
import heapq
import math
import numpy as np
from .soil import SoilLoad,cargo_volume


@dataclass
class ConventionalConfig:
    enabled: bool = True
    excavator_count: int = 1
    truck_count: int = 1
    base_x: float = 10.
    base_y: float = 10.
    work_rect: tuple = (24.,24.,24.,16.)
    shifts_per_day: int = 1
    shift_start_hour: float = 6.
    shift_span_hours: float = 9.
    lunch_after_hours: float = 4.
    lunch_hours: float = 1.
    weekly_rest_day: int = 4
    start_weekday: int = 0
    excavator_wage_egp_month: float = 16625.
    driver_wage_egp_month: float = 16218.
    egp_per_usd: float = 53.55
    diesel_egp_l: float = 20.5
    food_egp_unit: float = 100.
    water_egp_l: float = 5.
    fuel_stock_l: float = 5000.
    water_stock_l: float = 1000.
    food_stock_units: float = 100.
    water_l_per_person_hour: float = 1.
    food_units_per_person_shift: float = 1.
    excavator_purchase_usd: float = 120000.
    truck_purchase_usd: float = 80000.
    base_purchase_usd: float = 25000.
    maintenance_usd_per_engine_hour: float = 8.
    bucket_m3: float = 1.2
    bucket_cycle_s: float = 30.
    excavator_reach_m: float = 9.
    excavator_dig_depth_m: float = 6.
    truck_payload_kg: float = 18000.
    truck_body_m3: float = 10.
    excavator_speed_m_s: float = .7
    truck_speed_m_s: float = 4.
    excavator_tank_l: float = 300.
    truck_tank_l: float = 250.
    excavator_work_l_h: float = 18.
    truck_drive_l_h: float = 16.
    idle_l_h: float = 2.
    refuel_l_min: float = 30.
    refuel_slots: int = 1
    fuel_reserve_fraction: float = .15
    ramp_gradient: float = .12
    ramp_width_m: float = 9.
    route_step_m: float = 2.
    max_route_nodes: int = 5000

    def validate(self):
        numbers=[v for k,v in vars(self).items() if isinstance(v,(float,int)) and not isinstance(v,bool)]
        if not all(math.isfinite(v) and v>=0 for v in numbers):raise ValueError('Обычная техника: параметры должны быть конечными и неотрицательными')
        if self.excavator_count!=1 or not isinstance(self.truck_count,int) or not 1<=self.truck_count<=16:
            raise ValueError('Первый этап: один экскаватор и от 1 до 16 самосвалов')
        if not isinstance(self.refuel_slots,int) or not 1<=self.refuel_slots<=4:raise ValueError('Требуется от 1 до 4 заправочных постов')
        if self.shifts_per_day not in (1,2,3) or not 0<=self.shift_start_hour<24 or not 0<=self.start_weekday<7 or not 0<=self.weekly_rest_day<7:
            raise ValueError('Неверный календарь смен')
        if (not 1<=self.lunch_hours or not 0<self.lunch_after_hours<=5 or
            not 0<self.shift_span_hours<=min(10.,24/self.shifts_per_day) or
            not 0<self.shift_span_hours-self.lunch_hours<=8 or
            self.lunch_after_hours+self.lunch_hours>=self.shift_span_hours or
            self.shift_span_hours-self.lunch_after_hours-self.lunch_hours>5):
            raise ValueError('Смена: до 8 часов труда, обед не менее часа, не более 5 часов подряд; смены не пересекаются')
        if len(self.work_rect)!=4 or not all(math.isfinite(v) for v in self.work_rect) or min(self.work_rect[2:])<6:
            raise ValueError('Нужен рабочий участок размером не менее 6 × 6 м')
        for name in ('egp_per_usd','bucket_m3','bucket_cycle_s','excavator_reach_m','excavator_dig_depth_m',
                     'truck_payload_kg','truck_body_m3','excavator_speed_m_s','truck_speed_m_s',
                     'excavator_tank_l','truck_tank_l','refuel_l_min','route_step_m'):
            if getattr(self,name)<=0:raise ValueError('Параметр должен быть положительным: '+name)
        if not .02<=self.ramp_gradient<=.2 or not math.hypot(7.,2.8)+1.<=self.ramp_width_m or not 0<self.fuel_reserve_fraction<.5 or not 100<=self.max_route_nodes<=20000:
            raise ValueError('Неверные ограничения рампы, резерва или планировщика')

    @classmethod
    def for_simulation(cls,sim):
        t=getattr(sim.terrain,'core',sim.terrain);side=t.cell*t.height.shape[0]
        ids=np.flatnonzero(t.dig_mask & (t.height>t.target+1e-6))
        if len(ids):
            x=float(t.x.ravel()[ids].mean());y=float(t.y.ravel()[ids].mean())
        else:x=y=side/2
        w=min(24.,side/2);h=min(16.,side/2)
        rect=(max(1.,min(side-w-1,x-w/2)),max(1.,min(side-h-1,y-h/2)),w,h)
        from datetime import date,timedelta
        weekday=(date(2025,1,1)+timedelta(days=sim.config.base.start_day_of_year-1)).weekday()
        return cls(base_x=max(5.,min(side-5.,rect[0]-9)),base_y=max(5.,min(side-5.,rect[1]-9)),work_rect=rect,start_weekday=weekday)


@dataclass
class GroundMachine:
    id: str
    kind: str
    position: np.ndarray
    home: np.ndarray
    state: str = 'PARKED'
    reason: str = ''
    fuel_l: float = 0.
    cargo_kg: float = 0.
    soil: object = None
    path: list = field(default_factory=list)
    route_bounds: tuple | None = None
    route_token: tuple | None = None
    destination: tuple | None = None
    after_drive: str = 'PARKED'
    timer: float = 0.
    target: tuple | None = None
    excavated_m3: float = 0.
    delivered_m3: float = 0.
    engine_seconds: float = 0.
    working_seconds: float = 0.
    waiting_seconds: float = 0.
    distance_m: float = 0.
    fuel_used_l: float = 0.
    crew_key: tuple | None = None
    load_started: float = 0.
    next_plan_at: float = 0.
    blocked_revision: int = -1
    emergency_work_seconds: float = 0.
    budget_cache: dict | None = None
    retry_route_at: float = 0.

    @property
    def dimensions_m(self):return (6.,3.,3.) if self.kind=='excavator' else (7.,2.8,3.2)

    @property
    def radius(self):return math.hypot(*self.dimensions_m[:2])/2


class ConventionalFleet:
    def __init__(self,sim,config):
        config.validate();self.sim=sim;self.config=config
        self.terrain=getattr(sim.terrain,'core',sim.terrain);t=self.terrain;c=config
        side=t.cell*t.height.shape[0];x,y,w,h=c.work_rect
        if x<0 or y<0 or x+w>side or y+h>side:raise ValueError('Рабочий участок выходит за загруженное окно')
        self.work_mask=(t.x>=x)&(t.x<x+w)&(t.y>=y)&(t.y<y+h)
        if not np.any(self.work_mask&t.dig_mask):raise ValueError('На рабочем участке нет проектной выемки; сначала нанесите план')
        self.mask=self.work_mask
        self.base_position=np.array([c.base_x,c.base_y,t.sample(c.base_x,c.base_y)])
        self.machines=[];self.enabled=bool(c.enabled);self.reason=''
        self.fuel_stock_l=c.fuel_stock_l;self.water_stock_l=c.water_stock_l;self.food_stock_units=c.food_stock_units
        self.fuel_imported_l=self.water_imported_l=self.food_imported_units=0.
        self.fuel_used_l=self.water_used_l=self.food_used_units=0.
        self.excavated_mass_kg=self.excavated_m3=self.delivered_m3=self.delivered_mass_kg=0.
        self.extracted_materials=np.zeros(len(t.soil.profile.materials)) if t.soil else None
        self.delivered_materials=np.zeros_like(self.extracted_materials) if t.soil else None
        self.payroll_usd=self.maintenance_usd=0.;self.route_calculations=0
        self._marine=None;self._marine_stamp=None;self._crew_paid=set();self._work_complete=False
        self._last_time=None;self._last_report_time=None
        if not self.passable(self.base_position[:2],4.,ignore_vehicles=True):raise ValueError('Топливная база не помещается на безопасной суше')
        if np.any((self.work_mask|t.work_mask)[t.patch(c.base_x,c.base_y,4.)]):raise ValueError('Топливная база пересекает проектную выемку')
        self.ramp=self._make_ramp()
        for i in range(c.excavator_count+c.truck_count):
            kind='excavator' if i<c.excavator_count else 'truck';serial=i+1 if kind=='excavator' else i-c.excavator_count+1
            point=self._parking_point(math.hypot(6.,3.)/2 if kind=='excavator' else math.hypot(7.,2.8)/2)
            if point is None:raise ValueError('Нет места для стоянки всего парка; расширьте площадку базы')
            p=np.array([*point,t.sample(*point)])
            self.machines.append(GroundMachine(f'{kind}_{serial}',kind,p,p.copy()))
        self.excavator=self.machines[0]

    def _make_ramp(self):
        t=self.terrain;c=self.config;x,y,w,h=c.work_rect;along_x=w>=h
        start=np.array([x,y+h/2]) if along_x else np.array([x+w/2,y])
        end=np.array([x+w,y+h/2]) if along_x else np.array([x+w/2,y+h])
        if np.linalg.norm(self.base_position[:2]-end)<np.linalg.norm(self.base_position[:2]-start):start,end=end,start
        delta=end-start;length=float(np.linalg.norm(delta));unit=delta/length
        along=(t.x-start[0])*unit[0]+(t.y-start[1])*unit[1]
        lateral=np.abs((t.x-start[0])*unit[1]-(t.y-start[1])*unit[0])
        mask=self.work_mask&(lateral<=c.ramp_width_m/2)
        entry_height=t.sample(*start)
        surface=entry_height-np.clip(along,0,length)*c.ramp_gradient
        self.ramp_mask=mask;self.ramp_surface=surface
        return dict(entry_m=start.tolist(),end_m=end.tolist(),width_m=c.ramp_width_m,max_gradient=c.ramp_gradient)

    def _parking_point(self,radius):
        x,y=self.base_position[:2]
        for ring in range(1,9):
            # Circumscribed truck bodies plus a full truck-width access lane.
            step=16.
            offsets=sorted(((dx,dy) for dx in range(-ring,ring+1) for dy in range(-ring,ring+1)
                            if max(abs(dx),abs(dy))==ring),key=lambda d:(d[0]*d[0]+d[1]*d[1],d))
            for dx,dy in offsets:
                p=(x+dx*step,y+dy*step)
                if not self.passable(p,radius):continue
                patch=self.terrain.patch(*p,radius)
                if np.any((self.work_mask|self.terrain.work_mask|self.terrain.dump_mask)[patch]):continue
                return p
        return None

    def _other_rectangles(self,bounds):
        s=self.sim;t=self.terrain;site=getattr(s,'shared_site',None)
        if site is not None:return [row[1:5] for row in site.ground_token('conventional',bounds)]
        if getattr(s.config,'conventional_only',False):return []
        if s.base.config.layout is not None:
            from .base_layout import layout_footprints
            from .self_solar import deployment_rectangles
            from .battery_platforms import deployment_rectangles as battery_rectangles
            return [*layout_footprints(s.base.layout_spec,s.base.position),
                    *deployment_rectangles(s.base.layout_spec,s.base.position),
                    *battery_rectangles(s.base.layout_spec,s.base.position)]
        r=s.base.config.radius
        return [(s.base.position[0]-r,s.base.position[1]-r,2*r,2*r)]

    def passable(self,point,radius,machine=None,*,ignore_vehicles=False):
        t=self.terrain;x,y=map(float,point[:2]);side=t.cell*t.height.shape[0]
        if min(x,y)<radius+t.cell/2 or max(x,y)>side-radius-t.cell/2:return False
        patch=t.patch(x,y,radius+t.cell/2);xx=t.x[patch]-x;yy=t.y[patch]-y
        footprint=xx*xx+yy*yy<=(radius+t.cell*.71)**2
        if not np.any(footprint):return False
        stamp=(id(t),id(t.protected),id(t.reference),id(t.config.plan))
        if self._marine_stamp!=stamp:
            from .site_ground import marine_mask
            self._marine=marine_mask(t);self._marine_stamp=stamp
        if np.any(self._marine[patch]&footprint):return False
        heights=t.height[patch];z=t.sample(x,y);distance=np.maximum(np.hypot(xx,yy),t.cell/2)
        # Full circumscribed support footprint, including every contained fine cell.
        if np.any((np.abs(heights-z)>.2*distance+.12)&footprint):return False
        if heights.shape[0]>1 and np.any((np.abs(np.diff(heights,axis=0))>.2*t.cell+.12)&footprint[:-1]&footprint[1:]):return False
        if heights.shape[1]>1 and np.any((np.abs(np.diff(heights,axis=1))>.2*t.cell+.12)&footprint[:,:-1]&footprint[:,1:]):return False
        for bx,by,w,h in self._other_rectangles((x-radius,y-radius,x+radius,y+radius)):
            nearest=np.array([np.clip(x,bx,bx+w),np.clip(y,by,by+h)])
            if np.linalg.norm(nearest-[x,y])<radius+.2:return False
        if not ignore_vehicles:
            for other in self.machines:
                if other is not machine and np.linalg.norm(other.position[:2]-[x,y])<other.radius+radius+.25:return False
        # Every autonomous unit shares this road network. Inspect actual
        # grounded aircraft and committed low-altitude work on the whole site.
        from .conventional_integration import simulations_for
        for owner in simulations_for(self.sim):
            for drone in owner.drones:
                if drone.grounded and np.linalg.norm(drone.position[:2]-[x,y])<radius+max(.5,owner.config.fleet.separation/2):return False
                if drone.trip is not None:
                    for point in (drone.trip.dig,drone.trip.dump):
                        if np.linalg.norm(np.asarray(point)[:2]-[x,y])<radius+max(drone.config.grab_radius or .5,owner.config.fleet.separation):return False
            for e in (owner.logistics.excavators if owner.logistics else ()):
                if np.linalg.norm(e.position[:2]-[x,y])<e.config.support_half_width_m+radius+.3:return False
        return True

    def segment_safe(self,a,b,radius,machine=None):
        length=float(np.linalg.norm(np.asarray(b)[:2]-np.asarray(a)[:2]));step=min(1.,self.terrain.cell)
        points=np.linspace(np.asarray(a)[:2],np.asarray(b)[:2],max(2,int(math.ceil(length/step))+1))
        if any(not self.passable(p,radius,machine) for p in points):return False
        heights=self.terrain.sample_many(points);runs=np.linalg.norm(np.diff(points,axis=0),axis=1)
        return bool(np.all(np.abs(np.diff(heights))<=.2*runs+.12))

    def route(self,machine,destination):
        """Bounded deterministic A*; routes are computed only on assignment/change."""
        self.route_calculations+=1;start=machine.position[:2].copy();goal=np.asarray(destination)[:2]
        if not self.passable(goal,machine.radius,machine):return None
        if self.segment_safe(start,goal,machine.radius,machine):return [tuple(goal)]
        step=max(self.config.route_step_m,self.terrain.cell)
        side=self.terrain.cell*self.terrain.height.shape[0]
        origin=np.floor(np.minimum(start,goal)/step)*step-32
        lo=np.maximum(machine.radius+self.terrain.cell/2,origin)
        hi=np.minimum(side-machine.radius-self.terrain.cell/2,np.maximum(start,goal)+32)
        def key(p):return tuple(np.rint((np.asarray(p)-lo)/step).astype(int))
        def point(k):return lo+np.asarray(k)*step
        first=key(start);last=key(goal);queue=[(float(np.linalg.norm(start-goal)),0.,first)]
        costs={first:0.};parents={};checked={};seen=set();limit=min(self.config.max_route_nodes,1200)
        def clear(k):
            if k not in checked:
                p=point(k);checked[k]=bool(np.all(p>=lo)&np.all(p<=hi)) and self.passable(p,machine.radius,machine)
            return checked[k]
        while queue and len(seen)<limit:
            _,cost,current=heapq.heappop(queue)
            if current in seen:continue
            seen.add(current);p=point(current)
            if current==last and self.segment_safe(p,goal,machine.radius,machine):
                nodes=[tuple(goal)];node=current
                while node!=first:nodes.append(tuple(point(node)));node=parents[node]
                nodes.reverse()
                if nodes and all(self.segment_safe(a,b,machine.radius,machine) for a,b in zip([start,*nodes[:-1]],nodes)):return nodes
                return None
            for dx,dy in ((1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)):
                nxt=(current[0]+dx,current[1]+dy)
                if nxt in seen or not clear(nxt):continue
                if dx and dy and (not clear((current[0]+dx,current[1])) or not clear((current[0],current[1]+dy))):continue
                q=point(nxt)
                if abs(self.terrain.sample(*q)-self.terrain.sample(*p))>.2*step*math.hypot(dx,dy)+.12:continue
                # Adjacent grid nodes still require swept-footprint validation.
                tentative=cost+float(np.linalg.norm(q-p))
                if tentative+1e-9>=costs.get(nxt,math.inf):continue
                costs[nxt]=tentative;parents[nxt]=current
                heapq.heappush(queue,(tentative+float(np.linalg.norm(q-goal)),tentative,nxt))
        return None

    def _assign_route(self,m,destination,after):
        path=self.route(m,destination)
        if path is None:m.reason='Нет безопасного наземного маршрута';m.retry_route_at=self.sim.time+30;return False
        if after!='PARKED':
            shift=self.shift();outbound=self._path_length(m,path)/self._speed(m)
            old=m.position.copy();m.position[:2]=destination[:2];m.position[2]=self.terrain.sample(*destination[:2])
            home_path=self.route(m,m.home[:2]);back=self._path_length(m,home_path)/self._speed(m) if home_path is not None else math.inf
            m.position[:]=old
            duration=outbound+back+max(60.,self.config.bucket_cycle_s)+45.
            fuel=duration*max(self._rate(m,True),self.config.truck_drive_l_h)/3600+self._tank(m)*self.config.fuel_reserve_fraction
            if not shift['active'] or shift['seconds_left']<duration:
                m.reason='Недостаточно времени на рейс и возвращение до обеда / конца смены';m.retry_route_at=self.sim.time+30;return False
            if m.fuel_l<fuel:m.reason='Недостаточно топлива на рейс, возврат и резерв';m.retry_route_at=self.sim.time+30;return False
        m.budget_cache=None;m.retry_route_at=0.
        m.path=path;m.destination=tuple(destination[:2]);m.after_drive=after;m.state='DRIVING'
        points=np.array([m.position[:2],*path]);lo=points.min(axis=0)-m.radius;hi=points.max(axis=0)+m.radius
        m.route_bounds=tuple(np.r_[lo,hi]);m.route_token=self.terrain.route_token(m.route_bounds);m.reason=''
        return True

    def _path_length(self,m,path):
        points=np.array([m.position[:2],*path]);return float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())

    def _tank(self,m):return self.config.excavator_tank_l if m.kind=='excavator' else self.config.truck_tank_l
    def _speed(self,m):return self.config.excavator_speed_m_s if m.kind=='excavator' else self.config.truck_speed_m_s
    def _rate(self,m,working=False):
        if m.kind=='excavator':return self.config.excavator_work_l_h if working or m.state=='DRIVING' else self.config.idle_l_h
        return self.config.truck_drive_l_h if m.state=='DRIVING' else self.config.idle_l_h

    def _burn(self,m,working=False):
        needed=self._rate(m,working)/3600
        if m.fuel_l+1e-12<needed:m.reason='Нет топлива: требуется подвоз';return False
        m.fuel_l=max(0.,m.fuel_l-needed);m.fuel_used_l+=needed;self.fuel_used_l+=needed;m.engine_seconds+=1
        self.maintenance_usd+=self.config.maintenance_usd_per_engine_hour/3600
        return True

    def _move(self,m):
        if not m.path:m.state=m.after_drive;return
        if self.sim.time<m.retry_route_at:m.waiting_seconds+=1;return
        if m.route_token!=self.terrain.route_token(m.route_bounds):
            path=self.route(m,m.destination)
            if path is None:m.reason='Маршрут изменён грунтом или препятствием; груз сохранён';m.retry_route_at=self.sim.time+30;m.waiting_seconds+=1;return
            m.path=path;m.route_token=self.terrain.route_token(m.route_bounds)
        destination=np.asarray(m.path[0]);delta=destination-m.position[:2];length=float(np.linalg.norm(delta))
        next_point=m.position[:2]+delta*min(1.,self._speed(m)/max(length,1e-9))
        if not self.segment_safe(m.position,next_point,m.radius,m):
            m.reason='Проезд занят или уклон изменился';m.waiting_seconds+=1
            if self.sim.time>=m.next_plan_at:
                m.next_plan_at=self.sim.time+30;path=self.route(m,m.destination)
                if path is not None:m.path=path
            return
        if not self._burn(m):return
        m.position[:2]=next_point;m.position[2]=self.terrain.sample(*next_point);m.distance_m+=min(length,self._speed(m))
        if length<=self._speed(m)+1e-9:m.path.pop(0)
        if not m.path:m.state=m.after_drive;m.reason=''

    def shift(self,when=None):
        c=self.config;seconds=self.sim.config.start_hour*3600+(self.sim.time if when is None else when)
        day=math.floor(seconds/86400);period=86400/c.shifts_per_day
        for startday in (day,day-1):
            if (c.start_weekday+startday)%7==c.weekly_rest_day:continue
            for crew in range(c.shifts_per_day):
                start=startday*86400+c.shift_start_hour*3600+crew*period
                elapsed=seconds-start;span=c.shift_span_hours*3600
                if not 0<=elapsed<span:continue
                lunch=c.lunch_after_hours*3600;end=(c.lunch_after_hours+c.lunch_hours)*3600
                if lunch<=elapsed<end:return dict(active=False,crew=crew+1,reason='Обеденный перерыв',seconds_left=0.,key=(startday,crew))
                boundary=lunch if elapsed<lunch else span
                return dict(active=True,crew=crew+1,reason='Рабочая смена',seconds_left=boundary-elapsed,key=(startday,crew))
        return dict(active=False,crew=None,reason='Междусменный или еженедельный отдых',seconds_left=0.,key=None)

    def _remaining(self):
        t=self.terrain;floor=np.where(self.ramp_mask,np.maximum(t.target,self.ramp_surface),t.target)
        return np.where(self.work_mask&t.dig_mask,np.maximum(t.height-floor,0.),0.)

    def _return_budget(self,m,extra_s=0.):
        key=tuple(m.position[:2]);cache=m.budget_cache
        if (cache is not None and cache['position']==key and
            cache['token']==self.terrain.route_token(cache['bounds'])):path=cache['path']
        else:
            path=self.route(m,m.home[:2])
            if path is None:return None
            points=np.array([m.position[:2],*path]);bounds=tuple(np.r_[points.min(axis=0)-m.radius,points.max(axis=0)+m.radius])
            m.budget_cache=dict(position=key,path=path,bounds=bounds,token=self.terrain.route_token(bounds))
        seconds=self._path_length(m,path)/self._speed(m)+extra_s+15
        rate=max(self.config.excavator_work_l_h,self.config.idle_l_h) if m.kind=='excavator' else self.config.truck_drive_l_h
        fuel=seconds*rate/3600+self._tank(m)*self.config.fuel_reserve_fraction
        return seconds,fuel,path

    def _go_home(self,m):
        if np.linalg.norm(m.position[:2]-m.home[:2])<.1:m.state='PARKED';m.path=[];m.reason='';return True
        return self._assign_route(m,m.home[:2],'PARKED')

    def _available_target(self,e):
        t=self.terrain;remaining=self._remaining();ids=np.flatnonzero(remaining>1e-6)
        if not len(ids):return None
        distance=(t.x.ravel()[ids]-e.position[0])**2+(t.y.ravel()[ids]-e.position[1])**2
        valid=(distance<=self.config.excavator_reach_m**2)&(distance>=(e.radius+1.)**2)
        ids=ids[valid]
        if not len(ids):return None
        # Keep the present supporting footprint and a retained ramp uncut.
        ids=ids[np.argsort((t.x.ravel()[ids]-e.position[0])**2+(t.y.ravel()[ids]-e.position[1])**2,kind='stable')]
        for index in ids[:128]:
            x,y=float(t.x.ravel()[index]),float(t.y.ravel()[index])
            floor=max(float(t.target.ravel()[index]),e.position[2]-self.config.excavator_dig_depth_m)
            if self.ramp_mask.ravel()[index]:floor=max(floor,float(self.ramp_surface.ravel()[index]))
            if t.height.ravel()[index]>floor+1e-5:return x,y,floor
        return None

    def _choose_excavator_site(self,e):
        target=self._available_target(e)
        if target is not None:e.target=target;e.state='WAIT_TRUCK';return True
        t=self.terrain;remaining=self._remaining();ids=np.flatnonzero(remaining>1e-6)
        if not len(ids):
            self._work_complete=True;self.reason='Захватка обработана; выездная рампа остаётся в проектной выемке';return False
        order=np.argsort((t.x.ravel()[ids]-e.position[0])**2+(t.y.ravel()[ids]-e.position[1])**2,kind='stable')
        samples=ids[order[::max(1,len(order)//8)][:8]]
        attempts=0
        for index in samples:
            x,y=t.x.ravel()[index],t.y.ravel()[index]
            for dx,dy in ((-1,0),(0,-1),(1,0),(0,1)):
                p=np.array([x+dx*self.config.excavator_reach_m*.8,y+dy*self.config.excavator_reach_m*.8])
                if not self.passable(p,e.radius,e):continue
                attempts+=1
                if attempts>8:
                    e.reason='Ограниченный поиск не нашёл безопасной полки с выездом';self.reason=e.reason;return False
                path=self.route(e,p)
                if path is None:continue
                old=e.position.copy();e.position[:2]=p;e.position[2]=t.sample(*p)
                escape=self.route(e,e.home[:2]);e.position[:]=old
                if escape is None:continue
                if self._assign_route(e,p,'WAIT_TRUCK'):e.target=None;return True
        self.reason='Нет доступной рабочей полки с безопасным выездом; рампа и остаток сохранены'
        e.reason=self.reason;return False

    def _loading_point(self,truck):
        e=self.excavator
        # A truck parks beside the counterweight, not on the cell being cut.
        candidates=[e.position[:2]+8.*np.array([math.cos(a),math.sin(a)]) for a in np.arange(8)*math.pi/4]
        candidates.sort(key=lambda p:(float(np.linalg.norm(p-truck.position[:2])),tuple(p)))
        for p in candidates:
            if e.target is not None and np.linalg.norm(p-np.array(e.target[:2]))<truck.radius+1.:continue
            if self.passable(p,truck.radius,truck) and self._assign_route(truck,p,'WAIT_LOAD'):return True
        return False

    def _dump_point(self,truck):
        t=self.terrain;volume=cargo_volume(truck.cargo_kg,truck.soil,t.config.terrain.density)
        ids=t._dump_indices
        available=np.maximum(t.dump_cap.ravel()[ids]-t.height.ravel()[ids],0.)
        ids=ids[available>1e-8]
        if not len(ids):return None
        distance=(t.x.ravel()[ids]-truck.position[0])**2+(t.y.ravel()[ids]-truck.position[1])**2
        for index in ids[np.argsort(distance,kind='stable')[::max(1,len(ids)//32)][:32]]:
            x,y=float(t.x.ravel()[index]),float(t.y.ravel()[index])
            # At most three metres of tipping reach; no remote map-wide dump.
            if t.local_dump_space(x,y)+1e-9<volume:continue
            for dx,dy in ((-1,0),(0,-1),(1,0),(0,1)):
                p=np.array([x+dx*6.,y+dy*6.])
                if not self.passable(p,truck.radius,truck):continue
                if self._assign_route(truck,p,'UNLOADING'):
                    truck.target=(x,y);truck.timer=30.;return p
        return None

    def _cut(self,e,truck):
        target=self._available_target(e)
        if target is None:return False
        x,y,floor=target;t=self.terrain;c=self.config
        radius=max(t.cell*.71,.5);patch=t.patch(x,y,radius)
        # Terrain already supports a per-cell lower surface. Keep every native
        # project cell and each ramp cell above its own envelope, not merely
        # above the centre cell's elevation.
        floor=np.maximum(t.target[patch],e.position[2]-c.excavator_dig_depth_m)
        floor=np.where(self.ramp_mask[patch],np.maximum(floor,self.ramp_surface[patch]),floor)
        # Exact support masks for both vehicles. A bucket does not excavate
        # beneath tracks, wheels, or the reserved exit ramp envelope.
        # A conservative loose-soil support envelope prevents the excavation
        # from leaving an unstable cliff below any parked machine. Lower layers
        # may be stronger, but the support guard deliberately uses loose repose.
        slope=t.dump_slope
        for machine in self.machines:
            distance=np.hypot(t.x[patch]-machine.position[0],t.y[patch]-machine.position[1])
            support=machine.position[2]-slope*np.maximum(distance-machine.radius-.5,0.)
            floor=np.maximum(floor,support)
        allowed=self.work_mask.copy()
        for m in self.machines:
            patch=t.patch(*m.position[:2],m.radius+1.)
            allowed[patch]&=(t.x[patch]-m.position[0])**2+(t.y[patch]-m.position[1])**2>(m.radius+1.)**2
        available_volume=min(c.bucket_m3,c.truck_body_m3-cargo_volume(truck.cargo_kg,truck.soil,t.config.terrain.density))
        mass_limit=c.truck_payload_kg-truck.cargo_kg
        if min(available_volume,mass_limit)<=1e-9:return False
        radius=max(t.cell*.71,.5)
        if t.soil is not None:
            load=t.scoop_load(x,y,mass_limit,source='conventional',radius=radius,max_volume=available_volume,allowed_mask=allowed,minimum_height=floor)
            mass=load.mass
            if mass>0:
                if truck.soil is None:truck.soil=SoilLoad.empty(t.soil.profile)
                truck.soil.add(load);self.extracted_materials+=load.masses
            volume=load.volume
        else:
            mass=t.scoop(x,y,min(mass_limit,available_volume*t.config.terrain.density),source='conventional',radius=radius,allowed_mask=allowed,minimum_height=floor)
            volume=mass/t.config.terrain.density
        if mass>0:self._terrain_changed()
        truck.cargo_kg+=mass;self.excavated_mass_kg+=mass;self.excavated_m3+=volume;e.excavated_m3+=volume
        e.target=target
        return mass>1e-9

    @property
    def ground_stage_drained(self):
        """A parked finished crew can hand its ramp/buffer to the aircraft."""
        return bool(self._work_complete and self.carried_mass_kg<=1e-8 and
                    all(m.state in ('PARKED','REFUELING','WAIT_FUEL') and not m.path and
                        np.linalg.norm(m.position[:2]-m.home[:2])<.1 for m in self.machines))

    def rebind_terrain(self,terrain):
        """Extend the common window at the same resolution without moving gear."""
        old=self.terrain
        if terrain.cell!=old.cell:raise ValueError('Изменение разрешения с обычной техникой пока не поддерживается')
        old_surface=self.ramp_surface;entry=np.asarray(self.ramp['entry_m']);end=np.asarray(self.ramp['end_m'])
        delta=end-entry;length=float(np.linalg.norm(delta));unit=delta/length
        old_along=(old.x[0,0]-entry[0])*unit[0]+(old.y[0,0]-entry[1])*unit[1]
        entry_height=float(old_surface[0,0]+np.clip(old_along,0,length)*self.config.ramp_gradient)
        self.terrain=terrain;x,y,w,h=self.config.work_rect
        self.work_mask=(terrain.x>=x)&(terrain.x<x+w)&(terrain.y>=y)&(terrain.y<y+h);self.mask=self.work_mask
        along=(terrain.x-entry[0])*unit[0]+(terrain.y-entry[1])*unit[1]
        lateral=np.abs((terrain.x-entry[0])*unit[1]-(terrain.y-entry[1])*unit[0])
        self.ramp_mask=self.work_mask&(lateral<=self.config.ramp_width_m/2)
        self.ramp_surface=entry_height-np.clip(along,0,length)*self.config.ramp_gradient
        rows,cols=old_surface.shape;self.ramp_surface[:rows,:cols]=old_surface
        self._marine=None;self._marine_stamp=None
        for machine in self.machines:machine.route_token=None;machine.budget_cache=None

    def _terrain_changed(self):
        from .conventional_integration import simulations_for
        for owner in simulations_for(self.sim):owner._terrain_active=True
        # Terrain revision/local route tokens already invalidate affected data.
        # Do not discard every photovoltaic/base geometry cache for one scoop.

    def set_enabled(self,enabled):
        self.enabled=bool(enabled);self.config.enabled=self.enabled
        for machine in self.machines:machine.retry_route_at=0.
        if not self.enabled:
            for m in self.machines:
                m.retry_route_at=0.
                if m.state!='PARKED':self._go_home(m)

    def replenish(self,fuel_l=0.,water_l=0.,food_units=0.):
        if any(not math.isfinite(v) or v<0 for v in (fuel_l,water_l,food_units)):raise ValueError('Пополнение должно быть неотрицательным')
        self.fuel_stock_l+=fuel_l;self.fuel_imported_l+=fuel_l
        self.water_stock_l+=water_l;self.water_imported_l+=water_l
        self.food_stock_units+=food_units;self.food_imported_units+=food_units
        for m in self.machines:m.next_plan_at=0.;m.retry_route_at=0.

    @property
    def carried_mass_kg(self):return sum(m.cargo_kg for m in self.machines)
    def loads(self):return [m.soil for m in self.machines if m.soil is not None]
    @property
    def carried_soil(self):
        if self.terrain.soil is None:return None
        result=SoilLoad.empty(self.terrain.soil.profile)
        for load in self.loads():result.add(load)
        return result

    def step(self):
        """Called exactly once per world second, also while autonomous fleet idle."""
        if self._last_time==self.sim.time:return
        self._last_time=self.sim.time;c=self.config;shift=self.shift()
        wage=(c.excavator_count*c.excavator_wage_egp_month+c.truck_count*c.driver_wage_egp_month)*c.shifts_per_day
        self.payroll_usd+=wage/(30*86400*c.egp_per_usd)
        active=shift['active'] and self.enabled
        if active and shift['key'] not in self._crew_paid:
            meals=(c.excavator_count+c.truck_count)*c.food_units_per_person_shift
            if self.food_stock_units+1e-9<meals:active=False;self.reason='Нет питания для выходящей смены'
            else:self.food_stock_units-=meals;self.food_used_units+=meals;self._crew_paid.add(shift['key'])
        if active:
            water=(c.excavator_count+c.truck_count)*c.water_l_per_person_hour/3600
            if self.water_stock_l+1e-9<water:active=False;self.reason='Нет питьевой воды: возвращение на базу'
            else:self.water_stock_l-=water;self.water_used_l+=water
        pumps=[]
        if active and self.fuel_stock_l>0:
            queued=[m for m in self.machines if np.linalg.norm(m.position[:2]-m.home[:2])<.1 and m.fuel_l<self._tank(m)-1e-9]
            pumps=sorted(queued,key=lambda m:(m.state!='REFUELING',m.id))[:c.refuel_slots]
        for m in self.machines:
            if m.state=='DRIVING':
                # A committed return continues after clock/lunch boundaries.
                if not active and m.after_drive!='PARKED':
                    self._go_home(m)
                if not shift['active'] and m.after_drive=='PARKED':
                    m.emergency_work_seconds+=1;m.reason='Аварийное возвращение: задержка маршрута после окончания рабочего времени'
                self._move(m);continue
            at_home=np.linalg.norm(m.position[:2]-m.home[:2])<.1
            if active and m in pumps and at_home and m.fuel_l<self._tank(m)-1e-9 and self.fuel_stock_l>0:
                refill=min(c.refuel_l_min/60,self._tank(m)-m.fuel_l,self.fuel_stock_l)
                m.fuel_l+=refill;self.fuel_stock_l-=refill;m.state='REFUELING';m.timer+=1;continue
            if m.state=='REFUELING':m.state='PARKED'
            if active and at_home and m.fuel_l<self._tank(m)*.95:
                m.state='WAIT_FUEL';m.reason='Очередь к заправочному посту' if self.fuel_stock_l>0 else 'Топливо на базе исчерпано';continue
            if not active:
                m.reason=shift['reason'] if not shift['active'] else self.reason or 'Работа приостановлена'
                if not at_home and self.sim.time>=m.retry_route_at:self._go_home(m)
                elif at_home:m.state='PARKED'
                continue
            if m.fuel_l<self._tank(m)*(.95 if at_home else c.fuel_reserve_fraction):
                m.reason='Нужна дозаправка';
                if not at_home:self._go_home(m)
                continue
            if self.sim.time<m.retry_route_at:
                m.waiting_seconds+=1;continue
            # Bound return time/fuel once on work assignment and periodically.
            if True:
                budget=self._return_budget(m,c.bucket_cycle_s if m.kind=='excavator' else 60.)
                if budget is None:m.reason='Нет безопасного выезда на базу';m.retry_route_at=self.sim.time+30;continue
                seconds,fuel,_=budget
                if shift['seconds_left']<=seconds or m.fuel_l<fuel:
                    self._go_home(m);continue
            if m.kind=='excavator':
                if self._work_complete:
                    if not at_home:self._go_home(m)
                    continue
                if m.state=='PARKED' or m.target is None:
                    if self.sim.time<m.next_plan_at:continue
                    if not self._choose_excavator_site(m):m.next_plan_at=self.sim.time+60;continue
                if m.state=='DRIVING':continue
                truck=next((v for v in self.machines if v.kind=='truck' and v.state=='WAIT_LOAD' and
                            np.linalg.norm(v.position[:2]-m.position[:2])<=8.5),None)
                if truck is None:m.state='WAIT_TRUCK';m.waiting_seconds+=1;continue
                if not self._burn(m,True):continue
                m.state='DIGGING';m.working_seconds+=1;m.timer+=1
                factor=1.
                if self.terrain.soil is not None and m.target is not None:
                    load=self.terrain.preview_load(*m.target[:2],c.truck_payload_kg,source='conventional',radius=max(.5,self.terrain.cell*.71),max_volume=c.bucket_m3)
                    factor=max(1.,load.factor)
                if m.timer>=c.bucket_cycle_s*factor:
                    m.timer=0.;cut=self._cut(m,truck)
                    body=cargo_volume(truck.cargo_kg,truck.soil,self.terrain.config.terrain.density)
                    if truck.cargo_kg>=c.truck_payload_kg*.9 or body>=c.truck_body_m3*.9 or not cut:
                        if truck.cargo_kg>0:
                            if self._dump_point(truck) is None:truck.reason='Нет доступного места отвала; груз сохранён'
                        m.target=None
            else:
                if self._work_complete and m.cargo_kg>0 and m.state not in ('DRIVING','UNLOADING'):
                    if self._dump_point(m) is None:m.reason='Нет доступного отвала; груз сохранён'
                    continue
                if self._work_complete and m.cargo_kg<=0:
                    if not at_home:self._go_home(m)
                    continue
                if m.state in ('PARKED','WAIT_LOAD'):
                    if m.cargo_kg>0 and m.state=='PARKED':
                        if self._dump_point(m) is None:m.reason='Нет доступного отвала; груз сохранён'
                    elif self.excavator.state in ('WAIT_TRUCK','DIGGING') and self.excavator.target is not None:
                        if m.state=='PARKED':
                            bay=next((other for other in self.machines if other is not m and other.kind=='truck' and
                                (other.state=='WAIT_LOAD' or other.state=='DRIVING' and other.after_drive=='WAIT_LOAD')),None)
                            if bay is not None:
                                m.reason='Очередь к единственному месту погрузки; ожидание на базе';m.waiting_seconds+=1;continue
                            self._loading_point(m);m.load_started=self.sim.time
                        elif self.sim.time-m.load_started>300 and m.cargo_kg>0:self._dump_point(m)
                    continue
                if m.state=='UNLOADING':
                    if not self._burn(m):continue
                    m.timer-=1
                    if m.timer<=0:
                        volume=cargo_volume(m.cargo_kg,m.soil,self.terrain.config.terrain.density)
                        if self.terrain.local_dump_space(*m.target)+1e-9<volume:
                            m.reason='Отвал изменился: выбирается другая точка; груз сохранён';self._dump_point(m);continue
                        self.terrain.deposit(*m.target,m.cargo_kg,radius=2.,load=m.soil)
                        self._terrain_changed()
                        self.delivered_mass_kg+=m.cargo_kg
                        if m.soil is not None:self.delivered_materials+=m.soil.masses
                        self.delivered_m3+=volume;m.delivered_m3+=volume;m.cargo_kg=0.;m.soil=None;m.target=None;m.state='PARKED'
        if self._work_complete and not self.carried_mass_kg:
            for m in self.machines:
                if np.linalg.norm(m.position[:2]-m.home[:2])>.1 and m.after_drive!='PARKED':self._go_home(m)

    def validate(self):
        self.config.validate()
        if self.terrain is not getattr(self.sim.terrain,'core',self.sim.terrain):raise ValueError('Обычная техника: потеряна общая карта')
        if len(self.machines)!=1+self.config.truck_count or len({m.id for m in self.machines})!=len(self.machines):raise ValueError('Повреждён состав обычной техники')
        for value in (self.fuel_stock_l,self.water_stock_l,self.food_stock_units,self.fuel_used_l,self.water_used_l,self.food_used_units):
            if not math.isfinite(value) or value < -1e-8:raise ValueError('Повреждён запас расходных материалов')
        for m in self.machines:
            if (not isinstance(m.position,np.ndarray) or m.position.shape!=(3,) or not np.all(np.isfinite(m.position)) or
                not isinstance(m.home,np.ndarray) or m.home.shape!=(3,) or not np.all(np.isfinite(m.home))):
                raise ValueError('Повреждена позиция машины')
            if not math.isfinite(m.fuel_l) or not -1e-8<=m.fuel_l<=self._tank(m)+1e-8 or not math.isfinite(m.cargo_kg) or not 0<=m.cargo_kg<=self.config.truck_payload_kg+1e-7:
                raise ValueError('Повреждён бак или груз машины')
            if self.terrain.soil is not None and m.cargo_kg>0:
                if m.soil is None or m.soil.profile!=self.terrain.soil.profile:raise ValueError('Потерян состав грунта в кузове')
                m.soil.validate(m.cargo_kg)
            elif m.soil is not None:m.soil.validate(m.cargo_kg)
        if abs(self.excavated_mass_kg-self.delivered_mass_kg-self.carried_mass_kg)>1e-6:raise ValueError('Нарушен баланс перевезённого грунта')
        if self.extracted_materials is not None:
            if np.max(np.abs(self.extracted_materials-self.delivered_materials-self.carried_soil.masses))>1e-6:raise ValueError('Нарушен состав перевезённого грунта')
        r=self.report()
        if any(abs(r[k])>1e-6 for k in ('fuel_balance_error_l','water_balance_error_l','food_balance_error_units')):raise ValueError('Нарушен баланс запасов базы')
        return True

    def report(self):
        c=self.config;currency=c.egp_per_usd
        fuel=self.fuel_used_l*c.diesel_egp_l/currency
        food=self.food_used_units*c.food_egp_unit/currency;water=self.water_used_l*c.water_egp_l/currency
        operating=fuel+food+water+self.payroll_usd+self.maintenance_usd
        capital=c.excavator_count*c.excavator_purchase_usd+c.truck_count*c.truck_purchase_usd+c.base_purchase_usd
        t=self.terrain;retained=np.where(self.ramp_mask&t.dig_mask,np.maximum(np.minimum(t.height,self.ramp_surface)-t.target,0.),0.)
        ramp={**self.ramp,'remaining_m3':float(retained.sum()*t.area)}
        return dict(enabled=self.enabled,reason=self.reason,work_rect=list(c.work_rect),base_position_m=self.base_position.tolist(),
            excavated_m3=self.excavated_m3,excavated_mass_kg=self.excavated_mass_kg,delivered_m3=self.delivered_m3,
            carried_mass_kg=self.carried_mass_kg,capital_cost_usd=capital,operating_cost_usd=operating,total_cost_usd=capital+operating,
            cost_per_m3_usd=(capital+operating)/self.excavated_m3 if self.excavated_m3>0 else None,
            fuel_stock_l=self.fuel_stock_l,water_stock_l=self.water_stock_l,food_stock_units=self.food_stock_units,
            fuel_used_l=self.fuel_used_l,water_used_l=self.water_used_l,food_used_units=self.food_used_units,
            payroll_usd=self.payroll_usd,fuel_cost_usd=fuel,food_cost_usd=food,water_cost_usd=water,maintenance_usd=self.maintenance_usd,
            inventory_purchases_usd=((c.fuel_stock_l+self.fuel_imported_l)*c.diesel_egp_l+(c.food_stock_units+self.food_imported_units)*c.food_egp_unit+(c.water_stock_l+self.water_imported_l)*c.water_egp_l)/currency,
            consumable_inventory_value_usd=((self.fuel_stock_l+sum(m.fuel_l for m in self.machines))*c.diesel_egp_l+self.food_stock_units*c.food_egp_unit+self.water_stock_l*c.water_egp_l)/currency,
            fuel_balance_error_l=c.fuel_stock_l+self.fuel_imported_l-self.fuel_stock_l-self.fuel_used_l-sum(m.fuel_l for m in self.machines),
            water_balance_error_l=c.water_stock_l+self.water_imported_l-self.water_stock_l-self.water_used_l,
            food_balance_error_units=c.food_stock_units+self.food_imported_units-self.food_stock_units-self.food_used_units,
            diesel_energy_consumed_mj=self.fuel_used_l*35.8,shift=self.shift(),ramp=ramp,remaining_work_m3=float(self._remaining().sum()*t.area),
            refuel_slots=c.refuel_slots,route_calculations=self.route_calculations,crew_count=c.shifts_per_day*(c.excavator_count+c.truck_count),
            machines=[dict(id=m.id,kind=m.kind,state=m.state,reason=m.reason,position_m=m.position.tolist(),dimensions_m=list(m.dimensions_m),footprint_radius_m=m.radius,fuel_l=m.fuel_l,
                           cargo_kg=m.cargo_kg,excavated_m3=m.excavated_m3,delivered_m3=m.delivered_m3,path_m=[list(p) for p in m.path],
                           tank_capacity_l=self._tank(m),emergency_work_seconds=m.emergency_work_seconds,engine_seconds=m.engine_seconds,working_seconds=m.working_seconds,distance_m=m.distance_m)
                      for m in self.machines])
