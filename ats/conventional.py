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
    auto_work_area: bool = False
    dump_order: str = 'far_to_near'
    excavator_dimensions_m: tuple = (6.,3.,3.)
    truck_dimensions_m: tuple = (7.,2.8,3.2)
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
    ramp_entry_m: tuple | None = None
    ramp_end_m: tuple | None = None
    route_step_m: float = 2.
    max_route_nodes: int = 1200
    cut_tolerance_m: float = .25

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
        if type(self.auto_work_area) is not bool:raise ValueError('Автоматическое назначение участка должно быть включено или выключено')
        if self.dump_order not in ('far_to_near','nearest'):raise ValueError('Неизвестный порядок заполнения отвала')
        if not isinstance(self.cut_tolerance_m,(int,float)) or not 0<=self.cut_tolerance_m<=.30:
            raise ValueError('Допуск выемки по высоте должен быть от 0 до 30 см')
        if (self.ramp_entry_m is None)!=(self.ramp_end_m is None):raise ValueError('Задайте оба конца съезда')
        if self.ramp_entry_m is not None:
            if any(len(p)!=2 or any(not math.isfinite(v) for v in p) for p in (self.ramp_entry_m,self.ramp_end_m)):
                raise ValueError('Неверные координаты съезда')
            if np.linalg.norm(np.asarray(self.ramp_end_m)-self.ramp_entry_m)<1.:raise ValueError('Съезд короче 1 м')
        for dimensions in (self.excavator_dimensions_m,self.truck_dimensions_m):
            if len(dimensions)!=3 or any(not math.isfinite(v) or not .5<=v<=30 for v in dimensions):
                raise ValueError('Габариты машин: три размера от 0,5 до 30 м')
        clearance=max(math.hypot(*d[:2]) for d in (self.excavator_dimensions_m,self.truck_dimensions_m))+1.
        if not .02<=self.ramp_gradient<=.2 or not clearance<=self.ramp_width_m or not 0<self.fuel_reserve_fraction<.5 or not 100<=self.max_route_nodes<=20000:
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
        return cls(base_x=max(5.,min(side-5.,rect[0]-9)),base_y=max(5.,min(side-5.,rect[1]-9)),work_rect=rect,start_weekday=weekday,
                   auto_work_area=bool(sim.config.plan and sim.config.plan.channel))


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
    next_advance_at: float = 0.
    site_search_cursor: int = 0
    advance_search_cursor: int = 0
    blocked_revision: int = -1
    emergency_work_seconds: float = 0.
    budget_cache: dict | None = None
    retry_route_at: float = 0.
    body_dimensions_m: tuple | None = None
    heading_rad: float = 0.
    expose_target: tuple | None = None
    side_bucket: bool = False
    site_search_stage: int = 0

    @property
    def dimensions_m(self):return self.body_dimensions_m or ((6.,3.,3.) if self.kind=='excavator' else (7.,2.8,3.2))

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
        if not self.passable(self.base_position[:2],math.hypot(4.,4.),ignore_vehicles=True):raise ValueError('Топливная база не помещается на безопасной суше')
        if np.any((self.work_mask|t.work_mask)[t.patch(c.base_x,c.base_y,4.)]):raise ValueError('Топливная база пересекает проектную выемку')
        self.ramp=self._make_ramp()
        for i in range(c.excavator_count+c.truck_count):
            kind='excavator' if i<c.excavator_count else 'truck';serial=i+1 if kind=='excavator' else i-c.excavator_count+1
            dimensions=c.excavator_dimensions_m if kind=='excavator' else c.truck_dimensions_m
            point=self._parking_point(math.hypot(*dimensions[:2])/2)
            if point is None:raise ValueError('Нет места для стоянки всего парка; расширьте площадку базы')
            p=np.array([*point,t.sample(*point)])
            self.machines.append(GroundMachine(f'{kind}_{serial}',kind,p,p.copy(),body_dimensions_m=tuple(dimensions)))
        self.excavator=self.machines[0]

    def _make_ramp(self):
        t=self.terrain;c=self.config;x,y,w,h=c.work_rect;along_x=w>=h
        start=np.array([x,y+h/2]) if along_x else np.array([x+w/2,y])
        end=np.array([x+w,y+h/2]) if along_x else np.array([x+w/2,y+h])
        if c.ramp_entry_m is not None:
            start=np.asarray(c.ramp_entry_m,dtype=float);end=np.asarray(c.ramp_end_m,dtype=float)
            if any(not(x<=p[0]<=x+w and y<=p[1]<=y+h) for p in (start,end)):
                raise ValueError('Ось съезда выходит за рабочую захватку')
        elif np.linalg.norm(self.base_position[:2]-end)<np.linalg.norm(self.base_position[:2]-start):start,end=end,start
        delta=end-start;length=float(np.linalg.norm(delta));unit=delta/length
        # Begin the grade at the actual cut, rather than lowering it invisibly
        # across the unexcavated margin of the rectangular allocation.
        along=(t.x-start[0])*unit[0]+(t.y-start[1])*unit[1]
        lateral=np.abs((t.x-start[0])*unit[1]-(t.y-start[1])*unit[0])
        entrance=self.work_mask&t.dig_mask&(lateral<=c.ramp_width_m/2)
        if np.any(entrance):start=start+unit*max(0.,float(along[entrance].min())-t.cell/2)
        length=float(np.linalg.norm(end-start))
        along=(t.x-start[0])*unit[0]+(t.y-start[1])*unit[1]
        lateral=np.abs((t.x-start[0])*unit[1]-(t.y-start[1])*unit[0])
        entry_height=t.sample(*start)
        # The retained road includes supporting shoulders. Excavating both
        # flanks to the channel bottom must not leave a narrow raised wall.
        surface=(entry_height-np.clip(along,0,length)*c.ramp_gradient-
                 t.dump_slope*np.maximum(lateral-c.ramp_width_m/2,0.))
        mask=self.work_mask&(surface>t.target+1e-8)
        self.ramp_mask=mask;self.ramp_surface=surface
        self.access_mask=self.work_mask&t.dig_mask&(lateral<=c.ramp_width_m/2)&(along>=0)&(along<=length)
        depth=max(0.,entry_height-float(t.target[self.work_mask&t.dig_mask].min()))
        return dict(entry_m=start.tolist(),end_m=end.tolist(),width_m=c.ramp_width_m,max_gradient=c.ramp_gradient,
                    model='supported_access_v2',entry_height_m=entry_height,required_length_m=depth/c.ramp_gradient)

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

    def _marine_mask(self):
        t=self.terrain
        stamp=(id(t),id(t.protected),id(t.reference),id(t.config.plan))
        if self._marine_stamp!=stamp:
            from .site_ground import marine_mask
            self._marine=marine_mask(t);self._marine_stamp=stamp
        return self._marine

    def _ground_obstacles(self,bounds,radius,machine,ignore_vehicles=False,*,describe=False):
        """Closed rectangle (or point) plus the existing body clearance."""
        for x,y,width,height in self._other_rectangles(bounds):
            yield x,y,x+width,y+height,radius+.2,'оборудование комплекса'
        if not ignore_vehicles:
            x,y=self.base_position[:2]
            yield x-4.,y-4.,x+4.,y+4.,radius+.2,'топливная база'
            for other in self.machines:
                if other is not machine:
                    x,y=other.position[:2]
                    yield x,y,x,y,other.radius+radius+.25,other.id
        from .conventional_integration import simulations_for
        for owner in simulations_for(self.sim):
            for drone in owner.drones:
                if drone.grounded:
                    x,y=drone.position[:2]
                    label=f'дрон №{drone.serial} на земле' if describe else ''
                    yield x,y,x,y,radius+max(.5,owner.config.fleet.separation/2),label
                if drone.trip is not None:
                    for point in (drone.trip.dig,drone.trip.dump):
                        x,y=point[:2]
                        label=f'рабочая точка дрона №{drone.serial}' if describe else ''
                        yield x,y,x,y,radius+max(drone.config.grab_radius or .5,owner.config.fleet.separation),label
            for robot in (owner.logistics.excavators if owner.logistics else ()):
                x,y=robot.position[:2]
                label=f'Крот №{robot.serial}' if describe else ''
                yield x,y,x,y,robot.config.support_half_width_m+radius+.3,label

    def passable(self,point,radius,machine=None,*,ignore_vehicles=False):
        return self._footprint_issue(point,radius,machine,ignore_vehicles) is None

    def _footprint_issue(self,point,radius,machine,ignore_vehicles=False,*,describe=False):
        """Same support law for routing and on-demand reports about a failed pose."""
        t=self.terrain;x,y=map(float,point[:2]);side=t.cell*t.height.shape[0]
        if min(x,y)<radius+t.cell/2 or max(x,y)>side-radius-t.cell/2:return 'габарит машины выходит за карту'
        patch=t.patch(x,y,radius+t.cell/2);xx=t.x[patch]-x;yy=t.y[patch]-y
        footprint=xx*xx+yy*yy<=(radius+t.cell*.71)**2
        if not np.any(footprint):return 'нет ячеек опоры под машиной'
        if np.any(self._marine_mask()[patch]&footprint):return 'вода под габаритом машины'
        heights=t.height[patch];z=t.sample(x,y);distance=np.maximum(np.hypot(xx,yy),t.cell/2)
        # Full circumscribed support footprint, including every contained fine cell.
        if np.any((np.abs(heights-z)>.2*distance+.12)&footprint):return 'слишком большой перепад от центра до края опоры'
        if heights.shape[0]>1 and np.any((np.abs(np.diff(heights,axis=0))>.2*t.cell+.12)&footprint[:-1]&footprint[1:]):return 'ступень между ячейками под машиной'
        if heights.shape[1]>1 and np.any((np.abs(np.diff(heights,axis=1))>.2*t.cell+.12)&footprint[:,:-1]&footprint[:,1:]):return 'ступень между ячейками под машиной'
        bounds=(x-radius,y-radius,x+radius,y+radius)
        for left,bottom,right,top,clearance,label in self._ground_obstacles(bounds,radius,machine,ignore_vehicles,describe=describe):
            dx=max(left-x,0.,x-right);dy=max(bottom-y,0.,y-top)
            if np.linalg.norm((dx,dy))<clearance:
                return f'недостаточный зазор до препятствия ({label})' if describe else 'препятствие'
        return None

    def _clear_ground_corridor(self,a,b,radius,machine):
        """Sufficient live proof for all original drive and return samples."""
        t=self.terrain;cell=t.cell
        left,right=sorted((a[0],b[0]));bottom,top=sorted((a[1],b[1]))
        # Allow for floating point interpolation at the ends of a grid edge.
        margin=radius+cell/2+1e-8
        side=cell*t.height.shape[0]
        if min(left,bottom)<margin or max(right,top)>side-margin:return False
        col0=math.floor((left-margin)/cell);col1=math.floor((right+margin)/cell)+1
        row0=math.floor((bottom-margin)/cell);row1=math.floor((top+margin)/cell)+1
        if (col1-col0)*(row1-row0)>65536:return False
        heights=t.height[row0:row1,col0:col1]
        # Even a zero-length grade check allows 0.12 m. This range over the
        # whole swept box bounds every pair of heights in every sample.
        if np.any(self._marine_mask()[row0:row1,col0:col1]):return False
        if np.ptp(heights)>.12:
            # Bound the absolute height difference for any two grid cells by
            # their numbers of x/y steps times the maximum local increments.
            # Nearest-cell sampling adds at most one cell per axis to the
            # separation of two actual points. Thus the gradient and the
            # sampling slack below imply every original 0.2*run+0.12 test.
            dx=np.abs(np.diff(heights,axis=1)).max(initial=0.)
            dy=np.abs(np.diff(heights,axis=0)).max(initial=0.)
            if dx+dy>=.12-1e-12 or math.hypot(dx,dy)>=.2*cell-1e-12:return False
        bounds=(left-margin,bottom-margin,right+margin,top+margin)
        for x0,y0,x1,y1,clearance,_ in self._ground_obstacles(bounds,radius,machine):
            # If the boxes are close on both axes use the normal disk check.
            if max(left-x1,x0-right,bottom-y1,y0-top)<=clearance+1e-8:return False
        return True

    def segment_safe(self,a,b,radius,machine=None,*,reason=None):
        a=np.asarray(a)[:2];b=np.asarray(b)[:2]
        if self._clear_ground_corridor(a,b,radius,machine):return True
        length=float(np.linalg.norm(b-a));step=min(1.,self.terrain.cell)
        def samples(start,end):
            distance=float(np.linalg.norm(end-start))
            return np.linspace(start,end,max(2,int(math.ceil(distance/step))+1))
        def checks():
            yield samples(a,b)
            if machine is not None and length>self._speed(machine)+1e-9:
                # The support mask includes individual terrain cells. A long
                # edge's samples can miss a cut lip which the differently phased
                # one-second driving steps then hit forever. Check those steps
                # too, retaining the original grade checks. Both directions are
                # needed: an outbound is reversed as a checked escape route.
                speed=self._speed(machine)
                for start,end in ((a,b),(b,a)):
                    direction=(end-start)/length
                    for distance in np.arange(0.,length,speed):
                        yield samples(start+direction*distance,
                                      start+direction*min(length,distance+speed))
        checked=set()
        for points in checks():
            heights=self.terrain.sample_many(points);runs=np.linalg.norm(np.diff(points,axis=0),axis=1)
            steep=np.abs(np.diff(heights))>.2*runs+.12
            if np.any(steep):
                if reason is not None:
                    x,y=points[np.flatnonzero(steep)[0]+1]
                    reason.append(f'Слишком крутой перепад по ходу движения в точке ({x:.1f}, {y:.1f}) м')
                return False
            for point in points:
                key=tuple(point)
                if key not in checked:
                    if not self.passable(point,radius,machine):
                        if reason is not None:
                            issue=self._footprint_issue(point,radius,machine,describe=True)
                            reason.append(f'В точке ({point[0]:.1f}, {point[1]:.1f}) м — {issue}')
                        return False
                    checked.add(key)
        return True

    def route(self,machine,destination):
        """Bounded deterministic A*; routes are computed only on assignment/change."""
        self.route_calculations+=1;start=machine.position[:2].copy();goal=np.asarray(destination)[:2]
        self._route_failure=''
        if not self.passable(goal,machine.radius,machine):
            issue=self._footprint_issue(goal,machine.radius,machine,describe=True)
            self._route_failure=f'Точка назначения ({goal[0]:.1f}, {goal[1]:.1f}) м — {issue}'
            return None
        if self.segment_safe(start,goal,machine.radius,machine):return [tuple(goal)]
        if not self.passable(start,machine.radius,machine):
            # Every outgoing edge includes the current footprint.
            issue=self._footprint_issue(start,machine.radius,machine,describe=True)
            self._route_failure=f'Текущая стоянка ({start[0]:.1f}, {start[1]:.1f}) м — {issue}'
            return None
        step=max(self.config.route_step_m,self.terrain.cell)
        side=self.terrain.cell*self.terrain.height.shape[0]
        waypoints=[start,goal];x,y,w,h=self.config.work_rect
        if any(x<=p[0]<=x+w and y<=p[1]<=y+h for p in (start,goal)):
            entry=np.asarray(self.ramp['entry_m']);delta=np.asarray(self.ramp['end_m'])-entry
            unit=delta/max(float(np.linalg.norm(delta)),1e-9)
            # A deep haul may have to leave in the opposite direction through
            # the known ramp crest. Do not crop that real exit out of A* merely
            # because it lies beyond start/goal's arbitrary 32 m rectangle.
            waypoints.extend([entry,entry-unit*(machine.radius+2.)])
        extent=np.asarray(waypoints)
        origin=np.floor(extent.min(axis=0)/step)*step-32
        lo=np.maximum(machine.radius+self.terrain.cell/2,origin)
        hi=np.minimum(side-machine.radius-self.terrain.cell/2,extent.max(axis=0)+32)
        # Anchor the lattice to the actual pose. Replacing just its first grid
        # vertex with an off-grid pose skews every first edge; that can miss a
        # narrow but safe straight withdrawal from an excavated shelf.
        def key(p):return tuple(np.rint((np.asarray(p)-start)/step).astype(int))
        first=(0,0)
        def point(k):return start+np.asarray(k)*step
        diagonal=math.sqrt(2.)-1.
        finish=key(goal);terminals=[]
        for ox in range(-2,3):
            for oy in range(-2,3):
                q=point(np.asarray(finish)+(ox,oy))
                tail=float(np.linalg.norm(q-goal))
                if tail<=step*1.5:terminals.append((float(q[0]),float(q[1]),tail))
        def heuristic(p):
            # Exact relaxed cost on the eight-neighbour lattice plus its
            # off-grid final connector. Ignoring obstacles gives an admissible
            # lower bound; a constant connector discount instead leaves a huge
            # cheaper plateau ahead of the goal on long diagonal journeys.
            px,py=map(float,p)
            return min(max(dx,dy)+diagonal*min(dx,dy)+tail
                       for qx,qy,tail in terminals
                       for dx,dy in [(abs(px-qx),abs(py-qy))])
        queue=[(float(np.linalg.norm(start-goal)),0.,first)]
        costs={first:0.};parents={};checked={};edges={};seen=set();limit=self.config.max_route_nodes
        def clear(k):
            if k not in checked:
                p=point(k);checked[k]=bool(np.all(p>=lo)&np.all(p<=hi)) and self.passable(p,machine.radius,machine)
            return checked[k]
        while queue and len(seen)<limit:
            # Equal lower-bound costs may describe thousands of equivalent
            # diagonal-grid paths. Prefer greater progress within that tie;
            # the physical cost and admissible primary priority stay intact.
            _,negative_cost,current=heapq.heappop(queue);cost=-negative_cost
            if current in seen:continue
            seen.add(current);p=point(current)
            if np.linalg.norm(p-goal)<=step*1.5 and self.segment_safe(p,goal,machine.radius,machine):
                nodes=[tuple(goal)];node=current
                while node!=first:nodes.append(tuple(point(node)));node=parents[node]
                nodes.reverse()
                return nodes
            for dx,dy in ((1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)):
                nxt=(current[0]+dx,current[1]+dy)
                if nxt in seen or not clear(nxt):continue
                if dx and dy and (not clear((current[0]+dx,current[1])) or not clear((current[0],current[1]+dy))):continue
                q=point(nxt)
                if abs(self.terrain.sample(*q)-self.terrain.sample(*p))>.2*step*math.hypot(dx,dy)+.12:continue
                # Adjacent grid nodes still require swept-footprint validation.
                tentative=cost+float(np.linalg.norm(q-p))
                if tentative+1e-9>=costs.get(nxt,math.inf):continue
                edge=tuple(sorted((current,nxt)))
                if edge not in edges:edges[edge]=self.segment_safe(p,q,machine.radius,machine)
                if not edges[edge]:continue
                costs[nxt]=tentative;parents[nxt]=current
                heapq.heappush(queue,(tentative+heuristic(q),-tentative,nxt))
        self._route_failure=(f'Лимит узлов локального поиска ({limit}) исчерпан, путь пока не найден' if queue else
                             'Нет безопасного пути в локальном окне поиска')
        return None

    def _assign_route(self,m,destination,after,home_path=None,outbound_path=None):
        path=self.route(m,destination) if outbound_path is None else list(outbound_path)
        if path is None:m.reason=getattr(self,'_route_failure','') or 'Нет безопасного наземного маршрута';m.retry_route_at=self.sim.time+30;return False
        path=self._without_loops(m.position[:2],path)
        if after!='PARKED':
            shift=self.shift();outbound=self._path_length(m,path)/self._speed(m)
            checked_home=None
            if home_path is None:
                # A coarse grid may find an approach but miss its reverse
                # off-grid connection. Retain the physical way back: reverse
                # this actual outbound, then follow the current checked exit.
                checked_home=self._home_route(m,validate=True)
            old=m.position.copy();m.position[:2]=destination[:2];m.position[2]=self.terrain.sample(*destination[:2])
            if home_path is None and checked_home is not None:
                visited=[tuple(old[:2]),*path]
                backtrack=[*reversed(visited[:-1]),*checked_home]
                if all(self.segment_safe(a,b,m.radius,m)
                       for a,b in zip([m.position[:2],*backtrack[:-1]],backtrack)):
                    home_path=backtrack
            if home_path is None:home_path=self.route(m,m.home[:2])
            if home_path is None:
                m.position[:]=old
                m.reason='Нет безопасного обратного пути на базу';m.retry_route_at=self.sim.time+30;return False
            home_path=self._without_loops(m.position[:2],home_path)
            back=self._path_length(m,home_path)/self._speed(m) if home_path is not None else math.inf
            m.position[:]=old
            duration=outbound+back+max(60.,self.config.bucket_cycle_s)+45.
            fuel=duration*max(self._rate(m,True),self.config.truck_drive_l_h)/3600+self._tank(m)*self.config.fuel_reserve_fraction
            if not shift['active'] or shift['seconds_left']<duration:
                m.reason='Недостаточно времени на рейс и возвращение до обеда / конца смены';m.retry_route_at=self.sim.time+30;return False
            if m.fuel_l<fuel:m.reason='Недостаточно топлива на рейс, возврат и резерв';m.retry_route_at=self.sim.time+30;return False
        m.budget_cache=None;m.retry_route_at=0.
        self._set_drive_path(m,path)
        m.destination=tuple(destination[:2]);m.after_drive=after;m.state='DRIVING';m.reason=''
        if after!='PARKED':
            points=np.array([np.asarray(destination)[:2],*home_path]);lo=points.min(axis=0)-m.radius;hi=points.max(axis=0)+m.radius
            bounds=tuple(np.r_[lo,hi])
            m.budget_cache=dict(position=tuple(destination[:2]),path=list(home_path),bounds=bounds,token=self._home_route_token(m,bounds))
        return True

    def _set_drive_path(self,m,path):
        m.path=self._without_loops(m.position[:2],path)
        points=np.array([m.position[:2],*m.path]);lo=points.min(axis=0)-m.radius;hi=points.max(axis=0)+m.radius
        m.route_bounds=tuple(np.r_[lo,hi]);m.route_token=self.terrain.route_token(m.route_bounds)

    @staticmethod
    def _without_loops(start,path):
        """Erase repeated exact vertices; every remaining edge was checked."""
        points=[tuple(map(float,start))];seen={points[0]:0}
        for point in path:
            point=tuple(map(float,point))
            if point in seen:
                index=seen[point]
                for old in points[index+1:]:seen.pop(old,None)
                del points[index+1:]
            else:seen[point]=len(points);points.append(point)
        return points[1:]

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

    def _recover_drive(self,m,details):
        """Recover a blocked leg without abandoning the load or moving the vehicle."""
        issue='Проезд занят или уклон изменился'
        if details:issue+=': '+details[0]
        m.next_plan_at=self.sim.time+30
        returning=m.after_drive=='PARKED' and np.linalg.norm(np.asarray(m.destination)-m.home[:2])<1e-7
        path=self._home_route(m,validate=True) if returning else self.route(m,m.destination)
        if path is not None:
            if m.kind=='truck' and m.after_drive!='PARKED':
                ready=self._assign_route(m,m.destination,m.after_drive,outbound_path=path)
            else:
                self._set_drive_path(m,path)
                ready=True
            if ready:
                m.reason=issue+'. Найден объезд'
                return True
            failure=m.reason
        else:
            failure=getattr(self,'_route_failure','')
        if (m.kind=='truck' and m.after_drive in ('UNLOADING','WAIT_LOAD') and
                self.enabled and self.shift()['active'] and self.passable(m.position,m.radius,m)):
            # A loading or tipping stand may cease to exist during a trip.
            # Try the same bounded dispatcher used for the original order. It
            # verifies access, a way back and the trip's time/fuel budget.
            # Each candidate gets its own budget even when the old trip failed.
            m.retry_route_at=0.
            if m.after_drive=='UNLOADING' and m.cargo_kg>0:
                if self._dump_point(m) is not None:
                    m.reason=issue+'. Выбрана другая стоянка для разгрузки'
                    return True
                failure='; '.join(filter(None,(failure,m.reason)))
            elif (m.after_drive=='WAIT_LOAD' and self.excavator.target is not None and
                  self.excavator.state in ('WAIT_TRUCK','DIGGING')):
                if self._loading_point(m):
                    m.reason=issue+'. Выбрана другая стоянка для погрузки'
                    return True
                failure='; '.join(filter(None,(failure,m.reason)))
        m.reason='. '.join(filter(None,(issue,failure)))
        if m.cargo_kg>0 and 'груз сохранён' not in m.reason:m.reason+='; груз сохранён'
        return False

    def _move(self,m):
        if not m.path:
            m.state=m.after_drive;m.reason=''
            if m.kind=='truck' and m.state=='WAIT_LOAD' and m.cargo_kg<=1e-9:m.load_started=self.sim.time
            return
        if self.sim.time<m.retry_route_at:m.waiting_seconds+=1;return
        if m.route_bounds is None or m.route_token!=self.terrain.route_token(m.route_bounds):
            # A changed block can be behind the truck or off the road. The
            # saved waypoints may also include a detour outside a new A* box.
            details=[]
            for a,b in zip([m.position[:2],*m.path[:-1]],m.path):
                if not self.segment_safe(a,b,m.radius,m,reason=details):
                    if not self._recover_drive(m,details):
                        m.retry_route_at=self.sim.time+30
                    m.waiting_seconds+=1
                    return
            self._set_drive_path(m,m.path)
            if not m.path:return self._move(m)
        destination=np.asarray(m.path[0]);delta=destination-m.position[:2];length=float(np.linalg.norm(delta))
        next_point=m.position[:2]+delta*min(1.,self._speed(m)/max(length,1e-9))
        details=[] if self.sim.time>=m.next_plan_at else None
        if not self.segment_safe(m.position,next_point,m.radius,m,reason=details):
            m.waiting_seconds+=1
            if details is not None:
                self._recover_drive(m,details)
            elif not m.reason:m.reason='Проезд занят; ожидание повторной проверки'
            return
        if not self._burn(m):return
        m.reason=''
        if length>1e-9:m.heading_rad=math.atan2(delta[1],delta[0])
        m.position[:2]=next_point;m.position[2]=self.terrain.sample(*next_point);m.distance_m+=min(length,self._speed(m))
        if length<=self._speed(m)+1e-9:m.path.pop(0)
        if not m.path:
            m.state=m.after_drive;m.reason=''
            # Approach time is not idle loading time. A loaded truck following
            # a new shelf retains its last actual bucket timestamp instead of
            # extending an unproductive wait through repeated repositioning.
            if m.kind=='truck' and m.state=='WAIT_LOAD' and m.cargo_kg<=1e-9:m.load_started=self.sim.time

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

    def _work_plan(self):
        from .conventional_plan import work_geometry
        t=self.terrain
        key=(id(t),id(self.work_mask),id(self.ramp_surface),t.cell,
             tuple(self.ramp['entry_m']),tuple(self.ramp['end_m']),self.config.ramp_width_m,self.excavator.radius)
        cached=getattr(self,'_work_plan_cache',None)
        if cached is None or cached['key']!=key:
            cached=work_geometry(self);cached['key']=key;self._work_plan_cache=cached
            self._work_plan_builds=getattr(self,'_work_plan_builds',0)+1
        return cached

    def _work_remaining(self):
        """Live residual on cached work cells, without a full-map allocation."""
        t=self.terrain;ids=self._work_plan()['ids']
        floor=np.where(self.ramp_mask.ravel()[ids],np.maximum(t.target.ravel()[ids],self.ramp_surface.ravel()[ids]),t.target.ravel()[ids])
        return np.where(t.dig_mask.ravel()[ids],np.maximum(t.height.ravel()[ids]-floor,0.),0.)

    def _road_tolerance(self):
        return min(.12,max(.03,.2*self.terrain.cell))

    def _cut_threshold(self,on_access):
        """Vertical acceptance above the final cut or retained ramp surface."""
        tolerance=self.config.cut_tolerance_m
        if tolerance==0:return 1e-6
        road_tolerance=self._road_tolerance()
        if tolerance<=road_tolerance:return tolerance+1e-9
        return np.where(on_access,road_tolerance,tolerance)+1e-9

    def _actionable_work(self,remaining):
        ids=self._work_plan()['ids']
        on_access=getattr(self,'access_mask',self.ramp_mask).ravel()[ids]
        return remaining>self._cut_threshold(on_access)

    def _dump_plan(self):
        from .conventional_plan import dump_geometry
        t=self.terrain;key=(id(t),id(t._dump_indices),tuple(self.base_position[:2]))
        cached=getattr(self,'_dump_plan_cache',None)
        if cached is None or cached['key']!=key:
            cached=dump_geometry(self);cached['key']=key;self._dump_plan_cache=cached
            self._dump_plan_builds=getattr(self,'_dump_plan_builds',0)+1
        return cached

    def set_dump_order(self,order):
        if order not in ('far_to_near','nearest'):raise ValueError('Неизвестный порядок заполнения отвала')
        if self.config.dump_order!=order:
            self.config.dump_order=order;self._dump_cursors={}
            # Existing routes, their cargo and destination remain committed.
            for m in self.machines:m.retry_route_at=0.

    def plan_report(self):
        p=self._work_plan();ids=p['ids'];t=self.terrain
        access=getattr(self,'access_mask',None)
        sections=(len(np.unique(np.floor(p['along'][access.ravel()[ids]]/max(1.,t.cell)))) if access is not None else 0)
        return dict(policy=self.config.dump_order,work_cells=int(np.count_nonzero(t.dig_mask.ravel()[ids])),access_sections=sections,
                    dump_candidates=len(self._dump_plan()['ids']),
                    work_plan_builds=getattr(self,'_work_plan_builds',0),dump_plan_builds=getattr(self,'_dump_plan_builds',0))

    def _home_route(self,m,validate=False):
        """Retain fine stance connectors instead of rediscovering a coarse grid.

        New ground may invalidate a broad route tile without invalidating the
        actual swept path. Validate that path first, then fall back to A*.
        """
        key=tuple(m.position[:2]);cache=m.budget_cache
        if cache is not None and np.linalg.norm(np.asarray(cache['position'])-m.position[:2])<1e-7:
            token=self._home_route_token(m,cache['bounds'])
            if not validate and cache['token']==token:return cache['path']
            path=cache['path']
            if all(self.segment_safe(a,b,m.radius,m) for a,b in zip([m.position[:2],*path[:-1]],path)):
                cache['token']=token;return path
        path=self.route(m,m.home[:2])
        if path is None:return None
        points=np.array([m.position[:2],*path]);bounds=tuple(np.r_[points.min(axis=0)-m.radius,points.max(axis=0)+m.radius])
        m.budget_cache=dict(position=key,path=path,bounds=bounds,token=self._home_route_token(m,bounds))
        return path

    def _home_route_token(self,m,bounds):
        x0,y0,x1,y1=bounds
        own=tuple((v.id,tuple(v.position),v.radius,v.dimensions_m) for v in self.machines if v is not m and
                  x0-v.radius<=v.position[0]<=x1+v.radius and y0-v.radius<=v.position[1]<=y1+v.radius)
        return (id(self.terrain.height),self.terrain.cell,self.terrain.height.shape,m.radius,m.dimensions_m,
                tuple(self.base_position),own,self._cooperative_token(bounds))

    def _route_seconds(self,m,path):
        return float(sum(max(1,math.ceil(float(np.linalg.norm(np.asarray(b)-np.asarray(a)))/self._speed(m)-1e-12))
                         for a,b in zip([m.position[:2],*path[:-1]],path)))

    def _cooperative_stamp(self,e,bounds=None):
        def relevant(m):
            return (bounds is None or m is e or
                    bounds[0]-m.radius<=m.position[0]<=bounds[2]+m.radius and
                    bounds[1]-m.radius<=m.position[1]<=bounds[3]+m.radius)
        return (id(self.terrain),id(self.terrain.height),self.terrain.cell,self.terrain.height.shape,
                e.id,tuple((m.id,tuple(m.position),tuple(m.home),m.state,m.radius,m.dimensions_m) for m in self.machines if relevant(m)),
                self.config.truck_speed_m_s,self.config.excavator_speed_m_s)

    def _cooperative_token(self,bounds):
        from .conventional_integration import simulations_for
        x0,y0,x1,y1=bounds;actors=[]
        def near(point,margin):return x0-margin<=point[0]<=x1+margin and y0-margin<=point[1]<=y1+margin
        for owner in simulations_for(self.sim):
            for d in owner.drones:
                radius=max(.5,owner.config.fleet.separation/2)
                if d.grounded and near(d.position,radius):actors.append(('drone',d.serial,tuple(d.position),radius))
                if d.trip is not None:
                    for point in (d.trip.dig,d.trip.dump):
                        radius=max(owner.config.fleet.separation,d.config.grab_radius or .5)
                        if near(point,radius):actors.append(('trip',d.serial,tuple(point),radius))
            for robot in (owner.logistics.excavators if owner.logistics else ()):
                if near(robot.position,robot.config.support_half_width_m):actors.append(('robot',tuple(robot.position),robot.config.support_half_width_m))
        return self.terrain.route_token(bounds),tuple(self._other_rectangles(bounds)),tuple(actors)

    def _cached_cooperative_return(self,e,truck=None):
        plan=getattr(self,'_cooperative_return_cache',None)
        if (plan is not None and plan['stamp']==self._cooperative_stamp(e,plan['bounds']) and
                (truck is None or plan['truck_id']==truck.id) and
                plan['token']==self._cooperative_token(plan['bounds'])):return plan
        return None

    def _remember_cooperative_return(self,e,vehicle,truck_path,path):
        points=np.array([e.position[:2],e.home[:2],vehicle.position[:2],vehicle.home[:2],*path,*truck_path])
        margin=max(e.radius,vehicle.radius)+1.
        bounds=tuple(np.r_[points.min(axis=0)-margin,points.max(axis=0)+margin])
        truck_seconds=self._route_seconds(vehicle,truck_path)
        plan=dict(stamp=self._cooperative_stamp(e,bounds),truck_id=vehicle.id,truck_path=list(truck_path),
                  path=list(path),wait_seconds=truck_seconds+2.,truck_seconds=truck_seconds,
                  e_seconds=self._route_seconds(e,path),bounds=bounds)
        plan['token']=self._cooperative_token(bounds);self._cooperative_return_cache=plan
        self.cooperative_return_hints=dict(truck_id=vehicle.id,e_start=tuple(e.position),truck_start=tuple(vehicle.position),
            e_home=tuple(e.home),truck_home=tuple(vehicle.home),path=list(path),truck_path=list(truck_path))
        return plan

    def _checked_cooperative_hint(self,e,truck=None):
        hint=getattr(self,'cooperative_return_hints',None)
        if hint is None:return None
        vehicle=next((m for m in self.machines if m.id==hint['truck_id']),None)
        if vehicle is None or truck is not None and truck is not vehicle:return None
        if truck is None and not (vehicle.state=='WAIT_LOAD' or vehicle.state=='DRIVING' and vehicle.after_drive=='WAIT_LOAD'):return None
        for current,saved in ((e.position,hint['e_start']),(vehicle.position,hint['truck_start']),
                              (e.home,hint['e_home']),(vehicle.home,hint['truck_home'])):
            if not np.allclose(current,saved,rtol=0.,atol=1e-8):return None
        path=hint['path'];truck_path=hint['truck_path']
        for machine,route in ((e,path),(vehicle,truck_path)):
            endpoint=route[-1] if route else machine.position[:2]
            if not np.allclose(endpoint,machine.home[:2],rtol=0.,atol=1e-8):return None
        if not all(self.segment_safe(a,b,vehicle.radius,vehicle) for a,b in zip([vehicle.position[:2],*truck_path[:-1]],truck_path)):return None
        old=vehicle.position.copy()
        try:
            vehicle.position[:]=vehicle.home
            clear=all(self.segment_safe(a,b,e.radius,e) for a,b in zip([e.position[:2],*path[:-1]],path))
        finally:vehicle.position[:]=old;e.budget_cache=None
        return self._remember_cooperative_return(e,vehicle,truck_path,path) if clear else None

    def _cooperative_return(self,e,truck=None,max_routes=8):
        """Budget-only proof: a real truck withdrawal precedes E's return."""
        plan=self._cached_cooperative_return(e,truck)
        if plan is None:plan=self._checked_cooperative_hint(e,truck)
        if plan is None:
            candidates=[truck] if truck is not None else [m for m in self.machines if m.kind=='truck' and
                (m.state=='WAIT_LOAD' or m.state=='DRIVING' and m.after_drive=='WAIT_LOAD') and
                np.linalg.norm(m.position[:2]-e.position[:2])<=self._loading_distance(m)+m.radius]
            started=self.route_calculations
            cursor=0 if truck is not None or not candidates else getattr(e,'cooperative_search_cursor',0)%len(candidates)
            for index in range(cursor,len(candidates)):
                vehicle=candidates[index]
                if self.route_calculations-started>=max_routes:
                    if truck is None:e.cooperative_search_cursor=index
                    return None
                truck_path=self._home_route(vehicle,validate=True)
                if truck_path is None:continue
                if self.route_calculations-started>=max_routes:
                    if truck is None:e.cooperative_search_cursor=index
                    return None
                old=vehicle.position.copy()
                try:
                    vehicle.position[:]=vehicle.home
                    e.budget_cache=None;path=self._home_route(e,validate=True)
                finally:
                    vehicle.position[:]=old;e.budget_cache=None
                if path is None:continue
                plan=self._remember_cooperative_return(e,vehicle,truck_path,path)
                if truck is None:e.cooperative_search_cursor=0
                break
            else:
                if truck is None:e.cooperative_search_cursor=0
        if plan is None:return None
        vehicle=next((m for m in self.machines if m.id==plan['truck_id']),None)
        needed=(plan['truck_seconds']+15)*self.config.truck_drive_l_h/3600+self._tank(vehicle)*self.config.fuel_reserve_fraction if vehicle else math.inf
        return plan if vehicle is not None and vehicle.fuel_l>=needed else None

    def _return_budget(self,m,extra_s=0.):
        started=self.route_calculations
        plan=self._cached_cooperative_return(m) if m.kind=='excavator' else None
        if plan is None and m.kind=='excavator':plan=self._checked_cooperative_hint(m)
        path=self._home_route(m) if plan is None else None
        if path is None and m.kind=='excavator':plan=self._cooperative_return(m,max_routes=max(0,8-(self.route_calculations-started)))
        rate=max(self.config.excavator_work_l_h,self.config.idle_l_h) if m.kind=='excavator' else self.config.truck_drive_l_h
        if plan is not None:
            seconds=plan['wait_seconds']+plan['e_seconds']+extra_s+15
            fuel=((plan['e_seconds']+extra_s+15)*rate+plan['wait_seconds']*self.config.idle_l_h)/3600+self._tank(m)*self.config.fuel_reserve_fraction
            return seconds,fuel,plan['path']
        if path is None:return None
        seconds=self._path_length(m,path)/self._speed(m)+extra_s+15
        fuel=seconds*rate/3600+self._tank(m)*self.config.fuel_reserve_fraction
        return seconds,fuel,path

    def _retreat_loading_truck(self,e,plan,return_requested=False):
        vehicle=next(m for m in self.machines if m.id==plan['truck_id'])
        if not self._assign_route(vehicle,vehicle.home[:2],'PARKED',outbound_path=plan['truck_path']):return False
        vehicle.loading_hold_for=e.id;vehicle.loading_hold_since=self.sim.time
        e.wait_for_truck=vehicle.id;e.return_requested=bool(return_requested)
        e.state='WAIT_TRUCK';e.retry_route_at=0.
        e.reason='Ожидание фактического освобождения съезда самосвалом'
        self._cooperative_return_cache=None
        return True

    def _prepare_productive_relocation(self,e,goal,max_routes=8):
        """Prove the joint move, then move only the truck until it is home."""
        if max_routes<3:return False
        started=self.route_calculations;plan=self._cooperative_return(e,max_routes=max_routes-1)
        if plan is None or self.route_calculations-started+1>max_routes:return False
        vehicle=next(m for m in self.machines if m.id==plan['truck_id']);old=vehicle.position.copy()
        try:
            vehicle.position[:]=vehicle.home
            path=self.route(e,goal)
            if path is None:return False
            visited=[tuple(e.position[:2]),*path];home=[*reversed(visited[:-1]),*plan['path']]
            if not all(self.segment_safe(a,b,e.radius,e) for a,b in zip([goal,*home[:-1]],home)):return False
            out_seconds=self._route_seconds(e,path)
            position=e.position.copy()
            try:
                e.position[:2]=goal;e.position[2]=self.terrain.sample(*goal)
                back_seconds=self._route_seconds(e,home)
            finally:e.position[:]=position
        finally:vehicle.position[:]=old;e.budget_cache=None
        seconds=plan['wait_seconds']+out_seconds+back_seconds+max(60.,self.config.bucket_cycle_s)+45.
        fuel=seconds*max(self._rate(e,True),self.config.truck_drive_l_h)/3600+self._tank(e)*self.config.fuel_reserve_fraction
        if self.shift()['seconds_left']<seconds or e.fuel_l<fuel:return False
        if not self._retreat_loading_truck(e,plan):return False
        e.pending_relocation=dict(goal=tuple(map(float,goal)),requested_at=self.sim.time)
        return True

    def _go_home(self,m):
        if m.kind=='excavator':
            m.expose_target=None;m.return_requested=True;m.pending_relocation=None
        if np.linalg.norm(m.position[:2]-m.home[:2])<.1:
            m.state='PARKED';m.path=[];m.reason=''
            if m.kind=='excavator':m.return_requested=False
            return True
        path=self._home_route(m,validate=True)
        if path is None:
            plan=self._cooperative_return(m) if m.kind=='excavator' else None
            if plan is not None and self._retreat_loading_truck(m,plan,return_requested=True):return True
            m.reason=getattr(self,'_route_failure','') or 'Нет безопасного пути на базу';m.retry_route_at=self.sim.time+30;return False
        return self._assign_route(m,m.home[:2],'PARKED',outbound_path=path)

    def _road_band(self,e,plan,remaining,unfinished):
        """Pose-independent road cells and first station for one planning pass."""
        access=getattr(self,'access_mask',None)
        if access is not None:
            pending=np.flatnonzero(access.ravel()[plan['ids']] & (remaining>self._road_tolerance()+1e-9))
            if len(pending):
                along=plan['along'][pending]
                first=float(along.min())
                width=max(1.,self.terrain.cell,min(e.radius,self.config.excavator_reach_m/2))
                return plan['ids'][pending[along<=first+width]],first
        return unfinished,None

    def _target_indices(self,e,remaining=None,focus_access=True,*,eligible_ids=None):
        """Current reachable design centres, retaining the first road section."""
        t=self.terrain
        plan=self._work_plan();work_ids=plan['ids']
        if remaining is None:remaining=self._work_remaining()
        elif remaining.shape==t.height.shape:remaining=remaining.ravel()[work_ids]
        ids=eligible_ids
        if ids is None:
            ids=work_ids[self._actionable_work(remaining)]
            if focus_access:ids,_=self._road_band(e,plan,remaining,ids)
        if not len(ids):return ids,remaining
        distance=(t.x.ravel()[ids]-e.position[0])**2+(t.y.ravel()[ids]-e.position[1])**2
        valid=(distance<=self.config.excavator_reach_m**2)&(distance>=(e.radius+1.)**2)
        ids=ids[valid]
        return ids,remaining

    def target_yield(self,e,target=None):
        """Next bucket's actual geometric volume, without changing the soil."""
        target=self._available_target(e) if target is None else target
        if target is None:return 0.
        t=self.terrain;x,y=target[:2];radius=max(t.cell*.71,.5)
        patch=t.patch(x,y,radius);floor,allowed=self._bucket_geometry(e,patch)
        footprint=(t.x[patch]-x)**2+(t.y[patch]-y)**2<=radius**2
        volume=float(np.where(allowed&footprint,np.maximum(t.height[patch]-floor,0.),0.).sum()*t.area)
        return min(self.config.bucket_m3,volume)

    def available_work_at_stance(self,e,remaining=None,focus_access=True,*,eligible_ids=None):
        """Distinct removable volume in the actionable road band, not future cuts.

        A broad reachable annulus may contain much soil behind an unfinished
        road section. Counting that future soil would favour a stance that can
        currently take only crumbs. All floors/support guards are the same as
        an actual bucket's, and no capacity is reserved or soil transferred.
        """
        ids,_=self._target_indices(e,remaining,focus_access,eligible_ids=eligible_ids)
        if not len(ids):return 0.
        t=self.terrain;n=t.height.shape[1]
        patch=t.patch(*e.position[:2],self.config.excavator_reach_m)
        floor,allowed=self._bucket_geometry(e,patch)
        rows=ids//n-patch[0].start;cols=ids%n-patch[1].start
        volume=np.where(allowed[rows,cols],np.maximum(t.height[patch][rows,cols]-floor[rows,cols],0.),0.)
        return float(volume.sum()*t.area)

    def _available_target(self,e,remaining=None,focus_access=True,*,eligible_ids=None):
        t=self.terrain
        committed=getattr(e,'expose_target',None)
        if committed is not None and focus_access:
            x,y=committed;row,col=int(y/t.cell),int(x/t.cell)
            patch=t.patch(x,y,max(t.cell*.71,.5));floor,allowed=self._bucket_geometry(e,patch)
            i,j=row-patch[0].start,col-patch[1].start
            if np.linalg.norm(e.position[:2]-[x,y])<=self.config.excavator_reach_m:
                target=(x,y,float(floor[i,j]))
                if getattr(e,'side_bucket',False):
                    # A side bite can reach a remaining cell with the edge of
                    # its scoop even when the centre is at the depth limit.
                    if self.target_yield(e,target)>1e-12:return target
                elif allowed[i,j] and t.height[row,col]>floor[i,j]+1e-6:
                    return target
            # New terrain or a parked truck can close a formerly useful bite.
            # A trial pose must never cancel the live excavator's commitment.
            if remaining is None:
                e.expose_target=None
                e.side_bucket=False
        ids,remaining=self._target_indices(e,remaining,focus_access,eligible_ids=eligible_ids)
        if not len(ids):return None
        plan=self._work_plan();work_ids=plan['ids']
        score=(t.x.ravel()[ids]-e.position[0])**2+(t.y.ravel()[ids]-e.position[1])**2
        if self.ramp.get('model')=='supported_access_v2':
            selected=np.searchsorted(work_ids,ids)
            along=plan['along'][selected];lateral=plan['lateral'][selected]
            ids=ids[np.lexsort((score,along,lateral>self.config.ramp_width_m/2))]
        else:ids=ids[np.argsort(score,kind='stable')]

        # Evaluate the physical support and depth floor once for this pose.
        # Every centre in reach shares these same cells, including the rim of
        # its bucket. Padding the local rectangle treats off-map cells as empty.
        radius=max(t.cell*.71,.5)
        patch=t.patch(*e.position[:2],self.config.excavator_reach_m+radius)
        floor,allowed=self._bucket_geometry(e,patch)
        removable=np.where(allowed,np.maximum(t.height[patch]-floor,0.),0.)
        span=int(math.floor(radius/t.cell))
        depth=np.pad(removable,span) if span else removable
        rows=ids//t.height.shape[1]-patch[0].start
        cols=ids%t.height.shape[1]-patch[1].start
        volume=np.zeros(len(ids))
        for dy in range(-span,span+1):
            for dx in range(-span,span+1):
                if (dx*t.cell)**2+(dy*t.cell)**2<=radius**2:
                    volume+=depth[rows+span+dy,cols+span+dx]
        volume*=t.area

        # Keep the preferred approach order for ordinary productive scoops.
        # If the leading block cannot provide a useful bite, include all the
        # remaining cells in reach before declaring the stance exhausted.
        limit=min(128,len(ids))
        minimum=min(.005,self.config.bucket_m3*.005,max(1e-12,float(remaining.sum()*t.area)*.001))
        if self.config.cut_tolerance_m>0:
            # A support-limited slice can be millimetres deep even on a cell
            # far above grade. Prefer a productive scoop elsewhere in reach.
            minimum=max(minimum,.15*self.config.bucket_m3)
        if volume[:limit].max()<=minimum+1e-10:
            limit=len(ids)
        volume=volume[:limit]
        full=np.flatnonzero(volume>=self.config.bucket_m3)
        best=int(full[0]) if len(full) else int(np.argmax(volume))
        if volume[best]<=1e-12:return None
        index=ids[best]
        return float(t.x.ravel()[index]),float(t.y.ravel()[index]),float(floor[rows[best],cols[best]])

    def _bucket_geometry(self,e,patch):
        """One effective floor/support law for selection and actual extraction."""
        t=self.terrain;c=self.config
        grade=np.where(self.ramp_mask[patch],np.maximum(t.target[patch],self.ramp_surface[patch]),t.target[patch])
        floor=np.maximum(grade,e.position[2]-c.excavator_dig_depth_m)
        allowed=(self.work_mask[patch]&t.dig_mask[patch]).copy()
        if c.cut_tolerance_m>0:
            access=getattr(self,'access_mask',self.ramp_mask)
            allowed&=(t.height[patch]-grade)>self._cut_threshold(access[patch])
        for machine in self.machines:
            distance=np.hypot(t.x[patch]-machine.position[0],t.y[patch]-machine.position[1])
            floor=np.maximum(floor,machine.position[2]-t.dump_slope*np.maximum(distance-machine.radius-.5,0.))
            allowed&=distance>machine.radius+1.
        return floor,allowed

    def _obstructed_access_target(self,e):
        """The earliest road bump hidden inside the current track safety guard."""
        access=getattr(self,'access_mask',None)
        if access is None:return None
        t=self.terrain;plan=self._work_plan();remaining=self._work_remaining()
        positions=np.flatnonzero(access.ravel()[plan['ids']]&(remaining>self._road_tolerance()+1e-9))
        ids=plan['ids'][positions]
        if not len(ids):return None
        along=plan['along'][positions]
        distance=np.hypot(t.x.ravel()[ids]-e.position[0],t.y.ravel()[ids]-e.position[1])
        ids=ids[(along<=float(along.min())+max(1.,t.cell))&(distance<e.radius+1.)]
        if not len(ids):return None
        index=ids[int(np.argmax(remaining[np.searchsorted(plan['ids'],ids)]))]
        return float(t.x.ravel()[index]),float(t.y.ravel()[index])

    def _choose_excavator_site(self,e,advance_only=False,expose=None,productive_only=False):
        target=self._available_target(e)
        current_yield=self.target_yield(e,target) if productive_only and target is not None else 0.
        if target is not None and not advance_only and expose is None and not productive_only:
            e.target=target;e.state='WAIT_TRUCK';e.reason='';self.reason=''
            e.site_search_cursor=0;e.site_search_stage=0
            return True
        t=self.terrain;plan=self._work_plan();remaining=self._work_remaining()
        actionable=self._actionable_work(remaining);ids=plan['ids'][actionable]
        if not len(ids):
            if productive_only:return False
            self._work_complete=True;self._completed_revision=t.revision
            self.reason='Захватка обработана в пределах допуска; выездная рампа сохранена'
            return False
        routes_started=self.route_calculations
        current_escape=self._home_route(e,validate=True)
        if current_escape is None and not productive_only:
            if self._release_loading_bay(e):
                e.reason='Ожидание освобождения однополосного съезда самосвалом'
                e.retry_route_at=self.sim.time+5;e.next_plan_at=self.sim.time+5
            elif not advance_only:e.reason='Нет безопасного выезда от текущей полки'
            return False
        # Trial positions share the same terrain snapshot. Their reach,
        # support, bucket depth and route are still checked independently.
        focused_ids,first_station=self._road_band(e,plan,remaining,ids)
        order=np.argsort((t.x.ravel()[ids]-e.position[0])**2+(t.y.ravel()[ids]-e.position[1])**2,kind='stable')
        samples=ids[order[::max(1,len(order)//8)][:8]]
        # Fine two-dimensional repositioning exposes critical cells just beyond
        # the track exclusion. A centreline-only route can strand a shoulder
        # cell forever even though a half-metre sideways stance reaches it.
        step=t.cell;span=max(1,int(math.ceil(1.5/step)))
        candidates=sorted([e.position[:2]+[dx*step,dy*step]
            for dx in range(-span,span+1) for dy in range(-span,span+1) if dx or dy],
            key=lambda p:(t.sample(*p),float(np.sum((p-e.position[:2])**2))))
        # A road is built by actual bucket cuts before the next support-sized
        # section becomes passable. Prefer useful reachable lower road stations
        # over repeatedly sampling radial positions on an uncut high bank.
        entry=np.asarray(self.ramp['entry_m']);end=np.asarray(self.ramp['end_m'])
        delta=end-entry;length=float(np.linalg.norm(delta));unit=delta/max(length,1e-9)
        if length>0:
            # A half-metre backshift can expose the row just beyond the track
            # guard. Coarse two-metre stations can strand that row underneath
            # the next footprint despite there being a safe intermediate stance.
            stations=plan['stations'];points=plan['points']
            order=np.argsort(np.sum((points-e.position[:2])**2,axis=1),kind='stable')
            side=plan['side']
            chosen=list(order[:24])
            # Work starts at the first unfinished access cross-section, which
            # can be far from a camp beside the middle of a long ramp. Nearby
            # parking stations alone cannot expose that section. Retain both
            # local repositioning and stations at the actual excavation front.
            if first_station is not None:
                chosen.extend(np.argsort(np.abs(stations-first_station),kind='stable')[:24])
            road_points=points[list(dict.fromkeys(chosen))]
            candidates.extend(sorted([p+offset for p in road_points for offset in (side,-side,np.zeros(2))],
                key=lambda p:(t.sample(*p),float(np.sum((p-e.position[:2])**2)))))
            # Include the very approach already checked by automatic siting.
            # It must still pass current body, terrain, route and return tests.
            candidates.insert(0,entry-unit*(e.radius+t.cell))
        for index in samples:
            x,y=t.x.ravel()[index],t.y.ravel()[index]
            for dx,dy in ((-1,0),(0,-1),(1,0),(0,1)):
                candidates.append(np.array([x+dx*self.config.excavator_reach_m*.8,y+dy*self.config.excavator_reach_m*.8]))
        productive_targets={}
        if productive_only:
            # Rank all cheap geometric candidates before spending the bounded
            # route budget. A move must pay for a material improvement, rather
            # than alternate between equivalent centimetre-scale leftovers.
            minimum=max(current_yield*2.,current_yield+self.config.bucket_m3*.1)
            ranked=[];seen=set()
            for p in candidates:
                key=tuple(map(float,p))
                if key in seen:continue
                seen.add(key)
                if np.linalg.norm(p-e.position[:2])<max(.5,t.cell) or not self.passable(p,e.radius,e):continue
                old=e.position.copy();old_expose=e.expose_target
                e.position[:2]=p;e.position[2]=t.sample(*p)
                try:
                    useful=self._available_target(e,remaining,eligible_ids=focused_ids)
                    volume=self.target_yield(e,useful) if useful is not None else 0.
                    if volume+1e-10<minimum:continue
                    batch=self.available_work_at_stance(e,remaining,eligible_ids=focused_ids)
                finally:e.position[:]=old;e.expose_target=old_expose
                productive_targets[key]=useful
                ranked.append(((-volume,-batch,float(np.linalg.norm(p-e.position[:2]))),p))
            if not ranked:return False
            candidates=[p for _,p in sorted(ranked,key=lambda row:row[0])]
        ordinary=not (productive_only or advance_only or expose is not None)
        cursor_field='productive_search_cursor' if productive_only else 'advance_search_cursor' if advance_only or expose is not None else 'site_search_cursor'
        cursor=getattr(e,cursor_field,0)
        def search_order():
            if not ordinary:
                first=cursor%len(candidates)
                for offset in range(len(candidates)):
                    index=(first+offset)%len(candidates)
                    yield index,candidates[index],False
                return
            for index in range(cursor,len(candidates)):
                yield index,candidates[index],False
            # After the fast ramp-first search, check for work on the flanks.
            # Try reaching it from the current tracks before asking for a move.
            if cursor<=len(candidates):
                yield len(candidates),e.position[:2].copy(),True
            from .conventional_plan import shelf_candidates
            scan_start=len(candidates)+1
            for index,point in shelf_candidates(t,ids,self.config.excavator_reach_m,max(0,cursor-scan_start)):
                yield scan_start+index,point,True

        limited=False;dispatch_wait='';sampled=0
        stage=getattr(e,'site_search_stage',0) if cursor and ordinary else 0
        for index,p,expanded in search_order():
            if expanded:
                if sampled>=48:
                    setattr(e,cursor_field,index);limited=True;break
                sampled+=1
            if advance_only and t.sample(*p)>=e.position[2]-.05:continue
            same_pose=expanded and index==len(candidates)
            minimum_move=1e-7 if expanded else max(.5,t.cell)
            if not same_pose and np.linalg.norm(p-e.position[:2])<minimum_move:continue
            if not self.passable(p,e.radius,e):continue
            stage=max(stage,1)
            old=e.position.copy();e.position[:2]=p;e.position[2]=t.sample(*p)
            side_target=False
            try:
                if productive_only:useful=productive_targets[tuple(map(float,p))]
                elif expose is None:
                    useful=self._available_target(e,remaining,focus_access=not advance_only,
                                                  eligible_ids=ids if advance_only else focused_ids)
                    if useful is None and expanded:
                        useful=self._available_target(e,remaining,focus_access=False,eligible_ids=ids)
                        side_target=useful is not None
                else:
                    patch=t.patch(*expose,max(t.cell*.71,.5));floor,allowed=self._bucket_geometry(e,patch)
                    row=int(expose[1]/t.cell)-patch[0].start;col=int(expose[0]/t.cell)-patch[1].start
                    goal=max(t.target[patch][row,col],self.ramp_surface[patch][row,col] if self.ramp_mask[patch][row,col] else -math.inf)
                    reachable=(np.linalg.norm(p-expose)<=self.config.excavator_reach_m and allowed[row,col]
                               and floor[row,col]<=goal+min(.12,max(.03,.2*t.cell))
                               and t.height[patch][row,col]>floor[row,col]+1e-5)
                    useful=(*expose,float(floor[row,col])) if reachable else None
            finally:e.position[:]=old
            if useful is None:continue
            stage=max(stage,2)
            if same_pose:
                e.side_bucket=side_target
                e.expose_target=tuple(useful[:2]) if side_target else None
                e.target=useful;e.state='WAIT_TRUCK';e.reason='';self.reason=''
                setattr(e,cursor_field,0)
                return True
            if self.route_calculations-routes_started>=8:
                setattr(e,cursor_field,index);limited=True;break
            if productive_only and current_escape is None:
                # A loading truck may occupy the only road. Its partial load
                # must not be sent away merely to investigate a tiny target:
                # reach this branch only after a material gain was ranked.
                # The cooperative helper proves both routes and queues only
                # the truck's physical retreat. E stays still until it clears.
                if self._prepare_productive_relocation(e,p,max_routes=8-(self.route_calculations-routes_started)):
                    setattr(e,cursor_field,0);self.reason='';return True
                continue
            outbound=self.route(e,p)
            if outbound is None:
                if productive_only and self.route_calculations-routes_started<8:
                    if self._prepare_productive_relocation(e,p,max_routes=8-(self.route_calculations-routes_started)):
                        setattr(e,cursor_field,0);self.reason='';return True
                continue
            # Footprint and absolute-grade checks are symmetric. Reversing the
            # validated outbound path then following the already checked home
            # path gives a real safe escape, without another A* per candidate.
            points=[tuple(e.position[:2]),*outbound]
            escape=[*reversed(points[:-1]),*current_escape]
            if self._assign_route(e,p,'WAIT_TRUCK',home_path=escape,outbound_path=outbound):
                setattr(e,cursor_field,0)
                e.side_bucket=side_target
                e.expose_target=tuple(expose) if expose is not None else None
                if side_target:e.expose_target=tuple(useful[:2])
                e.target=useful if side_target or expose is not None else None
                self.reason=''
                return True
            if e.reason.startswith(('Недостаточно топлива','Недостаточно времени')):dispatch_wait=e.reason
        if not limited:setattr(e,cursor_field,0)
        if advance_only or expose is not None or productive_only:return False
        e.site_search_stage=stage if limited else 0
        if limited:
            message='Лимит поиска за один шаг; продолжается проверка рабочих позиций'
        elif stage==0:
            message='Рабочая позиция не найдена: грунт или препятствия не дают разместить гусеницы'
        elif stage==1:
            message='Рабочая позиция не найдена: с проверенных опор ковш не достаёт до доступного грунта'
        else:
            message='Рабочая позиция не найдена: до проверенных мест не найден безопасный подъезд с возвратом'
        self.reason=dispatch_wait or f'{message}. С учётом съезда и допуска осталось вынуть {float(remaining[actionable].sum()*t.area):.3f} м³'
        e.reason=self.reason;return False

    def _loading_point(self,truck,*,clear_exit=False,excavator_escape=None):
        e=self.excavator
        if self.sim.time<truck.retry_route_at:return False
        routes_started=self.route_calculations
        escape=excavator_escape if excavator_escape is not None else self._home_route(e,validate=True)
        if escape is None:
            truck.reason='Нет проверенного выезда экскаватора для места погрузки'
            truck.retry_route_at=self.sim.time+30;return False
        # A truck parks beside the counterweight, not on the cell being cut.
        distance=self._loading_distance(truck)
        candidates=[e.position[:2]+distance*np.array([math.cos(a),math.sin(a)]) for a in np.arange(8)*math.pi/4]
        candidates.sort(key=lambda p:(float(np.linalg.norm(p-truck.position[:2])),tuple(p)))
        cursor=truck.site_search_cursor%len(candidates)
        for index in range(cursor,len(candidates)):
            p=candidates[index]
            if clear_exit and np.linalg.norm(p-truck.position[:2])<max(.5,self.terrain.cell):continue
            if e.target is not None and np.linalg.norm(p-np.array(e.target[:2]))<truck.radius+1.:continue
            if not self.passable(p,truck.radius,truck):continue
            old=truck.position.copy()
            exit_limited=False
            try:
                truck.position[:2]=p;truck.position[2]=self.terrain.sample(*p)
                exit_clear=all(self.segment_safe(a,b,e.radius,e)
                    for a,b in zip([e.position[:2],*escape[:-1]],escape))
                if not exit_clear:
                    # A single-lane bay need not leave simultaneous room for
                    # both machines: the truck must have a real withdrawal,
                    # followed by a checked E route after it reaches home.
                    if self.route_calculations-routes_started+5>8:exit_limited=True
                    else:
                        exit_clear=self._cooperative_return(e,truck,max_routes=8-(self.route_calculations-routes_started)-3) is not None
                        if not exit_clear and self.route_calculations-routes_started+4<=8:
                            exit_clear=self._home_route(e,validate=True) is not None
            finally:
                truck.position[:]=old
                # The probe describes a future bay, not the current obstacle.
                e.budget_cache=None
            if exit_limited:
                truck.site_search_cursor=index;truck.retry_route_at=self.sim.time+30
                truck.reason='Лимит поиска места погрузки; следующие места проверяются после ожидания'
                return False
            if not exit_clear:continue
            # _assign_route can require outbound, present return and fallback
            # return searches. Keep the existing eight-search ceiling and
            # resume the remaining circle after a bounded wait.
            if self.route_calculations-routes_started+3>8:
                truck.site_search_cursor=index;truck.retry_route_at=self.sim.time+30
                truck.reason='Лимит поиска места погрузки; следующие места проверяются после ожидания'
                return False
            if self._assign_route(truck,p,'WAIT_LOAD'):
                truck.site_search_cursor=0;return True
        truck.site_search_cursor=0
        truck.reason='Нет безопасного места погрузки с выездом экскаватора; груз сохранён'
        truck.retry_route_at=self.sim.time+30
        return False

    def _loading_distance(self,truck):return max(8.,self.excavator.radius+truck.radius+.5)

    def _release_loading_bay(self,e):
        """Withdraw the actual truck without converting its partial load to a tip."""
        plan=self._cooperative_return(e)
        return plan is not None and self._retreat_loading_truck(e,plan,getattr(e,'return_requested',False))

    def _dump_point(self,truck):
        if self.sim.time<truck.retry_route_at:return None
        t=self.terrain;volume=cargo_volume(truck.cargo_kg,truck.soil,t.config.terrain.density)
        fixed=self.config.dump_order=='far_to_near'
        ids=self._dump_plan()['ids'] if fixed else t._dump_indices
        available=np.maximum(t.dump_cap.ravel()[ids]-t.height.ravel()[ids],0.)
        free=(available>1e-8)&t.dump_mask.ravel()[ids]
        if not np.any(free):
            truck.retry_route_at=self.sim.time+30;truck.reason='Отвал заполнен; груз сохранён';return None
        obstacles=self._dump_obstacles()
        cursors=getattr(self,'_dump_cursors',{});self._dump_cursors=cursors
        if fixed:
            from .conventional_plan import far_dump_candidates
            candidates=far_dump_candidates(self,free,cursors.get(truck.id,0))
        else:
            # Compatibility policy: choose near the truck as older saves did.
            ids=ids[free]
            distance=(t.x.ravel()[ids]-truck.position[0])**2+(t.y.ravel()[ids]-truck.position[1])**2
            points=[(float(t.x.ravel()[i]),float(t.y.ravel()[i]))
                    for i in ids[np.argsort(distance,kind='stable')[::max(1,len(ids)//32)][:32]]]
            values=[(x,y,dx,dy) for x,y in points for dx,dy in ((-1,0),(0,-1),(1,0),(0,1))]
            cursor=cursors.get(truck.id,0)%len(values)
            candidates=(((cursor+offset)%len(values),value) for offset,value in enumerate(values[cursor:]+values[:cursor]))
        routes_started=self.route_calculations;limited=False
        for checked,(candidate,(x,y,dx,dy)) in enumerate(candidates):
            if fixed and checked>=128:
                cursors[truck.id]=candidate;limited=True;break
            distance=truck.dimensions_m[0]/2+2.5
            p=np.array([x+dx*distance,y+dy*distance])
            if fixed and any(other is not truck and other.cargo_kg>0. and other.target is not None and
                (other.state=='UNLOADING' or (other.state=='DRIVING' and other.after_drive=='UNLOADING')) and
                (np.linalg.norm(np.asarray(other.target)-[x,y])<4. or
                 (other.destination is not None and np.linalg.norm(np.asarray(other.destination)-p)<other.radius+truck.radius+.25))
                for other in self.machines):continue
            if not self.passable(p,truck.radius,truck):continue
            minimum=min(volume,max(.25,volume*.15))
            # Cheap local upper bound first; the expensive road search only
            # runs for a tip which can actually accept a meaningful amount.
            _,upper=self._dump_capacity(truck,(x,y),p,[],obstacles)
            if upper.sum()*t.area<minimum-1e-9:continue
            if self.route_calculations-routes_started+2>8:
                cursors[truck.id]=candidate;limited=True;break
            escape=self._escape_from(truck,p)
            if escape is None:continue
            _,capacity=self._dump_capacity(truck,(x,y),p,escape,obstacles)
            if capacity.sum()*t.area<minimum-1e-9:continue
            if self._assign_route(truck,p,'UNLOADING',home_path=escape):
                cursors[truck.id]=0;truck.target=(x,y);truck.timer=30.;return p
        if not limited:cursors[truck.id]=0
        truck.retry_route_at=self.sim.time+30
        truck.reason=('Лимит поиска разгрузки; следующие варианты проверяются после ожидания; груз сохранён' if limited else
                      'Нет безопасной точки разгрузки с выездом; груз сохранён')
        return None

    def _escape_from(self,m,point):
        old=m.position.copy()
        try:
            m.position[:2]=point;m.position[2]=self.terrain.sample(*point)
            return self.route(m,m.home[:2])
        finally:m.position[:]=old

    def _dump_obstacles(self):
        from .conventional_integration import simulations_for
        from .site_ground import equipment_mask
        owners=simulations_for(self.sim)
        return equipment_mask(self.terrain,owners) if any(owner.drones for owner in owners) else False

    def _dump_capacity(self,truck,target,point,escape,obstacles=None):
        """Only actual local room that cannot avalanche onto wheels or egress.

        The repose cone anchored along each support/road contour caps height,
        rather than merely masking cells under the current vehicle. This keeps
        the deposit and its subsequent conservative relaxation below the road's
        stable envelope, without a second material ledger or speculative add.
        """
        t=self.terrain;radius=2.;patch=t.patch(*target,radius)
        xx,yy=t.x[patch],t.y[patch];height=t.height[patch]
        capacity=np.where(t.dump_mask[patch]&((xx-target[0])**2+(yy-target[1])**2<=radius**2),
                          np.maximum(t.dump_cap[patch]-height,0.),0.)
        def support(position,r):
            distance=np.hypot(xx-position[0],yy-position[1])
            ceiling=position[2]+t.dump_slope*np.maximum(distance-r-.75,0.)
            np.minimum(capacity,np.maximum(ceiling-height,0.),out=capacity)
            capacity[distance<=r+.75]=0.
        for m in self.machines:
            support(np.r_[point,t.sample(*point)] if m is truck else m.position,m.radius)
            support(m.home,m.radius)
        support(self.base_position,math.hypot(4.,4.))
        # Protect the complete swept return corridor, not just its vertices.
        road=[np.asarray(point),*[np.asarray(p) for p in escape]]
        for a,b in zip(road,road[1:]):
            delta=b-a;denom=float(delta@delta)
            fraction=np.clip(((xx-a[0])*delta[0]+(yy-a[1])*delta[1])/max(denom,1e-12),0.,1.)
            distance=np.hypot(xx-a[0]-fraction*delta[0],yy-a[1]-fraction*delta[1])
            z=min(float(t.sample(*a)),float(t.sample(*b)))
            ceiling=z+t.dump_slope*np.maximum(distance-truck.radius-.75,0.)
            np.minimum(capacity,np.maximum(ceiling-height,0.),out=capacity)
            capacity[distance<=truck.radius+.75]=0.
        if obstacles is None:obstacles=self._dump_obstacles()
        if obstacles is not False:capacity[obstacles[patch]]=0.
        return patch,capacity

    def _unload(self,m):
        """Commit one bounded safe tip; all unplaced mass remains in the body."""
        if m.cargo_kg==0.:
            if m.soil is not None:m.soil.validate(0.)
            m.soil=None;m.target=None;m.state='PARKED';return True
        t=self.terrain;escape=self._home_route(m,validate=True)
        if escape is None:
            m.reason='Нет безопасного выезда перед разгрузкой; груз сохранён';m.retry_route_at=self.sim.time+30;return False
        patch,capacity=self._dump_capacity(m,m.target,m.position[:2],escape)
        volume=cargo_volume(m.cargo_kg,m.soil,t.config.terrain.density)
        placed=min(volume,float(capacity.sum()*t.area))
        if (volume<.25 and placed<volume) or (volume>=.25 and placed<.25-1e-9):
            m.reason='Для остатка груза нужна другая безопасная точка разгрузки';return False
        if placed<=0.:
            m.reason='Нет безопасного места для разгрузки вне колёс и выезда; груз сохранён';return False
        fraction=min(1.,placed/volume);mass=m.cargo_kg*fraction
        load=m.soil.scaled(fraction) if m.soil is not None else None
        t._deposit_capacity(*m.target,placed,capacity,radius=2.,load=load,bounds=patch)
        if m.soil is not None:m.soil.remove(mass);self.delivered_materials+=load.masses
        m.cargo_kg=max(0.,m.cargo_kg-mass)
        self.delivered_mass_kg+=mass;self.delivered_m3+=placed;m.delivered_m3+=placed
        if fraction==1.:m.cargo_kg=0.;m.soil=None
        m.target=None;m.state='PARKED';m.reason=''
        self._terrain_changed()
        return True

    def _cut(self,e,truck):
        target=self._available_target(e)
        if target is None:return False
        x,y=target[:2];t=self.terrain;c=self.config
        radius=max(t.cell*.71,.5);patch=t.patch(x,y,radius)
        # Terrain already supports a per-cell lower surface. Keep every native
        # project cell and each ramp cell above its own envelope, not merely
        # above the centre cell's elevation.
        floor,allowed=self._bucket_geometry(e,patch)
        # The existing floor parameter is local to the bucket footprint.
        # Leaving protected cells at their current height gives zero removable
        # volume there, without allocating a mask the size of the whole map.
        floor=np.where(allowed,floor,t.height[patch])
        available_volume=min(c.bucket_m3,c.truck_body_m3-cargo_volume(truck.cargo_kg,truck.soil,t.config.terrain.density))
        mass_limit=c.truck_payload_kg-truck.cargo_kg
        if min(available_volume,mass_limit)<=1e-9:return False
        if t.soil is not None:
            load=t.scoop_load(x,y,mass_limit,source='conventional',radius=radius,max_volume=available_volume,minimum_height=floor)
            mass=load.mass
            if mass>0:
                if truck.soil is None:truck.soil=SoilLoad.empty(t.soil.profile)
                truck.soil.add(load);self.extracted_materials+=load.masses
            volume=load.volume
        else:
            mass=t.scoop(x,y,min(mass_limit,available_volume*t.config.terrain.density),source='conventional',radius=radius,minimum_height=floor)
            volume=mass/t.config.terrain.density
        if mass>0:
            self._terrain_changed()
            truck.load_started=self.sim.time
        truck.cargo_kg+=mass;self.excavated_mass_kg+=mass;self.excavated_m3+=volume;e.excavated_m3+=volume
        e.target=target
        return mass>1e-9

    def _ground_released(self,offset=(0.,0.)):
        if getattr(self,'_ground_stage_released',False):
            return True
        # Older checkpoints record the handoff in the shared terrain mask.
        # It covers the entire rectangle or none of it. Sample one interior cell.
        mask=getattr(self.terrain,'_conventional_mask',None)
        if mask is None or self.sim.shared_site is None:
            return False
        x,y,w,h=self.config.work_rect
        row=int((y+h/2-offset[1])/self.terrain.cell);col=int((x+w/2-offset[0])/self.terrain.cell)
        return not bool(mask[row,col])

    def _recheck_finished_work(self):
        if not self._work_complete or getattr(self,'_completed_revision',None)==self.terrain.revision:
            return
        if self._ground_released() or self.carried_mass_kg>1e-8:
            return
        # A late avalanche can refill an accepted cell. Finish the committed
        # haul before reopening, and leave any machine already en route alone.
        self._completed_revision=self.terrain.revision
        if np.any(self._actionable_work(self._work_remaining())):
            self._work_complete=False
            self.reason='На захватке снова есть грунт выше допуска'
            self.excavator.next_plan_at=0.

    def set_cut_tolerance(self,value):
        from dataclasses import replace
        replace(self.config,cut_tolerance_m=value).validate()
        if value==self.config.cut_tolerance_m:
            return
        if self._ground_released():
            raise ValueError('Наземный этап завершён: участок уже передан дронам')
        self.config.cut_tolerance_m=value
        self._completed_revision=None
        self._recheck_finished_work()
        self.excavator.site_search_cursor=0;self.excavator.site_search_stage=0
        if self.excavator.state!='DRIVING':
            self.excavator.next_plan_at=0.

    @property
    def ground_stage_drained(self):
        """A parked finished crew can hand its ramp/buffer to the aircraft."""
        self._recheck_finished_work()
        return bool(self._work_complete and self.carried_mass_kg<=1e-8 and
                    all(m.state in ('PARKED','REFUELING','WAIT_FUEL') and not m.path and
                        np.linalg.norm(m.position[:2]-m.home[:2])<.1 for m in self.machines))

    def rebind_terrain(self,terrain,*,offset_m=(0.,0.)):
        """Extend the common window at the same resolution without moving gear."""
        old=self.terrain
        if terrain.cell!=old.cell:raise ValueError('Изменение разрешения с обычной техникой пока не поддерживается')
        offset=np.asarray(offset_m,dtype=float)
        cells=np.rint(offset/terrain.cell).astype(int)
        if np.any(cells<0) or np.any(np.abs(cells*terrain.cell-offset)>1e-8):raise ValueError('Смещение участка должно совпадать с сеткой')
        old_surface=self.ramp_surface;entry=np.asarray(self.ramp['entry_m']);end=np.asarray(self.ramp['end_m'])
        delta=end-entry;length=float(np.linalg.norm(delta));unit=delta/length
        old_along=(old.x[0,0]+offset[0]-entry[0])*unit[0]+(old.y[0,0]+offset[1]-entry[1])*unit[1]
        entry_height=self.ramp.get('entry_height_m',float(old_surface[0,0]+np.clip(old_along,0,length)*self.config.ramp_gradient))
        self._ground_stage_released=self._ground_released(offset)
        self._completed_revision=None
        self.terrain=terrain;x,y,w,h=self.config.work_rect
        self.work_mask=(terrain.x>=x)&(terrain.x<x+w)&(terrain.y>=y)&(terrain.y<y+h);self.mask=self.work_mask
        along=(terrain.x-entry[0])*unit[0]+(terrain.y-entry[1])*unit[1]
        lateral=np.abs((terrain.x-entry[0])*unit[1]-(terrain.y-entry[1])*unit[0])
        self.ramp_surface=entry_height-np.clip(along,0,length)*self.config.ramp_gradient
        if self.ramp.get('model')=='supported_access_v2':
            self.ramp_surface-=terrain.dump_slope*np.maximum(lateral-self.config.ramp_width_m/2,0.)
            self.ramp_mask=self.work_mask&(self.ramp_surface>terrain.target+1e-8)
            self.access_mask=self.work_mask&terrain.dig_mask&(lateral<=self.config.ramp_width_m/2)&(along>=0)&(along<=length)
        else:self.ramp_mask=self.work_mask&(lateral<=self.config.ramp_width_m/2)
        rows,cols=old_surface.shape;dx,dy=cells;self.ramp_surface[dy:dy+rows,dx:dx+cols]=old_surface
        self._marine=None;self._marine_stamp=None
        self._work_plan_cache=None;self._dump_plan_cache=None;self._dump_cursors={}
        self._cooperative_return_cache=None;self.cooperative_return_hints=None
        for machine in self.machines:
            machine.route_token=None;machine.budget_cache=None
            pending=getattr(machine,'pending_relocation',None)
            if pending is not None:
                # _fleet_preview shallow-copies machines; never mutate the
                # live command dictionary during an expansion preflight.
                machine.pending_relocation={**pending,'goal':tuple(np.asarray(pending['goal'])+offset)}

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
        self._recheck_finished_work()
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
            if m.kind=='excavator':
                waiting=getattr(m,'wait_for_truck',None)
                if waiting:
                    vehicle=next((v for v in self.machines if v.id==waiting),None)
                    if vehicle is not None and (vehicle.state=='DRIVING' or np.linalg.norm(vehicle.position[:2]-vehicle.home[:2])>=.1):
                        m.waiting_seconds+=1;m.reason='Ожидание фактического освобождения съезда самосвалом';continue
                    m.wait_for_truck=None;m.retry_route_at=0.;m.next_plan_at=0.
                if getattr(m,'return_requested',False) and m.state!='DRIVING':
                    if np.linalg.norm(m.position[:2]-m.home[:2])<.1:m.return_requested=False
                    elif self.sim.time>=m.retry_route_at:self._go_home(m);continue
                    else:m.waiting_seconds+=1;continue
                pending=getattr(m,'pending_relocation',None)
                if pending is not None and m.state!='DRIVING':
                    if not active:self._go_home(m);continue
                    if self.sim.time<m.retry_route_at:m.waiting_seconds+=1;continue
                    if self._assign_route(m,pending['goal'],'WAIT_TRUCK'):
                        m.pending_relocation=None;m.target=None;continue
                    if self.sim.time-pending['requested_at']<300:
                        m.waiting_seconds+=1;continue
                    m.pending_relocation=None;m.target=None;m.next_plan_at=0.
            if m.state=='DRIVING':
                # A committed return continues after clock/lunch boundaries.
                if not active and m.after_drive!='PARKED':
                    if self.sim.time<m.retry_route_at or not self._go_home(m):
                        m.waiting_seconds+=1
                        continue
                    if m.state!='DRIVING':continue
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
            if (m.kind=='truck' and at_home and (getattr(self.excavator,'wait_for_truck',None) or
                    getattr(self.excavator,'pending_relocation',None) is not None or
                    getattr(self.excavator,'return_requested',False) or self.excavator.state=='DRIVING')):
                m.state='PARKED';m.reason='Ожидание безопасного завершения манёвра экскаватора';m.waiting_seconds+=1;continue
            if m.kind=='truck' and at_home and getattr(m,'loading_hold_for',None):
                owner=self.excavator
                busy=(getattr(owner,'wait_for_truck',None)==m.id or getattr(owner,'pending_relocation',None) is not None or
                      getattr(owner,'return_requested',False) or owner.state=='DRIVING' or
                      (owner.target is None and np.linalg.norm(owner.position[:2]-owner.home[:2])>=.1 and
                       self.sim.time-getattr(m,'loading_hold_since',self.sim.time)<300))
                if busy:
                    m.state='PARKED';m.reason='Ожидание безопасного завершения манёвра экскаватора';m.waiting_seconds+=1;continue
                m.loading_hold_for=None
                body=cargo_volume(m.cargo_kg,m.soil,self.terrain.config.terrain.density)
                if (active and not self._work_complete and m.cargo_kg>0 and m.cargo_kg<c.truck_payload_kg*.9 and body<c.truck_body_m3*.9 and
                        owner.state in ('WAIT_TRUCK','DIGGING') and owner.target is not None):
                    m.state='WAIT_LOAD';m.load_started=self.sim.time
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
                if budget is None:
                    yielded=m.kind=='excavator' and self._release_loading_bay(m)
                    m.reason=('Ожидание освобождения однополосного съезда самосвалом' if yielded else 'Нет безопасного выезда на базу')
                    m.retry_route_at=self.sim.time+(5 if yielded else 30);continue
                if m.reason in ('Нет безопасного выезда на базу','Ожидание освобождения однополосного съезда самосвалом'):m.reason=''
                seconds,fuel,_=budget
                if shift['seconds_left']<=seconds or m.fuel_l<fuel:
                    self._go_home(m);continue
            if m.kind=='excavator':
                if self._work_complete:
                    if not at_home:self._go_home(m)
                    continue
                loading=any(v.kind=='truck' and (v.state=='WAIT_LOAD' or v.state=='DRIVING' and v.after_drive=='WAIT_LOAD')
                            for v in self.machines)
                if (self.ramp.get('model')=='supported_access_v2' and m.target is not None and m.expose_target is None
                        and self.sim.time>=m.next_advance_at):
                    incoming=any(v.kind=='truck' and v.state=='DRIVING' and v.after_drive=='WAIT_LOAD' for v in self.machines)
                    at_bay=any(v.kind=='truck' and v.state=='WAIT_LOAD' and
                               np.linalg.norm(v.position[:2]-m.position[:2])<=self._loading_distance(v)+.5 for v in self.machines)
                    weak_loading=at_bay and not incoming and m.timer<=1e-9 and self.target_yield(m)<c.bucket_m3*.15
                    if not loading or weak_loading:
                        m.next_advance_at=self.sim.time+60
                        if weak_loading:
                            # A waiting truck must not pin the excavator to
                            # tiny scraps forever. Only a materially better
                            # focused stance may interrupt a loading batch.
                            moved=self._choose_excavator_site(m,productive_only=True)
                        else:
                            # Keep the existing road-opening/downward advance
                            # while the truck is away hauling.
                            obstructed=self._obstructed_access_target(m)
                            moved=self._choose_excavator_site(m,advance_only=obstructed is None,expose=obstructed)
                        if moved:m.next_plan_at=0.;continue
                if m.state=='PARKED' or m.target is None:
                    if self.sim.time<m.next_plan_at:continue
                    if not self._choose_excavator_site(m):
                        if m.reason!='Ожидание освобождения однополосного съезда самосвалом':m.next_plan_at=self.sim.time+60
                        continue
                if m.state=='DRIVING':continue
                truck=next((v for v in self.machines if v.kind=='truck' and v.state=='WAIT_LOAD' and
                            np.linalg.norm(v.position[:2]-m.position[:2])<=self._loading_distance(v)+.5),None)
                if truck is None:m.state='WAIT_TRUCK';m.waiting_seconds+=1;continue
                if c.cut_tolerance_m>0 and m.timer==0 and self.target_yield(m,m.target)<=1e-12:
                    # A previous bite or intervening slump can finish this
                    # footprint. Replan before spending another bucket cycle.
                    m.target=self._available_target(m)
                    if m.target is None:continue
                if not self._burn(m,True):continue
                m.state='DIGGING';m.working_seconds+=1;m.timer+=1
                factor=1.
                # All material factors are >= 1. Until the base cycle elapses
                # no bucket can finish, even if the exposed mixture changes.
                if m.timer>=c.bucket_cycle_s and self.terrain.soil is not None and m.target is not None:
                    load=self.terrain.preview_load(*m.target[:2],c.truck_payload_kg,source='conventional',radius=max(.5,self.terrain.cell*.71),max_volume=c.bucket_m3)
                    factor=max(1.,load.factor)
                if m.timer>=c.bucket_cycle_s*factor:
                    m.timer=0.;cut=self._cut(m,truck)
                    body=cargo_volume(truck.cargo_kg,truck.soil,self.terrain.config.terrain.density)
                    full=truck.cargo_kg>=c.truck_payload_kg*.9 or body>=c.truck_body_m3*.9
                    if full or not cut:
                        # Losing one bucket target means replanning the cut,
                        # not dispatching an otherwise useful partial load.
                        if full and truck.cargo_kg>0:
                            if self._dump_point(truck) is None and not truck.reason:truck.reason='Нет доступного места отвала; груз сохранён'
                        m.target=None
            else:
                if self._work_complete and m.cargo_kg>0 and m.state not in ('DRIVING','UNLOADING'):
                    if self._dump_point(m) is None and not m.reason:m.reason='Нет доступного отвала; груз сохранён'
                    continue
                if self._work_complete and m.cargo_kg<=0:
                    if not at_home:self._go_home(m)
                    continue
                if m.state in ('PARKED','WAIT_LOAD'):
                    body=cargo_volume(m.cargo_kg,m.soil,self.terrain.config.terrain.density)
                    full=m.cargo_kg>=c.truck_payload_kg*.9 or body>=c.truck_body_m3*.9
                    if m.cargo_kg>0 and (m.state=='PARKED' or full or self.sim.time-m.load_started>300):
                        if self._dump_point(m) is None and not m.reason:m.reason='Нет доступного отвала; груз сохранён'
                    elif self.excavator.state in ('WAIT_TRUCK','DIGGING') and self.excavator.target is not None:
                        following=m.state=='WAIT_LOAD' and np.linalg.norm(m.position[:2]-self.excavator.position[:2])>self._loading_distance(m)+.5
                        if m.state=='PARKED' or following:
                            bay=next((other for other in self.machines if other is not m and other.kind=='truck' and
                                (other.state=='WAIT_LOAD' or other.state=='DRIVING' and other.after_drive=='WAIT_LOAD')),None)
                            if bay is not None:
                                m.reason='Очередь к единственному месту погрузки; ожидание на базе';m.waiting_seconds+=1;continue
                            if not self._loading_point(m):
                                m.retry_route_at=max(m.retry_route_at,self.sim.time+30)
                                if not m.reason:m.reason='Нет безопасной точки погрузки; груз сохранён'
                        else:m.waiting_seconds+=1
                    else:m.waiting_seconds+=1
                    continue
                if m.state=='UNLOADING':
                    if not self._burn(m):continue
                    m.timer-=1
                    if m.timer<=0:
                        if not self._unload(m):
                            if self._dump_point(m) is None:m.retry_route_at=self.sim.time+30
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
            if (len(m.dimensions_m)!=3 or any(not math.isfinite(v) or not .5<=v<=30 for v in m.dimensions_m)
                    or not math.isfinite(m.heading_rad)):raise ValueError('Повреждены габариты или направление машины')
            if not math.isfinite(m.fuel_l) or not -1e-8<=m.fuel_l<=self._tank(m)+1e-8 or not math.isfinite(m.cargo_kg) or not 0<=m.cargo_kg<=self.config.truck_payload_kg+1e-7:
                raise ValueError('Повреждён бак или груз машины')
            if cargo_volume(m.cargo_kg,m.soil,self.terrain.config.terrain.density)>self.config.truck_body_m3+1e-7:
                raise ValueError('Объём грунта превышает вместимость кузова')
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
        t=self.terrain;ids=self._work_plan()['ids']
        retained=np.where(self.ramp_mask.ravel()[ids]&t.dig_mask.ravel()[ids],
            np.maximum(np.minimum(t.height.ravel()[ids],self.ramp_surface.ravel()[ids])-t.target.ravel()[ids],0.),0.)
        ramp={**self.ramp,'remaining_m3':float(retained.sum()*t.area)}
        remaining=self._work_remaining()
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
            diesel_energy_consumed_mj=self.fuel_used_l*35.8,shift=self.shift(),ramp=ramp,remaining_work_m3=float(remaining.sum()*t.area),
            cut_tolerance_m=c.cut_tolerance_m,remaining_actionable_m3=float(remaining[self._actionable_work(remaining)].sum()*t.area),
            refuel_slots=c.refuel_slots,route_calculations=self.route_calculations,crew_count=c.shifts_per_day*(c.excavator_count+c.truck_count),
            machines=[dict(id=m.id,kind=m.kind,state=m.state,reason=m.reason,position_m=m.position.tolist(),dimensions_m=list(m.dimensions_m),footprint_radius_m=m.radius,fuel_l=m.fuel_l,
                           heading_rad=m.heading_rad,cargo_kg=m.cargo_kg,excavated_m3=m.excavated_m3,delivered_m3=m.delivered_m3,path_m=[list(p) for p in m.path],
                           tank_capacity_l=self._tank(m),emergency_work_seconds=m.emergency_work_seconds,engine_seconds=m.engine_seconds,working_seconds=m.working_seconds,distance_m=m.distance_m)
                      for m in self.machines])
