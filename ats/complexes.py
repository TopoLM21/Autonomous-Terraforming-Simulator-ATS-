"""Reusable equipment templates and isolated work zones in one regional project."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import numpy as np
from pathlib import Path
from .config import TerrainConfig,TrenchConfig
from .power_design import equipment_copy,sized_configuration,PowerRequest
from .experiments import configuration_data,configuration_from_data


@dataclass
class ComplexTemplate:
    name: str
    config: object
    power_request: dict | None = None
    power_result: dict | None = None
    stock: list | None = None  # additional physical packs beyond the standard recipe

    @classmethod
    def capture(cls,config,name='Рабочий комплекс',power_result=None):
        if not isinstance(name,str) or not name.strip():raise ValueError('Укажите название комплекса')
        c=sized_configuration(config,power_result) if power_result is not None else equipment_copy(config)
        # A template contains a hardware recipe, never terrain edits, cargo,
        # existing reservations, battery SOC histories or a running simulation.
        c.initial_height=c.reference_height=c.initial_soil=c.georeference=c.plan=None
        c.local_reference=False
        c.soil_profile=None;c.terrain=TerrainConfig();c.trench=TrenchConfig()
        c.base.x=5.;c.base.y=16.
        # Initial staging-map estimate only. Real field placement is checked by
        # Simulation; vehicle dimensions are deliberately not inferred from it.
        side=max(64,math.ceil((math.sqrt(c.base.panel_area)+32)/32)*32)
        if c.energy_design:side=max(side,math.ceil(c.energy_design['required_map_side_m']/32)*32)
        from .maps import workspace_cell
        c.terrain.cell_size=workspace_cell(min(1024,side),c.terrain.cell_size)
        c.terrain.n=round(min(1024,side)/c.terrain.cell_size)
        c.validate()
        return cls(name.strip(),c,deepcopy(power_result['request']) if power_result else None,deepcopy(power_result))

    def to_data(self):
        return dict(format='ATS_COMPLEX_TEMPLATE',version=1,name=self.name,
                    configuration=configuration_data(self.config),power_request=self.power_request,power_result=self.power_result,stock=self.stock)

    @classmethod
    def from_simulation(cls,simulation,name='Текущая комплектация'):
        template=cls.capture(simulation.config,name);w=simulation.logistics
        if w is None:return template
        from .inventory import BatteryModule
        from collections import Counter
        expected=Counter()
        for d in simulation.drones:
            for recipe in d.config.equipment.batteries:
                expected[f'drone:{recipe.capacity_kwh:g}:{recipe.mass_kg:g}']+=recipe.quantity*(1+w.config.drone_spare_sets)
        expected['excavator']=w.config.excavator_battery_count if w.excavator else 0
        groups={}
        for pack in w.inventory.select(BatteryModule):
            if pack.compatibility=='base':continue
            key=pack.compatibility
            if key not in groups:groups[key]=dict(count=0,compatibility=key,capacity_kwh=pack.capacity_j/3_600_000,
                mass_kg=pack.mass_kg,max_charge_w=pack.max_charge_w,max_discharge_w=pack.max_discharge_w,initial_soc=1.)
            groups[key]['count']+=1
        template.stock=[{**row,'count':row['count']-expected[key]} for key,row in groups.items() if row['count']>expected[key]]
        template.validate();return template

    def validate(self):
        if not isinstance(self.name,str) or not self.name.strip():raise ValueError('У шаблона нет названия')
        self.config.validate()
        if any(getattr(self.config,k) is not None for k in ('initial_height','reference_height','initial_soil','plan','georeference')):
            raise ValueError('Шаблон содержит состояние участка')
        if self.power_request is not None:PowerRequest(**self.power_request).validate()
        for row in self.stock or []:
            if (type(row['count']) is not int or not 1<=row['count']<=4096 or not isinstance(row['compatibility'],str) or
                    any(not math.isfinite(row[k]) or row[k]<=0 for k in ('capacity_kwh','mass_kg','max_discharge_w')) or
                    not math.isfinite(row['max_charge_w']) or row['max_charge_w']<0 or not 0<=row['initial_soc']<=1):raise ValueError('Неверный дополнительный запас батарей')

    def equipment_mass(self):
        c=self.config;log=c.logistics
        mass=c.base.equipment.mass+sum(d.equipment.mass for d in c.fleet.models)
        if log is not None:
            drone_packs=sum(sum(m.quantity*m.mass_kg for m in d.equipment.batteries) for d in c.fleet.models)
            drone_modules=sum(sum(m.quantity for m in d.equipment.batteries) for d in c.fleet.models)
            mass+=drone_packs*log.drone_spare_sets+log.container_count*log.container_tare_kg
            mass+=len(c.fleet.models)*(2.+log.hoist_mass_kg)+log.rigging_mass_kg
            robot_modules=log.excavator_battery_count if log.excavator_enabled else 0
            if log.excavator_enabled:mass+=log.excavator_dry_mass_kg+robot_modules*log.pack_mass_kg
            if log.individual_chargers:mass+=(drone_modules*(1+log.drone_spare_sets)+robot_modules)*log.charger_slot_mass_kg
            mass+=sum(r['count']*(r['mass_kg']+(log.charger_slot_mass_kg if log.individual_chargers else 0.)) for r in self.stock or [])
        return float(mass)

    @classmethod
    def from_data(cls,data):
        if data.get('format')!='ATS_COMPLEX_TEMPLATE' or data.get('version')!=1:raise ValueError('Неизвестный формат шаблона комплекса')
        name=data['name']
        if not isinstance(name,str) or not name.strip():raise ValueError('У шаблона нет названия')
        c=configuration_from_data(data['configuration'])
        if any(getattr(c,k) is not None for k in ('initial_height','reference_height','initial_soil','plan','georeference')):
            raise ValueError('Шаблон должен содержать оборудование, а не состояние участка')
        request=data.get('power_request')
        if request is not None:PowerRequest(**request).validate()
        template=cls(name,c,request,data.get('power_result'),data.get('stock'));template.validate();return template

    def save(self,path):
        Path(path).write_text(json.dumps(self.to_data(),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):return cls.from_data(json.loads(Path(path).read_text(encoding='utf-8')))


@dataclass
class ComplexUnit:
    id: str
    name: str
    simulation: object
    born_at: float = 0.
    section: int | None = None
    committed: bool = False

    def bounds(self):
        c=self.simulation.config;origin=(c.georeference or {}).get('origin_m',[0.,0.]);side=c.terrain.n*c.terrain.cell_size
        return [float(origin[0]),float(origin[1]),float(side),float(side)]


class ComplexProject:
    shared_site=None
    transition=None
    """One clock/project; each complex owns an exclusive, separated work zone.

    Existing simulators keep their physical inventories and routing. Overlapping
    work windows are rejected before creating hardware. No soil or energy is
    exchanged between the isolated zones. The coarse regional DEM is shared.
    """
    def __init__(self,simulation,landscape=None):
        self.landscape=landscape;self.units=[ComplexUnit('complex_1','Комплекс 1',simulation,0.,self._section(simulation))]
        self.time=simulation.time;self.active_id='complex_1';self.next_id=2
        self.initial_energy_kwh=simulation.report(full=False)['initial_energy_kwh'];self.delivered_energy_kwh=0.
        w=simulation.logistics
        self.initial_equipment_mass=self.hardware_mass(simulation)-(w.imported_mass_kg-w.exported_mass_kg if w else 0.)
        self.delivered_equipment_mass=0.
        self.templates=[]

    @staticmethod
    def hardware_mass(simulation):
        w=simulation.logistics
        if w is None:return simulation.base.config.mass+sum(d.config.empty_mass for d in simulation.drones)
        return (sum(item.mass_kg for item in w.inventory.items.values())+
                sum(d.config.equipment.frame_mass_kg for d in simulation.drones)+
                (w.config.excavator_dry_mass_kg if w.excavator is not None else 0.))

    def inventory_ids(self):
        return [f'{u.id}/{id}' for u in self.units if u.simulation.logistics is not None
                for id in u.simulation.logistics.inventory.items]

    @staticmethod
    def _section(simulation):return ((simulation.config.georeference or {}).get('channel') or {}).get('section')

    @property
    def active(self):return next(u for u in self.units if u.id==self.active_id)

    @property
    def running(self):return not self.blocked and any(u.simulation.state.value=='RUNNING' for u in self.units)

    @property
    def blocked(self):return any(u.simulation.state.value=='BLOCKED' for u in self.units)

    def select(self,id):
        if not any(u.id==id for u in self.units):raise ValueError('Неизвестный комплекс')
        self.active_id=id;return self.active.simulation

    def _overlaps(self,bounds,margin):
        x,y,w,h=bounds
        for u in self.units:
            xx,yy,ww,hh=u.bounds()
            if x<xx+ww+margin and x+w+margin>xx and y<yy+hh+margin and y+h+margin>yy:return True
        return False

    def add(self,template,origin=None,section=None):
        template.validate()
        if len(self.units)>=20:raise ValueError('Одновременно допускается до 20 рабочих комплексов')
        if self.landscape is None:raise ValueError('Для работы нескольких комплексов откройте региональный DEM или проект канала')
        from .simulation import Simulation
        from .landscape import workspace_config
        from .channel import ChannelProject
        c=deepcopy(template.config)
        c.slope_model=self.active.simulation.config.slope_model
        if section is not None:
            if self.landscape.channel_project is None:raise ValueError('Сначала разметьте общий проект канала')
            project=ChannelProject.from_data(self.landscape.channel_project)
            if not isinstance(section,int) or not 0<=section<project.section_count:raise ValueError('Участок вне проекта')
            c=project.configuration(self.landscape,c,section)
        else:
            if origin is None:raise ValueError('Укажите начало отдельного рабочего участка')
            source=self.active.simulation.config;side=source.terrain.n*source.terrain.cell_size
            c=workspace_config(self.landscape,c,origin,side)
            if source.plan is not None:
                # Global canal geometry is assigned only through section numbers.
                if source.plan.channel:raise ValueError('Для канала выберите номер участка проекта')
                c.plan=deepcopy(source.plan)
        bounds=[*(c.georeference['origin_m']),c.terrain.n*c.terrain.cell_size,c.terrain.n*c.terrain.cell_size]
        fleets=[c.fleet,*[u.simulation.config.fleet for u in self.units]]
        margin=2*max(f.detour_distance for f in fleets)+max(f.separation for f in fleets)
        if self._overlaps(bounds,margin):raise ValueError('Рабочие окна комплексов пересекаются или не оставляют места для обхода дронов; выберите более дальний участок')
        # A newly delivered complex starts at the current solar date/time.
        leader=self.units[0].simulation
        solar_hours=leader.config.start_hour+self.time/3600
        c.start_hour=solar_hours%24;c.base.start_day_of_year=(leader.config.base.start_day_of_year-1+int(solar_hours//24))%365+1
        sim=Simulation(c,initial_stock=template.stock)
        if sim.state.value=='BLOCKED':raise ValueError(sim.reason)
        id=f'complex_{self.next_id}';unit=ComplexUnit(id,f'{template.name} · {self.next_id}',sim,self.time,section)
        energy=sim.report(full=False)['initial_energy_kwh']
        self.units.append(unit);self.next_id+=1;self.delivered_energy_kwh+=energy
        self.delivered_equipment_mass+=self.hardware_mass(sim)
        return unit

    def enable_shared(self):
        if self.shared_site is not None:return self.shared_site
        if len(self.units)!=1:raise ValueError('Общий участок создаётся из одного действующего комплекса; раздельные региональные участки сохраняются в прежнем режиме')
        from .shared_site import SharedSite
        self.shared_site=SharedSite(self);return self.shared_site

    def add_shared(self,template,position=None):
        """Deliver a real independent complex onto the existing physical map."""
        if self.transition is not None:raise ValueError('Сначала завершите или отмените региональный переезд')
        if len(self.units)>=20:raise ValueError('Одновременно допускается до 20 рабочих комплексов')
        template.validate();site=self.enable_shared();source=self.units[0].simulation.config
        from .power_design import equipment_copy
        from .base_layout import configure_layout,rotate_offset,layout_footprints
        from .shared_site import WorkTerrain
        from .simulation import Simulation
        from .solar_base import SolarBase
        c=equipment_copy(template.config)
        c.slope_model=source.slope_model
        for key in ('terrain','trench','plan','georeference'):setattr(c,key,deepcopy(getattr(source,key)))
        for key in ('initial_height','reference_height','initial_soil','soil_profile','local_reference'):setattr(c,key,getattr(source,key))
        c.base.latitude_deg=source.base.latitude_deg
        solar_hours=source.start_hour+self.time/3600;c.start_hour=solar_hours%24
        c.base.start_day_of_year=(source.base.start_day_of_year-1+int(solar_hours//24))%365+1
        id=f'complex_{self.next_id}';view=WorkTerrain(site.core,c,site,id)
        extra=[(r['count']*(r['mass_kg']+(c.logistics.charger_slot_mass_kg if c.logistics and c.logistics.individual_chargers else 0.)),r['count']*r['capacity_kwh']) for r in template.stock or []]
        spec=configure_layout(c,extra=extra);side=site.core.height.shape[0]*site.core.cell
        goal=site.placement_goal()
        extra_sites=[(p[1],p[2]) for p in (c.plan.channel or {}).get('operations',{}).get('base_sites',[])] if c.plan else []
        candidates=[position] if position is not None else sorted(
            extra_sites+[(float(x),float(y)) for x in np.linspace(5,side-5,23) for y in np.linspace(5,side-5,23)],
            key=lambda p:float(np.sum((np.asarray(p)-goal)**2)))
        chosen=None
        pad=math.ceil((max(c.fleet.count,c.fleet.charge_slots)-1)/2)*c.fleet.pad_spacing+c.fleet.separation
        for point in candidates:
            point=np.asarray(point,dtype=float)
            if point.shape!=(2,) or not np.all(np.isfinite(point)):raise ValueError('Неверные координаты базы')
            rect=site.pads_bounds(c,point)
            if site.ground_occupied(id,rect) or not site.pads_clear(c,point):continue
            base=SolarBase(c.base,np.r_[point,view.sample(*point)])
            if spec:base.layout_spec=spec
            if base.safe_path(point,view):chosen=point;break
        if chosen is None:raise ValueError('Нет безопасного места для комплекса на общей карте. Увеличьте карту или уменьшите комплектацию')
        c.base.x,c.base.y=map(float,chosen)
        sim=Simulation(c,terrain=view,initial_stock=template.stock)
        if sim.state.value=='BLOCKED':raise ValueError(sim.reason)
        # The newly delivered excavator and racks start beside their own base.
        # No old equipment or battery history is moved by cloning.
        w=sim.logistics
        if w and w.excavator:
            e=w.excavator;old=e.position.copy();radius=c.base.radius+w.config.support_half_width_m+2
            for delta in ((radius,0),(-radius,0),(0,radius),(0,-radius)):
                point=sim.base.position.copy();point[:2]+=rotate_offset(c.base.layout,delta);point[2]=view.sample(*point[:2])
                boxes=list(layout_footprints(sim.base.layout_spec,sim.base.position)) if c.base.layout is not None else []
                if w.field_rect:boxes.append(w.field_rect)
                support=w.config.support_half_width_m
                if any(x-support<point[0]<x+width+support and y-support<point[1]<y+height+support for x,y,width,height in boxes):continue
                if e.stable(point,view) and not site.ground_occupied(id,(point[0]-1,point[1]-1,2,2)):
                    shift=point-old;e.position=point;w.rack=point.copy()
                    for item in w.inventory.items.values():
                        if np.linalg.norm(np.asarray(item.position)-old)<1e-7:item.position=tuple(np.asarray(item.position)+shift)
                    break
            else:raise ValueError('Нет безопасной площадки Крота рядом с новой базой')
        unit=ComplexUnit(id,f'{template.name} · {self.next_id}',sim,self.time,self._section(sim))
        energy=sim.report(full=False)['initial_energy_kwh']
        self.units.append(unit);self.next_id+=1;self.delivered_energy_kwh+=energy
        self.delivered_equipment_mass+=self.hardware_mass(sim);site.invalidate_geometry();site.partition()
        return unit

    def find_free_section(self,template,start=0):
        if self.landscape is None or self.landscape.channel_project is None:return None
        from .channel import ChannelProject
        project=ChannelProject.from_data(self.landscape.channel_project)
        side=project.side_m;cell=self.landscape.cell_size
        fleets=[template.config.fleet,*[u.simulation.config.fleet for u in self.units]]
        margin=2*max(f.detour_distance for f in fleets)+max(f.separation for f in fleets)
        for i in range(start,project.section_count):
            origin=project.origin(self.landscape.raster,i)
            if not self._overlaps([*origin,side,side],margin):return i
        return None

    def step(self):
        if not self.running:return
        if self.transition is not None:self.transition.before_step()
        if self.shared_site is not None:self.shared_site.start_tick()
        for unit in self.units:
            s=unit.simulation
            if s.state.value=='RUNNING':unit.committed=False;s.step()
            elif s.state.value=='COMPLETED':
                # Finished machinery is parked and powered off. Its calendar
                # still follows the common project clock, without generating
                # unaccounted power or freezing its recorded idle time.
                s.time+=1.;s.record_activities(1.);s.base.solar_power=0.
                if s.logistics is not None:
                    s.logistics.base_load_power_w=s.logistics.base_battery_power_w=0.
                    for slot in s.logistics.battery_service.slots.values():slot.input_power_w=0.
            if self.shared_site is None and s.state.value=='COMPLETED' and not unit.committed:
                self.commit(unit);unit.committed=True
        if self.shared_site is not None and self.transition is None:self.shared_site.advance_tasks()
        self.time+=1.
        if self.transition is not None:self.transition.after_step()

    def move_window(self,origin=None,*,section=None):
        from .regional import RegionalTransition
        if self.transition is not None:raise ValueError('Переезд группы уже выполняется; дождитесь завершения или отмените его')
        move=RegionalTransition.prepare(self,origin,section)
        self.transition=move;move.start()
        return move

    def cancel_window_move(self):
        if self.transition is None:return False
        self.transition.cancel();return True

    def persist_window(self):
        """Commit a parked whole shared group, including unfinished earthwork."""
        if self.transition is not None:raise ValueError('Сохраните переезд всей группы в .atsstate; регион можно экспортировать после остановки')
        if self.shared_site is None:
            for u in self.units:self.commit(u)
            return
        if self.landscape is None:return
        for u in self.units:
            s=u.simulation;w=s.logistics;e=w.excavator if w else None
            if (s.base.destination is not None or not all(d.grounded and abs(d.cargo)<1e-8 for d in s.drones)
                    or w and (w.active or w.group or any(c.soil_kg for c in w.containers) or e and (e.bucket_kg or e.active_container))):
                raise ValueError('Сначала посадите все комплексы и завершите перевозки всей группы')
        t=self.shared_site.core;s=self.units[0].simulation
        self.landscape.commit_window(s.config.georeference['origin_m'],t.height,cell_size=t.cell,soil=t.soil)
        if self.landscape.channel_project is not None and t.remaining<=self.shared_site.completion_volume:
            from .channel import ChannelProject
            channel=ChannelProject.from_data(self.landscape.channel_project);r=self.report()
            r.update(state='COMPLETED',georeference=s.config.georeference,
                     solar_generated_kwh=sum(u.simulation.base.generated for u in self.units)/3_600_000)
            channel.record(r);self.landscape.channel_project=channel.to_data()

    def commit(self,unit):
        if self.landscape is None:return
        if self.shared_site is not None and any(u.simulation.state.value!='COMPLETED' for u in self.units):
            raise ValueError('Дождитесь завершения общей карты; текущее состояние всей группы сохраняется в .atsstate')
        s=unit.simulation
        if not all(d.grounded and abs(d.cargo)<=1e-8 for d in s.drones):raise ValueError('Завершите перевозки до фиксации рабочего участка')
        w=s.logistics;e=w.excavator if w else None
        if w and (w.active or w.group or any(c.soil_kg for c in w.containers) or e and (e.bucket_kg or e.active_container)):
            raise ValueError('Завершите обслуживание и выгрузку контейнеров')
        self.landscape.commit_window(s.config.georeference['origin_m'],s.terrain.height,cell_size=s.terrain.cell,soil=s.terrain.soil)
        if unit.section is not None and self.landscape.channel_project is not None:
            from .channel import ChannelProject
            project=ChannelProject.from_data(self.landscape.channel_project);active=project.active_section
            project.active_section=unit.section;project.record(s.report(full=False));project.active_section=active
            self.landscape.channel_project=project.to_data()

    def report(self):
        rows=[];mass_error=energy_error=0.;initial=0.
        for unit in self.units:
            s=unit.simulation;r=s.report(full=False)
            mass_error+=r['mass_error_kg'];energy_error+=r['energy_error_j']
            initial+=r['initial_energy_kwh']
            rows.append(dict(id=unit.id,name=unit.name,state=r['state'],reason=r['reason'],born_at=unit.born_at,
                time_seconds=s.time,section=unit.section,bounds=unit.bounds(),drones=len(s.drones),
                excavated_m3=r['excavated_m3'],consumed_kwh=r['consumed_kwh'],base_capacity_kwh=r['base_capacity_kwh'],
                panel_area_m2=r['panel_area_m2'],mass_error_kg=r['mass_error_kg'],energy_error_j=r['energy_error_j']))
        energy_error+=(initial-self.initial_energy_kwh-self.delivered_energy_kwh)*3_600_000
        if self.shared_site is not None:mass_error=self.shared_site.mass_error()
        hardware=sum(self.hardware_mass(u.simulation) for u in self.units)
        hardware_error=None
        if all(u.simulation.logistics is not None for u in self.units):
            transfers=sum(u.simulation.logistics.imported_mass_kg-u.simulation.logistics.exported_mass_kg for u in self.units)
            hardware_error=hardware-self.initial_equipment_mass-self.delivered_equipment_mass-transfers
        state='BLOCKED' if self.blocked else 'RUNNING' if self.running else 'COMPLETED'
        reason=next((u.simulation.reason for u in self.units if u.simulation.state.value=='BLOCKED'),'')
        result=dict(state=state,reason=reason,time_seconds=self.time,active_id=self.active_id,complexes=rows,mass_error_kg=mass_error,
                    energy_error_j=energy_error,initial_energy_kwh=self.initial_energy_kwh,
                    delivered_energy_kwh=self.delivered_energy_kwh,
                    equipment_mass_kg=hardware,delivered_equipment_mass_kg=self.delivered_equipment_mass,
                    equipment_mass_error_kg=hardware_error,
                    excavated_m3=sum(r['excavated_m3'] for r in rows),consumed_kwh=sum(r['consumed_kwh'] for r in rows),
                    drone_count=sum(r['drones'] for r in rows),complex_count=len(rows),
                    trips=sum(u.simulation.trips for u in self.units))
        if self.transition is not None:result['regional_transition']=self.transition.report()
        if hasattr(self,'window_history'):result['window_history']=self.window_history
        if self.shared_site is not None:
            site=self.shared_site;errors=site.material_errors()
            result['shared_worksite']=dict(remaining_m3=site.core.remaining,assignments=site.assignments,
                unique_ground_cells=site.core.height.size,shared_height_arrays=1,
                max_material_mass_error_kg=float(np.abs(errors).max()) if errors is not None else None)
        return result

    def validate(self):
        if not self.units or len(self.units)>20 or len({u.id for u in self.units})!=len(self.units):raise ValueError('Повреждён состав комплексов')
        if self.active_id not in {u.id for u in self.units} or not math.isfinite(self.time) or self.time<0:raise ValueError('Повреждено время или выбор комплекса')
        if any(not isinstance(u.id,str) or not u.id.startswith('complex_') or not u.id[8:].isdigit() for u in self.units):
            raise ValueError('Повреждены идентификаторы комплексов')
        if type(self.next_id) is not int or self.next_id<=max(int(u.id[8:]) for u in self.units):raise ValueError('Повреждён счётчик комплексов')
        simulations=set();items=set()
        for u in self.units:
            if not math.isfinite(u.born_at) or not 0<=u.born_at<=self.time:raise ValueError('Повреждено время поставки комплекса')
            if abs(u.born_at+u.simulation.time-self.time)>1e-7:raise ValueError('Нарушены часы комплексов')
            if id(u.simulation) in simulations:raise ValueError('Два комплекса ссылаются на одну симуляцию')
            simulations.add(id(u.simulation))
            if u.simulation.logistics:
                for item in u.simulation.logistics.inventory.items.values():
                    if id(item) in items:raise ValueError('Один предмет принадлежит двум комплексам')
                    items.add(id(item))
        for i,u in enumerate(self.units):
            a=u.bounds()
            for v in self.units[i+1:]:
                b=v.bounds()
                f,g=u.simulation.config.fleet,v.simulation.config.fleet
                gap=2*max(f.detour_distance,g.detour_distance)+max(f.separation,g.separation)
                if self.shared_site is None and a[0]<b[0]+b[2]+gap and a[0]+a[2]+gap>b[0] and a[1]<b[1]+b[3]+gap and a[1]+a[3]+gap>b[1]:
                    raise ValueError('Сохранение содержит пересекающиеся рабочие участки')
        if self.shared_site is not None:self.shared_site.validate()
        if self.transition is not None:
            from .regional import RegionalTransition
            if type(self.transition) is not RegionalTransition or self.transition.project is not self:raise ValueError('Потеряна связь с переездом группы')
            self.transition.validate()
        for t in self.templates:t.validate()
        initial=sum((u.simulation._initial_energy+(u.simulation.logistics.initial_extra_energy if u.simulation.logistics else 0.))/3_600_000 for u in self.units)
        if (not all(math.isfinite(v) and v>=0 for v in (self.initial_energy_kwh,self.delivered_energy_kwh,self.initial_equipment_mass,self.delivered_equipment_mass)) or
                abs(initial-self.initial_energy_kwh-self.delivered_energy_kwh)>max(1e-8,initial*1e-12)):
            raise ValueError('Повреждён учёт энергии поставленных комплексов')
        r=self.report()
        if r['equipment_mass_error_kg'] is not None and abs(r['equipment_mass_error_kg'])>max(1e-7,r['equipment_mass_kg']*1e-12):
            raise ValueError('Повреждён учёт массы поставленных комплексов')
