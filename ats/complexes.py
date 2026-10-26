"""Reusable equipment templates and isolated work zones in one regional project."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
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

    @classmethod
    def capture(cls,config,name='Рабочий комплекс',power_result=None):
        if not isinstance(name,str) or not name.strip():raise ValueError('Укажите название комплекса')
        c=sized_configuration(config,power_result) if power_result is not None else equipment_copy(config)
        # A template contains a hardware recipe, never terrain edits, cargo,
        # existing reservations, battery SOC histories or a running simulation.
        c.initial_height=c.reference_height=c.initial_soil=c.georeference=c.plan=None
        c.soil_profile=None;c.terrain=TerrainConfig();c.trench=TrenchConfig()
        c.base.x=5.;c.base.y=16.
        # Initial staging-map estimate only. Real field placement is checked by
        # Simulation; vehicle dimensions are deliberately not inferred from it.
        side=max(64,math.ceil((math.sqrt(c.base.panel_area)+32)/32)*32)
        c.terrain.n=round(min(1024,side)/c.terrain.cell_size)
        c.validate()
        return cls(name.strip(),c,deepcopy(power_result['request']) if power_result else None,deepcopy(power_result))

    def to_data(self):
        return dict(format='ATS_COMPLEX_TEMPLATE',version=1,name=self.name,
                    configuration=configuration_data(self.config),power_request=self.power_request,power_result=self.power_result)

    def validate(self):
        if not isinstance(self.name,str) or not self.name.strip():raise ValueError('У шаблона нет названия')
        self.config.validate()
        if any(getattr(self.config,k) is not None for k in ('initial_height','reference_height','initial_soil','plan','georeference')):
            raise ValueError('Шаблон содержит состояние участка')
        if self.power_request is not None:PowerRequest(**self.power_request).validate()

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
        template=cls(name,c,request,data.get('power_result'));template.validate();return template

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
        if len(self.units)>=8:raise ValueError('Одновременно допускается до восьми рабочих комплексов')
        if self.landscape is None:raise ValueError('Для работы нескольких комплексов откройте региональный DEM или проект канала')
        from .simulation import Simulation
        from .landscape import workspace_config
        from .channel import ChannelProject
        c=deepcopy(template.config)
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
        sim=Simulation(c)
        if sim.state.value=='BLOCKED':raise ValueError(sim.reason)
        id=f'complex_{self.next_id}';unit=ComplexUnit(id,f'{template.name} · {self.next_id}',sim,self.time,section)
        energy=sim.report(full=False)['initial_energy_kwh']
        self.units.append(unit);self.next_id+=1;self.delivered_energy_kwh+=energy
        self.delivered_equipment_mass+=self.hardware_mass(sim)
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
            if s.state.value=='COMPLETED' and not unit.committed:
                self.commit(unit);unit.committed=True
        self.time+=1.

    def commit(self,unit):
        if self.landscape is None:return
        s=unit.simulation
        if not all(d.grounded and d.cargo==0 for d in s.drones):raise ValueError('Завершите перевозки до фиксации рабочего участка')
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
        hardware=sum(self.hardware_mass(u.simulation) for u in self.units)
        hardware_error=None
        if all(u.simulation.logistics is not None for u in self.units):
            transfers=sum(u.simulation.logistics.imported_mass_kg-u.simulation.logistics.exported_mass_kg for u in self.units)
            hardware_error=hardware-self.initial_equipment_mass-self.delivered_equipment_mass-transfers
        state='BLOCKED' if self.blocked else 'RUNNING' if self.running else 'COMPLETED'
        reason=next((u.simulation.reason for u in self.units if u.simulation.state.value=='BLOCKED'),'')
        return dict(state=state,reason=reason,time_seconds=self.time,active_id=self.active_id,complexes=rows,mass_error_kg=mass_error,
                    energy_error_j=energy_error,initial_energy_kwh=self.initial_energy_kwh,
                    delivered_energy_kwh=self.delivered_energy_kwh,
                    equipment_mass_kg=hardware,delivered_equipment_mass_kg=self.delivered_equipment_mass,
                    equipment_mass_error_kg=hardware_error,
                    excavated_m3=sum(r['excavated_m3'] for r in rows),consumed_kwh=sum(r['consumed_kwh'] for r in rows),
                    drone_count=sum(r['drones'] for r in rows),complex_count=len(rows),
                    trips=sum(u.simulation.trips for u in self.units))

    def validate(self):
        if not self.units or len(self.units)>8 or len({u.id for u in self.units})!=len(self.units):raise ValueError('Повреждён состав комплексов')
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
                if a[0]<b[0]+b[2]+gap and a[0]+a[2]+gap>b[0] and a[1]<b[1]+b[3]+gap and a[1]+a[3]+gap>b[1]:
                    raise ValueError('Сохранение содержит пересекающиеся рабочие участки')
        for t in self.templates:t.validate()
        initial=sum((u.simulation._initial_energy+(u.simulation.logistics.initial_extra_energy if u.simulation.logistics else 0.))/3_600_000 for u in self.units)
        if (not all(math.isfinite(v) and v>=0 for v in (self.initial_energy_kwh,self.delivered_energy_kwh,self.initial_equipment_mass,self.delivered_equipment_mass)) or
                abs(initial-self.initial_energy_kwh-self.delivered_energy_kwh)>max(1e-8,initial*1e-12)):
            raise ValueError('Повреждён учёт энергии поставленных комплексов')
        r=self.report()
        if r['equipment_mass_error_kg'] is not None and abs(r['equipment_mass_error_kg'])>max(1e-7,r['equipment_mass_kg']*1e-12):
            raise ValueError('Повреждён учёт массы поставленных комплексов')
