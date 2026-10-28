"""Fixed-step fleet simulation; all energy and sand are finite and conserved."""
from collections import deque
from enum import Enum
from datetime import date, timedelta
import math
import numpy as np
from .config import Config, KWH
from .terrain import Terrain
from .drone import Drone, DroneState as DS
from .solar_base import SolarBase
from .planner import Planner
from .dust import Dust
from .traffic import conflicts
from .equipment import resolve_config,retained_energy,station_offset
from .laser import prepare_links
from .soil import SoilLoad,cargo_volume


class MissionState(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"


def elapsed_text(seconds):
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days} д {hours:02}:{minutes:02}:{seconds:02}"


class Simulation:
    shared_site=None
    def __init__(self, config=None,terrain=None,initial_stock=None):
        self.config = config or Config()
        self.config.validate()
        resolve_config(self.config);self.config.validate()
        from .base_layout import configure_layout
        self.initial_stock=initial_stock or []
        extra=[(r['count']*(r['mass_kg']+(self.config.logistics.charger_slot_mass_kg if self.config.logistics and self.config.logistics.individual_chargers else 0.)),r['count']*r['capacity_kwh']) for r in self.initial_stock]
        layout=configure_layout(self.config,extra)
        self.terrain = Terrain(self.config) if terrain is None else terrain
        b = self.config.base
        self.base = SolarBase(b, np.array([b.x, b.y, self.terrain.sample(b.x, b.y)]))
        if layout is not None:
            self.base.layout_spec=layout
            side=self.terrain.cell*self.config.terrain.n
            def pads_clear():
                return all(self._safe_pad(self.slot_position(i)) for i in range(self.config.fleet.charge_slots)) and all(
                    self._safe_pad(self.wait_position(i)) for i in range(self.config.fleet.count))
            if not self.base.safe_path(self.base.position[:2],self.terrain) or not pads_clear():
                original=self.base.position.copy()
                candidates=sorted([(float(x),float(y)) for x in np.linspace(b.radius+1,side-b.radius-1,15)
                    for y in np.linspace(b.radius+1,side-b.radius-1,15)],key=lambda p:(p[0]-b.x)**2+(p[1]-b.y)**2)
                for point in candidates:
                    self.base.position=np.array([*point,self.terrain.sample(*point)])
                    if self.base.safe_path(point,self.terrain) and pads_clear():
                        b.x,b.y=point;break
                else:self.base.position=original
        self.drones = []
        self.slots = [None]*self.config.fleet.charge_slots
        self.queue = []
        for index, model in enumerate(self.config.drone_models()):
            slot = index if index < len(self.slots) else None
            point = self.slot_position(slot) if slot is not None else self.wait_position(index)
            d = Drone(model, point, identifier=index, slot=slot,serial=index+1)
            if slot is None:
                d.state = DS.WAIT_GROUND
                if d.charge_requested:
                    self.queue.append(index)
            else:
                self.slots[slot] = index
            self.drones.append(d)
        self.planner = Planner(self.config)
        self.dust = Dust(self.config)
        self.time = 0.0
        self.state = MissionState.RUNNING
        self.reason = ""
        self.events = deque(maxlen=self.config.event_limit)
        self.event_serial = 0
        self._event_keys = {}
        self.trips = 0
        self.excavated_mass = 0.0
        self.filled_mass = 0.0
        self.excavated_volume=0.
        self.filled_volume=0.
        self.landing_requested = False
        self.pending_equipment = None
        self.retired_drones = []
        self._next_serial = len(self.drones)+1
        self.equipment_exported_energy = 0.0
        self.equipment_imported_energy = 0.0
        self._relocation_attempt_time = -float("inf")
        self.relocation_pending=None
        self.base_move_wait=''
        self._base_front_anchor=(self.terrain.front.settings['base_chainage_m'] if self.terrain.front is not None else 0.)
        self._terrain_active = self.config.terrain.roughness > 0
        self._initial_energy = sum(d.battery for d in self.drones)+self.base.battery
        self._starts = {d.identifier:d.position.copy() for d in self.drones}
        self.laser_links = []
        self.event(f"Сценарий готов: дронов {len(self.drones)}, зарядных слотов {len(self.slots)}")
        points = [self.slot_position(i) for i in range(len(self.slots))]+[self.wait_position(i) for i in range(len(self.drones))]
        if not self.base.safe_path(self.base.position[:2], self.terrain) or any(not self._safe_pad(p) for p in points):
            self.block("База или посадочные площадки пересекают канаву, отвал либо край карты")
        elif self.terrain.dump_space < max(0, self.terrain.remaining_cut-self.terrain.remaining_fill):
            self.block("Недостаточно места в отвале для проектного объёма песка")
        self.logistics = None
        if self.config.logistics is not None:
            from .logistics import Logistics
            self.logistics = Logistics(self)

    @property
    def drone(self):
        return self.drones[0]  # compatibility for scripts using the single drone

    @property
    def trip(self):
        return self.drone.trip

    @trip.setter
    def trip(self, value):
        self.drone.trip = value

    @property
    def minimum_drone_charge(self):
        return self.drone.minimum_charge

    def slot_position(self, slot):
        offset = station_offset(slot,self.config.fleet.pad_spacing)
        if self.base.config.layout is None:return self.base.position+np.array([0,offset,self.config.base.roof_height])
        from .base_layout import rotate_offset
        xy=rotate_offset(self.base.config.layout,[0,offset])
        return self.base.position+np.array([*xy,self.config.base.roof_height])

    def base_specification(self):
        """Nominal envelope from existing geometry; live mass from actual equipment."""
        b=self.base;diameter=2*b.config.radius
        offsets=[station_offset(i,self.config.fleet.pad_spacing) for i in range(len(self.slots))]
        docked=sum(d.config.empty_mass+d.cargo for d in self.drones if d.docked)
        rect=self.logistics.field_rect if self.logistics is not None else None
        result=dict(body_dimensions_m=[diameter,diameter,b.config.roof_height],
                    landing_zone_dimensions_m=[diameter,max(offsets)-min(offsets)+diameter],
                    solar_field_dimensions_m=list(rect[2:]) if rect is not None else None,
                    installed_equipment_mass_kg=b.config.equipment.mass,
                    stored_items_mass_kg=b.config.mass-b.config.equipment.mass,
                    docked_drones_mass_kg=docked,gross_mass_kg=b.config.mass+docked)
        if b.config.layout is not None:
            from .base_layout import support_report
            result['body_dimensions_m']=list(b.layout_spec['dimensions_m'])
            result['landing_zone_dimensions_m']=[max(diameter,b.layout_spec['dimensions_m'][0]),max(max(offsets)-min(offsets)+diameter,b.layout_spec['dimensions_m'][1])]
            result['layout']=b.layout_spec;result['ground_support']=support_report(b,self.terrain,self.drones)
        return result

    def wait_position(self, index):
        f = self.config.fleet
        from .base_layout import rotate_offset
        if self.base.config.layout is None:x,y=self.base.position[0]-f.waiting_offset,self.base.position[1]+station_offset(index,f.pad_spacing)
        else:x,y=self.base.position[:2]+rotate_offset(self.base.config.layout,[-f.waiting_offset,station_offset(index,f.pad_spacing)])
        return np.array([x, y, self.terrain.sample(x, y)+f.ground_clearance])

    def cruise(self, d):
        surface = self.terrain.clearance_surface if self.config.reference_height is not None or self.shared_site is not None else max(float(self.terrain.reference.max()), self.config.terrain.dump_max_height)
        return surface+d.config.work_height+self.config.base.roof_height+d.identifier*self.config.fleet.altitude_spacing

    def _safe_pad(self, p):
        t = self.terrain
        side = t.config.terrain.n*t.cell
        if not (t.cell < p[0] < side-t.cell and t.cell < p[1] < side-t.cell):
            return False
        ix, iy = int(p[0]/t.cell), int(p[1]/t.cell)
        margin = self.config.fleet.separation
        patch=t.patch(*p[:2],margin)
        near = (t.x[patch]-p[0])**2+(t.y[patch]-p[1])**2 < margin**2
        blocked=t.design_work[patch]|t.dump_mask[patch]
        if t.protected is not None:blocked |= t.protected[patch]
        return not np.any(blocked&near) and not (t.excavation[iy,ix] or t.fill_mask[iy,ix])

    def apply_plan(self, plan):
        if self.shared_site is not None:return self.shared_site.apply_plan(plan)
        from copy import deepcopy
        if self.pending_equipment is not None:raise ValueError("Сначала дождитесь установки оборудования или отмените её")
        if self.base.destination is not None or not all(d.grounded and abs(d.cargo) <= 1e-8 for d in self.drones):
            raise ValueError("Применяйте план после посадки всех дронов и выгрузки груза; при полёте можно подготовить рисунок")
        if self.logistics is not None:
            w=self.logistics;e=w.excavator
            if w.group is not None or w.active or e is not None and (e.bucket_kg>0 or e.active_container is not None):
                raise ValueError('Дождитесь завершения цикла ковша и посадки логистического флота')
            if any(c.soil_kg>0 for c in w.containers):
                raise ValueError('Перед изменением плана завершите выгрузку контейнеров командой посадки флота')
        candidate = deepcopy(self.terrain)
        candidate.set_plan(plan)
        if self.logistics is not None and self.logistics.field_rect is not None:
            x,y,w,h=self.logistics.field_rect
            area=(candidate.x>=x)&(candidate.x<x+w)&(candidate.y>=y)&(candidate.y<y+h)
            if np.any(area&candidate.work_mask):raise ValueError('План пересекает поле солнечных панелей')
        points = [self.slot_position(i) for i in range(len(self.slots))]+[self.wait_position(i) for i in range(len(self.drones))]
        for p in points:
            near = (candidate.x-p[0])**2+(candidate.y-p[1])**2 < self.config.fleet.separation**2
            if np.any(candidate.work_mask & near):
                raise ValueError("Кисть пересекает базу или посадочную площадку")
        t = self.terrain
        for key in ("target", "segment_labels", "excavation", "fill_mask", "bottom", "_cut_indices", "_fill_indices"):
            setattr(t, key, getattr(candidate, key))
        t.revision += 1
        t.initial_excavation = t.remaining
        self.config.plan = deepcopy(plan)
        if not (plan.channel or {}).get('operations'):
            t.front=None;self.relocation_pending=None;self.base_move_wait=''
            t.design_work=t.work_mask
        t.minimum_surface=float(min(t.reference.min(),t.target.min()))
        t._clearance_surface=max(float(t.reference.max()),float(t.dump_cap.max()),float(t.height.max()),float(t.target.max()))
        if self.config.georeference and self.config.georeference.get('channel'):
            if (plan.channel or {}).get('project_id')!=self.config.georeference['channel']['project_id']:
                self.config.georeference.pop('channel')
        self.config.initial_height = t.height.tolist()
        if t.soil is not None:self.config.initial_soil=t.soil.export()
        self.config.base.x,self.config.base.y = map(float,self.base.position[:2])
        self.state, self.reason = MissionState.RUNNING, ""
        self.landing_requested = False
        self._terrain_active = True
        if self.logistics is not None:
            for job in self.logistics.jobs:
                if job.status=='QUEUED':self.logistics.cancel_job(job.id)
            self.logistics.progress_at=self.time
        self.event(f"Применён план: сегментов {len(plan.segments)}, выемка {t.remaining_cut:.3f} м³, засыпка {t.remaining_fill:.3f} м³")

    def request_landing(self):
        self.landing_requested = True
        self.cancel_relocation()
        if self.base.destination is None:self.base_move_wait=''
        self.event("Запрошена посадка флота: завершение текущих переносов, без новых рейсов")

    def cancel_relocation(self):
        if self.base.destination is not None:
            if self.relocation_pending is not None:raise ValueError('База уже движется; дождитесь остановки')
            return False
        if self.relocation_pending is None:return False
        self.relocation_pending=None;self.base_move_wait='';self._relocation_attempt_time=self.time
        w=self.logistics
        if w is not None and w.base_pack_elapsed:
            # Reverse only the part already folded. Deployed area is unchanged
            # at cancellation and unfolding still consumes finite station power.
            w.field_elapsed=max(0.,w.config.field_deployment_seconds-w.base_pack_elapsed)
            w.base_pack_elapsed=0.;w.field_ready=False
        self.event('Подготовка переезда отменена; панели разворачиваются, маршрут будет перепланирован')
        return True

    @property
    def base_move_status(self):
        labels={'LANDING_FLEET_FOR_BASE_MOVE':'завершает рейсы и ждёт посадки флота',
                'WAITING_SERVICE_COMPLETION':'завершает начатую смену батарей',
                'WAITING_GROUP_LIFT':'ждёт завершения совместного переноса',
                'WAITING_BASE_ENERGY':'накапливает энергию для переезда и развёртывания панелей',
                'WAITING_BASE_DRIVE_POWER':'недостаточная мощность привода от батарей или инвертора',
                'FOLDING_SOLAR_FIELD':'складывает солнечные панели',
                'UNFOLDING_SOLAR_FIELD':'завершает развёртывание солнечных панелей',
                'MOVING_WITH_WORKFRONT':'переезжает вдоль канала',
                'NO_SAFE_FORWARD_BASE_ROUTE':'ищет безопасный маршрут вперёд'}
        return labels.get(self.base_move_wait,'на рабочей позиции')

    def set_panel_area(self, area):
        from copy import deepcopy
        from .equipment import set_panel_area,resolve_config
        if area==self.base.config.panel_area:return 'applied'
        candidate=deepcopy(self.config)
        set_panel_area(candidate.base.equipment,area,self.base.config.panel_efficiency)
        resolve_config(candidate);candidate.validate()
        if self.logistics is not None:
            self.logistics.configure_panels(candidate.base.equipment.panels)
            return 'applied'
        return self.request_equipment(candidate)

    def set_base_battery_count(self, count):
        from copy import deepcopy
        from .equipment import set_base_battery_count
        if type(count) is int and count==sum(b.quantity for b in self.base.config.equipment.batteries):return 'applied'
        candidate=deepcopy(self.config)
        set_base_battery_count(candidate.base.equipment,count)
        return self.request_equipment(candidate)

    def configure_platforms(self,settings):
        from copy import deepcopy
        from .base_layout import parameters,solve_layout,apply_live_layout,plan_heading
        from .power_design import equipment_copy
        if self.logistics is None:raise ValueError('Платформы настраиваются в ATS 2.0')
        if self.base.destination is not None or not all(d.grounded for d in self.drones) or self.logistics.group:
            raise ValueError('Для изменения шасси остановите базу и посадите дронов')
        c=equipment_copy(self.config);c.base.layout=parameters(settings)
        if c.base.layout['auto_heading']:c.base.layout['heading_deg']=plan_heading(c)
        spec=solve_layout(c,self.logistics);candidate=deepcopy(self.base);candidate.config=c.base;candidate.layout_spec=spec
        c.base.radius=spec['dimensions_m'][0]/2;c.base.roof_height=spec['dimensions_m'][2]
        c.base.mass=self.base.config.mass+spec['chassis_mass_kg']-self.base.config.equipment.chassis_mass_kg
        if not candidate.safe_path(candidate.position[:2],self.terrain):raise ValueError('Платформы не помещаются на текущей площадке или превышают несущую способность')
        self.base.config.layout=c.base.layout;apply_live_layout(self.logistics,spec);self.logistics.update_base_mass()
        for d in self.drones:
            if d.docked:d.position=self.slot_position(d.slot)
        self.event(f"Автокомпоновка: {spec['platform_count']} платформ, шасси {spec['chassis_mass_kg']:.0f} кг")
        return spec

    def request_equipment(self,proposed):
        """Apply stationary base changes now; airborne workshop work waits for landing."""
        if getattr(self,'regional_stage',None) in ('PACK','DRIVE','ARRIVE'):raise ValueError('Сначала завершите или отмените движение; поставки запасных батарей доступны на складе')
        from copy import deepcopy
        from dataclasses import asdict
        if proposed.soil_profile!=self.config.soil_profile:
            raise ValueError('Свойства грунта задаются до земляных работ; загрузка сохраняет прежний состав')
        if self.logistics is not None:
            # Old live equipment imports remain available in the legacy engine.
            # ATS 2.0 cannot fabricate physical inventory through recipe editing.
            old,new=asdict(self.config.base.equipment),asdict(proposed.base.equipment)
            proposed.validate()
            controls=('empty_speed','loaded_speed','reserve','work_height','scoop_seconds','dig_rate_kg_s','digging_j_m3','dump_seconds','winch_seconds','initial_charge')
            if any(any(getattr(d.config,key)!=getattr(m,key) for key in controls) for d,m in zip(self.drones,proposed.drone_models())):
                raise ValueError('ATS 2.0: скорости, операции и стартовый заряд меняются в новом эксперименте; порог зарядки и мощность лазера доступны сразу')
            for values in (old,new):
                values.pop('charge_power_kw');values.pop('charge_efficiency')
                values.pop('panels')
                values.pop('batteries')
                for emitter in values['emitters']:
                    emitter.pop('enabled');emitter.pop('input_power_kw')
            if old!=new or [asdict(d.config.equipment) for d in self.drones]!=[asdict(d.equipment) for d in proposed.drone_models()]:
                raise ValueError('ATS 2.0: изменение физических предметов требует нового эксперимента; зарядка, смена готовых батарей и перевозки выполняются автоматически')
            panels_changed=asdict(self.config.base.equipment)['panels']!=asdict(proposed.base.equipment)['panels']
            batteries_changed=asdict(self.config.base.equipment)['batteries']!=asdict(proposed.base.equipment)['batteries']
            if panels_changed:
                if self.base.destination is not None:raise ValueError('Дождитесь остановки базы перед изменением панелей')
                self.logistics.solar_field_site(proposed.base.equipment.panel_area)
            if batteries_changed:self.logistics.configure_base_batteries(proposed.base.equipment.batteries)
            if panels_changed:
                self.logistics.configure_panels(proposed.base.equipment.panels)
            self.config.base.equipment=deepcopy(proposed.base.equipment)
            self.config.base.charge_power=proposed.base.equipment.charge_power_kw*1000
            self.config.base.charge_efficiency=proposed.base.equipment.charge_efficiency
            self.logistics.update_base_mass()
            for d,model,saved in zip(self.drones,proposed.drone_models(),self.config.fleet.models):
                d.config.name=saved.name=model.name
                d.config.charge_threshold=saved.charge_threshold=model.charge_threshold
            self.event('Обновлены лимиты зарядки / включение и мощность лазеров')
            return 'applied'
        candidate=deepcopy(self.config)
        candidate.base.equipment=deepcopy(proposed.base.equipment)
        candidate.fleet.models=proposed.drone_models();candidate.fleet.count=len(candidate.fleet.models)
        candidate.fleet.wait_mode=proposed.fleet.wait_mode
        resolve_config(candidate);candidate.validate()
        count,slots=candidate.fleet.count,candidate.fleet.charge_slots
        for index in range(slots):
            if not self._safe_pad(self.slot_position(index)):raise ValueError("Новое посадочное место пересекает план, отвал или край карты")
        for index in range(count):
            if not self._safe_pad(self.wait_position(index)):raise ValueError("Площадка нового дрона пересекает план или край карты")
        if any(v is not None and v<count for v in self.slots[slots:]):
            raise ValueError("Можно удалить только свободное посадочное место; дождитесь вылета занимающего его дрона")
        changed=count!=len(self.drones) or any(asdict(d.config)!=asdict(m) for d,m in zip(self.drones,candidate.fleet.models))
        if self.base.destination is not None or (changed and not all(d.grounded and d.cargo==0 for d in self.drones)):
            if self.pending_equipment is None:self._equipment_previous=(self.state,self.reason,self.landing_requested)
            self.pending_equipment=candidate;self.request_landing()
            self.state,self.reason=MissionState.RUNNING,""
            self.event("Комплектация ожидает остановки базы и посадки флота; груз будет доставлен до установки")
            return "pending"
        self._install_equipment(candidate)
        return "applied"

    def cancel_equipment(self):
        if self.pending_equipment is not None:
            self.state,self.reason,self.landing_requested=self._equipment_previous
            self.pending_equipment=None;self.event("Ожидающая установка оборудования отменена")

    def _install_equipment(self,candidate):
        count,slots=candidate.fleet.count,candidate.fleet.charge_slots
        if any(v is not None and v<count for v in self.slots[slots:]):
            raise ValueError("Удаляемое посадочное место ещё занято")
        available=self.slots[:slots]+[None]*max(0,slots-len(self.slots))
        available=[v if v is None or v<count else None for v in available]
        additions=[];positions=[d.position for d in self.drones[:count]]
        for index in range(len(self.drones),count):
            slot=next((i for i,v in enumerate(available) if v is None),None)
            point=self.slot_position(slot) if slot is not None else self.wait_position(index)
            if any(np.linalg.norm(point-p)<candidate.fleet.separation-1e-9 for p in positions):raise ValueError("Площадка доставки нового дрона занята")
            model=candidate.fleet.models[index]
            if slot is None:
                probe=Drone(model,point,identifier=index)
                if not any(probe.battery>=model.battery*model.reserve+self._transfer_energy(probe,i) for i in range(slots)):
                    raise ValueError("Новому дрону нужен заряд для перелёта с площадки или свободное зарядное место")
            additions.append((index,slot,point));positions.append(point)
            if slot is not None:available[slot]=index
        old_base=self.config.base.equipment
        new_energy=retained_energy(old_base.batteries,candidate.base.equipment.batteries,self.base.battery)
        self.equipment_exported_energy+=self.base.battery-new_energy
        self.base.battery=new_energy
        # Removal is external transport from the ground; sand and flight history stay.
        for d in self.drones[count:]:
            self._clear_slot(d);self.equipment_exported_energy+=d.battery;d.retired_at=self.time;self.retired_drones.append(d)
        self.drones=self.drones[:count]
        self.slots=self.slots[:slots]+[None]*max(0,slots-len(self.slots))
        for d,model in zip(self.drones,candidate.fleet.models):
            new_energy=retained_energy(d.config.equipment.batteries,model.equipment.batteries,d.battery)
            self.equipment_exported_energy+=d.battery-new_energy
            d.battery=new_energy;d.config=model;d.minimum_charge=min(d.minimum_charge,d.battery)
            if d.battery<d.config.battery-1e-7 and d.battery<=d.config.battery*d.config.charge_threshold:d.charge_requested=True
        self.config=candidate;self.base.config=candidate.base;self.terrain.config=candidate
        self.planner.config=candidate;self.dust.config=candidate.dust
        for index,slot,point in additions:
            d=Drone(candidate.fleet.models[index],point,identifier=index,slot=slot,serial=self._next_serial,created_at=self.time)
            self._next_serial+=1;self.equipment_imported_energy+=d.battery
            if slot is None:d.state=DS.WAIT_GROUND
            else:self.slots[slot]=index
            self.drones.append(d)
        self.queue=[i for i in self.queue if i<count and self.drones[i].charge_requested]
        for d in self.drones:
            if d.state==DS.WAIT_GROUND:self._enqueue(d)
        self._capability_key=None
        if self.pending_equipment is not None:
            state,reason,landing=self._equipment_previous
            self.state,self.reason,self.landing_requested=(MissionState.RUNNING,"",landing) if state==MissionState.BLOCKED else (state,reason,landing)
        elif self.state==MissionState.BLOCKED:self.state,self.reason=MissionState.RUNNING,""
        self.pending_equipment=None
        self.laser_links=[];self.base.laser_input_power=0.
        for d in self.drones:d.laser_power_w=0.;d.laser_status="Ожидание следующего шага"
        self.event(f"Оборудование установлено без сброса: база {candidate.base.mass:.1f} кг, панели {candidate.base.panel_area:.1f} м², слотов {slots}, дронов {count}; энергия снятого оборудования {self.equipment_exported_energy/KWH:.3f} кВт·ч")

    @property
    def hour(self):
        return (self.config.start_hour+self.time/3600) % 24

    def day_at(self, seconds):
        passed_midnights = int((self.config.start_hour*3600+seconds)//86400)
        return (self.config.base.start_day_of_year-1+passed_midnights) % 365+1

    @property
    def day_of_year(self):
        return self.day_at(self.time)

    @property
    def clock_text(self):
        seconds = int(round((self.config.start_hour*3600+self.time) % 86400)) % 86400
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    @property
    def date_text(self):
        return (date(2025, 1, 1)+timedelta(days=self.day_of_year-1)).strftime("%d.%m")

    def event(self, message, key=None):
        if key is not None and self._event_keys.get(key) == message:
            return
        if key is not None:
            self._event_keys[key] = message
        self.event_serial += 1
        self.events.append((self.event_serial, self.time, message))

    def drone_event(self, d, message, key=None):
        self.event(f"Дрон {d.serial} ({d.config.name}): {message}", (d.serial,key) if key else None)

    def block(self, reason):
        self.state = MissionState.BLOCKED
        self.reason = reason
        self.event(f"Работа заблокирована: {reason}")

    def consume_drone(self, joules, drone=None):
        d = drone or self.drone
        supplied=d.laser_step_energy
        if self.logistics is not None and joules-supplied>sum(self.logistics.inventory.items[id].max_discharge_w for id in d.battery_ids)*self.config.dt+1e-8:
            self._settle_laser(d);self.block(f'BLOCKED_BATTERY_POWER: предел разряда модулей дрона {d.serial}')
            return False
        if joules > d.battery+supplied+1e-9:
            self._settle_laser(d)
            self.block(f"Недостаточно энергии дрона {d.serial}; расчёт остановлен до разряда ниже нуля")
            return False
        d.laser_step_energy=0.
        energy=d.battery+supplied-joules
        self._laser_spill(d,max(0.,energy-d.config.battery))
        d.battery = max(0,min(d.config.battery,energy))
        d.consumed += joules
        d.minimum_charge = min(d.minimum_charge, d.battery)
        return True

    def _laser_spill(self,d,energy):
        if energy<=0:return
        self.base.laser_losses+=energy;d.laser_received-=energy
        d.laser_power_w=max(0.,d.laser_power_w-energy/self.config.dt)
        for link in self.laser_links:
            if link.drone is d:link.received_w=d.laser_power_w

    def _settle_laser(self,d):
        # If an earlier actor stopped the engine, every delivered joule still
        # ends up in storage or explicitly reported heat, never an orphan buffer.
        energy=d.laser_step_energy;d.laser_step_energy=0.
        stored=min(energy,max(0.,d.config.battery-d.battery))
        d.battery+=stored;self._laser_spill(d,energy-stored)

    def flight_power(self,d):
        movement=d.state in (DS.TAKEOFF,DS.TO_DIG,DS.TO_DUMP,DS.RETURN,DS.LANDING,DS.TRANSFER_TAKEOFF,DS.TO_SLOT,DS.SLOT_LANDING,DS.WAIT_LANDING)
        winching=d.state in (DS.LOWER_GRABBER,DS.LIFT_GRABBER,DS.DUMP)
        lift_height=max(0,d.position[2]-self.terrain.sample(*d.position[:2])-d.config.stowed_cable) if d.state==DS.LIFT_GRABBER else 0
        digging=0.
        if d.state==DS.SCOOP and d.trip is not None:
            digging=self._scoop_request(d,d.elapsed,self.config.dt)/self.config.terrain.density*d.config.digging_j_m3/self.config.dt
            if self.terrain.soil is not None:digging=d._scoop_preview.effort_volume*d.config.digging_j_m3/self.config.dt
        return d.power(movement,winching,lift_height=lift_height)+digging

    def _scoop_request(self,d,elapsed,dt):
        trip,c=d.trip,d.config
        duration=c.scoop_duration(trip.mass)*trip.soil_factor
        request=min(max(0.,trip.mass-d.cargo),trip.mass/duration*min(dt,max(0.,duration-elapsed)))
        if self.terrain.soil is not None:
            from .equipment import physical_payload_limit
            request=min(request,max(0.,physical_payload_limit(d)-d.cargo))
            volume=max(0.,c.equipment.bucket.volume_m3-(d.soil.volume if d.soil is not None else 0.))
            d._scoop_preview=self.terrain.preview_load(*trip.dig,request,trip.source,c.grab_radius,max_volume=volume)
            return d._scoop_preview.mass
        return min(request,self.terrain.available_mass(*trip.dig,trip.source,c.grab_radius))

    def _clear_slot(self, d):
        if d.slot is not None and self.slots[d.slot] == d.identifier:
            self.slots[d.slot] = None
        d.slot = None

    def _enqueue(self, d):
        if d.charge_requested and d.identifier not in self.queue:
            self.queue.append(d.identifier)

    def _reserve_return(self, d):
        limit=self.pending_equipment.fleet.charge_slots if self.pending_equipment is not None else len(self.slots)
        available = next((i for i,v in enumerate(self.slots[:limit]) if v is None), None)
        if available is not None and not self.queue:
            d.slot = available
            self.slots[available] = d.identifier
            d.destination = self.slot_position(available)
        else:
            d.destination = self.wait_position(d.identifier)
        d.detour = None

    def _assign_slots(self):
        if not self.queue or self.base.destination is not None or (self.logistics is None and (self.landing_requested or self.terrain.remaining <= self.config.terrain.completion_volume)):
            return
        for slot in range(len(self.slots)):
            if self.slots[slot] is not None:
                continue
            eligible = next((i for i in self.queue if self.drones[i].charge_requested and self.drones[i].state in (DS.WAIT_GROUND,DS.WAIT_HOVER) and self.drones[i].battery < self.drones[i].config.battery-1e-7 and self.drones[i].battery >= self.drones[i].config.battery*self.drones[i].config.reserve+self._transfer_energy(self.drones[i],slot)), None)
            if eligible is None:
                break
            d = self.drones[eligible]
            self.queue.remove(eligible)
            self.slots[slot] = eligible
            d.slot = slot
            d.destination = self.slot_position(slot)
            if d.state == DS.WAIT_GROUND:
                d.transition(DS.TRANSFER_TAKEOFF)
            else:
                d.transition(DS.TO_SLOT)
            self.drone_event(d, f"получен зарядный слот {slot+1}")

    def _transfer_energy(self,d,slot):
        c=d.config
        altitude=self.cruise(d)
        target=self.slot_position(slot)
        seconds=math.ceil(max(0,altitude-target[2])/c.vertical_speed)+math.ceil(np.linalg.norm(target[:2]-d.position[:2])/c.empty_speed)+3
        if d.grounded:
            seconds+=math.ceil(max(0,altitude-d.position[2])/c.vertical_speed)
        return d.power(True)*seconds

    def _move(self, d, destination, next_state, dt):
        previous = d.position.copy()
        target = np.asarray(d.detour if d.detour is not None else destination)
        arrived = d.move_towards(target,dt)
        other = conflicts(d,previous,d.position,self.drones,self._starts,self.config.fleet.separation)
        if other is None and self.shared_site is not None:other=self.shared_site.collision(self,d,previous,d.position)
        if other is not None:
            d.position[:] = previous
            d.velocity[:] = 0
            d.traffic_yields += 1
            self.drone_event(d,f"уступает путь дрону {other.serial}","traffic")
            direction = np.asarray(destination)[:2]-previous[:2]
            length = float(np.linalg.norm(direction))
            if length > 1e-9:
                normal = np.array([-direction[1],direction[0]])/length
                side = self.config.terrain.n*self.terrain.cell
                for sign in (1,-1):
                    point = previous.copy()
                    point[:2] += sign*normal*self.config.fleet.detour_distance
                    if (np.all(point[:2] > self.terrain.cell) and np.all(point[:2] < side-self.terrain.cell) and
                            conflicts(d,previous,point,self.drones,self._starts,self.config.fleet.separation) is None and
                            (self.shared_site is None or self.shared_site.collision(self,d,previous,point) is None)):
                        d.detour = point
                        break
            return
        self._event_keys.pop((d.identifier,"traffic"),None)
        distance = np.linalg.norm(d.position-previous)
        count = max(2,int(math.ceil(distance/(self.terrain.cell/2)))+1)
        for p in np.linspace(previous,d.position,count):
            ground = self.terrain.sample(*p[:2])
            if self.shared_site is not None:ground=max(ground,self.shared_site.obstacle_height(p))
            if np.linalg.norm(p[:2]-self.base.position[:2]) <= self.config.base.radius:
                ground = max(ground,self.base.position[2]+self.config.base.roof_height)
            if p[2] < ground-1e-9:
                d.position[:] = previous
                d.velocity[:] = 0
                self.block(f"Траектория дрона {d.serial} пересекает рельеф или базу")
                return
        if arrived:
            if d.detour is not None:
                d.detour = None
            else:
                d.transition(next_state)

    def _reserved_dig(self,d):
        working = (DS.TAKEOFF,DS.TO_DIG,DS.LOWER_GRABBER,DS.SCOOP,DS.LIFT_GRABBER)
        points=[o.trip.dig for o in self.drones if o is not d and o.trip is not None and o.state in working]
        if self.logistics is not None and self.logistics.excavator is not None:
            points.append(tuple(self.logistics.excavator.position[:2]))
        if self.shared_site is not None:points+=self.shared_site.reservations(self,'dig')
        return points

    def _reserved_dump(self,d):
        points=[o.trip.dump for o in self.drones if o is not d and o.trip is not None and o.state not in (DS.RETURN,DS.LANDING)]
        if self.logistics is not None:
            points += [j.destination[:2] for j in self.logistics.jobs
                       if j.kind=='PICKUP_FULL_CONTAINER' and j.status in ('ACTIVE','QUEUED')]
        if self.shared_site is not None:points+=self.shared_site.reservations(self,'dump')
        return points

    def _choose_trip(self, d, battery=None, unreserved=False, airborne=False):
        incoming = sum(o.trip.mass for o in self.drones if o is not d and o.trip is not None and o.trip.destination == "fill" and not o.dump_released and o.state not in (DS.RETURN,DS.LANDING))
        if self.terrain.soil is not None:
            incoming=sum(o.trip.volume or 0. for o in self.drones if o is not d and o.trip is not None and o.trip.destination=='fill' and not o.dump_released and o.state not in (DS.RETURN,DS.LANDING))
            if self.logistics is not None:
                incoming+=sum(self.logistics.inventory.items[j.resource_ids[0]].soil_volume(self.config.terrain.density)
                    for j in self.logistics.jobs if j.kind=='PICKUP_FULL_CONTAINER' and j.soil_destination=='fill' and j.status in ('ACTIVE','QUEUED'))
        home = self.slot_position(d.slot) if d.slot is not None else self.wait_position(d.identifier)
        return self.planner.choose(self.terrain,self.base,d,() if unreserved else self._reserved_dig(d),home,() if unreserved else self._reserved_dump(d),0 if unreserved else incoming,battery,d.position if airborne or self.terrain.front is not None else None)

    def _request_charge(self, d, reason):
        if not d.charge_requested:
            self.drone_event(d, f"нужна зарядка: {reason}; текущий заряд {100*d.battery/d.config.battery:.1f}%")
        d.charge_requested = True

    def _charging_needed(self, d):
        if d.charge_requested and d.battery >= d.config.battery-1e-7:
            d.charge_requested = False
            d.charge_cycles += 1
            self.drone_event(d, "зарядка завершена: 100%")
        if d.battery < d.config.battery-1e-7 and d.battery <= d.config.battery*d.config.charge_threshold and self.terrain.remaining > self.config.terrain.completion_volume:
            self._request_charge(d, "достигнут нижний порог")
        return d.charge_requested

    def _docked(self,d):
        if self.base.destination is not None or self.relocation_pending is not None:
            return
        if self.landing_requested or self.terrain.remaining <= self.config.terrain.completion_volume:
            return
        if self._charging_needed(d):
            d.state = DS.CHARGING
            self.drone_event(d,"заряжается до 100% после запроса зарядки","charge")
            return
        d.state = DS.DOCKED
        if d.trip is None:
            d.trip = self._choose_trip(d)
        if d.trip is None:
            if self._choose_trip(d, unreserved=True) is None and self._choose_trip(d, battery=d.config.battery, unreserved=True) is not None:
                self._request_charge(d, "не хватает энергии на следующий рейс с резервом")
            self.drone_event(d,"ожидает свободного участка выемки","job")
            return
        self._event_keys.pop((d.identifier,"job"),None)
        self._event_keys.pop((d.identifier,"charge"),None)
        d.transition(DS.TAKEOFF)
        self.drone_event(d,f"вылет с зарядом {100*d.battery/d.config.battery:.1f}%; план до {d.trip.mass:.1f} кг")

    def _ready_ground(self,d):
        if self.landing_requested or self.relocation_pending is not None or self.base.destination is not None or self.terrain.remaining <= self.config.terrain.completion_volume:
            return
        if self._charging_needed(d):
            self._enqueue(d)
            return
        trip=self._choose_trip(d)
        if trip is not None:
            d.trip=trip
            if d.identifier in self.queue:self.queue.remove(d.identifier)
            d.transition(DS.TAKEOFF)
            self.drone_event(d,f"вылет с наземной площадки с зарядом {100*d.battery/d.config.battery:.1f}%")
        elif self._choose_trip(d, unreserved=True) is None and self._choose_trip(d, battery=d.config.battery, unreserved=True) is not None:
            self._request_charge(d, "не хватает энергии на рейс с резервом")
            self._enqueue(d)

    def _continue_or_return(self, d):
        d.trip = None
        d.dump_released = False
        if (not self.landing_requested and self.relocation_pending is None and self.terrain.remaining > self.config.terrain.completion_volume
                and (self.logistics is None or not self.logistics.direct_should_return(d))):
            if not self._charging_needed(d):
                trip = self._choose_trip(d, airborne=True)
                if trip is not None:
                    d.trip = trip
                    d.transition(DS.TO_DIG)
                    self.drone_event(d, f"продолжает работу без зарядки: {100*d.battery/d.config.battery:.1f}%")
                    return
                if self._choose_trip(d, unreserved=True, airborne=True) is None and self._choose_trip(d,battery=d.config.battery,unreserved=True,airborne=True) is not None:
                    self._request_charge(d,"не хватает энергии на следующий рейс с возвращением и резервом")
        d.transition(DS.RETURN)
        self._reserve_return(d)

    def _landed(self,d):
        d.trip = None
        d.destination = None
        d.detour = None
        if d.slot is None:
            self._enqueue(d)
            self.drone_event(d,"посадка на площадку ожидания; очередь на зарядку","queue")
        else:
            self.drone_event(d,f"посадка в слот {d.slot+1}")

    def _airborne(self,d,dt):
        c, trip = d.config,d.trip
        d.last_flight_power_w=self.flight_power(d)
        if not self.consume_drone(d.last_flight_power_w*dt,d):
            return
        d.airborne_seconds+=dt;d.flight_seconds+=dt
        d.longest_flight_seconds=max(d.longest_flight_seconds,d.flight_seconds)
        # A fulfilled wireless charge must release an air queue/reserved socket.
        if self.logistics is None and d.charge_requested and d.battery>=c.battery-1e-7 and d.state in (DS.RETURN,DS.LANDING,DS.WAIT_HOVER,DS.WAIT_LANDING,DS.TRANSFER_TAKEOFF,DS.TO_SLOT,DS.SLOT_LANDING) and not self.landing_requested and self.terrain.remaining>self.config.terrain.completion_volume:
            self._charging_needed(d)
            next_trip=self._choose_trip(d,airborne=True)
            if next_trip is not None:
                self._clear_slot(d)
                if d.identifier in self.queue:self.queue.remove(d.identifier)
                d.trip=trip=next_trip;d.destination=None;d.detour=None;d.transition(DS.TO_DIG)
                self.drone_event(d,"заряжен лазером до 100%: продолжает работу в воздухе")
        d.elapsed += dt
        if d.state == DS.TAKEOFF:
            self._move(d,[*d.position[:2],trip.altitude],DS.TO_DIG,dt)
            if d.state == DS.TO_DIG:
                self._clear_slot(d)
        elif d.state == DS.TO_DIG:
            self._move(d,[*trip.dig,trip.altitude],DS.LOWER_GRABBER,dt)
        elif d.state == DS.LOWER_GRABBER:
            d.grabber_extension = min(1,d.elapsed/c.winch_seconds)
            if d.elapsed >= c.winch_seconds:
                d.transition(DS.SCOOP)
        elif d.state == DS.SCOOP:
            duration=c.scoop_duration(trip.mass)*trip.soil_factor
            request=self._scoop_request(d,d.elapsed-dt,dt)
            load=None
            if self.terrain.soil is not None:
                volume=max(0.,c.equipment.bucket.volume_m3-(d.soil.volume if d.soil is not None else 0.))
                load=self.terrain.scoop_load(*trip.dig,request,trip.source,c.grab_radius,max_volume=volume)
                if d.soil is None:d.soil=SoilLoad.empty(self.config.soil_profile)
                d.soil.add(load);taken=load.mass
            else:taken=self.terrain.scoop(*trip.dig,request,trip.source,c.grab_radius)
            d.cargo += taken
            if trip.source == "cut":
                d.excavated_mass += taken
                self.excavated_mass += taken
                if load is not None:self.excavated_volume+=load.volume
            else:
                d.reclaimed_mass += taken
            if taken>0:self._terrain_active = True
            if d.elapsed >= duration:
                self.drone_event(d,f"{'Взято из отвала' if trip.source == 'stockpile' else 'Завершена выемка'}: {d.cargo:.1f} кг в грейфере за {d.elapsed:.0f} с")
                d.transition(DS.LIFT_GRABBER)
        elif d.state == DS.LIFT_GRABBER:
            d.grabber_extension = max(0,1-d.elapsed/c.winch_seconds)
            if d.elapsed >= c.winch_seconds:
                d.transition(DS.TO_DUMP if d.cargo > 1e-9 else DS.RETURN)
                if d.state == DS.RETURN:
                    self._reserve_return(d)
        elif d.state == DS.TO_DUMP:
            self._move(d,[*trip.dump,trip.altitude],DS.DUMP,dt)
            if d.state == DS.DUMP:
                self.drone_event(d,"завис над отвалом: опускает грейфер к поверхности")
        elif d.state == DS.DUMP:
            if not d.dump_released:
                if (trip.destination=='dump' and self.terrain.front is not None and
                        self.terrain.local_dump_space(*trip.dump)<cargo_volume(d.cargo,d.soil,self.config.terrain.density)-1e-12):
                    point=self.terrain.dump_location(cargo_volume(d.cargo,d.soil,self.config.terrain.density),
                        self._reserved_dump(d),self.config.fleet.separation,near=d.position)
                    if point is None:self.block('Нет локального места для груза в боковых отвалах');return
                    trip.dump=point;d.transition(DS.TO_DUMP);return
                d.grabber_extension = min(1,d.elapsed/c.winch_seconds)
                if d.elapsed >= c.winch_seconds:
                    try:
                        mass = self.terrain.deposit_fill(*trip.dump,d.cargo,c.grab_radius,load=d.soil) if trip.destination == "fill" else d.cargo
                        if trip.destination == "dump":
                            self.terrain.deposit(*trip.dump,d.cargo,c.grab_radius,load=d.soil)
                    except ValueError as error:
                        self.block(str(error)); return
                    d.cargo = max(0, d.cargo-mass)
                    delivered=d.soil.remove(mass) if d.soil is not None else None
                    stamp=self.time+self.config.dt
                    day=int((self.config.start_hour*3600+stamp)//86400)
                    d.productivity.record(stamp,day,mass,self.config.terrain.density,trip.destination,volume=delivered.volume if delivered is not None else None)
                    if trip.destination == "fill":
                        d.filled_mass += mass
                        self.filled_mass += mass
                        if delivered is not None:self.filled_volume+=delivered.volume
                    d.dump_released = True
                    d.trips += 1
                    self.trips += 1
                    self._terrain_active = True
                    self.drone_event(d,f"{'Засыпка по плану' if trip.destination == 'fill' else 'Выгрузка в отвал'}: {mass:.1f} кг у поверхности; рейс {d.trips}")
            else:
                d.grabber_extension = max(0,(c.dump_seconds-d.elapsed)/(c.dump_seconds-c.winch_seconds))
                if d.elapsed >= c.dump_seconds:
                    if d.cargo > 1e-8:
                        point = self.terrain.fill_location(self._reserved_dump(d),self.config.fleet.separation)
                        kind = "fill"
                        if point is None:
                            point = self.terrain.dump_location(cargo_volume(d.cargo,d.soil,self.config.terrain.density),self._reserved_dump(d),self.config.fleet.separation,near=d.position if self.terrain.front is not None else None)
                            kind = "dump"
                        if point is None:
                            self.block("Нет места для оставшегося груза после изменения объёма засыпки");return
                        trip.dump, trip.destination = point, kind
                        d.transition(DS.TO_DUMP)
                    else:
                        self._continue_or_return(d)
        elif d.state == DS.RETURN:
            if getattr(d,'regional_launch',False):
                self._move(d,[*d.position[:2],self.cruise(d)],DS.RETURN,dt)
                if d.position[2]>=self.cruise(d)-1e-8:d.regional_launch=False
            else:self._move(d,[*d.destination[:2],self.cruise(d)],DS.LANDING,dt)
        elif d.state == DS.LANDING:
            air_wait = d.slot is None and d.charge_requested and not self.landing_requested and self.relocation_pending is None and self.config.fleet.wait_mode == "air" and self.terrain.remaining > self.config.terrain.completion_volume and (self.logistics is None or d.identifier not in self.logistics.recall_ids)
            goal = [*d.destination[:2],self.cruise(d)] if air_wait else d.destination
            self._move(d,goal,DS.WAIT_HOVER if air_wait else DS.DOCKED if d.slot is not None else DS.WAIT_GROUND,dt)
            if d.grounded:
                self._landed(d)
            elif d.state == DS.WAIT_HOVER:
                d.trip = None
                self._enqueue(d)
        elif d.state == DS.WAIT_HOVER:
            d.waiting_seconds += dt
            landing_seconds=math.ceil((d.position[2]-self.wait_position(d.identifier)[2])/c.vertical_speed)
            future_transfer=max(self._transfer_energy_from_ground(d,i) for i in range(len(self.slots)))
            landing_budget = d.power(True)*(landing_seconds+self.config.fleet.emergency_landing_seconds)+future_transfer
            if self.landing_requested or self.relocation_pending is not None or self.terrain.remaining <= self.config.terrain.completion_volume or d.battery <= c.battery*c.reserve+landing_budget or self.logistics is not None and d.identifier in self.logistics.recall_ids:
                d.destination = self.wait_position(d.identifier)
                d.transition(DS.WAIT_LANDING)
                self.drone_event(d,"садится на площадку ожидания, чтобы сохранить резерв")
        elif d.state == DS.WAIT_LANDING:
            self._move(d,d.destination,DS.WAIT_GROUND,dt)
            if d.grounded:
                self._landed(d)
        elif d.state == DS.TRANSFER_TAKEOFF:
            self._move(d,[*d.position[:2],self.cruise(d)],DS.TO_SLOT,dt)
        elif d.state == DS.TO_SLOT:
            self._move(d,[*d.destination[:2],self.cruise(d)],DS.SLOT_LANDING,dt)
        elif d.state == DS.SLOT_LANDING:
            self._move(d,d.destination,DS.DOCKED,dt)
            if d.docked:
                self._landed(d)

    def _update_workfront(self):
        t=self.terrain;front=t.front
        if front is None or self.base.destination is not None or self.relocation_pending is not None:return
        front.update()
        if self.landing_requested or t.remaining<=self.config.terrain.completion_volume:return
        advance=front.position-self._base_front_anchor
        if advance<max(4.,front.settings['front_length_m']/2):return
        if self.time-self._relocation_attempt_time<30:return
        self._relocation_attempt_time=self.time
        points=[(0,station_offset(i,self.config.fleet.pad_spacing)) for i in range(max(len(self.slots),len(self.drones)))]
        points += [(-self.config.fleet.waiting_offset,p[1]) for p in points.copy()]
        sites=sorted(front.settings['base_sites'],key=lambda p:2*abs(p[0]-front.position)+.25*np.linalg.norm(np.asarray(p[1:])-self.base.position[:2]))
        self.base_move_wait='NO_SAFE_FORWARD_BASE_ROUTE'
        for along,x,y in sites:
            target=np.array([x,y])
            if along<self._base_front_anchor+2 or np.linalg.norm(target-self.base.position[:2])<3:continue
            if all(self.base.safe_path(target,t,offset) for offset in points):
                self.relocation_pending=target
                self._pending_front_anchor=front.position
                self.base_move_wait='LANDING_FLEET_FOR_BASE_MOVE'
                self.event('Фронт продвинулся: завершаем рейсы и перемещаем базу вдоль канала')
                return

    def _start_front_move(self):
        if self.relocation_pending is None or self.landing_requested or not all(d.grounded for d in self.drones):return False
        if any(d.trip is not None or d.cargo>1e-8 for d in self.drones):return False
        point=self.relocation_pending
        cost=self.base.route_energy(point,self.terrain,self.drones)
        if self.base.battery<cost*1.3:
            self.base_move_wait='WAITING_BASE_ENERGY';return False
        self.base.destination=point.copy();self.relocation_pending=None
        for d in self.drones:
            if d.state==DS.CHARGING:d.state=DS.DOCKED
        self._base_front_anchor=self._pending_front_anchor
        self.base.last_relocation=self.time;self.base.last_relocation_trip=self.trips;self.base.relocations+=1
        self.base_move_wait='MOVING_WITH_WORKFRONT'
        return True

    def _maybe_relocate(self):
        if self.terrain.front is not None:
            self._start_front_move();return
        if self.landing_requested or self.config.plan is not None or not all(d.docked for d in self.drones) or any(d.trip for d in self.drones):
            return
        b,q,f = self.config.base,self.config.trench,self.config.fleet
        if self.trips-self.base.last_relocation_trip < b.relocation_trips or self.time-self.base.last_relocation < b.relocation_cooldown:
            return
        if self.time-self._relocation_attempt_time < b.relocation_cooldown:
            return
        self._relocation_attempt_time=self.time
        point = np.array([q.x-q.length/2-q.depth/self.terrain.slope-b.relocation_clearance-b.radius,q.y])
        if np.linalg.norm(point-self.base.position[:2]) < self.terrain.cell:
            return
        cost = np.linalg.norm(point-self.base.position[:2])/b.speed*self.base.motion_power(self.drones)
        if self.base.battery < cost:
            return
        for slot in range(len(self.slots)):
            offset = (0,station_offset(slot,f.pad_spacing))
            if not self.base.safe_path(point,self.terrain,offset):
                return
        trip = self.planner.choose(self.terrain,self.base,self.drone)
        if trip and self.base.maybe_relocate(self.time,self.trips,trip.dig,trip.dump,self.terrain,self.drones):
            self.event("База перемещается со всеми пристыкованными дронами")

    def _transfer_energy_from_ground(self,d,slot):
        point=self.wait_position(d.identifier)
        altitude=self.cruise(d)
        target=self.slot_position(slot)
        seconds=math.ceil((altitude-point[2])/d.config.vertical_speed)+math.ceil((altitude-target[2])/d.config.vertical_speed)+math.ceil(np.linalg.norm(target[:2]-point[:2])/d.config.empty_speed)+3
        return d.power(True)*seconds

    def _finish_or_block(self):
        if self.pending_equipment is not None:return
        if not all(d.grounded for d in self.drones):
            return
        if self.terrain.remaining <= self.config.terrain.completion_volume and not self._terrain_active and all(d.cargo == 0 for d in self.drones):
            self.state = MissionState.COMPLETED
            self.event("Работа завершена: все дроны приземлились, достигнут допуск выемки")
            return
        if self.landing_requested:
            return
        if self.terrain.remaining > self.config.terrain.completion_volume:
            key = (self.terrain.revision,tuple(self.base.position))
            if getattr(self,"_capability_key",None) != key:
                self._capable_ids = {d.identifier for d in self.drones if self._choose_trip(d,battery=d.config.battery,unreserved=True) is not None}
                self._capability_key = key
            ready = [d for d in self.drones if d.identifier in self._capable_ids and not d.charge_requested]
            if not self._capable_ids:
                if self.terrain.remaining_fill > self.config.terrain.completion_volume and self.terrain.stockpile_volume+self.terrain.remaining_cut < 1e-9:
                    self.block("Недостаточно грунта для засыпки: запас в отвале и запланированная выемка исчерпаны")
                else:
                    self.block("Нет доступного рейса ни для одного дрона: отвал заполнен или ёмкости батарей недостаточно")
            elif self.config.base.charge_power == 0 and not ready:
                self.block("Зарядное устройство отключено; полная зарядка перед продолжением невозможна")
            elif self.base.battery <= 1e-8 and self.config.base.panel_area*self.config.base.panel_efficiency*self.config.base.peak_irradiance == 0 and not ready:
                self.block("Полная зарядка невозможна: база разряжена, солнечная генерация или зарядное устройство отключены")
            elif all(d.state == DS.WAIT_GROUND for d in self.drones) and all(not any(d.battery >= d.config.battery*d.config.reserve+self._transfer_energy(d,i) for i in range(len(self.slots))) for d in self.drones):
                self.block("Дронам на площадках не хватает энергии для перелёта к зарядке с резервом; требуется внешняя доставка")
            elif self.queue and all(i is not None and self.drones[i].battery >= self.drones[i].config.battery-1e-7 and i not in self._capable_ids for i in self.slots) and not ready:
                self.block("Зарядные слоты заняты полностью заряженными дронами, которым недостаточно ёмкости для рейса; добавьте слот или измените состав")
            elif self.base.battery <= 1e-8 and self.config.base.idle_power >= self.config.base.panel_area*self.config.base.panel_efficiency*self.config.base.peak_irradiance:
                self.block("Собственное потребление базы поглощает всю солнечную мощность")

    def step(self,dt=None):
        dt = self.config.dt if dt is None else dt
        if dt != self.config.dt:
            raise ValueError("Передавайте фиксированный шаг 1 с; ускорение = число шагов")
        if self.state != MissionState.RUNNING:
            return
        self._update_workfront()
        if self.config.georeference is not None:
            from .channel import latitude_at
            key=tuple(self.base.position[:2])
            if getattr(self,'_latitude_position',None)!=key:
                self._latitude_value=latitude_at(self.config.georeference,*key,self.base.config.latitude_deg)
                self._latitude_position=key
            self.base.config.latitude_deg=self._latitude_value
        if self.logistics is not None:
            self.logistics.step()
            self.record_activities(dt)
            return
        self._starts = {d.identifier:d.position.copy() for d in self.drones}
        moving = self.base.destination is not None
        for d in self.drones:
            if d.docked and self.terrain.remaining > self.config.terrain.completion_volume and self._charging_needed(d):
                d.state = DS.CHARGING
                d.charging_seconds += dt
        flags = [d.state == DS.CHARGING and d.slot is not None for d in self.drones]
        if not moving:
            for d in self.drones:
                if d.state in (DS.WAIT_GROUND,DS.WAIT_HOVER) and d.charge_requested:self._enqueue(d)
            self._assign_slots()  # predict optical demand in the actual flight state
        self.laser_links=prepare_links(self.base,self.drones,self.terrain,dt,self.flight_power,self.config.fleet.separation/4,dust=self.dust)
        for link in self.laser_links:link.drone.laser_status=link.status
        message = self.base.tick(dt,(self.hour+dt/7200)%24,self.drones,flags,self.day_at(self.time+dt/2),self.laser_links,terrain=self.terrain)
        if message:
            self.event(message)
        if not moving:
            self._maybe_relocate()
        for d in self.drones:
            if self.state != MissionState.RUNNING:
                break
            if d.docked:
                old = d.position.copy()
                d.position[:] = self.slot_position(d.slot)
                d.velocity = (d.position-old)/dt
                if not moving:
                    self._docked(d)
            elif d.state == DS.WAIT_GROUND:
                d.velocity[:] = 0
                d.waiting_seconds += dt
                self._ready_ground(d)
            else:
                self._airborne(d,dt)
                if d.grounded:d.last_flight_seconds=d.flight_seconds;d.flight_seconds=0.
        for d in self.drones:self._settle_laser(d)
        if self._terrain_active:
            moved = self.terrain.relax()
            if moved > 1e-8:
                self.event("Началось осыпание песка","avalanche")
            if moved <= self.config.terrain.avalanche_stop_volume:
                self._terrain_active = False
                self._event_keys.pop("avalanche",None)
        if self.state == MissionState.RUNNING:
            active = [d for d in self.drones if d.state in (DS.LOWER_GRABBER,DS.SCOOP,DS.LIFT_GRABBER,DS.DUMP)]
            self.dust.tick(dt,self.drone.position,0,1,False)
            for d in active:
                self.dust.emit(dt,d.position,d.position[2]-self.terrain.sample(*d.position[:2]),d.power()/d.config.hover_power)
            self._finish_or_block()
        self.time += dt
        self.record_activities(dt)
        if self.pending_equipment is not None and self.base.destination is None and all(d.grounded and d.cargo==0 for d in self.drones):
            try:self._install_equipment(self.pending_equipment)
            except ValueError as error:
                self.cancel_equipment();self.event(f"Оборудование не установлено: {error}")

    def run(self,steps):
        for _ in range(steps):
            if self.state != MissionState.RUNNING:
                break
            self.step()
        return self.report()

    def record_activities(self, dt):
        from .activity import drone_activity,excavator_activity
        start=self.config.start_hour*3600+self.time-dt
        for drone in self.drones:
            drone.activity.record(drone_activity(drone,self.logistics),start,dt)
        if self.logistics is not None and self.logistics.excavator is not None:
            robot=self.logistics.excavator
            robot.activity.record(excavator_activity(robot),start,dt)

    def report(self, *, full=True):
        t,b = self.terrain,self.base
        cargo = sum(d.cargo for d in self.drones)
        consumed = sum(d.consumed for d in self.drones+self.retired_drones)
        batteries = sum(d.battery for d in self.drones)
        ground_error = t.mass_change+cargo
        energy_error = batteries+b.battery+consumed+b.consumed+b.charging_losses+b.laser_losses+b.curtailed+self.equipment_exported_energy-self._initial_energy-b.generated-self.equipment_imported_energy
        remaining = t.remaining
        sun = b.sun(self.hour,self.day_of_year)
        drone_rows = []
        for d in self.drones:
            drone_rows.append({"id":d.identifier+1,"name":d.config.name,"state":d.state.value,"position":d.position.tolist(),"battery_kwh":d.battery/KWH,"battery_percent":100*d.battery/d.config.battery,"capacity_kwh":d.config.battery/KWH,"cargo_kg":d.cargo,"payload_kg":d.config.payload,"empty_mass_kg":d.config.empty_mass,"trips":d.trips,"excavated_mass_kg":d.excavated_mass,"consumed_kwh":d.consumed/KWH,"waiting_seconds":d.waiting_seconds,"charging_seconds":d.charging_seconds,"traffic_yields":d.traffic_yields,"slot":d.slot+1 if d.slot is not None else None,"minimum_kwh":d.minimum_charge/KWH,"cable_length_m":d.cable_length(t.sample(*d.position[:2]))})
            drone_rows[-1].update(charge_threshold_percent=100*d.config.charge_threshold,charge_requested=d.charge_requested,charge_cycles=d.charge_cycles,filled_mass_kg=d.filled_mass,reclaimed_mass_kg=d.reclaimed_mass)
            drone_rows[-1].update(serial=d.serial,created_at_seconds=d.created_at,equipment=d.config.equipment.breakdown(),**d.productivity.report(self.time,int((self.config.start_hour*3600+self.time)//86400)))
            drone_rows[-1]['activity']=d.activity.report()
            drone_rows[-1].update(dig_rate_kg_s=d.config.dig_rate_kg_s,full_scoop_seconds=d.config.scoop_duration(d.config.payload),digging_j_m3=d.config.digging_j_m3)
            drone_rows[-1].update(laser_received_kwh=d.laser_received/KWH,laser_power_w=d.laser_power_w,laser_status=d.laser_status,flight_power_w=d.last_flight_power_w,airborne_seconds=d.airborne_seconds,current_flight_seconds=d.flight_seconds,last_flight_seconds=d.last_flight_seconds,longest_flight_seconds=d.longest_flight_seconds)
        extra = {"remaining_cut_m3":t.remaining_cut,"remaining_fill_m3":t.remaining_fill,"stockpile_m3":t.stockpile_volume,"filled_mass_kg":self.filled_mass,"filled_m3":self.filled_mass/self.config.terrain.density,"plan_segments":len(self.config.plan.segments) if self.config.plan is not None else 1,"landing_requested":self.landing_requested}
        extra.update(map_side_m=t.cell*t.height.shape[0],grid_cells=t.height.shape[0],cell_size_m=t.cell)
        extra['georeference']=self.config.georeference
        extra.update(base_capacity_kwh=b.config.battery/KWH,
                     base_battery_count=sum(pack.quantity for pack in b.config.equipment.batteries),
                     base_specification=self.base_specification())
        extra.update(workfront=t.front.report() if t.front is not None else None,
                     base_position_m=b.position.tolist(),base_move_wait=self.base_move_wait,
                     base_move_pending=self.relocation_pending is not None,base_move_status=self.base_move_status)
        extra.update(base_equipment=b.config.equipment.breakdown(),equipment_imported_kwh=self.equipment_imported_energy/KWH,equipment_exported_kwh=self.equipment_exported_energy/KWH,equipment_pending=self.pending_equipment is not None,laser_transfer_enabled=True,retired_drones=[{"serial":d.serial,"name":d.config.name,"created_at_seconds":d.created_at,"retired_at_seconds":d.retired_at,"consumed_kwh":d.consumed/KWH,"laser_received_kwh":d.laser_received/KWH,"airborne_seconds":d.airborne_seconds,"longest_flight_seconds":d.longest_flight_seconds,**d.productivity.report(self.time,int((self.config.start_hour*3600+self.time)//86400))} for d in self.retired_drones])
        extra.update(lasers=[link.report() for link in self.laser_links],laser_input_power_w=b.laser_input_power,laser_received_power_w=sum(d.laser_power_w for d in self.drones),laser_input_kwh=b.laser_input/KWH,laser_received_kwh=sum(d.laser_received for d in self.drones+self.retired_drones)/KWH,laser_losses_kwh=b.laser_losses/KWH)
        extra['laser_receiver_heat_kwh']=sum(d.receiver_heat_j for d in self.drones+self.retired_drones)/KWH
        for row,drone in zip(extra['retired_drones'],self.retired_drones):row['activity']=drone.activity.report()
        result = {**extra,"state":self.state.value,"reason":self.reason,"time_seconds":self.time,"elapsed_display":elapsed_text(self.time),"solar_clock":self.clock_text,"date":self.date_text,"day_of_year":self.day_of_year,"latitude_deg":b.config.latitude_deg,"base_mass_kg":b.config.mass,"panel_area_m2":b.config.panel_area,"irradiance_w_m2":sun.irradiance,"sun_elevation_deg":sun.elevation_deg,"daylight_hours":sun.daylight_hours,"sunrise_hour":sun.sunrise_hour,"sunset_hour":sun.sunset_hour,"base_motion_power_w":b.motion_power(self.drones) if b.destination is not None else 0.0,"cable_length_m":drone_rows[0]["cable_length_m"],"hour":self.hour,"trips":self.trips,"relocations":b.relocations,"initial_excavation_m3":t.initial_excavation,"remaining_m3":remaining,"completion_percent":max(0,min(100,100*(1-remaining/max(t.initial_excavation,1e-12)))),"excavated_m3":self.excavated_mass/self.config.terrain.density,"excavated_mass_kg":self.excavated_mass,"cargo_kg":cargo,"drone_state":self.drone.state.value,"base_state":b.state.value,"drone_kwh":batteries/KWH,"base_kwh":b.battery/KWH,"solar_power_w":b.solar_power,"solar_generated_kwh":b.generated/KWH,"solar_curtailed_kwh":b.curtailed/KWH,"consumed_kwh":(consumed+b.consumed)/KWH,"charging_losses_kwh":b.charging_losses/KWH,"mass_error_kg":ground_error,"energy_error_j":energy_error,"minimum_drone_kwh":min(d.minimum_charge/KWH for d in self.drones),"initial_energy_kwh":self._initial_energy/KWH,"drone_count":len(self.drones),"charge_slots":len(self.slots),"slot_occupants":[v+1 if v is not None else None for v in self.slots],"queue":[i+1 for i in self.queue],"wait_mode":self.config.fleet.wait_mode,"drones":drone_rows}
        if self.logistics is not None:result=self.logistics.update_report(result,full=full)
        if t.soil is not None:
            result['excavated_m3']=self.excavated_volume;result['filled_m3']=self.filled_volume
            loads=[d.soil for d in self.drones if d.soil is not None]
            if self.logistics is not None:
                loads.extend(c.soil for c in self.logistics.containers if c.soil is not None)
                e=self.logistics.excavator
                if e is not None and e.soil is not None:loads.append(e.soil)
            error=(t.soil.volume_delta_by_material-t._initial_soil_volume)*t.soil.profile.densities
            for load in loads:error+=load.masses
            if self.shared_site is not None:error=self.shared_site.material_errors()
            result['soil']=dict(name=t.soil.profile.name,conditional=True,
                modified_tiles=len(t.soil.tiles),storage_bytes=sum(a.nbytes for a in t.soil.tiles.values()),
                materials=[dict(name=m.name,density_kg_m3=m.density,digging_factor=m.digging_factor,
                    mass_error_kg=float(v)) for m,v in zip(t.soil.profile.materials,error)],
                max_material_mass_error_kg=float(np.abs(error).max()))
        if self.shared_site is not None:result['mass_error_kg']=self.shared_site.mass_error()
        if full and getattr(self.config,'energy_design',None):
            from .winter_power import design_status
            result['energy_design']=self.config.energy_design
            state,message=design_status(self.config)
            result['energy_design_status']=dict(state=state,message=message)
        return result
