"""Physical equipment inventories; masses and ratings use SI-compatible units."""
from dataclasses import dataclass,field,asdict
from copy import deepcopy
import math


def retained_energy(old_modules,new_modules,energy):
    """All packs share SoC; removed packs export their share, added packs are empty."""
    old,new={},{}
    for modules,counts in [(old_modules,old),(new_modules,new)]:
        for b in modules:counts[b.capacity_kwh]=counts.get(b.capacity_kwh,0)+b.quantity
    capacity=sum(k*q for k,q in old.items())
    retained=sum(k*min(q,new.get(k,0)) for k,q in old.items())
    return energy*retained/capacity if capacity else 0.0


def station_offset(index,spacing):
    return (0 if index==0 else (index+1)//2*(1 if index%2 else -1))*spacing

KWH=3_600_000.0
BASE_BATTERY_LIMIT=1000
PANEL_QUANTITY_LIMIT=10000


def positive(value,label,zero=False):
    if not isinstance(value,(int,float)) or not math.isfinite(value) or (value<0 if zero else value<=0):
        raise ValueError(f"{label}: требуется {'неотрицательное' if zero else 'положительное'} конечное число")


def set_panel_area(equipment, area, efficiency=.20):
    """Resize panel recipes, preserving their efficiency and mass per square metre."""
    positive(area,'Площадь панелей',True)
    previous=equipment.panel_area
    if area==0:
        equipment.panels=[]
    elif len(equipment.panels)==1 and abs(area/equipment.panels[0].area_m2-round(area/equipment.panels[0].area_m2))<1e-8 and 1<=round(area/equipment.panels[0].area_m2)<=PANEL_QUANTITY_LIMIT:
        # Whole standard panels change in count, retaining their physical size.
        equipment.panels[0].quantity=round(area/equipment.panels[0].area_m2)
    elif previous>0:
        factor=area/previous
        for panel in equipment.panels:
            panel.area_m2*=factor;panel.mass_kg*=factor
    else:
        count=round(area) if abs(area-round(area))<1e-8 and 1<=round(area)<=PANEL_QUANTITY_LIMIT else 1
        equipment.panels=[PanelModule(quantity=count,area_m2=area/count,mass_kg=area*4/count,efficiency=efficiency or .20)]


@dataclass
class BatteryModule:
    name: str="Аккумулятор"
    quantity: int=1
    capacity_kwh: float=5.0
    mass_kg: float=50.0

    def validate(self,maximum=64):
        if not isinstance(self.quantity,int) or isinstance(self.quantity,bool) or not 0<=self.quantity<=maximum:raise ValueError(f"Количество батарей: 0–{maximum}")
        positive(self.capacity_kwh,"Ёмкость батареи");positive(self.mass_kg,"Масса батареи")


@dataclass
class PanelModule:
    name: str="Солнечная панель"
    quantity: int=20
    area_m2: float=1.0
    mass_kg: float=4.0
    efficiency: float=.20

    def validate(self):
        if type(self.quantity) is not int or not 0<=self.quantity<=PANEL_QUANTITY_LIMIT:
            raise ValueError(f"Количество панелей: 0–{PANEL_QUANTITY_LIMIT}")
        positive(self.area_m2,"Площадь панели");positive(self.mass_kg,"Масса панели")
        if not 0<=self.efficiency<=1:raise ValueError("КПД панели: 0–100%")


@dataclass
class LaserEmitter:
    drone_number: int=1
    mass_kg: float=30.0
    input_power_kw: float=20.0
    efficiency: float=.5
    range_m: float=30.0
    enabled: bool=True
    beam_radius_m: float=.10
    divergence_mrad: float=2.0
    profile: str='existing_legacy'
    reference_capture: float=1.0
    attenuation_per_m: float=0.0
    dust_attenuation: float=0.0
    shared_source: str=''
    shared_limit_kw: float=0.0

    def validate(self):
        if not isinstance(self.drone_number,int) or not 1<=self.drone_number<=6:raise ValueError("Номер позиции дрона для передатчика: 1–6")
        positive(self.mass_kg,"Масса передатчика");positive(self.input_power_kw,"Мощность передатчика",True);positive(self.range_m,"Дальность передатчика")
        positive(self.beam_radius_m,"Радиус луча");positive(self.divergence_mrad,"Расходимость луча",True)
        if not isinstance(self.enabled,bool):raise ValueError("Состояние передатчика должно быть логическим")
        if not 0<self.efficiency<=1:raise ValueError("КПД передатчика: больше 0 и не больше 100%")
        if self.profile not in ('existing_legacy','realistic_reference') or not 0<=self.reference_capture<=1:
            raise ValueError('Неверный профиль/захват лазера')
        positive(self.attenuation_per_m,'Затухание',True);positive(self.dust_attenuation,'Пыль',True)
        positive(self.shared_limit_kw,'Общий предел источника',True)


@dataclass
class LaserReceiver:
    installed: bool=False
    mass_kg: float=5.0
    area_m2: float=.25
    max_power_kw: float=8.0
    efficiency: float=.45
    optical_cap_kw: float | None=None  # legacy limit is electrical, preserved
    orientation_cosine: float=1.0
    thermal_capacity_j_k: float=4000.
    cooling_w_k: float=20.
    max_temperature_c: float=70.
    ambient_temperature_c: float=25.

    def validate(self):
        positive(self.mass_kg,"Масса приёмника");positive(self.area_m2,"Площадь приёмника");positive(self.max_power_kw,"Мощность приёмника")
        if not isinstance(self.installed,bool) or not 0<self.efficiency<=1:raise ValueError("Неверные параметры лазерного приёмника")
        if self.optical_cap_kw is not None:positive(self.optical_cap_kw,'Оптический предел')
        if not 0<=self.orientation_cosine<=1:raise ValueError('Ориентация приёмника: 0..1')
        positive(self.thermal_capacity_j_k,'Теплоёмкость');positive(self.cooling_w_k,'Охлаждение',True)
        if self.max_temperature_c<=self.ambient_temperature_c:raise ValueError('Тепловой предел ниже среды')


@dataclass
class BucketModule:
    mass_kg: float=15.0
    volume_m3: float=.0375
    radius_m: float=.5

    def validate(self):
        positive(self.mass_kg,"Масса ковша");positive(self.volume_m3,"Объём ковша");positive(self.radius_m,"Радиус захвата")


@dataclass
class BaseEquipment:
    chassis_mass_kg: float=585.0  # tracks, frame, drive and control equipment
    batteries: list[BatteryModule]=field(default_factory=lambda:[BatteryModule(quantity=6)])
    panels: list[PanelModule]=field(default_factory=lambda:[PanelModule()])
    landing_slots: int=1
    slot_mass_kg: float=20.0
    charger_mass_kg: float=15.0
    charge_power_kw: float=8.0
    charge_efficiency: float=.9
    emitters: list[LaserEmitter]=field(default_factory=list)

    @property
    def capacity(self):return sum(b.quantity*b.capacity_kwh*KWH for b in self.batteries)
    @property
    def panel_area(self):return sum(p.quantity*p.area_m2 for p in self.panels)
    @property
    def panel_efficiency(self):
        return sum(p.quantity*p.area_m2*p.efficiency for p in self.panels)/self.panel_area if self.panel_area else 0.0
    @property
    def mass(self):
        return self.chassis_mass_kg+self.charger_mass_kg+self.landing_slots*self.slot_mass_kg+sum(b.quantity*b.mass_kg for b in self.batteries)+sum(p.quantity*p.mass_kg for p in self.panels)+sum(e.mass_kg for e in self.emitters)

    def breakdown(self):
        return {"chassis_kg":self.chassis_mass_kg,"batteries_kg":sum(b.quantity*b.mass_kg for b in self.batteries),"panels_kg":sum(p.quantity*p.mass_kg for p in self.panels),"landing_slots_kg":self.landing_slots*self.slot_mass_kg,"charger_kg":self.charger_mass_kg,"emitters_kg":sum(e.mass_kg for e in self.emitters),"total_kg":self.mass}

    def validate(self):
        positive(self.chassis_mass_kg,"Масса шасси");positive(self.slot_mass_kg,"Масса посадочного места");positive(self.charger_mass_kg,"Масса зарядного устройства")
        positive(self.charge_power_kw,"Общая мощность зарядки",True)
        if not isinstance(self.landing_slots,int) or not 1<=self.landing_slots<=6:raise ValueError("Посадочных мест: 1–6")
        if not 0<self.charge_efficiency<=1:raise ValueError("КПД зарядки: 0–100%")
        if len(self.batteries)>12 or len(self.panels)>12 or len(self.emitters)>6:raise ValueError("Слишком много типов оборудования базы")
        for module in self.batteries:module.validate(BASE_BATTERY_LIMIT)
        if sum(b.quantity for b in self.batteries)>BASE_BATTERY_LIMIT:raise ValueError(f'На базе допускается до {BASE_BATTERY_LIMIT} аккумуляторов')
        for module in [*self.panels,*self.emitters]:module.validate()
        if len({e.drone_number for e in self.emitters})!=len(self.emitters):raise ValueError("Для позиции дрона допускается один лазерный передатчик")

    @classmethod
    def from_legacy(cls,b,slots=1):
        battery_count=max(1,math.ceil(b.battery/KWH/5)) if b.battery else 0
        batteries=[BatteryModule(quantity=battery_count,capacity_kwh=b.battery/KWH/battery_count,mass_kg=b.battery/KWH*10/battery_count)] if battery_count else []
        panel_count=min(PANEL_QUANTITY_LIMIT,max(1,math.ceil(b.panel_area))) if b.panel_area else 0
        panels=[PanelModule(quantity=panel_count,area_m2=b.panel_area/panel_count,mass_kg=b.panel_area*4/panel_count,efficiency=b.panel_efficiency)] if panel_count else []
        chassis=b.mass-sum(v.quantity*v.mass_kg for v in batteries+panels)-20-15
        return cls(chassis,batteries,panels,slots,20,15,b.charge_power/1000,b.charge_efficiency)

    def resolve(self,b,fleet):
        self.validate();b.mass=self.mass;b.battery=self.capacity;b.panel_area=self.panel_area;b.panel_efficiency=self.panel_efficiency
        b.charge_power=self.charge_power_kw*1000;b.charge_efficiency=self.charge_efficiency;fleet.charge_slots=self.landing_slots


@dataclass
class DroneEquipment:
    frame_mass_kg: float=60.0
    payload_limit_kg: float=120.0
    reference_mass_kg: float=105.0
    reference_hover_power_kw: float=18.0
    batteries: list[BatteryModule]=field(default_factory=lambda:[BatteryModule(quantity=2,capacity_kwh=2.5,mass_kg=15)])
    bucket: BucketModule=field(default_factory=BucketModule)
    receiver: LaserReceiver=field(default_factory=LaserReceiver)

    @property
    def capacity(self):return sum(b.quantity*b.capacity_kwh*KWH for b in self.batteries)
    @property
    def mass(self):return self.frame_mass_kg+sum(b.quantity*b.mass_kg for b in self.batteries)+self.bucket.mass_kg+(self.receiver.mass_kg if self.receiver.installed else 0)
    @property
    def hover_power(self):return self.reference_hover_power_kw*1000*(self.mass/self.reference_mass_kg)**1.5

    def breakdown(self):
        return {"frame_kg":self.frame_mass_kg,"batteries_kg":sum(b.quantity*b.mass_kg for b in self.batteries),"bucket_kg":self.bucket.mass_kg,"receiver_kg":self.receiver.mass_kg if self.receiver.installed else 0,"total_kg":self.mass}

    def validate(self):
        for value,label in [(self.frame_mass_kg,"Масса рамы"),(self.payload_limit_kg,"Грузоподъёмность рамы"),(self.reference_mass_kg,"Опорная масса"),(self.reference_hover_power_kw,"Опорная мощность")]:positive(value,label)
        if len(self.batteries)>12:raise ValueError("У дрона слишком много типов батарей")
        for b in self.batteries:b.validate()
        if self.capacity<=0:raise ValueError("Дрону требуется хотя бы одна батарея")
        self.bucket.validate();self.receiver.validate()

    @classmethod
    def from_legacy(cls,d,density=1600,grab_radius=.5):
        battery_share,bucket_share=(.24,.16) if d.payload<=30 else (2/7,1/7)
        return cls(d.empty_mass*(1-battery_share-bucket_share),d.payload*2,d.empty_mass,d.hover_power/1000,[BatteryModule(quantity=2,capacity_kwh=d.battery/KWH/2,mass_kg=d.empty_mass*battery_share/2)],BucketModule(d.empty_mass*bucket_share,d.payload/density,grab_radius))

    def resolve(self,d,density):
        self.validate();d.empty_mass=self.mass;d.battery=self.capacity;d.payload=min(self.payload_limit_kg,self.bucket.volume_m3*density);d.hover_power=self.hover_power;d.grab_radius=self.bucket.radius_m


def resolve_config(config):
    if config.base.equipment is None:config.base.equipment=BaseEquipment.from_legacy(config.base,config.fleet.charge_slots)
    config.base.equipment.resolve(config.base,config.fleet)
    models=config.drone_models()
    for d in [config.drone,*models]:
        if d.equipment is None:d.equipment=DroneEquipment.from_legacy(d,config.terrain.density,config.terrain.grab_radius)
        d.equipment.resolve(d,config.terrain.density)
        if config.soil_profile is not None:d.payload=d.equipment.payload_limit_kg
    config.fleet.models=models
    return config


def physical_payload_limit(drone):
    e=drone.config.equipment
    batteries=sum(b.quantity*b.mass_kg for b in e.batteries)
    reference_batteries=max(0.,e.reference_mass_kg-e.frame_mass_kg-e.bucket.mass_kg)
    receiver=e.receiver.mass_kg if e.receiver.installed else 0.
    structural=max(0.,e.payload_limit_kg-max(0.,batteries-reference_batteries)-receiver)
    output=getattr(drone,'max_battery_output_w',math.inf)
    power_limit=drone.config.empty_mass*((output/(drone.config.hover_power*drone.config.flight_factor))**(2/3)-1)
    return max(0.,min(structural,power_limit))


def set_base_battery_count(equipment, count):
    """Resize installed storage; additions use its first type, removals the last."""
    if type(count) is not int or not 0<=count<=BASE_BATTERY_LIMIT:
        raise ValueError(f'Аккумуляторов базы: 0–{BASE_BATTERY_LIMIT}')
    delta=count-sum(pack.quantity for pack in equipment.batteries)
    if delta>0:
        if not equipment.batteries:equipment.batteries=[BatteryModule(quantity=0)]
        equipment.batteries[0].quantity+=delta
    elif delta<0:
        for pack in reversed(equipment.batteries):
            take=min(pack.quantity,-delta);pack.quantity-=take;delta+=take
    equipment.validate()


def equipment_data(config):
    return {"version":1,"base":asdict(config.base.equipment),"models":[asdict(d) for d in config.drone_models()],"wait_mode":config.fleet.wait_mode}


def load_equipment_data(data,config):
    from .config import DroneConfig
    if data.get("version")!=1:raise ValueError("Неизвестная версия комплектации")
    clone=deepcopy(config);b=dict(data["base"])
    b["batteries"]=[BatteryModule(**v) for v in b["batteries"]];b["panels"]=[PanelModule(**v) for v in b["panels"]];b["emitters"]=[LaserEmitter(**v) for v in b["emitters"]]
    clone.base.equipment=BaseEquipment(**b);models=[]
    for raw in data["models"]:
        values=dict(raw);e=dict(values["equipment"]);e["batteries"]=[BatteryModule(**v) for v in e["batteries"]];e["bucket"]=BucketModule(**e["bucket"]);e["receiver"]=LaserReceiver(**e["receiver"]);values["equipment"]=DroneEquipment(**e);models.append(DroneConfig(**values))
    clone.fleet.models=models;clone.fleet.count=len(models);clone.fleet.wait_mode=data["wait_mode"]
    resolve_config(clone);clone.validate();return clone
