"""Physical instances, not equipment recipes. All energy is J and mass is kg."""
from dataclasses import dataclass, field, asdict
from enum import Enum
import math


class Location(str, Enum):
    BASE_MOUNT = 'BASE_MOUNT'
    BASE_STORAGE = 'BASE_STORAGE'
    BASE_CHARGER = 'BASE_CHARGER'
    DRONE_MOUNT = 'DRONE_MOUNT'
    DRONE_CARGO = 'DRONE_CARGO'
    EXCAVATOR_SLOT = 'EXCAVATOR_SLOT'
    WORKSITE_RACK = 'WORKSITE_RACK'
    FILLED_QUEUE = 'FILLED_QUEUE'
    DUMP_SITE = 'DUMP_SITE'
    GROUND = 'GROUND'
    IN_TRANSFER = 'IN_TRANSFER'


@dataclass
class Item:
    id: str
    mass_kg: float
    location: Location = Location.BASE_STORAGE
    state: str = 'STORED_READY'
    holder_id: str = 'base'
    reserved_by: str | None = None
    position: tuple = (0., 0., 0.)
    kind: str = 'Equipment'

    @property
    def gross_mass(self):
        return self.mass_kg


@dataclass
class BatteryModule(Item):
    compatibility: str = 'excavator'
    capacity_j: float = 5_400_000.
    energy_j: float = 5_400_000.
    max_charge_w: float = 1500.
    max_discharge_w: float = 6000.
    charged_energy: float = 0.
    swaps: int = 0
    soh: float = 1.

    @property
    def soc(self):
        return self.energy_j / self.capacity_j


@dataclass
class SoilContainer(Item):
    volume_m3: float = .055
    max_gross_kg: float = 90.
    soil_kg: float = 0.
    cycles: int = 0
    wear: float = 0.
    assigned_limit_kg: float = 0.
    soil: object = None

    def soil_volume(self,density):
        return self.soil.volume if self.soil is not None else self.soil_kg/density

    def full(self,density):
        return self.soil_kg>=self.assigned_limit_kg-1e-8 or self.soil_volume(density)>=self.volume_m3-1e-10

    @property
    def gross_mass(self):
        return self.mass_kg + self.soil_kg

    def capacity_for(self, payload_kg, rigging_kg, density):
        return max(0., min(payload_kg-self.mass_kg-rigging_kg,
                          self.max_gross_kg-self.mass_kg, self.volume_m3*density))


@dataclass
class Rigging(Item):
    kind: str = 'CONTAINER_HOOK'


@dataclass
class ChargingSlot(Item):
    kind: str = 'MODULE_CHARGING_SLOT'
    module_id: str = ''
    rated_w: float = 1500.
    input_power_w: float = 0.


@dataclass
class Inventory:
    items: dict = field(default_factory=dict)

    def add(self, item):
        self.add_many([item])
        return item

    def add_many(self, items):
        """One atomic delivery, validating once even for a large module shipment."""
        added={}
        for item in items:
            if item.id in self.items or item.id in added:
                raise ValueError(f'Duplicate inventory ID: {item.id}')
            added[item.id]=item
        self.items.update(added)
        try:self.validate()
        except Exception:
            for id in added:del self.items[id]
            raise

    def reserve(self, ids, owner):
        objects = [self.items[i] for i in ids]
        if any(o.reserved_by not in (None, owner) for o in objects):
            return False
        for o in objects:
            o.reserved_by = owner
        return True

    def release(self, owner):
        for o in self.items.values():
            if o.reserved_by == owner:
                o.reserved_by = None

    def move(self, id, location, holder, position, state, owner=None):
        item = self.items[id]
        if item.reserved_by is not None and item.reserved_by != owner:
            raise ValueError(f'{id} reserved by {item.reserved_by}')
        item.location, item.holder_id = Location(location), holder
        item.position, item.state = tuple(map(float, position)), state

    def select(self, cls, location=None):
        signature=(id(self.items),len(self.items))
        if getattr(self,'_index_signature',None)!=signature:
            self._index_signature=signature;self._type_index={}
        key=cls.__module__+'.'+cls.__name__
        if key not in self._type_index:self._type_index[key]=[o for o in self.items.values() if isinstance(o,cls)]
        return [o for o in self._type_index[key] if location is None or o.location == location]

    def mutable_items(self):
        signature=(id(self.items),len(self.items))
        if getattr(self,'_mutable_signature',None)!=signature:
            self._mutable_signature=signature
            static={'SOLAR_PANEL','LANDING_PAD','CHASSIS','PLATFORM_CHASSIS','CHARGER_AND_AUTOMATION','LASER_TRANSMITTER'}
            self._mutable_items=[o for o in self.items.values() if o.kind not in static]
        return self._mutable_items

    def validate(self,*,dynamic=False):
        slot_modules=set()
        entries=((o.id,o) for o in self.mutable_items()) if dynamic else self.items.items()
        for key, o in entries:
            if key != o.id or not o.holder_id or not isinstance(o.location, Location):
                raise ValueError('Invalid inventory identity/location')
            if not math.isfinite(o.mass_kg) or o.mass_kg <= 0:
                raise ValueError('Invalid inventory mass')
            if isinstance(o, BatteryModule):
                if o.capacity_j <= 0 or not -1e-7 <= o.energy_j <= o.capacity_j+1e-7:
                    raise ValueError('Battery SOC outside physical capacity')
                if o.location == Location.BASE_CHARGER and o.reserved_by:
                    raise ValueError('Battery cannot charge while reserved for transport')
            if isinstance(o, SoilContainer):
                if o.soil_kg < -1e-8 or o.gross_mass > o.max_gross_kg+1e-7:
                    raise ValueError('Container overloaded')
                if o.soil is not None:
                    o.soil.validate(o.soil_kg)
                    if o.soil.volume>o.volume_m3+1e-9:raise ValueError('Container volume exceeded')
            if isinstance(o, ChargingSlot):
                if (not isinstance(self.items.get(o.module_id),BatteryModule) or o.module_id in slot_modules
                        or o.location!=Location.BASE_MOUNT or o.holder_id!='base'
                        or not math.isfinite(o.rated_w) or o.rated_w<0
                        or not math.isfinite(o.input_power_w) or not 0<=o.input_power_w<=o.rated_w+1e-7):
                    raise ValueError('Invalid individual battery charging slot')
                slot_modules.add(o.module_id)

    def snapshot(self):
        rows=[]
        for o in self.items.values():
            row=dict(asdict(o),type=type(o).__name__,gross_mass_kg=o.gross_mass)
            if isinstance(o,SoilContainer):
                row.pop('soil',None)
                if o.soil is not None:
                    row.update(soil_volume_m3=o.soil.volume,soil_materials_kg=o.soil.masses.tolist())
            rows.append(row)
        return rows

    def grouped(self, query=''):
        """Read-only overview; all instances belong to exactly one group."""
        query=query.casefold().strip();groups={}
        names={'GRABBER':'Грейферы / ковши','CONTAINER_HOOK':'Контейнерные подвесы','HOIST':'Лебёдки',
               'HEAVY_LIFT_RIG':'Стропа коллективного подъёма','MODULE_CHARGING_SLOT':'Зарядные слоты модулей',
               'SOLAR_PANEL':'Солнечные панели','LANDING_PAD':'Посадочные места','CHASSIS':'Шасси базы',
               'CHARGER_AND_AUTOMATION':'Зарядное оборудование','LASER_TRANSMITTER':'Передатчики лазера',
               'LASER_RECEIVER':'Приёмники лазера'}
        for item in self.items.values():
            scope=('На складе / зарядке' if item.location in (Location.BASE_STORAGE,Location.BASE_CHARGER) else
                   'Установлены' if item.location in (Location.BASE_MOUNT,Location.DRONE_MOUNT,Location.EXCAVATOR_SLOT) else
                   'На участке / в пути')
            status='';unit=''
            if isinstance(item,BatteryModule):
                name='Аккумуляторы базы' if item.compatibility=='base' else 'Батареи Крота' if item.compatibility=='excavator' else 'Батареи дронов'
                status='Полные' if item.soc>=1-1e-6 else 'Разряженные (≤10%)' if item.soc<=.1 else 'Частично заряженные'
                unit=f'{item.capacity_j/3600000:g} кВт·ч / {item.mass_kg:g} кг'
                kind=(item.compatibility,item.capacity_j,item.mass_kg)
            elif isinstance(item,SoilContainer):
                name='Ёмкости';kind='container'
                status='Пустые' if item.soil_kg<=1e-8 else 'Заполненные' if item.location==Location.FILLED_QUEUE or item.state=='FULL_IN_TRANSIT' else 'С грунтом'
            else:name=names.get(item.kind,item.kind);kind=item.kind
            label=' · '.join(part for part in (name,status,scope,unit) if part)
            searchable=f'{label} {item.id} {type(item).__name__} {item.location.value} {item.state} {item.holder_id} {item.reserved_by or ""}'
            if query and query not in searchable.casefold():continue
            key=(kind,status,scope)
            row=groups.setdefault(key,dict(key=key,label=label,count=0,available=0,reserved=0,mass_kg=0.,soil_kg=0.,energy_j=0.,member_ids=[]))
            row['count']+=1;row['reserved']+=bool(item.reserved_by);row['mass_kg']+=item.gross_mass
            row['available']+=bool(item.location==Location.BASE_STORAGE and not item.reserved_by
                and (not isinstance(item,BatteryModule) or item.soc>=1-1e-6))
            row['soil_kg']+=getattr(item,'soil_kg',0.);row['energy_j']+=getattr(item,'energy_j',0.)
            row['member_ids'].append(item.id)
        return sorted(groups.values(),key=lambda row:row['label'])
