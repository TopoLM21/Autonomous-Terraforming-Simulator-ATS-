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
        return [o for o in self.items.values() if isinstance(o, cls)
                and (location is None or o.location == location)]

    def validate(self):
        slot_modules=set()
        for key, o in self.items.items():
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
            if isinstance(o, ChargingSlot):
                if (not isinstance(self.items.get(o.module_id),BatteryModule) or o.module_id in slot_modules
                        or o.location!=Location.BASE_MOUNT or o.holder_id!='base'
                        or not math.isfinite(o.rated_w) or o.rated_w<0
                        or not math.isfinite(o.input_power_w) or not 0<=o.input_power_w<=o.rated_w+1e-7):
                    raise ValueError('Invalid individual battery charging slot')
                slot_modules.add(o.module_id)

    def snapshot(self):
        return [dict(asdict(o), type=type(o).__name__, gross_mass_kg=o.gross_mass)
                for o in self.items.values()]
