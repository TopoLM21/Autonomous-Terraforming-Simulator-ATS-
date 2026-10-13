"""Solar input, shared contact/optical charging and tracked relocation."""
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
from .solar import sunlight
from .laser import transfer


class BaseState(str, Enum):
    PARKED_GENERATING = "PARKED_GENERATING"
    CHARGING = "CHARGING"
    MOVING = "MOVING"
    WAITING_FOR_SUN = "WAITING_FOR_SUN"


@dataclass
class SolarBase:
    config: object
    position: np.ndarray
    battery: float = field(init=False)
    state: BaseState = BaseState.PARKED_GENERATING
    destination: object = None
    generated: float = 0.0
    curtailed: float = 0.0
    consumed: float = 0.0
    charging_losses: float = 0.0
    charging_input: float = 0.0
    solar_power: float = 0.0
    laser_input: float = 0.0
    laser_losses: float = 0.0
    laser_input_power: float = 0.0
    last_relocation: float = -float("inf")
    last_relocation_trip: int = 0
    relocations: int = 0

    def __post_init__(self):
        self.battery = self.config.battery*self.config.initial_charge

    def sun(self, hour, day_of_year=None):
        b = self.config
        return sunlight(b.latitude_deg, b.start_day_of_year if day_of_year is None else day_of_year, hour, b.peak_irradiance)

    def irradiance(self, hour, day_of_year=None):
        return self.sun(hour, day_of_year).irradiance

    def motion_power(self, drone):
        drones = drone if isinstance(drone, list) else [drone]
        total_mass = self.config.mass+sum(d.config.empty_mass+d.cargo for d in drones if d.docked)
        if not isinstance(drone, list) and not drone.docked:
            total_mass += drone.config.empty_mass+drone.cargo
        return self.config.move_power*total_mass/self.config.reference_total_mass

    def tick(self, dt, hour, drone, charging, day_of_year=None,laser_links=(),modules=(),bus_limit_w=None,
             storage_charge_w=None,storage_discharge_w=None,auxiliary_w=0.):
        b = self.config
        drones = drone if isinstance(drone, list) else [drone]
        flags = charging if isinstance(charging, list) else [charging]*len(drones)
        self.laser_input_power=0.
        if self.destination is not None:
            self.state = BaseState.MOVING
            self.solar_power = 0.0
            distance = float(np.linalg.norm(self.destination-self.position[:2]))
            needed = self.motion_power(drone)*min(dt, distance/b.speed)
            if self.battery < needed:
                self.destination = None  # unfold panels at the actual stopping point
                self.state = BaseState.WAITING_FOR_SUN
                return "База остановилась: недостаточно энергии для движения"
            fraction = min(1.0, b.speed*dt/max(distance, 1e-12))
            self.position[:2] += (self.destination-self.position[:2])*fraction
            energy = needed
            self.battery -= energy
            self.consumed += energy
            if distance <= b.speed*dt:
                self.destination = None
                self.state = BaseState.PARKED_GENERATING
                return "База закончила перемещение"
            return None
        self.solar_power = self.irradiance(hour, day_of_year)*b.panel_area*b.panel_efficiency
        solar = self.solar_power*dt
        self.generated += solar
        total_available = self.battery+solar
        available = total_available if bus_limit_w is None else min(total_available,bus_limit_w*dt)
        if storage_discharge_w is not None:
            available=min(available,solar+min(self.battery,storage_discharge_w*dt))
        inaccessible=total_available-available
        idle = min(available, (b.idle_power+auxiliary_w)*dt)
        available -= idle
        self.consumed += idle
        charging_energy = 0.0
        active = [(d,min(b.charge_power,getattr(d,'max_charge_input_w',b.charge_power))) for d, flag in zip(drones, flags) if d.docked and flag and d.battery < d.config.battery-1e-8]
        active += [(module,limit) for module,limit in modules if module.battery<module.config.battery-1e-8]
        budget = min(b.charge_power*dt, available)
        # Equal shares, redistributing unused power when a battery reaches 100%.
        while active and budget > 1e-9:
            share = budget/len(active)
            pending = []
            spent = 0.0
            for d,limit in active:
                energy = min(share,limit*dt, (d.config.battery-d.battery)/b.charge_efficiency)
                d.battery = min(d.config.battery, d.battery+energy*b.charge_efficiency)
                d.charged_energy += energy*b.charge_efficiency
                self.charging_input += energy
                self.charging_losses += energy*(1-b.charge_efficiency)
                spent += energy
                if d.battery < d.config.battery-1e-8 and energy < limit*dt-1e-9:
                    pending.append((d,limit-energy/dt))
            budget -= spent
            charging_energy += spent
            available -= spent
            active = pending
        laser_input,laser_received=transfer(laser_links,max(0.,available),dt)
        available-=laser_input;self.laser_input+=laser_input
        self.laser_input_power=laser_input/dt;self.laser_losses+=laser_input-laser_received
        # Integrate simultaneous generation and discharge before applying capacity.
        # A full base can still feed solar power into a docked drone this tick.
        available += inaccessible
        if storage_charge_w is not None:
            overflow=max(0.,available-self.battery-storage_charge_w*dt)
            self.curtailed+=overflow;available-=overflow
        self.curtailed += max(0.0, available-b.battery)
        self.battery = max(0.0,min(available, b.battery))
        self.state = BaseState.PARKED_GENERATING if self.solar_power > 0 or self.battery > 0 else BaseState.WAITING_FOR_SUN
        if modules or any(d.docked and flag for d, flag in zip(drones, flags)):
            self.state = BaseState.CHARGING if charging_energy > 0 else BaseState.WAITING_FOR_SUN
        elif laser_input>0:self.state=BaseState.CHARGING
        return None

    def safe_path(self, target, terrain, offset=(0, 0)):
        b = self.config
        distance = float(np.linalg.norm(np.asarray(target)-self.position[:2]))
        steps = max(2, int(math.ceil(distance/(terrain.cell/2)))+1)
        points = np.linspace(self.position[:2]+offset, np.asarray(target)+offset, steps)
        side = terrain.config.terrain.n*terrain.cell
        q = terrain.config.trench
        margin = q.depth/terrain.slope + b.radius
        x, y, w, h = terrain.config.terrain.dump_rect
        previous = None
        for p in points:
            if np.any(p < b.radius+terrain.cell) or np.any(p > side-b.radius-terrain.cell):
                return False
            dx = max(abs(p[0]-q.x)-q.length/2, 0)
            dy = max(abs(p[1]-q.y)-q.width/2, 0)
            if math.hypot(dx, dy) < margin:
                return False
            if x-b.radius <= p[0] <= x+w+b.radius and y-b.radius <= p[1] <= y+h+b.radius:
                return False
            z = terrain.sample(*p)
            if previous is not None and abs(z-previous[2]) > b.max_slope*np.linalg.norm(p-previous[:2])+1e-9:
                return False
            for offset in [(b.radius, 0), (-b.radius, 0), (0, b.radius), (0, -b.radius)]:
                if abs(terrain.sample(*(p+offset))-z) > b.max_slope*b.radius:
                    return False
            previous = np.array([*p, z])
        return True

    def maybe_relocate(self, time, trips, dig, dump, terrain, drone):
        b = self.config
        drones = drone if isinstance(drone, list) else [drone]
        if not all(d.docked for d in drones) or self.destination is not None or trips-self.last_relocation_trip < b.relocation_trips or time-self.last_relocation < b.relocation_cooldown:
            return False
        q = terrain.config.trench
        point = np.array([q.x-q.length/2-q.depth/terrain.slope-b.relocation_clearance-b.radius, q.y])
        old_distance = np.linalg.norm(self.position[:2]-dig)+np.linalg.norm(self.position[:2]-dump)
        new_distance = np.linalg.norm(point-dig)+np.linalg.norm(point-dump)
        if new_distance >= old_distance*(1-b.relocation_min_gain) or not self.safe_path(point, terrain):
            return False
        cost = np.linalg.norm(point-self.position[:2])/b.speed*self.motion_power(drone)
        if self.battery < cost:
            return False
        self.destination = point
        self.last_relocation = time
        self.last_relocation_trip = trips
        self.relocations += 1
        return True
