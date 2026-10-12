"""Fixed-step autonomous engine; no GUI imports, no wall-clock dependence."""
from collections import deque
from enum import Enum
import math
import numpy as np
from .config import Config, KWH
from .terrain import Terrain
from .drone import Drone, DroneState as DS
from .solar_base import SolarBase, BaseState
from .planner import Planner
from .dust import Dust


class MissionState(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"


class Simulation:
    def __init__(self, config=None):
        self.config = config or Config()
        self.config.validate()
        self.terrain = Terrain(self.config)
        b = self.config.base
        self.base = SolarBase(b, np.array([b.x, b.y, self.terrain.sample(b.x, b.y)]))
        self.drone = Drone(self.config.drone, self.base.position+np.array([0, 0, b.roof_height]))
        self.planner = Planner(self.config)
        self.dust = Dust(self.config)
        self.time = 0.0
        self.state = MissionState.RUNNING
        self.reason = ""
        self.events = deque(maxlen=self.config.event_limit)
        self.event_serial = 0
        self._event_keys = {}
        self.trip = None
        self.trips = 0
        self.excavated_mass = 0.0
        self._terrain_active = self.config.terrain.roughness > 0
        self._initial_energy = self.drone.battery+self.base.battery
        self.minimum_drone_charge = self.drone.battery
        self.event("Сценарий готов: автономный планировщик включён")
        if not self.base.safe_path(self.base.position[:2], self.terrain):
            self.block("База находится на канаве, отвале или непроходимой поверхности")
        elif self.terrain.dump_space < self.terrain.initial_excavation:
            self.block("Недостаточно места в отвале для проектного объёма песка")

    @property
    def hour(self):
        return (self.config.start_hour+self.time/3600) % 24

    def event(self, message, key=None):
        if key is not None and self._event_keys.get(key) == message:
            return
        if key is not None:
            self._event_keys[key] = message
        self.event_serial += 1
        self.events.append((self.event_serial, self.time, message))

    def block(self, reason):
        self.state = MissionState.BLOCKED
        self.reason = reason
        self.event(f"Работа заблокирована: {reason}")

    def consume_drone(self, joules):
        if joules > self.drone.battery+1e-9:
            self.block("Недостаточно энергии дрона: остановка расчёта в безопасной позиции")
            return False
        self.drone.battery = max(0, self.drone.battery-joules)
        self.drone.consumed += joules
        self.minimum_drone_charge = min(self.minimum_drone_charge, self.drone.battery)
        return True

    def _move(self, destination, next_state, dt):
        d = self.drone
        previous = d.position.copy()
        arrived = d.move_towards(destination, dt)
        distance = np.linalg.norm(d.position-previous)
        # Inspect the entire swept segment, not only its end.
        count = max(2, int(math.ceil(distance/(self.terrain.cell/2)))+1)
        for point in np.linspace(previous, d.position, count):
            ground = self.terrain.sample(*point[:2])
            if np.linalg.norm(point[:2]-self.base.position[:2]) <= self.config.base.radius:
                ground = max(ground, self.base.position[2]+self.config.base.roof_height)
            if point[2] < ground-1e-9:
                d.position[:] = previous
                d.velocity[:] = 0
                self.block("Прямая траектория пересекает рельеф или базу")
                return
        if arrived:
            d.transition(next_state)

    def _docked(self):
        d, t = self.drone, self.terrain
        if self.base.destination is not None:
            return
        if t.remaining <= self.config.terrain.completion_volume and d.cargo <= 1e-9:
            if self._terrain_active:
                return  # let residual avalanches settle before acceptance
            self.state = MissionState.COMPLETED
            self.event("Работа завершена: достигнут допуск проектной выемки")
            return
        if self.trip is None:
            self.trip = self.planner.choose(t, self.base)
        if self.trip is None:
            self.block("Нет доступного рейса: отвал заполнен либо батареи недостаточно даже после полной зарядки")
            return
        if self.base.maybe_relocate(self.time, self.trips, self.trip.dig, self.trip.dump, t, d):
            self.event("База перемещается: маршрут рейсов станет короче")
            self.trip = None
            return
        needed = self.trip.energy+self.config.drone.battery*self.config.drone.reserve
        if d.battery+1e-9 < needed:
            d.transition(DS.CHARGING)
            self.event("Ожидает зарядки перед рейсом с резервом на возвращение", "wait")
            if self.base.battery <= 1e-9 and (self.config.base.panel_area*self.config.base.panel_efficiency*self.config.base.peak_irradiance == 0 or self.config.base.charge_power == 0):
                self.block("Зарядка невозможна: база разряжена и солнечная генерация отключена")
            elif self.config.base.charge_power == 0:
                self.block("Зарядка невозможна: мощность зарядного устройства равна нулю")
            elif self.base.battery <= 1e-9 and self.config.base.idle_power >= self.config.base.panel_area*self.config.base.panel_efficiency*self.config.base.peak_irradiance:
                self.block("Зарядка невозможна: собственное потребление базы поглощает всю солнечную мощность")
            return
        self._event_keys.pop("wait", None)
        d.transition(DS.TAKEOFF)
        self.event(f"Рейс {self.trips+1}: запланировано до {self.trip.mass:.1f} кг; бюджет {self.trip.energy/KWH:.3f} кВт·ч")

    def _airborne(self, dt):
        d, c, trip = self.drone, self.config.drone, self.trip
        movement = d.state in (DS.TAKEOFF, DS.TO_DIG, DS.TO_DUMP, DS.RETURN, DS.LANDING)
        winching = d.state in (DS.LOWER_GRABBER, DS.LIFT_GRABBER)
        lift_height = max(0.0, d.position[2]-self.terrain.sample(*d.position[:2])) if d.state == DS.LIFT_GRABBER else 0.0
        if not self.consume_drone(d.power(movement, winching, lift_height=lift_height)*dt):
            return
        d.elapsed += dt
        if d.state == DS.TAKEOFF:
            self._move([*self.base.position[:2], trip.altitude], DS.TO_DIG, dt)
        elif d.state == DS.TO_DIG:
            self._move([*trip.dig, trip.altitude], DS.LOWER_GRABBER, dt)
        elif d.state == DS.LOWER_GRABBER:
            d.grabber_extension = min(1, d.elapsed/c.winch_seconds)
            if d.elapsed >= c.winch_seconds:
                d.transition(DS.SCOOP)
        elif d.state == DS.SCOOP and d.elapsed >= c.scoop_seconds:
            d.cargo = self.terrain.scoop(*trip.dig, trip.mass)
            self.excavated_mass += d.cargo
            self._terrain_active = True
            if d.cargo <= 1e-9:
                self.event("Захват пуст: планировщик выберет другой участок")
            else:
                self.event(f"Завершена выемка: {d.cargo:.1f} кг в грейфере")
            d.transition(DS.LIFT_GRABBER)
        elif d.state == DS.LIFT_GRABBER:
            d.grabber_extension = max(0, 1-d.elapsed/c.winch_seconds)
            if d.elapsed >= c.winch_seconds:
                d.transition(DS.TO_DUMP if d.cargo > 1e-9 else DS.RETURN)
        elif d.state == DS.TO_DUMP:
            self._move([*trip.dump, trip.altitude], DS.DUMP, dt)
        elif d.state == DS.DUMP and d.elapsed >= c.dump_seconds:
            try:
                self.terrain.deposit(*trip.dump, d.cargo)
            except ValueError as error:
                self.block(str(error))  # cargo remains in the aircraft
                return
            d.cargo = 0.0
            self.trips += 1
            self._terrain_active = True
            self.event(f"Рейс {self.trips}: песок выгружен в отвал")
            d.transition(DS.RETURN)
        elif d.state == DS.RETURN:
            self._move([*self.base.position[:2], trip.altitude], DS.LANDING, dt)
        elif d.state == DS.LANDING:
            self._move(self.base.position+[0, 0, self.config.base.roof_height], DS.DOCKED, dt)
            if d.docked:
                self.trip = None

    def step(self, dt=None):
        dt = self.config.dt if dt is None else dt
        if dt != self.config.dt:
            raise ValueError("Передавайте фиксированный шаг 1 с; ускорение = число шагов")
        if self.state != MissionState.RUNNING:
            return
        d = self.drone
        base_was_moving = self.base.destination is not None
        message = self.base.tick(dt, (self.hour+dt/7200) % 24, d, d.state == DS.CHARGING)
        if message:
            self.event(message)
        if d.docked:
            self.base.position[2] = self.terrain.sample(*self.base.position[:2])
            old = d.position.copy()
            d.position[:] = self.base.position+[0, 0, self.config.base.roof_height]
            d.velocity[:] = (d.position-old)/dt
            if not base_was_moving:
                self._docked()
        else:
            self._airborne(dt)
        if self._terrain_active:
            moved = self.terrain.relax()
            if moved > 1e-8:
                self.event("Началось осыпание песка", "avalanche")
            if moved <= self.config.terrain.avalanche_stop_volume:
                self._terrain_active = False
                self._event_keys.pop("avalanche", None)
        if self.state == MissionState.RUNNING:
            working = d.state in (DS.LOWER_GRABBER, DS.SCOOP, DS.LIFT_GRABBER, DS.DUMP)
            self.dust.tick(dt, d.position, d.position[2]-self.terrain.sample(*d.position[:2]), d.power()/self.config.drone.hover_power, working)
        self.time += dt

    def run(self, steps):
        for _ in range(steps):
            if self.state != MissionState.RUNNING:
                break
            self.step()
        return self.report()

    def report(self):
        t, d, b = self.terrain, self.drone, self.base
        ground_error = t.mass+d.cargo-t.initial_volume*self.config.terrain.density
        energy_error = (d.battery+b.battery+d.consumed+b.consumed+b.charging_losses+b.curtailed)-(self._initial_energy+b.generated)
        remaining = t.remaining
        return {
            "state": self.state.value, "reason": self.reason, "time_seconds": self.time,
            "hour": self.hour, "trips": self.trips, "relocations": b.relocations,
            "initial_excavation_m3": t.initial_excavation, "remaining_m3": remaining,
            "completion_percent": max(0.0, min(100.0, 100*(1-remaining/max(t.initial_excavation, 1e-12)))),
            "excavated_m3": self.excavated_mass/self.config.terrain.density,
            "excavated_mass_kg": self.excavated_mass, "cargo_kg": d.cargo,
            "drone_state": d.state.value, "base_state": b.state.value,
            "drone_kwh": d.battery/KWH, "base_kwh": b.battery/KWH,
            "solar_power_w": b.solar_power, "solar_generated_kwh": b.generated/KWH,
            "solar_curtailed_kwh": b.curtailed/KWH,
            "consumed_kwh": (d.consumed+b.consumed)/KWH,
            "charging_losses_kwh": b.charging_losses/KWH,
            "mass_error_kg": ground_error, "energy_error_j": energy_error,
            "minimum_drone_kwh": self.minimum_drone_charge/KWH,
        }
