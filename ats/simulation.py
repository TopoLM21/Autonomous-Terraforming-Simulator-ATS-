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
    def __init__(self, config=None):
        self.config = config or Config()
        self.config.validate()
        self.terrain = Terrain(self.config)
        b = self.config.base
        self.base = SolarBase(b, np.array([b.x, b.y, self.terrain.sample(b.x, b.y)]))
        self.drones = []
        self.slots = [None]*self.config.fleet.charge_slots
        self.queue = []
        for index, model in enumerate(self.config.drone_models()):
            slot = index if index < len(self.slots) else None
            point = self.slot_position(slot) if slot is not None else self.wait_position(index)
            d = Drone(model, point, identifier=index, slot=slot)
            if slot is None:
                d.state = DS.WAIT_GROUND
                if d.battery < d.config.battery-1e-7:
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
        self._relocation_attempt_time = -float("inf")
        self._terrain_active = self.config.terrain.roughness > 0
        self._initial_energy = sum(d.battery for d in self.drones)+self.base.battery
        self._starts = {d.identifier:d.position.copy() for d in self.drones}
        self.event(f"Сценарий готов: дронов {len(self.drones)}, зарядных слотов {len(self.slots)}")
        points = [self.slot_position(i) for i in range(len(self.slots))]+[self.wait_position(i) for i in range(len(self.drones))]
        if not self.base.safe_path(self.base.position[:2], self.terrain) or any(not self._safe_pad(p) for p in points):
            self.block("База или посадочные площадки пересекают канаву, отвал либо край карты")
        elif self.terrain.dump_space < self.terrain.initial_excavation:
            self.block("Недостаточно места в отвале для проектного объёма песка")

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
        offset = (slot-(len(self.slots)-1)/2)*self.config.fleet.pad_spacing
        return self.base.position+np.array([0, offset, self.config.base.roof_height])

    def wait_position(self, index):
        f = self.config.fleet
        x = self.base.position[0]-f.waiting_offset
        y = self.base.position[1]+(index-(self.config.fleet.count-1)/2)*f.pad_spacing
        return np.array([x, y, self.terrain.sample(x, y)+f.ground_clearance])

    def cruise(self, d):
        return max(float(self.terrain.reference.max()), self.config.terrain.dump_max_height)+d.config.work_height+self.config.base.roof_height+d.identifier*self.config.fleet.altitude_spacing

    def _safe_pad(self, p):
        t = self.terrain
        side = t.config.terrain.n*t.cell
        if not (t.cell < p[0] < side-t.cell and t.cell < p[1] < side-t.cell):
            return False
        ix, iy = int(p[0]/t.cell), int(p[1]/t.cell)
        margin = self.config.fleet.separation/2
        near = (t.x-p[0])**2+(t.y-p[1])**2 < margin**2
        return not np.any((t.excavation|t.dump_mask)&near) and not t.excavation[iy, ix]

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
        self.event(f"Дрон {d.identifier+1} ({d.config.name}): {message}", (d.identifier,key) if key else None)

    def block(self, reason):
        self.state = MissionState.BLOCKED
        self.reason = reason
        self.event(f"Работа заблокирована: {reason}")

    def consume_drone(self, joules, drone=None):
        d = drone or self.drone
        if joules > d.battery+1e-9:
            self.block(f"Недостаточно энергии дрона {d.identifier+1}; расчёт остановлен до разряда ниже нуля")
            return False
        d.battery = max(0, d.battery-joules)
        d.consumed += joules
        d.minimum_charge = min(d.minimum_charge, d.battery)
        return True

    def _clear_slot(self, d):
        if d.slot is not None and self.slots[d.slot] == d.identifier:
            self.slots[d.slot] = None
        d.slot = None

    def _enqueue(self, d):
        if d.identifier not in self.queue:
            self.queue.append(d.identifier)

    def _reserve_return(self, d):
        available = next((i for i,v in enumerate(self.slots) if v is None), None)
        if available is not None and not self.queue:
            d.slot = available
            self.slots[available] = d.identifier
            d.destination = self.slot_position(available)
        else:
            d.destination = self.wait_position(d.identifier)
        d.detour = None

    def _assign_slots(self):
        if not self.queue or self.base.destination is not None or self.terrain.remaining <= self.config.terrain.completion_volume:
            return
        for slot in range(len(self.slots)):
            if self.slots[slot] is not None:
                continue
            eligible = next((i for i in self.queue if self.drones[i].state in (DS.WAIT_GROUND,DS.WAIT_HOVER) and self.drones[i].battery < self.drones[i].config.battery-1e-7 and self.drones[i].battery >= self.drones[i].config.battery*self.drones[i].config.reserve+self._transfer_energy(self.drones[i],slot)), None)
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
        if other is not None:
            d.position[:] = previous
            d.velocity[:] = 0
            d.traffic_yields += 1
            self.drone_event(d,f"уступает путь дрону {other.identifier+1}","traffic")
            direction = np.asarray(destination)[:2]-previous[:2]
            length = float(np.linalg.norm(direction))
            if length > 1e-9:
                normal = np.array([-direction[1],direction[0]])/length
                side = self.config.terrain.n*self.terrain.cell
                for sign in (1,-1):
                    point = previous.copy()
                    point[:2] += sign*normal*self.config.fleet.detour_distance
                    if np.all(point[:2] > self.terrain.cell) and np.all(point[:2] < side-self.terrain.cell) and conflicts(d,previous,point,self.drones,self._starts,self.config.fleet.separation) is None:
                        d.detour = point
                        break
            return
        self._event_keys.pop((d.identifier,"traffic"),None)
        distance = np.linalg.norm(d.position-previous)
        count = max(2,int(math.ceil(distance/(self.terrain.cell/2)))+1)
        for p in np.linspace(previous,d.position,count):
            ground = self.terrain.sample(*p[:2])
            if np.linalg.norm(p[:2]-self.base.position[:2]) <= self.config.base.radius:
                ground = max(ground,self.base.position[2]+self.config.base.roof_height)
            if p[2] < ground-1e-9:
                d.position[:] = previous
                d.velocity[:] = 0
                self.block(f"Траектория дрона {d.identifier+1} пересекает рельеф или базу")
                return
        if arrived:
            if d.detour is not None:
                d.detour = None
            else:
                d.transition(next_state)

    def _reserved_dig(self,d):
        working = (DS.TAKEOFF,DS.TO_DIG,DS.LOWER_GRABBER,DS.SCOOP,DS.LIFT_GRABBER)
        return [o.trip.dig for o in self.drones if o is not d and o.trip is not None and o.state in working]

    def _reserved_dump(self,d):
        return [o.trip.dump for o in self.drones if o is not d and o.trip is not None and o.state not in (DS.RETURN,DS.LANDING)]

    def _docked(self,d):
        if self.base.destination is not None:
            return
        if self.terrain.remaining <= self.config.terrain.completion_volume:
            return
        if d.battery < d.config.battery-1e-7:
            d.state = DS.CHARGING
            self.drone_event(d,"заряжается до 100% перед следующим вылетом","charge")
            return
        d.state = DS.DOCKED
        if d.trip is None:
            d.trip = self.planner.choose(self.terrain,self.base,d,self._reserved_dig(d),self.slot_position(d.slot),self._reserved_dump(d))
        if d.trip is None:
            self.drone_event(d,"ожидает свободного участка выемки","job")
            return
        self._event_keys.pop((d.identifier,"job"),None)
        self._event_keys.pop((d.identifier,"charge"),None)
        d.transition(DS.TAKEOFF)
        self.drone_event(d,f"вылет с зарядом {100*d.battery/d.config.battery:.1f}%; план до {d.trip.mass:.1f} кг")

    def _ready_ground(self,d):
        if d.battery < d.config.battery-1e-7 or self.terrain.remaining <= self.config.terrain.completion_volume:
            return
        trip=self.planner.choose(self.terrain,self.base,d,self._reserved_dig(d),d.position,self._reserved_dump(d))
        if trip is not None:
            d.trip=trip
            if d.identifier in self.queue:self.queue.remove(d.identifier)
            d.transition(DS.TAKEOFF)
            self.drone_event(d,"вылет с наземной площадки с зарядом 100%")

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
        movement = d.state in (DS.TAKEOFF,DS.TO_DIG,DS.TO_DUMP,DS.RETURN,DS.LANDING,DS.TRANSFER_TAKEOFF,DS.TO_SLOT,DS.SLOT_LANDING,DS.WAIT_LANDING)
        winching = d.state in (DS.LOWER_GRABBER,DS.LIFT_GRABBER,DS.DUMP)
        lift_height = max(0,d.position[2]-self.terrain.sample(*d.position[:2])-c.stowed_cable) if d.state == DS.LIFT_GRABBER else 0
        if not self.consume_drone(d.power(movement,winching,lift_height=lift_height)*dt,d):
            return
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
        elif d.state == DS.SCOOP and d.elapsed >= c.scoop_seconds:
            d.cargo = self.terrain.scoop(*trip.dig,trip.mass)
            d.excavated_mass += d.cargo
            self.excavated_mass += d.cargo
            self._terrain_active = True
            self.drone_event(d,f"завершена выемка: {d.cargo:.1f} кг в грейфере")
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
                d.grabber_extension = min(1,d.elapsed/c.winch_seconds)
                if d.elapsed >= c.winch_seconds:
                    try:
                        self.terrain.deposit(*trip.dump,d.cargo)
                    except ValueError as error:
                        self.block(str(error)); return
                    mass = d.cargo
                    d.cargo = 0
                    d.dump_released = True
                    d.trips += 1
                    self.trips += 1
                    self._terrain_active = True
                    self.drone_event(d,f"грейфер раскрыт у поверхности, выгружено {mass:.1f} кг; рейс {d.trips}")
            else:
                d.grabber_extension = max(0,(c.dump_seconds-d.elapsed)/(c.dump_seconds-c.winch_seconds))
                if d.elapsed >= c.dump_seconds:
                    d.transition(DS.RETURN)
                    self._reserve_return(d)
        elif d.state == DS.RETURN:
            self._move(d,[*d.destination[:2],self.cruise(d)],DS.LANDING,dt)
        elif d.state == DS.LANDING:
            air_wait = d.slot is None and self.config.fleet.wait_mode == "air" and self.terrain.remaining > self.config.terrain.completion_volume
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
            if self.terrain.remaining <= self.config.terrain.completion_volume or d.battery <= c.battery*c.reserve+landing_budget:
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

    def _maybe_relocate(self):
        if not all(d.docked for d in self.drones) or any(d.trip for d in self.drones):
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
            offset = (0,(slot-(len(self.slots)-1)/2)*f.pad_spacing)
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
        if not all(d.grounded for d in self.drones):
            return
        if self.terrain.remaining <= self.config.terrain.completion_volume and not self._terrain_active and all(d.cargo == 0 for d in self.drones):
            self.state = MissionState.COMPLETED
            self.event("Работа завершена: все дроны приземлились, достигнут допуск выемки")
            return
        if self.terrain.remaining > self.config.terrain.completion_volume:
            full = [d for d in self.drones if d.battery >= d.config.battery-1e-7]
            if full and len(full) == len(self.drones) and all(self.planner.choose(self.terrain,self.base,d) is None for d in full):
                self.block("Нет доступного рейса ни для одного дрона: отвал заполнен или ёмкости батарей недостаточно")
            elif self.config.base.charge_power == 0 and not any(self.planner.choose(self.terrain,self.base,d) is not None for d in full):
                self.block("Зарядное устройство отключено; полная зарядка перед продолжением невозможна")
            elif self.base.battery <= 1e-8 and (self.config.base.charge_power == 0 or self.config.base.panel_area*self.config.base.panel_efficiency*self.config.base.peak_irradiance == 0) and not any(d.docked and d.battery >= d.config.battery-1e-7 for d in self.drones):
                self.block("Полная зарядка невозможна: база разряжена, солнечная генерация или зарядное устройство отключены")
            elif all(d.state == DS.WAIT_GROUND for d in self.drones) and all(not any(d.battery >= d.config.battery*d.config.reserve+self._transfer_energy(d,i) for i in range(len(self.slots))) for d in self.drones):
                self.block("Дронам на площадках не хватает энергии для перелёта к зарядке с резервом; требуется внешняя доставка")
            elif self.base.battery <= 1e-8 and self.config.base.idle_power >= self.config.base.panel_area*self.config.base.panel_efficiency*self.config.base.peak_irradiance:
                self.block("Собственное потребление базы поглощает всю солнечную мощность")

    def step(self,dt=None):
        dt = self.config.dt if dt is None else dt
        if dt != self.config.dt:
            raise ValueError("Передавайте фиксированный шаг 1 с; ускорение = число шагов")
        if self.state != MissionState.RUNNING:
            return
        self._starts = {d.identifier:d.position.copy() for d in self.drones}
        moving = self.base.destination is not None
        for d in self.drones:
            if d.docked and d.battery < d.config.battery-1e-7:
                d.state = DS.CHARGING
                d.charging_seconds += dt
        flags = [d.state == DS.CHARGING and d.slot is not None for d in self.drones]
        message = self.base.tick(dt,(self.hour+dt/7200)%24,self.drones,flags,self.day_at(self.time+dt/2))
        if message:
            self.event(message)
        if not moving:
            for d in self.drones:
                if d.state in (DS.WAIT_GROUND,DS.WAIT_HOVER) and d.battery < d.config.battery-1e-7:
                    self._enqueue(d)
            self._assign_slots()
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

    def run(self,steps):
        for _ in range(steps):
            if self.state != MissionState.RUNNING:
                break
            self.step()
        return self.report()

    def report(self):
        t,b = self.terrain,self.base
        cargo = sum(d.cargo for d in self.drones)
        consumed = sum(d.consumed for d in self.drones)
        batteries = sum(d.battery for d in self.drones)
        ground_error = t.mass+cargo-t.initial_volume*self.config.terrain.density
        energy_error = batteries+b.battery+consumed+b.consumed+b.charging_losses+b.curtailed-self._initial_energy-b.generated
        remaining = t.remaining
        sun = b.sun(self.hour,self.day_of_year)
        drone_rows = []
        for d in self.drones:
            drone_rows.append({"id":d.identifier+1,"name":d.config.name,"state":d.state.value,"position":d.position.tolist(),"battery_kwh":d.battery/KWH,"battery_percent":100*d.battery/d.config.battery,"capacity_kwh":d.config.battery/KWH,"cargo_kg":d.cargo,"payload_kg":d.config.payload,"empty_mass_kg":d.config.empty_mass,"trips":d.trips,"excavated_mass_kg":d.excavated_mass,"consumed_kwh":d.consumed/KWH,"waiting_seconds":d.waiting_seconds,"charging_seconds":d.charging_seconds,"traffic_yields":d.traffic_yields,"slot":d.slot+1 if d.slot is not None else None,"minimum_kwh":d.minimum_charge/KWH,"cable_length_m":d.cable_length(t.sample(*d.position[:2]))})
        return {"state":self.state.value,"reason":self.reason,"time_seconds":self.time,"elapsed_display":elapsed_text(self.time),"solar_clock":self.clock_text,"date":self.date_text,"day_of_year":self.day_of_year,"latitude_deg":b.config.latitude_deg,"base_mass_kg":b.config.mass,"panel_area_m2":b.config.panel_area,"irradiance_w_m2":sun.irradiance,"sun_elevation_deg":sun.elevation_deg,"daylight_hours":sun.daylight_hours,"sunrise_hour":sun.sunrise_hour,"sunset_hour":sun.sunset_hour,"base_motion_power_w":b.motion_power(self.drones) if b.destination is not None else 0.0,"cable_length_m":drone_rows[0]["cable_length_m"],"hour":self.hour,"trips":self.trips,"relocations":b.relocations,"initial_excavation_m3":t.initial_excavation,"remaining_m3":remaining,"completion_percent":max(0,min(100,100*(1-remaining/max(t.initial_excavation,1e-12)))),"excavated_m3":self.excavated_mass/self.config.terrain.density,"excavated_mass_kg":self.excavated_mass,"cargo_kg":cargo,"drone_state":self.drone.state.value,"base_state":b.state.value,"drone_kwh":batteries/KWH,"base_kwh":b.battery/KWH,"solar_power_w":b.solar_power,"solar_generated_kwh":b.generated/KWH,"solar_curtailed_kwh":b.curtailed/KWH,"consumed_kwh":(consumed+b.consumed)/KWH,"charging_losses_kwh":b.charging_losses/KWH,"mass_error_kg":ground_error,"energy_error_j":energy_error,"minimum_drone_kwh":min(d.minimum_charge/KWH for d in self.drones),"initial_energy_kwh":self._initial_energy/KWH,"drone_count":len(self.drones),"charge_slots":len(self.slots),"slot_occupants":[v+1 if v is not None else None for v in self.slots],"queue":[i+1 for i in self.queue],"wait_mode":self.config.fleet.wait_mode,"drones":drone_rows}
