from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class DroneState(str, Enum):
    DOCKED = "DOCKED"
    CHARGING = "CHARGING"
    TAKEOFF = "TAKEOFF"
    TO_DIG = "TO_DIG"
    LOWER_GRABBER = "LOWER_GRABBER"
    SCOOP = "SCOOP"
    LIFT_GRABBER = "LIFT_GRABBER"
    TO_DUMP = "TO_DUMP"
    DUMP = "DUMP"
    RETURN = "RETURN"
    LANDING = "LANDING"


@dataclass
class Drone:
    config: object
    position: np.ndarray
    battery: float = field(init=False)
    cargo: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    state: DroneState = DroneState.DOCKED
    elapsed: float = 0.0
    consumed: float = 0.0
    grabber_extension: float = 0.0
    dump_released: bool = False

    def __post_init__(self):
        self.battery = self.config.battery*self.config.initial_charge

    @property
    def docked(self):
        return self.state in (DroneState.DOCKED, DroneState.CHARGING)

    def power(self, moving=False, winching=False, cargo=None, lift_height=0.0):
        load = self.cargo if cargo is None else cargo
        p = self.config.hover_power*((self.config.empty_mass+load)/self.config.empty_mass)**1.5
        winch = 0.0
        if winching:
            winch = self.config.winch_power
            winch += self.config.gravity*load*max(0.0, lift_height)/(self.config.winch_efficiency*self.config.winch_seconds)
        return p*(self.config.flight_factor if moving else 1) + winch

    def transition(self, state):
        self.state = state
        self.elapsed = 0.0
        self.velocity[:] = 0
        if state == DroneState.DUMP:
            self.dump_released = False

    def cable_length(self, surface_height):
        if self.docked:
            return 0.0
        clearance = max(0.0, self.position[2]-surface_height)
        short = min(clearance, self.config.stowed_cable)
        return short+self.grabber_extension*(clearance-short)

    def move_towards(self, target, dt):
        target = np.asarray(target, dtype=float)
        previous = self.position.copy()
        difference = target-self.position
        horizontal = float(np.linalg.norm(difference[:2]))
        speed = self.config.loaded_speed if self.cargo > 0 else self.config.empty_speed
        if horizontal:
            self.position[:2] += difference[:2]*min(1.0, speed*dt/horizontal)
        self.position[2] += np.clip(difference[2], -self.config.vertical_speed*dt, self.config.vertical_speed*dt)
        self.velocity = (self.position-previous)/dt
        return bool(np.linalg.norm(target-self.position) < 1e-9)
