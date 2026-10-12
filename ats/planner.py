from dataclasses import dataclass
import math
import numpy as np


@dataclass
class Trip:
    dig: tuple
    dump: tuple
    mass: float
    energy: float
    altitude: float


class Planner:
    def __init__(self, config):
        self.config = config

    def estimate(self, base, dig, dump, altitude, cargo, drone_config=None, home=None):
        d = drone_config or self.config.drone
        home_position = np.asarray(home) if home is not None else base.position+[0, 0, self.config.base.roof_height]
        empty = d.hover_power
        loaded = empty*((d.empty_mass+cargo)/d.empty_mass)**1.5
        # Round each state separately because integration uses fixed 1 s ticks.
        up = math.ceil(max(0, altitude-home_position[2])/d.vertical_speed)
        out = math.ceil(np.linalg.norm(np.asarray(dig)-home_position[:2])/d.empty_speed)
        haul = math.ceil(np.linalg.norm(np.asarray(dig)-dump)/d.loaded_speed)
        home = math.ceil(np.linalg.norm(np.asarray(dump)-home_position[:2])/d.empty_speed)
        lower = math.ceil(d.winch_seconds)
        scoop = math.ceil(d.scoop_seconds)
        unload = math.ceil(d.dump_seconds)
        # Scooping is charged at empty power, lifting/haul/unload at full load.
        joules = empty*d.flight_factor*(2*up+out+home) + loaded*d.flight_factor*haul
        joules += (empty+d.winch_power)*lower + empty*scoop + (loaded+d.winch_power)*lower + (loaded+d.winch_power)*unload
        joules += cargo*d.gravity*(altitude+self.config.trench.depth+self.config.terrain.roughness)/d.winch_efficiency
        if self.config.fleet.count > 1:
            joules += empty*d.flight_factor*(self.config.fleet.emergency_landing_seconds+2*self.config.fleet.detour_distance/d.empty_speed)
        return joules*d.budget_margin

    def choose(self, terrain, base, drone=None, reserved=(), home=None, reserved_dump=()):
        d, t = (drone.config if drone is not None else self.config.drone), self.config.terrain
        excess = np.where(terrain.excavation, np.maximum(terrain.height-terrain.target, 0), 0)
        for point in reserved:
            excess[(terrain.x-point[0])**2+(terrain.y-point[1])**2 < self.config.fleet.separation**2] = 0
        candidates = np.flatnonzero(excess.ravel() > 1e-12)
        if not len(candidates):
            return None
        # First shortlist useful cells, then score actual grabber footprint volume.
        heights = excess.ravel()[candidates]
        order = np.argsort(-heights, kind="stable")[:32]
        best = None
        altitude = max(float(terrain.reference.max()), t.dump_max_height) + d.work_height + self.config.base.roof_height
        altitude += (drone.identifier if drone is not None else 0)*self.config.fleet.altitude_spacing
        for index in candidates[order]:
            dig = (float(terrain.x.ravel()[index]), float(terrain.y.ravel()[index]))
            _, available = terrain.local_available(*dig)
            mass = min(d.payload, float(available.sum()*terrain.area*t.density))
            if mass <= 1e-9:
                continue
            dump = terrain.dump_location(mass/t.density,reserved_dump,self.config.fleet.separation)
            if dump is None:
                return None
            energy = self.estimate(base, dig, dump, altitude, mass, d, home)
            if energy > d.battery*(1-d.reserve):
                # Smaller loads reduce flight power while preserving the reserve.
                for factor in [0.75, 0.5, 0.25, 0.1]:
                    reduced = mass*factor
                    estimate = self.estimate(base, dig, dump, altitude, reduced, d, home)
                    if estimate <= d.battery*(1-d.reserve):
                        mass, energy = reduced, estimate
                        break
                else:
                    continue
            score = mass/energy
            if best is None or score > best[0]:
                best = (score, Trip(dig, dump, mass, energy, altitude))
        return best[1] if best else None
