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
    source: str = "cut"
    destination: str = "dump"


class Planner:
    def __init__(self, config):
        self.config = config

    def estimate(self, base, dig, dump, altitude, cargo, drone_config=None, home=None, start=None, min_surface=None):
        d = drone_config or self.config.drone
        home_position = np.asarray(home) if home is not None else base.position+[0, 0, self.config.base.roof_height]
        start_position = np.asarray(start) if start is not None else home_position
        empty = d.hover_power
        loaded = empty*((d.empty_mass+cargo)/d.empty_mass)**1.5
        # Round each state separately because integration uses fixed 1 s ticks.
        up = math.ceil(max(0, altitude-start_position[2])/d.vertical_speed)
        down = math.ceil(max(0, altitude-home_position[2])/d.vertical_speed)
        out = math.ceil(np.linalg.norm(np.asarray(dig)-start_position[:2])/d.empty_speed)
        haul = math.ceil(np.linalg.norm(np.asarray(dig)-dump)/d.loaded_speed)
        home = math.ceil(np.linalg.norm(np.asarray(dump)-home_position[:2])/d.empty_speed)
        lower = math.ceil(d.winch_seconds)
        scoop = math.ceil(d.scoop_duration(cargo))
        unload = math.ceil(d.dump_seconds)
        # Conservative full-load bound covers gradually filling the bucket.
        joules = empty*d.flight_factor*(up+down+out+home) + loaded*d.flight_factor*haul
        joules += (empty+d.winch_power)*lower + loaded*scoop + (loaded+d.winch_power)*lower + (loaded+d.winch_power)*unload
        joules += cargo/self.config.terrain.density*d.digging_j_m3
        surface = -self.config.trench.depth-self.config.terrain.roughness if min_surface is None else min_surface
        joules += cargo*d.gravity*max(0, altitude-surface)/d.winch_efficiency
        if self.config.fleet.count > 1:
            joules += empty*d.flight_factor*(self.config.fleet.emergency_landing_seconds+2*self.config.fleet.detour_distance/d.empty_speed)
        return joules*d.budget_margin

    def choose(self, terrain, base, drone=None, reserved=(), home=None, reserved_dump=(), incoming_fill=0.0, battery=None, start=None):
        d = drone.config if drone is not None else self.config.drone
        charge = (drone.battery if drone is not None else d.battery) if battery is None else battery
        fill_mass = max(0, terrain.remaining_fill*self.config.terrain.density-incoming_fill)
        fill_point = terrain.fill_location(reserved_dump, self.config.fleet.separation) if fill_mass > 1e-8 else None
        sources = ["stockpile", "cut"] if fill_point is not None else ["cut"]
        for source in sources:
            result = self._choose_source(terrain, base, drone, source, reserved, home, reserved_dump, fill_point, fill_mass, charge, start)
            if result is not None:
                return result
        return None

    def _choose_source(self, terrain, base, drone, source, reserved, home, reserved_dump, fill_point, fill_mass, charge, start):
        d, t = (drone.config if drone is not None else self.config.drone), self.config.terrain
        excess = terrain.source_excess(source)
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
        if self.config.reference_height is not None:
            altitude = terrain.clearance_surface+d.work_height+self.config.base.roof_height
        altitude += (drone.identifier if drone is not None else 0)*self.config.fleet.altitude_spacing
        for index in candidates[order]:
            dig = (float(terrain.x.ravel()[index]), float(terrain.y.ravel()[index]))
            mass = min(d.payload, terrain.available_mass(*dig,source,d.grab_radius))
            if fill_point is not None:
                mass = min(mass, fill_mass)
            if mass <= 1e-9:
                continue
            dump = fill_point if fill_point is not None else terrain.dump_location(mass/t.density,reserved_dump,self.config.fleet.separation)
            if dump is None:
                return None
            min_surface = float(min(terrain.height.min(), terrain.target.min()))
            energy = self.estimate(base, dig, dump, altitude, mass, d, home, start, min_surface)
            usable = max(0, charge-d.battery*d.reserve)
            if energy > usable:
                # Smaller loads reduce flight power while preserving the reserve.
                for factor in [0.75, 0.5, 0.25, 0.1]:
                    reduced = mass*factor
                    estimate = self.estimate(base, dig, dump, altitude, reduced, d, home, start, min_surface)
                    if estimate <= usable:
                        mass, energy = reduced, estimate
                        break
                else:
                    continue
            score = mass/energy
            if best is None or score > best[0]:
                best = (score, Trip(dig, dump, mass, energy, altitude, source, "fill" if fill_point is not None else "dump"))
        return best[1] if best else None
