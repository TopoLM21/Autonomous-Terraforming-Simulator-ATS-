"""Closed-boundary, float64 heightfield with conservative sand transfers."""
import math
import numpy as np


class Terrain:
    def __init__(self, config):
        config.validate()
        self.config = config
        t, q = config.terrain, config.trench
        self.cell = t.cell_size
        self.area = self.cell**2
        self.axis = (np.arange(t.n) + 0.5) * self.cell
        self.x, self.y = np.meshgrid(self.axis, self.axis)
        rng = np.random.default_rng(config.seed)
        self.height = rng.uniform(-t.roughness, t.roughness, (t.n, t.n)).astype(np.float64)
        self.reference = self.height.copy()
        self.slope = math.tan(math.radians(q.angle_deg))
        dx = np.maximum(np.abs(self.x-q.x)-q.length/2, 0)
        dy = np.maximum(np.abs(self.y-q.y)-q.width/2, 0)
        self.bottom = (dx == 0) & (dy == 0)
        self.target = np.minimum(self.reference, -q.depth + self.slope*np.hypot(dx, dy))
        self.excavation = self.target < self.reference - 1e-12
        x, y, w, h = t.dump_rect
        self.dump_mask = (self.x >= x) & (self.x < x+w) & (self.y >= y) & (self.y < y+h)
        if np.any(self.dump_mask & self.excavation):
            raise ValueError("Отвал пересекает канаву с откосами")
        # The cap is a stable tent with a one-cell setback. It reserves space for
        # avalanches and prevents sand leaving the designated dump rectangle.
        distance = np.maximum(0, np.minimum.reduce([self.x-x, x+w-self.x, self.y-y, y+h-self.y])-self.cell)
        self.dump_cap = self.reference + np.where(self.dump_mask, np.minimum(t.dump_max_height, distance*self.slope*t.dump_slope_factor), 0)
        self.initial_volume = self.volume
        self.initial_excavation = self.remaining
        self.revision = 0
        self._pairs = []
        grid = np.arange(t.n*t.n).reshape(t.n, t.n)
        # Eight-neighbour slopes, split into disjoint matchings. Sequential
        # equal-and-opposite pair updates cannot overshoot or double-spend sand.
        for oy, ox in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            ya = slice(0, t.n-oy)
            xa = slice(max(0, -ox), min(t.n, t.n-ox))
            yb = slice(oy, t.n)
            xb = slice(max(0, ox), min(t.n, t.n+ox))
            a, b = grid[ya, xa], grid[yb, xb]
            colors = np.indices(a.shape)[1 if oy == 0 else 0] % 2
            for color in [0, 1]:
                self._pairs.append((a[colors == color], b[colors == color], self.slope*self.cell*math.hypot(oy, ox)))

    @property
    def volume(self):
        return float(np.sum(self.height-self.config.terrain.floor, dtype=np.float64)*self.area)

    @property
    def mass(self):
        return self.volume*self.config.terrain.density

    @property
    def remaining(self):
        revision=getattr(self,"revision",0)
        if getattr(self,"_remaining_revision",None)!=revision:
            self._remaining_cache=float(np.maximum(self.height-self.target, 0)[self.excavation].sum()*self.area)
            self._remaining_revision=revision
        return self._remaining_cache

    @property
    def dump_space(self):
        return float(np.maximum(self.dump_cap-self.height, 0)[self.dump_mask].sum()*self.area)

    def sample(self, x, y):
        # Nearest-cell sampling uses exactly the same cell convention as edits.
        ix = min(max(int(x/self.cell), 0), self.height.shape[1]-1)
        iy = min(max(int(y/self.cell), 0), self.height.shape[0]-1)
        return float(self.height[iy, ix])

    def local_available(self, x, y):
        mask = self.excavation & ((self.x-x)**2+(self.y-y)**2 <= self.config.terrain.grab_radius**2)
        return mask, np.where(mask, np.maximum(self.height-self.target, 0), 0)

    def scoop(self, x, y, max_mass):
        if not math.isfinite(max_mass) or max_mass < 0:
            raise ValueError("Неверная масса захвата")
        _, excess = self.local_available(x, y)
        available = float(excess.sum()*self.area)
        volume = min(max_mass/self.config.terrain.density, available)
        if available <= 0:
            return 0.0
        # Proportional removal never digs below the design and changes many cells.
        self.height -= excess*(volume/available)
        self.revision += 1
        return volume*self.config.terrain.density

    def dump_location(self, volume, reserved=(), separation=0.0):
        capacity = np.where(self.dump_mask, np.maximum(self.dump_cap-self.height, 0), 0)
        if capacity.sum()*self.area < volume-1e-12:
            return None
        candidates = np.flatnonzero(capacity.ravel() > 1e-12)
        for point in reserved:
            candidates=candidates[(self.x.ravel()[candidates]-point[0])**2+(self.y.ravel()[candidates]-point[1])**2 >= separation**2]
        if not len(candidates):
            return None
        # Lowest available site produces distributed heaps, not a single spire.
        score = self.height.ravel()[candidates] + 0.02*np.hypot(self.x.ravel()[candidates]-self.x[self.dump_mask].mean(), self.y.ravel()[candidates]-self.y[self.dump_mask].mean())
        i = candidates[int(np.argmin(score))]
        return (float(self.x.ravel()[i]), float(self.y.ravel()[i]))

    def deposit(self, x, y, mass):
        if not math.isfinite(mass) or mass < 0:
            raise ValueError("Неверная масса выгрузки")
        volume = mass/self.config.terrain.density
        capacity = np.where(self.dump_mask, np.maximum(self.dump_cap-self.height, 0), 0)
        if capacity.sum()*self.area < volume-1e-12:
            raise ValueError("Отвал заполнен: груз не выгружен")
        local = ((self.x-x)**2+(self.y-y)**2 <= self.config.terrain.grab_radius**2)
        local_cap = np.where(local, capacity, 0)
        if local_cap.sum()*self.area < volume-1e-12:
            # Expand the footprint around the same unload site only as needed.
            order = np.argsort(((self.x-x)**2+(self.y-y)**2).ravel(), kind="stable")
            cap = capacity.ravel()[order]*self.area
            needed = min(len(order), int(np.searchsorted(np.cumsum(cap), volume))+1)
            local = np.zeros(self.height.size, dtype=bool)
            local[order[:needed]] = True
            local_cap = np.where(local.reshape(self.height.shape), capacity, 0)
        total = float(local_cap.sum()*self.area)
        if volume > 0:
            self.height += local_cap*(volume/total)
            self.revision += 1

    def relax(self, sweeps=None):
        t = self.config.terrain
        height = self.height.ravel()
        moved = 0.0
        for _ in range(t.avalanche_sweeps if sweeps is None else sweeps):
            sweep_moved = 0.0
            for a, b, limit in self._pairs:
                difference = height[a]-height[b]
                active = np.abs(difference) > limit+t.avalanche_epsilon
                if not np.any(active):
                    continue
                a, b, difference = a[active], b[active], difference[active]
                transfer = np.sign(difference)*(np.abs(difference)-limit-t.avalanche_epsilon)*t.relaxation
                height[a] -= transfer
                height[b] += transfer
                sweep_moved += float(np.abs(transfer).sum())*self.area
            moved += sweep_moved
            if sweep_moved == 0:
                break
        if moved > 0:
            self.revision += 1
        return moved

    def max_slope_excess(self):
        h = self.height.ravel()
        return max(float(np.maximum(np.abs(h[a]-h[b])-limit, 0).max()) for a, b, limit in self._pairs)
