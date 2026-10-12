"""Qualitative visibility field; deliberately independent of sand mass."""
import math
import numpy as np


class Dust:
    def __init__(self, config):
        self.config = config.dust
        self.side = config.terrain.n*config.terrain.cell_size
        self.n = math.ceil(config.terrain.n/self.config.downsample)
        self.cell = self.side/self.n
        self.field = np.zeros((self.n, self.n), dtype=np.float64)
        self.y, self.x = np.indices(self.field.shape, dtype=float)

    def tick(self, dt, position, altitude, power_ratio, working):
        c = self.config
        if not working and self.field.max() < 1e-8:
            self.field.fill(0)
            return
        # Semi-Lagrangian backtracing is stable even with strong wind; dust can
        # leave the map. This is not a conservative physical sand transport.
        source_x = self.x-c.wind_x*dt/self.cell
        source_y = self.y-c.wind_y*dt/self.cell
        ix, iy = np.floor(source_x).astype(int), np.floor(source_y).astype(int)
        fx, fy = source_x-ix, source_y-iy
        new = np.zeros_like(self.field)
        for ox, oy, weight in [(0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)), (0, 1, (1-fx)*fy), (1, 1, fx*fy)]:
            xx, yy = ix+ox, iy+oy
            inside = (xx >= 0) & (xx < self.n) & (yy >= 0) & (yy < self.n)
            new[inside] += weight[inside]*self.field[yy[inside], xx[inside]]
        self.field = new
        count = max(1, math.ceil(c.diffusion*dt/self.cell**2/0.24))
        alpha = c.diffusion*dt/self.cell**2/count
        for _ in range(count):
            padded = np.pad(self.field, 1, mode="edge")
            self.field += alpha*(padded[1:-1, 2:]+padded[1:-1, :-2]+padded[2:, 1:-1]+padded[:-2, 1:-1]-4*self.field)
        self.field *= math.exp(-dt/c.lifetime)
        if working and c.strength:
            distance = ((self.x+0.5)*self.cell-position[0])**2+((self.y+0.5)*self.cell-position[1])**2
            emission = c.strength*power_ratio*math.exp(-max(0, altitude)/c.height_scale)
            self.field += emission*dt*np.exp(-distance/(2*self.cell**2))
        np.maximum(self.field, 0, out=self.field)
