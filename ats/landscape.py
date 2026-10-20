"""Offline coarse DEM with sparse, conservative fine-grid work patches.

Coordinates in the simulator are always metres; elevations are metres above MSL.
No hydrology and no automatic regional excavation/transport are implied.
"""
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import json
import math
import numpy as np

EARTH_RADIUS = 6_371_008.8


@dataclass
class ElevationRaster:
    height: np.ndarray  # row zero is SOUTH, samples at (x0 + col*dx, y0 + row*dy)
    metadata: dict

    def __post_init__(self):
        self.height = np.array(self.height, dtype=np.float64, copy=True)
        self.metadata = dict(self.metadata)
        m = self.metadata
        if self.height.ndim != 2 or min(self.height.shape) < 2 or self.height.size > 16_000_000:
            raise ValueError('DEM: требуется двумерная сетка 2×2…16 млн отсчётов')
        if m.get('vertical_datum') != 'MSL':
            raise ValueError('DEM: нужны метры относительно уровня моря (MSL); эллипсоидальные высоты сначала преобразуйте')
        if m.get('crs') != 'EPSG:4326' and m.get('units') != 'metres':
            raise ValueError('DEM: задайте EPSG:4326 или проекцию с единицами metres')
        if not all(math.isfinite(m.get(k, float('nan'))) for k in ('x0', 'y0', 'dx', 'dy')) or min(m['dx'], m['dy']) <= 0:
            raise ValueError('DEM: неверная привязка или шаг')
        if not np.any(np.isfinite(self.height)) or np.any(np.isinf(self.height)):
            raise ValueError('DEM: нет конечных высот или обнаружена бесконечность')
        if m['crs'] == 'EPSG:4326':
            if not (-90 < m['y0'] < 90 and self.north < 90 and -180 <= m['x0'] < self.east <= 180):
                raise ValueError('DEM: неверные границы WGS84')
        self.height.flags.writeable = False

    @property
    def east(self):
        return self.metadata['x0']+(self.height.shape[1]-1)*self.metadata['dx']

    @property
    def north(self):
        return self.metadata['y0']+(self.height.shape[0]-1)*self.metadata['dy']

    @property
    def scale(self):
        if self.metadata['crs'] == 'EPSG:4326':
            lat = (self.metadata['y0']+self.north)/2
            return EARTH_RADIUS*math.pi/180*math.cos(math.radians(lat)), EARTH_RADIUS*math.pi/180
        return 1., 1.

    @property
    def extent_m(self):
        sx, sy = self.scale
        return (self.east-self.metadata['x0'])*sx, (self.north-self.metadata['y0'])*sy

    @property
    def resolution_m(self):
        sx, sy = self.scale
        return self.metadata['dx']*sx, self.metadata['dy']*sy

    def to_metres(self, x, y):
        sx, sy = self.scale
        return (np.asarray(x)-self.metadata['x0'])*sx, (np.asarray(y)-self.metadata['y0'])*sy

    def to_source(self, x, y):
        sx, sy = self.scale
        return self.metadata['x0']+np.asarray(x)/sx, self.metadata['y0']+np.asarray(y)/sy

    def sample(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        width, height = self.extent_m
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(x < 0) or np.any(x > width) or np.any(y < 0) or np.any(y > height):
            raise ValueError('Участок выходит за границы DEM')
        rx, ry = self.resolution_m
        fx, fy = x/rx, y/ry
        ix, iy = np.minimum(fx.astype(int), self.height.shape[1]-2), np.minimum(fy.astype(int), self.height.shape[0]-2)
        ux, uy = fx-ix, fy-iy
        corners = []
        for ox, oy, weight in [(0,0,(1-ux)*(1-uy)), (1,0,ux*(1-uy)), (0,1,(1-ux)*uy), (1,1,ux*uy)]:
            z = self.height[iy+oy,ix+ox]
            if np.any((weight > 1e-12) & ~np.isfinite(z)):
                raise ValueError('В рабочем участке есть NoData; импортируйте DEM без пробелов')
            corners.append(np.nan_to_num(z))
        z00,z10,z01,z11 = corners
        south=z00+ux*(z10-z00)
        north=z01+ux*(z11-z01)
        return south+uy*(north-south)

    def save(self, path):
        with open(path, 'wb') as f:
            np.savez_compressed(f, height=self.height, metadata=json.dumps(self.metadata, ensure_ascii=False))


def load_dem(path, *, crs=None, vertical_datum=None):
    """NPZ needs no GIS dependency. ASC requires explicit CRS/datum or sidecar.

    GeoTIFF is optional (pip install '.[dem]'); rotated/feet rasters are rejected.
    All importers convert north-first rows to south-first; no silent NoData fill.
    """
    path = Path(path)
    if path.suffix.lower() in ('.tif', '.tiff'):
        try:
            import rasterio
        except ImportError as error:
            raise ValueError('Для GeoTIFF установите: python -m pip install -e ".[dem]"; ASC/NPZ работают без rasterio') from error
        with rasterio.open(path) as src:
            a = src.transform
            if a.b != 0 or a.d != 0 or a.a <= 0 or a.e >= 0 or src.crs is None:
                raise ValueError('GeoTIFF: нужна привязанная сетка north-up; перепроецируйте повёрнутый растр')
            if src.width*src.height > 16_000_000:
                raise ValueError('DEM: сначала вырежьте область, предел 16 млн отсчётов')
            geographic = src.crs.to_epsg() == 4326
            if not geographic and (not src.crs.is_projected or src.crs.linear_units not in ('metre','meter')):
                raise ValueError('GeoTIFF: нужны WGS84 или проекция в метрах')
            scale, offset = src.scales[0], src.offsets[0]
            if src.units[0] not in (None, '', 'm', 'metre', 'meter'):
                raise ValueError('GeoTIFF: вертикальные единицы должны быть метрами')
            h = (src.read(1, masked=True).astype(float).filled(np.nan)*scale+offset)[::-1]
            m = dict(crs=src.crs.to_string(), units='degrees' if geographic else 'metres',
                     vertical_datum=vertical_datum, x0=a.c+a.a/2, y0=a.f+a.e*(src.height-.5),
                     dx=a.a, dy=-a.e, source=str(path.name))
            from rasterio.warp import transform
            _, lat = transform(src.crs,'EPSG:4326',[(src.bounds.left+src.bounds.right)/2],[(src.bounds.bottom+src.bounds.top)/2])
            m['latitude_deg'] = lat[0]
            return ElevationRaster(h, m)
    if path.suffix.lower() == '.asc':
        sidecar = path.with_suffix('.json')
        m = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
        with path.open(encoding='utf-8-sig') as f:
            header = {}
            # ESRI ASCII: six header lines, north row first, square cells.
            for _ in range(6):
                key, value = f.readline().split()
                header[key.lower()] = float(value)
            cols, rows = int(header['ncols']), int(header['nrows'])
            if cols < 2 or rows < 2 or cols*rows > 16_000_000:
                raise ValueError('DEM: сначала вырежьте область, предел 16 млн отсчётов')
            h = np.loadtxt(f, dtype=float, ndmin=2)
        if h.shape != (rows, cols):
            raise ValueError('ASC: размеры данных не совпадают с заголовком')
        h[h == header.get('nodata_value', -9999)] = np.nan
        cell = header['cellsize']
        x = header.get('xllcenter', header.get('xllcorner', float('nan'))+cell/2)
        y = header.get('yllcenter', header.get('yllcorner', float('nan'))+cell/2)
        crs = crs or m.get('crs')
        m.update(crs=crs, units='degrees' if crs == 'EPSG:4326' else 'metres',
                 vertical_datum=vertical_datum or m.get('vertical_datum'), x0=x,y0=y,dx=cell,dy=cell,source=path.name)
        if not crs:
            raise ValueError('ASC: укажите CRS (EPSG:4326 либо LOCAL_METRES)')
        return ElevationRaster(h[::-1], m)
    with np.load(path, allow_pickle=False) as data:
        return ElevationRaster(data['height'], json.loads(str(data['metadata'])))


def qattara_dem():
    return load_dem(Path(__file__).parent/'data'/'qattara_etopo1.npz')


class Landscape:
    """Fine tiles are allocated only when modified. One live local engine at a time.

    An unchanged region costs the coarse DEM only. Reading a window has no
    persistent allocation. All patch edits store FLOAT64 elevation deltas, so a
    cut in one patch and spoil in another are counted in the same volume ledger.
    """
    def __init__(self, raster, cell_size=.25, tile_cells=64):
        if not math.isfinite(cell_size) or cell_size <= 0 or not isinstance(tile_cells, int) or not 8 <= tile_cells <= 256:
            raise ValueError('Неверное разрешение или размер блока ландшафта')
        self.raster, self.cell_size, self.tile_cells = raster, cell_size, tile_cells
        self.tiles = {}
        self.corridors = []

    def _window(self, origin, n):
        if len(origin) != 2 or not all(math.isfinite(v) for v in origin) or not isinstance(n,int) or not 16 <= n <= 1024:
            raise ValueError('Неверный рабочий участок; максимум 1024×1024')
        index = np.rint(np.asarray(origin)/self.cell_size).astype(np.int64)
        if np.any(np.abs(index*self.cell_size-origin) > 1e-7):
            raise ValueError('Начало участка должно быть кратно мелкой ячейке')
        maximum = np.asarray(origin)+n*self.cell_size
        if np.any(index < 0) or np.any(maximum > self.raster.extent_m):
            raise ValueError('Участок выходит за границы DEM')
        axis = (np.arange(n)+.5)*self.cell_size
        x, y = np.meshgrid(axis+origin[0],axis+origin[1])
        return index, self.raster.sample(x,y)

    def _overlaps(self, index, n):
        size = self.tile_cells
        for ty in range(index[1]//size,(index[1]+n-1)//size+1):
            for tx in range(index[0]//size,(index[0]+n-1)//size+1):
                lo = np.maximum(index,(tx*size,ty*size))
                hi = np.minimum(index+n,((tx+1)*size,(ty+1)*size))
                local = (slice(lo[1]-index[1],hi[1]-index[1]),slice(lo[0]-index[0],hi[0]-index[0]))
                tile = (slice(lo[1]-ty*size,hi[1]-ty*size),slice(lo[0]-tx*size,hi[0]-tx*size))
                yield (tx,ty), local, tile

    def read_window(self, origin, n=256):
        index, baseline = self._window(origin,n)
        current = baseline.copy()
        for key, local, tile in self._overlaps(index,n):
            if key in self.tiles:
                current[local] += self.tiles[key][tile]
        return baseline, current

    def commit_window(self, origin, height):
        height = np.asarray(height,dtype=float)
        if height.ndim != 2 or height.shape[0] != height.shape[1] or not np.all(np.isfinite(height)):
            raise ValueError('Неверные высоты рабочего участка')
        n = height.shape[0]
        index, baseline = self._window(origin,n)
        delta = height-baseline
        for key, local, tile in self._overlaps(index,n):
            values = delta[local]
            if key not in self.tiles and not np.any(values != 0):
                continue
            block = self.tiles.get(key, np.zeros((self.tile_cells,self.tile_cells))).copy()
            block[tile] = values
            if np.any(block != 0):
                self.tiles[key] = block
            else:
                self.tiles.pop(key,None)

    def report(self):
        return dict(source=self.raster.metadata,vertical_datum='MSL', extent_m=list(self.raster.extent_m),
                    source_resolution_m=list(self.raster.resolution_m),fine_cell_size_m=self.cell_size,
                    modified_tiles=len(self.tiles),fine_storage_bytes=sum(t.nbytes for t in self.tiles.values()),
                    dem_storage_bytes=self.raster.height.nbytes,
                    soil_delta_m3=math.fsum(float(t.sum(dtype=np.float64))*self.cell_size**2 for t in self.tiles.values()))

    def save(self, path):
        metadata = dict(version=1,raster=self.raster.metadata,cell_size=self.cell_size,tile_cells=self.tile_cells,corridors=self.corridors)
        arrays = {f'tile_{x}_{y}':t for (x,y),t in self.tiles.items()}
        with open(path,'wb') as f:
            np.savez_compressed(f,height=self.raster.height,metadata=json.dumps(metadata,ensure_ascii=False),**arrays)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            m = json.loads(str(data['metadata']))
            if m.get('version') != 1:
                raise ValueError('Неизвестная версия ландшафта')
            result = cls(ElevationRaster(data['height'],m['raster']),m['cell_size'],m['tile_cells'])
            result.corridors = m.get('corridors',[])
            for name in data.files:
                if name.startswith('tile_'):
                    _,x,y = name.split('_')
                    key = int(x),int(y)
                    t = np.array(data[name],dtype=np.float64)
                    if min(key) < 0 or t.shape != (result.tile_cells,result.tile_cells) or not np.all(np.isfinite(t)):
                        raise ValueError('Повреждённый блок ландшафта')
                    result.tiles[key] = t
            return result


def workspace_config(landscape, config, origin, side_m=64.):
    """Start a local SI workspace without flattening or rebasing the DEM to zero."""
    from .plan import PlanConfig
    c = deepcopy(config)
    cell = landscape.cell_size
    n = round(side_m/cell)
    if not 32 <= side_m <= 256 or abs(n*cell-side_m) > 1e-8:
        raise ValueError('Рабочий участок: 32…256 м, кратно ячейке')
    baseline, current = landscape.read_window(origin,n)
    c.terrain.n, c.terrain.cell_size = n, cell
    c.terrain.roughness = 0.
    c.terrain.floor = min(-1000.,float(current.min())-100.,float(baseline.min())-100.)
    c.terrain.grab_radius = max(c.terrain.grab_radius,cell)
    c.terrain.dump_rect = (side_m*.6875, side_m*.59375, side_m*.28125, side_m*.375)
    c.trench.x = c.trench.y = side_m/2
    c.trench.length, c.trench.width, c.trench.depth = 2.,1.,.25
    c.base.x, c.base.y = min(10.,side_m*.15625),side_m/2
    c.plan = PlanConfig([])
    c.reference_height, c.initial_height = baseline.tolist(),current.tolist()
    source = landscape.raster
    anchor = [float(v) for v in source.to_source(*origin)]
    c.georeference = dict(vertical_datum='MSL',source=source.metadata,origin_m=list(origin),
                          source_origin=anchor,source_resolution_m=list(source.resolution_m),
                          projection='local equirectangular' if source.metadata['crs']=='EPSG:4326' else source.metadata['crs'])
    if source.metadata['crs']=='EPSG:4326':
        c.base.latitude_deg = anchor[1]
    elif 'latitude_deg' in source.metadata:
        c.base.latitude_deg = source.metadata['latitude_deg']
    c.validate()
    return c


def corridor_profile(raster, points, target_msl=-5., width_m=20., spacing_m=500., angle_deg=35.):
    """Coarse route screening, suitable as the cost API for a future optimizer.

    Integrates cross sections from sampled centre heights and fixed side slopes. This never
    digs or allocates fine tiles, and is not a geotechnical quantity survey.
    """
    from .plan import sample_line
    if not math.isfinite(target_msl) or not math.isfinite(width_m) or width_m <= 0 or not math.isfinite(spacing_m) or spacing_m <= 0 or not 5 <= angle_deg <= 55:
        raise ValueError('Неверные параметры коридора')
    p = np.asarray(points,dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.all(np.isfinite(p)):
        raise ValueError('Коридору нужна линия как минимум из двух точек в метрах')
    if np.linalg.norm(np.diff(p,axis=0),axis=1).sum()/spacing_m > 20000:
        raise ValueError('Слишком подробный региональный профиль; увеличьте шаг')
    line, axis = sample_line(p,spacing_m)
    if axis[-1] <= 0:
        raise ValueError('Длина коридора должна быть положительной')
    ground = raster.sample(line[:,0],line[:,1])
    depth = np.maximum(ground-target_msl,0.)
    area = width_m*depth+depth**2/math.tan(math.radians(angle_deg))
    volume = float(np.sum((area[:-1]+area[1:])/2*np.diff(axis)))
    return dict(distance_m=axis.tolist(),ground_msl=ground.tolist(),target_msl=target_msl,
                estimated_cut_m3=volume,length_m=float(axis[-1]),width_m=width_m,
                source_resolution_m=list(raster.resolution_m),method='coarse constant-cross-slope screening')
