"""Offline coarse DEM with sparse, conservative fine-grid work patches.

Coordinates in the simulator are always metres; elevations are metres above MSL.
No hydrology and no automatic regional excavation/transport are implied.
"""
from dataclasses import dataclass
from copy import copy,deepcopy
from contextlib import nullcontext
from pathlib import Path
import json
import math
import os
import tempfile
import zipfile
import numpy as np
from .tile_cache import TileCache,TileMap,OverlayTiles,DEFAULT_TILE_RAM

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
    def __init__(self, raster, cell_size=.25, tile_cells=64,soil_profile=None,*,cache_bytes=DEFAULT_TILE_RAM):
        if not math.isfinite(cell_size) or cell_size <= 0 or not isinstance(tile_cells, int) or not 8 <= tile_cells <= 256:
            raise ValueError('Неверное разрешение или размер блока ландшафта')
        self.raster, self.cell_size, self.tile_cells = raster, cell_size, tile_cells
        self._tile_cache=TileCache(cache_bytes)
        self.tiles = TileMap(self._tile_cache,'height')
        self.corridors = []
        self.channel_project = None
        self.soil_profile=soil_profile
        self.soil_tiles=TileMap(self._tile_cache,'soil')
        self.prefetch_enabled=True;self.prefetch_request=None;self.prefetch_cancelled=None
        self._prefetch_runtime=None
        self._prefetch_auto_key=None
        if soil_profile is not None:soil_profile.validate()

    def _grid(self, origin, n, cell_size=None):
        if len(origin) != 2 or not all(math.isfinite(v) for v in origin) or not isinstance(n,int) or not 16 <= n <= 1024:
            raise ValueError('Неверный рабочий участок; максимум 1024×1024')
        cell=self.cell_size if cell_size is None else cell_size
        factor=round(cell/self.cell_size)
        if factor not in (1,2,4) or abs(factor*self.cell_size-cell)>1e-9:
            raise ValueError('Рабочая ячейка должна совпадать с сеткой хранения или быть в 2/4 раза крупнее')
        index = np.rint(np.asarray(origin)/self.cell_size).astype(np.int64)
        if np.any(np.abs(index*self.cell_size-origin) > 1e-7):
            raise ValueError('Начало участка должно быть кратно мелкой ячейке')
        maximum = np.asarray(origin)+n*cell
        if np.any(index < 0) or np.any(maximum > self.raster.extent_m):
            raise ValueError('Участок выходит за границы DEM')
        return index,factor

    def _window(self, origin, n, cell_size=None,check=None):
        index,factor=self._grid(origin,n,cell_size)
        cell=self.cell_size if cell_size is None else cell_size
        axis = (np.arange(n)+.5)*cell
        baseline=np.empty((n,n),dtype=np.float64)
        # Bound interpolation scratch arrays and offer cancellation between
        # strips. Coordinates/operations per sample remain bitwise identical.
        for row in range(0,n,64):
            if check:check(.25*row/n)
            baseline[row:row+64]=self.raster.sample((axis+origin[0])[None,:],(axis[row:row+64]+origin[1])[:,None])
        return index,baseline,factor

    def _overlaps(self, index, n):
        size = self.tile_cells
        for ty in range(index[1]//size,(index[1]+n-1)//size+1):
            for tx in range(index[0]//size,(index[0]+n-1)//size+1):
                lo = np.maximum(index,(tx*size,ty*size))
                hi = np.minimum(index+n,((tx+1)*size,(ty+1)*size))
                local = (slice(lo[1]-index[1],hi[1]-index[1]),slice(lo[0]-index[0],hi[0]-index[0]))
                tile = (slice(lo[1]-ty*size,hi[1]-ty*size),slice(lo[0]-tx*size,hi[0]-tx*size))
                yield (tx,ty), local, tile

    def _tokens(self,mapping,index,n):
        return {key:mapping.token(key) for key,_,_ in self._overlaps(index,n)}

    def read_window(self, origin, n=256, *, cell_size=None,prepared=None,check=None):
        cell=self.cell_size if cell_size is None else cell_size
        index,factor=self._grid(origin,n,cell)
        usable=(prepared is not None and prepared.key==(*map(float,origin),n,cell)
                and prepared.fine_cell==self.cell_size and prepared.tile_cells==self.tile_cells
                and prepared.raster is self.raster)
        if usable:
            baseline=prepared.baseline.copy()
            if self._tokens(self.tiles,index,n*factor)==prepared.height_tokens:
                return baseline,prepared.current.copy()
        else:_,baseline,_=self._window(origin,n,cell,check)
        if check:check(.25)
        current = baseline.copy()
        changes=np.zeros_like(current) if factor>1 else None
        for key, local, tile in self._overlaps(index,n*factor):
            if check:check()
            if key in self.tiles:
                if factor==1:current[local] += self.tiles[key][tile]
                else:
                    rows=np.arange(local[0].start,local[0].stop)//factor
                    cols=np.arange(local[1].start,local[1].stop)//factor
                    np.add.at(changes,(rows[:,None],cols[None,:]),self.tiles[key][tile]/factor**2)
        if factor>1:current+=changes
        return baseline, current

    def commit_window(self, origin, height, *, cell_size=None,soil=None):
        cache=getattr(self,'_tile_cache',None)
        with cache.transaction() if cache is not None else nullcontext():
            self._commit_window(origin,height,cell_size=cell_size,soil=soil)

    def _commit_window(self, origin, height, *, cell_size=None,soil=None):
        height = np.asarray(height,dtype=float)
        if height.ndim != 2 or height.shape[0] != height.shape[1] or not np.all(np.isfinite(height)):
            raise ValueError('Неверные высоты рабочего участка')
        n = height.shape[0]
        index, baseline,factor = self._window(origin,n,cell_size)
        if self.soil_profile is not None:
            if (soil is None or soil.profile!=self.soil_profile or soil.factor!=factor or
                    soil.fine_cell!=self.cell_size or soil.tile_cells!=self.tile_cells or
                    not np.array_equal(soil.origin,index) or soil.n!=n):
                raise ValueError('Для сохранения слоёв нужен состав текущего рабочего участка')
            soil.validate()
        elif soil is not None:raise ValueError('Профиль участка не совпадает с ландшафтом')
        delta = height-baseline
        _,previous=self.read_window(origin,n,cell_size=cell_size)
        increments=height-previous
        for key, local, tile in self._overlaps(index,n*factor):
            if factor==1:values = delta[local]
            else:
                rows=np.arange(local[0].start,local[0].stop)//factor
                cols=np.arange(local[1].start,local[1].stop)//factor
                values=increments[rows[:,None],cols[None,:]]
            if key not in self.tiles and not np.any(values != 0):
                continue
            old=self.tiles.get(key)
            block = np.zeros((self.tile_cells,self.tile_cells)) if old is None else old.copy()
            if factor==1:block[tile] = values
            else:block[tile] += values
            if np.any(block != 0):
                if old is None or not np.array_equal(old,block):self.tiles[key] = block
            else:
                self.tiles.pop(key,None)
        if soil is not None:
            for key,local,tile in self._overlaps(index,n*factor):
                if key not in soil.tiles:continue
                old=self.soil_tiles.get(key)
                block=np.zeros_like(soil.tiles[key]) if old is None else old.copy()
                block[:,tile[0],tile[1]]=soil.tiles[key][:,tile[0],tile[1]]
                if old is None or not np.array_equal(old,block):self.soil_tiles[key]=block

    def soil_window(self,origin,n,cell_size,*,prepared=None,check=None):
        if self.soil_profile is None:return None
        index,factor=self._grid(origin,n,cell_size)
        if (prepared is not None and prepared.key==(*map(float,origin),n,cell_size)
                and prepared.soil_profile==self.soil_profile and prepared.fine_cell==self.cell_size
                and prepared.tile_cells==self.tile_cells
                and prepared.raster is self.raster
                and self._tokens(self.soil_tiles,index,n*factor)==prepared.soil_tokens):
            return {**prepared.soil,'tiles':[[x,y,a.copy()] for x,y,a in prepared.soil['tiles']]}
        keys=[key for key,_,_ in self._overlaps(index,n*factor) if key in self.soil_tiles]
        tiles=[]
        for x,y in keys:
            if check:check()
            tiles.append([x,y,self.soil_tiles[x,y].copy()])
        return dict(fine_cell=self.cell_size,tile_cells=self.tile_cells,origin_m=list(origin),
                    tiles=tiles)

    def snapshot(self):
        """Freeze version references; previews allocate only their edited blocks."""
        result=copy(self)
        result.tiles=OverlayTiles(self._tile_cache.snapshot('height'))
        result.soil_tiles=OverlayTiles(self._tile_cache.snapshot('soil'))
        result._tile_cache=None;result._prefetch_runtime=None;result.prefetch_request=None
        result.soil_profile=deepcopy(self.soil_profile);result.channel_project=deepcopy(self.channel_project)
        return result

    def prepare_window(self,origin,n,cell_size,*,budget,check):
        from .window_prefetch import PreparedWindow
        index,factor=self._grid(origin,n,cell_size)
        height_tokens=self._tokens(self.tiles,index,n*factor)
        soil_tokens=self._tokens(self.soil_tiles,index,n*factor)
        estimated=2*n*n*8+sum(v.nbytes for v in soil_tokens.values() if v is not None)
        if estimated>budget:raise ValueError('Окно превышает лимит памяти предзагрузки; используйте обычный переезд или меньшее окно')
        baseline,current=self.read_window(origin,n,cell_size=cell_size,check=check)
        check(.65);soil=self.soil_window(origin,n,cell_size,check=check)
        baseline.flags.writeable=False;current.flags.writeable=False
        for _,_,a in (soil or {}).get('tiles',[]):a.flags.writeable=False
        return PreparedWindow(tuple(map(float,origin)),n,cell_size,baseline,current,soil,height_tokens,soil_tokens,
            self.cell_size,self.tile_cells,self.soil_profile,self.raster)

    def prefetch_window(self,origin,n=256,*,cell_size=None,force=True,automatic=False):
        from .window_prefetch import WindowPrefetch,PREFETCH_BYTES
        cell=self.cell_size if cell_size is None else cell_size;self._grid(origin,n,cell)
        key=(*map(float,origin),n,cell)
        if not force and (not self.prefetch_enabled or self.prefetch_cancelled==list(key)):return False
        if self.prefetch_request and tuple(self.prefetch_request['key'])==key and self._prefetch_runtime is not None and not force:return False
        snapshot=self.snapshot()
        if self._prefetch_runtime is None:self._prefetch_runtime=WindowPrefetch()
        self.prefetch_request=dict(key=list(key),automatic=automatic)
        self.prefetch_cancelled=None
        self._prefetch_runtime.request(snapshot,key,PREFETCH_BYTES)
        return True

    def resume_prefetch(self):
        if self.prefetch_request is not None and self._prefetch_runtime is None:
            key=self.prefetch_request['key'];automatic=self.prefetch_request.get('automatic',False)
            self.prefetch_window(key[:2],key[2],cell_size=key[3],automatic=automatic)

    def cancel_prefetch(self):
        if self.prefetch_request:self.prefetch_cancelled=self.prefetch_request['key'].copy()
        self.prefetch_request=None
        if self._prefetch_runtime is not None:self._prefetch_runtime.cancel()

    def clear_prefetch(self):
        self.cancel_prefetch();self._prefetch_runtime=None;self.prefetch_cancelled=None

    def prepared_window(self,origin,n,cell):
        self.resume_prefetch()
        return self._prefetch_runtime.prepared((*map(float,origin),n,cell)) if self._prefetch_runtime else None

    def cache_report(self):
        report=self._tile_cache.report()
        report['prefetch']=(self._prefetch_runtime.report() if self._prefetch_runtime else
            dict(state='QUEUED' if self.prefetch_request else 'CANCELLED' if self.prefetch_cancelled else 'IDLE',
                key=self.prefetch_request['key'] if self.prefetch_request else None,error='',progress=0.,read_seconds=0.,prepared_bytes=0))
        return report

    def ensure_next_prefetch(self,simulation):
        self.resume_prefetch()
        if not self.prefetch_enabled or self.prefetch_request is not None:return
        geo=simulation.config.georeference or {};info=geo.get('channel',{})
        if not self.channel_project or not info:return
        key=(id(geo),id(self.channel_project),tuple(geo['origin_m']),info['section'],
             simulation.config.terrain.n,simulation.config.terrain.cell_size)
        if getattr(self,'_prefetch_auto_key',None)==key:return
        self._prefetch_auto_key=key
        from .channel import ChannelProject
        from .regional import window_spec
        channel=ChannelProject.from_data(self.channel_project);index=info['section']+1
        if (index>=channel.section_count or channel.side_m!=simulation.config.terrain.n*simulation.config.terrain.cell_size
                or info.get('project_id')!=channel.identity):return
        origin,n,cell=window_spec(self,simulation.config,None,index)
        if origin==geo['origin_m']:return
        self.prefetch_window(origin,n,cell_size=cell,force=False,automatic=True)

    def restore_cache(self):
        """Migrate a legacy in-memory region without changing any array values."""
        if isinstance(self.tiles,TileMap):return
        cache=TileCache();height,soil=self.tiles,self.soil_tiles
        self._tile_cache=cache;self.tiles=TileMap(cache,'height');self.soil_tiles=TileMap(cache,'soil')
        with cache.transaction():
            for k,a in height.items():self.tiles[k]=a
            for k,a in soil.items():self.soil_tiles[k]=a
        self.prefetch_enabled=True;self.prefetch_request=None;self.prefetch_cancelled=None;self._prefetch_runtime=None;self._prefetch_auto_key=None

    def report(self):
        height_bytes,sums=self.tiles.totals()
        report=dict(source=self.raster.metadata,vertical_datum='MSL', extent_m=list(self.raster.extent_m),
                    source_resolution_m=list(self.raster.resolution_m),fine_cell_size_m=self.cell_size,
                    modified_tiles=len(self.tiles),fine_storage_bytes=height_bytes,
                    dem_storage_bytes=self.raster.height.nbytes,
                    soil_delta_m3=math.fsum(total*self.cell_size**2 for total in sums))
        if self.channel_project:
            from .channel import ChannelProject
            p=ChannelProject.from_data(self.channel_project)
            report['channel']=dict(project_id=p.identity,length_m=p.length,width_m=p.width_m,depth_m=p.depth_m,
                target_msl=-p.depth_m,shore_offset_m=p.shore_offset_m,sections=p.section_count,
                active_section=p.active_section,completed=list(p.completed))
            from .sector_progress import merge
            spans=merge(p.finished_spans)
            report['channel'].update(finished_spans=spans,finished_profile_m=sum(b-a for a,b in spans),
                project_profile_complete=bool(spans and spans[0][0]<=1e-8 and spans[0][1]>=p.length-1e-8))
        if self.soil_profile is not None:
            report.update(soil_profile=self.soil_profile.name,soil_modified_tiles=len(self.soil_tiles),
                          soil_storage_bytes=self.soil_tiles.totals()[0])
        return report

    def save(self, path):
        from dataclasses import asdict
        metadata = dict(version=2,raster=self.raster.metadata,cell_size=self.cell_size,tile_cells=self.tile_cells,corridors=self.corridors,channel_project=self.channel_project,
                        soil_profile=asdict(self.soil_profile) if self.soil_profile is not None else None,
                        tile_cache_limit_bytes=self._tile_cache.limit_bytes,prefetch_enabled=self.prefetch_enabled)
        path=Path(path);temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as f:
                temporary=Path(f.name)
                with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
                    def write(name,array):
                        with archive.open(name+'.npy','w',force_zip64=True) as stream:
                            np.lib.format.write_array(stream,np.asarray(array),allow_pickle=False)
                    write('height',self.raster.height);write('metadata',json.dumps(metadata,ensure_ascii=False))
                    for prefix,mapping in (('tile',self.tiles),('soil',self.soil_tiles)):
                        for (x,y),array in mapping.items():write(f'{prefix}_{x}_{y}',array)
                f.flush();os.fsync(f.fileno())
            os.replace(temporary,path);temporary=None
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            m = json.loads(str(data['metadata']))
            if m.get('version') not in (1,2):
                raise ValueError('Неизвестная версия ландшафта')
            result = cls(ElevationRaster(data['height'],m['raster']),m['cell_size'],m['tile_cells'],cache_bytes=m.get('tile_cache_limit_bytes',DEFAULT_TILE_RAM))
            result.prefetch_enabled=m.get('prefetch_enabled',True)
            if type(result.prefetch_enabled) is not bool:raise ValueError('Неверная настройка предзагрузки')
            result.corridors = m.get('corridors',[])
            result.channel_project = m.get('channel_project')
            if m.get('soil_profile') is not None:
                from .soil import SoilProfile
                result.soil_profile=SoilProfile.from_data(m['soil_profile'])
            for name in data.files:
                if name.startswith('tile_'):
                    _,x,y = name.split('_')
                    key = int(x),int(y)
                    t = np.array(data[name],dtype=np.float64)
                    if min(key) < 0 or t.shape != (result.tile_cells,result.tile_cells) or not np.all(np.isfinite(t)):
                        raise ValueError('Повреждённый блок ландшафта')
                    result.tiles[key] = t
                elif name.startswith('soil_'):
                    _,x,y=name.split('_');key=int(x),int(y);t=np.array(data[name],dtype=np.float64)
                    if (result.soil_profile is None or min(key)<0 or
                            t.shape!=(len(result.soil_profile.materials)+1,result.tile_cells,result.tile_cells) or
                            not np.all(np.isfinite(t)) or np.any(t< -1e-9)):
                        raise ValueError('Повреждён состав грунта ландшафта')
                    result.soil_tiles[key]=t
            return result


def workspace_config(landscape, config, origin, side_m=64.,*,prepared=None):
    """Start a local SI workspace without flattening or rebasing the DEM to zero."""
    from .plan import PlanConfig
    from .maps import workspace_cell
    # These matrices and the old plan/frame are replaced below. Copying them
    # first doubles a large window's transient RAM for no physical purpose.
    c=copy(config)
    c.initial_height=c.reference_height=c.initial_soil=c.plan=c.georeference=None
    c=deepcopy(c)
    cell = workspace_cell(side_m,landscape.cell_size)
    n = round(side_m/cell)
    if not 32 <= side_m <= 1024 or abs(n*cell-side_m) > 1e-8:
        raise ValueError('Рабочий участок: 32…1024 м, кратно ячейке')
    baseline, current = landscape.read_window(origin,n,cell_size=cell,prepared=prepared)
    c.terrain.n, c.terrain.cell_size = n, cell
    c.terrain.roughness = 0.
    c.terrain.floor = min(-1000.,float(current.min())-100.,float(baseline.min())-100.)
    c.terrain.grab_radius = max(c.terrain.grab_radius,cell)
    c.terrain.dump_rect = (side_m*.6875, side_m*.59375, side_m*.28125, side_m*.375)
    c.trench.x = c.trench.y = side_m/2
    c.trench.length, c.trench.width, c.trench.depth = max(2.,cell),max(1.,cell),.25
    c.base.x, c.base.y = min(10.,side_m*.15625),side_m/2
    c.plan = PlanConfig([])
    c.reference_height, c.initial_height = baseline.tolist(),current.tolist()
    c.soil_profile=deepcopy(landscape.soil_profile)
    c.initial_soil=landscape.soil_window(origin,n,cell,prepared=prepared)
    source = landscape.raster
    anchor = [float(v) for v in source.to_source(*origin)]
    c.georeference = dict(vertical_datum='MSL',source=source.metadata,origin_m=list(origin),
                          source_origin=anchor,source_resolution_m=list(source.resolution_m),
                          projection='local equirectangular' if source.metadata['crs']=='EPSG:4326' else source.metadata['crs'])
    from .channel import marine_window
    if source.metadata['crs']!='LOCAL_METRES' or source.metadata.get('ocean_boundary',False):
        c.georeference['marine']=marine_window(source,origin,side_m)
    if source.metadata['crs']=='EPSG:4326':
        c.base.latitude_deg = float(source.to_source(origin[0]+c.base.x,origin[1]+c.base.y)[1])
    elif 'latitude_deg' in source.metadata:
        c.base.latitude_deg = source.metadata['latitude_deg']
    c.validate()
    return c


def corridor_profile(raster, points, target_msl=-5., width_m=20., spacing_m=500., angle_deg=35.,soil_profile=None):
    """Coarse route screening, suitable as the cost API for a future optimizer.

    Integrates cross sections from sampled centre heights and fixed side slopes. This never
    digs or allocates fine tiles, and is not a geotechnical quantity survey.
    """
    from .plan import sample_line
    if not math.isfinite(target_msl) or not math.isfinite(width_m) or width_m <= 0 or not math.isfinite(spacing_m) or spacing_m <= 0 or (angle_deg is not None and not 5 <= angle_deg <= 75):
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
    if angle_deg is None and soil_profile:
        bounds=np.r_[0.,soil_profile.boundaries,np.inf];area=width_m*depth
        for lo,hi,m in zip(bounds[:-1],bounds[1:],soil_profile.materials):
            area+=(np.minimum(depth,hi)**2-np.minimum(depth,lo)**2)/math.tan(math.radians(m.native_angle_deg))
    else:area = width_m*depth+depth**2/math.tan(math.radians(angle_deg or 35.))
    volume = float(np.sum((area[:-1]+area[1:])/2*np.diff(axis)))
    return dict(distance_m=axis.tolist(),ground_msl=ground.tolist(),target_msl=target_msl,
                estimated_cut_m3=volume,length_m=float(axis[-1]),width_m=width_m,
                source_resolution_m=list(raster.resolution_m),method='coarse constant-cross-slope screening')
