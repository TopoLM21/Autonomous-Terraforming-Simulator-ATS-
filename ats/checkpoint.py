"""Versioned, atomic snapshots: JSON object references and non-pickle NumPy data.

The explicit class registry preserves live job/inventory aliases and cycles.
Loading never imports names or executes constructors supplied by the file.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
import json
import math
import os
import platform
from pathlib import Path
import tempfile
from types import SimpleNamespace
import zipfile
import numpy as np


FORMAT='ATS_SIMULATION'
VERSION=11
MAX_BYTES=1_500_000_000
MAX_NODES=2_000_000


def _classes():
    from . import (config,equipment,simulation,solar_base,drone,terrain,planner,dust,
                   logistics,logistics_config,inventory,batteries,excavator,cooperative_lift,
                   productivity,activity,plan,laser,workfront,landscape,soil,complexes,power_design,shared_site,regional,solar_array,battery_platforms,self_solar)
    modules=(config,equipment,simulation,solar_base,drone,terrain,planner,dust,logistics,
             logistics_config,inventory,batteries,excavator,cooperative_lift,productivity,
             activity,plan,laser,workfront,landscape,soil,complexes,power_design,shared_site,regional,solar_array,battery_platforms,self_solar)
    classes=[value for module in modules for value in vars(module).values()
             if isinstance(value,type) and value.__module__==module.__name__]
    return {c.__module__+'.'+c.__name__:c for c in [*classes,SimpleNamespace]}


@dataclass
class Session:
    simulation: object
    landscape: object = None
    ui: dict = None
    runtime: dict = None
    complex_project: object = None


class _Encoder:
    def __init__(self, archive):
        self.archive=archive;self.nodes=[];self.seen={};self.classes=_classes()

    def encode(self, value, matrix=False):
        if isinstance(value,np.generic):
            if value.dtype.kind not in 'biuf' or value.dtype.itemsize>8:raise ValueError('Неподдерживаемый скаляр')
            return {'scalar':value.dtype.str,'value':self.encode(value.item())}
        if isinstance(value,Enum):
            return {'enum':type(value).__module__+'.'+type(value).__name__,'value':value.value}
        if value is None or type(value) in (str,bool,int):return value
        if type(value) is float:
            return value if math.isfinite(value) else {'number':str(value)}
        if isinstance(value,slice):return {'slice':[value.start,value.stop,value.step]}
        identity=id(value)
        if identity in self.seen:return {'ref':self.seen[identity]}
        index=len(self.nodes);self.seen[identity]=index;node={};self.nodes.append(node)
        if len(self.nodes)>MAX_NODES:raise ValueError('Слишком много объектов для одного сохранения')
        if matrix and isinstance(value,list):
            value=np.asarray(value,dtype=np.float64);node['kind']='matrix_list'
        if isinstance(value,np.ndarray):
            if value.dtype.kind not in 'biuf' or value.dtype.itemsize>8:raise ValueError('Неподдерживаемый тип массива')
            name=f'arrays/{index}.npy';buffer=BytesIO();np.save(buffer,value,allow_pickle=False)
            self.archive.writestr(name,buffer.getvalue())
            node.update(kind=node.get('kind','array'),file=name,writeable=bool(value.flags.writeable))
        elif isinstance(value,dict):node.update(kind='dict',items=[[self.encode(k),self.encode(v)] for k,v in value.items()])
        elif isinstance(value,(list,tuple,set,frozenset,deque)):
            node.update(kind=type(value).__name__,items=[self.encode(v) for v in value])
            if isinstance(value,deque):node['maxlen']=value.maxlen
        else:
            name=type(value).__module__+'.'+type(value).__name__
            if name not in self.classes:raise ValueError(f'Нельзя сохранить объект {name}')
            node.update(kind='object',type=name,attrs={k:self.encode(v,matrix=name=='ats.config.Config' and
                k in ('initial_height','reference_height') and v is not None) for k,v in vars(value).items()})
        return {'ref':index}


class _Decoder:
    def __init__(self, archive, nodes):
        if not isinstance(nodes,list) or len(nodes)>MAX_NODES:raise ValueError('Неверный список объектов')
        self.archive=archive;self.nodes=nodes;self.classes=_classes();self.objects=[];self.building=set()
        for node in nodes:
            kind=node['kind']
            if kind in ('array','matrix_list'):
                name=node['file']
                if not name.startswith('arrays/') or not name.endswith('.npy'):raise ValueError('Неверное имя массива')
                payload=BytesIO(archive.read(name));version=np.lib.format.read_magic(payload)
                if version not in ((1,0),(2,0)):raise ValueError('Неподдерживаемый формат массива')
                reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                shape,_,dtype=reader(payload)
                if (dtype.kind not in 'biuf' or dtype.itemsize>8 or len(shape)>4 or
                        math.prod(shape)*dtype.itemsize!=payload.getbuffer().nbytes-payload.tell()):
                    raise ValueError('Неверный размер или тип массива')
                payload.seek(0);value=np.load(payload,allow_pickle=False)
                if not isinstance(value,np.ndarray) or value.dtype.kind not in 'biuf' or value.dtype.itemsize>8:
                    raise ValueError('Неподдерживаемый массив')
                if kind=='matrix_list':value=value.tolist()
                else:value.flags.writeable=node.get('writeable',True)
            elif kind=='object':
                cls=self.classes.get(node['type'])
                if cls is None or issubclass(cls,Enum):raise ValueError('Неизвестный тип объекта')
                value=cls.__new__(cls)
            elif kind=='dict':value={}
            elif kind=='list':value=[]
            elif kind=='set':value=set()
            elif kind=='deque':value=deque(maxlen=node['maxlen'])
            elif kind in ('tuple','frozenset'):value=None
            else:raise ValueError('Неизвестный вид объекта')
            self.objects.append(value)
        for i,node in enumerate(nodes):
            kind=node['kind'];value=self.objects[i]
            if kind=='object':value.__dict__.update({k:self.decode(v) for k,v in node['attrs'].items()})
            elif kind=='dict':value.update((self.decode(k),self.decode(v)) for k,v in node['items'])
            elif kind in ('list','deque'):value.extend(self.decode(v) for v in node['items'])
            elif kind=='set':value.update(self.decode(v) for v in node['items'])
            elif kind in ('tuple','frozenset'):self.reference(i)

    def reference(self, index):
        if type(index) is not int or not 0<=index<len(self.nodes):raise ValueError('Повреждённая ссылка объекта')
        if self.objects[index] is None:
            if index in self.building:raise ValueError('Недопустимый цикл неизменяемых объектов')
            self.building.add(index);node=self.nodes[index]
            values=[self.decode(v) for v in node['items']]
            self.objects[index]=tuple(values) if node['kind']=='tuple' else frozenset(values)
            self.building.remove(index)
        return self.objects[index]

    def decode(self, value):
        if not isinstance(value,dict):return value
        if 'ref' in value:return self.reference(value['ref'])
        if 'enum' in value:
            cls=self.classes.get(value['enum'])
            if cls is None or not issubclass(cls,Enum):raise ValueError('Неизвестное состояние оборудования')
            return cls(value['value'])
        if 'number' in value:
            if value['number'] not in ('inf','-inf','nan'):raise ValueError('Неверное специальное число')
            return float(value['number'])
        if 'scalar' in value:
            dtype=np.dtype(value['scalar'])
            if dtype.kind not in 'biuf' or dtype.itemsize>8:raise ValueError('Неверный тип скаляра')
            return dtype.type(self.decode(value['value']))
        if 'slice' in value:
            if len(value['slice'])!=3 or any(v is not None and type(v) is not int for v in value['slice']):
                raise ValueError('Неверный диапазон массива')
            return slice(*value['slice'])
        raise ValueError('Неверное значение сохранения')


def _migrate_v1(decoder):
    """V1 never contained stratigraphy. Keep its density, cargo and jobs intact."""
    from .config import Config
    from .simulation import Simulation
    from .terrain import Terrain
    from .drone import Drone
    from .inventory import SoilContainer
    from .excavator import Excavator
    from .planner import Trip
    from .landscape import Landscape
    for value in decoder.objects:
        if isinstance(value,Config):value.soil_profile=None;value.initial_soil=None
        elif isinstance(value,Terrain):value.soil=None
        elif isinstance(value,Simulation):
            value.excavated_volume=value.excavated_mass/value.config.terrain.density
            value.filled_volume=value.filled_mass/value.config.terrain.density
        elif isinstance(value,(Drone,SoilContainer)):value.soil=None
        elif isinstance(value,Excavator):value.soil=None;value.cycle_factor=1.
        elif isinstance(value,Trip):value.soil_factor=1.;value.volume=None
        elif isinstance(value,Landscape):value.soil_profile=None;value.soil_tiles={}


def _validate(session):
    from .simulation import Simulation,MissionState
    from .drone import DroneState
    from .landscape import Landscape
    s=session.simulation
    if type(s) is not Simulation or not isinstance(s.state,MissionState):raise ValueError('Нет состояния симуляции')
    s.config.validate();t=s.terrain
    if t.front is not None and t.front.sectors:
        f=t.front
        if (type(f.stationed) is not bool or type(f.moving) is not bool or
                not all(math.isfinite(v) for v in (f.sector_start,f.sector_end,f.sector_centre,f.sector_tolerance)) or
                f.sector_start<0 or f.sector_end<=f.sector_start or
                not f.sector_start<=f.sector_centre<=f.sector_end or f.sector_tolerance<=0):
            raise ValueError('Повреждено состояние рабочего сектора')
    shape=(s.config.terrain.n,)*2
    for key in ('height','reference','target','_initial_surface','dump_cap'):
        array=getattr(t,key)
        if not isinstance(array,np.ndarray) or array.shape!=shape or not np.all(np.isfinite(array)):
            raise ValueError('Повреждён рельеф')
    if s.config.soil_profile is not None:
        from .soil import SoilField
        if not isinstance(t.soil,SoilField) or t.soil.profile!=s.config.soil_profile:
            raise ValueError('Потерян профиль грунта')
        t.soil.validate()
        if t.soil.n!=shape[0] or t.soil.cell!=t.cell:raise ValueError('Сетка грунта не совпадает с рельефом')
        # Audit independent height and material representations on bounded strips.
        for y in range(0,shape[0],16):
            yy,xx=np.mgrid[y:min(shape[0],y+16),0:shape[0]];ids=(yy*shape[0]+xx).ravel()
            delta=t.soil.height_delta(ids).reshape(yy.shape)
            if not np.allclose(delta,(t.height-t.reference)[y:y+16],rtol=0,atol=1e-7):
                raise ValueError('Высоты не совпадают с составом грунта')
    elif t.soil is not None:raise ValueError('Лишний состав грунта без профиля')
    if not math.isfinite(s.time) or s.time<0:raise ValueError('Неверное модельное время')
    if len(s.drones)!=s.config.fleet.count or [d.identifier for d in s.drones]!=list(range(len(s.drones))):
        raise ValueError('Повреждён состав флота')
    if len({d.serial for d in s.drones})!=len(s.drones):raise ValueError('Повторяющийся экземпляр дрона')
    if t.config is not s.config or s.base.config is not s.config.base or s.planner.config is not s.config:
        raise ValueError('Нарушены ссылки на конфигурацию')
    if not -1e-7<=s.base.battery<=s.base.config.battery+1e-7:raise ValueError('Неверный заряд базы')
    if s.base.position.shape!=(3,) or not np.all(np.isfinite(s.base.position)):raise ValueError('Неверная позиция базы')
    if s.dust.field.shape!=(s.dust.n,s.dust.n) or not np.all(np.isfinite(s.dust.field)):raise ValueError('Повреждено поле пыли')
    for d in s.drones:
        if not isinstance(d.state,DroneState) or not -1e-7<=d.battery<=d.config.battery+1e-7 or d.cargo<0:
            raise ValueError('Неверное состояние дрона')
        if d.position.shape!=(3,) or not np.all(np.isfinite(d.position)) or d.velocity.shape!=(3,) or not np.all(np.isfinite(d.velocity)):
            raise ValueError('Неверная позиция или скорость дрона')
        if d.soil is not None:
            d.soil.validate(d.cargo if d.soil.mass>1e-8 else None)
            if d.soil.profile!=s.config.soil_profile:raise ValueError('Неверный материал груза дрона')
        if d.slot is not None and (not 0<=d.slot<len(s.slots) or s.slots[d.slot]!=d.identifier):
            raise ValueError('Нарушена бронь посадочного места')
    occupants=[value for value in s.slots if value is not None]
    if len(occupants)!=len(set(occupants)) or any(not 0<=i<len(s.drones) for i in occupants):
        raise ValueError('Повреждены посадочные места')
    if len(s.queue)!=len(set(s.queue)) or any(not 0<=i<len(s.drones) for i in s.queue):
        raise ValueError('Повреждена очередь флота')
    if s.logistics is not None:
        w=s.logistics
        if w.sim is not s or w.battery_service.world is not w or w.active.keys()&w.direct_ids:
            raise ValueError('Нарушены ссылки диспетчера')
        w.inventory.validate()
        if w.battery_platforms:
            w.battery_platforms.validate();w.battery_platforms.audit_site()
        w.audit_solar_site()
        for container in w.containers:
            if container.soil is not None and container.soil.profile!=s.config.soil_profile:
                raise ValueError('Неверный материал контейнера')
        if w.excavator is not None and w.excavator.soil is not None:
            w.excavator.soil.validate(w.excavator.bucket_kg)
            if w.excavator.soil.profile!=s.config.soil_profile:raise ValueError('Неверный материал ковша')
        jobs={id(job) for job in w.jobs}
        if any(id(job) not in jobs for job in w.active.values()):raise ValueError('Потеряно активное задание')
        for d in s.drones:
            if any(w.inventory.items[id].holder_id!=f'drone_{d.serial}' for id in d.battery_ids):
                raise ValueError('Потеряна установленная батарея')
    if session.landscape is not None and type(session.landscape) is not Landscape:raise ValueError('Неверный ландшафт')
    if not isinstance(session.ui,dict):raise ValueError('Неверные параметры просмотра')
    ui=session.ui
    if 'speed' in ui and ui['speed'] not in (1,10,50,200,1000,2000):raise ValueError('Неверное ускорение просмотра')
    if 'render_mode' in ui and ui['render_mode'] not in ('auto','detail','off'):raise ValueError('Неверный режим карты')
    if 'view_range' in ui:
        bounds=ui['view_range']
        if (not isinstance(bounds,(tuple,list)) or len(bounds)!=2 or any(not isinstance(pair,(tuple,list)) or
                len(pair)!=2 or any(not isinstance(v,(float,int)) or not math.isfinite(v) for v in pair) or
                pair[0]>=pair[1] for pair in bounds)):raise ValueError('Неверный масштаб карты')
    if 'profile_position' in ui and (not isinstance(ui['profile_position'],(float,int)) or not 0<=ui['profile_position']<=100):
        raise ValueError('Неверное положение профиля')
    if 'profile_direction' in ui and ui['profile_direction'] not in ('cross','along'):raise ValueError('Неверное направление профиля')
    if 'brush_segments' in ui:
        from .plan import PlanSegment
        if not isinstance(ui['brush_segments'],list) or any(type(p) is not PlanSegment for p in ui['brush_segments']):
            raise ValueError('Неверный рисунок плана')
        for segment in ui['brush_segments']:segment.validate(s.config,allow_outside=True)
    if 'profile_segment' in ui and ui['profile_segment'] is not None and type(ui['profile_segment']) is not int:
        raise ValueError('Неверный сегмент профиля')
    if ui.get('power_request') is not None:
        from .power_design import PowerRequest
        PowerRequest(**ui['power_request']).validate()
    if 'complex_templates' in ui:
        from .complexes import ComplexTemplate
        if not isinstance(ui['complex_templates'],list) or any(type(t) is not ComplexTemplate for t in ui['complex_templates']):
            raise ValueError('Неверная библиотека комплексов')
        for t in ui['complex_templates']:t.validate()
    r=s.report(full=False)
    if not all(math.isfinite(r[key]) for key in ('mass_error_kg','energy_error_j')):
        raise ValueError('Неверные физические балансы')
    if session.complex_project is not None:
        from .complexes import ComplexProject
        p=session.complex_project
        if type(p) is not ComplexProject or p.landscape is not session.landscape:
            raise ValueError('Потеряна связь комплекса с общим проектом')
        p.validate()
        if p.active.simulation is not s:raise ValueError('Потерян выбранный комплекс')
        for unit in p.units:
            if unit.simulation is not s:_validate(Session(unit.simulation,session.landscape,{},{}))


def save_state(path, simulation, landscape=None, ui=None,complex_project=None):
    """Save between physical steps, including airborne cargo and active services."""
    path=Path(path);temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as file:
            temporary=Path(file.name)
            with zipfile.ZipFile(file,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
                encoder=_Encoder(archive)
                root=encoder.encode(dict(simulation=simulation,landscape=landscape,ui=ui or {},complex_project=complex_project))
                manifest=dict(format=FORMAT,version=VERSION,root=root,nodes=encoder.nodes,
                              runtime=dict(python=platform.python_version(),numpy=np.__version__))
                archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,allow_nan=False,separators=(',',':')))
            file.flush();os.fsync(file.fileno())
        os.replace(temporary,path);temporary=None
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def load_state(path):
    """Read a self-contained snapshot; leave the caller's current world untouched."""
    try:
        with zipfile.ZipFile(path) as archive:
            entries=archive.infolist()
            if len(entries)>20000 or len({e.filename for e in entries})!=len(entries) or sum(e.file_size for e in entries)>MAX_BYTES:
                raise ValueError('Сохранение превышает допустимый размер')
            if archive.getinfo('manifest.json').file_size>256_000_000:raise ValueError('Слишком большой список объектов')
            manifest=json.loads(archive.read('manifest.json'))
            if manifest.get('format')!=FORMAT or type(manifest.get('version')) is not int or manifest['version'] not in range(1,VERSION+1):
                raise ValueError('Неизвестный формат или версия сохранения')
            decoder=_Decoder(archive,manifest['nodes']);root=decoder.decode(manifest['root'])
            if manifest['version']==1:_migrate_v1(decoder)
            # Old missions keep their exact target and original common-angle
            # law. No automatic re-excavation or silent material conversion.
            from .config import Config
            from .config import TerrainConfig
            from .logistics_config import LogisticsConfig
            from .terrain import Terrain
            for value in decoder.objects:
                if isinstance(value,Config) and 'slope_model' not in vars(value):value.slope_model='legacy'
                if isinstance(value,TerrainConfig) and 'compact_spoil' not in vars(value):value.compact_spoil=False
                if isinstance(value,LogisticsConfig) and 'solar' not in vars(value):value.solar=None
                if isinstance(value,LogisticsConfig) and 'battery_platforms' not in vars(value):value.battery_platforms=None
                if isinstance(value,LogisticsConfig) and 'self_solar' not in vars(value):value.self_solar=None
                if isinstance(value,Terrain) and not hasattr(value,'dump_slope'):value.dump_slope=value.slope
            if manifest['version']<11:
                from .workfront import Workfront
                from .simulation import Simulation
                # Scheduling-only migration. Keep the exact terrain, inventory,
                # actor coordinates and every already committed move or flight.
                for value in decoder.objects:
                    if isinstance(value,Workfront):
                        value.settings['sector_length_m']=200.
                        value.stationed=True;value.moving=False;value.legacy_pickups=[]
                        value._set_sector(value.position);value.update()
                    elif isinstance(value,Config) and value.plan and (value.plan.channel or {}).get('operations'):
                        value.plan.channel['operations']['sector_length_m']=200.
                for value in decoder.objects:
                    if isinstance(value,Simulation) and value.terrain.front is not None:
                        value.terrain._spoil_front_key=None
                        value.terrain.front.moving=value.base.destination is not None
                        if value.terrain.front.moving:value.terrain.front.stationed=False;value.terrain.front.revision=-1
                        value.terrain.front.legacy_pickups=[(*d.trip.dig,d.config.grab_radius) for d in value.drones
                            if d.trip is not None and d.trip.source=='cut']
                        value.event('Сохранение принято: фиксированные сектора 200 м; текущая позиция и начатые рейсы сохранены')
        session=Session(root['simulation'],root['landscape'],root['ui'],manifest.get('runtime',{}),root.get('complex_project'))
        if session.landscape is not None:
            from .channel import marine_window
            simulations=[u.simulation for u in session.complex_project.units] if session.complex_project else [session.simulation]
            masks={}
            for s in simulations:
                geo=s.config.georeference
                if geo and 'marine' not in geo and 'origin_m' in geo and (session.landscape.raster.metadata['crs']!='LOCAL_METRES' or session.landscape.raster.metadata.get('ocean_boundary',False)):
                    key=(*geo['origin_m'],s.terrain.cell*s.terrain.height.shape[0])
                    if key not in masks:masks[key]=marine_window(session.landscape.raster,geo['origin_m'],key[-1])
                    geo['marine']=masks[key]
        _validate(session)
        return session
    except (zipfile.BadZipFile,EOFError,KeyError,TypeError,AttributeError,IndexError,OverflowError,RecursionError) as error:
        raise ValueError('Повреждённое сохранение ATS') from error
