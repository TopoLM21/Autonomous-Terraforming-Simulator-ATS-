"""Brush target surfaces and geometry for independent plan/profile segments."""
from dataclasses import dataclass, field, asdict
import json
import math
import base64
import numpy as np


@dataclass
class PlanSegment:
    points: list
    mode: str = "dig"
    width: float = 1.0
    height: float = -0.25
    name: str = "Сегмент"

    def validate(self, config, *, allow_outside=False):
        side = config.terrain.n*config.terrain.cell_size
        if self.mode not in ("dig", "fill") or not self.points or len(self.points) > 4096:
            raise ValueError("Неверный режим кисти или линия сегмента")
        if not isinstance(self.name,str) or not self.name.strip() or len(self.name)>80:
            raise ValueError("Имя сегмента должно содержать от 1 до 80 символов")
        upper = config.terrain.dump_max_height
        if config.reference_height is not None:
            upper = max(10000.,max(max(row) for row in config.reference_height)+upper)
        if not math.isfinite(self.width) or not math.isfinite(self.height) or self.width < config.terrain.cell_size or not config.terrain.floor < self.height <= upper:
            raise ValueError("Неверная ширина или целевая высота кисти")
        if any(len(p) != 2 or any(not math.isfinite(v) or (not allow_outside and not 0 < v < side) for v in p) for p in self.points):
            raise ValueError("Линия кисти выходит за карту")


@dataclass
class PlanConfig:
    segments: list[PlanSegment] = field(default_factory=list)
    channel: dict | None = None

    def validate(self, config):
        if len(self.segments) > 100:
            raise ValueError("План ограничен 100 сегментами")
        for segment in self.segments:
            segment.validate(config,allow_outside=self.channel is not None)
        if self.channel is not None:
            if any(s.mode!='dig' or s.height>=0 for s in self.segments):
                raise ValueError('Проект канала задаёт выемку ниже моря. Для отдельной засыпки сначала очистите план канала')
            for key in ('start','tangent'):
                if len(self.channel.get(key,[]))!=2 or not all(math.isfinite(v) for v in self.channel[key]):
                    raise ValueError('Неверная геометрия берегового отступа')
            if abs(math.hypot(*self.channel['tangent'])-1)>1e-6:
                raise ValueError('Направление берегового отступа должно быть единичным')
            length=self.channel.get('project_length_m')
            if length is not None and (not isinstance(length,(int,float)) or not math.isfinite(length) or length<=0):
                raise ValueError('Неверная длина регионального проекта')
            spans=self.channel.get('finished_spans',[])
            if not isinstance(spans,list) or any(not isinstance(p,(list,tuple)) or len(p)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in p) or not 0<=p[0]<p[1] or length is not None and p[1]>length+1e-6 for p in spans):
                raise ValueError('Неверные интервалы завершённого профиля')
            if 'chainage_starts' in self.channel:
                starts=self.channel['chainage_starts'];first=self.channel.get('first_edges',[])
                if len(starts)!=len(self.segments) or len(first)!=len(starts) or any(not math.isfinite(v) or v<0 for v in starts):
                    raise ValueError('Неверная разбивка маршрута по длине')
            operations=self.channel.get('operations')
            if operations is not None:
                if not isinstance(operations,dict):raise ValueError('Неверные параметры фронта канала')
                positive=('front_length_m','setback_m','bank_width_m')
                if any(not isinstance(operations.get(key),(int,float)) or not math.isfinite(operations[key]) or operations[key]<=0 for key in positive):
                    raise ValueError('Неверные параметры фронта канала')
                length=operations.get('sector_length_m',200.)
                if isinstance(length,bool) or not isinstance(length,(int,float)) or not math.isfinite(length) or not 1<=length<=10000:
                    raise ValueError('Длина рабочего сектора должна быть от 1 до 10000 м')
                anchor=operations.get('base_chainage_m')
                if not isinstance(anchor,(int,float)) or not math.isfinite(anchor) or anchor<0:
                    raise ValueError('Неверное начало движения базы')
                sites=operations.get('base_sites',[])
                extent=config.terrain.n*config.terrain.cell_size
                if not sites or len(sites)>4096 or any(len(p)!=3 or not all(math.isfinite(v) for v in p) or
                        p[0]<0 or not 0<p[1]<extent or not 0<p[2]<extent for p in sites):
                    raise ValueError('Неверные площадки движения базы')


def line_distance(x, y, points):
    result = np.full(np.broadcast_shapes(np.shape(x), np.shape(y)), np.inf)
    pairs = zip(points, points[1:]) if len(points) > 1 else [(points[0], points[0])]
    for a, b in pairs:
        a, b = np.asarray(a), np.asarray(b)
        v = b-a
        length = float(np.dot(v, v))
        fraction = np.clip(((x-a[0])*v[0]+(y-a[1])*v[1])/length, 0, 1) if length else 0
        result = np.minimum(result, np.hypot(x-a[0]-fraction*v[0], y-a[1]-fraction*v[1]))
    return result


def target_surface(terrain, plan):
    baseline = terrain.height.copy()
    target = baseline.copy()
    labels = np.zeros(baseline.shape, dtype=np.int16)
    protected,shore_cap=channel_shore(terrain,plan)
    for number, segment in enumerate(plan.segments, 1):
        distance = line_distance(terrain.x, terrain.y, segment.points)
        from .slopes import envelope
        offset=np.maximum(0,distance-segment.width/2)
        if segment.mode == "dig":
            bound = envelope(terrain,offset,segment.height,'dig')
            influence = (distance <= segment.width/2) | (bound < baseline-1e-12)
            surface = np.minimum(baseline, bound)
        else:
            bound = envelope(terrain,offset,segment.height,'fill')
            influence = (distance <= segment.width/2) | (bound > baseline+1e-12)
            surface = np.maximum(baseline, bound)
        if plan.channel is not None:
            if 'chainage_starts' in plan.channel:
                a,b=np.asarray(segment.points[0]),np.asarray(segment.points[-1]);v=b-a
                length=float(np.linalg.norm(v))
                along=((terrain.x-a[0])*v[0]+(terrain.y-a[1])*v[1])/max(length,1e-12)
                if not plan.channel['first_edges'][number-1]:along=np.clip(along,0,length)
                along=along+plan.channel['chainage_starts'][number-1]
            else:
                along=(terrain.x-plan.channel['start'][0])*plan.channel['tangent'][0]+(terrain.y-plan.channel['start'][1])*plan.channel['tangent'][1]
            original=getattr(terrain,'reference',baseline)
            front=original-np.maximum(0,along)*terrain.slope*.9
            surface=np.maximum(surface,front)
            if shore_cap is not None:surface=np.maximum(surface,shore_cap)
            influence &= (along>0) & (surface<baseline-1e-12)
            if protected is not None:influence &= ~protected
            surface=np.minimum(target,surface)
        target[influence] = surface[influence]
        labels[influence] = number
    return target, labels


def channel_shore(terrain,plan):
    """Preserve marine ground and a stable dry bank, including diagonal routes."""
    marine=plan.channel.get('marine') if plan.channel else None
    if marine is None:return None,None
    rows,cols=marine['shape']
    if not 2<=rows<=16000000 or not 2<=cols<=16000000 or rows*cols>16000000:
        raise ValueError('Неверная береговая маска проекта')
    raw=base64.b64decode(marine['bits'],validate=True)
    if len(raw)!=(rows*cols+7)//8:raise ValueError('Повреждена береговая маска')
    nodes=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),count=rows*cols).reshape(rows,cols).astype(bool)
    rx,ry=marine['spacing'];ox,oy=marine['origin']
    ix=np.clip(np.floor((terrain.x-ox)/rx).astype(int),0,cols-2)
    iy=np.clip(np.floor((terrain.y-oy)/ry).astype(int),0,rows-2)
    original=getattr(terrain,'reference',terrain.height)
    protected=(original<0)&(nodes[iy,ix]|nodes[iy+1,ix]|nodes[iy,ix+1]|nodes[iy+1,ix+1])
    if not np.any(protected):return protected,None
    depth=max(-s.height for s in plan.segments) if plan.segments else 0.
    cap=np.full(terrain.height.shape,-float(depth),dtype=np.float64)
    cell=plan.channel['cell_size'];slope=terrain.slope*.8
    reached=protected.copy()
    for k in range(1,min(max(terrain.height.shape),int(math.ceil(depth/(cell*slope)))+1)+1):
        padded=np.pad(reached,1)
        expanded=(padded[:-2,:-2]|padded[:-2,1:-1]|padded[:-2,2:]|padded[1:-1,:-2]|
                  padded[1:-1,2:]|padded[2:,:-2]|padded[2:,1:-1]|padded[2:,2:]|reached)
        shell=expanded&~reached;cap[shell]=max(-depth,-k*cell*slope)
        if not np.any(shell):break
        reached=expanded
    return protected,cap


def sample_line(points, spacing):
    points = np.asarray(points, dtype=float)
    if len(points) == 1:
        return points.copy(), np.array([0.])
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distances = np.r_[0., np.cumsum(lengths)]
    keep = np.r_[True, lengths > 1e-12]
    points, distances = points[keep], distances[keep]
    if len(points) == 1:
        return points, np.array([0.])
    axis = np.linspace(0, distances[-1], max(2, int(math.ceil(distances[-1]/spacing))+1))
    return np.column_stack([np.interp(axis, distances, points[:, i]) for i in range(2)]), axis


def cross_section(segment, fraction, half_extent, spacing):
    line, axis = sample_line(segment.points, spacing)
    i = int(np.argmin(abs(axis-axis[-1]*fraction)))
    tangent = line[min(i+1, len(line)-1)]-line[max(i-1, 0)]
    length = np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])/length if length else np.array([0., 1.])
    offset = np.linspace(-half_extent, half_extent, max(3, int(math.ceil(2*half_extent/spacing))+1))
    return line[i]+offset[:, None]*normal, offset


def save_plan(path, config):
    data = {"version": 1, "segments": [asdict(s) for s in config.plan.segments], "initial_height": config.initial_height,
            "grid_size": config.terrain.n, "cell_size": config.terrain.cell_size, "seed": config.seed, "roughness": config.terrain.roughness,
            "angle_deg": config.trench.angle_deg, "base_position": [config.base.x,config.base.y]}
    data.update(slope_model=config.slope_model,repose_deg=config.terrain.repose_deg,
                compact_spoil=config.terrain.compact_spoil,spoil_reserve_margin=config.terrain.spoil_reserve_margin,
                spoil_front_sections=config.terrain.spoil_front_sections,
                reference_height=config.reference_height,georeference=config.georeference,
                floor=config.terrain.floor,dump_rect=config.terrain.dump_rect,channel=config.plan.channel)
    from .soil import json_soil_state
    data.update(soil_profile=asdict(config.soil_profile) if config.soil_profile is not None else None,
                initial_soil=json_soil_state(config.initial_soil))
    from pathlib import Path
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def load_plan(path, config):
    from pathlib import Path
    from copy import deepcopy
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("grid_size") != config.terrain.n or data.get("cell_size") != config.terrain.cell_size:
        raise ValueError("Несовместимая версия плана или сетка")
    clone = deepcopy(config)
    clone.slope_model=data.get('slope_model','legacy')
    clone.terrain.repose_deg=data.get('repose_deg',35.)
    clone.terrain.compact_spoil=data.get('compact_spoil',False)
    clone.terrain.spoil_reserve_margin=data.get('spoil_reserve_margin',.2)
    clone.terrain.spoil_front_sections=data.get('spoil_front_sections',2)
    clone.plan = PlanConfig([PlanSegment(**s) for s in data["segments"]],channel=data.get('channel'))
    clone.initial_height = data.get("initial_height")
    clone.reference_height = data.get('reference_height')
    clone.georeference = data.get('georeference')
    from .soil import SoilProfile
    clone.soil_profile=SoilProfile.from_data(data['soil_profile']) if data.get('soil_profile') is not None else None
    clone.initial_soil=data.get('initial_soil')
    clone.terrain.floor = data.get('floor',clone.terrain.floor)
    clone.terrain.dump_rect = tuple(data.get('dump_rect',clone.terrain.dump_rect))
    clone.seed, clone.terrain.roughness = data["seed"], data["roughness"]
    clone.trench.angle_deg = data.get("angle_deg",clone.trench.angle_deg)
    clone.base.x,clone.base.y = data.get("base_position",[clone.base.x,clone.base.y])
    clone.validate()
    return clone
