"""Brush target surfaces and geometry for independent plan/profile segments."""
from dataclasses import dataclass, field, asdict
import json
import math
import numpy as np


@dataclass
class PlanSegment:
    points: list
    mode: str = "dig"
    width: float = 1.0
    height: float = -0.25
    name: str = "Сегмент"

    def validate(self, config):
        side = config.terrain.n*config.terrain.cell_size
        if self.mode not in ("dig", "fill") or not self.points or len(self.points) > 4096:
            raise ValueError("Неверный режим кисти или линия сегмента")
        if not isinstance(self.name,str) or not self.name.strip() or len(self.name)>80:
            raise ValueError("Имя сегмента должно содержать от 1 до 80 символов")
        if not math.isfinite(self.width) or not math.isfinite(self.height) or self.width < config.terrain.cell_size or not config.terrain.floor < self.height <= config.terrain.dump_max_height:
            raise ValueError("Неверная ширина или целевая высота кисти")
        if any(len(p) != 2 or any(not math.isfinite(v) or not 0 < v < side for v in p) for p in self.points):
            raise ValueError("Линия кисти выходит за карту")


@dataclass
class PlanConfig:
    segments: list[PlanSegment] = field(default_factory=list)

    def validate(self, config):
        if len(self.segments) > 100:
            raise ValueError("План ограничен 100 сегментами")
        for segment in self.segments:
            segment.validate(config)


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
    for number, segment in enumerate(plan.segments, 1):
        distance = line_distance(terrain.x, terrain.y, segment.points)
        ramp = np.maximum(0, distance-segment.width/2)*terrain.slope
        if segment.mode == "dig":
            envelope = segment.height+ramp
            influence = (distance <= segment.width/2) | (envelope < baseline-1e-12)
            surface = np.minimum(baseline, envelope)
        else:
            envelope = segment.height-ramp
            influence = (distance <= segment.width/2) | (envelope > baseline+1e-12)
            surface = np.maximum(baseline, envelope)
        target[influence] = surface[influence]
        labels[influence] = number
    return target, labels


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
    from pathlib import Path
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def load_plan(path, config):
    from pathlib import Path
    from copy import deepcopy
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("grid_size") != config.terrain.n or data.get("cell_size") != config.terrain.cell_size:
        raise ValueError("Несовместимая версия плана или сетка")
    clone = deepcopy(config)
    clone.plan = PlanConfig([PlanSegment(**s) for s in data["segments"]])
    clone.initial_height = data.get("initial_height")
    clone.seed, clone.terrain.roughness = data["seed"], data["roughness"]
    clone.trench.angle_deg = data.get("angle_deg",clone.trench.angle_deg)
    clone.base.x,clone.base.y = data.get("base_position",[clone.base.x,clone.base.y])
    clone.validate()
    return clone
