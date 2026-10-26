"""All model constants in SI units; electrical powers are estimates."""
from dataclasses import dataclass, field, asdict,fields,is_dataclass
import math
from .equipment import KWH


@dataclass
class TrenchConfig:
    length: float = 2.0
    width: float = 1.0  # flat bottom, NOT top opening
    depth: float = 0.25
    x: float = 16.0
    y: float = 16.0
    angle_deg: float = 35.0


@dataclass
class TerrainConfig:
    n: int = 128
    cell_size: float = 0.25
    density: float = 1600.0
    floor: float = -3.0
    roughness: float = 0.0
    grab_radius: float = 0.5
    dump_rect: tuple = (22.0, 19.0, 9.0, 12.0)
    dump_max_height: float = 2.5
    dump_slope_factor: float = 0.90
    relaxation: float = 0.45
    avalanche_sweeps: int = 3
    avalanche_epsilon: float = 1e-9
    avalanche_stop_volume: float = 1e-10
    completion_volume: float = 0.05


@dataclass
class DroneConfig:
    name: str = "Тяжёлый 60"
    empty_mass: float = 105.0
    payload: float = 60.0
    battery: float = 5 * KWH
    initial_charge: float = 1.0
    reserve: float = 0.20
    charge_threshold: float = 0.30
    empty_speed: float = 6.0
    loaded_speed: float = 4.0
    vertical_speed: float = 2.0
    work_height: float = 6.0
    scoop_seconds: float = 10.0
    dig_rate_kg_s: float = 6.0
    digging_j_m3: float = 100_000.0  # loose sand estimate, not rock cutting
    dump_seconds: float = 10.0
    winch_seconds: float = 5.0
    stowed_cable: float = 1.0
    hover_power: float = 18_000.0
    flight_factor: float = 1.08
    winch_power: float = 500.0
    winch_efficiency: float = 0.8
    gravity: float = 9.80665
    budget_margin: float = 1.25
    equipment: object = None
    grab_radius: float | None = None

    def scoop_duration(self, mass):
        return max(self.scoop_seconds, mass/self.dig_rate_kg_s)


@dataclass
class BaseConfig:
    x: float = 5.0
    y: float = 16.0
    roof_height: float = 1.5
    radius: float = 0.75
    mass: float = 1000.0  # includes battery, panels, tracks and equipment
    reference_total_mass: float = 1105.0  # calibration for move_power
    panel_area: float = 20.0
    panel_efficiency: float = 0.20
    peak_irradiance: float = 1000.0
    latitude_deg: float = 55.0
    start_day_of_year: int = 172  # 21 June in the non-leap model calendar
    battery: float = 30 * KWH
    initial_charge: float = 1.0
    charge_power: float = 8000.0  # input from the base, output = input * efficiency
    charge_efficiency: float = 0.90
    speed: float = 0.20
    move_power: float = 1000.0
    max_slope: float = 0.15
    relocation_trips: int = 5
    relocation_cooldown: float = 900.0
    relocation_min_gain: float = 0.15
    relocation_clearance: float = 2.0
    idle_power: float = 0.0
    equipment: object = None


@dataclass
class DustConfig:
    downsample: int = 4
    wind_x: float = 1.0
    wind_y: float = 0.4
    strength: float = 1.0
    lifetime: float = 18.0
    diffusion: float = 0.08
    height_scale: float = 4.0
    concentration_scale: float = 4.0


@dataclass
class FleetConfig:
    count: int = 1
    charge_slots: int = 1
    models: list[DroneConfig] = field(default_factory=list)
    wait_mode: str = "ground"
    separation: float = 2.0
    pad_spacing: float = 3.0
    waiting_offset: float = 3.0
    ground_clearance: float = 0.4
    altitude_spacing: float = 3.0
    detour_distance: float = 3.0
    emergency_landing_seconds: float = 20.0
    comparison_step_limit: int = 2_000_000


@dataclass
class Config:
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    trench: TrenchConfig = field(default_factory=TrenchConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    base: BaseConfig = field(default_factory=BaseConfig)
    dust: DustConfig = field(default_factory=DustConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    plan: object = None  # PlanConfig, imported lazily to keep configuration simple
    initial_height: list | None = None
    reference_height: list | None = None  # immutable imported DEM, metres above MSL
    georeference: dict | None = None
    seed: int = 42
    dt: float = 1.0
    start_hour: float = 9.0
    event_limit: int = 300
    frame_budget_ms: float = 12.0
    max_frame_steps: int = 200
    display_interval_ms: int = 40
    terrain_display_seconds: float = 0.20
    logistics: object = None  # LogisticsConfig; None preserves the original engine
    legacy_laser_equipment: dict | None = None
    soil_profile: object = None  # SoilProfile; None preserves pre-layer scenarios/saves
    initial_soil: dict | None = None  # sparse fine columns for a regional window

    def drone_models(self):
        from copy import deepcopy
        return deepcopy(self.fleet.models) if self.fleet.models else [deepcopy(self.drone) for _ in range(self.fleet.count)]

    def validate(self):
        if self.soil_profile is not None:
            self.soil_profile.validate()
        elif self.initial_soil is not None:
            raise ValueError('Состав грунта требует профиля материалов')
        if self.logistics is not None:
            self.logistics.validate()
            if self.logistics.direct_allowed and self.logistics.excavator_enabled and self.fleet.count<2:
                raise ValueError('Смешанной команде нужны как минимум два дрона; одиночный грейфер доступен в старом режиме')
        def require(condition, message):
            if not condition:
                raise ValueError(message)
        def finite_values(value):
            if is_dataclass(value):
                return all(finite_values(getattr(value,f.name)) for f in fields(value)
                           if f.name not in ('initial_height','reference_height'))
            if isinstance(value, dict):
                return all(finite_values(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite_values(v) for v in value)
            return not isinstance(value, (int, float)) or math.isfinite(value)
        require(finite_values(self), "Все параметры должны быть конечными")
        t, q, d, b, dust = self.terrain, self.trench, self.drone, self.base, self.dust
        f = self.fleet
        require(isinstance(f.count, int) and 1 <= f.count <= 6 and isinstance(f.charge_slots, int) and 1 <= f.charge_slots <= 6, "Число дронов и зарядных слотов: от 1 до 6")
        require(not f.models or len(f.models) == f.count, "Число моделей не совпадает с числом дронов")
        require(f.wait_mode in ("ground", "air"), "Неизвестный режим очереди")
        require(f.separation > 0 and f.pad_spacing > f.separation and f.altitude_spacing > f.separation and f.waiting_offset > f.separation and f.ground_clearance > 0 and f.detour_distance > f.separation and f.emergency_landing_seconds > 0 and f.comparison_step_limit > 0, "Неверные расстояния безопасности или предел сравнения")
        for model in f.models:
            clone = Config(terrain=t, trench=q, drone=model, base=b, dust=dust, seed=self.seed, dt=self.dt, start_hour=self.start_hour)
            clone.validate()
        require(isinstance(t.n, int) and t.n >= 16 and t.cell_size > 0, "Неверная сетка")
        require(self.dt == 1.0, "Движок использует фиксированный шаг 1 с")
        require(t.density > 0 and t.floor < -q.depth - t.roughness, "Неверная плотность/нижняя граница грунта")
        require(t.roughness >= 0 and 0 < t.relaxation <= 0.5, "Неверные параметры рельефа")
        require(isinstance(t.avalanche_sweeps, int) and t.avalanche_sweeps > 0, "Неверное число проходов осыпания")
        require(t.grab_radius >= t.cell_size and t.completion_volume > 0 and t.avalanche_epsilon > 0, "Неверный радиус/допуск")
        require(t.avalanche_stop_volume > 0, "Неверный допуск остановки осыпания")
        require(isinstance(self.seed, int) and self.seed >= 0, "Seed должен быть неотрицательным целым числом")
        require(self.event_limit > 0 and self.frame_budget_ms > 0 and self.max_frame_steps > 0 and self.display_interval_ms > 0 and self.terrain_display_seconds > 0, "Неверные параметры обновления интерфейса")
        require(q.length > 0 and q.width > 0 and q.depth > 0 and 5 <= q.angle_deg <= 55, "Неверная геометрия канавы")
        side = t.n * t.cell_size
        margin = q.depth / math.tan(math.radians(q.angle_deg)) + t.cell_size
        require(q.length >= t.cell_size and q.width >= t.cell_size, "Дно меньше разрешения сетки")
        require(q.x-q.length/2-margin > 0 and q.x+q.length/2+margin < side and q.y-q.width/2-margin > 0 and q.y+q.width/2+margin < side, "Канава с откосами выходит за карту")
        x, y, w, h = t.dump_rect
        require(w >= t.cell_size and h >= t.cell_size and x >= 0 and y >= 0 and x+w <= side and y+h <= side, "Отвал выходит за карту или слишком мал")
        require(t.dump_max_height > 0 and 0 < t.dump_slope_factor < 1, "Неверная высота/уклон отвала")
        require(0 < b.x < side and 0 < b.y < side and b.radius > 0 and b.roof_height > 0, "Неверное положение/размер базы")
        require(b.mass > 0 and b.reference_total_mass > 0, "Масса базы должна быть положительной")
        require(-90 <= b.latitude_deg <= 90 and isinstance(b.start_day_of_year, int) and 1 <= b.start_day_of_year <= 365, "Неверная широта или день года")
        require(d.stowed_cable > 0 and d.dump_seconds >= 2*d.winch_seconds, "Выгрузка должна включать опускание и подъём грейфера")
        for obj, keys in [(d, ("empty_mass", "payload", "battery", "empty_speed", "loaded_speed", "vertical_speed", "work_height", "hover_power", "scoop_seconds", "dump_seconds", "winch_seconds")), (b, ("speed", "move_power", "max_slope", "relocation_trips", "relocation_cooldown", "relocation_clearance"))]:
            require(all(getattr(obj, k) > 0 for k in keys), "Масса, скорость, ёмкость и длительности должны быть положительными")
        require(0 <= d.initial_charge <= 1 and 0 <= b.initial_charge <= 1 and 0 <= d.reserve < 1, "Неверный заряд/резерв")
        require(d.reserve <= d.charge_threshold <= 1, "Порог зарядки должен быть между резервом и 100%")
        require(b.battery>=0,"Ёмкость базы не может быть отрицательной")
        if b.equipment is not None:b.equipment.validate()
        if d.equipment is not None:d.equipment.validate()
        if d.grab_radius is not None:require(min(t.cell_size,.25)<=d.grab_radius<=f.separation/2,"Радиус ковша должен помещаться в расстояние между дронами")
        if self.initial_height is not None:
            require(len(self.initial_height) == t.n and all(len(row) == t.n for row in self.initial_height), "Размер исходного рельефа не совпадает с сеткой")
            import numpy as np
            h=np.asarray(self.initial_height,dtype=float)
            require(bool(np.all(np.isfinite(h))), "Все параметры должны быть конечными")
            require(bool(np.all(h>=t.floor)), "Исходный рельеф ниже неподвижного основания")
        if self.plan is not None:
            self.plan.validate(self)
        require(bool(d.name.strip()), "У дрона должно быть имя")
        require(0 < b.charge_efficiency <= 1 and 0 <= b.panel_efficiency <= 1 and b.panel_area >= 0 and b.peak_irradiance >= 0 and b.charge_power >= 0, "Неверные параметры солнечной энергии/зарядки")
        require(0 <= self.start_hour < 24, "Неверный стартовый час")
        require(d.flight_factor >= 1 and d.budget_margin >= 1 and d.winch_power >= 0 and b.idle_power >= 0 and 0 <= b.relocation_min_gain < 1, "Неверные энергетические параметры")
        require(d.dig_rate_kg_s > 0 and d.digging_j_m3 >= 0, "Неверная скорость или удельная энергия копания")
        if self.reference_height is not None:
            require(self.georeference is not None and self.georeference.get('vertical_datum') == 'MSL', "DEM должен иметь высоты относительно уровня моря (MSL)")
            require(len(self.reference_height) == t.n and all(len(row) == t.n for row in self.reference_height), "Размер DEM не совпадает с сеткой")
            import numpy as np
            h=np.asarray(self.reference_height,dtype=float)
            require(bool(np.all(np.isfinite(h))), "Все параметры должны быть конечными")
            require(bool(np.all(h>=t.floor)), "DEM ниже неподвижного основания")
        require(0 < d.winch_efficiency <= 1 and d.gravity > 0, "Неверные параметры лебёдки/гравитации")
        require(dust.downsample >= 1 and isinstance(dust.downsample, int) and dust.lifetime > 0 and dust.diffusion >= 0 and dust.strength >= 0 and dust.height_scale > 0 and dust.concentration_scale > 0, "Неверная модель пыли")
