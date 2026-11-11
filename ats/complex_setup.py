"""A recipe is reviewed before any Simulation, inventory or vehicle is created."""
from copy import deepcopy
from dataclasses import dataclass
from .equipment import MAX_DRONES,resolve_config
from .power_design import equipment_copy
from .logistics_config import LogisticsConfig
from .work_schedule import WorkSchedule
from .scheduled_power import design_scheduled
from .costs import default_catalog,validate_catalog,quote


@dataclass(frozen=True)
class SetupRequest:
    drones: int = 4
    excavator: bool = False
    containers: int = 8
    name: str = 'Рабочий комплекс'
    excavators: int | None = None

    @property
    def robot_count(self):return int(self.excavator) if self.excavators is None else self.excavators

    def validate(self):
        if type(self.drones) is not int or not 1<=self.drones<=MAX_DRONES:
            raise ValueError(f'Выберите от 1 до {MAX_DRONES} дронов')
        if type(self.excavator) is not bool:raise ValueError('Неверный выбор Крота')
        if type(self.robot_count) is not int or not 0<=self.robot_count<=10:raise ValueError('Выберите от 0 до 10 Кротов')
        if self.robot_count and self.drones<2:raise ValueError('Кроту нужен курьер: выберите минимум два дрона')
        if type(self.containers) is not int or not 0<=self.containers<=128:raise ValueError('Количество грунтовых ёмкостей: 0–128')
        if self.robot_count and self.containers<self.robot_count:raise ValueError('Нужна минимум одна грунтовая ёмкость на каждого Крота')
        if not isinstance(self.name,str) or not self.name.strip():raise ValueError('Укажите название комплекса')


def recipe_config(source,request,catalog=None):
    request.validate();c=equipment_copy(source)
    # Equipment editing and sizing must not clone a million-cell DEM or change
    # the active site's initial/reference/soil arrays. These belong to the site.
    fields=('initial_height','reference_height','initial_soil','georeference','plan')
    site={key:getattr(c,key) for key in fields}
    for key in fields:setattr(c,key,None)
    models=c.drone_models();c.fleet.count=request.drones
    c.fleet.models=[deepcopy(models[i%len(models)]) for i in range(request.drones)]
    c.base.equipment.landing_slots=request.drones
    if c.logistics is None:c.logistics=LogisticsConfig()
    c.logistics.container_count=request.containers if request.robot_count else 0
    for key,value in site.items():setattr(c,key,deepcopy(value) if key in ('georeference','plan') else value)
    c.logistics.excavator_enabled=bool(request.robot_count)
    c.logistics.excavator_count=max(1,request.robot_count)
    c.logistics.direct_allowed=True;c.logistics.idle_digging=True
    if catalog is None:catalog=getattr(source,'cost_catalog',None)
    c.cost_catalog=deepcopy(default_catalog() if catalog is None else catalog)
    validate_catalog(c.cost_catalog);c.energy_design=None
    resolve_config(c);c.validate();return c


@dataclass
class ComplexDesign:
    config: object
    design: dict
    estimate: dict
    request: SetupRequest

    def template(self):
        from .complexes import ComplexTemplate
        return ComplexTemplate.capture(self.config,self.request.name)


def calculate(source,request,schedule=None,power_request=None,catalog=None,*,cancelled=lambda:False):
    config=recipe_config(source,request,catalog)
    config,design=design_scheduled(config,schedule or WorkSchedule(),power_request,cancelled=cancelled)
    return ComplexDesign(config,design,quote(config),request)


def place(design,landscape=None,*,position=None,allow_resize=True):
    """The single commit point for first deployment, after a complete review."""
    from .winter_power import ready_design
    from .checkpoint import Session
    from .complexes import ComplexProject
    c=equipment_copy(design.config)
    if position is not None:c.base.x,c.base.y=map(float,position)
    sim,_=ready_design(c,design.design,allow_resize=allow_resize)
    if position is not None:
        import numpy as np
        if np.linalg.norm(sim.base.position[:2]-position)>1e-6:
            raise ValueError('Выбранная точка не вмещает комплекс с раскрытыми опорами и проходами')
    project=ComplexProject(sim,landscape);project.units[0].name=design.request.name
    project.templates.append(design.template())
    return Session(sim,landscape,{},complex_project=project)
