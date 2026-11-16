"""Deliver the first autonomous complex onto a live conventional worksite.

Only hardware is constructed. The existing Terrain, material ledger and diesel
fleet stay live objects; no snapshot or new heightfield is used as a staging map.
"""
from collections import deque
from copy import deepcopy
import numpy as np


def _configuration(source,template,point):
    from .power_design import equipment_copy
    c=equipment_copy(template.config);c.conventional_only=False
    for key in ('terrain','trench','plan','georeference'):
        setattr(c,key,deepcopy(getattr(source.config,key)))
    for key in ('initial_height','reference_height','initial_soil','soil_profile','local_reference','slope_model'):
        setattr(c,key,getattr(source.config,key))
    # The diesel fleet has absolute deadlines against this original clock.
    c.start_hour=source.config.start_hour
    c.base.start_day_of_year=source.config.base.start_day_of_year
    c.base.latitude_deg=source.config.base.latitude_deg
    c.base.x,c.base.y=map(float,point)
    return c


def _swept_rectangle(a,b,radius):
    a,b=np.asarray(a)[:2],np.asarray(b)[:2];delta=b-a
    length=float(np.linalg.norm(delta))
    direction=delta/length if length>1e-9 else np.array([1.,0.])
    side=np.array([-direction[1],direction[0]])*radius
    lo=a-direction*radius;hi=b+direction*radius
    return np.array([lo+side,hi+side,hi-side,lo-side])


def _check_envelope(source,template,side):
    """Reject a geometrically impossible translation before any world spawn.

    The heading is exactly configure_layout's accepted heading, including the
    source plan's automatic heading. No alternative packing or map is invented.
    """
    from .base_layout import configure_layout,platform_polygons,rotate_offset
    from .equipment import station_offset
    from .self_solar import deployment_polygons as solar_polygons
    from .battery_platforms import deployment_polygons as battery_polygons
    from .site_ground import rectangle_polygon
    c=_configuration(source,template,(0.,0.));extra=[]
    for row in template.stock or []:
        charger=c.logistics.charger_slot_mass_kg if c.logistics and c.logistics.individual_chargers else 0.
        extra.extend([(row['mass_kg']+charger,row['capacity_kwh'],f'initial_stock_{len(extra)+i+1:05}')
                      for i in range(row['count'])])
    spec=configure_layout(c,extra)
    if spec:
        polygons=[*platform_polygons(spec,(0.,0.)),*solar_polygons(spec,(0.,0.)),*battery_polygons(spec,(0.,0.))]
    else:
        r=c.base.radius;polygons=[rectangle_polygon((-r,-r,2*r,2*r))]
    f=c.fleet;offsets=(spec or {}).get('dock_offsets',[])
    distance=max(f.waiting_offset,c.base.radius+3.+f.separation) if offsets else f.waiting_offset
    for i in range(max(f.count,f.charge_slots)):
        for dx in (0.,-distance):
            offset=offsets[i] if i<len(offsets) else (0.,station_offset(i,f.pad_spacing))
            x,y=rotate_offset(c.base.layout,(dx+offset[0],offset[1]));r=f.separation
            polygons.append(rectangle_polygon((x-r,y-r,2*r,2*r)))
    points=np.concatenate(polygons);span=points.max(axis=0)-points.min(axis=0)
    if np.any(span>side+1e-8):
        raise ValueError(f'Габариты развёрнутого комплекса {span[0]:.1f} × {span[1]:.1f} м превышают размер карты {side:g} м. Увеличьте карту или уменьшите комплектацию')


def _check_placement(project,fleet):
    """Full future deployment and access, not just vehicle/base centres."""
    from .site_ground import polygons_overlap,rectangle_polygon
    from .self_solar import deployment_polygons as solar_polygons
    from .battery_platforms import deployment_polygons as battery_polygons
    site=project.shared_site;s=project.units[0].simulation
    polygons=[polygon for _,polygon in site._unit_geometry(project.units[0])['entries']]
    spec=getattr(s.base,'layout_spec',None)
    if spec:
        polygons.extend(solar_polygons(spec,s.base.position))
        polygons.extend(battery_polygons(spec,s.base.position))
    if fleet is None:return
    foreign=[polygon for _,polygon in site._conventional_geometry(fleet)['entries']]
    x,y,w,h=fleet.config.work_rect
    margin=max([site.core.cell,*[d.config.grab_radius or site.core.cell for d in s.drones]])
    if s.logistics:margin=max(margin,s.logistics.config.reach_m)
    foreign.append(rectangle_polygon((x-margin,y-margin,w+2*margin,h+2*margin)))
    for machine in fleet.machines:
        points=[machine.position[:2],*[np.asarray(p)[:2] for p in machine.path]]
        for a,b in zip(points,points[1:]):foreign.append(_swept_rectangle(a,b,machine.radius+.3))
    if any(polygons_overlap(a,b,clearance=.3) for a in polygons for b in foreign):
        raise ValueError('Комплекс или его раскрытие пересекает обычную технику, её захватку либо проверенный наземный маршрут')


def _stage_new_robot(sim):
    """A delivered single Krot starts beside its own delivered base and racks."""
    from .base_layout import rotate_offset,platform_polygons
    from .site_ground import rectangle_polygon,polygons_overlap
    from .self_solar import deployment_polygons as solar_polygons
    from .battery_platforms import deployment_polygons as battery_polygons
    w=sim.logistics
    if w is None or len(w.excavators)!=1:return
    robot=w.excavators[0];old=robot.position.copy()
    radius=sim.config.base.radius+w.config.support_half_width_m+2.
    spec=getattr(sim.base,'layout_spec',None)
    if spec:
        occupied=[*platform_polygons(spec,sim.base.position),*solar_polygons(spec,sim.base.position),
                  *battery_polygons(spec,sim.base.position)]
    else:
        r=sim.config.base.radius;x,y=sim.base.position[:2]
        occupied=[rectangle_polygon((x-r,y-r,2*r,2*r))]
    occupied.extend(w.solar_ground_polygons())
    for delta in ((radius,0.),(-radius,0.),(0.,radius),(0.,-radius)):
        point=sim.base.position.copy();point[:2]+=rotate_offset(sim.config.base.layout,delta)
        point[2]=sim.terrain.sample(*point[:2])
        support=w.config.support_half_width_m
        footprint=rectangle_polygon((point[0]-support,point[1]-support,2*support,2*support))
        if any(polygons_overlap(footprint,polygon) for polygon in occupied):continue
        if not robot.stable(point,sim.terrain):continue
        shift=point-old;robot.position=point;w.rack=point.copy()
        for item in w.inventory.items.values():
            if np.linalg.norm(np.asarray(item.position)-old)<1e-7:
                item.position=tuple(np.asarray(item.position)+shift)
        return
    raise ValueError('Нет безопасной площадки нового Крота рядом с базой')


def adopt_first(source_sim,template,landscape=None,position=None):
    """Return a complete mixed Session, or leave the old worksite untouched.

    Call between simulation steps with the source GUI timer paused. Map size and
    cell resolution are kept: insufficient space never silently resets terrain.
    """
    from .simulation import Simulation
    from .complexes import ComplexProject
    from .checkpoint import Session
    from .conventional_integration import reservation_mask
    if not source_sim.config.conventional_only or source_sim.drones or source_sim.logistics is not None:
        raise ValueError('Первый комплекс добавляется только на площадку без автономной техники')
    if source_sim.shared_site is not None or getattr(source_sim,'regional_stage',None):
        raise ValueError('Сначала завершите изменение рабочего окна')
    template.validate()
    if template.config.conventional_only or template.config.fleet.count<1:
        raise ValueError('Выберите настоящую автономную комплектацию с дронами')
    t=source_sim.terrain;fleet=getattr(source_sim,'conventional',None)
    side=t.height.shape[0]*t.cell
    _check_envelope(source_sim,template,side)
    if position is not None:
        point=np.asarray(position,dtype=float)
        if point.shape!=(2,) or not np.all(np.isfinite(point)):raise ValueError('Неверные координаты нового комплекса')
        candidates=[tuple(point)]
    else:
        goal=np.array([template.config.base.x,template.config.base.y])
        candidates=[tuple(goal),*sorted([(float(x),float(y)) for x in np.linspace(5.,side-5.,15)
            for y in np.linspace(5.,side-5.,15)],key=lambda p:float(np.sum((np.asarray(p)-goal)**2)))]
    # Constructors may populate caches and temporarily assign _site_owner.
    # Height/material arrays are neither copied nor written during this phase.
    baseline=vars(t).copy()
    cache_copies={key:deepcopy(value) for key,value in baseline.items()
        if 'cache' in key and isinstance(value,(dict,list,deque))}
    previous_fleet_sim=fleet.sim if fleet is not None else None

    def restore():
        vars(t).clear();vars(t).update(baseline)
        vars(t).update({key:deepcopy(value) for key,value in cache_copies.items()})
        if fleet is not None:fleet.sim=previous_fleet_sim

    last_error='Нет безопасного места'
    try:
        for point in candidates:
            restore()
            c=_configuration(source_sim,template,point)
            permanent=False
            try:
                sim=Simulation(c,terrain=t,initial_stock=deepcopy(template.stock))
                if sim.state.value=='BLOCKED':raise ValueError(sim.reason)
                if position is not None and np.linalg.norm(sim.base.position[:2]-point)>1e-6:
                    raise ValueError('Выбранная точка не вмещает полный развёрнутый комплекс')
                sim.time=source_sim.time;sim._terrain_active=source_sim._terrain_active
                sim.conventional=fleet
                if fleet is not None:
                    fleet.sim=sim
                    t._conventional_mask=reservation_mask(t,fleet.config,[sim])
                    free=(t.dig_mask|t.fill_mask)&~t._conventional_mask
                    volume=float(np.abs(t.height-t.target)[free].sum()*t.area)
                    if volume<=t.config.terrain.completion_volume:
                        permanent=True
                        raise ValueError('За пределами захватки обычной техники нет работы для дронов; расширьте план или уменьшите захватку')
                _stage_new_robot(sim)
                project=ComplexProject(sim,landscape)
                project.units[0].name=template.name
                project.templates.append(deepcopy(template))
                project.enable_shared()
                _check_placement(project,fleet)
                project.delivered_energy_kwh=project.initial_energy_kwh;project.initial_energy_kwh=0.
                project.delivered_equipment_mass=project.initial_equipment_mass;project.initial_equipment_mass=0.
                # born_at is the existing affine clock offset. New hardware did
                # not work before arrival; its own productivity starts now.
                for drone in sim.drones:drone.created_at=sim.time
                sim.events=deque(source_sim.events,maxlen=source_sim.events.maxlen)
                sim.event_serial=source_sim.event_serial;sim._event_keys=dict(source_sim._event_keys)
                sim.event(f'Первый автономный комплекс поставлен на общую карту: {template.name}')
                if fleet is not None:fleet.validate()
                project.validate()
                source_sim.conventional=None
                return Session(sim,landscape,{},complex_project=project)
            except ValueError as error:
                last_error=str(error)
                if position is not None or permanent:raise
        raise ValueError('Комплекс не помещается на текущей общей карте. Увеличьте участок или уменьшите комплектацию. '+last_error)
    except BaseException:
        restore()
        raise
