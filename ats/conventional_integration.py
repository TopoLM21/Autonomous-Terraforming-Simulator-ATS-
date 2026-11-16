"""Attach a separate diesel fleet to the existing single physical heightfield."""
import numpy as np


def simulations_for(sim):
    site=getattr(sim,'shared_site',None)
    return [unit.simulation for unit in site.project.units] if site is not None else [sim]


def fleets_for(sim):
    """Discover physical fleets site-wide without attaching aliases to units."""
    seen=set();result=[]
    for owner in simulations_for(sim):
        fleet=getattr(owner,'conventional',None)
        if fleet is not None and id(fleet) not in seen:seen.add(id(fleet));result.append(fleet)
    return result


def fleet_for(sim):
    fleets=fleets_for(sim)
    return fleets[0] if fleets else None


def reservation_mask(terrain,config,simulations):
    """Reserve the contractor's rectangle and the actual grab/reach margin."""
    margin=terrain.cell
    for s in simulations:
        margin=max([margin,*[float(d.config.grab_radius or terrain.cell) for d in s.drones]])
        if s.logistics:margin=max(margin,s.logistics.config.reach_m)
    x,y,w,h=config.work_rect
    return ((terrain.x>=x-margin)&(terrain.x<=x+w+margin)&
            (terrain.y>=y-margin)&(terrain.y<=y+h+margin))


def cut_exclusion(terrain,patch=None):
    """Reserved work plus complete physical camp/body footprints, site-wide.

    Cached local painting avoids full-grid distance calculations every tick.
    The work reservation can be released, while parked physical gear remains.
    """
    core=getattr(terrain,'core',terrain);site=getattr(terrain,'shared_site',None)
    owner=site.project.units[0].simulation if site is not None else getattr(core,'_site_owner',None)
    fleet=fleet_for(owner) if owner is not None else None
    reserved=getattr(core,'_conventional_mask',None)
    if fleet is None:return reserved[patch] if reserved is not None and patch is not None else reserved
    key=(id(core.height),id(reserved),tuple(fleet.base_position[:2]),tuple((m.id,tuple(m.position[:2]),tuple(m.home[:2]),m.radius) for m in fleet.machines))
    cached=getattr(fleet,'_cut_exclusion_cache',None)
    if cached is None or cached[0]!=key:
        mask=reserved.copy() if reserved is not None else np.zeros(core.height.shape,dtype=bool)
        from .site_ground import paint_rect
        x,y=fleet.base_position[:2];paint_rect(core,mask,(x-4,y-4,8,8))
        for machine in fleet.machines:
            radius=machine.radius+.5
            for position in (machine.position,machine.home):
                area=core.patch(*position[:2],radius)
                mask[area]|=(core.x[area]-position[0])**2+(core.y[area]-position[1])**2<=radius**2
        cached=(key,mask);fleet._cut_exclusion_cache=cached
    return cached[1][patch] if patch is not None else cached[1]


def _overlap_circle(terrain,mask,point,radius):
    patch=terrain.patch(*point[:2],radius)
    return bool(np.any(mask[patch]&((terrain.x[patch]-point[0])**2+(terrain.y[patch]-point[1])**2<=radius**2)))


def install(sim,config):
    from .conventional import ConventionalFleet
    from .site_ground import equipment_mask,rectangle_patch
    if fleets_for(sim):raise ValueError('Техника уже приобретена: используйте обслуживание или новый проект')
    owners=simulations_for(sim);site=getattr(sim,'shared_site',None)
    if site is not None and site.project.transition is not None:raise ValueError('Дождитесь завершения регионального переезда')
    config.validate()
    if getattr(config,'auto_work_area',False):
        from .conventional_siting import prepare
        config=prepare(sim,config)
    t=getattr(sim.terrain,'core',sim.terrain)
    patch=rectangle_patch(t,config.work_rect)
    if patch is None:raise ValueError('Рабочий участок должен полностью помещаться на карте')
    reserved=reservation_mask(t,config,owners)
    for s in owners:
        if s.base.destination is not None or s.relocation_pending is not None:raise ValueError('Дождитесь завершения переезда базы')
        if s.drones and np.any(equipment_mask(t,[s])[patch]):raise ValueError('Рабочий участок пересекает установленную автономную технику')
        for drone in s.drones:
            if drone.trip is not None and any(_overlap_circle(t,reserved,p,drone.config.grab_radius or t.cell) for p in (drone.trip.dig,drone.trip.dump)):
                raise ValueError('Участок занят действующим рейсом дрона; завершите или сбросьте его приказ')
        if s.logistics:
            for job in s.logistics.jobs:
                if job.status in ('QUEUED','ACTIVE') and any(_overlap_circle(t,reserved,p,t.cell) for p in (job.origin,job.destination)):
                    raise ValueError('Участок зарезервирован заданием автономного комплекса')
            for e in s.logistics.excavators:
                if (e.bucket_kg or e.active_container or e.state.value in ('DIGGING','FILLING_CONTAINER')) and _overlap_circle(t,reserved,e.position,s.logistics.config.reach_m):
                    raise ValueError('На участке выполняется цикл Крота; завершите текущую выемку')
    if site is not None and any(s.drones for s in owners):
        free=(t.dig_mask|t.fill_mask)&~reserved
        volume=float(np.abs(t.height-t.target)[free].sum()*t.area)
        if volume<=site.completion_volume:
            raise ValueError('После выделения участка обычной техники не остаётся работы для дронов; уменьшите захватку или расширьте план')
    # Construction only reads physical ground. No ownership, purchase or cargo
    # changes are committed until all placement and reservation checks pass.
    fleet=ConventionalFleet(sim,config)
    sim.conventional=fleet;t._conventional_mask=reserved
    if site is not None:site.partition();site.invalidate_geometry()
    sim.event('Обычная техника приобретена: отдельные топливо, зарплаты и смета')
    return fleet


def standalone_report(sim):
    from .simulation import elapsed_text
    fleet=getattr(sim,'conventional',None);t=sim.terrain
    r=dict(state=sim.state.value,reason=sim.reason,time_seconds=sim.time,
        elapsed_display=elapsed_text(sim.time),date=sim.date_text,hour=sim.hour,
        day_of_year=sim.day_of_year,latitude_deg=sim.config.base.latitude_deg,
        drone_count=0,drones=[],autonomous_equipment_cost_usd=0.,
        # Shared-window bookkeeping uses the same report interface, but an
        # ordinary-only site has no autonomous battery, panels or electrical
        # loads. Diesel work remains in the separate conventional report.
        initial_energy_kwh=0.,consumed_kwh=0.,base_capacity_kwh=0.,panel_area_m2=0.,excavated_m3=0.,
        mass_error_kg=t.mass_change+(fleet.carried_mass_kg if fleet else 0.),energy_error_j=0.,
        remaining_m3=t.remaining,remaining_cut_m3=t.remaining_cut,
        map_side_m=t.cell*t.height.shape[0],cell_size_m=t.cell)
    if fleet is not None:r['conventional']=fleet.report()
    if t.soil is not None:
        error=(t.soil.volume_delta_by_material-t._initial_soil_volume)*t.soil.profile.densities
        if fleet:
            for load in fleet.loads():error+=load.masses
        r['soil']=dict(max_material_mass_error_kg=float(np.abs(error).max()))
    return r


def refresh_map(host):
    """Lightweight map overlays; no per-cell or physics work in the renderer."""
    import pyqtgraph as pg
    from .conventional_visuals import refresh_geometry
    fleet=fleet_for(host.sim)
    refresh_geometry(host,host.view_box,fleet)
    if not hasattr(host,'ground_vehicle_markers'):
        if fleet is None:return
        host.ground_vehicle_markers=pg.ScatterPlotItem(size=6,pen=pg.mkPen('#17202b',width=1))
        host.ground_work_outline=pg.PlotDataItem(pen=pg.mkPen('#ffcf66',width=2))
        host.ground_routes=pg.PlotDataItem(pen=pg.mkPen('#f5d69a',width=1))
        host.ground_vehicle_markers.setZValue(3);host.ground_routes.setZValue(2)
        for item in (host.ground_vehicle_markers,host.ground_work_outline,host.ground_routes):host.view_box.addItem(item)
    if fleet is None:
        host.ground_vehicle_markers.clear();host.ground_work_outline.clear();host.ground_routes.clear();return
    x,y,w,h=fleet.config.work_rect
    host.ground_work_outline.setData([x,x+w,x+w,x,x],[y,y,y+h,y+h,y])
    host.ground_vehicle_markers.setData([dict(pos=m.position[:2],symbol='s' if m.kind=='excavator' else 't',
        brush='#f0ba43' if m.kind=='excavator' else '#f07547') for m in fleet.machines]+
        [dict(pos=fleet.base_position[:2],symbol='+',brush='#eee8b0',size=19)])
    points=[p for m in fleet.machines if m.path for p in [m.position[:2],*m.path,(np.nan,np.nan)]]
    if points:
        a=np.asarray(points);host.ground_routes.setData(a[:,0],a[:,1],connect='finite')
    else:host.ground_routes.clear()


def occupied(terrain,rect):
    site=getattr(terrain,'shared_site',None)
    owner=site.project.units[0].simulation if site is not None else getattr(terrain,'_site_owner',None)
    fleet=fleet_for(owner) if owner is not None else None
    if fleet is None:return False
    x,y,w,h=rect;bx,by=fleet.base_position[:2]
    if x<bx+4 and x+w>bx-4 and y<by+4 and y+h>by-4:return True
    for m in fleet.machines:
        nearest=np.array([np.clip(m.position[0],x,x+w),np.clip(m.position[1],y,y+h)])
        if np.linalg.norm(nearest-m.position[:2])<=m.radius+.3:return True
    return False


def require_unloaded(sim):
    fleet=fleet_for(sim)
    if fleet is not None and fleet.carried_mass_kg>1e-8:
        raise ValueError('Сначала выгрузите самосвалы; работающая техника и груз сохраняются в .atsstate')
