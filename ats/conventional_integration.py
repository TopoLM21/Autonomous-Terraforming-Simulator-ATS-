"""Attach a separate diesel fleet to the existing single physical heightfield."""
import numpy as np


def install(sim,config):
    from .conventional import ConventionalFleet
    from .site_ground import equipment_mask,rectangle_patch
    if getattr(sim,'conventional',None) is not None:
        raise ValueError('Техника уже приобретена: используйте обслуживание или новый проект')
    if sim.shared_site is not None:
        raise ValueError('Первый этап обычной техники: отдельная площадка или один автономный комплекс')
    if sim.base.destination is not None or sim.relocation_pending is not None:
        raise ValueError('Дождитесь завершения переезда базы')
    if any(not d.grounded or d.trip is not None for d in sim.drones):
        raise ValueError('Перед разметкой посадите автономный флот и завершите рейсы')
    if sim.logistics and (sim.logistics.group or sim.logistics.active or
            any(e.bucket_kg or e.active_container for e in sim.logistics.excavators)):
        raise ValueError('Завершите текущие циклы Кротов и грузовые операции')
    config.validate()
    t=sim.terrain
    patch=rectangle_patch(t,config.work_rect)
    if patch is None:raise ValueError('Рабочий участок должен полностью помещаться на карте')
    if sim.drones and np.any(equipment_mask(t,[sim])[patch]):
        raise ValueError('Рабочий участок пересекает установленную автономную технику')
    fleet=ConventionalFleet(sim,config)
    sim.conventional=fleet
    # Leave physical design/volume intact. Only new autonomous target choices
    # exclude the separate contractor's patch and a grabber-width boundary.
    x,y,w,h=config.work_rect
    margin=max([d.config.grab_radius for d in sim.drones]+[t.cell])
    if sim.logistics:margin=max(margin,sim.logistics.config.reach_m)
    t._conventional_mask=((t.x>=x-margin)&(t.x<=x+w+margin)&
                          (t.y>=y-margin)&(t.y<=y+h+margin))
    sim.event('Обычная техника приобретена: отдельные топливо, зарплаты и смета')
    return fleet


def standalone_report(sim):
    from .simulation import elapsed_text
    fleet=getattr(sim,'conventional',None);t=sim.terrain
    r=dict(state=sim.state.value,reason=sim.reason,time_seconds=sim.time,
        elapsed_display=elapsed_text(sim.time),date=sim.date_text,hour=sim.hour,
        day_of_year=sim.day_of_year,latitude_deg=sim.config.base.latitude_deg,
        drone_count=0,drones=[],autonomous_equipment_cost_usd=0.,
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
    fleet=getattr(host.sim,'conventional',None)
    if not hasattr(host,'ground_vehicle_markers'):
        if fleet is None:return
        host.ground_vehicle_markers=pg.ScatterPlotItem(size=13,pen=pg.mkPen('#17202b',width=2))
        host.ground_work_outline=pg.PlotDataItem(pen=pg.mkPen('#ffcf66',width=2))
        host.ground_routes=pg.PlotDataItem(pen=pg.mkPen('#f5d69a',width=1))
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
    owner=getattr(terrain,'_site_owner',None);fleet=getattr(owner,'conventional',None)
    if fleet is None:return False
    x,y,w,h=rect;bx,by=fleet.base_position[:2]
    if x<bx+4 and x+w>bx-4 and y<by+4 and y+h>by-4:return True
    for m in fleet.machines:
        nearest=np.array([np.clip(m.position[0],x,x+w),np.clip(m.position[1],y,y+h)])
        if np.linalg.norm(nearest-m.position[:2])<=m.radius+.3:return True
    return False


def require_unloaded(sim):
    fleet=getattr(sim,'conventional',None)
    if fleet is not None and fleet.carried_mass_kg>1e-8:
        raise ValueError('Сначала выгрузите самосвалы; работающая техника и груз сохраняются в .atsstate')
