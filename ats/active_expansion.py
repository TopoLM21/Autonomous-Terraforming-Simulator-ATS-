"""Join neighbouring DEM data to the one active physical heightfield.

The old window is wholly retained at its original resolution. Moving the local
origin is a coordinate change at a stationary boundary, never transport of gear.
Preparation is detached; disk, geometry and busy-state errors leave actors alone.
"""
from copy import copy,deepcopy
import math
import numpy as np


def _point(value,shift):
    if value is None:return None
    result=np.asarray(value,dtype=float).copy();result[:2]+=shift
    return result if isinstance(value,np.ndarray) else tuple(map(float,result))


def _rect(value,shift):
    if value is None:return None
    return (float(value[0]+shift[0]),float(value[1]+shift[1]),*value[2:])


def _stationary(site):
    """No half-finished action may retain coordinates in the obsolete frame."""
    for u in site.project.units:
        s=u.simulation;w=s.logistics
        if s.pending_equipment is not None or s.base.destination is not None or s.relocation_pending is not None:
            raise ValueError('Перед присоединением области завершите изменение оборудования и движение базы')
        if any(not d.grounded or d.trip is not None or abs(d.cargo)>1e-8 or d.destination is not None or d.detour is not None for d in s.drones):
            raise ValueError('Перед сменой начала карты посадите все дроны и завершите их рейсы без груза')
        if w:
            service=w.battery_service;platforms=w.battery_platforms
            if (w.active or w.group or w.base_move_job or w.tool_changes or w.tool_targets or w.dock_owner is not None
                    or service.drone_swaps or service.robot_swap
                    or platforms and (platforms.handlers or platforms.returns)
                    or any(j.status=='ACTIVE' for j in w.jobs)):
                raise ValueError('Перед сменой начала карты завершите перевозки, замену батарей и обслуживание')
            if any(c.soil_kg>1e-8 for c in w.containers):
                raise ValueError('Перед сменой начала карты выгрузите контейнеры с грунтом')
            for e in w.excavators:
                if ((e.state.value!='IDLE' and (e.destination is not None or e.target is not None)) or e.bucket_kg>1e-8 or
                        e.state.value in ('MOVING_LOCAL','AIR_TRANSPORT','RIGGED_FOR_LIFT','SWAPPING_BATTERIES','FILLING_CONTAINER','DIGGING')):
                    raise ValueError('Перед сменой начала карты завершите работу, выгрузку и движение Кротов')
            array=w.solar_array
            if array and any(c.active is not None or c.carrying for c in array.crews):
                raise ValueError('Перед сменой начала карты завершите монтаж и перевозку солнечных секций')
    for fleet in site.conventional_fleets():
        if any(m.state not in ('PARKED','REFUELING','WAIT_FUEL') or m.path or m.destination is not None or m.cargo_kg>1e-8 for m in fleet.machines):
            raise ValueError('Перед сменой начала карты остановите обычную технику на стоянке и выгрузите груз')


def _configuration(old,origin,shift,n,cell,reference,current,soil,landscape):
    from .power_design import equipment_copy
    c=equipment_copy(old.config)
    for key in ('terrain','trench','plan','georeference'):setattr(c,key,deepcopy(getattr(old.config,key)))
    c.terrain.n=n;c.terrain.cell_size=cell;c.terrain.grab_radius=max(cell,c.terrain.grab_radius)
    c.terrain.dump_rect=_rect(c.terrain.dump_rect,shift)
    c.trench.x+=shift[0];c.trench.y+=shift[1];c.base.x+=shift[0];c.base.y+=shift[1]
    if c.plan:
        for segment in c.plan.segments:segment.points=[list(_point(p,shift)) for p in segment.points]
        channel=c.plan.channel
        if channel:
            channel['start']=list(_point(channel['start'],shift));channel['cell_size']=cell
            operations=channel.get('operations')
            if operations:
                operations['base_sites']=[[a,x+shift[0],y+shift[1]] for a,x,y in operations['base_sites']]
            if landscape is not None and landscape.channel_project is not None:
                from .channel import ChannelProject
                project=ChannelProject.from_data(landscape.channel_project)
                # A regional plan intentionally omitted remote edges. Include
                # those reached by this larger frame, while retaining explicit
                # user-edited local segment parameters rather than replacing them.
                canonical=(project.identity==channel.get('project_id') and all(
                    q.mode=='dig' and q.width==project.width_m and q.height==-project.depth_m
                    for q in c.plan.segments))
                if canonical:
                    fresh=project.plan(origin,side_m=n*cell)
                    fresh.channel={**channel,**fresh.channel}
                    fresh.channel['finished_spans']=deepcopy(channel.get('finished_spans',[]))
                    c.plan=fresh;channel=fresh.channel
    if c.georeference:
        c.georeference['origin_m']=origin.tolist()
        c.georeference['source_origin']=list(map(float,landscape.raster.to_source(*origin)))
        if c.georeference.get('channel'):c.georeference['channel']['custom_window']=True
        from .channel import marine_window
        if ('marine' in c.georeference or c.plan and c.plan.channel or
                landscape.raster.metadata['crs']!='LOCAL_METRES' or landscape.raster.metadata.get('ocean_boundary',False)):
            marine=marine_window(landscape.raster,origin,n*cell);c.georeference['marine']=marine
            if c.plan and c.plan.channel:c.plan.channel['marine']=marine
    c.initial_height=current.tolist();c.reference_height=reference.tolist();c.initial_soil=soil
    c.local_reference=c.georeference is None
    c.terrain.floor=min(c.terrain.floor,float(current.min())-1.)
    if c.plan and c.plan.channel and c.plan.channel.get('shore_reference')=='crest':
        from .shore import prepare
        prepare(landscape,c,origin,n*cell)
    fit=(c.plan.channel or {}).get('window_fit') if c.plan else None
    if fit is not None:
        fit.update(actual_side_m=n*cell,cell_m=cell)
        if landscape is not None and landscape.channel_project is not None:
            from .channel import ChannelProject
            from .channel_site import cross_section
            project=ChannelProject.from_data(landscape.channel_project)
            station=fit.get('cross_section',{}).get('station_m')
            if project.identity==c.plan.channel.get('project_id') and station is not None:
                fit['cross_section']=cross_section(project,landscape,c,origin,n*cell,station)
    c.validate();return c


def _fleet_preview(fleet,replacement,shift):
    """Validate a translated road and masks without touching the live fleet."""
    proposed=copy(fleet);proposed.config=deepcopy(fleet.config)
    proposed.config.work_rect=_rect(proposed.config.work_rect,shift)
    proposed.config.base_x+=shift[0];proposed.config.base_y+=shift[1]
    for key in ('ramp_entry_m','ramp_end_m'):
        if getattr(proposed.config,key,None) is not None:setattr(proposed.config,key,_point(getattr(proposed.config,key),shift))
    proposed.ramp=deepcopy(fleet.ramp)
    for key in ('entry_m','end_m'):proposed.ramp[key]=list(_point(proposed.ramp[key],shift))
    proposed.base_position=_point(fleet.base_position,shift)
    proposed.machines=[]
    for original in fleet.machines:
        m=copy(original)
        for key in ('position','home','destination','target','expose_target'):setattr(m,key,_point(getattr(m,key,None),shift))
        m.path=[_point(p,shift) for p in m.path]
        m.route_token=None;m.budget_cache=None
        if m.route_bounds is not None:
            m.route_bounds=tuple(np.asarray(m.route_bounds)+np.r_[shift,shift])
        proposed.machines.append(m)
    proposed.excavator=proposed.machines[0]
    proposed.rebind_terrain(replacement,offset_m=shift)
    return proposed


def _move_solar(array,shift):
    if array is None:return
    array.reserved_rects=[_rect(r,shift) for r in array.reserved_rects]
    for q in array.sections:
        q.rect=_rect(q.rect,shift)
        if hasattr(q,'route'):q.route=[list(_point(p,shift)) for p in q.route]
    for crew in array.crews:
        for key in ('home','position','mount','lift_origin'):setattr(crew,key,_point(getattr(crew,key),shift))
        crew.path=[list(_point(p,shift)) for p in crew.path]
        crew.payload_origins={key:_point(p,shift) for key,p in crew.payload_origins.items()}
    if hasattr(array,'storage_mounts'):array.storage_mounts={key:_point(p,shift) for key,p in array.storage_mounts.items()}
    # All pose/ground caches are derived. Keep finite actuation progress intact.
    for key in ('_audit_stamp','_own_geometry_key','_obstacle_key','_position_key','_position_stamp'):
        if hasattr(array,key):setattr(array,key,None)
    if hasattr(array,'_panel_pose_cache'):array._panel_pose_cache={}


def _prepare_dust(sim,config,shift):
    """Allocate the derived display raster before the physical commit."""
    from .dust import Dust
    prototype=copy(sim.config);prototype.terrain=config.terrain
    old=sim.dust;fresh=Dust(prototype)
    pixels=np.rint(shift/fresh.cell).astype(int)
    if abs(old.cell-fresh.cell)<1e-9 and np.all(np.abs(pixels*fresh.cell-shift)<1e-8):
        x,y=pixels;rows,cols=old.field.shape;fresh.field[y:y+rows,x:x+cols]=old.field
    return fresh


def _apply_actor_frame(sim,shift,old_origin):
    sim.base.position=_point(sim.base.position,shift)
    sim.config.base.x,sim.config.base.y=map(float,sim.base.position[:2])
    sim.config.trench.x+=shift[0];sim.config.trench.y+=shift[1]
    for d in sim.drones:
        d.position=_point(d.position,shift)
        for key in ('destination','detour'):setattr(d,key,_point(getattr(d,key),shift))
    sim._starts={key:_point(value,shift) for key,value in sim._starts.items()}
    sim.laser_links=[]
    if sim.terrain.front is not None:
        sim.terrain.front.legacy_pickups=[_point(p,shift) for p in getattr(sim.terrain.front,'legacy_pickups',[])]
    w=sim.logistics
    if w:
        w.rack=_point(w.rack,shift);w.field_rect=_rect(w.field_rect,shift)
        for item in w.inventory.items.values():item.position=_point(item.position,shift)
        for e in w.excavators:
            for key in ('position','destination','target','rack_position'):
                if hasattr(e,key):setattr(e,key,_point(getattr(e,key),shift))
            e._drive_cache=None
        for job in [*(w.archived_jobs or []),*w.jobs]:
            if job.status=='QUEUED':
                # A queued request has not moved its reserved equipment. Keep
                # its ID/status/reservation and future physical endpoints live.
                job.origin=_point(job.origin,shift);job.destination=_point(job.destination,shift)
                if hasattr(job,'region_origin_m'):job.region_origin_m=(old_origin-shift).tolist()
            elif not hasattr(job,'region_origin_m'):job.region_origin_m=old_origin.tolist()
        _move_solar(w.solar_array,shift)
        if w.battery_platforms:w.battery_platforms._rack_schema=None


def enlarge(site,side_m,*,origin_m=None,preserve_resolution=False):
    """Prepare and join neighbours; actor mutation occurs only after validation."""
    p=site.project;old=site.core;old_side=old.height.shape[0]*old.cell;side=float(side_m)
    if p.transition is not None:raise ValueError('Сначала завершите или отмените региональный переезд')
    if not math.isfinite(side) or not old_side<=side<=2048:raise ValueError('Присоединённая карта должна сохранять старое окно и иметь сторону до 2048 м')
    from .maps import workspace_cell
    cell=old.cell if preserve_resolution else workspace_cell(side,old.cell)
    n=round(side/cell)
    if n>2048 or abs(n*cell-side)>1e-8:raise ValueError('Присоединение: не более 2048 ячеек по стороне, размер кратен текущей ячейке')
    if cell!=old.cell:raise ValueError('Присоединение соседних областей требует сохранения разрешения')
    old_origin=np.asarray((old.config.georeference or {}).get('origin_m',[0.,0.]),dtype=float)
    origin=old_origin.copy() if origin_m is None else np.asarray(origin_m,dtype=float)
    if origin.shape!=(2,) or not np.all(np.isfinite(origin)):raise ValueError('Нужны конечные координаты начала карты X/Y')
    origin=old_origin+np.rint((origin-old_origin)/cell)*cell
    shift=old_origin-origin
    if np.any(shift< -1e-8) or np.any(shift+old_side>side+1e-8):raise ValueError('Присоединённое окно должно целиком сохранять прежнюю расчётную область')
    if side==old_side and np.all(np.abs(shift)<1e-8):return
    landscape=p.landscape
    if np.any(shift) and (landscape is None or old.config.georeference is None):raise ValueError('Для присоединения слева или снизу нужна региональная привязка DEM')
    if np.any(shift):_stationary(site)
    if landscape is not None:
        if np.any(origin<0) or np.any(origin+side>landscape.raster.extent_m+np.array([1e-8,1e-8])):
            raise ValueError('Присоединённая область выходит за границы исходного DEM')
        # Include the latest live material columns, even in partially shared
        # persistent tiles, without committing anything to the real region.
        preview=landscape.snapshot()
        preview.commit_window(old_origin,old.height,cell_size=old.cell,soil=old.soil)
        prepared=landscape.prepared_window(origin,n,cell)
        reference,current=preview.read_window(origin,n,cell_size=cell,prepared=prepared)
        soil=preview.soil_window(origin,n,cell,prepared=prepared)
    else:
        reference=np.zeros((n,n),dtype=float);current=reference.copy()
        soil=old.soil.export() if old.soil is not None else None
    dx,dy=np.rint(shift/cell).astype(int);rows=old.height.shape[0]
    overlap=np.s_[dy:dy+rows,dx:dx+rows]
    reference[overlap]=old.reference;current[overlap]=old.height
    baseline=current.copy();baseline[overlap]=old._initial_surface
    c=_configuration(old,origin,shift,n,cell,reference,current,soil,landscape)
    from .terrain import Terrain
    replacement=Terrain(c)
    # Previously planned/damaged ground is authoritative; new neighbours gain
    # the same design law without regenerating the old target from today's cut.
    for key in ('target','segment_labels','excavation','fill_mask','bottom','core_mask','dump_mask','dump_cap','protected','design_work','dump_allowed_mask','_relax_dirty'):
        before=getattr(old,key,None);after=getattr(replacement,key,None)
        if isinstance(before,np.ndarray) and before.shape==old.height.shape:
            if not isinstance(after,np.ndarray) or after.shape!=replacement.height.shape:
                after=np.zeros(replacement.height.shape,dtype=before.dtype);setattr(replacement,key,after)
            after[overlap]=before
    if c.plan and old.config.plan:
        names=[q.name for q in c.plan.segments]
        if len(set(names))==len(names):
            labels=old.segment_labels
            for number,segment in enumerate(old.config.plan.segments,1):
                if segment.name in names:replacement.segment_labels[overlap][labels==number]=names.index(segment.name)+1
    replacement._initial_surface=baseline;replacement.initial_excavation=old.initial_excavation
    replacement.revision=old.revision+1;replacement._dump_content_revision=getattr(old,'_dump_content_revision',0)+1
    replacement._set_dig_indices();replacement._fill_indices=np.flatnonzero(replacement.fill_mask)
    replacement._dump_indices=np.flatnonzero(replacement.dump_mask&(replacement.dump_cap>replacement.reference))
    if replacement.soil is not None:
        replacement._initial_soil_mass=replacement.soil.mass_delta-(old.soil.mass_delta-old._initial_soil_mass)
        replacement._initial_soil_volume=replacement.soil.volume_delta_by_material-(old.soil.volume_delta_by_material-old._initial_soil_volume)
    # Newly connected borders participate in the next conservative relaxation.
    dirty=replacement._relax_dirty
    for x in (dx,dx+rows-1):dirty[max(0,dy-1):min(n,dy+rows+1),max(0,x-1):min(n,x+2)]=True
    for y in (dy,dy+rows-1):dirty[max(0,y-1):min(n,y+2),max(0,dx-1):min(n,dx+rows+1)]=True
    yy,xx=np.nonzero(dirty)
    replacement._dirty_bounds=(int(yy.min()),int(yy.max()+1),int(xx.min()),int(xx.max()+1)) if len(yy) else None
    if old.front is not None and replacement.front is not None:
        replacement.front.refresh_design()
        if old.front.sectors and replacement.front.sectors:replacement.front._set_sector(old.front.global_sector_start)
        for key in ('stationed','moving','followers_launched','support_guard_end'):
            if hasattr(old.front,key):setattr(replacement.front,key,getattr(old.front,key))
        replacement.front.legacy_pickups=[_point(p,shift) for p in getattr(old.front,'legacy_pickups',[])]
    replacement._clearance_surface=max(old.clearance_surface,replacement._clearance_surface)
    previews=[(fleet,_fleet_preview(fleet,replacement,shift)) for fleet in site.conventional_fleets()]
    dusts=[_prepare_dust(u.simulation,c,shift) for u in p.units]
    ownership=np.ones((n,n),dtype=np.int16);ownership[overlap]=site.owners
    # Everything above is detached. No disk writes, actor moves, purchases or
    # model-time advances precede this commit boundary between physics steps.
    for u,dust in zip(p.units,dusts):
        s=u.simulation
        s.config.terrain=deepcopy(c.terrain);s.config.plan=deepcopy(c.plan)
        s.config.initial_height=c.initial_height;s.config.reference_height=c.reference_height;s.config.initial_soil=c.initial_soil
        s.config.local_reference=c.local_reference;s.config.georeference=deepcopy(c.georeference)
        if np.any(shift):_apply_actor_frame(s,shift,old_origin)
        s.dust=dust
        s.terrain.core=replacement;s._terrain_active=True;s._capability_key=None;s._latitude_position=None
        if s.logistics:
            for e in s.logistics.excavators:e._drive_cache=None
    for fleet,proposed in previews:
        fleet.config=proposed.config;fleet.base_position=proposed.base_position;fleet.ramp=proposed.ramp
        # Keep machine identities that UI/debug callers may already reference.
        for original,m in zip(fleet.machines,proposed.machines):original.__dict__.update(m.__dict__)
        for key in ('terrain','work_mask','mask','ramp_surface','ramp_mask','access_mask','_marine','_marine_stamp',
                    '_ground_stage_released','_completed_revision'):
            if hasattr(proposed,key):setattr(fleet,key,getattr(proposed,key))
    site.core=replacement;site.owners=ownership;site._mass_revision=-1
    replacement.config=p.units[0].simulation.config;replacement._site_owner=p.units[0].simulation
    if previews:
        from .conventional_integration import reservation_mask
        fleet=previews[0][0]
        replacement._conventional_mask=(np.zeros(replacement.height.shape,dtype=bool) if fleet.ground_stage_drained else
                                        reservation_mask(replacement,fleet.config,[u.simulation for u in p.units]))
    site._starts={};site._airspace={};site.invalidate_geometry();site.partition()
    from .simulation import MissionState
    for u in p.units:
        s=u.simulation;front=s.terrain.front
        # A larger physical domain can resolve these two window-related
        # diagnoses. It grants only a fresh check: the normal range, energy,
        # footprint and route rules run again on the next simulation step.
        if (s.state==MissionState.BLOCKED and front is not None and front.sectors and
                s.reason.partition(':')[0] in ('BLOCKED_SECTOR_RANGE','BLOCKED_SECTOR_SITE')):
            s.state=MissionState.RUNNING;s.reason='';s.base_move_wait=''
            s._sector_range_key=None;s._work_window_cache=None;s._relocation_attempt_time=-math.inf
    p.active.simulation.event(f'Присоединены соседние области: карта {side:g} м, ячейка {cell:g} м; грунт, техника и время сохранены')
