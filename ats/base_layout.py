"""Conditional modular crawler engineering, in metres, kg, Pa and joules.

The parameters are editable design assumptions, not a manufacturer's rating.
Geometry, chassis mass and contact pressure follow the installed equipment.
"""
from copy import deepcopy
import math
import numpy as np
from .equipment import KWH,station_offset

G=9.80665
DEFAULTS=dict(platform_length_m=6.,platform_width_m=3.,platform_height_m=1.8,
    platform_payload_kg=5000.,frame_base_kg=400.,frame_kg_m2=55.,drive_mass_kg=180.,
    drive_kw=12.,platform_gap_m=1.,track_contact_fraction=.8,max_pressure_kpa=30.,
    min_track_width_m=.45,battery_litres_kwh=8.,panel_folded_thickness_m=.04,
    packing_fraction=.75,other_equipment_kg_m3=300.,drone_stowed_m3=2.,
    landing_pad_m2=9.,drive_efficiency=.8,rolling_resistance=.10,
    traction_coefficient=.55,bearing_capacity_kpa=60.,control_power_w=25.)
DEFAULTS['heading_deg']=0.
DEFAULTS['auto_heading']=1.


def parameters(value=None):
    p={**DEFAULTS,**(value or {})}
    if set(p)!=set(DEFAULTS) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for k,v in p.items() if k not in ('heading_deg','auto_heading')) or not math.isfinite(p['heading_deg']) or p['auto_heading'] not in (0,1):
        raise ValueError('Неверные параметры платформ')
    if (p['platform_width_m']<2 or p['platform_length_m']<3 or p['platform_height_m']<.5 or
            not 0<p['packing_fraction']<=1 or not 0<p['drive_efficiency']<=1 or
            not 0<p['track_contact_fraction']<=1 or p['min_track_width_m']>p['platform_width_m']*.38):
        raise ValueError('Неверная геометрия или КПД платформ')
    return p


def equipment_totals(config,world=None,panels=None,batteries=None,extra=()):
    """Maximum packed load, including machines currently working off the base."""
    c=config;b=c.base.equipment;log=c.logistics
    panel_recipes=b.panels if panels is None else panels
    battery_recipes=b.batteries if batteries is None else batteries
    panel_mass=sum(p.quantity*p.mass_kg for p in panel_recipes)
    battery_mass=sum(p.quantity*p.mass_kg for p in battery_recipes)
    capacity=sum(p.quantity*p.capacity_kwh for p in battery_recipes)
    area=sum(p.quantity*p.area_m2 for p in panel_recipes)
    if world is None:
        mass=b.mass-b.chassis_mass_kg
        mass-=sum(p.quantity*p.mass_kg for p in b.panels+b.batteries)
        mass+=panel_mass+battery_mass+sum(d.equipment.mass for d in c.fleet.models)
        capacity+=sum(d.battery/KWH for d in c.fleet.models)
        battery_mass+=sum(sum(p.quantity*p.mass_kg for p in d.equipment.batteries) for d in c.fleet.models)
        if log:
            mass+=sum(sum(p.quantity*p.mass_kg for p in d.equipment.batteries) for d in c.fleet.models)*log.drone_spare_sets
            battery_mass+=sum(sum(p.quantity*p.mass_kg for p in d.equipment.batteries) for d in c.fleet.models)*log.drone_spare_sets
            capacity+=sum(d.battery/KWH for d in c.fleet.models)*log.drone_spare_sets
            modules=sum(sum(p.quantity for p in d.equipment.batteries) for d in c.fleet.models)*(1+log.drone_spare_sets)
            if log.excavator_enabled:
                mass+=log.excavator_dry_mass_kg+log.excavator_battery_count*log.pack_mass_kg
                capacity+=log.excavator_battery_count*log.pack_capacity_kwh+log.buffer_kwh
                battery_mass+=log.excavator_battery_count*log.pack_mass_kg
                modules+=log.excavator_battery_count
            mass+=log.container_count*log.container_tare_kg+len(c.fleet.models)*(2+log.hoist_mass_kg)+log.rigging_mass_kg
            if log.individual_chargers:mass+=modules*log.charger_slot_mass_kg
    else:
        from .inventory import BatteryModule
        items=list(world.inventory.items.values())
        # Base batteries are replaced as a recipe; mobile packs keep real IDs.
        mass=sum(o.mass_kg for o in items if o.kind not in ('CHASSIS','PLATFORM_CHASSIS','SOLAR_PANEL','SOLAR_FRAME','SOLAR_INSTALLER','BATTERY_PLATFORM_HARDWARE') and
                 o.id not in world.base_battery_ids)
        mass+=panel_mass+battery_mass+sum(d.config.equipment.frame_mass_kg for d in world.sim.drones)
        if world.excavator:mass+=log.excavator_dry_mass_kg
        capacity+=sum(o.capacity_j/KWH for o in items if isinstance(o,BatteryModule) and o.id not in world.base_battery_ids)
        battery_mass+=sum(o.mass_kg for o in items if isinstance(o,BatteryModule) and o.id not in world.base_battery_ids)
        if world.excavator:capacity+=log.buffer_kwh
    solar_volume=None;solar_frame_mass=0.
    retained_pv=bool(world and world.sim.base.config.layout is not None and any(a.get('role')=='PV' for a in world.sim.base.layout_spec['platforms']))
    if log and getattr(log,'self_solar',None) is not None and (area>20 or retained_pv):
        from .self_solar import recipe
        groups,_=recipe(panel_recipes,log.self_solar, [a for a in world.sim.base.layout_spec['platforms'] if a.get('role')=='PV'] if world and world.sim.base.config.layout is not None else ())
        solar_frame_mass=sum(g['hardware_kg'] for g in groups);mass+=solar_frame_mass
    elif log and log.solar is not None and area>20:
        from .solar_array import recipe,parameters
        _,crews,solar_frame_mass,solar_volume=recipe(panel_recipes,log.solar)
        mass+=solar_frame_mass+crews*parameters(log.solar)['installer_mass_kg']
    for entry in extra:
        mass_kg,kwh=entry[:2];mass+=mass_kg;capacity+=kwh;battery_mass+=mass_kg
    return dict(payload_kg=float(mass),battery_kwh=float(capacity),panel_area_m2=float(area),
                panel_mass_kg=panel_mass,battery_mass_kg=battery_mass,drones=len(c.fleet.models),pads=b.landing_slots,
                solar_packed_m3=solar_volume,solar_frame_mass_kg=solar_frame_mass)


def solve_layout(config,world=None,panels=None,batteries=None,extra=(),base_records=None):
    p=parameters(config.base.layout);totals=equipment_totals(config,world,panels,batteries,extra)
    from .self_solar import enabled as self_enabled,recipe,roof_recipes,compose
    retained_pv=bool(world and world.sim.base.config.layout is not None and any(a.get('role')=='PV' for a in world.sim.base.layout_spec['platforms']))
    if self_enabled(config) and (totals['panel_area_m2']>20 or retained_pv):
        recipes=config.base.equipment.panels if panels is None else panels
        existing=[a for a in world.sim.base.layout_spec['platforms'] if a.get('role')=='PV'] if world and world.sim.base.config.layout is not None else ()
        groups,_=recipe(recipes,config.logistics.self_solar,existing)
        core=dict(totals);external_mass=sum(g['panel_mass_kg']+g['hardware_kg'] for g in groups)
        roof=roof_recipes(recipes);core.update(payload_kg=totals['payload_kg']-external_mass,panel_area_m2=sum(q.quantity*q.area_m2 for q in roof),panel_mass_kg=sum(q.quantity*q.mass_kg for q in roof),solar_frame_mass_kg=0.,solar_packed_m3=None)
        return compose(config,_solve_layout(config,p,core,world,roof,batteries,extra,base_records),groups,world)
    return _solve_layout(config,p,totals,world,panels,batteries,extra,base_records)


def _solve_layout(config,p,totals,world,panels,batteries,extra,base_records):
    from .battery_platforms import enabled,solve
    if enabled(config):return solve(config,p,totals,world,panels,batteries,extra,base_records)
    length,width,height=p['platform_length_m'],p['platform_width_m'],p['platform_height_m']
    deck=length*width;dry=p['frame_base_kg']+deck*p['frame_kg_m2']+p['drive_mass_kg']
    volume=(totals['battery_kwh']*p['battery_litres_kwh']/1000+
            (totals['solar_packed_m3'] if totals['solar_packed_m3'] is not None else totals['panel_area_m2']*p['panel_folded_thickness_m'])+
            max(0.,totals['payload_kg']-totals['panel_mass_kg']-totals['battery_mass_kg']-totals['solar_frame_mass_kg'])/p['other_equipment_kg_m3']+
            totals['drones']*p['drone_stowed_m3'])
    required_area=max(volume/(height*p['packing_fraction']),totals['pads']*p['landing_pad_m2'])
    count=max(1,math.ceil(totals['payload_kg']/p['platform_payload_kg']),math.ceil(required_area/deck))
    previous=getattr(getattr(getattr(world,'sim',None),'base',None),'layout_spec',{})
    retained=[a for a in previous.get('platforms',()) if a.get('role')!='PV'] if previous.get('self_solar') else []
    count=max(count,len(retained))
    while count<=64:
        gross=dry+totals['payload_kg']/count
        track_width=max(p['min_track_width_m'],gross*G/(p['max_pressure_kpa']*1000*2*length*p['track_contact_fraction']))
        if track_width<=width*.38:break
        count+=1
    if count>64:raise ValueError('Комплектации требуется более 64 платформ')
    platforms=[]
    for i in range(count):
        offset=station_offset(i,length+p['platform_gap_m'])
        identifier=retained[i]['id'] if i<len(retained) else max((a['id'] for a in previous.get('platforms',())),default=0)+i-len(retained)+1 if retained else i+1
        xy=retained[i]['offset_m'] if i<len(retained) else [0.,offset]
        platforms.append(dict(id=identifier,offset_m=list(xy),dimensions_m=[width,length,height],
            payload_kg=totals['payload_kg']/count,dry_mass_kg=dry,gross_mass_kg=gross,
            track_width_m=track_width,track_contact_length_m=length*p['track_contact_fraction'],
            contact_area_m2=2*track_width*length*p['track_contact_fraction'],
            pressure_kpa=gross*G/(2*track_width*length*p['track_contact_fraction'])/1000,
            drive_power_kw=p['drive_kw'],centre_of_gravity_m=[0.,offset,(dry*.45+totals['payload_kg']/count*height*.45)/gross]))
    lower=min(a['offset_m'][1]-length/2 for a in platforms);upper=max(a['offset_m'][1]+length/2 for a in platforms)
    return dict(parameters=p,platforms=platforms,platform_count=count,chassis_mass_kg=count*dry,
        packed_payload_kg=totals['payload_kg'],packed_gross_mass_kg=count*dry+totals['payload_kg'],
        packed_volume_m3=volume,dimensions_m=[width,upper-lower,height],bounds_m=[-width/2,lower,width,upper-lower],
        centre_of_gravity_m=[*rotate_offset(p,[0.,sum(a['centre_of_gravity_m'][1] for a in platforms)/count]),platforms[0]['centre_of_gravity_m'][2]],
        contact_area_m2=sum(a['contact_area_m2'] for a in platforms),max_pressure_kpa=max(a['pressure_kpa'] for a in platforms),
        drive_power_kw=count*p['drive_kw'],control_power_w=count*p['control_power_w'])


def configure_layout(config,extra=()):
    from .self_solar import enabled
    if config.base.layout is None and enabled(config) and config.base.equipment.panel_area>20:config.base.layout={}
    if config.base.layout is None:return None
    config.base.layout=parameters(config.base.layout)
    if config.base.layout['auto_heading']:config.base.layout['heading_deg']=plan_heading(config)
    spec=solve_layout(config,extra=extra)
    config.base.equipment.chassis_mass_kg=spec['chassis_mass_kg']
    config.base.radius=spec.get('hub_radius_m',spec['dimensions_m'][0]/2);config.base.roof_height=spec.get('deck_height_m',spec['dimensions_m'][2])
    config.base.idle_power=spec['control_power_w']
    config.base.equipment.resolve(config.base,config.fleet)
    return spec


def plan_heading(config):
    if config.plan is not None:
        for segment in config.plan.segments:
            if len(segment.points)>1:
                delta=np.asarray(segment.points[-1])-segment.points[0]
                if np.linalg.norm(delta)>1e-9:return math.degrees(math.atan2(delta[1],delta[0]))-90.
    return -90.


def rotate_offset(settings,offset):
    angle=math.radians((settings or {}).get('heading_deg',0.));x,y=offset
    return np.array([math.cos(angle)*x-math.sin(angle)*y,math.sin(angle)*x+math.cos(angle)*y])


def safe_layout_route(base,target,terrain,offset=(0,0)):
    spec=base.layout_spec;p=spec['parameters'];side=terrain.height.shape[0]*terrain.cell
    offset=rotate_offset(p,offset);start=base.position[:2]+offset;end=np.asarray(target)+offset
    distance=float(np.linalg.norm(end-start));steps=max(2,math.ceil(distance/max(.25,terrain.cell))+1)
    previous=None
    for point in np.linspace(start,end,steps):
        condition=support_report(base,terrain,point=point,spec=spec)
        if not condition['safe']:return False
        for x,y,w,h in layout_footprints(spec,point):
            if min(x,y)<terrain.cell or max(x+w,y+h)>side-terrain.cell:return False
            patch=(slice(max(0,int(y/terrain.cell)),min(terrain.height.shape[0],math.ceil((y+h)/terrain.cell))),
                   slice(max(0,int(x/terrain.cell)),min(terrain.height.shape[1],math.ceil((x+w)/terrain.cell))))
            blocked=terrain.design_work[patch]|terrain.dump_mask[patch]
            if terrain.protected is not None:blocked=blocked|terrain.protected[patch]
            if np.any(blocked):return False
            from .site_ground import footprint_safe
            if not footprint_safe(terrain,(x,y,w,h),grade_deg=math.degrees(math.atan(base.config.max_slope)),relief_m=float('inf')):return False
            site=getattr(terrain,'shared_site',None)
            if site is not None and site.ground_occupied(getattr(terrain,'owner_id',None),(x,y,w,h)):return False
            z=[terrain.sample(px,py) for px,py in ((x,y),(x+w,y),(x,y+h),(x+w,y+h))]
            sx=max(abs(z[1]-z[0]),abs(z[3]-z[2]))/w
            sy=max(abs(z[2]-z[0]),abs(z[3]-z[1]))/h
            if (math.hypot(sx,sy)>base.config.max_slope or sx>condition['tipping_slope_x'] or sy>condition['tipping_slope_y']):return False
        z=terrain.sample(*point)
        if previous is not None:
            slope=abs(z-previous[2])/max(1e-12,float(np.linalg.norm(point-previous[:2])))
            if slope>min(base.config.max_slope,condition['traction']-condition['rolling_resistance']):return False
        previous=np.r_[point,z]
    return True


def ground_properties(terrain,point,settings):
    profile=terrain.soil.profile if terrain.soil is not None else None
    if profile is None:return settings['bearing_capacity_kpa'],settings['traction_coefficient'],settings['rolling_resistance']
    n=terrain.height.shape[0];x,y=np.clip(np.asarray(point[:2])/terrain.cell,0,n-1).astype(int)
    m=profile.materials[int(terrain.soil.surface_ids(np.array([y*n+x]))[0])]
    return m.bearing_kpa,m.traction,m.rolling_resistance


def support_report(base,terrain,drones=(),point=None,spec=None):
    spec=spec or getattr(base,'layout_spec',None) or solve_layout(terrain.config)
    p=spec['parameters'];point=base.position[:2] if point is None else np.asarray(point)
    actual=base.config.mass+sum(d.config.empty_mass+d.cargo for d in drones if d.docked)
    scale=actual/spec['packed_gross_mass_kg'];rows=[]
    if 'bindings' not in spec:
        # Historical missions keep their original aggregate reactions and cost.
        for module in spec['platforms']:
            centre=point+rotate_offset(p,module['offset_m']);bearing,traction,rolling=ground_properties(terrain,centre,p)
            pressure=module['pressure_kpa']*scale
            rows.append(dict(id=module['id'],pressure_kpa=pressure,bearing_kpa=bearing,
                traction=traction,rolling_resistance=rolling,safe=pressure<=bearing))
        return dict(platforms=rows,max_pressure_kpa=max(r['pressure_kpa'] for r in rows),
            minimum_bearing_kpa=min(r['bearing_kpa'] for r in rows),safe=all(r['safe'] for r in rows),
            rolling_resistance=max(r['rolling_resistance'] for r in rows),traction=min(r['traction'] for r in rows),
            tipping_slope_x=p['platform_width_m']/2/max(.01,spec['centre_of_gravity_m'][2])/1.3,
            tipping_slope_y=p['platform_length_m']/2/max(.01,spec['centre_of_gravity_m'][2])/1.3)
    from .battery_platforms import owner_world
    world=owner_world(base,terrain)
    manager=getattr(world,'battery_platforms',None)
    loads=manager.loads() if manager and spec is base.layout_spec else None
    for module in spec['platforms']:
        centre=point+rotate_offset(p,module['offset_m']);bearing,traction,rolling=ground_properties(terrain,centre,p)
        mass=loads[module['id']]['mass_kg'] if loads else module['gross_mass_kg']*scale
        cg=loads[module['id']]['centre_of_gravity_m'] if loads else [module['centre_of_gravity_m'][0]-module['offset_m'][0],module['centre_of_gravity_m'][1]-module['offset_m'][1],module['centre_of_gravity_m'][2]]
        pressure=mass*G/module['contact_area_m2']/1000
        loaded=mass-module['dry_mass_kg']<=p['platform_payload_kg']+1e-7
        balanced=abs(cg[0])<p['platform_width_m']/2/1.3 and abs(cg[1])<p['platform_length_m']/2/1.3
        rows.append(dict(id=module['id'],pressure_kpa=pressure,bearing_kpa=bearing,
            mass_kg=mass,centre_of_gravity_m=cg,overloaded=not loaded,balanced=balanced,
            traction=traction,rolling_resistance=rolling,safe=pressure<=bearing and loaded and balanced))
    return dict(platforms=rows,max_pressure_kpa=max(r['pressure_kpa'] for r in rows),
        minimum_bearing_kpa=min(r['bearing_kpa'] for r in rows),safe=all(r['safe'] for r in rows),
        rolling_resistance=max(r['rolling_resistance'] for r in rows),traction=min(r['traction'] for r in rows),
        tipping_slope_x=min((p['platform_width_m']/2-abs(r['centre_of_gravity_m'][0]))/max(.01,r['centre_of_gravity_m'][2])/1.3 for r in rows),
        tipping_slope_y=min((p['platform_length_m']/2-abs(r['centre_of_gravity_m'][1]))/max(.01,r['centre_of_gravity_m'][2])/1.3 for r in rows))


def layout_footprints(spec,point):
    for corners in platform_polygons(spec,point):
        lower=corners.min(axis=0);upper=corners.max(axis=0)
        yield float(lower[0]),float(lower[1]),float(upper[0]-lower[0]),float(upper[1]-lower[1])


def platform_polygons(spec,point):
    angle=math.radians(spec['parameters'].get('heading_deg',0.));cs,sn=math.cos(angle),math.sin(angle)
    rotation=np.array([[cs,sn],[-sn,cs]])
    for a in spec['platforms']:
        centre=np.asarray(a['offset_m']);w,l,_=a['dimensions_m']
        corners=np.array([[-w/2,-l/2],[w/2,-l/2],[w/2,l/2],[-w/2,l/2],[-w/2,-l/2]])+centre
        yield corners@rotation+np.asarray(point[:2])


def apply_live_layout(world,spec):
    """Frames are physical delivered equipment, with zero stored electricity."""
    from .inventory import Item,Location as L
    s=world.sim;eq=s.base.config.equipment
    old=sum(o.mass_kg for o in world.inventory.items.values() if o.kind in ('CHASSIS','PLATFORM_CHASSIS'))
    for key in [key for key,o in world.inventory.items.items() if o.kind=='PLATFORM_CHASSIS']:del world.inventory.items[key]
    world.inventory.items['base_chassis'].mass_kg=spec['platforms'][0]['dry_mass_kg']
    for a in spec['platforms'][1:]:
        key=f"platform_{a['id']:02}_chassis"
        xy=s.base.position[:2]+rotate_offset(spec['parameters'],a['offset_m'])
        position=(*map(float,xy),s.terrain.sample(*xy))
        world.inventory.add(Item(key,a['dry_mass_kg'],L.BASE_MOUNT,'INSTALLED','base',position=position,kind='PLATFORM_CHASSIS'))
    delta=spec['chassis_mass_kg']-old
    world.imported_mass_kg+=max(0.,delta);world.exported_mass_kg+=max(0.,-delta)
    eq.chassis_mass_kg=spec['chassis_mass_kg'];s.base.layout_spec=spec
    s.base.config.layout=dict(spec['parameters'])
    s.base.config.radius=spec.get('hub_radius_m',spec['dimensions_m'][0]/2);s.base.config.roof_height=spec.get('deck_height_m',spec['dimensions_m'][2])
    s.base.config.idle_power=spec['control_power_w']
    if getattr(world,'battery_platforms',None) is not None:world.battery_platforms.install(spec)
    if s.shared_site is not None:s.shared_site.invalidate_geometry()


def preview_live_layout(world,**changes):
    config=world.sim.config
    if world.sim.base.config.layout is None:
        if getattr(world.config,'self_solar',None) is None or sum(p.quantity*p.area_m2 for p in changes.get('panels',config.base.equipment.panels))<=20:return None
        from .power_design import equipment_copy
        config=equipment_copy(config);config.base.layout=parameters(dict(heading_deg=0.,auto_heading=0.,platform_height_m=config.base.roof_height))
        config.logistics.battery_platforms=None
    spec=solve_layout(config,world,**changes)
    if 'self_solar' in spec:
        from .self_solar import place
        spec=place(config,spec,world.sim.base.position,world.sim.terrain)
    base=deepcopy(world.sim.base);base.layout_spec=spec
    base.config.radius=spec.get('hub_radius_m',spec['dimensions_m'][0]/2)
    if not base.safe_path(base.position[:2],world.sim.terrain):raise ValueError('Новая компоновка платформ не помещается на текущей площадке')
    from .battery_platforms import deployment_safe
    if not deployment_safe(spec,base.position,world.sim.terrain,solar_polygons=world.solar_ground_polygons()):raise ValueError('Нет места для опор или сервисных проходов батарейных платформ')
    for x,y,w,h in layout_footprints(spec,base.position):
        for xx,yy,ww,hh in ([] if 'self_solar' in spec else world.solar_ground_rects()):
            if x<xx+ww and x+w>xx and y<yy+hh and y+h>yy:
                raise ValueError('Новая компоновка пересекает солнечную площадку; сначала сложите секции')
    return spec


def design_continuous(config,request=None):
    from .power_design import PowerRequest,calculate_power,sized_configuration,equipment_copy
    c=equipment_copy(config);request=deepcopy(request or PowerRequest());request.work_hours=24.;request.source='equipment'
    site={key:getattr(c,key) for key in ('initial_height','reference_height','initial_soil','plan','georeference','local_reference')}
    c.initial_height=c.reference_height=c.initial_soil=c.georeference=None;c.local_reference=False
    automatic=c.base.layout is None or 'heading_deg' not in c.base.layout
    c.base.layout=parameters(c.base.layout)
    if automatic:c.base.layout['heading_deg']=plan_heading(config)
    previous=None
    for _ in range(16):
        r=calculate_power(c,request);c=sized_configuration(c,r);spec=configure_layout(c)
        if spec.get('battery_platforms'):
            c.logistics.inverter_power_w=max(c.logistics.inverter_power_w,spec['drive_power_kw']*1000)
        key=(r['panel_count'],r['base_battery_count'],spec['platform_count'],c.base.idle_power)
        if key==previous:
            for name,value in site.items():setattr(c,name,value)
            r['base_layout']=spec;r['design_mode']='continuous';return c,r
        previous=key
    raise ValueError('Подбор круглосуточной комплектации не сошёлся')
