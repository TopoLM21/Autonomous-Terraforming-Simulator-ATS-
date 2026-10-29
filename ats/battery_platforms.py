"""Conditional tracked BESS and battery-service carriers; SI units throughout.

This is an engineering model, not a manufacturer's product specification.
Cells stay clamped in racks. Only ground supports and service aprons unfold.
The common DC bus retains the existing, single energy ledger.
"""
from dataclasses import dataclass
import math
import numpy as np
from .equipment import KWH,station_offset
from .inventory import BatteryModule,Item,Location as L


DEFAULTS=dict(bess_converter_kw=150.,service_bus_kw=100.,bess_hardware_kg=360.,service_hardware_kg=280.,
    bess_aux_w=100.,service_aux_w=25.,support_extension_m=.8,apron_width_m=1.,
    support_seconds=30.,apron_seconds=20.,connect_seconds=5.,actuator_w=200.,
    handling_w=200.,handling_speed_m_s=1.5,handling_min_seconds=5.,rack_clearance_m=.03)


def parameters(value=None):
    p={**DEFAULTS,**(value or {})}
    if (set(p)!=set(DEFAULTS) or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in p.values())
            or p['support_extension_m']>2 or p['apron_width_m']>3):
        raise ValueError('Неверные параметры батарейных платформ')
    return p


def enabled(config):
    return config.base.layout is not None and config.logistics is not None and config.logistics.battery_platforms is not None


def owner_world(base,terrain):
    site=getattr(terrain,'shared_site',None)
    if site:
        s=next((u.simulation for u in site.project.units if u.simulation.base is base),None)
    else:s=getattr(terrain,'_site_owner',None)
    return getattr(s,'logistics',None) if s is not None and s.base is base else None


def pack_dimensions(kwh,litres_kwh=8.):
    # 5 kWh cabinet module: 0.5 x 0.4 x 0.2 m. Other recipes scale in volume.
    scale=(kwh*litres_kwh/40.)**(1/3)
    return [.5*scale,.4*scale,.2*scale]


def converter_rating(config,count,recipe):
    """Candidate BESS rating in kW for the power designer's capacity search."""
    if not enabled(config):return float('inf')
    from .base_layout import parameters as layout_parameters
    p=layout_parameters(config.base.layout);q=parameters(config.logistics.battery_platforms)
    dims=pack_dimensions(recipe.capacity_kwh,p['battery_litres_kwh'])
    slots=len(_grid(p['platform_width_m']-.4,p['platform_length_m']-.4,p['platform_height_m']-.3,dims,q['rack_clearance_m']))
    slots=min(slots,math.floor((p['platform_payload_kg']-q['bess_hardware_kg'])/recipe.mass_kg))
    if slots<=0:raise ValueError('Батарея не помещается на энергетической платформе')
    return math.ceil(count/slots)*q['bess_converter_kw']


def deployment_polygons(spec,point):
    from .base_layout import rotate_offset
    q=spec.get('battery_platforms')
    if not q:return []
    polygons=[]
    for a in spec['platforms']:
        if a['role'] not in ('BESS','SERVICE'):continue
        w,l,_=a['dimensions_m'];ww=w+2*max(q['support_extension_m'],q['apron_width_m'])
        l+=spec['parameters']['platform_gap_m']  # bridges connect adjacent carrier aisles
        xy=np.asarray(point[:2])+rotate_offset(spec['parameters'],a['offset_m'])
        corners=np.array([rotate_offset(spec['parameters'],v) for v in ((-ww/2,-l/2),(ww/2,-l/2),(ww/2,l/2),(-ww/2,l/2))])+xy
        polygons.append(corners)
    return polygons


def deployment_rectangles(spec,point):
    from .site_ground import polygon_bounds
    return [polygon_bounds(p) for p in deployment_polygons(spec,point)]


def path_position(points,fraction):
    path=np.asarray(points,dtype=float);lengths=np.linalg.norm(np.diff(path,axis=0),axis=1);left=float(lengths.sum())*fraction
    for i,length in enumerate(lengths):
        if left<=length:return path[i]+(path[i+1]-path[i])*left/max(length,1e-12)
        left-=float(length)
    return path[-1].copy()


def deployment_safe(spec,point,terrain,*,solar_rects=(),solar_polygons=None):
    from .site_ground import footprint_safe,rectangle_polygon,polygon_bounds,polygons_overlap
    blocked=terrain.design_work|terrain.dump_mask
    if terrain.protected is not None:blocked=blocked|terrain.protected
    if solar_polygons is None:solar_polygons=[rectangle_polygon(r) for r in solar_rects]
    for polygon in deployment_polygons(spec,point):
        rect=polygon_bounds(polygon)
        if not footprint_safe(terrain,rect,blocked,grade_deg=8.,relief_m=.6):return False
        if any(polygons_overlap(polygon,other) for other in solar_polygons):return False
        site=getattr(terrain,'shared_site',None)
        if site and site.ground_occupied(getattr(terrain,'owner_id',None),rect):return False
    return True


def _grid(width,length,height,dimensions,gap=.03,centre=(0.,0.)):
    w,l,h=dimensions
    nx=int((width+gap)/(w+gap));ny=int((length+gap)/(l+gap));nz=int((height+gap)/(h+gap))
    points=[(ix,iy) for iy in range(ny) for ix in range(nx)]
    points.sort(key=lambda q:((q[0]-(nx-1)/2)**2+(q[1]-(ny-1)/2)**2,q[1],q[0]))
    return [[centre[0]+(ix-(nx-1)/2)*(w+gap),centre[1]+(iy-(ny-1)/2)*(l+gap),(iz+.5)*h+iz*gap]
            for iz in range(nz) for ix,iy in points]


def battery_records(config,world,batteries,extra,base_records=None):
    records=[];recipes=config.base.equipment.batteries if batteries is None else batteries
    if world is not None and base_records is None and batteries is not None:
        signature=lambda recipes:[(b.quantity,b.capacity_kwh,b.mass_kg) for b in recipes]
        if signature(batteries)==signature(world.sim.base.config.equipment.batteries):
            base_records=[world.inventory.items[id] for id in world.base_battery_ids]
    for recipe in recipes:
        for _ in range(recipe.quantity):
            i=len(records)
            id=world.base_battery_ids[i] if world is not None and batteries is None and i<len(world.base_battery_ids) else f'base_battery_{i+1:02}'
            records.append(dict(id=id,mass_kg=recipe.mass_kg,kwh=recipe.capacity_kwh,role='BESS',kind='BATTERY'))
    if base_records is not None:
        records=[dict(id=b.id,mass_kg=b.mass_kg,kwh=b.capacity_j/KWH,role='BESS',kind='BATTERY') for b in base_records]
    if world is not None:
        records.extend(dict(id=b.id,mass_kg=b.mass_kg,kwh=b.capacity_j/KWH,role='SERVICE',kind='BATTERY')
                       for b in world.inventory.select(BatteryModule) if b.compatibility!='base')
    else:
        c=config.logistics
        for i,d in enumerate(config.fleet.models):
            slot=0
            for recipe in d.equipment.batteries:
                for _ in range(recipe.quantity):
                    slot+=1
                    records.append(dict(id=f'drone_{i+1}_battery_{slot}',mass_kg=recipe.mass_kg,kwh=recipe.capacity_kwh,role='SERVICE',kind='BATTERY'))
                    for j in range(c.drone_spare_sets):
                        records.append(dict(id=f'drone_{i+1}_spare_{j+1}_{slot}',mass_kg=recipe.mass_kg,kwh=recipe.capacity_kwh,role='SERVICE',kind='BATTERY'))
        if c.excavator_enabled:
            records.extend(dict(id=f'excavator_battery_{i+1:02}',mass_kg=c.pack_mass_kg,kwh=c.pack_capacity_kwh,role='SERVICE',kind='BATTERY') for i in range(c.excavator_battery_count))
    # Initial-stock deliveries are resolved to real IDs in the live layout.
    for i,entry in enumerate(extra):
        mass,kwh=entry[:2];id=entry[2] if len(entry)>2 else f'initial_stock_reserved_{i}'
        if config.logistics.individual_chargers:mass-=config.logistics.charger_slot_mass_kg
        if kwh:records.append(dict(id=id,mass_kg=mass,kwh=kwh,role='SERVICE',kind='BATTERY'))
    return records


def solve(config,settings,totals,world=None,panels=None,batteries=None,extra=(),base_records=None):
    """Deterministic load manifest; modules have individual bolted rack positions.

    BESS uses the full cabinet deck. A service carrier has a fixed 3x3 m
    landing/exchange bay and a separate, accessible battery rack. PV cassettes
    occupy cargo carriers, never the battery cabinets or exchange bay.
    """
    p=settings;q=parameters(config.logistics.battery_platforms)
    if not config.logistics.individual_chargers:
        raise ValueError('Физическим батарейным платформам нужны индивидуальные зарядные гнёзда; прежний общий зарядник доступен с battery_platforms=None')
    w,l,h=p['platform_width_m'],p['platform_length_m'],p['platform_height_m']
    if w<3 or l<6 or h<1.5:raise ValueError('Батарейным шкафам и сервисному месту нужна платформа не меньше 3×6×1,5 м')
    dry=p['frame_base_kg']+w*l*p['frame_kg_m2']+p['drive_mass_kg'];limit=p['platform_payload_kg']
    records=battery_records(config,world,batteries,extra,base_records);platforms=[];bindings={};grids={}
    previous=getattr(getattr(world,'sim',None),'base',None)
    previous=getattr(previous,'layout_spec',{})
    old_bindings=previous.get('bindings',{})
    if old_bindings and any(p[k]!=previous['parameters'][k] for k in ('heading_deg','platform_length_m','platform_width_m','platform_height_m','platform_gap_m')):
        raise ValueError('Габариты и взаимное положение уже закреплённых платформ меняются в новом шаблоне; текущая техника не переставляется мгновенно')

    def add(role):
        cap=256 if getattr(config.logistics,'self_solar',None) is not None else 64
        if len(platforms)>=cap:raise ValueError(f'Комплектации требуется более {cap} платформ с реальными местами груза')
        id=len(platforms)+1;hardware=q['bess_hardware_kg'] if role=='BESS' else q['service_hardware_kg'] if role=='SERVICE' else 0.
        a=dict(id=id,role=role,offset_m=[0.,station_offset(id-1,l+p['platform_gap_m'])],dimensions_m=[w,l,h],
            payload_kg=hardware,dry_mass_kg=dry,hardware_kg=hardware,loads=[],reserved_volume_m3=0.)
        platforms.append(a);return a

    def bind(a,r,offset,dimensions):
        if a['payload_kg']+r['mass_kg']>limit+1e-7:raise ValueError('Перегруз платформы')
        a['payload_kg']+=r['mass_kg']
        entry={**r,'platform_id':a['id'],'offset_m':list(offset),'dimensions_m':list(dimensions),
               'mount_id':f"platform_{a['id']:02}_mount_{len(a['loads'])+1:04}",'fasteners':4}
        a['loads'].append(entry);bindings[r['id']]=entry
        if r['kind']=='BATTERY':a['reserved_volume_m3']+=math.prod(dimensions)

    if old_bindings:
        for a in previous['platforms']:
            current=add(a['role']);current['offset_m']=list(a['offset_m'])

    for role in ('BESS','SERVICE'):
        subset=[r for r in records if r['role']==role]
        if role=='SERVICE':
            for _ in range(max(0,max(1,totals['pads'])-sum(a['role']==role for a in platforms))):add(role)
        for r in subset:
            dims=pack_dimensions(r['kwh'],p['battery_litres_kwh'])
            choices=[a for a in platforms if a['role']==role]
            # Charge socket travels with its assigned rack, even when pack is away.
            socket=config.logistics.charger_slot_mass_kg if role=='SERVICE' and config.logistics.individual_chargers else 0.
            chosen=None
            key=(role,*dims)
            if key not in grids:grids[key]=_grid(w-.4,l-.4 if role=='BESS' else l/2-.4,h-.3 if role=='BESS' else .8,dims,q['rack_clearance_m'],centre=(0.,0. if role=='BESS' else -l/4))
            preferred=old_bindings.get(r['id'])
            home=None
            if role=='SERVICE' and r['id'].startswith('drone_'):
                serial=int(r['id'].split('_')[1]);services=[a['id'] for a in platforms if a['role']=='SERVICE']
                if services:home=services[(serial-1)%min(len(services),max(1,totals['pads']))]
            choices.sort(key=lambda a:(a['id']!=preferred['platform_id'] if preferred else a['id']!=home if home else False,len(a['loads']),a['id']))
            for a in choices:
                places=[preferred['offset_m']] if preferred and a['id']==preferred['platform_id'] else grids[key]
                # Different pack sizes use non-overlapping bounding cells.
                used=a.setdefault('rack_cells',[])
                same=all(old[1]==dims for old in used)
                candidates=places[len(used):] if same and not preferred else places
                for point in candidates:
                    if not same or preferred:
                        if any(all(abs(point[k]-old[0][k])<(dims[k]+old[1][k])/2+q['rack_clearance_m']-.000001 for k in range(3)) for old in used):continue
                    if a['payload_kg']+r['mass_kg']+socket<=limit-(400 if role=='SERVICE' else 0):
                        chosen=(a,point);break
                if chosen:break
            if chosen is None:
                if preferred:raise ValueError(f"Нельзя переставить закреплённый модуль {r['id']} без переоснащения платформы")
                a=add(role);places=grids[key]
                if not places or r['mass_kg']+socket+a['payload_kg']>limit- (400 if role=='SERVICE' else 0):raise ValueError('Модуль не помещается в батарейный шкаф по массе или габаритам')
                chosen=(a,places[0]);a['rack_cells']=[]
            a,point=chosen;a['rack_cells'].append((point,dims));bind(a,r,point,dims)
            if socket:bind(a,dict(id='charge_slot_'+r['id'],mass_kg=socket,kwh=0.,role=role,kind='CHARGER'),[point[0],point[1],point[2]-.02],[.1,.1,.02])

    services=[a for a in platforms if a['role']=='SERVICE']
    dock_offsets=[]
    for i in range(totals['pads']):
        a=services[i];dock_offsets.append([a['offset_m'][0],a['offset_m'][1]+l/4])
        bind(a,dict(id=f'landing_pad_{i+1}',mass_kg=config.base.equipment.slot_mass_kg,kwh=0.,role='SERVICE',kind='EXCHANGE_BAY'),
             [0.,l/4,h-.075],[w,l/2,.15])
        if i<len(config.fleet.models):
            serial=world.sim.drones[i].serial if world else i+1;drone=config.fleet.models[i]
            bind(a,dict(id=f'hook_{serial}',mass_kg=2.,kwh=0.,role='SERVICE',kind='TOOL'),[w/2-.2,l/4,h-.6],[.2,.3,.2])
            bind(a,dict(id=f'grabber_{serial}',mass_kg=drone.equipment.bucket.mass_kg,kwh=0.,role='SERVICE',kind='TOOL'),[w/2-.45,l/4,h-.4],[.5,.5,.5])
    assigned_mass=sum(a['payload_kg']-a['hardware_kg'] for a in platforms)
    panel_recipes=config.base.equipment.panels if panels is None else panels
    # Every folded cassette gets its own load envelope and tie-down position.
    solar_mass=0.;cargo=[]
    if config.logistics.solar is not None and getattr(config.logistics,'self_solar',None) is None and totals['panel_area_m2']>20:
        from .solar_array import recipe,parameters as solar_parameters,roof_counts
        groups,crews,_,_=recipe(panel_recipes,config.logistics.solar);sp=solar_parameters(config.logistics.solar)
        for i,(_,area,mass) in enumerate(groups):
            frame=max(1.,area/sp['section_area_m2'])*sp['frame_mass_kg'];solar_mass+=mass+frame
            cargo.append((dict(id=f'solar_section_{i+1:04}',mass_kg=mass+frame,kwh=0.,role='CARGO',kind='SOLAR_CASSETTE'),
                          [sp['folded_width_m'],sp['folded_length_m'],sp['folded_height_m']]))
    # All remaining gear and vehicle frames have a reserved cargo envelope.
    rest=max(0.,totals['payload_kg']-assigned_mass-solar_mass)
    # Fixed docking frames are carried at their own exchange bays.
    for i,d in enumerate(config.fleet.models[:len(dock_offsets)]):
        a=services[i];mass=min(rest,d.equipment.frame_mass_kg)
        bind(a,dict(id=f'drone_frame_{i+1}',mass_kg=mass,kwh=0.,role='SERVICE',kind='VEHICLE'),[0.,l/4,h+.5],[2.,2.,1.]);rest-=mass
        a['dimensions_m'][2]=h+1.  # conditional aircraft transport envelope, deck stays at h
    for i in range(math.ceil(rest/150.)):
        mass=min(150.,rest-i*150.)
        if mass>1e-7:cargo.append((dict(id=f'other_cargo_{i+1:04}',mass_kg=mass,kwh=0.,role='CARGO',kind='OTHER'),[1.,1.,max(.1,mass/p['other_equipment_kg_m3'])]))
    cells={}
    for r,dims in cargo:
        chosen=None
        for a in [a for a in platforms if a['role']=='CARGO']:
            volume=math.prod(dims)
            if a['payload_kg']+r['mass_kg']>limit or a['reserved_volume_m3']+volume>w*l*h*p['packing_fraction']+1e-8:continue
            used=cells.setdefault(a['id'],[])
            for point in _grid(w,l,h,dims,0.):
                if not any(all(abs(point[k]-old[0][k])<(dims[k]+old[1][k])/2-1e-7 for k in range(3)) for old in used):chosen=(a,point);break
            if chosen:break
        if chosen is None:
            a=add('CARGO');cells[a['id']]=[];places=_grid(w,l,h,dims,0.)
            if not places or r['mass_kg']>limit:raise ValueError('Груз не помещается на грузовой платформе')
            chosen=(a,places[0])
        a,point=chosen;cells[a['id']].append((point,dims));a['reserved_volume_m3']+=math.prod(dims);bind(a,r,point,dims)
    if not platforms:add('CARGO')
    hardware=sum(a['hardware_kg'] for a in platforms)
    for a in platforms:
        a.pop('rack_cells',None);mass=a['payload_kg'];gross=dry+mass
        cg=np.array([0.,0.,dry*.45+a['hardware_kg']*.6])
        for r in a['loads']:cg+=r['mass_kg']*np.asarray(r['offset_m'])
        cg/=gross;local_x=float(cg[0]);cg[:2]+=a['offset_m']
        tw=max(p['min_track_width_m'],gross*9.80665/(p['max_pressure_kpa']*1000*2*l*p['track_contact_fraction']))
        if tw>w*.38:raise ValueError('Не хватает площади гусениц для фактической нагрузки платформы')
        contact=2*tw*l*p['track_contact_fraction']
        a.update(gross_mass_kg=gross,track_width_m=tw,track_contact_length_m=l*p['track_contact_fraction'],contact_area_m2=contact,
                 pressure_kpa=gross*9.80665/contact/1000,drive_power_kw=p['drive_kw'],centre_of_gravity_m=cg.tolist())
        if abs(local_x)>w/2/1.3 or abs(cg[1]-a['offset_m'][1])>l/2/1.3:raise ValueError('Центр тяжести за безопасным контуром платформы')
    lo=min(a['offset_m'][1]-l/2 for a in platforms);hi=max(a['offset_m'][1]+l/2 for a in platforms)
    gross=sum(a['gross_mass_kg'] for a in platforms);cg=sum(np.asarray(a['centre_of_gravity_m'])*a['gross_mass_kg'] for a in platforms)/gross
    from .base_layout import rotate_offset
    cg[:2]=rotate_offset(p,cg[:2]);count=len(platforms)
    return dict(parameters=p,platforms=platforms,platform_count=count,chassis_mass_kg=count*dry,
        packed_payload_kg=totals['payload_kg']+hardware,packed_gross_mass_kg=gross,packed_volume_m3=sum(a['reserved_volume_m3'] for a in platforms),
        dimensions_m=[w,hi-lo,max(a['dimensions_m'][2] for a in platforms)],deck_height_m=h,bounds_m=[-w/2,lo,w,hi-lo],centre_of_gravity_m=cg.tolist(),
        contact_area_m2=sum(a['contact_area_m2'] for a in platforms),max_pressure_kpa=max(a['pressure_kpa'] for a in platforms),
        drive_power_kw=count*p['drive_kw'],control_power_w=count*p['control_power_w']+sum(q['bess_aux_w'] if a['role']=='BESS' else q['service_aux_w'] if a['role']=='SERVICE' else 0. for a in platforms),
        battery_platforms=q,bindings=bindings,dock_offsets=dock_offsets,battery_hardware_kg=hardware,
        converter_power_w=sum(a['role']=='BESS' for a in platforms)*q['bess_converter_kw']*1000.)


@dataclass
class CarrierState:
    id: int
    mode: str = 'DEPLOY'
    state: str = 'STOWED'
    supports: float = 0.
    apron: float = 0.
    connection: float = 0.
    elapsed: float = 0.
    reason: str = ''


class BatteryPlatforms:
    def __init__(self,world):
        self.world=world;self.settings=parameters(world.config.battery_platforms)
        self.carriers={};self.bindings={};self.issue='';self.site_issues={};self.route_preview=False;self.handlers={};self.returns={}
        self.install(world.sim.base.layout_spec,initial=True)

    def install(self,spec,initial=False):
        """Equipment delivery adds cabinets, never energy or replacement batteries."""
        w=self.world;self.bindings=spec['bindings']
        added=[]
        for a in spec['platforms']:
            if a['role'] not in ('BESS','SERVICE'):continue
            if a['id'] not in self.carriers:self.carriers[a['id']]=CarrierState(a['id'])
            id=f"battery_platform_{a['id']:02}_hardware"
            if id not in w.inventory.items:
                added.append(Item(id,a['hardware_kg'],L.BASE_MOUNT,'INSTALLED','base',position=tuple(w.sim.base.position),kind='BATTERY_PLATFORM_HARDWARE'))
        if added:
            w.inventory.add_many(added)
            if not initial:w.imported_mass_kg+=sum(o.mass_kg for o in added)
        self.position_items()

    def slot_position(self,id):
        if id not in self.bindings:return self.world.sim.base.position.copy()
        self._cache_positions()
        return np.array(self._rack_positions[id])

    def _slot_coordinates(self,id):
        from .base_layout import rotate_offset
        entry=self.bindings.get(id)
        if entry is None:return self.world.sim.base.position.copy()
        b=self.world.sim.base;a=b.layout_spec['platforms'][entry['platform_id']-1]
        offset=np.asarray(a['offset_m'])+np.asarray(entry['offset_m'][:2]);xy=b.position[:2]+rotate_offset(b.config.layout,offset)
        return np.r_[xy,self.world.sim.terrain.sample(*(b.position[:2]+rotate_offset(b.config.layout,a['offset_m'])))+entry['offset_m'][2]]

    def _cache_positions(self):
        from .base_layout import rotate_offset
        w=self.world;b=w.sim.base;spec=b.layout_spec
        schema=(id(spec),tuple(b.position),(b.config.layout or {}).get('heading_deg',0.),
            tuple((a['id'],tuple(a['offset_m'])) for a in spec['platforms']),
            tuple((id,e['platform_id'],tuple(e['offset_m'])) for id,e in self.bindings.items()))
        moved=getattr(self,'_rack_schema',None)!=schema
        if moved:
            self._rack_xy=np.array([b.position[:2]+rotate_offset(b.config.layout,a['offset_m']) for a in spec['platforms']])
        heights=w.sim.terrain.sample_many(self._rack_xy)
        if not moved and np.array_equal(heights,self._rack_heights):return
        self._rack_schema=schema;self._rack_heights=heights.copy()
        self._rack_positions={id:tuple(self._slot_coordinates(id)) for id in self.bindings}
        self._hardware_positions={}
        for a,xy,z in zip(spec['platforms'],self._rack_xy,heights):
            self._hardware_positions[f"battery_platform_{a['id']:02}_hardware"]=(*map(float,xy),float(z+.6))

    def position_items(self):
        w=self.world;self._cache_positions()
        for id,entry in self.bindings.items():
            o=w.inventory.items.get(id)
            if o and id not in self.returns and o.holder_id=='base' and o.location in (L.BASE_MOUNT,L.BASE_STORAGE,L.BASE_CHARGER):o.position=self._rack_positions[id]
        for key,position in self._hardware_positions.items():
            if key in w.inventory.items:w.inventory.items[key].position=position

    @property
    def stowed(self):return not self.returns and not self.handlers and all(c.state=='STOWED' and c.mode=='PACK' for c in self.carriers.values())

    @property
    def ready(self):return not self.issue and all(c.state=='WORKING' and c.mode=='DEPLOY' for c in self.carriers.values())

    @property
    def settled(self):return not self.returns and all(c.state==('WORKING' if c.mode=='DEPLOY' else 'STOWED') for c in self.carriers.values())

    @property
    def converter_power_w(self):return self.world.sim.base.layout_spec['converter_power_w']

    def dock_carrier(self,slot):
        services=[a for a in self.world.sim.base.layout_spec['platforms'] if a['role']=='SERVICE']
        return services[slot]['id'] if slot is not None and slot<len(services) else None

    def service_ready(self,slot=None,*,module_id=None,continuing=False):
        id=self.bindings.get(module_id,{}).get('platform_id') if module_id else self.dock_carrier(slot)
        c=self.carriers.get(id)
        return bool(c and min(c.supports,c.apron,c.connection)>=1 and c.state in ('WORKING','WAIT_SERVICE') and (c.mode=='DEPLOY' or continuing) and (id not in self.site_issues or continuing))

    def busy(self,id):
        w=self.world
        return id in self.handlers or any(self.bindings[key]['platform_id']==id for key in self.returns) or any(self.dock_carrier(d.slot)==id and (d.serial in w.battery_service.drone_swaps or d.serial in w.tool_changes)
                   for d in w.sim.drones) or any(o.reserved_by and self.bindings.get(o.id,{}).get('platform_id')==id for o in w.inventory.select(BatteryModule))

    def acquire(self,owner,slot,module_ids=()):
        ids={self.dock_carrier(slot),*(self.bindings[id]['platform_id'] for id in module_ids)}
        if None in ids or any(self.handlers.get(id) not in (None,owner) for id in ids):return False
        for id in ids:self.handlers[id]=owner
        return True

    def release(self,owner):
        for id in [id for id,value in self.handlers.items() if value==owner]:del self.handlers[id]

    def loads(self):
        """Actual carrier reactions; absent packs/aircraft do not press on tracks.

        Fixed rack loads and aircraft use their own mounts. Remaining shared
        equipment follows its cargo manifest, not an equal load on every track.
        """
        w=self.world;b=w.sim.base;spec=b.layout_spec;rows={}
        for a in spec['platforms']:
            mass=a['dry_mass_kg']+a['hardware_kg']
            rows[a['id']]=dict(mass_kg=mass,moment=np.array([0.,0.,a['dry_mass_kg']*.45+a['hardware_kg']*.6]))
        def add(id,mass,offset):
            rows[id]['mass_kg']+=mass;rows[id]['moment']+=mass*np.asarray(offset)
        assigned=0.
        for id,e in self.bindings.items():
            o=w.inventory.items.get(id)
            if o is None or e['kind'] not in ('BATTERY','CHARGER','EXCHANGE_BAY','TOOL'):continue
            if o.holder_id=='base' or o.location==L.IN_TRANSFER and o.holder_id.startswith('station_transfer_'):
                add(e['platform_id'],o.mass_kg,e['offset_m']);assigned+=o.mass_kg
        for d in w.sim.drones:
            if d.docked:
                id=self.dock_carrier(d.slot)
                if id:add(id,d.config.empty_mass+d.cargo,[0.,spec['parameters']['platform_length_m']/4,b.config.roof_height+.5])
        array=getattr(w,'solar_array',None)
        if array:
            cranes=array.crane_supported()
            for section in array.sections:
                if getattr(array,'mechanism',None)=='self_deploying':
                    from .self_solar import parameters
                    sp=array.settings;frame=2*sp['leaf_pairs']*sp['span_m']*sp['leaf_width_m']*sp['frame_kg_m2']
                    z=sp['hinge_height_m']+sp['leaf_width_m']/2*math.cos(section.open_fraction*math.pi/2)
                    # Hardware was included at the low chassis centroid above.
                    pitch=sp['leaf_width_m']*math.sin(section.open_fraction*math.pi/2)+(sp['panel_thickness_m']+sp['hinge_allowance_m'])*math.cos(section.open_fraction*math.pi/2)
                    moment=sum(m*(1 if i%2==0 else -1)*(sp['root_offset_m']+(i//2+.5)*pitch) for i,m in enumerate(section.leaf_masses_kg))
                    add(section.platform_id,section.panel_mass_kg,[moment/max(1e-12,section.panel_mass_kg),0.,z])
                    rows[section.platform_id]['moment'][2]+=frame*(z-.6)+(section.frame_mass_kg-frame)*.1
                    continue
                entry=self.bindings.get(section.id)
                if entry and (section.state=='STOWED' or section.id in cranes):
                    add(entry['platform_id'],section.panel_mass_kg+section.frame_mass_kg,entry['offset_m'])
        total=b.config.mass+sum(d.config.empty_mass+d.cargo for d in w.sim.drones if d.docked)
        remainder=total-sum(r['mass_kg'] for r in rows.values())
        candidates=[a for a in spec['platforms'] if a['role']=='CARGO'] or spec['platforms']
        weights=[max(1.,sum(e['mass_kg'] for e in a['loads'] if e['kind']=='OTHER')) for a in candidates]
        for a,weight in zip(candidates,weights):add(a['id'],remainder*weight/sum(weights),[0.,0.,b.config.roof_height*.45])
        for id,r in rows.items():r['centre_of_gravity_m']=(r.pop('moment')/max(.001,r['mass_kg'])).tolist()
        return rows

    def request(self,mode,carrier_id=None):
        if mode not in ('PACK','DEPLOY'):raise ValueError('Неизвестная команда платформам')
        if self.world.sim.base.destination is not None:raise ValueError('Команда недоступна во время движения')
        if carrier_id is not None and carrier_id not in self.carriers:raise ValueError('Платформа не найдена')
        for id,c in self.carriers.items():
            if carrier_id is None or id==carrier_id:c.mode=mode;c.reason=''
        if mode=='PACK':
            for job in self.world.jobs:
                if job.status=='QUEUED' and any(self.bindings.get(key,{}).get('platform_id') in self.carriers and
                    (carrier_id is None or self.bindings[key]['platform_id']==carrier_id) for key in job.resource_ids):self.world.cancel_job(job.id)
        self.world.wake_for_solar_service()
        if self.world.sim.shared_site:self.world.sim.shared_site.invalidate_geometry(self.world.sim)

    def polygons(self,*,planned=False):
        if self.route_preview:return []
        b=self.world.sim.base
        return [polygon for (id,c),polygon in zip(self.carriers.items(),deployment_polygons(b.layout_spec,b.position))
                if planned or c.state!='STOWED' or c.mode!='PACK']

    def rectangles(self,*,planned=False):
        from .site_ground import polygon_bounds
        return [polygon_bounds(p) for p in self.polygons(planned=planned)]

    def site_safe(self):
        w=self.world;b=w.sim.base
        return deployment_safe(b.layout_spec,b.position,w.sim.terrain,solar_polygons=w.solar_ground_polygons())

    def audit_site(self):
        """Recheck changed support cells, avoiding full PV scans for distant digs."""
        from .site_ground import rectangle_patch,marine_mask
        from hashlib import sha256
        w=self.world;t=w.sim.terrain;b=w.sim.base
        site=w.sim.shared_site
        actors=tuple((u.id,tuple(u.simulation.base.position),tuple(u.simulation.logistics.excavator.position) if u.simulation.logistics.excavator else None) for u in site.project.units) if site else ()
        stamp=(id(t.height),t.revision,tuple(b.position),id(t.config.georeference),w.field_rect,actors)
        if getattr(self,'_audit_stamp',None)==stamp and 'site_issues' in vars(self):return
        self._audit_stamp=stamp;digest=sha256()
        for rect in self.rectangles(planned=True):
            patch=rectangle_patch(t,rect)
            if patch is None:digest.update(b'outside');continue
            for values in (t.height[patch],t.design_work[patch],t.dump_mask[patch],marine_mask(t)[patch]):digest.update(values.tobytes())
            if t.protected is not None:digest.update(t.protected[patch].tobytes())
        signature=(tuple(b.position),w.field_rect,actors,digest.hexdigest())
        if getattr(self,'_audit_signature',None)==signature and 'site_issues' in vars(self):return
        self._audit_signature=signature
        self.site_issues={}
        solar_polygons=w.solar_ground_polygons()
        for carrier_id in self.carriers:
            spec={**b.layout_spec,'platforms':[b.layout_spec['platforms'][carrier_id-1]]}
            if not deployment_safe(spec,b.position,t,solar_polygons=solar_polygons):
                self.site_issues[carrier_id]='Небезопасный грунт, вода или занятая зона опор/прохода'
        self.issue=('BATTERY_SITE_BLOCKED: '+', '.join(f'P{carrier_id:02}' for carrier_id in self.site_issues)+'; оборудование сохранено') if self.site_issues else ''

    def step(self):
        w=self.world;p=self.settings
        if w.sim.base.destination is not None:
            if not self.stowed:raise ValueError('Движение базы с неубранными батарейными платформами')
            return
        if any(c.mode=='DEPLOY' for c in self.carriers.values()):self.audit_site()
        self._return_step()
        for id,c in self.carriers.items():
            c.reason=self.site_issues.get(id,'')
            busy=self.busy(id)
            # An incoming module is first secured at a working receiving port,
            # even if packing was requested during partial commissioning.
            draining=c.mode=='PACK' and busy and min(c.supports,c.apron,c.connection)<1
            if c.mode=='DEPLOY' or draining:
                if c.state=='WORKING':continue
                if id in self.site_issues:continue
                if c.supports<1:attr,duration,state='supports',p['support_seconds'],'DEPLOY_SUPPORTS'
                elif c.apron<1:attr,duration,state='apron',p['apron_seconds'],'OPEN_APRON'
                elif c.connection<1:attr,duration,state='connection',p['connect_seconds'],'CONNECTING'
                else:c.state='WORKING';continue
                sign=1.
            else:
                if c.state=='STOWED':continue
                if busy:c.state='WAIT_SERVICE';c.reason='Завершение занятого обменного слота';continue
                if c.connection>0:attr,duration,state='connection',p['connect_seconds'],'DISCONNECTING'
                elif c.apron>0:attr,duration,state='apron',p['apron_seconds'],'CLOSE_APRON'
                elif c.supports>0:attr,duration,state='supports',p['support_seconds'],'RETRACT_SUPPORTS'
                else:c.state='STOWED';continue
                sign=-1.
            c.state=state
            if not w.use_station(p['actuator_w']):c.reason='Недостаточно питания привода';continue
            setattr(c,attr,min(1.,max(0.,getattr(c,attr)+sign/duration)));c.elapsed+=1.
            # Exact endpoints avoid an extra powered tick from floating error.
            if abs(getattr(c,attr)-1)<1e-10:setattr(c,attr,1.)
            if abs(getattr(c,attr))<1e-10:setattr(c,attr,0.)
    def receiving_position(self,id):
        from .base_layout import rotate_offset
        w=self.world;b=w.sim.base;a=b.layout_spec['platforms'][self.bindings[id]['platform_id']-1]
        offset=np.asarray(a['offset_m'])+[a['dimensions_m'][0]/2+.5,-a['dimensions_m'][1]/4]
        xy=b.position[:2]+rotate_offset(b.config.layout,offset)
        return np.r_[xy,w.sim.terrain.sample(*xy)+.4]

    def handling_path(self,start,target):
        """A conditional powered side rail, through connected service aprons."""
        from .base_layout import rotate_offset
        b=self.world.sim.base;p=b.layout_spec['parameters'];origin=b.position
        start=np.asarray(start);target=np.asarray(target)
        angle=math.radians(p['heading_deg']);cs,sn=math.cos(angle),math.sin(angle)
        rotation=np.array([[cs,-sn],[sn,cs]])
        a=(start[:2]-origin[:2])@rotation;z=(target[:2]-origin[:2])@rotation
        rail=p['platform_width_m']/2+.5;height=max(start[2],target[2],origin[2]+p['platform_height_m'])+.4
        def point(x,y):return [*(origin[:2]+rotate_offset(p,[x,y])),height]
        if b.layout_spec.get('formation')=='compact_grid_v1':
            aisles=b.layout_spec['service_aisles_x'];header=b.layout_spec['service_header_y']
            if aisles:
                x1=min(aisles,key=lambda x:abs(x-a[0]));x2=min(aisles,key=lambda x:abs(x-z[0]))
                middle=[point(x1,a[1]),point(x2,z[1])] if abs(x1-x2)<1e-9 else [point(x1,a[1]),point(x1,header),point(x2,header),point(x2,z[1])]
                return [start.tolist(),point(a[0],a[1]),*middle,point(z[0],z[1]),target.tolist()]
        return [start.tolist(),point(a[0],a[1]),point(rail,a[1]),point(rail,z[1]),point(z[0],z[1]),target.tolist()]

    def receive(self,id):
        o=self.world.inventory.items[id]
        self.returns[id]=dict(start=list(o.position),elapsed=0.)
        o.state='AWAITING_RACK'

    def _return_step(self):
        w=self.world;p=self.settings
        for id,job in list(self.returns.items()):
            owner='rack_return_'+id;carrier=self.bindings[id]['platform_id'];c=self.carriers[carrier];b=w.inventory.items[id]
            if not self.service_ready(module_id=id,continuing=True) or self.handlers.get(carrier) not in (None,owner) or b.reserved_by not in (None,owner):continue
            self.handlers[carrier]=owner;w.inventory.reserve([id],owner)
            target=self.slot_position(id);start=np.asarray(job['start'])
            if 'path' not in job:job['path']=self.handling_path(start,target)
            segments=np.diff(np.asarray(job['path']),axis=0);distance=float(np.linalg.norm(segments,axis=1).sum())
            duration=max(p['handling_min_seconds'],math.ceil(distance/p['handling_speed_m_s']))
            mechanical=b.mass_kg*(9.80665*(float(np.maximum(segments[:,2],0).sum())+.1*float(np.linalg.norm(segments[:,:2],axis=1).sum()))+p['handling_speed_m_s']**2)/.8
            if not w.use_station(p['handling_w']+mechanical/duration):c.reason='Нет питания для возврата модуля в шкаф';continue
            job['elapsed']+=1.
            w.inventory.move(id,L.IN_TRANSFER,'station_transfer_rack',path_position(job['path'],min(1.,job['elapsed']/duration)),'BATTERY_HANDLING',owner)
            if job['elapsed']>=duration:
                w.inventory.move(id,L.BASE_STORAGE,'base',target,'STORED_READY' if b.soc>=1-1e-12 else 'QUEUED_FOR_CHARGE',owner)
                w.inventory.release(owner);self.release(owner);del self.returns[id]

    def energy_budget(self):
        p=self.settings
        return len(self.carriers)*p['actuator_w']*(p['support_seconds']+p['apron_seconds']+p['connect_seconds'])

    def report(self):
        w=self.world;rows=[];actual=self.loads()
        for id,c in self.carriers.items():
            a=w.sim.base.layout_spec['platforms'][id-1];packs=[w.inventory.items[k] for k,e in self.bindings.items() if e['platform_id']==id and k in w.inventory.items and isinstance(w.inventory.items[k],BatteryModule)]
            rows.append(dict(id=id,role=a['role'],state=c.state,mode=c.mode,reason=c.reason,supports=c.supports,apron=c.apron,connection=c.connection,
                capacity_kwh=sum(b.capacity_j for b in packs)/KWH,energy_kwh=sum(b.energy_j for b in packs if b.holder_id=='base')/KWH,
                mass_kg=a['gross_mass_kg'],payload_kg=a['payload_kg'],dimensions_m=a['dimensions_m'],centre_of_gravity_m=a['centre_of_gravity_m'],
                actual_mass_kg=actual[id]['mass_kg'],actual_centre_of_gravity_m=actual[id]['centre_of_gravity_m'],
                power_kw=self.settings['bess_converter_kw'] if a['role']=='BESS' else min(self.settings['service_bus_kw'],sum(w.battery_service.slots[b.id].rated_w for b in packs if b.id in w.battery_service.slots)/1000),
                modules=len(packs)))
        return rows

    def validate(self):
        w=self.world;spec=w.sim.base.layout_spec
        if set(self.carriers)!={a['id'] for a in spec['platforms'] if a['role'] in ('BESS','SERVICE')}:raise ValueError('Потеряны батарейные платформы')
        if w.battery_platforms is not self or parameters(self.settings)!=self.settings:raise ValueError('Повреждены параметры батарейных платформ')
        states={'STOWED','DEPLOY_SUPPORTS','OPEN_APRON','CONNECTING','WORKING','WAIT_SERVICE','DISCONNECTING','CLOSE_APRON','RETRACT_SUPPORTS'}
        for c in self.carriers.values():
            if c.state not in states or c.mode not in ('PACK','DEPLOY') or not all(0<=v<=1 for v in (c.supports,c.apron,c.connection)) or not math.isfinite(c.elapsed) or c.elapsed<0:raise ValueError('Повреждено состояние батарейной платформы')
            if c.state=='STOWED' and max(c.supports,c.apron,c.connection)>0 or c.state=='WORKING' and min(c.supports,c.apron,c.connection)<1:raise ValueError('Несогласованное состояние опор и подключения')
        mounts=[e['mount_id'] for a in spec['platforms'] for e in a['loads']]
        if len(mounts)!=len(set(mounts)):raise ValueError('Повторное назначение крепления')
        for b in w.inventory.select(BatteryModule):
            if b.id not in self.bindings:raise ValueError(f'Нет крепления батареи {b.id}')
            e=self.bindings[b.id]
            if e['role']!=('BESS' if b.compatibility=='base' else 'SERVICE') or abs(e['mass_kg']-b.mass_kg)>1e-7:raise ValueError('Модуль на неподходящем носителе')
        for id in self.returns:
            if id not in w.inventory.items or id not in self.bindings:raise ValueError('Потерян возвращаемый модуль')
        if w.sim.base.destination is not None and not self.stowed:raise ValueError('Переезд с неубранными опорами')
