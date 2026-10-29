"""Permanently attached accordion PV wings on tracked carriers.

Conditional engineering model: 6x2 m leaves fold upright, then open symmetrically
on rolling ground supports. No installer, unloading or detached ground panels.
"""
from copy import deepcopy
from dataclasses import dataclass,field
import math
import numpy as np
from .inventory import Item,Location as L

DEFAULTS=dict(leaf_pairs=6,leaf_width_m=2.,span_m=6.,panel_thickness_m=.04,
    hinge_allowance_m=.02,root_offset_m=.6,hinge_height_m=1.,transport_clearance_m=.2,
    frame_kg_m2=1.5,support_mass_kg=8.,support_pad_m2=.25,fixed_mechanism_kg=180.,
    motor_w=750.,support_w=250.,connect_w=100.,idle_w=5.,support_seconds=30.,
    unfold_seconds=180.,connect_seconds=10.,rolling_resistance=.08,drive_efficiency=.75,
    max_grade_deg=8.,support_travel_m=.6,row_access_m=2.,column_access_m=3.)
MAX_PLATFORMS=256


def parameters(value=None):
    p={**DEFAULTS,**(value or {})}
    if set(p)!=set(DEFAULTS) or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in p.values()):raise ValueError('Неверный паспорт самораскрывающихся солнечных платформ')
    if type(p['leaf_pairs']) is not int or not 2<=p['leaf_pairs']<=12 or p['leaf_pairs']%2 or p['drive_efficiency']>1 or p['max_grade_deg']>12:raise ValueError('Неверное число шарниров, КПД или допустимый уклон PV')
    return p


def enabled(config):return config.logistics is not None and config.logistics.self_solar is not None


def roof_recipes(panels):
    from .solar_array import roof_counts
    result=[]
    for panel,count in zip(panels,roof_counts(panels)):
        if count:q=deepcopy(panel);q.quantity=count;result.append(q)
    return result


def hardware(p):
    area=2*p['leaf_pairs']*p['leaf_width_m']*p['span_m']
    supports=4+4*math.ceil(p['leaf_pairs']/2)
    return area*p['frame_kg_m2']+supports*p['support_mass_kg']+p['fixed_mechanism_kg']


def recipe(panels,settings,existing=()):
    from .solar_array import roof_counts
    p=parameters(settings);leaf_area=p['leaf_width_m']*p['span_m'];n=2*p['leaf_pairs']
    groups=[];roof=[];index=0;current=None;leaf=0
    for panel,mounted in zip(panels,roof_counts(panels)):
        for k in range(panel.quantity):
            index+=1;id=f'solar_panel_{index:03}'
            if k<mounted:roof.append(id);continue
            if panel.area_m2>leaf_area+1e-9:raise ValueError('Цельная панель больше листа гармошки; выберите меньший модуль')
            if current is None or leaf>=n:
                current=dict(panel_ids=[],leaves=[[] for _ in range(n)],areas=[0.]*n,area_m2=0.,panel_mass_kg=0.,hardware_kg=hardware(p));groups.append(current);leaf=0
            if current['areas'][leaf]+panel.area_m2>leaf_area+1e-9:
                leaf+=1
                if leaf>=n:
                    current=dict(panel_ids=[],leaves=[[] for _ in range(n)],areas=[0.]*n,area_m2=0.,panel_mass_kg=0.,hardware_kg=hardware(p));groups.append(current);leaf=0
            current['panel_ids'].append(id);current['leaves'][leaf].append(id);current['areas'][leaf]+=panel.area_m2
            current['area_m2']+=panel.area_m2;current['panel_mass_kg']+=panel.mass_kg
    # Empty delivered mechanisms retain their identity and mass after hot removal.
    while len(groups)<len(existing):groups.append(dict(panel_ids=[],leaves=[[] for _ in range(n)],areas=[0.]*n,area_m2=0.,panel_mass_kg=0.,hardware_kg=hardware(p)))
    return groups,roof


def _rect(spec,module,point,*,swept=True):
    from .base_layout import rotate_offset
    p=spec['self_solar'];w=2*(p['root_offset_m']+p['leaf_pairs']*math.hypot(p['leaf_width_m'],p['panel_thickness_m']+p['hinge_allowance_m'])+.25) if swept else module['dimensions_m'][0]
    length=p['span_m']+.5 if swept else module['dimensions_m'][1]
    corners=np.array([rotate_offset(spec['parameters'],v) for v in ((-w/2,-length/2),(w/2,-length/2),(w/2,length/2),(-w/2,length/2))])
    centre=np.asarray(point[:2])+rotate_offset(spec['parameters'],module['offset_m']);corners+=centre
    lo=corners.min(axis=0);hi=corners.max(axis=0);return (*map(float,lo),*map(float,hi-lo))


def deployment_rectangles(spec,point):
    return [_rect(spec,a,point) for a in spec.get('platforms',()) if a.get('role')=='PV'] if 'self_solar' in spec else []


def deployment_safe(spec,point,terrain):
    from .site_ground import footprint_safe
    if 'self_solar' not in spec:return True
    from .compact_layout import safe
    if not safe(spec,point,terrain):return False
    p=spec['self_solar'];blocked=terrain.design_work|terrain.dump_mask
    if terrain.protected is not None:blocked=blocked|terrain.protected
    site=getattr(terrain,'shared_site',None)
    return all(footprint_safe(terrain,r,blocked,grade_deg=p['max_grade_deg'],relief_m=p['support_travel_m']) and support_safe(spec,a,point,terrain) and
               (site is None or not site.ground_occupied(getattr(terrain,'owner_id',None),r)) for a,r in zip((a for a in spec['platforms'] if a.get('role')=='PV'),deployment_rectangles(spec,point)))


def support_centres(spec,module,point,fraction=1.):
    from .base_layout import rotate_offset
    p=spec['self_solar'];theta=fraction*math.pi/2
    pitch=p['leaf_width_m']*math.sin(theta)+(p['panel_thickness_m']+p['hinge_allowance_m'])*math.cos(theta)
    centre=np.asarray(point[:2])+rotate_offset(spec['parameters'],module['offset_m'])
    for sign in (-1,1):
        for j in (0,*range(2,p['leaf_pairs']+1,2)):
            for end in (-1,1):yield centre+rotate_offset(spec['parameters'],[sign*(p['root_offset_m']+j*pitch),end*p['span_m']/2])


def support_safe(spec,module,point,terrain):
    from .base_layout import ground_properties
    p=spec['self_solar'];count=4+2*p['leaf_pairs']
    # Conservative: each leg is sized for an equal share of the entire packed
    # vehicle, although its tracks continue supporting the central chassis.
    pressure=module['gross_mass_kg']*9.80665/(count*p['support_pad_m2'])/1000
    return all(ground_properties(terrain,q,spec['parameters'])[0]>=pressure for q in support_centres(spec,module,point))


def refresh_spec(spec):
    from .base_layout import layout_footprints,rotate_offset
    boxes=list(layout_footprints(spec,[0.,0.]));lo=np.min(np.array(boxes)[:,:2],axis=0);hi=np.max(np.array(boxes)[:,:2]+np.array(boxes)[:,2:],axis=0)
    # The displayed envelope follows the actual world heading; the hub proxy is separate.
    spec['bounds_m']=[*map(float,lo),*map(float,hi-lo)];spec['dimensions_m']=[*map(float,hi-lo),max(a['dimensions_m'][2] for a in spec['platforms'])]
    spec['chassis_mass_kg']=sum(a['dry_mass_kg'] for a in spec['platforms']);spec['platform_count']=len(spec['platforms'])
    spec['packed_payload_kg']=sum(a['payload_kg'] for a in spec['platforms']);spec['packed_gross_mass_kg']=sum(a['gross_mass_kg'] for a in spec['platforms'])
    cg=np.zeros(3)
    for a in spec['platforms']:
        c=np.asarray(a['centre_of_gravity_m']);cg+=a['gross_mass_kg']*np.r_[rotate_offset(spec['parameters'],c[:2]),c[2]]
    spec['centre_of_gravity_m']=(cg/spec['packed_gross_mass_kg']).tolist()
    spec['contact_area_m2']=sum(a['contact_area_m2'] for a in spec['platforms']);spec['max_pressure_kpa']=max(a['pressure_kpa'] for a in spec['platforms'])
    spec['drive_power_kw']=sum(a['drive_power_kw'] for a in spec['platforms'])


def compose(config,core,groups,world=None):
    from .base_layout import station_offset
    spec=core;p=parameters(config.logistics.self_solar);lp=spec['parameters'];spec['self_solar']=p
    if p['span_m']>lp['platform_length_m'] or p['root_offset_m']+p['leaf_pairs']*(p['panel_thickness_m']+p['hinge_allowance_m'])>lp['platform_width_m']/2-.1:raise ValueError('Сложенные крылья не помещаются на выбранном шасси')
    old=[a for a in spec['platforms'] if a.get('role')=='PV'];included_controls=len(old)
    if not old and world and getattr(world.sim.base,'layout_spec',{}).get('self_solar'):
        old=[deepcopy(a) for a in world.sim.base.layout_spec['platforms'] if a.get('role')=='PV'];spec['platforms'].extend(old)
    count=max(len(groups),len(old));groups=list(groups)
    if count+len(spec['platforms'])-len(old)>MAX_PLATFORMS:raise ValueError('Солнечному комплексу требуется более 256 носителей')
    dummy=dict(panel_ids=[],leaves=[[] for _ in range(2*p['leaf_pairs'])],areas=[0.]*(2*p['leaf_pairs']),area_m2=0.,panel_mass_kg=0.,hardware_kg=hardware(p))
    while len(groups)<count:groups.append(deepcopy(dummy))
    hub=[a for a in spec['platforms'] if a.get('role')!='PV'];spec['hub_radius_m']=lp['platform_width_m']/2
    half=max(abs(a['offset_m'][1])+a['dimensions_m'][1]/2 for a in hub)
    sweep=2*(p['root_offset_m']+p['leaf_pairs']*math.hypot(p['leaf_width_m'],p['panel_thickness_m']+p['hinge_allowance_m'])+.25)
    pitch=sweep+p['column_access_m'];rows=max(1,math.ceil(math.sqrt(max(1,count)*(lp['platform_length_m']+p['row_access_m'])/pitch)))
    dry=lp['frame_base_kg']+lp['platform_width_m']*lp['platform_length_m']*lp['frame_kg_m2']+lp['drive_mass_kg']
    spec['pv_groups']=[]
    for i,g in enumerate(groups):
        if i<len(old):a=old[i];fixed=bool(world and (getattr(world,'solar_array',None) or getattr(world.sim.base,'layout_spec',{}).get('pv_placed')))
        else:
            a=dict(id=max(a['id'] for a in spec['platforms'])+1);spec['platforms'].append(a);fixed=False
        previous=next((row for row in world.sim.base.layout_spec['platforms'] if row['id']==a['id']),a) if fixed else a
        offset=previous.get('offset_m') if fixed else [(i%rows-(rows-1)/2)*pitch,(1 if (i//rows)%2==0 else -1)*(half+8+(i//(2*rows))*(lp['platform_length_m']+p['row_access_m']))]
        payload=g['panel_mass_kg']+g['hardware_kg']
        if payload>lp['platform_payload_kg']:raise ValueError('Солнечные панели и собственные приводы перегружают шасси')
        gross=dry+payload;tw=max(lp['min_track_width_m'],gross*9.80665/(lp['max_pressure_kpa']*1000*2*lp['platform_length_m']*lp['track_contact_fraction']))
        if tw>lp['platform_width_m']*.38:raise ValueError('Недостаточная площадь гусениц солнечного носителя')
        contact=2*tw*lp['platform_length_m']*lp['track_contact_fraction'];frame=p['span_m']*p['leaf_width_m']*2*p['leaf_pairs']*p['frame_kg_m2']
        z=(dry*.45+(g['panel_mass_kg']+frame)*(p['hinge_height_m']+p['leaf_width_m']/2)+(g['hardware_kg']-frame)*.7)/gross
        a.update(role='PV',offset_m=list(offset),dimensions_m=[lp['platform_width_m'],lp['platform_length_m'],p['hinge_height_m']+p['leaf_width_m']+p['transport_clearance_m']],payload_kg=payload,hardware_kg=g['hardware_kg'],dry_mass_kg=dry,gross_mass_kg=gross,
            track_width_m=tw,track_contact_length_m=lp['platform_length_m']*lp['track_contact_fraction'],contact_area_m2=contact,pressure_kpa=gross*9.80665/contact/1000,drive_power_kw=lp['drive_kw'],centre_of_gravity_m=[offset[0],offset[1],z],loads=[],reserved_volume_m3=2*p['leaf_pairs']*(p['panel_thickness_m']+p['hinge_allowance_m'])*p['leaf_width_m']*p['span_m'],pv_fixed=fixed)
        spec['pv_groups'].append(dict(g,platform_id=a['id']))
    spec['packed_volume_m3']+=sum(a['reserved_volume_m3'] for a in spec['platforms'] if a.get('role')=='PV')
    spec['control_power_w']-=included_controls*lp['control_power_w']
    spec['platforms'].sort(key=lambda a:a['id'])
    spec['control_power_w']+=count*(lp['control_power_w']+p['idle_w']);spec['deck_height_m']=lp['platform_height_m']
    from .compact_layout import arrange
    arrange(config,spec,world);refresh_spec(spec)
    return spec


def place(config,spec,point,terrain):
    """Choose initial/delivered carrier positions; existing carriers never teleport."""
    from .site_ground import footprint_safe,paint_rect,marine_mask
    from .base_layout import layout_footprints,rotate_offset,station_offset
    from .battery_platforms import deployment_rectangles as battery_rects
    if 'self_solar' not in spec:return spec
    result={**spec,'platforms':[dict(a) for a in spec['platforms']]};p=spec['self_solar'];lp=spec['parameters']
    blocked=terrain.design_work|terrain.dump_mask|marine_mask(terrain)
    if terrain.protected is not None:blocked=blocked|terrain.protected
    blocked=blocked.copy();site=getattr(terrain,'shared_site',None)
    if site:
        from .site_ground import equipment_mask
        others=[u.simulation for u in site.project.units if u.id!=getattr(terrain,'owner_id',None)]
        blocked|=equipment_mask(terrain,others)
    from .compact_layout import safe as access_safe,rectangles as access_rectangles
    if not access_safe(result,point,terrain):raise ValueError('COMPACT_ACCESS_BLOCKED: рабочие проходы пересекают воду, выемку, отвал или чужую технику')
    hub={**spec,'platforms':[a for a in spec['platforms'] if a.get('role')!='PV']}
    hub_boxes=[*layout_footprints(hub,point),*battery_rects(spec,point)]
    for rect in hub_boxes:
        if not footprint_safe(terrain,rect,blocked,grade_deg=p['max_grade_deg'],relief_m=float('inf')):raise ValueError('Нет безопасного места для центральных платформ')
    for rect in hub_boxes:
        paint_rect(terrain,blocked,rect,1.)
    f=config.fleet;offsets=spec.get('dock_offsets',[])
    for i in range(max(f.count,f.charge_slots)):
        along=offsets[i][1] if i<len(offsets) else station_offset(i,f.pad_spacing)
        across=offsets[i][0] if i<len(offsets) else 0.
        distance=max(f.waiting_offset,spec['hub_radius_m']+3+f.separation) if offsets else f.waiting_offset
        for dx in (0.,-distance):
            x,y=np.asarray(point[:2])+rotate_offset(lp,[across+dx,along]);rect=(x-f.separation,y-f.separation,2*f.separation,2*f.separation)
            if not footprint_safe(terrain,rect,terrain.design_work|terrain.dump_mask,grade_deg=8.,relief_m=float('inf')):raise ValueError('Посадочные площадки не помещаются')
            paint_rect(terrain,blocked,rect)
    if config.logistics.excavator_enabled:
        q=config.trench;c=config.logistics;y=q.y-q.width/2-q.depth/terrain.slope-c.support_half_width_m-.1
        paint_rect(terrain,blocked,(q.x-c.support_half_width_m-1,y-c.support_half_width_m-1,2*c.support_half_width_m+2,2*c.support_half_width_m+2))
    access_mask=np.zeros_like(blocked)
    for rect in access_rectangles(result,point):paint_rect(terrain,access_mask,rect)
    example=next(a for a in result['platforms'] if a.get('role')=='PV');rect=_rect(spec,example,[0.,0.]);ww,hh=rect[2:]
    side=terrain.height.shape[0]*terrain.cell;xs=np.arange(ww/2+1,side-ww/2,ww+p['column_access_m']);ys=np.arange(hh/2+1,side-hh/2,hh+p['row_access_m'])
    candidates=[np.array([x,y]) for y in ys for x in xs];candidates.sort(key=lambda c:(float(np.sum((c-np.asarray(point[:2]))**2)),c[1],c[0]))
    for a in [a for a in result['platforms'] if a.get('role')=='PV']:
        if a.get('pv_fixed'):
            rect=_rect(result,a,point)
            if not footprint_safe(terrain,rect,blocked,grade_deg=p['max_grade_deg'],relief_m=p['support_travel_m']):raise ValueError('Существующий солнечный носитель нельзя переставить при поставке')
        else:
            candidates_blocked=blocked|access_mask
            chosen=None
            for candidate in candidates:
                rect=(candidate[0]-ww/2,candidate[1]-hh/2,ww,hh)
                if footprint_safe(terrain,rect,candidates_blocked,grade_deg=p['max_grade_deg'],relief_m=p['support_travel_m']):chosen=candidate;break
            if chosen is None:raise ValueError('BLOCKED_SOLAR_SPACE: не хватает сухой ровной площади для собственных крыльев и опор')
            angle=math.radians(lp['heading_deg']);delta=chosen-np.asarray(point[:2]);offset=[math.cos(angle)*delta[0]+math.sin(angle)*delta[1],-math.sin(angle)*delta[0]+math.cos(angle)*delta[1]]
            a['offset_m']=offset;a['centre_of_gravity_m']=[*offset,a['centre_of_gravity_m'][2]];rect=_rect(result,a,point)
        paint_rect(terrain,blocked,rect,.25)
        if not support_safe(result,a,point,terrain):raise ValueError('Недостаточная несущая способность грунта под опорами солнечной платформы')
    result['pv_placed']=True
    refresh_spec(result);return result


@dataclass
class SolarPlatform:
    id: str
    platform_id: int
    panel_ids: list
    leaves: list
    area_m2: float
    panel_mass_kg: float
    frame_mass_kg: float
    state: str = 'STOWED'
    supports: float = 0.
    open_fraction: float = 0.
    connection: float = 0.
    elapsed: float = 0.
    actuation_energy_j: float = 0.
    reason: str = ''
    rect: tuple = ()
    leaf_masses_kg: list = field(default_factory=list)


class SelfSolarArray:
    mechanism='self_deploying'
    def __init__(self,world):
        self.settings=parameters(world.config.self_solar);self.mode='DEPLOY';self.issue='';self.invalid_sections=[];self.crews=[];self.available_installers=0
        groups,roof=recipe(world.sim.base.config.equipment.panels,self.settings,[a for a in world.sim.base.layout_spec['platforms'] if a.get('role')=='PV'])
        self.roof_panel_ids=roof;self.roof_area_m2=sum(p.quantity*p.area_m2 for p in roof_recipes(world.sim.base.config.equipment.panels))
        carriers=[a for a in world.sim.base.layout_spec['platforms'] if a.get('role')=='PV'];self.sections=[]
        for a,g in zip(carriers,groups):
            q=SolarPlatform(f"solar_platform_{a['id']:03}_mechanism",a['id'],g['panel_ids'],g['leaves'],g['area_m2'],g['panel_mass_kg'],g['hardware_kg'])
            q.leaf_masses_kg=[sum(world.inventory.items[id].mass_kg for id in ids) for ids in q.leaves];self.sections.append(q)
        self.support_ids=[q.id for q in self.sections];self.reserved_rects=[];self.plan(world)

    def module(self,world,q):return world.sim.base.layout_spec['platforms'][q.platform_id-1]

    def centre(self,world,q):
        from .base_layout import rotate_offset
        xy=world.sim.base.position[:2]+rotate_offset(world.sim.base.config.layout,self.module(world,q)['offset_m'])
        return np.r_[xy,world.sim.terrain.sample(*xy)]

    def plan(self,world,count=None):
        # Replanning validates the fixed carriers; it cannot choose new positions.
        self.reserved_rects=[]
        for q in self.sections:
            q.rect=_rect(world.sim.base.layout_spec,self.module(world,q),world.sim.base.position);self.reserved_rects.append(q.rect)
        self._audit_stamp=None

    def add_inventory(self,world):
        items=[]
        for q in self.sections:
            centre=self.centre(world,q)
            if q.id not in world.inventory.items:items.append(Item(q.id,q.frame_mass_kg,L.BASE_MOUNT,'ATTACHED','base',position=tuple(centre),kind='SOLAR_FRAME'))
        world.inventory.add_many(items);self.sync_positions(world,force=True)

    @property
    def stowed(self):return all(q.state=='STOWED' and q.supports==q.open_fraction==q.connection==0 for q in self.sections)
    @property
    def ready(self):return not self.invalid_sections and all(q.state=='DEPLOYED' for q in self.sections)
    @property
    def installed_area(self):return self.roof_area_m2+sum(q.area_m2 for q in self.sections if q.state=='DEPLOYED' and q.id not in self.invalid_sections and self.mode=='DEPLOY')
    def offboard_panel_mass(self):return 0.
    def crane_supported(self):return set()
    def duration(self):return self.settings['support_seconds']+self.settings['unfold_seconds']+self.settings['connect_seconds']+1
    def energy_budget(self):
        p=self.settings
        return sum(p['support_w']*p['support_seconds']+p['connect_w']*p['connect_seconds']+p['motor_w']*p['unfold_seconds']+
            (q.panel_mass_kg+2*p['leaf_pairs']*p['span_m']*p['leaf_width_m']*p['frame_kg_m2'])*9.80665*(p['leaf_width_m']/2+p['rolling_resistance']*p['leaf_pairs']*p['leaf_width_m'])/p['drive_efficiency'] for q in self.sections)

    def request_fold(self):self.mode='FOLD'
    def request_deploy(self):self.mode='DEPLOY'

    def audit(self,world):
        from .site_ground import footprint_safe
        t=world.sim.terrain;b=world.sim.base;site=world.sim.shared_site
        actors=tuple((u.id,tuple(u.simulation.base.position),id(getattr(u.simulation.base,'layout_spec',None)),
            getattr(getattr(u.simulation.logistics,'solar_array',None),'mode',None),
            getattr(getattr(u.simulation.logistics,'solar_array',None),'stowed',True)) for u in site.project.units) if site else ()
        excavator=tuple(world.excavator.position) if world.excavator else ()
        stamp=(id(t.height),t.revision,id(b.layout_spec),tuple(b.position),actors,excavator,id(t.config.georeference))
        if getattr(self,'_audit_stamp',None)==stamp:return
        self._audit_stamp=stamp;self.invalid_sections=[];p=self.settings
        from .base_layout import layout_footprints
        bodies=[(a['id'],r) for a,r in zip(b.layout_spec['platforms'],layout_footprints(b.layout_spec,b.position))]
        spaces={q.id:_rect(b.layout_spec,self.module(world,q),b.position) for q in self.sections}
        extras=list(world.battery_platforms.rectangles()) if world.battery_platforms else []
        if world.excavator:
            e=world.excavator;r=world.config.support_half_width_m+.3;extras.append((e.position[0]-r,e.position[1]-r,2*r,2*r))
        blocked=t.design_work|t.dump_mask
        if t.protected is not None:blocked=blocked|t.protected
        for q in self.sections:
            rect=spaces[q.id];q.rect=rect
            safe=footprint_safe(t,rect,blocked,grade_deg=p['max_grade_deg'],relief_m=p['support_travel_m']) and support_safe(b.layout_spec,self.module(world,q),b.position,t)
            if safe and site:safe=not site.ground_occupied(getattr(t,'owner_id',None),rect)
            if safe:
                boxes=[r for id,r in bodies if id!=q.platform_id]+extras+[r for id,r in spaces.items() if id!=q.id]
                for xx,yy,w,h in boxes:
                    x,y,ww,hh=rect
                    if x<xx+w and x+ww>xx and y<yy+h and y+hh>yy:safe=False;break
            if not safe:self.invalid_sections.append(q.id)
        self.issue=('SELF_SOLAR_BLOCKED: '+', '.join(self.invalid_sections)+'; панели закреплены, перемещение остановлено') if self.invalid_sections else ''
        world.solar_site_issue=self.issue

    def panel_pose(self,world,q,leaf):
        from .base_layout import rotate_offset
        p=self.settings;theta=q.open_fraction*math.pi/2;side=1 if leaf%2==0 else -1;j=leaf//2
        pitch=p['leaf_width_m']*math.sin(theta)+(p['panel_thickness_m']+p['hinge_allowance_m'])*math.cos(theta)
        offset=[side*(p['root_offset_m']+(j+.5)*pitch),0.];centre=self.centre(world,q)
        return centre+np.r_[rotate_offset(world.sim.base.config.layout,offset),p['hinge_height_m']+p['leaf_width_m']/2*math.cos(theta)]

    def sync_positions(self,world,force=False,only=None):
        key=tuple(world.sim.base.position)
        if only is None and not force and getattr(self,'_position_key',None)==key:return
        self._position_key=key
        for q in self.sections if only is None else only:
            chassis=world.inventory.items.get(f'platform_{q.platform_id:02}_chassis')
            if chassis is not None:chassis.position=tuple(map(float,self.centre(world,q)))
            for i,ids in enumerate(q.leaves):
                if not ids:continue
                position=tuple(map(float,self.panel_pose(world,q,i)))
                for id in ids:
                    o=world.inventory.items[id];o.position=position;o.location=L.BASE_MOUNT;o.holder_id='base';o.state='ATTACHED'
            if q.id in world.inventory.items:world.inventory.items[q.id].position=tuple(map(float,self.centre(world,q)))

    def step(self,world):
        b=world.sim.base;p=self.settings
        if b.destination is not None:
            if not self.stowed:raise ValueError('Движение со раскрытыми собственными солнечными крыльями запрещено')
            return
        self.audit(world);changed=[]
        for q in self.sections:
            q.reason=''
            if self.mode=='DEPLOY':
                if q.state=='DEPLOYED':continue
                if q.id in self.invalid_sections:q.reason='Недоступна зона раскрытия или опоры';continue
                if q.supports<1:attr,seconds,power,state='supports',p['support_seconds'],p['support_w'],'DEPLOY_SUPPORTS'
                elif q.open_fraction<1:attr,seconds,power,state='open_fraction',p['unfold_seconds'],p['motor_w'],'UNFOLDING'
                elif q.connection<1:attr,seconds,power,state='connection',p['connect_seconds'],p['connect_w'],'CONNECTING'
                else:q.state='DEPLOYED';continue
                direction=1.
            else:
                if q.state=='STOWED':continue
                if q.id in self.invalid_sections and q.open_fraction>0:q.reason='Препятствие в ходе складывания; требуется устранить';continue
                if q.connection>0:attr,seconds,power,state='connection',p['connect_seconds'],p['connect_w'],'DISCONNECTING'
                elif q.open_fraction>0:attr,seconds,power,state='open_fraction',p['unfold_seconds'],p['motor_w'],'FOLDING'
                elif q.supports>0:attr,seconds,power,state='supports',p['support_seconds'],p['support_w'],'RETRACT_SUPPORTS'
                else:q.state='STOWED';continue
                direction=-1.
            before=getattr(q,attr);after=min(1.,max(0.,before+direction/seconds));energy=power
            if attr=='open_fraction':
                a,c=before*math.pi/2,after*math.pi/2
                moving=q.panel_mass_kg+2*p['leaf_pairs']*p['span_m']*p['leaf_width_m']*p['frame_kg_m2']
                lift=max(0.,math.cos(c)-math.cos(a))*p['leaf_width_m']/2
                travel=abs(math.sin(c)-math.sin(a))*p['leaf_width_m']*p['leaf_pairs']/2
                energy+=moving*9.80665*(lift+p['rolling_resistance']*travel)/p['drive_efficiency']
            q.state=state
            if not world.use_station(energy):q.reason='Недостаточно мощности или энергии собственного привода';continue
            if abs(after-1)<1e-10:after=1.
            if abs(after)<1e-10:after=0.
            setattr(q,attr,after);q.elapsed+=1.;q.actuation_energy_j+=energy;changed.append(q)
        if changed:self.sync_positions(world,only=changed)
        self.sync_positions(world)
        world.field_ready=self.ready;world.field_elapsed=self.duration()*sum(q.state=='DEPLOYED' for q in self.sections)/max(1,len(self.sections))
        world.base_pack_elapsed=self.duration() if self.mode=='FOLD' and self.stowed else self.duration()*.999*sum(q.state=='STOWED' for q in self.sections)/max(1,len(self.sections)) if self.mode=='FOLD' else 0.

    def leaf_polygons(self,world,q):
        from .base_layout import rotate_offset
        p=self.settings;theta=q.open_fraction*math.pi/2;pitch=p['leaf_width_m']*math.sin(theta)+(p['panel_thickness_m']+p['hinge_allowance_m'])*math.cos(theta)
        for i in range(2*p['leaf_pairs']):
            pos=self.panel_pose(world,q,i);half=max(.025,pitch/2)
            corners=np.array([rotate_offset(world.sim.base.config.layout,v) for v in ((-half,-p['span_m']/2),(half,-p['span_m']/2),(half,p['span_m']/2),(-half,p['span_m']/2),(-half,-p['span_m']/2))])+pos[:2]
            yield corners

    def obstacle_height(self,world,point):
        """Packed bodies and the complete reserved sweep, including single-base runs."""
        key=(world.sim.time,tuple(world.sim.base.position))
        if getattr(self,'_obstacle_key',None)!=key:
            self._obstacle_key=key;self._obstacle_grid={}
            spec=world.sim.base.layout_spec;b=world.sim.base
            expanded=not self.stowed or self.mode=='DEPLOY'
            for q in self.sections:
                body=_rect(spec,self.module(world,q),b.position,swept=False)
                rect=_rect(spec,self.module(world,q),b.position)
                ground=self.centre(world,q)[2]
                boxes=[(*body,ground+self.module(world,q)['dimensions_m'][2])]
                if expanded:boxes.append((*rect,ground+self.settings['hinge_height_m']+self.settings['leaf_width_m']*math.cos(q.open_fraction*math.pi/2)))
                for x,y,w,h,z in boxes:
                    for gx in range(math.floor(x/16),math.floor((x+w)/16)+1):
                        for gy in range(math.floor(y/16),math.floor((y+h)/16)+1):self._obstacle_grid.setdefault((gx,gy),[]).append((x,y,w,h,z))
        return max((z for x,y,w,h,z in self._obstacle_grid.get((math.floor(point[0]/16),math.floor(point[1]/16)),()) if x<=point[0]<=x+w and y<=point[1]<=y+h),default=-math.inf)

    def validate(self,world):
        if parameters(self.settings)!=self.settings or self.mode not in ('DEPLOY','FOLD'):raise ValueError('Поврежден паспорт собственного PV')
        all_ids=[*self.roof_panel_ids,*(id for q in self.sections for id in q.panel_ids)]
        if len(all_ids)!=len(set(all_ids)) or set(all_ids)!={o.id for o in world.inventory.items.values() if o.kind=='SOLAR_PANEL'}:raise ValueError('Панель потеряна или назначена двум носителям')
        for q in self.sections:
            if self.module(world,q).get('role')!='PV' or q.id not in world.inventory.items:raise ValueError('Отсутствует собственный PV-носитель')
            if q.state not in ('STOWED','DEPLOY_SUPPORTS','UNFOLDING','CONNECTING','DEPLOYED','DISCONNECTING','FOLDING','RETRACT_SUPPORTS') or not all(0<=v<=1 for v in (q.supports,q.open_fraction,q.connection)):raise ValueError('Повреждена стадия раскрытия PV')
            if q.state=='STOWED' and max(q.supports,q.open_fraction,q.connection)>0 or q.state=='DEPLOYED' and min(q.supports,q.open_fraction,q.connection)<1:raise ValueError('Несогласованная геометрия закреплённых крыльев')
            if (not math.isfinite(q.elapsed) or q.elapsed<0 or not math.isfinite(q.actuation_energy_j) or q.actuation_energy_j<0 or
                [id for leaf in q.leaves for id in leaf]!=q.panel_ids or len(q.leaves)!=2*self.settings['leaf_pairs'] or
                abs(sum(q.leaf_masses_kg)-q.panel_mass_kg)>1e-7 or abs(world.inventory.items[q.id].mass_kg-q.frame_mass_kg)>1e-7):raise ValueError('Повреждены крепления или масса солнечной платформы')
            for id in q.panel_ids:
                o=world.inventory.items[id]
                if o.location!=L.BASE_MOUNT or o.holder_id!='base':raise ValueError('Закреплённая панель оказалась выгруженной')
        if world.sim.base.destination is not None and not self.stowed:raise ValueError('Сохранение содержит движение с раскрытыми крыльями')

    def report(self):
        return dict(mechanism=self.mechanism,mode=self.mode,issue=self.issue,roof_area_m2=self.roof_area_m2,installed_area_m2=self.installed_area,
            platform_count=len(self.sections),available_installers=0,settings=dict(self.settings),setup_seconds_estimate=self.duration(),
            states={s:sum(q.state==s for q in self.sections) for s in ('STOWED','DEPLOY_SUPPORTS','UNFOLDING','CONNECTING','DEPLOYED','DISCONNECTING','FOLDING','RETRACT_SUPPORTS')},
            platforms=[dict(id=q.id,platform_id=q.platform_id,state=q.state,area_m2=q.area_m2,supports=q.supports,open_fraction=q.open_fraction,connection=q.connection,
                mass_kg=q.panel_mass_kg+q.frame_mass_kg,energy_j=q.actuation_energy_j,reason=q.reason) for q in self.sections])
