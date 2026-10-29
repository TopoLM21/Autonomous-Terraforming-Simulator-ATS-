"""Finite logistics layered on the original terrain, drone and solar physics."""
from dataclasses import dataclass, field, asdict
import math
import numpy as np
from .inventory import Inventory, Item, BatteryModule, SoilContainer, Rigging, ChargingSlot, Location as L
from .config import KWH
from .drone import DroneState as DS
from .simulation import MissionState
from .laser import prepare_links
from .soil import cargo_volume


@dataclass
class TransportJob:
    id: str
    kind: str
    resource_ids: list
    origin: tuple
    destination: tuple
    priority: int
    created_at: float
    status: str = 'QUEUED'
    executor: int | None = None
    phase: str = 'PREPARE'
    elapsed: float = 0.
    estimated_energy_j: float = 0.
    eta_seconds: float = 0.
    reason: str = ''
    soil_destination: str = 'dump'


class Logistics:
    archived_jobs=None
    battery_platforms=None  # old checkpoints keep their original service geometry
    def __init__(self, sim):
        self.sim, self.config = sim, sim.config.logistics
        self.inventory = Inventory()
        self.jobs = []
        self.archived_jobs=[]
        self.active = {}
        self._serial = 0
        self.delivered_mass = 0.
        self.progress_at = 0.
        self.excavator = None
        self.rack = np.array([sim.config.trench.x,sim.config.trench.y,0.])
        self.dock_owner = None
        self.initial_extra_energy = 0.
        self.excavator_consumed = 0.
        self.station_consumed = 0.
        self.swaps = 0
        self.group=None
        self.lift_count=0
        self.direct_ids=set()
        self.dedicated_ids=set()
        self.tool_changes={}
        self.tool_targets={}
        self.recall_ids=set()
        self.idle_since={}
        self.parking=set()
        self.wait_reasons={}
        self._bus_used=0.
        self._pending_station_energy=0.
        self._solar_ticked=False
        self._step_curtailed=0.
        self.wait_reason=''
        self.base_move_job=None
        self.base_pack_elapsed=0.
        self._base_moving_step=False
        self.base_battery_power_w=0.
        self.base_load_power_w=0.
        self.base_battery_ids=[]
        self._base_battery_serial=0
        self.courier_id=None
        self.courier_active=False
        self.courier_reason=''
        self.courier_batches=0
        self.deliveries=[]
        self._delivery_serial=0
        self.imported_mass_kg=0.
        self.exported_mass_kg=0.
        self.field_elapsed=0.
        self.field_ready=sim.base.config.panel_area<=20.
        self.field_rect=None
        self.solar_array=None
        base_packs=[]
        for recipe in sim.base.config.equipment.batteries:
            for _ in range(recipe.quantity):
                id=f'base_battery_{len(self.base_battery_ids)+1:02}'
                b=BatteryModule(id,recipe.mass_kg,L.BASE_MOUNT,'INSTALLED','base',
                    position=tuple(sim.base.position),compatibility='base',capacity_j=recipe.capacity_kwh*KWH,
                    energy_j=recipe.capacity_kwh*KWH*sim.base.config.initial_charge,
                    max_charge_w=recipe.capacity_kwh*2000.,max_discharge_w=recipe.capacity_kwh*10000.)
                base_packs.append(b);self.base_battery_ids.append(id)
        self.inventory.add_many(base_packs)
        self._base_battery_serial=len(self.base_battery_ids)
        eq=sim.base.config.equipment
        self.inventory.add(Item('base_chassis',eq.chassis_mass_kg,L.BASE_MOUNT,'INSTALLED','base',position=tuple(sim.base.position),kind='CHASSIS'))
        self.inventory.add(Item('base_charger',eq.charger_mass_kg,L.BASE_MOUNT,'INSTALLED','base',position=tuple(sim.base.position),kind='CHARGER_AND_AUTOMATION'))
        for i in range(eq.landing_slots):
            self.inventory.add(Item(f'landing_pad_{i+1}',eq.slot_mass_kg,L.BASE_MOUNT,'INSTALLED','base',position=tuple(sim.slot_position(i)),kind='LANDING_PAD'))
        panel_index=0;panel_items=[]
        for recipe in eq.panels:
            for _ in range(recipe.quantity):
                panel_index+=1
                panel_items.append(Item(f'solar_panel_{panel_index:03}',recipe.mass_kg,L.BASE_MOUNT,'INSTALLED','base',position=tuple(sim.base.position),kind='SOLAR_PANEL'))
        self.inventory.add_many(panel_items)
        for i,emitter in enumerate(eq.emitters):
            self.inventory.add(Item(f'laser_channel_{i+1}',emitter.mass_kg,L.BASE_MOUNT,'INSTALLED','base',position=tuple(sim.base.position),kind='LASER_TRANSMITTER'))
        if not self.field_ready and self.config.solar is None:
            try:self.field_rect=self.solar_field_site(sim.base.config.panel_area)
            except ValueError as error:sim.block(str(error))
        for i in range(self.config.container_count):
            self.inventory.add(SoilContainer(f'container_{i+1:02}',self.config.container_tare_kg,
                state='EMPTY_AT_BASE',position=tuple(sim.base.position),
                volume_m3=self.config.container_volume_m3,max_gross_kg=self.config.container_gross_kg))
        for d in sim.drones:
            d.battery_ids = []
            receiver=d.config.equipment.receiver
            if receiver.installed:
                self.inventory.add(Item(f'laser_receiver_{d.serial}',receiver.mass_kg,L.DRONE_MOUNT,'INSTALLED',f'drone_{d.serial}',position=tuple(d.position),kind='LASER_RECEIVER'))
            for recipe in d.config.equipment.batteries:
                for _ in range(recipe.quantity):
                    id=f'drone_{d.serial}_battery_{len(d.battery_ids)+1}'
                    self.inventory.add(BatteryModule(id,recipe.mass_kg,L.DRONE_MOUNT,'INSTALLED',
                        f'drone_{d.serial}',position=tuple(d.position),compatibility=f'drone:{recipe.capacity_kwh:g}:{recipe.mass_kg:g}',
                        capacity_j=recipe.capacity_kwh*KWH,energy_j=recipe.capacity_kwh*KWH*d.config.initial_charge,
                        max_charge_w=self.config.drone_pack_charge_w,max_discharge_w=self.config.drone_pack_discharge_w))
                    d.battery_ids.append(id)
            self.inventory.add(Rigging(f'grabber_{d.serial}',d.config.equipment.bucket.mass_kg,
                position=tuple(sim.base.position),kind='GRABBER'))
            d.max_battery_output_w=sum(self.inventory.items[id].max_discharge_w for id in d.battery_ids)
            self.inventory.add(Rigging(f'hook_{d.serial}',2.,L.DRONE_MOUNT,'INSTALLED',
                f'drone_{d.serial}',position=tuple(d.position),kind='CONTAINER_HOOK'))
            self.inventory.add(Rigging(f'hoist_{d.serial}',self.config.hoist_mass_kg,
                L.GROUND,'STORED_READY',f'drone_{d.serial}',position=tuple(d.position),kind='HOIST'))
            # Actual mounted hook replaces the grabber. Legacy configurations stay intact.
            d.attachment = 'CONTAINER_HOOK'
            self._tool_mass(d, 2.)
            if self.batch_enabled or self.config.direct_allowed and (not self.config.excavator_enabled or d is sim.drones[-1]):
                self.inventory.move(f'hook_{d.serial}',L.BASE_STORAGE,'base',sim.base.position,'STORED_READY')
                self.inventory.move(f'grabber_{d.serial}',L.DRONE_MOUNT,f'drone_{d.serial}',d.position,'INSTALLED')
                d.attachment='GRABBER';self._tool_mass(d,d.config.equipment.bucket.mass_kg)
                if self.config.direct_allowed:self.dedicated_ids.add(d.identifier)
        self.inventory.add(Rigging('group_rig',self.config.rigging_mass_kg,
                                   position=tuple(sim.base.position),kind='HEAVY_LIFT_RIG'))
        if self.config.excavator_enabled:
            from .excavator import Excavator
            c=self.config;q=sim.config.trench
            point=np.array([q.x,q.y-q.width/2-q.depth/sim.terrain.slope-c.support_half_width_m-.1,0.])
            point[2]=sim.terrain.sample(*point[:2])
            self.excavator=Excavator(c,point,buffer_j=c.buffer_kwh*KWH)
            self.rack=point.copy()
            for i in range(c.excavator_battery_count):
                installed=i<c.excavator_battery_slots
                b=BatteryModule(f'excavator_battery_{i+1:02}',c.pack_mass_kg,
                    L.EXCAVATOR_SLOT if installed else L.BASE_STORAGE,
                    'INSTALLED' if installed else 'STORED_READY','excavator' if installed else 'base',
                    position=tuple(point if installed else sim.base.position),
                    capacity_j=c.pack_capacity_kwh*KWH,
                    energy_j=c.pack_capacity_kwh*KWH*(c.excavator_initial_soc if installed else 1.),
                    max_charge_w=c.pack_charge_kw*1000.)
                self.inventory.add(b);self.initial_extra_energy+=b.energy_j
                if installed:self.excavator.battery_ids.append(b.id)
            self.initial_extra_energy+=self.excavator.buffer_j
            for container in self.containers:
                container.assigned_limit_kg=max(container.capacity_for(self.payload_limit(d),c.hoist_mass_kg,
                    max(sim.config.soil_profile.densities) if sim.config.soil_profile is not None else sim.config.terrain.density) for d in sim.drones)
                if self.batch_enabled:
                    self.inventory.move(container.id,L.WORKSITE_RACK,'worksite',self.rack,'EMPTY_AT_EXCAVATOR')
        for d in sim.drones:
            for set_index in range(self.config.drone_spare_sets):
                for slot,id in enumerate(d.battery_ids):
                    old=self.inventory.items[id]
                    b=BatteryModule(f'drone_{d.serial}_spare_{set_index+1}_{slot+1}',old.mass_kg,
                        position=tuple(sim.base.position),compatibility=old.compatibility,
                        capacity_j=old.capacity_j,energy_j=old.capacity_j*self.config.drone_spare_initial_soc,max_charge_w=old.max_charge_w,
                        max_discharge_w=old.max_discharge_w)
                    self.inventory.add(b);self.initial_extra_energy+=b.energy_j
        stock=[]
        for recipe in sim.initial_stock:
            for _ in range(recipe['count']):
                pack=BatteryModule(f'initial_stock_{len(stock)+1:05}',recipe['mass_kg'],position=tuple(sim.base.position),
                    compatibility=recipe['compatibility'],capacity_j=recipe['capacity_kwh']*KWH,
                    energy_j=recipe['capacity_kwh']*KWH*recipe['initial_soc'],max_charge_w=recipe['max_charge_w'],max_discharge_w=recipe['max_discharge_w'])
                stock.append(pack);self.initial_extra_energy+=pack.energy_j
        self.inventory.add_many(stock)
        from .batteries import BatteryService
        self.battery_service=BatteryService(self)
        if sim.base.config.layout is not None:
            from .base_layout import solve_layout,apply_live_layout
            apply_live_layout(self,solve_layout(sim.config,self))
            from .battery_platforms import enabled,BatteryPlatforms
            if enabled(sim.config):self.battery_platforms=BatteryPlatforms(self)
        if sim.base.config.panel_area>20 and (self.config.solar is not None or self.config.self_solar is not None):
            try:
                from .solar_array import SolarArray
                from .self_solar import SelfSolarArray
                self.solar_array=(SelfSolarArray if self.config.self_solar is not None else SolarArray)(self);self.solar_array.add_inventory(self)
                self._solar_bounds()
            except ValueError as error:sim.block(str(error))
        self.update_base_mass()

    def _solar_bounds(self):
        if self.solar_array and self.solar_array.reserved_rects:
            boxes=self.solar_array.reserved_rects;x=min(r[0] for r in boxes);y=min(r[1] for r in boxes)
            self.field_rect=(x,y,max(r[0]+r[2] for r in boxes)-x,max(r[1]+r[3] for r in boxes)-y)
        else:self.field_rect=None

    def field_setup_seconds(self):
        array=getattr(self,'solar_array',None)
        return max(1.,array.duration()) if array else self.config.field_deployment_seconds

    def solar_ground_rects(self):
        array=getattr(self,'solar_array',None)
        if self.field_rect is None:return []
        if array:return [] if getattr(self,'regional_folded',False) or array.mode=='FOLD' and array.stowed else array.reserved_rects
        return [self.field_rect]

    def plan_solar_field(self):
        array=getattr(self,'solar_array',None)
        if array:
            if not array.stowed:raise ValueError('Сначала физически сложите солнечные секции')
            if getattr(array,'mechanism',None)=='self_deploying':
                array.plan(self);self._solar_bounds();array.request_deploy();return
            from copy import deepcopy
            old=self.solar_array;self.solar_array=None;rect=self.field_rect;self.field_rect=None
            try:
                proposed=deepcopy(old);proposed.reserved_rects=[];proposed.plan(self,old.available_installers)
                ids=[id for id in old.support_ids if self.inventory.items[id].kind=='SOLAR_INSTALLER']
                for crew,id in zip(proposed.crews,ids):crew.id=id
                proposed.support_ids=list(old.support_ids)
            finally:self.solar_array=old;self.field_rect=rect
            self.solar_array=proposed;self._solar_bounds();proposed.request_deploy()
        else:self.field_rect=self.solar_field_site(self.sim.base.config.panel_area)

    def wake_for_solar_service(self):
        # A completed excavation can still receive equipment or fold its field.
        # Completion resumes after this finite service, without a new mission.
        if self.sim.state==MissionState.COMPLETED:
            self.sim.state=MissionState.RUNNING;self.sim.reason=''
        self.progress_at=self.sim.time

    def audit_solar_site(self):
        """Report unsafe historical layouts without removing any equipment."""
        from .site_ground import footprint_safe
        array=getattr(self,'solar_array',None);t=self.sim.terrain
        previous=getattr(self,'solar_site_issue','');self.solar_site_issue=''
        if array:
            array.validate(self)
            if getattr(array,'mechanism',None)=='self_deploying':
                array.audit(self);self.solar_site_issue=array.issue;return array.issue
            if array.issue.startswith('BLOCKED_SOLAR_SITE:'):array.issue=''
            deployed=[q for q in array.sections if q.state=='DEPLOYED']
            array.invalid_sections=[q.id for q in deployed if not footprint_safe(t,q.rect,t.design_work|t.dump_mask,
                   grade_deg=array.settings['max_grade_deg'],relief_m=array.settings['support_travel_m'])]
            if array.invalid_sections:
                array.issue='BLOCKED_SOLAR_SITE: сохранённая секция на воде или непригодном грунте; генерация остановлена, оборудование сохранено'
                self.solar_site_issue=array.issue
        elif self.field_rect is not None:
            rect=self.field_rect
            if not footprint_safe(t,rect,t.design_work|t.dump_mask,grade_deg=8.,relief_m=max(rect[2:])*math.tan(math.radians(8))):
                self.solar_site_issue='BLOCKED_SOLAR_SITE: старое поле на воде или непригодном грунте; сложите и перепланируйте поле'
        from .base_layout import layout_footprints
        b=self.sim.base
        boxes=list(layout_footprints(b.layout_spec,b.position)) if b.config.layout is not None else [(b.position[0]-b.config.radius,b.position[1]-b.config.radius,2*b.config.radius,2*b.config.radius)]
        if any(not footprint_safe(t,box,grade_deg=math.degrees(math.atan(b.config.max_slope)),relief_m=float('inf')) for box in boxes):
            self.solar_site_issue+=' UNSAFE_BASE_SITE: сохранённая база на воде или непригодном грунте; техника сохранена, требуется безопасное размещение'
        if self.solar_site_issue and self.solar_site_issue!=previous:self.sim.event(self.solar_site_issue)
        return self.solar_site_issue

    @property
    def batch_enabled(self):
        return self.config.batch_courier and self.config.idle_digging and self.config.excavator_enabled

    def solar_field_site(self, area):
        if area<=20:return None
        s=self.sim;margin=s.terrain.cell;side=margin*s.terrain.height.shape[0]
        width=min(max(20.,math.sqrt(area)),side-2*margin);height=area/width
        pads=[s.slot_position(i) for i in range(len(s.slots))]+[s.wait_position(i) for i in range(len(s.drones))]
        pads += [d.position for d in s.drones if d.grounded]
        if self.excavator is not None:pads.append(self.excavator.position)
        sites=[]
        shared=getattr(s.terrain,'shared_site',None);grid=11 if shared is not None else 5
        for x in np.linspace(margin,max(margin,side-margin-width),grid):
            for y in np.linspace(margin,max(margin,side-margin-height),grid):
                if shared is not None and shared.ground_occupied(s.terrain.owner_id,(x,y,width,height)):continue
                mask=(s.terrain.x>=x)&(s.terrain.x<x+width)&(s.terrain.y>=y)&(s.terrain.y<y+height)
                from .site_ground import footprint_safe
                if height>side-2*margin or np.any(mask&(s.terrain.design_work|s.terrain.dump_mask)):continue
                if not footprint_safe(s.terrain,(x,y,width,height),grade_deg=8.,relief_m=max(width,height)*math.tan(math.radians(8))):continue
                if any(x-margin<=p[0]<=x+width+margin and y-margin<=p[1]<=y+height+margin for p in pads):continue
                sites.append((np.linalg.norm(np.array([x+width/2,y+height/2])-s.base.position[:2]),
                              (float(x),float(y),width,height)))
        if not sites:raise ValueError('BLOCKED_SOLAR_FIELD_SITE: поле панелей не помещается вне рабочих и посадочных площадок')
        return min(sites,key=lambda site:(site[0],site[1]))[1]

    def configure_panels(self, panels,*,dry_run=False,prospective_batteries=None):
        """Apply a validated external panel installation without recreating the world."""
        from copy import copy,deepcopy
        s=self.sim
        if s.base.destination is not None:raise ValueError('Дождитесь остановки базы перед изменением панелей')
        equipment=deepcopy(s.base.config.equipment);equipment.panels=deepcopy(panels);equipment.validate()
        if prospective_batteries is not None:equipment.batteries=deepcopy(prospective_batteries);equipment.validate()
        from .base_layout import preview_live_layout,apply_live_layout
        layout=preview_live_layout(self,panels=panels,batteries=equipment.batteries)
        attached=self.config.self_solar is not None
        if not attached and self.config.solar is not None and equipment.panel_area>20:
            from .solar_array import recipe,parameters
            settings=parameters(self.config.solar);groups,_,_,_=recipe(equipment.panels,settings)
            if any(mass+max(1.,area/settings['section_area_m2'])*settings['frame_mass_kg']>settings['installer_payload_kg'] for _,area,mass in groups):raise ValueError('Солнечная кассета тяжелее грузоподъёмности монтажника')
        array=getattr(self,'solar_array',None)
        if array and not array.stowed:
            # Validate the requested installation before scheduling teardown.
            # A conservative tile-area bound rejects impossible deliveries
            # atomically, without folding a perfectly working existing field.
            p=array.settings;side=s.terrain.height.shape[0]*s.terrain.cell
            packed=0. if attached else equipment.panel_area*(1+p['column_lane_m']/p['section_width_m'])*(1+p['row_lane_m']/(p['section_area_m2']/p['section_width_m']))
            if packed>side**2:raise ValueError('Солнечные секции с проездами не помещаются на карте')
            if dry_run:return 'waiting'
            self.pending_solar_panels=deepcopy(panels);array.request_fold()
            self.wake_for_solar_service()
            s.event('Изменение панелей ожидает физического возврата солнечных секций')
            return 'waiting'
        modern=(attached or self.config.solar is not None) and (equipment.panel_area>20 or attached and layout is not None and 'self_solar' in layout)
        rect=None if modern else self.solar_field_site(equipment.panel_area)
        deployed=min(self.generating_area(),equipment.panel_area)
        old_support=sum(o.mass_kg for o in self.inventory.items.values() if o.kind in ('SOLAR_FRAME','SOLAR_INSTALLER'))
        items={id:item for id,item in self.inventory.items.items() if item.kind not in ('SOLAR_PANEL','SOLAR_FRAME','SOLAR_INSTALLER')}
        index=0
        for recipe in equipment.panels:
            for _ in range(recipe.quantity):
                index+=1;id=f'solar_panel_{index:03}'
                items[id]=Item(id,recipe.mass_kg,L.BASE_MOUNT,'INSTALLED','base',
                               position=tuple(s.base.position),kind='SOLAR_PANEL')
        Inventory(items).validate()
        proposed=None
        if modern:
            from .solar_array import SolarArray
            from .self_solar import SelfSolarArray
            from .power_design import equipment_copy
            probe=copy(self);actor=copy(s);actor.config=equipment_copy(s.config);actor.base=copy(s.base)
            actor.base.config=actor.config.base;actor.base.config.equipment=equipment
            if layout is not None:
                actor.base.layout_spec=layout;actor.base.config.radius=layout.get('hub_radius_m',layout['dimensions_m'][0]/2);actor.base.config.roof_height=layout.get('deck_height_m',layout['dimensions_m'][2])
                actor.base.config.layout=dict(layout['parameters'])
            probe.sim=actor;probe.config=actor.config.logistics;actor.logistics=probe
            probe.inventory=Inventory(items);probe.solar_array=None;probe.field_rect=None
            if self.battery_platforms and layout is not None:
                from .battery_platforms import CarrierState
                probe.battery_platforms=copy(self.battery_platforms);probe.battery_platforms.world=probe
                probe.battery_platforms.bindings=layout['bindings']
                probe.battery_platforms.carriers={a['id']:deepcopy(self.battery_platforms.carriers.get(a['id'],CarrierState(a['id']))) for a in layout['platforms'] if a['role'] in ('BESS','SERVICE')}
            proposed=(SelfSolarArray if attached else SolarArray)(probe)
        if dry_run:return 'applied'
        old_mass=sum(o.mass_kg for o in self.inventory.items.values() if o.kind=='SOLAR_PANEL')
        new_mass=sum(p.quantity*p.mass_kg for p in equipment.panels)
        self.imported_mass_kg+=max(0.,new_mass-old_mass);self.exported_mass_kg+=max(0.,old_mass-new_mass)
        self.inventory.items=items
        self.solar_array=None
        s.base.config.equipment=equipment;equipment.resolve(s.base.config,s.config.fleet)
        if layout is not None:
            if s.base.config.layout is None and attached:self.config.battery_platforms=None
            apply_live_layout(self,layout)
        self.field_rect=rect;self.base_pack_elapsed=0.
        self.field_ready=rect is None or deployed>=equipment.panel_area
        self.field_elapsed=(self.config.field_deployment_seconds*deployed/equipment.panel_area
                            if rect is not None else 0.)
        if proposed:
            self.solar_array=proposed;proposed.add_inventory(self);self._solar_bounds()
            self.field_ready=False;self.field_elapsed=0.
        self.solar_site_issue=''
        new_support=sum(o.mass_kg for o in self.inventory.items.values() if o.kind in ('SOLAR_FRAME','SOLAR_INSTALLER'))
        self.imported_mass_kg+=max(0.,new_support-old_support);self.exported_mass_kg+=max(0.,old_support-new_support)
        if s.shared_site is not None:s.shared_site.invalidate_geometry()
        self.update_base_mass()
        s.event(f'Панели изменены без сброса: {equipment.panel_area:g} м², {new_mass:g} кг; '+
                ('развёртывание поля' if rect else 'компактная установка'))
        self.wake_for_solar_service()
        return 'applied'

    def configure_base_batteries(self, recipes):
        """Install empty base packs, retaining real IDs, charge and all jobs."""
        from copy import deepcopy
        from collections import defaultdict,deque
        s=self.sim
        if s.base.destination is not None:raise ValueError('Дождитесь остановки базы перед изменением аккумуляторов')
        equipment=deepcopy(s.base.config.equipment)
        equipment.batteries=deepcopy(recipes);equipment.validate()
        from .base_layout import preview_live_layout,apply_live_layout
        available=defaultdict(deque)
        old=[self.inventory.items[id] for id in self.base_battery_ids]
        for pack in old:available[(pack.capacity_j,pack.mass_kg)].append(pack)
        retained=[];added=[];serial=self._base_battery_serial
        for recipe in equipment.batteries:
            for _ in range(recipe.quantity):
                key=(recipe.capacity_kwh*KWH,recipe.mass_kg)
                if available[key]:retained.append(available[key].popleft())
                else:
                    serial+=1
                    added.append(BatteryModule(f'base_battery_{serial:02}',recipe.mass_kg,
                        L.BASE_MOUNT,'INSTALLED','base',position=tuple(s.base.position),
                        compatibility='base',capacity_j=key[0],energy_j=0.,
                        max_charge_w=recipe.capacity_kwh*2000.,max_discharge_w=recipe.capacity_kwh*10000.))
        removed=[pack for queue in available.values() for pack in queue]
        if any(pack.reserved_by for pack in removed):raise ValueError('Нельзя снять зарезервированный аккумулятор базы')
        layout=preview_live_layout(self,batteries=recipes,base_records=[*retained,*added])
        items=dict(self.inventory.items)
        for pack in removed:del items[pack.id]
        for pack in added:items[pack.id]=pack
        Inventory(items).validate()
        old_capacity=sum(pack.capacity_j for pack in old)
        retained_capacity=sum(pack.capacity_j for pack in retained)
        retained_energy=min(s.base.battery,s.base.battery*retained_capacity/old_capacity) if old_capacity else 0.
        exported_energy=s.base.battery-retained_energy
        # The existing shared base bus represents a common state of charge.
        for pack in retained:pack.energy_j=retained_energy*pack.capacity_j/retained_capacity
        self.inventory.items=items
        self.base_battery_ids=[pack.id for pack in [*retained,*added]]
        self._base_battery_serial=serial
        self.imported_mass_kg+=sum(pack.mass_kg for pack in added)
        self.exported_mass_kg+=sum(pack.mass_kg for pack in removed)
        s.equipment_exported_energy+=exported_energy
        s.base.battery=retained_energy
        s.base.config.equipment=equipment;equipment.resolve(s.base.config,s.config.fleet)
        if layout is not None:apply_live_layout(self,layout)
        self.update_base_mass();self.progress_at=s.time
        s.event(f'Аккумуляторы базы: {len(self.base_battery_ids)} шт., {equipment.capacity/KWH:g} кВт·ч; '+
                f'добавлено {len(added)} пустых, снято {len(removed)}; вывоз {exported_energy/KWH:.3f} кВт·ч')

    def replenish_batteries(self, template_id, count, soc=1.):
        """External shipment: new IDs, slot hardware and explicitly imported energy."""
        template=self.inventory.items.get(template_id)
        if not isinstance(template,BatteryModule) or template.compatibility=='base':
            raise ValueError('Выберите тип сменной батареи дрона или Крота')
        if not isinstance(count,int) or isinstance(count,bool) or not 1<=count<=128:
            raise ValueError('Поставка: от 1 до 128 модулей за операцию')
        if not isinstance(soc,(int,float)) or not math.isfinite(soc) or not 0<=soc<=1:
            raise ValueError('Заряд поставки: от 0 до 100%')
        packs=[]
        for i in range(count):
            id=f'delivered_battery_{self._delivery_serial+i+1:05}'
            packs.append(BatteryModule(id,template.mass_kg,position=tuple(self.sim.base.position),
                compatibility=template.compatibility,capacity_j=template.capacity_j,energy_j=template.capacity_j*soc,
                max_charge_w=template.max_charge_w,max_discharge_w=template.max_discharge_w))
        slots=self.battery_service.make_slots(packs)
        from .base_layout import preview_live_layout,apply_live_layout
        layout=preview_live_layout(self,extra=[(pack.mass_kg+(self.config.charger_slot_mass_kg if self.config.individual_chargers else 0.),pack.capacity_j/KWH,pack.id) for pack in packs])
        self.inventory.add_many([*packs,*slots])
        self.battery_service.slots.update((slot.module_id,slot) for slot in slots)
        self._delivery_serial+=count
        energy=sum(b.energy_j for b in packs);mass=sum(item.mass_kg for item in [*packs,*slots])
        self.sim.equipment_imported_energy+=energy;self.imported_mass_kg+=mass
        if layout is not None:apply_live_layout(self,layout)
        self.deliveries.append(dict(time_seconds=self.sim.time,module_ids=[b.id for b in packs],
            compatibility=template.compatibility,soc=soc,energy_j=energy,mass_kg=mass))
        self.battery_service.update_slot_states();self.update_base_mass()
        self.progress_at=self.sim.time
        if self.sim.state==MissionState.BLOCKED and any(reason in self.sim.reason for reason in ('NO_SPARE_BATTERIES','NO_ENERGY')):
            self.sim.state=MissionState.RUNNING;self.sim.reason=''
        self.sim.event(f'Поставка: {count} батарей {template.compatibility}, SOC {soc:.0%}; '+
                       f'ввоз {energy/KWH:.3f} кВт·ч и {mass:g} кг вместе со слотами')
        return [b.id for b in packs]

    def mobile_items(self):
        # Installed base packs are already counted by the base equipment and
        # use their own bus synchronization. Locations of the returned items
        # remain live; only this structural selection is cached.
        signature=(id(self.inventory.items),len(self.inventory.items),id(self.base_battery_ids),len(self.base_battery_ids))
        if getattr(self,'_mobile_signature',None)!=signature:
            self._mobile_signature=signature;base=set(self.base_battery_ids)
            self._mobile_items=[o for o in self.inventory.mutable_items() if o.id not in base]
        return self._mobile_items

    def update_base_mass(self):
        self.sim.base.config.mass=self.sim.base.config.equipment.mass+sum(
            o.gross_mass for o in self.mobile_items()
            if o.holder_id=='base' and (o.location in (L.BASE_STORAGE,L.BASE_CHARGER) or isinstance(o,ChargingSlot)))
        array=getattr(self,'solar_array',None)
        if array:
            self.sim.base.config.mass-=array.offboard_panel_mass()
            crane=array.crane_supported()
            self.sim.base.config.mass+=sum(self.inventory.items[id].mass_kg for id in array.support_ids if self.inventory.items[id].holder_id=='base' or id in crane)
        if self.battery_platforms:
            self.sim.base.config.mass+=sum(o.mass_kg for o in self.inventory.items.values() if o.kind=='BATTERY_PLATFORM_HARDWARE')
            self.sim.base.config.mass+=sum(o.mass_kg for o in self.inventory.select(BatteryModule) if o.location==L.IN_TRANSFER and o.holder_id.startswith('station_transfer_'))

    def _tool_mass(self, d, tool_mass):
        e=d.config.equipment
        d.config.empty_mass=e.mass-e.bucket.mass_kg+tool_mass
        d.config.hover_power=e.reference_hover_power_kw*1000*(d.config.empty_mass/e.reference_mass_kg)**1.5

    def payload_limit(self, d):
        from .equipment import physical_payload_limit
        return physical_payload_limit(d)

    @property
    def containers(self):
        return self.inventory.select(SoilContainer)

    def new_job(self, kind, item, origin, destination, priority=3):
        if item.reserved_by or any(j.status in ('QUEUED','ACTIVE') and item.id in j.resource_ids for j in self.jobs):
            return None
        self._serial+=1
        job=TransportJob(f'job_{self._serial:05}',kind,[item.id],tuple(origin),tuple(destination),
                         priority,self.sim.time)
        if not self.inventory.reserve(job.resource_ids,job.id):
            return None
        self.jobs.append(job)
        return job

    def cancel_job(self, id):
        job=next(j for j in self.jobs if j.id==id)
        if job.status=='ACTIVE':
            raise ValueError('Airborne cargo must finish or enter explicit recovery')
        job.status='CANCELLED';self.inventory.release(job.id)

    def generate_jobs(self):
        s=self.sim
        regional=getattr(s,'regional_stage',None)
        if regional=='COLLECT':
            for item in self.inventory.mutable_items():
                if item.location in (L.WORKSITE_RACK,L.FILLED_QUEUE,L.DUMP_SITE,L.GROUND):
                    if item.kind=='HOIST' and item.holder_id.startswith('drone_'):continue
                    destination=self.battery_platforms.receiving_position(item.id) if self.battery_platforms and isinstance(item,BatteryModule) else s.base.position
                    self.new_job('PACK_FOR_RELOCATION',item,item.position,destination,2)
            return
        if regional in ('PACK','DRIVE','ARRIVE','DEPLOY'):return
        self.battery_service.generate_jobs()
        for c in self.containers:
            # Reservations already own an immutable route until pickup. Do not
            # redo the spatial search, or report a full dump for assigned cargo.
            if c.reserved_by:continue
            if c.soil_kg > 1e-8 and c.location==L.FILLED_QUEUE:
                incoming=sum(self.inventory.items[j.resource_ids[0]].soil_kg for j in self.jobs
                    if j.kind=='PICKUP_FULL_CONTAINER' and j.soil_destination=='fill' and j.status in ('QUEUED','ACTIVE'))
                fill=s.terrain.fill_location() if s.terrain.remaining_fill*s.config.terrain.density>incoming+1e-8 else None
                if s.terrain.soil is not None:
                    incoming=sum(self.inventory.items[j.resource_ids[0]].soil_volume(s.config.terrain.density) for j in self.jobs
                        if j.kind=='PICKUP_FULL_CONTAINER' and j.soil_destination=='fill' and j.status in ('QUEUED','ACTIVE'))
                    incoming+=sum(d.trip.volume or 0. for d in s.drones if d.trip is not None and d.trip.destination=='fill'
                        and not d.dump_released and d.state not in (DS.RETURN,DS.LANDING))
                    fill=s.terrain.fill_location() if s.terrain.remaining_fill>incoming+1e-10 else None
                dump=fill if fill is not None else s.terrain.dump_location(c.soil_volume(s.config.terrain.density),
                    s._reserved_dump(None),s.config.fleet.separation,near=c.position)
                if dump is not None:
                    job=self.new_job('PICKUP_FULL_CONTAINER',c,c.position,(*dump,s.terrain.sample(*dump)),3)
                    if job is not None:job.soil_destination='fill' if fill is not None else 'dump'
                else:
                    s.block('BLOCKED_DUMP_FULL: нет места для контейнерного груза')
            elif regional is None and c.location==L.BASE_STORAGE and c.soil_kg==0 and self.excavator is not None:
                self.new_job('DELIVER_EMPTY_CONTAINER',c,c.position,self.rack,4)
            elif (regional is None and c.location==L.WORKSITE_RACK and c.soil_kg==0 and self.excavator is not None
                  and np.linalg.norm(np.array(c.position[:2])-self.rack[:2])>self.config.reach_m):
                self.new_job('RETURN_EMPTY_CONTAINER',c,c.position,self.rack,4)

    def estimate(self, d, job):
        altitude=self.sim.cruise(d)
        item=self.inventory.items[job.resource_ids[0]]
        load=item.gross_mass+self.config.hoist_mass_kg
        if load>self.payload_limit(d)+1e-7:
            return math.inf,math.inf
        if d.power(winching=True,cargo=load,lift_height=max(0.,altitude-job.origin[2]))>d.max_battery_output_w:
            return math.inf,math.inf
        home=d.position
        empty=d.power(moving=True,cargo=0)
        loaded=d.power(moving=True,cargo=load)
        out=math.ceil(np.linalg.norm(home[:2]-job.origin[:2])/d.config.empty_speed)
        haul=math.ceil(np.linalg.norm(np.array(job.origin[:2])-job.destination[:2])/d.config.loaded_speed)
        back=math.ceil(np.linalg.norm(np.array(job.destination[:2])-home[:2])/d.config.empty_speed)
        vertical=2*math.ceil(max(0.,altitude-home[2])/d.config.vertical_speed)
        # Includes return of a reusable empty container and traffic allowance.
        if job.kind=='PICKUP_FULL_CONTAINER':
            back += math.ceil(np.linalg.norm(np.array(job.destination[:2])-job.origin[:2])/d.config.loaded_speed)
        handling=6*self.config.handling_seconds
        seconds=out+haul+back+vertical+handling+self.sim.config.fleet.emergency_landing_seconds
        energy=(empty*(out+back+vertical+self.sim.config.fleet.emergency_landing_seconds)
                +loaded*(haul+handling)+load*d.config.gravity*max(0.,altitude-job.origin[2])/d.config.winch_efficiency)
        return energy*d.config.budget_margin,seconds

    def dispatch(self):
        s=self.sim
        if self._base_moving_step or s.base.destination is not None or s.relocation_pending is not None:return
        pending=sorted((j for j in self.jobs if j.status=='QUEUED'),
                       key=lambda j:(j.priority-(s.time-j.created_at)/600,j.created_at,j.id))
        for d in s.drones:
            if (d.identifier in self.active or not d.grounded
                    or d.identifier in self.recall_ids
                    or self.group is not None or d.identifier in self.direct_ids
                    or d.serial in self.battery_service.drone_swaps):
                continue
            if self.batch_enabled and (not self.courier_active or d.identifier!=self.courier_id):continue
            if d.attachment!='CONTAINER_HOOK' or d.identifier in self.tool_targets:continue
            if s._charging_needed(d):
                if d.docked:d.state=DS.CHARGING
                else:s._enqueue(d)
                continue
            for job in pending:
                if job.status!='QUEUED':continue
                if s.landing_requested and job.kind!='PICKUP_FULL_CONTAINER' and not (
                        getattr(s,'regional_stage',None) and job.kind in ('PACK_FOR_RELOCATION','DELIVER_EXCAVATOR_BATTERY','RECOVER_EMPTY_EXCAVATOR_BATTERY')):continue
                energy,eta=self.estimate(d,job)
                if energy > d.battery-d.config.battery*d.config.reserve:
                    job.reason='WAITING_ENERGY_OR_PAYLOAD'
                    if energy <= d.config.battery*(1-d.config.reserve):
                        s._request_charge(d,'резерв следующего грузового рейса')
                    continue
                hoist=self.inventory.items[f'hoist_{d.serial}']
                if not self.inventory.reserve([hoist.id],job.id):continue
                job.status,job.executor,job.estimated_energy_j,job.eta_seconds='ACTIVE',d.serial,energy,eta
                job.resource_ids.append(hoist.id)
                self.active[d.identifier]=job
                if d.identifier in s.queue:s.queue.remove(d.identifier)
                self.inventory.move(hoist.id,L.DRONE_CARGO,f'drone_{d.serial}',d.position,'IN_USE',job.id)
                d.cargo=hoist.mass_kg
                s.drone_event(d,f'{job.kind}: {job.resource_ids[0]}')
                break

    def try_lift(self):
        if getattr(self.sim,'regional_stage',None):return
        from .cooperative_lift import CollectiveLift,lift_check
        from copy import copy
        e=self.excavator;s=self.sim
        if e is None or e.state.value!='WAITING_RELOCATION' or self.group is not None or self._base_moving_step or s.base.destination is not None or s.relocation_pending is not None:return
        if (self.active or self.direct_ids or self.tool_targets or self.recall_ids or self.battery_service.drone_swaps
                or any(not d.grounded for d in s.drones)):return
        rig=self.inventory.items['group_rig']
        if np.linalg.norm(np.asarray(rig.position[:2])-e.position[:2])>self.config.reach_m:
            self.new_job('DELIVER_RIGGING',rig,rig.position,e.position,6)
            return
        if not self.config.air_relocation:
            s.block('BLOCKED_LOCAL_PATH: воздушный перенос отключён');return
        preview=copy(e);preview.battery_ids=[]
        distance=float(np.linalg.norm(e.destination[:2]-e.position[:2]))
        check=lift_check(preview,self.inventory,s.drones,self.config,distance)
        if not check['feasible']:
            e.reason=check['reason']
            if check['reason']=='BLOCKED_INSUFFICIENT_LIFT':s.block(check['reason'])
            else:
                for d in s.drones:
                    if d.serial in check['waiting_drones']:s._request_charge(d,'запас для коллективного подъёма')
            return
        self.group=CollectiveLift(self,e.destination.copy(),list(s.drones))
        s.event('Команда зарезервирована: снятие батарей и перенос сухого Крота')

    def direct_should_return(self, d):
        return (self.sim.relocation_pending is not None or d.identifier in self.tool_targets or d.identifier in self.recall_ids or
                self.excavator is not None and self.excavator.state.value=='WAITING_RELOCATION')

    def request_tool(self, d, target):
        if d.attachment==target:
            self.inventory.release(f'tool_{d.serial}')
            if self.battery_platforms:self.battery_platforms.release(f'tool_{d.serial}')
            self.tool_targets.pop(d.identifier,None);self.tool_changes.pop(d.serial,None)
            if d.state==DS.CHANGING_TOOL:d.state=DS.DOCKED
        else:self.tool_targets[d.identifier]=target

    def reset_drone_order(self, serial):
        """Cancel atomic dock work; return flying cargo through its existing safe route."""
        d=next((d for d in self.sim.drones if d.serial==serial),None)
        if d is None:raise ValueError('Дрон не найден')
        in_group=self.group is not None and any(member is d for member in self.group.participants)
        if d.grounded and d.cargo>1e-8 and d.identifier not in self.active and not in_group:
            raise ValueError('Груз без маршрута: требуется восстановление перевозки, сброс груза запрещён')
        self.request_tool(d,d.attachment)
        self.battery_service.cancel_drone_swap(d.serial)
        self.recall_ids.add(d.identifier)
        if d.identifier in self.active:
            self.active[d.identifier].reason='RECALL_REQUESTED'
            result='Отмена принята: дрон завершит доставку груза и вернётся'
        elif in_group:
            result='Отмена принята: дрон завершит совместный перенос и вернётся'
        elif d.identifier in self.direct_ids:
            if d.cargo<=1e-8 and d.grabber_extension==0 and d.state in (DS.TAKEOFF,DS.TO_DIG):
                d.trip=None
                if d.slot is not None and self.sim.slots[d.slot]==d.identifier:
                    d.destination=self.sim.slot_position(d.slot);d.transition(DS.LANDING)
                else:
                    d.transition(DS.RETURN);self.sim._reserve_return(d)
            result='Отмена принята: груз будет выгружен, затем дрон вернётся'
        elif d.grounded and d.serial not in self.battery_service.drone_swaps:
            self.finish_drone_recall(d)
            return 'Приказ сброшен, резервы освобождены; автономный диспетчер продолжит работу'
        elif d.grounded:result='Отмена принята: завершается безопасный возврат батарей в шкаф'
        else:
            if d.state==DS.WAIT_HOVER:
                d.destination=self.sim.wait_position(d.identifier);d.transition(DS.WAIT_LANDING)
            result='Отмена принята: дрон завершает посадку'
        self.sim.drone_event(d,result)
        return result

    def finish_drone_recall(self, d):
        self.recall_ids.discard(d.identifier)
        self.request_tool(d,d.attachment)
        self.idle_since.pop(d.identifier,None);self.wait_reasons.pop(d.identifier,None)
        if self.courier_id==d.identifier:self.courier_id=None
        d.trip=d.destination=d.detour=None;d.grabber_extension=0.
        d.transition(DS.CHARGING if d.slot is not None and d.charge_requested else
                     DS.DOCKED if d.slot is not None else DS.WAIT_GROUND)
        if not d.charge_requested and d.identifier in self.sim.queue:self.sim.queue.remove(d.identifier)
        elif d.slot is None:self.sim._enqueue(d)
        self.sim.drone_event(d,'Текущий приказ сброшен; автономное планирование возобновлено')

    def batch_tool_requests(self, pending, relocating):
        """Latch a service batch, assigning exactly one courier until it is drained."""
        s=self.sim;e=self.excavator
        filled=sum(c.location==L.FILLED_QUEUE and c.soil_kg>1e-8 for c in self.containers)
        threshold=max(1,math.ceil(len(self.containers)*self.config.courier_fill_fraction))
        capacity=e.capacity(self.inventory)
        low=(not e.battery_ids or e.energy(self.inventory)<capacity*self.config.courier_low_soc)
        finishing=s.terrain.remaining_cut<=s.config.terrain.completion_volume or s.landing_requested
        stranded=e.state.value in ('WAITING_TRANSPORT','WAITING_EMPTY_CONTAINER','LOW_BATTERY')
        reason=('REGIONAL_MOVE' if getattr(s,'regional_stage',None) and pending else 'LOW_SOC' if low else 'FILLED_THRESHOLD' if filled>=threshold else
                'FINISH_BATCH' if finishing and filled else 'RELOCATION' if relocating and pending else
                'CLEAR_WORKSITE' if stranded and pending else '')
        if not self.courier_active and reason:
            self.courier_active=True;self.courier_reason=reason;self.courier_batches+=1
            s.event(f'Партия курьера: {reason}; готово ёмкостей {filled}/{len(self.containers)}')
        if (self.courier_active and not reason and not pending and not self.active
                and self.battery_service.robot_swap is None):
            self.courier_active=False;self.courier_reason=''
            s.event('Партия курьера завершена: возвращается к копанию')
        if not self.courier_active:
            if self.courier_id is not None and not getattr(s,'regional_stage',None):
                self.request_tool(s.drones[self.courier_id],'GRABBER')
            self.courier_id=None
            return
        previous=s.drones[self.courier_id] if self.courier_id is not None else None
        # Never hand an airborne transport job to another aircraft. A grounded
        # courier that needs its own full recharge may be relieved by a ready digger.
        if previous is not None and (previous.identifier in self.active or
                not previous.charge_requested and previous.identifier not in self.recall_ids):
            self.request_tool(previous,'CONTAINER_HOOK');return
        from copy import copy
        choices=[]
        for d in s.drones:
            if d.charge_requested or d.identifier in self.active or d.identifier in self.recall_ids or d.serial in self.battery_service.drone_swaps:continue
            preview=copy(d);preview.config=copy(d.config);self._tool_mass(preview,2.)
            if pending and not any(self.estimate(preview,j)[0]<=d.battery-d.config.battery*d.config.reserve for j in pending):continue
            choices.append(d)
        if choices:
            selected=min(choices,key=lambda d:(not d.grounded,d.attachment!='CONTAINER_HOOK',-d.battery/d.config.battery,d.identifier))
            if previous is not None and previous is not selected:self.request_tool(previous,'GRABBER')
            self.courier_id=selected.identifier;self.request_tool(selected,'CONTAINER_HOOK')
            s.drone_event(selected,'назначен единственным курьером партии')

    def service_tools(self):
        """Return to an actual pad, then exchange two physical attachments."""
        s=self.sim
        pending=sorted((j for j in self.jobs if j.status=='QUEUED'),
                       key=lambda j:(j.priority-(s.time-j.created_at)/600,j.created_at,j.id))
        relocating=self.excavator is not None and self.excavator.state.value=='WAITING_RELOCATION'
        if self.batch_enabled:
            self.batch_tool_requests(pending,relocating)
        elif relocating:
            for d in s.drones:
                if self.tool_targets.get(d.identifier)=='GRABBER':self.request_tool(d,'CONTAINER_HOOK')
        # A charging carrier cannot cover a waiting job. Match actual ready
        # aircraft (including pending tool exchanges) before recalling diggers.
        from copy import copy
        candidates=[] if self.batch_enabled else sorted(s.drones,key=lambda d:(d.attachment!='CONTAINER_HOOK',d.identifier))
        for d in candidates:
            if not pending:break
            if d.charge_requested or d.identifier in self.recall_ids or d.serial in self.battery_service.drone_swaps:continue
            if d.identifier in self.active or d.identifier in self.direct_ids:continue
            if not d.grounded and self.tool_targets.get(d.identifier)!='CONTAINER_HOOK':continue
            if d.attachment=='GRABBER' and not self.config.idle_digging:
                if any(o.attachment=='CONTAINER_HOOK' and not o.charge_requested for o in s.drones):continue
            preview=copy(d);preview.config=copy(d.config);self._tool_mass(preview,2.)
            job=next((j for j in pending if self.estimate(preview,j)[0]<=d.battery-d.config.battery*d.config.reserve),None)
            if job is None:continue
            self.request_tool(d,'CONTAINER_HOOK')
            pending.remove(job)
        if not self.batch_enabled and pending and not any(d.attachment=='CONTAINER_HOOK' and not d.charge_requested
                               or self.tool_targets.get(d.identifier)=='CONTAINER_HOOK'
                               for d in s.drones):
            # All ready diggers may currently be in flight. Finish one delivery
            # and bring its hook back instead of waiting for a charging carrier.
            recall=next((d for d in s.drones if d.identifier in self.direct_ids and not d.charge_requested and d.identifier not in self.recall_ids),None)
            if recall is not None:self.request_tool(recall,'CONTAINER_HOOK')
        for d in s.drones:
            if d.identifier not in self.tool_targets or d.identifier in self.direct_ids or d.identifier in self.recall_ids:continue
            if d.charge_requested or s.landing_requested and self.tool_targets[d.identifier]!='CONTAINER_HOOK':continue
            if d.state==DS.WAIT_GROUND and s.base.destination is None:
                slot=next((i for i,v in enumerate(s.slots) if v is None),None)
                if slot is None:continue
                energy=s._transfer_energy(d,slot)
                if d.battery-energy<d.config.battery*d.config.reserve:
                    s._request_charge(d,'переход к станции смены инструмента');s._enqueue(d);continue
                d.slot=slot;s.slots[slot]=d.identifier;d.destination=s.slot_position(slot)
                d.transition(DS.TRANSFER_TAKEOFF)
                s.drone_event(d,f'возвращается в слот {slot+1} для смены инструмента')

    def dock_service_ready(self,slot=None,continuing=False):
        """The workshop needs a stationary base and real bus power, not all PV.

        Historical aggregate fields retain their original commissioning policy.
        Existing swaps may finish while a local move is pending; BatteryService
        separately prevents starting another swap during that preparation.
        """
        s=self.sim
        if s.base.destination is not None or getattr(self,'regional_folded',False):return False
        if getattr(s,'regional_stage',None) in ('DRIVE','ARRIVE'):return False
        if self.battery_platforms:
            if slot is None:return any(self.battery_platforms.service_ready(d.slot,continuing=continuing) for d in s.drones if d.docked)
            return self.battery_platforms.service_ready(slot,continuing=continuing)
        if getattr(s,'regional_stage',None)=='PACK':return False
        if getattr(self,'solar_array',None) is not None:return True
        return self.field_ready and not self.base_pack_elapsed

    def tick_tool_changes(self):
        """Reserve workshop power before battery chargers, without changing inventory early."""
        s=self.sim
        if not self.dock_service_ready(continuing=True) or s.relocation_pending is not None:return
        for d in s.drones:
            if (d.identifier not in self.tool_targets or not d.docked or d.charge_requested
                    or d.identifier in self.active or d.identifier in self.direct_ids or d.identifier in self.recall_ids
                    or d.serial in self.battery_service.drone_swaps):continue
            if not self.dock_service_ready(d.slot,continuing=d.serial in self.tool_changes):
                self.wait_reasons[d.identifier]='Сервисная платформа не готова';continue
            if self.group is not None and any(member is d for member in self.group.participants):
                self.wait_reasons[d.identifier]='Ожидание завершения совместного переноса';continue
            owner=f'tool_{d.serial}'
            ids=[f'hook_{d.serial}',f'grabber_{d.serial}']
            missing=[id for id in ids if id not in self.inventory.items]
            if missing:
                self.wait_reasons[d.identifier]='Нет инструмента в инвентаре: '+', '.join(missing);continue
            mounted=f'grabber_{d.serial}' if d.attachment=='GRABBER' else f'hook_{d.serial}'
            wanted=f'grabber_{d.serial}' if self.tool_targets[d.identifier]=='GRABBER' else f'hook_{d.serial}'
            current,target=self.inventory.items[mounted],self.inventory.items[wanted]
            if current.location!=L.DRONE_MOUNT or current.holder_id!=f'drone_{d.serial}':
                self.wait_reasons[d.identifier]='Текущее крепление инструмента требует восстановления';continue
            if target.location!=L.BASE_STORAGE or target.holder_id!='base':
                self.wait_reasons[d.identifier]=f'Инструмент {wanted} ещё не доставлен на базу';continue
            if not self.inventory.reserve(ids,owner):
                self.wait_reasons[d.identifier]='Инструмент зарезервирован другим заданием';continue
            if self.battery_platforms and not self.battery_platforms.acquire(owner,d.slot):
                self.inventory.release(owner);self.wait_reasons[d.identifier]='Обменный механизм занят';continue
            d.state=DS.CHANGING_TOOL
            if not self.use_station(100.):
                self.wait_reasons[d.identifier]='нет энергии для привода станции';continue
            self.wait_reasons[d.identifier]='работает станция'
            elapsed=self.tool_changes.get(d.serial,0.)+1
            self.tool_changes[d.serial]=elapsed
            if elapsed<self.config.attachment_seconds:continue
            target=self.tool_targets[d.identifier]
            mounted=f'grabber_{d.serial}' if target=='GRABBER' else f'hook_{d.serial}'
            stored=next(id for id in ids if id!=mounted)
            stored_position=self.battery_platforms.slot_position(stored) if self.battery_platforms else s.base.position
            self.inventory.move(stored,L.BASE_STORAGE,'base',stored_position,'STORED_READY',owner)
            self.inventory.move(mounted,L.DRONE_MOUNT,f'drone_{d.serial}',d.position,'INSTALLED',owner)
            d.attachment=target;self._tool_mass(d,self.inventory.items[mounted].mass_kg)
            self.inventory.release(owner);self.tool_targets.pop(d.identifier);self.tool_changes.pop(d.serial)
            if self.battery_platforms:self.battery_platforms.release(owner)
            self.wait_reasons.pop(d.identifier,None)
            d.state=DS.DOCKED;s.drone_event(d,f'установлен инструмент {target}')

    def yield_service_pads(self):
        s=self.sim
        if self.group is not None:return
        waiting=bool(s.queue or any(s.drones[i].state==DS.WAIT_GROUND for i in self.tool_targets))
        if not waiting or s.base.destination is not None or s.landing_requested or s.relocation_pending is not None:return
        if any(slot is None for slot in s.slots):return
        if self.parking:return
        for d in s.drones:
            if (not d.docked or d.charge_requested or d.identifier in self.active or
                    d.identifier in self.direct_ids or d.identifier in self.tool_targets):continue
            energy=d.power(True)*(2*math.ceil(max(0.,s.cruise(d)-d.position[2])/d.config.vertical_speed)+
                                  math.ceil(s.config.fleet.waiting_offset/d.config.empty_speed)+10)
            if d.battery-energy<d.config.battery*d.config.reserve:continue
            d.destination=s.wait_position(d.identifier);d.transition(DS.RETURN)
            self.parking.add(d.identifier)
            s.drone_event(d,'освобождает сервисный слот; перелёт на площадку ожидания')
            break

    def direct_dispatch(self):
        if not (self.config.direct_allowed or self.config.idle_digging) or self.group is not None:return
        s=self.sim
        if self._base_moving_step or s.base.destination is not None or s.relocation_pending is not None:return
        if self.excavator is not None and self.excavator.state.value=='WAITING_RELOCATION':return
        # Transport assignments and necessary tool changes have already been
        # selected; unrelated queued cargo must not serialize the whole fleet.
        for d in s.drones:
            if (d.identifier in self.active or d.identifier in self.direct_ids or not d.grounded
                    or d.charge_requested or s.landing_requested or d.identifier in self.recall_ids):
                self.idle_since.pop(d.identifier,None);continue
            if d.identifier in self.tool_targets:continue
            if self.batch_enabled and self.courier_active and d.identifier==self.courier_id:continue
            if d.attachment!='GRABBER':
                if not self.config.idle_digging and d.identifier not in self.dedicated_ids:continue
                hooks=sum(o.attachment=='CONTAINER_HOOK' and self.tool_targets.get(o.identifier)!='GRABBER' for o in s.drones)
                if not self.batch_enabled and self.excavator is not None and hooks<=1 and len(s.drones)>1:continue
                since=self.idle_since.setdefault(d.identifier,s.time)
                if not self.batch_enabled and s.time-since<self.config.idle_dig_delay_s:continue
                # Preview with the real grabber mass, without changing hardware.
                from copy import copy
                preview=copy(d);preview.config=copy(d.config)
                self._tool_mass(preview,self.inventory.items[f'grabber_{d.serial}'].mass_kg)
                if self.direct_trip(d,preview) is None:continue
                self.request_tool(d,'GRABBER');continue
            trip=self.direct_trip(d)
            if trip is None:continue
            job_id=f'job_{self._serial+1:05}'
            resources=[f'grabber_{d.serial}']
            if not self.inventory.reserve(resources,job_id):
                self.wait_reasons[d.identifier]='Грейфер занят другим заданием'
                continue
            self._serial+=1
            job=TransportJob(job_id,'EXCAVATE_DIRECT',resources,
                trip.dig,trip.dump,5,s.time,status='ACTIVE',executor=d.serial)
            self.jobs.append(job);d.trip=trip;self.direct_ids.add(d.identifier)
            d.transition(DS.TAKEOFF)

    def direct_trip(self, d, preview=None):
        actor=preview if preview is not None else d
        trip=self.sim._choose_trip(actor)
        if trip is None:
            if self.sim._choose_trip(actor,battery=actor.config.battery) is not None:
                self.sim._request_charge(d,'недостаточно энергии на грейферный рейс с резервом')
                if d.docked:d.state=DS.CHARGING
                else:self.sim._enqueue(d)
            else:self.wait_reasons[d.identifier]='Нет свободного доступного участка'
        return trip

    def idle_reason(self, d):
        if not d.grounded:return ''
        if self.sim.relocation_pending is not None:
            return 'Подготовка переезда базы: '+self.sim.base_move_status
        if d.serial in self.battery_service.drone_swaps:return 'Замена батарей'
        if d.charge_requested:
            return 'Зарядка до 100%' if d.docked else 'Очередь к зарядному доку'
        if d.identifier in self.tool_targets:
            if d.state==DS.CHANGING_TOOL:return 'Смена инструмента: '+self.wait_reasons.get(d.identifier,'работает станция')
            detail=self.wait_reasons.get(d.identifier,'')
            return ('Ожидание станции смены инструмента'+(': '+detail if detail else '')) if d.docked else 'Очередь к станции смены инструмента'
        if self.group is not None:return 'Подготовка совместного переноса'
        if self.excavator is not None and self.excavator.state.value=='WAITING_RELOCATION':
            return 'Сбор команды для переноса Крота'
        if self.sim.landing_requested:return 'Посадка флота запрошена'
        if self.sim.terrain.remaining<=self.sim.config.terrain.completion_volume:return 'Завершение миссии'
        if d.identifier in self.active:return ''
        return self.wait_reasons.get(d.identifier,'Ожидание готового груза' if d.attachment=='CONTAINER_HOOK' else 'Ожидание участка')

    def generating_area(self):
        """Only physically unfolded panels contribute, including partial deployment."""
        if self.sim.base.destination is not None or getattr(self,'regional_folded',False):return 0.
        array=getattr(self,'solar_array',None)
        if array:return array.installed_area
        if getattr(self,'solar_site_issue',''):return 0.
        seconds=max(1.,self.field_setup_seconds())
        fraction=(max(0.,1-self.base_pack_elapsed/seconds) if self.base_pack_elapsed else
                  1. if self.field_ready else min(1.,self.field_elapsed/seconds))
        return self.sim.base.config.panel_area*fraction

    def use_station(self, energy):
        b=self.sim.base;s=self.sim
        if self._bus_used+energy>self.config.inverter_power_w:return False
        if not self._solar_ticked:
            solar=b.irradiance((s.hour+1/7200)%24,s.day_at(s.time+.5))*self.generating_area()*b.config.panel_efficiency
            discharge=self.storage_power('discharge')
            available=min(self.config.inverter_power_w,solar+min(b.battery,discharge))-b.config.idle_power
            if available<self._pending_station_energy+energy:return False
            self._pending_station_energy+=energy
        else:
            if b.battery+self._step_curtailed<energy:return False
            reclaimed=min(self._step_curtailed,energy);from_battery=energy-reclaimed
            used=max(0.,getattr(self,'_storage_before_tick',b.battery)-b.battery)
            if used+from_battery>self.storage_power('discharge')+1e-7:return False
            b.battery-=from_battery;b.curtailed-=reclaimed;self._step_curtailed-=reclaimed
            b.consumed+=energy
        self.station_consumed+=energy;self._bus_used+=energy
        return True

    def try_base_move(self):
        if getattr(self.sim,'regional_stage',None):return
        s=self.sim
        channel=s.terrain.front is not None
        if self.field_rect is not None and not channel:return
        if channel and s.relocation_pending is None:return
        if channel:
            s.base_move_wait=('LANDING_FLEET_FOR_BASE_MOVE' if self.active or self.direct_ids or not all(d.grounded for d in s.drones)
                else 'WAITING_GROUP_LIFT' if self.group else 'WAITING_SERVICE_COMPLETION' if
                self.battery_service.robot_swap or self.battery_service.drone_swaps else 'WAITING_BASE_ENERGY')
        if s.base.motion_power(s.drones)>min(self.config.inverter_power_w,
                self.storage_power('discharge')):
            if channel:s.base_move_wait='WAITING_BASE_DRIVE_POWER'
            return
        if (self.active or self.group or self.direct_ids or s.base.destination is not None
                or self.battery_service.robot_swap or self.battery_service.drone_swaps
                or not channel and any(j.status=='QUEUED' for j in self.jobs)
                or not all(d.grounded if channel else d.docked for d in s.drones) or s.landing_requested):return
        if self.battery_platforms and not self.battery_platforms.stowed:
            cost=s.base.route_energy(s.relocation_pending,s.terrain,s.drones) if channel else 0.
            if s.base.battery<cost*1.3+2*self.battery_platforms.energy_budget():
                s.base_move_wait='WAITING_BASE_ENERGY';return
            self.battery_platforms.request('PACK');s.base_move_wait='PACKING_BATTERY_PLATFORMS';return
        if channel and self.field_rect is not None:
            array=getattr(self,'solar_array',None)
            if not self.field_ready and not (array and array.mode=='FOLD'):
                s.base_move_wait='UNFOLDING_SOLAR_FIELD';return
            cost=s.base.route_energy(s.relocation_pending,s.terrain,s.drones)
            # Chargers are paused until redeployment. Budget the remaining fold,
            # the uphill drive and a full unfold even through a completely dark night.
            setup_seconds=2*self.field_setup_seconds()-self.base_pack_elapsed
            array=getattr(self,'solar_array',None)
            setup_energy=(2*array.energy_budget() if array else 200*setup_seconds)+s.base.config.idle_power*setup_seconds
            if s.base.battery<cost*1.3+setup_energy:
                s.base_move_wait='WAITING_BASE_ENERGY';return
            if array:array.request_fold()
            elif self.use_station(200.):self.base_pack_elapsed+=1.
            if self.base_pack_elapsed<self.field_setup_seconds():
                s.base_move_wait='FOLDING_SOLAR_FIELD';return
            self.field_rect=None;self.field_ready=False;self.field_elapsed=0.;self.base_pack_elapsed=0.
            return  # account the final folding step before the drive starts
        before=s.base.position.copy();s._maybe_relocate()
        if s.base.destination is not None:
            for job in self.jobs:
                if job.status=='QUEUED':self.cancel_job(job.id)
            self._serial+=1
            job=TransportJob(f'job_{self._serial:05}','RELOCATE_BASE',['base_chassis'],tuple(before),
                tuple([*s.base.destination,before[2]]),7,s.time,status='ACTIVE')
            self.inventory.reserve(job.resource_ids,job.id);self.jobs.append(job);self.base_move_job=job

    def change_phase(self, d, job, phase):
        job.phase,job.elapsed=phase,0.
        d.elapsed=0.
        states={'TAKEOFF':DS.TAKEOFF,'PICK':DS.LOWER_GRABBER,'LIFT':DS.LIFT_GRABBER,
                'LOWER':DS.DUMP,'DROP':DS.DUMP,'HOME':DS.RETURN,'LAND':DS.LANDING}
        d.transition(states.get(phase,DS.TO_DIG))

    def fly(self, d, job, point, next_phase):
        target=np.array(point,dtype=float)
        self.sim._move(d,target,d.state,1.)
        if np.linalg.norm(d.position-target)<1e-8 and d.detour is None:
            self.change_phase(d,job,next_phase)

    def carried(self, d):
        return [o for o in self.inventory.items.values()
                if o.location==L.DRONE_CARGO and o.holder_id==f'drone_{d.serial}']

    def _load(self, d, job):
        for id in job.resource_ids:
            o=self.inventory.items[id]
            state='FULL_IN_TRANSIT' if isinstance(o,SoilContainer) and o.soil_kg else 'EMPTY_IN_TRANSIT'
            if isinstance(o,BatteryModule):state='IN_TRANSIT_FULL' if o.soc>=.95 else 'IN_TRANSIT_EMPTY'
            self.inventory.move(id,L.DRONE_CARGO,f'drone_{d.serial}',job.origin,state,job.id)
        d.cargo=sum(o.gross_mass for o in self.carried(d))

    def _unload_soil(self, d, job):
        c=self.inventory.items[job.resource_ids[0]]
        mass=c.soil_kg
        if job.soil_destination=='fill':
            mass=self.sim.terrain.deposit_fill(*job.destination[:2],mass,load=c.soil)
            self.sim.filled_mass+=mass;d.filled_mass+=mass
        else:self.sim.terrain.deposit(*job.destination[:2],mass,load=c.soil)
        delivered=c.soil.remove(mass) if c.soil is not None else None
        if delivered is not None and job.soil_destination=='fill':self.sim.filled_volume+=delivered.volume
        c.soil_kg=max(0.,c.soil_kg-mass)
        if c.soil_kg==0:c.cycles+=1;c.wear+=.001
        d.cargo=sum(o.gross_mass for o in self.carried(d))
        d.trips+=1;self.sim.trips+=1;self.delivered_mass+=mass
        d.excavated_mass+=mass
        day=int((self.sim.config.start_hour*3600+self.sim.time)//86400)
        d.productivity.record(self.sim.time,day,mass,self.sim.config.terrain.density,job.soil_destination,volume=delivered.volume if delivered is not None else None)
        self.sim._terrain_active=True;self.progress_at=self.sim.time

    def _drop(self, d, job):
        item=self.inventory.items[job.resource_ids[0]]
        at_base=job.kind in ('RECOVER_EMPTY_EXCAVATOR_BATTERY','PACK_FOR_RELOCATION')
        location=L.BASE_STORAGE if at_base else L.WORKSITE_RACK
        holder='base' if at_base else 'excavator'
        state='QUEUED_FOR_CHARGE' if isinstance(item,BatteryModule) and at_base else (
            'EMPTY_AT_EXCAVATOR' if isinstance(item,SoilContainer) else 'STORED_READY')
        self.inventory.move(item.id,location,holder,job.destination,state,job.id)
        if at_base and isinstance(item,BatteryModule) and self.battery_platforms:self.battery_platforms.receive(item.id)
        hoist=self.inventory.items[job.resource_ids[-1]]
        # Hoist stays on board until the aircraft physically returns to base.
        d.cargo=hoist.mass_kg
        self.progress_at=self.sim.time

    def flight_tick(self, d, job):
        s=self.sim;phase=job.phase
        if phase=='PREPARE':
            if not s.consume_drone(d.config.winch_power,d):return
            job.elapsed+=1.
            if job.elapsed>=self.config.handling_seconds:self.change_phase(d,job,'TAKEOFF')
            return
        moving=phase in ('TAKEOFF','ORIGIN','DESTINATION','RETURN_EMPTY','HOME','LAND')
        winching=phase in ('PICK','LIFT','LOWER','RETURN_DROP')
        lift=max(0.,d.position[2]-s.terrain.sample(*d.position[:2])) if phase=='LIFT' else 0.
        if not s.consume_drone(d.power(moving,winching,lift_height=lift),d):return
        d.airborne_seconds+=1.;d.flight_seconds+=1.;d.longest_flight_seconds=max(d.longest_flight_seconds,d.flight_seconds)
        job.elapsed+=1.;phase_elapsed=job.elapsed;altitude=s.cruise(d)
        if phase=='TAKEOFF':
            self.fly(d,job,(*d.position[:2],altitude),'ORIGIN')
            if job.phase=='ORIGIN':s._clear_slot(d)
        elif phase=='ORIGIN':self.fly(d,job,(*job.origin[:2],altitude),'PICK')
        elif phase=='PICK':
            if self.dock_owner not in (None,job.id):
                d.waiting_seconds+=1.;job.elapsed=0.;return
            self.dock_owner=job.id
            if job.elapsed>=self.config.handling_seconds:
                self._load(d,job);self.change_phase(d,job,'LIFT')
        elif phase=='LIFT':
            if job.elapsed>=self.config.handling_seconds:
                self.dock_owner=None;self.change_phase(d,job,'DESTINATION')
        elif phase=='DESTINATION':self.fly(d,job,(*job.destination[:2],altitude),'LOWER')
        elif phase=='LOWER':
            if job.elapsed>=self.config.handling_seconds:self.change_phase(d,job,'DROP')
        elif phase=='DROP':
            if job.elapsed>=self.config.handling_seconds:
                if job.kind=='PICKUP_FULL_CONTAINER':
                    item=self.inventory.items[job.resource_ids[0]]
                    if (job.soil_destination=='dump' and s.terrain.front is not None and
                            s.terrain.local_dump_space(*job.destination[:2])<item.soil_volume(s.config.terrain.density)-1e-12):
                        dump=s.terrain.dump_location(item.soil_volume(s.config.terrain.density),
                            s._reserved_dump(None),s.config.fleet.separation,near=d.position)
                        if dump is None:s.block('BLOCKED_DUMP_FULL: груз сохранён в контейнере');return
                        job.destination=(*dump,s.terrain.sample(*dump));self.change_phase(d,job,'DESTINATION');return
                    self._unload_soil(d,job)
                    item=self.inventory.items[job.resource_ids[0]]
                    if item.soil_kg>1e-8:
                        dump=s.terrain.dump_location(item.soil_volume(s.config.terrain.density),near=d.position)
                        if dump is None:s.block('BLOCKED_DUMP_FULL: остаток после засыпки сохранён в контейнере');return
                        job.destination=(*dump,s.terrain.sample(*dump));job.soil_destination='dump'
                        self.change_phase(d,job,'DESTINATION')
                    else:self.change_phase(d,job,'RETURN_LIFT')
                else:
                    self._drop(d,job);self.change_phase(d,job,'HOME_LIFT')
        elif phase=='RETURN_LIFT':
            if job.elapsed>=self.config.handling_seconds:self.change_phase(d,job,'RETURN_EMPTY')
        elif phase=='RETURN_EMPTY':self.fly(d,job,(*job.origin[:2],altitude),'RETURN_DROP')
        elif phase=='RETURN_DROP':
            if job.elapsed>=self.config.handling_seconds:
                destination=job.destination;job.destination=job.origin
                self._drop(d,job);job.destination=destination
                self.change_phase(d,job,'HOME_LIFT')
        elif phase=='HOME_LIFT':
            if job.elapsed>=self.config.handling_seconds:self.change_phase(d,job,'HOME')
        elif phase=='HOME':
            self.fly(d,job,(*s.wait_position(d.identifier)[:2],altitude),'LAND')
        elif phase=='LAND':
            self.fly(d,job,s.wait_position(d.identifier),'DONE')
            if job.phase=='DONE':
                job.status='COMPLETED';self.active.pop(d.identifier)
                hoist=self.inventory.items[job.resource_ids[-1]]
                self.inventory.move(hoist.id,L.GROUND,f'drone_{d.serial}',d.position,'STORED_READY',job.id)
                self.inventory.release(job.id)
                d.cargo=0.;d.transition(DS.WAIT_GROUND)
                d.last_flight_seconds=d.flight_seconds;d.flight_seconds=0.
                if d.battery<=d.config.battery*d.config.charge_threshold:
                    s._request_charge(d,'нижний порог после грузового рейса')
                if d.charge_requested:s._enqueue(d)
        for item in self.carried(d):
            p=d.position.copy();p[2]-=d.config.stowed_cable
            fraction=min(1.,phase_elapsed/self.config.handling_seconds)
            ground=job.origin[2] if phase in ('PICK','LIFT','RETURN_DROP') else job.destination[2]
            if phase in ('LIFT','RETURN_LIFT','HOME_LIFT'):p[2]=ground+(p[2]-ground)*fraction
            if phase in ('PICK','LOWER','RETURN_DROP'):p[2]+=(ground-p[2])*fraction
            if phase=='DROP':p[2]=job.destination[2]
            item.position=tuple(p)

    def sync_batteries(self):
        if getattr(self,'regional_cleanup_pending',False) and not getattr(self.sim,'regional_stage',None):
            for d in self.sim.drones:
                if d.grounded and hasattr(d,'regional_mass'):
                    d.config.empty_mass,d.config.hover_power=d.regional_mass;del d.regional_mass
                    self.inventory.move(f'hoist_{d.serial}',L.GROUND,f'drone_{d.serial}',d.position,'STORED_READY')
            self.regional_cleanup_pending=any(hasattr(d,'regional_mass') for d in self.sim.drones)
        mounted=[item for item in self.mobile_items() if item.location==L.DRONE_MOUNT]
        positions={f'drone_{d.serial}':tuple(d.position) for d in self.sim.drones}
        for item in mounted:
            if item.holder_id in positions:item.position=positions[item.holder_id]
        base_position=tuple(self.sim.base.position);base_energy=self.sim.base.battery;base_capacity=self.sim.base.config.battery
        for id in self.base_battery_ids:
            b=self.inventory.items[id]
            b.energy_j=base_energy*b.capacity_j/base_capacity
            if not self.battery_platforms:b.position=base_position
        for d in self.sim.drones:
            packs=[self.inventory.items[id] for id in d.battery_ids]
            total=sum(b.energy_j for b in packs)
            difference=d.battery-total
            room=sum(b.capacity_j-b.energy_j for b in packs)
            for id in d.battery_ids:
                b=self.inventory.items[id]
                if difference<0 and total>0:b.energy_j=max(0.,b.energy_j*d.battery/total)
                elif difference>0 and room>0:
                    b.energy_j=min(b.capacity_j,b.energy_j+difference*(b.capacity_j-b.energy_j)/room)
                b.position=positions[f'drone_{d.serial}']
        self.battery_service.update_slot_states()
        if self.battery_platforms:self.battery_platforms.position_items()
        array=getattr(self,'solar_array',None)
        if getattr(array,'mechanism',None)=='self_deploying':array.sync_positions(self)
        self.inventory.validate(dynamic=int(self.sim.time)%60!=0)
        mounted=set()
        for d in self.sim.drones:
            for id in d.battery_ids:
                if id in mounted or self.inventory.items[id].location!=L.DRONE_MOUNT or self.inventory.items[id].holder_id!=f'drone_{d.serial}':
                    raise ValueError('Physical drone battery appears in two holders')
                mounted.add(id)
        if self.excavator is not None:
            for id in self.excavator.battery_ids:
                if self.inventory.items[id].location!=L.EXCAVATOR_SLOT:
                    raise ValueError('Excavator slot refers to a transported battery')

    def step(self):
        s=self.sim
        self.wait_reasons.clear()
        self.wait_reason=''
        self._bus_used=0.
        self._pending_station_energy=0.;self._solar_ticked=False;self._step_curtailed=0.
        before_bus=s.base.consumed+s.base.charging_input+s.base.laser_input
        battery_before=s.base.battery
        before_curtailed=s.base.curtailed
        if s.relocation_pending is not None:
            # Tool exchange commits atomically at the end. Release unfinished
            # exchanges now, rather than freezing their state during relocation.
            for identifier in list(self.tool_targets):
                d=s.drones[identifier];self.request_tool(d,d.attachment)
        self.try_base_move();base_before=s.base.position.copy()
        self._base_moving_step=s.base.destination is not None
        s._starts={d.identifier:d.position.copy() for d in s.drones}
        for d in s.drones:
            if d.identifier not in self.active and s._charging_needed(d):
                if d.docked and s.base.destination is None:d.state=DS.CHARGING;d.charging_seconds+=1.
                elif d.state==DS.WAIT_GROUND:s._enqueue(d)
        s._assign_slots()
        if self.dock_service_ready(continuing=True):
            self.battery_service.tick_drone_swaps()
        self.tick_tool_changes()
        if self.battery_platforms:self.battery_platforms.step()
        consumers=self.battery_service.charging()
        flags=[d.docked and d.charge_requested and d.serial not in self.battery_service.drone_swaps and
               (self.battery_platforms is None or self.battery_platforms.service_ready(d.slot)) for d in s.drones]
        for d in s.drones:
            d.max_charge_input_w=sum(self.inventory.items[id].max_charge_w for id in d.battery_ids
                                     if self.inventory.items[id].soc<1-1e-12)
        s.laser_links=prepare_links(s.base,s.drones,s.terrain,1.,s.flight_power,s.config.fleet.separation/4,
            dust=s.dust,obstacles=[self.excavator] if self.excavator is not None else ())
        area=s.base.config.panel_area
        array=getattr(self,'solar_array',None)
        if array and not getattr(self,'regional_folded',False) and s.base.destination is None:
            if getattr(s,'regional_stage',None)=='PACK':array.request_fold()
            array.step(self)
            if array.stowed and getattr(self,'pending_solar_panels',None) is not None:
                panels=self.pending_solar_panels;self.pending_solar_panels=None
                try:self.configure_panels(panels);array=self.solar_array
                except ValueError as error:self.solar_site_issue=str(error);s.event(str(error))
        setup=bool(self.base_pack_elapsed or not self.field_ready) and array is None
        if not self.field_ready and array is None:
            if not getattr(self,'regional_folded',False) and s.base.destination is None and (s.relocation_pending is None or self.field_rect is not None) and self.use_station(200.):
                self.field_elapsed+=1.
            if not getattr(self,'regional_folded',False) and self.field_elapsed>=self.field_setup_seconds():self.field_ready=True
        if getattr(s,'regional_stage',None)=='PACK' and not getattr(self,'regional_folded',False) and array is None:
            if self.use_station(200.):self.base_pack_elapsed+=1.
        # A pending hardware delivery can have committed during array.step.
        area=s.base.config.panel_area
        generating_area=self.generating_area()
        if setup:
            # Deployment and motion must retain their own energy reserve;
            # charging aircraft/spares must not strand folded panels overnight.
            consumers=[];flags=[False]*len(s.drones);s.laser_links=[]
        s.base.config.panel_area=generating_area
        charge_power=s.base.config.charge_power
        s.base.config.charge_power=min(charge_power,self.config.bus_power_w)
        self._storage_before_tick=s.base.battery
        s.base.tick(1.,(s.hour+1/7200)%24,s.drones,flags,s.day_at(s.time+.5),
                    s.laser_links,modules=consumers,bus_limit_w=self.config.inverter_power_w,
                    storage_charge_w=self.storage_power('charge'),
                    storage_discharge_w=self.storage_power('discharge'),
                    auxiliary_w=self._pending_station_energy,
                    module_bus_limit_w=sum(power for _,power in consumers) if self.config.individual_chargers else None,
                    terrain=s.terrain)
        self._solar_ticked=True;self._step_curtailed=max(0.,s.base.curtailed-before_curtailed)
        s.base.config.charge_power=charge_power
        self._bus_used=s.base.consumed+s.base.charging_input+s.base.laser_input-before_bus
        s.base.config.panel_area=area
        delta=s.base.position-base_before
        if np.linalg.norm(delta)>0:
            for item in self.inventory.items.values():
                if item.holder_id=='base':item.position=tuple(np.asarray(item.position)+delta)
            for d in s.drones:
                if d.docked:d.position[:]=s.slot_position(d.slot)
            if array:
                for crew in array.crews:crew.position+=delta;crew.home+=delta;crew.mount+=delta
        if self.base_move_job and s.base.destination is None:
            self.base_move_job.status='COMPLETED';self.inventory.release(self.base_move_job.id)
            self.base_move_job=None;self.progress_at=s.time
            s.base_move_wait=''
            if self.battery_platforms and not getattr(self,'regional_folded',False):self.battery_platforms.request('DEPLOY')
            if not getattr(self,'regional_folded',False) and not self.field_ready and s.base.config.panel_area>20:
                try:self.plan_solar_field()
                except ValueError as error:s.block(str(error))
        self.battery_service.after_charging()
        self.try_lift()
        if self.group is None and s.base.destination is None and s.relocation_pending is None:self.generate_jobs()
        if self.group is None and not self._base_moving_step and s.base.destination is None and s.relocation_pending is None:self.service_tools()
        self.dispatch()
        self.direct_dispatch()
        self.yield_service_pads()
        if self.group is None:self.battery_service.tick_robot_swap()
        if self.excavator is not None and self.group is None:
            self.excavator.tick(self)
            for id in self.excavator.battery_ids:self.inventory.items[id].position=tuple(self.excavator.position)
        participants={d.identifier for d in self.group.participants} if self.group else set()
        if self.group is not None:self.group.tick()
        for d in s.drones:
            if s.state!=MissionState.RUNNING:break
            if d.identifier in participants:continue
            if d.identifier in self.active:self.flight_tick(d,self.active[d.identifier])
            elif d.identifier in self.direct_ids:
                s._airborne(d,1.)
                if d.grounded:
                    self.direct_ids.remove(d.identifier)
                    job=next(j for j in reversed(self.jobs) if j.executor==d.serial and j.kind=='EXCAVATE_DIRECT' and j.status=='ACTIVE')
                    job.status='CANCELLED' if d.identifier in self.recall_ids else 'COMPLETED'
                    self.inventory.release(job.id);self.progress_at=s.time
            elif not d.grounded:
                s._airborne(d,1.)
            else:d.waiting_seconds+=1.
            if d.identifier in self.parking:
                if d.position[2]>=s.cruise(d)-1e-8:s._clear_slot(d)
                if d.grounded:self.parking.remove(d.identifier)
            s._settle_laser(d)
        for d in s.drones:s._settle_laser(d)
        for identifier in list(self.recall_ids):
            d=s.drones[identifier]
            if (d.grounded and d.cargo<=1e-8 and d.identifier not in self.active
                    and d.serial not in self.battery_service.drone_swaps
                    and d.identifier not in self.direct_ids and d.identifier not in participants):
                self.finish_drone_recall(d)
        if s._terrain_active:
            s._terrain_active=s.terrain.relax()>s.config.terrain.avalanche_stop_volume
        s.dust.tick(1.,s.drone.position,0.,1.,False)
        self.sync_batteries()
        self.update_base_mass()
        self.base_battery_power_w=s.base.battery-battery_before
        self.base_load_power_w=s.base.consumed+s.base.charging_input+s.base.laser_input-before_bus
        self.detect_stall()
        if (not getattr(s,'regional_stage',None) and s.terrain.remaining<=s.config.terrain.completion_volume and not s._terrain_active
                and s.base.destination is None and self.base_move_job is None
                and not self.battery_service.drone_swaps and not self.battery_service.robot_swap
                and (self.battery_platforms is None or self.battery_platforms.settled)
                and not self.active and all(c.soil_kg==0 for c in self.containers)
                and self.group is None and not self.direct_ids
                and (getattr(self,'solar_array',None) is None or (self.solar_array.stowed if self.solar_array.mode=='FOLD' else self.solar_array.ready))
                and (self.excavator is None or self.excavator.bucket_kg==0)
                and all(d.grounded for d in s.drones)):
            s.state=MissionState.COMPLETED;s.event('Логистическая миссия завершена')
            for job in self.jobs:
                if job.status=='QUEUED':self.cancel_job(job.id)
            for d in s.drones:
                self.inventory.release(f'tool_{d.serial}')
                if d.state==DS.CHANGING_TOOL:d.state=DS.DOCKED
            self.tool_targets.clear();self.tool_changes.clear()
            self.courier_active=False;self.courier_id=None;self.courier_reason=''
        s.time+=1.
        if len(self.jobs)>2048:self.archive_completed_jobs()

    def archive_completed_jobs(self):
        history=[j for j in self.jobs if j.status in ('COMPLETED','CANCELLED')]
        if len(history)<=256:return
        removed=history[:-256];keys={id(j) for j in removed}
        if self.archived_jobs is None:self.archived_jobs=[]
        self.archived_jobs.extend(removed);self.jobs[:]=[j for j in self.jobs if id(j) not in keys]

    def storage_power(self,mode):
        rating=sum(getattr(self.inventory.items[id],f'max_{mode}_w') for id in self.base_battery_ids)
        return min(rating,self.battery_platforms.converter_power_w) if self.battery_platforms else rating

    def detect_stall(self):
        if getattr(self.sim,'regional_stage',None):return
        if self.battery_platforms and not self.battery_platforms.settled:
            self.wait_reason=self.battery_platforms.issue or 'Подготовка батарейных платформ';self.progress_at=self.sim.time;return
        if getattr(self,'solar_array',None) is not None and not (self.solar_array.stowed if self.solar_array.mode=='FOLD' else self.solar_array.ready):
            if self.solar_array.issue:self.wait_reason=self.solar_array.issue
            else:self.progress_at=self.sim.time
            return
        s=self.sim
        if s.base.destination is not None or s.relocation_pending is not None:return
        if s.state!=MissionState.RUNNING or self.active or s.landing_requested:return
        if s.terrain.remaining<=s.config.terrain.completion_volume:return
        if self.excavator is not None:
            e=self.excavator
            if e.state.value=='BLOCKED':s.block('BLOCKED_EXCAVATOR: '+e.reason)
            elif not self.containers and not (self.config.direct_allowed or self.config.idle_digging):
                s.block('BLOCKED_NO_CONTAINERS: запас тары равен нулю')
            elif e.state.value=='LOW_BATTERY' and len([b for b in self.inventory.select(BatteryModule)
                    if b.compatibility=='excavator'])<=self.config.excavator_battery_slots and not (
                    self.config.direct_allowed or self.config.idle_digging):
                s.block('BLOCKED_NO_SPARE_BATTERIES: физическая смена невозможна')
        no_source=(s.base.battery<=1e-8 and s.config.base.panel_area*s.config.base.panel_efficiency*s.config.base.peak_irradiance==0)
        if no_source and any(d.charge_requested for d in s.drones):
            s.block('BLOCKED_NO_ENERGY: база пуста и нет солнечной генерации')
        pending=[j for j in self.jobs if j.status=='QUEUED']
        if pending and all(all(self.estimate(d,j)[0]>d.config.battery*(1-d.config.reserve) for d in s.drones) for j in pending):
            s.block('BLOCKED_TRANSPORT_CAPACITY: ни один аппарат не может выполнить грузовой рейс с резервом')
        if not self.config.excavator_enabled and not (self.config.direct_allowed or self.config.idle_digging) and not pending:
            s.block('BLOCKED_NO_EXECUTOR: нет машины для выемки')
        if s.time-self.progress_at>self.config.stall_timeout_s:
            charging=bool(self.battery_service.queue or self.battery_service.robot_swap or self.battery_service.drone_swaps
                          or any(d.charge_requested for d in s.drones))
            if charging:
                self.wait_reason='WAITING_SUN_OR_CHARGER: конечная зарядная очередь'
            elif self.tool_targets:
                self.wait_reason='WAITING_STATION_POWER'
            elif not self.field_ready:
                self.wait_reason='WAITING_SOLAR_DEPLOYMENT_POWER'
            elif self.excavator is not None and self.excavator.state.value=='WAITING_RELOCATION':
                self.wait_reason='WAITING_GROUP_SOC'
            elif not (self.config.direct_allowed or self.config.idle_digging):
                s.block('BLOCKED_NO_PROGRESS: '+(self.excavator.reason if self.excavator else 'NO_EXECUTOR'))

    def update_report(self, report, *, full=True):
        s=self.sim
        soil=sum(c.soil_kg for c in self.containers)
        soil+=sum(d.cargo for d in s.drones if d.identifier in self.direct_ids)
        if self.excavator is not None:soil+=self.excavator.bucket_kg
        report['mass_error_kg']=s.terrain.mass_change+soil
        if full:report['inventory']=self.inventory.snapshot()
        report['logistics_version']=2
        if self.battery_platforms:report['battery_platforms']=self.battery_platforms.report()
        report['courier_strategy']='batch' if self.batch_enabled else 'responsive'
        report['courier']=dict(active=self.courier_active,reason=self.courier_reason,
            drone_serial=s.drones[self.courier_id].serial if self.courier_id is not None else None,
            batches=self.courier_batches,filled_containers=sum(c.location==L.FILLED_QUEUE and c.soil_kg>1e-8 for c in self.containers),
            threshold=max(1,math.ceil(len(self.containers)*self.config.courier_fill_fraction)))
        report['inventory_imported_mass_kg']=self.imported_mass_kg
        report['inventory_exported_mass_kg']=self.exported_mass_kg
        report['battery_deliveries_count']=len(self.deliveries)
        if full:report['battery_deliveries']=list(self.deliveries)
        if full:
            jobs=(sorted([*self.archived_jobs,*self.jobs],key=lambda j:int(j.id.rsplit('_',1)[1])) if self.archived_jobs else self.jobs)
            report['jobs']=[asdict(j) for j in jobs]
        report['container_cycles']=sum(c.cycles for c in self.containers)
        report['battery_service']=self.battery_service.snapshot(include_slots=full)
        report['collective_lift']=self.group.report() if self.group else None
        report['collective_lift_count']=self.lift_count
        report['delivered_mass_kg']=self.delivered_mass
        report['solar_field_deployed']=self.field_ready
        report['solar_generating_area_m2']=self.generating_area()
        report['solar_field_folding_seconds']=self.base_pack_elapsed
        if getattr(self,'solar_site_issue',''):report['solar_site_issue']=self.solar_site_issue
        report['base_battery_power_w']=self.base_battery_power_w
        report['base_load_power_w']=self.base_load_power_w
        report['base_energy_status']=('База движется, генерация отключена' if s.base.destination is not None else
            'Складывание панелей: зарядка потребителей приостановлена' if self.base_pack_elapsed else
            'Панели сложены: недостаточно энергии для развёртывания' if not self.field_ready and
                self.generating_area()<=1e-8 and s.base.battery<200+s.base.config.idle_power else
            'Развёртывание панелей: зарядка потребителей приостановлена' if not self.field_ready else
            'База без накопителя: прямое питание от солнца' if not s.base.config.battery else
            'Аккумуляторы базы полностью заряжены' if s.base.battery>=s.base.config.battery-1e-7 else
            'Аккумуляторы базы заряжаются' if self.base_battery_power_w>1e-7 else
            'Аккумуляторы базы питают нагрузку' if self.base_battery_power_w< -1e-7 else
            'Ожидание солнца' if s.base.solar_power<=1e-8 else
            'Солнечная энергия поступает потребителям')
        report['solar_field_rect']=self.field_rect
        if getattr(self,'solar_array',None) is not None:
            report['solar_sections']=self.solar_array.report()
            if not self.field_ready and s.base.destination is None:
                action='Возврат солнечных секций' if self.solar_array.mode=='FOLD' else 'Монтаж солнечных секций'
                if getattr(self.solar_array,'mechanism',None)=='self_deploying':action='Складывание собственных крыльев' if self.solar_array.mode=='FOLD' else 'Самораскрытие солнечных платформ'
                report['base_energy_status']=f'{action}: в работе {self.generating_area():g} м²; питание и зарядка — в пределах доступной мощности'
        if report.get('solar_site_issue'):report['base_energy_status']=report['solar_site_issue']
        if getattr(s.terrain,'dump_reserve_info',None):report['spoil_reserve']=dict(s.terrain.dump_reserve_info)
        report['logistics_wait_reason']=self.wait_reason
        report['solar_field_deployment_seconds']=self.field_elapsed
        report['energy_drone_kwh']=sum(d.consumed for d in s.drones)/KWH
        report['energy_excavator_kwh']=self.excavator.consumed_j/KWH if self.excavator else 0.
        report['energy_laser_station_kwh']=s.base.laser_input/KWH
        report['energy_station_mechanics_kwh']=self.station_consumed/KWH
        report['drone_flight_time_seconds']=sum(d.airborne_seconds for d in s.drones)
        extra_energy=sum(b.energy_j for b in self.inventory.select(BatteryModule)
                         if b.location not in (L.DRONE_MOUNT,L.BASE_MOUNT))
        if self.excavator is not None:
            extra_energy+=self.excavator.buffer_j+self.excavator.consumed_j
            report['excavator']=self.excavator.report(self.inventory,s.time,
                int((s.config.start_hour*3600+s.time)//86400),s.config.terrain.density)
            if s.terrain.soil is not None:report['excavator']['excavated_m3']=self.excavator.productivity.total_volume
            report['consumed_kwh']+=self.excavator.consumed_j/KWH
        report['energy_error_j']+=extra_energy-self.initial_extra_energy
        report['initial_energy_kwh']+=self.initial_extra_energy/KWH
        for row,d in zip(report['drones'],s.drones):
            job=self.active.get(d.identifier)
            row.update(task=job.kind if job else '',attachment=d.attachment,
                       role='COURIER' if self.batch_enabled and d.identifier==self.courier_id else 'DIGGER' if d.attachment=='GRABBER' else 'CARRIER',
                       transport_payload_kg=self.payload_limit(d),battery_ids=list(d.battery_ids),
                       wait_reason=self.idle_reason(d))
            row['recall_requested']=d.identifier in self.recall_ids
            row['tool_change_remaining_seconds']=(max(0.,self.config.attachment_seconds-self.tool_changes.get(d.serial,0.))
                                                  if d.identifier in self.tool_targets else 0.)
            if d.identifier in self.direct_ids:row['task']='EXCAVATE_DIRECT'
            if d.identifier in self.tool_targets:
                row['task']='CHANGE_TOOL → '+self.tool_targets[d.identifier]
            swap=self.battery_service.drone_swaps.get(d.serial)
            row['battery_swaps']=self.battery_service.drone_swap_counts.get(d.serial,0)
            row['battery_swap_remaining_seconds']=max(0.,self.config.drone_swap_seconds-swap['elapsed']) if swap else 0.
            if swap:row['task']='SWAP_BATTERIES'
            row['lift_equivalent_kg']=d.equivalent_load
            if self.group is not None and any(o.identifier==d.identifier for o in self.group.participants):row['task']='RELOCATE_EXCAVATOR_AIR'
            row['equipment']['bucket_kg']=self.inventory.items[f'grabber_{d.serial}'].mass_kg if d.attachment=='GRABBER' else 0.
            row['equipment']['hook_kg']=self.inventory.items[f'hook_{d.serial}'].mass_kg if d.attachment=='CONTAINER_HOOK' else 0.
            row['equipment']['total_kg']=d.config.empty_mass
        return report
