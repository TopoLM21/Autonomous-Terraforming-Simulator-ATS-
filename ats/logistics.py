"""Finite logistics layered on the original terrain, drone and solar physics."""
from dataclasses import dataclass, field, asdict
import math
import numpy as np
from .inventory import Inventory, Item, BatteryModule, SoilContainer, Rigging, Location as L
from .config import KWH
from .drone import DroneState as DS
from .simulation import MissionState
from .laser import prepare_links


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
    def __init__(self, sim):
        self.sim, self.config = sim, sim.config.logistics
        self.inventory = Inventory()
        self.jobs = []
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
        self.tool_changes={}
        self.tool_targets={}
        self.idle_since={}
        self.parking=set()
        self._bus_used=0.
        self._pending_station_energy=0.
        self._solar_ticked=False
        self._step_curtailed=0.
        self.wait_reason=''
        self.base_move_job=None
        self.base_battery_ids=[]
        self.field_elapsed=0.
        self.field_ready=sim.base.config.panel_area<=20.
        self.field_rect=None
        for recipe in sim.base.config.equipment.batteries:
            for _ in range(recipe.quantity):
                id=f'base_battery_{len(self.base_battery_ids)+1:02}'
                b=BatteryModule(id,recipe.mass_kg,L.BASE_MOUNT,'INSTALLED','base',
                    position=tuple(sim.base.position),compatibility='base',capacity_j=recipe.capacity_kwh*KWH,
                    energy_j=recipe.capacity_kwh*KWH*sim.base.config.initial_charge,
                    max_charge_w=recipe.capacity_kwh*2000.,max_discharge_w=recipe.capacity_kwh*10000.)
                self.inventory.add(b);self.base_battery_ids.append(id)
        eq=sim.base.config.equipment
        self.inventory.add(Item('base_chassis',eq.chassis_mass_kg,L.BASE_MOUNT,'INSTALLED','base',position=tuple(sim.base.position),kind='CHASSIS'))
        self.inventory.add(Item('base_charger',eq.charger_mass_kg,L.BASE_MOUNT,'INSTALLED','base',position=tuple(sim.base.position),kind='CHARGER_AND_AUTOMATION'))
        for i in range(eq.landing_slots):
            self.inventory.add(Item(f'landing_pad_{i+1}',eq.slot_mass_kg,L.BASE_MOUNT,'INSTALLED','base',position=tuple(sim.slot_position(i)),kind='LANDING_PAD'))
        panel_index=0
        for recipe in eq.panels:
            for _ in range(recipe.quantity):
                panel_index+=1
                self.inventory.add(Item(f'solar_panel_{panel_index:03}',recipe.mass_kg,L.BASE_MOUNT,'INSTALLED','base',position=tuple(sim.base.position),kind='SOLAR_PANEL'))
        for i,emitter in enumerate(eq.emitters):
            self.inventory.add(Item(f'laser_channel_{i+1}',emitter.mass_kg,L.BASE_MOUNT,'INSTALLED','base',position=tuple(sim.base.position),kind='LASER_TRANSMITTER'))
        if not self.field_ready:
            area=sim.base.config.panel_area;side=sim.terrain.cell*sim.terrain.height.shape[0]
            width=min(20.,side-2*sim.terrain.cell);height=area/width
            margin=sim.terrain.cell
            pads=[sim.slot_position(i) for i in range(len(sim.slots))]+[sim.wait_position(i) for i in range(len(sim.drones))]
            sites=[]
            for x in np.linspace(margin,max(margin,side-margin-width),5):
                for y in np.linspace(margin,max(margin,side-margin-height),5):
                    mask=(sim.terrain.x>=x)&(sim.terrain.x<x+width)&(sim.terrain.y>=y)&(sim.terrain.y<y+height)
                    if height>side-2*margin or np.any(mask&(sim.terrain.work_mask|sim.terrain.dump_mask)):continue
                    if any(x-margin<=p[0]<=x+width+margin and y-margin<=p[1]<=y+height+margin for p in pads):continue
                    sites.append((np.linalg.norm(np.array([x+width/2,y+height/2])-sim.base.position[:2]),(float(x),float(y),width,height)))
            if sites:
                self.field_rect=min(sites,key=lambda s:(s[0],s[1]))[1]
            else:sim.block('BLOCKED_SOLAR_FIELD_SITE: поле панелей не помещается вне рабочих и посадочных площадок')
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
            if self.config.direct_allowed and (not self.config.excavator_enabled or d is sim.drones[-1]):
                self.inventory.move(f'hook_{d.serial}',L.BASE_STORAGE,'base',sim.base.position,'STORED_READY')
                self.inventory.move(f'grabber_{d.serial}',L.DRONE_MOUNT,f'drone_{d.serial}',d.position,'INSTALLED')
                d.attachment='GRABBER';self._tool_mass(d,d.config.equipment.bucket.mass_kg)
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
                    sim.config.terrain.density) for d in sim.drones)
        for d in sim.drones:
            for set_index in range(self.config.drone_spare_sets):
                for slot,id in enumerate(d.battery_ids):
                    old=self.inventory.items[id]
                    b=BatteryModule(f'drone_{d.serial}_spare_{set_index+1}_{slot+1}',old.mass_kg,
                        position=tuple(sim.base.position),compatibility=old.compatibility,
                        capacity_j=old.capacity_j,energy_j=old.capacity_j*self.config.drone_spare_initial_soc,max_charge_w=old.max_charge_w,
                        max_discharge_w=old.max_discharge_w)
                    self.inventory.add(b);self.initial_extra_energy+=b.energy_j
        from .batteries import BatteryService
        self.battery_service=BatteryService(self)

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
        self.battery_service.generate_jobs()
        for c in self.containers:
            if c.soil_kg > 1e-8 and c.location==L.FILLED_QUEUE:
                incoming=sum(self.inventory.items[j.resource_ids[0]].soil_kg for j in self.jobs
                    if j.kind=='PICKUP_FULL_CONTAINER' and j.soil_destination=='fill' and j.status in ('QUEUED','ACTIVE'))
                fill=s.terrain.fill_location() if s.terrain.remaining_fill*s.config.terrain.density>incoming+1e-8 else None
                dump=fill if fill is not None else s.terrain.dump_location(c.soil_kg/s.config.terrain.density)
                if dump is not None:
                    job=self.new_job('PICKUP_FULL_CONTAINER',c,c.position,(*dump,s.terrain.sample(*dump)),3)
                    if job is not None:job.soil_destination='fill' if fill is not None else 'dump'
                else:
                    s.block('BLOCKED_DUMP_FULL: нет места для контейнерного груза')
            elif c.location==L.BASE_STORAGE and c.soil_kg==0 and self.excavator is not None:
                self.new_job('DELIVER_EMPTY_CONTAINER',c,c.position,self.rack,4)
            elif (c.location==L.WORKSITE_RACK and c.soil_kg==0 and self.excavator is not None
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
        if s.base.destination is not None:return
        pending=sorted((j for j in self.jobs if j.status=='QUEUED'),
                       key=lambda j:(j.priority-(s.time-j.created_at)/600,j.created_at,j.id))
        for d in s.drones:
            if (d.identifier in self.active or not d.grounded
                    or self.group is not None or d.identifier in self.direct_ids
                    or d.serial in self.battery_service.drone_swaps):
                continue
            if d.attachment!='CONTAINER_HOOK' or d.identifier in self.tool_targets:continue
            if s._charging_needed(d):
                if d.docked:d.state=DS.CHARGING
                else:s._enqueue(d)
                continue
            for job in pending:
                if job.status!='QUEUED':continue
                if s.landing_requested and job.kind!='PICKUP_FULL_CONTAINER':continue
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
        from .cooperative_lift import CollectiveLift,lift_check
        from copy import copy
        e=self.excavator;s=self.sim
        if e is None or e.state.value!='WAITING_RELOCATION' or self.group is not None or s.base.destination is not None:return
        if (self.active or self.direct_ids or self.tool_targets or self.battery_service.drone_swaps
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
                for d in s.drones:s._request_charge(d,'запас для коллективного подъёма')
            return
        self.group=CollectiveLift(self,e.destination.copy(),list(s.drones))
        s.event('Команда зарезервирована: снятие батарей и перенос сухого Крота')

    def direct_should_return(self, d):
        return (d.identifier in self.tool_targets or
                self.excavator is not None and self.excavator.state.value=='WAITING_RELOCATION')

    def request_tool(self, d, target):
        if d.attachment==target:
            self.inventory.release(f'tool_{d.serial}')
            self.tool_targets.pop(d.identifier,None);self.tool_changes.pop(d.serial,None)
            if d.state==DS.CHANGING_TOOL:d.state=DS.DOCKED
        else:self.tool_targets[d.identifier]=target

    def service_tools(self):
        """Return to an actual pad, then exchange two physical attachments."""
        s=self.sim
        queued=any(j.status=='QUEUED' for j in self.jobs)
        relocating=self.excavator is not None and self.excavator.state.value=='WAITING_RELOCATION'
        if queued or relocating:
            for d in s.drones:
                if self.tool_targets.get(d.identifier)=='GRABBER':self.request_tool(d,'CONTAINER_HOOK')
        if queued:
            # Keep a carrier available; recall one digger if all hooks are away.
            if not any(d.attachment=='CONTAINER_HOOK' for d in s.drones):
                if not any(target=='CONTAINER_HOOK' for target in self.tool_targets.values()):
                    candidates=[d for d in s.drones if d.identifier not in self.active]
                    if candidates:self.request_tool(min(candidates,key=lambda d:(not d.grounded,d.identifier)),'CONTAINER_HOOK')
        for d in s.drones:
            if d.identifier not in self.tool_targets or d.identifier in self.direct_ids:continue
            if d.charge_requested or s.landing_requested:continue
            if d.state==DS.WAIT_GROUND and s.base.destination is None:
                slot=next((i for i,v in enumerate(s.slots) if v is None),None)
                if slot is None:continue
                energy=s._transfer_energy(d,slot)
                if d.battery-energy<d.config.battery*d.config.reserve:
                    s._request_charge(d,'переход к станции смены инструмента');s._enqueue(d);continue
                d.slot=slot;s.slots[slot]=d.identifier;d.destination=s.slot_position(slot)
                d.transition(DS.TRANSFER_TAKEOFF)
                s.drone_event(d,f'возвращается в слот {slot+1} для смены инструмента')
            if not d.docked:continue
            owner=f'tool_{d.serial}'
            ids=[f'hook_{d.serial}',f'grabber_{d.serial}']
            if not self.inventory.reserve(ids,owner):continue
            d.state=DS.CHANGING_TOOL
            if not self.use_station(100.):continue
            elapsed=self.tool_changes.get(d.serial,0.)+1
            self.tool_changes[d.serial]=elapsed
            if elapsed<self.config.attachment_seconds:continue
            target=self.tool_targets[d.identifier]
            mounted=f'grabber_{d.serial}' if target=='GRABBER' else f'hook_{d.serial}'
            stored=next(id for id in ids if id!=mounted)
            self.inventory.move(stored,L.BASE_STORAGE,'base',s.base.position,'STORED_READY',owner)
            self.inventory.move(mounted,L.DRONE_MOUNT,f'drone_{d.serial}',d.position,'INSTALLED',owner)
            d.attachment=target;self._tool_mass(d,self.inventory.items[mounted].mass_kg)
            self.inventory.release(owner);self.tool_targets.pop(d.identifier);self.tool_changes.pop(d.serial)
            d.state=DS.DOCKED;s.drone_event(d,f'установлен инструмент {target}')

    def yield_service_pads(self):
        s=self.sim
        if self.group is not None:return
        waiting=bool(s.queue or any(s.drones[i].state==DS.WAIT_GROUND for i in self.tool_targets))
        if not waiting or s.base.destination is not None or s.landing_requested:return
        if any(slot is None for slot in s.slots):return
        if self.parking:return
        for d in s.drones:
            if (not d.docked or d.charge_requested or d.identifier in self.active or
                    d.identifier in self.direct_ids or d.identifier in self.tool_targets):continue
            energy=d.power(True)*(2*math.ceil(s.cruise(d)/d.config.vertical_speed)+
                                  math.ceil(s.config.fleet.waiting_offset/d.config.empty_speed)+10)
            if d.battery-energy<d.config.battery*d.config.reserve:continue
            d.destination=s.wait_position(d.identifier);d.transition(DS.RETURN)
            self.parking.add(d.identifier)
            s.drone_event(d,'освобождает сервисный слот; перелёт на площадку ожидания')
            break

    def direct_dispatch(self):
        if not (self.config.direct_allowed or self.config.idle_digging) or self.group is not None:return
        s=self.sim
        if s.base.destination is not None:return
        if self.excavator is not None and self.excavator.state.value=='WAITING_RELOCATION':return
        # A standing transport reserve avoids a tool exchange for each new load.
        queued=any(j.status=='QUEUED' for j in self.jobs)
        for d in s.drones:
            if (d.identifier in self.active or d.identifier in self.direct_ids or not d.grounded
                    or d.charge_requested or s.landing_requested or queued):
                self.idle_since.pop(d.identifier,None);continue
            if d.identifier in self.tool_targets:continue
            if d.attachment!='GRABBER':
                if not self.config.idle_digging:continue
                hooks=sum(o.attachment=='CONTAINER_HOOK' and self.tool_targets.get(o.identifier)!='GRABBER' for o in s.drones)
                if self.excavator is not None and hooks<=1 and len(s.drones)>1:continue
                since=self.idle_since.setdefault(d.identifier,s.time)
                if s.time-since<self.config.idle_dig_delay_s:continue
                # Preview with the real grabber mass, without changing hardware.
                from copy import copy
                preview=copy(d);preview.config=copy(d.config)
                self._tool_mass(preview,self.inventory.items[f'grabber_{d.serial}'].mass_kg)
                if s._choose_trip(preview) is None:continue
                self.request_tool(d,'GRABBER');continue
            trip=s._choose_trip(d)
            if trip is None:continue
            d.trip=trip;self.direct_ids.add(d.identifier)
            self._serial+=1
            job=TransportJob(f'job_{self._serial:05}','EXCAVATE_DIRECT',[f'grabber_{d.serial}'],
                trip.dig,trip.dump,5,s.time,status='ACTIVE',executor=d.serial)
            self.inventory.reserve(job.resource_ids,job.id);self.jobs.append(job)
            d.transition(DS.TAKEOFF)

    def use_station(self, energy):
        b=self.sim.base;s=self.sim
        if self._bus_used+energy>self.config.inverter_power_w:return False
        if not self._solar_ticked:
            solar=(b.irradiance((s.hour+1/7200)%24,s.day_at(s.time+.5))*b.config.panel_area*b.config.panel_efficiency
                   if b.destination is None and self.field_ready else 0.)
            discharge=sum(self.inventory.items[id].max_discharge_w for id in self.base_battery_ids)
            available=min(self.config.inverter_power_w,solar+min(b.battery,discharge))-b.config.idle_power
            if available<self._pending_station_energy+energy:return False
            self._pending_station_energy+=energy
        else:
            if b.battery+self._step_curtailed<energy:return False
            reclaimed=min(self._step_curtailed,energy);from_battery=energy-reclaimed
            b.battery-=from_battery;b.curtailed-=reclaimed;self._step_curtailed-=reclaimed
            b.consumed+=energy
        self.station_consumed+=energy;self._bus_used+=energy
        return True

    def try_base_move(self):
        s=self.sim
        if self.field_rect is not None:return  # deployed large field is stationary
        if s.base.motion_power(s.drones)>min(self.config.inverter_power_w,
                sum(self.inventory.items[id].max_discharge_w for id in self.base_battery_ids)):
            return
        if (self.active or self.group or self.direct_ids or s.base.destination is not None
                or self.battery_service.robot_swap or self.battery_service.drone_swaps
                or any(j.status=='QUEUED' for j in self.jobs)
                or not all(d.docked for d in s.drones) or s.landing_requested):return
        before=s.base.position.copy();s._maybe_relocate()
        if s.base.destination is not None:
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
            mass=self.sim.terrain.deposit_fill(*job.destination[:2],mass)
            self.sim.filled_mass+=mass;d.filled_mass+=mass
        else:self.sim.terrain.deposit(*job.destination[:2],mass)
        c.soil_kg=max(0.,c.soil_kg-mass)
        if c.soil_kg==0:c.cycles+=1;c.wear+=.001
        d.cargo=sum(o.gross_mass for o in self.carried(d))
        d.trips+=1;self.sim.trips+=1;self.delivered_mass+=mass
        d.excavated_mass+=mass
        day=int((self.sim.config.start_hour*3600+self.sim.time)//86400)
        d.productivity.record(self.sim.time,day,mass,self.sim.config.terrain.density,job.soil_destination)
        self.sim._terrain_active=True;self.progress_at=self.sim.time

    def _drop(self, d, job):
        item=self.inventory.items[job.resource_ids[0]]
        at_base=job.kind=='RECOVER_EMPTY_EXCAVATOR_BATTERY'
        location=L.BASE_STORAGE if at_base else L.WORKSITE_RACK
        holder='base' if at_base else 'excavator'
        state='QUEUED_FOR_CHARGE' if isinstance(item,BatteryModule) and at_base else (
            'EMPTY_AT_EXCAVATOR' if isinstance(item,SoilContainer) else 'STORED_READY')
        self.inventory.move(item.id,location,holder,job.destination,state,job.id)
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
                    self._unload_soil(d,job)
                    item=self.inventory.items[job.resource_ids[0]]
                    if item.soil_kg>1e-8:
                        dump=s.terrain.dump_location(item.soil_kg/s.config.terrain.density)
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
        for d in self.sim.drones:
            for item in self.inventory.items.values():
                if item.location==L.DRONE_MOUNT and item.holder_id==f'drone_{d.serial}':
                    item.position=tuple(d.position)
        for id in self.base_battery_ids:
            b=self.inventory.items[id]
            b.energy_j=self.sim.base.battery*b.capacity_j/self.sim.base.config.battery
            b.position=tuple(self.sim.base.position)
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
                b.position=tuple(d.position)
        self.inventory.validate()
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
        self._bus_used=0.
        self._pending_station_energy=0.;self._solar_ticked=False;self._step_curtailed=0.
        before_bus=s.base.consumed+s.base.charging_input+s.base.laser_input
        before_curtailed=s.base.curtailed
        self.try_base_move();base_before=s.base.position.copy()
        s._starts={d.identifier:d.position.copy() for d in s.drones}
        for d in s.drones:
            if d.identifier not in self.active and s._charging_needed(d):
                if d.docked:d.state=DS.CHARGING;d.charging_seconds+=1.
                elif d.state==DS.WAIT_GROUND:s._enqueue(d)
        s._assign_slots()
        if s.base.destination is None:self.battery_service.tick_drone_swaps()
        consumers=self.battery_service.charging()
        flags=[d.docked and d.charge_requested and d.serial not in self.battery_service.drone_swaps for d in s.drones]
        for d in s.drones:
            d.max_charge_input_w=sum(self.inventory.items[id].max_charge_w for id in d.battery_ids
                                     if self.inventory.items[id].soc<1-1e-12)
        s.laser_links=prepare_links(s.base,s.drones,s.terrain,1.,s.flight_power,s.config.fleet.separation/4,
            dust=s.dust,obstacles=[self.excavator] if self.excavator is not None else ())
        area=s.base.config.panel_area
        if not self.field_ready:
            s.base.config.panel_area=0.
            if self.use_station(200.):
                self.field_elapsed+=1.
            if self.field_elapsed>=self.config.field_deployment_seconds:self.field_ready=True
        charge_power=s.base.config.charge_power
        s.base.config.charge_power=min(charge_power,self.config.bus_power_w)
        s.base.tick(1.,(s.hour+1/7200)%24,s.drones,flags,s.day_at(s.time+.5),
                    s.laser_links,modules=consumers,bus_limit_w=self.config.inverter_power_w,
                    storage_charge_w=sum(self.inventory.items[id].max_charge_w for id in self.base_battery_ids),
                    storage_discharge_w=sum(self.inventory.items[id].max_discharge_w for id in self.base_battery_ids),
                    auxiliary_w=self._pending_station_energy)
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
        if self.base_move_job and s.base.destination is None:
            self.base_move_job.status='COMPLETED';self.inventory.release(self.base_move_job.id)
            self.base_move_job=None;self.progress_at=s.time
        self.battery_service.after_charging()
        self.try_lift()
        if self.group is None and s.base.destination is None:self.generate_jobs()
        if self.group is None and s.base.destination is None:self.service_tools()
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
                    job.status='COMPLETED';self.inventory.release(job.id);self.progress_at=s.time
            elif not d.grounded:
                s._airborne(d,1.)
            else:d.waiting_seconds+=1.
            if d.identifier in self.parking:
                if d.position[2]>=s.cruise(d)-1e-8:s._clear_slot(d)
                if d.grounded:self.parking.remove(d.identifier)
            s._settle_laser(d)
        for d in s.drones:s._settle_laser(d)
        if s._terrain_active:
            s._terrain_active=s.terrain.relax()>s.config.terrain.avalanche_stop_volume
        s.dust.tick(1.,s.drone.position,0.,1.,False)
        self.sync_batteries()
        s.base.config.mass=s.base.config.equipment.mass+sum(o.gross_mass for o in self.inventory.items.values()
            if o.holder_id=='base' and o.location in (L.BASE_STORAGE,L.BASE_CHARGER))
        self.detect_stall()
        if (s.terrain.remaining<=s.config.terrain.completion_volume and not s._terrain_active
                and not self.active and all(c.soil_kg==0 for c in self.containers)
                and self.group is None and not self.direct_ids
                and (self.excavator is None or self.excavator.bucket_kg==0)
                and all(d.grounded for d in s.drones)):
            s.state=MissionState.COMPLETED;s.event('Логистическая миссия завершена')
            for job in self.jobs:
                if job.status=='QUEUED':self.cancel_job(job.id)
            for d in s.drones:
                self.inventory.release(f'tool_{d.serial}')
                if d.state==DS.CHANGING_TOOL:d.state=DS.DOCKED
            self.tool_targets.clear();self.tool_changes.clear()
        s.time+=1.

    def detect_stall(self):
        s=self.sim
        if s.state!=MissionState.RUNNING or self.active or s.landing_requested:return
        if s.terrain.remaining<=s.config.terrain.completion_volume:return
        if self.excavator is not None:
            e=self.excavator
            if e.state.value=='BLOCKED':s.block('BLOCKED_EXCAVATOR: '+e.reason)
            elif not self.containers and not (self.config.direct_allowed or self.config.idle_digging):
                s.block('BLOCKED_NO_CONTAINERS: запас тары равен нулю')
            elif e.state.value=='LOW_BATTERY' and self.config.excavator_battery_count<=self.config.excavator_battery_slots:
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
            charging=bool(self.battery_service.queue or self.battery_service.robot_swap or self.battery_service.drone_swaps)
            if charging:
                self.wait_reason='WAITING_SUN_OR_CHARGER: конечная зарядная очередь'
            elif self.excavator is not None and self.excavator.state.value=='WAITING_RELOCATION':
                self.wait_reason='WAITING_GROUP_SOC'
            elif not (self.config.direct_allowed or self.config.idle_digging):
                s.block('BLOCKED_NO_PROGRESS: '+(self.excavator.reason if self.excavator else 'NO_EXECUTOR'))

    def update_report(self, report, *, full=True):
        s=self.sim
        soil=sum(c.soil_kg for c in self.containers)
        soil+=sum(d.cargo for d in s.drones if d.identifier in self.direct_ids)
        if self.excavator is not None:soil+=self.excavator.bucket_kg
        report['mass_error_kg']=s.terrain.mass+soil-s.terrain.initial_volume*s.config.terrain.density
        if full:report['inventory']=self.inventory.snapshot()
        report['logistics_version']=2
        if full:report['jobs']=[asdict(j) for j in self.jobs]
        report['container_cycles']=sum(c.cycles for c in self.containers)
        report['battery_service']=self.battery_service.snapshot()
        report['collective_lift']=self.group.report() if self.group else None
        report['collective_lift_count']=self.lift_count
        report['delivered_mass_kg']=self.delivered_mass
        report['solar_field_deployed']=self.field_ready
        report['solar_field_rect']=self.field_rect
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
            report['excavator']=self.excavator.report(self.inventory)
            report['consumed_kwh']+=self.excavator.consumed_j/KWH
        report['energy_error_j']+=extra_energy-self.initial_extra_energy
        report['initial_energy_kwh']+=self.initial_extra_energy/KWH
        for row,d in zip(report['drones'],s.drones):
            job=self.active.get(d.identifier)
            row.update(task=job.kind if job else '',attachment=d.attachment,
                       transport_payload_kg=self.payload_limit(d),battery_ids=list(d.battery_ids))
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
