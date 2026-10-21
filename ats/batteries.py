"""Physical charging ports adapt to the existing shared solar bus."""
from dataclasses import dataclass
from types import SimpleNamespace
from .inventory import BatteryModule, ChargingSlot, Location as L


class ChargePort:
    def __init__(self, pack):
        self.pack=pack
        self.config=SimpleNamespace(battery=pack.capacity_j)

    @property
    def battery(self):return self.pack.energy_j

    @battery.setter
    def battery(self, value):self.pack.energy_j=value

    @property
    def charged_energy(self):return self.pack.charged_energy

    @charged_energy.setter
    def charged_energy(self, value):self.pack.charged_energy=value


class BatteryService:
    def __init__(self, logistics):
        self.world=logistics
        self.queue=[]
        self.ports=[]
        self.robot_swap=None
        self.drone_swaps={}
        self.drone_swap_count=0
        self.drone_swap_counts={}
        self.slots={}
        self._charge_before={}
        self._module_power={}
        slots=self.make_slots(logistics.inventory.select(BatteryModule))
        logistics.inventory.add_many(slots)
        self.slots.update((slot.module_id,slot) for slot in slots)
        self.after_charging()

    def make_slots(self, packs):
        w=self.world
        if not w.config.individual_chargers:return []
        return [ChargingSlot(f'charge_slot_{pack.id}',w.config.charger_slot_mass_kg,
                L.BASE_MOUNT,'EMPTY','base',position=tuple(w.sim.base.position),module_id=pack.id,
                rated_w=min(w.config.port_power_w,pack.max_charge_w))
                for pack in packs if pack.compatibility!='base']

    @property
    def capacity(self):
        return len(self.slots) if self.world.config.individual_chargers else self.world.config.charger_ports

    def ready(self, compatibility, location=L.BASE_STORAGE, minimum_soc=.999999):
        return [b for b in self.world.inventory.select(BatteryModule,location)
                if b.compatibility==compatibility and b.soc>=minimum_soc and not b.reserved_by]

    def generate_jobs(self):
        w=self.world;e=w.excavator;c=w.config
        if e is None:return
        capacity=e.capacity(w.inventory)
        # Conservative delivery lead time: round trip, side dock, charge queue.
        distance=float(((e.position[:2]-w.sim.base.position[:2])**2).sum()**.5)
        lead=2*distance/min(d.config.loaded_speed for d in w.sim.drones)+2*c.battery_swap_seconds+90.
        threshold=capacity*c.excavator_reserve+c.digging_power_w*lead
        due=e.energy(w.inventory)<=threshold or len(e.battery_ids)<c.excavator_battery_slots
        if w.batch_enabled and not w.courier_active:
            due=(len(e.battery_ids)<c.excavator_battery_slots or
                 e.energy(w.inventory)<capacity*c.courier_low_soc or e.state.value=='LOW_BATTERY')
        if due:
            rack=self.ready('excavator',L.WORKSITE_RACK)
            incoming=[j for j in w.jobs if j.kind=='DELIVER_EXCAVATOR_BATTERY' and j.status in ('QUEUED','ACTIVE')]
            needed=c.excavator_battery_slots-len(rack)-len(incoming)
            for b in self.ready('excavator')[:max(0,needed)]:
                w.new_job('DELIVER_EXCAVATOR_BATTERY',b,b.position,w.rack,1)
        for b in w.inventory.select(BatteryModule,L.WORKSITE_RACK):
            if b.compatibility=='excavator' and (b.state=='AWAITING_PICKUP' or
                    b.state=='STORED_READY' and b.soc<.999999):
                w.new_job('RECOVER_EMPTY_EXCAVATOR_BATTERY',b,b.position,w.sim.base.position,2)

    def charging(self):
        w=self.world;inv=w.inventory;c=w.config
        eligible={}
        for b in inv.select(BatteryModule):
            if b.reserved_by or b.soc>=1-1e-12:continue
            if b.location in (L.BASE_STORAGE,L.BASE_CHARGER):eligible[b.id]=b
        for id in list(self.queue):
            if id not in eligible:self.queue.remove(id)
        for id in eligible:
            if id not in self.queue:self.queue.append(id)
        self.ports=list(self.queue) if c.individual_chargers else self.queue[:c.charger_ports]
        if c.individual_chargers and any(id not in self.slots for id in self.ports):
            raise ValueError('Battery has no assigned physical charging slot')
        self._charge_before={id:inv.items[id].charged_energy for id in self.ports}
        self._module_power={}
        for slot in self.slots.values():slot.input_power_w=0.
        for b in inv.select(BatteryModule):
            if b.location==L.BASE_CHARGER and b.id not in self.ports:
                b.location=L.BASE_STORAGE;b.state='QUEUED_FOR_CHARGE'
        consumers=[]
        for id in self.ports:
            b=inv.items[id]
            if b.location!=L.DRONE_MOUNT:b.location=L.BASE_CHARGER
            b.state='CHARGING'
            consumers.append((ChargePort(b),min(c.port_power_w,b.max_charge_w)))
        return consumers

    def after_charging(self):
        w=self.world
        self._module_power={id:max(0.,w.inventory.items[id].charged_energy-before)/w.sim.base.config.charge_efficiency
                            for id,before in self._charge_before.items()}
        for b in w.inventory.select(BatteryModule):
            if b.state=='CHARGING' and b.soc>=1-1e-12:
                b.state='INSTALLED' if b.location==L.DRONE_MOUNT else 'STORED_READY'
                if b.location==L.BASE_CHARGER:b.location=L.BASE_STORAGE
        self.update_slot_states()

    def update_slot_states(self):
        w=self.world
        for id,slot in self.slots.items():
            pack=w.inventory.items[id]
            slot.input_power_w=self._module_power.get(id,0.)
            slot.state=('EMPTY' if pack.location not in (L.BASE_STORAGE,L.BASE_CHARGER) else
                        'RESERVED' if pack.reserved_by else 'FULL' if pack.soc>=1-1e-12 else
                        'CHARGING' if slot.input_power_w>0 else 'WAITING_POWER')

    def tick_robot_swap(self):
        from .excavator import ExcavatorState as ES
        w=self.world;e=w.excavator;c=w.config;inv=w.inventory
        if e is None:return
        if self.robot_swap is None:
            ready=self.ready('excavator',L.WORKSITE_RACK,minimum_soc=.5 if not e.battery_ids else .999999)
            if (len(ready)>=c.excavator_battery_slots and e.bucket_kg==0 and e.active_container is None
                    and (e.energy(inv)<e.capacity(inv)*.5 or len(e.battery_ids)<c.excavator_battery_slots)):
                ids=[b.id for b in ready[:c.excavator_battery_slots]]
                if inv.reserve(ids+e.battery_ids,'excavator_swap'):
                    self.robot_swap={'ids':ids,'index':0,'elapsed':0.}
                    e.state=ES.SWAPPING_BATTERIES;e.reason='SIDE_DOCK_SEQUENTIAL_SWAP'
                    e.supports=True
        if self.robot_swap is None:return
        if not e.consume(c.idle_power_w,inv,control=True):
            # A physically delivered pack powers the side dock through its
            # service connector. No mining is possible from the control buffer.
            pack=inv.items[self.robot_swap['ids'][self.robot_swap['index']]]
            if pack.energy_j<c.idle_power_w:
                e.state=ES.LOW_BATTERY;e.reason='DOCK_POWER_UNAVAILABLE';return
            pack.energy_j-=c.idle_power_w;e.consumed_j+=c.idle_power_w
        swap=self.robot_swap;swap['elapsed']+=1.
        if swap['elapsed']<c.battery_swap_seconds:return
        index=swap['index'];new=inv.items[swap['ids'][index]]
        if index<len(e.battery_ids):
            old=inv.items[e.battery_ids[index]]
            inv.move(old.id,L.WORKSITE_RACK,'excavator',e.position,'AWAITING_PICKUP','excavator_swap')
            e.battery_ids[index]=new.id
        else:e.battery_ids.append(new.id)
        inv.move(new.id,L.EXCAVATOR_SLOT,'excavator',e.position,'INSTALLED','excavator_swap')
        new.swaps+=1;swap['index']+=1;swap['elapsed']=0.
        w.sim.event(f'Боковой док: установлен {new.id}')
        if swap['index']>=len(swap['ids']):
            inv.release('excavator_swap');self.robot_swap=None
            e.swaps+=1;e.state=ES.IDLE;e.reason='';e.elapsed=0.
            w.progress_at=w.sim.time

    def tick_drone_swaps(self):
        from .drone import DroneState as DS
        w=self.world;c=w.config;inv=w.inventory
        if not c.swap_station or c.inverter_power_w<200.:return
        for d in w.sim.drones:
            if not d.docked or not d.charge_requested:continue
            owner=f'drone_swap_{d.serial}'
            if d.serial not in self.drone_swaps:
                if w.sim.relocation_pending is not None or d.identifier in w.recall_ids:continue
                mounted=[inv.items[id] for id in d.battery_ids]
                ids=[]
                for old in mounted:
                    ready=self.ready(old.compatibility)
                    match=next((b for b in ready if b.id not in ids and b.capacity_j==old.capacity_j and b.mass_kg==old.mass_kg),None)
                    if match is None:break
                    ids.append(match.id)
                if len(ids)==len(mounted) and inv.reserve(ids+d.battery_ids,owner):
                    self.drone_swaps[d.serial]={'ids':ids,'old_ids':list(d.battery_ids),'elapsed':0.}
                    w.sim.drone_event(d,f'смена батарей: {", ".join(d.battery_ids)} → {", ".join(ids)}; {c.drone_swap_seconds:g} с')
            swap=self.drone_swaps.get(d.serial)
            if swap is None:continue
            d.state=DS.SWAPPING_BATTERIES
            # Mechanical dock uses finite base energy, separately from charging.
            power=200.
            if not w.use_station(power):continue
            swap['elapsed']+=1.
            if swap['elapsed']<c.drone_swap_seconds:continue
            for id in d.battery_ids:
                inv.move(id,L.BASE_STORAGE,'base',w.sim.base.position,'QUEUED_FOR_CHARGE',owner)
            for id in swap['ids']:
                inv.move(id,L.DRONE_MOUNT,f'drone_{d.serial}',d.position,'INSTALLED',owner)
                inv.items[id].swaps+=1
            d.battery_ids=swap['ids'];d.battery=sum(inv.items[id].energy_j for id in d.battery_ids)
            d.charge_requested=False;d.charge_cycles+=1;d.state=DS.DOCKED
            inv.release(owner);del self.drone_swaps[d.serial];self.drone_swap_count+=1
            self.drone_swap_counts[d.serial]=self.drone_swap_counts.get(d.serial,0)+1
            w.sim.drone_event(d,'роботизированная смена батарей завершена')
            w.progress_at=w.sim.time

    def slot_snapshot(self):
        inv=self.world.inventory
        if self.world.config.individual_chargers:
            return [dict(id=slot.id,module_id=id,state=slot.state,location=inv.items[id].location.value,
                         soc=inv.items[id].soc,rated_w=slot.rated_w,input_power_w=slot.input_power_w)
                    for id,slot in self.slots.items()]
        return [dict(id=f'shared_charge_slot_{i+1}',module_id=id or '',
                     state='CHARGING' if id and self._module_power.get(id,0)>0 else 'WAITING_POWER' if id else 'EMPTY',
                     location=inv.items[id].location.value if id else '',soc=inv.items[id].soc if id else 0.,
                     rated_w=self.world.config.port_power_w,input_power_w=self._module_power.get(id,0.))
                for i in range(self.capacity) for id in [self.ports[i] if i<len(self.ports) else None]]

    def snapshot(self, *, include_slots=False):
        report=dict(ports=list(self.ports),queue=[] if self.world.config.individual_chargers else list(self.queue),
                    mode='individual' if self.world.config.individual_chargers else 'shared',
                    capacity=self.capacity,module_input_power_w=sum(self._module_power.values()),
                    robot_swap=dict(self.robot_swap) if self.robot_swap else None,
                    drone_swaps={serial:dict(swap,remaining_seconds=max(0.,self.world.config.drone_swap_seconds-swap['elapsed']))
                                 for serial,swap in self.drone_swaps.items()},drone_swap_count=self.drone_swap_count,drone_swap_counts=dict(self.drone_swap_counts),
                    dock_charging_drones=[d.serial for d in self.world.sim.drones
                                         if d.docked and d.charge_requested and d.serial not in self.drone_swaps])
        if include_slots:report['slots']=self.slot_snapshot()
        return report
