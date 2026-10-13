"""Physical charging ports adapt to the existing shared solar bus."""
from dataclasses import dataclass
from types import SimpleNamespace
from .inventory import BatteryModule, Location as L


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
        if e.energy(w.inventory)<=threshold or len(e.battery_ids)<c.excavator_battery_slots:
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
        self.ports=self.queue[:c.charger_ports]
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
        for b in w.inventory.select(BatteryModule):
            if b.state=='CHARGING' and b.soc>=1-1e-12:
                b.state='INSTALLED' if b.location==L.DRONE_MOUNT else 'STORED_READY'
                if b.location==L.BASE_CHARGER:b.location=L.BASE_STORAGE

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
        w=self.world;c=w.config;inv=w.inventory
        if not c.swap_station or c.inverter_power_w<200.:return
        for d in w.sim.drones:
            if not d.docked or not d.charge_requested:continue
            owner=f'drone_swap_{d.serial}'
            if d.serial not in self.drone_swaps:
                mounted=[inv.items[id] for id in d.battery_ids]
                ids=[]
                for old in mounted:
                    ready=self.ready(old.compatibility)
                    match=next((b for b in ready if b.id not in ids and b.capacity_j==old.capacity_j and b.mass_kg==old.mass_kg),None)
                    if match is None:break
                    ids.append(match.id)
                if len(ids)==len(mounted) and inv.reserve(ids+d.battery_ids,owner):
                    self.drone_swaps[d.serial]={'ids':ids,'elapsed':0.}
            swap=self.drone_swaps.get(d.serial)
            if swap is None:continue
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
            d.charge_requested=False;d.charge_cycles+=1
            inv.release(owner);del self.drone_swaps[d.serial];self.drone_swap_count+=1
            w.sim.drone_event(d,'роботизированная смена батарей завершена')
            w.progress_at=w.sim.time

    def snapshot(self):
        return dict(ports=list(self.ports),queue=list(self.queue),
                    capacity=self.world.config.charger_ports,robot_swap=self.robot_swap,
                    drone_swaps=self.drone_swaps,drone_swap_count=self.drone_swap_count,
                    dock_charging_drones=[d.serial for d in self.world.sim.drones if d.docked and d.charge_requested])
