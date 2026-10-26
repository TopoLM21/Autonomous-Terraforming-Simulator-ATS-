"""Krot-160 quasi-static sand excavation; all figures are model assumptions."""
from dataclasses import dataclass, field, asdict
from enum import Enum
import math
import numpy as np
from .inventory import BatteryModule, SoilContainer, Location as L
from .config import KWH
from .activity import ActivityStats
from .productivity import Productivity
from .soil import SoilLoad


class ExcavatorState(str, Enum):
    IDLE='IDLE'
    MOVING_LOCAL='MOVING_LOCAL'
    DEPLOYING_SUPPORTS='DEPLOYING_SUPPORTS'
    DIGGING='DIGGING'
    FILLING_CONTAINER='FILLING_CONTAINER'
    WAITING_EMPTY_CONTAINER='WAITING_EMPTY_CONTAINER'
    WAITING_TRANSPORT='WAITING_TRANSPORT'
    LOW_BATTERY='LOW_BATTERY'
    SWAP_PREPARATION='SWAP_PREPARATION'
    SWAPPING_BATTERIES='SWAPPING_BATTERIES'
    WAITING_RELOCATION='WAITING_RELOCATION'
    RIGGED_FOR_LIFT='RIGGED_FOR_LIFT'
    AIR_TRANSPORT='AIR_TRANSPORT'
    FAULT='FAULT'
    BLOCKED='BLOCKED'


@dataclass
class Excavator:
    config: object
    position: np.ndarray
    battery_ids: list = field(default_factory=list)
    state: ExcavatorState = ExcavatorState.DEPLOYING_SUPPORTS
    buffer_j: float = 0.
    bucket_kg: float = 0.
    soil: object = None
    cycle_factor: float = 1.
    active_container: str | None = None
    elapsed: float = 0.
    supports: bool = False
    target: tuple | None = None
    destination: np.ndarray | None = None
    reason: str = ''
    excavated_kg: float = 0.
    consumed_j: float = 0.
    buffer_charge_losses_j: float = 0.
    working_seconds: float = 0.
    waiting_seconds: float = 0.
    moving_seconds: float = 0.
    cycles: int = 0
    swaps: int = 0
    relocations: int = 0
    activity: ActivityStats = field(default_factory=ActivityStats)
    productivity: Productivity = field(default_factory=Productivity)

    def batteries(self, inventory):
        return [inventory.items[id] for id in self.battery_ids]

    def energy(self, inventory):
        return sum(b.energy_j for b in self.batteries(inventory))

    def capacity(self, inventory):
        return sum(b.capacity_j for b in self.batteries(inventory))

    def mass(self, inventory):
        return self.config.excavator_dry_mass_kg+sum(b.mass_kg for b in self.batteries(inventory))+self.bucket_kg

    def consume(self, joules, inventory, control=False):
        packs=self.batteries(inventory)
        energy=sum(b.energy_j for b in packs)
        if control and (self.state in (ExcavatorState.SWAPPING_BATTERIES,ExcavatorState.AIR_TRANSPORT)
                        or not packs or energy<joules):
            if self.buffer_j+1e-9<joules:return False
            self.buffer_j=max(0.,self.buffer_j-joules)
        else:
            if energy+1e-9<joules or sum(b.max_discharge_w for b in packs)<joules:
                return False
            for b in packs:b.energy_j=max(0.,b.energy_j-joules*b.energy_j/energy)
        self.consumed_j+=joules
        return True

    def stable(self, point, terrain):
        c=self.config;z=terrain.sample(*point[:2]);side=terrain.height.shape[0]*terrain.cell
        radius=c.support_half_width_m
        site=getattr(terrain,'shared_site',None)
        if site is not None and site.ground_occupied(terrain.owner_id,(point[0]-radius,point[1]-radius,2*radius,2*radius)):return False
        if np.any(np.asarray(point[:2])<radius+terrain.cell) or np.any(np.asarray(point[:2])>side-radius-terrain.cell):
            return False
        limit=math.tan(math.radians(c.max_slope_deg))
        for dx,dy in ((radius,0),(-radius,0),(0,radius),(0,-radius),(radius,radius),(-radius,-radius)):
            if abs(terrain.sample(point[0]+dx,point[1]+dy)-z)>limit*math.hypot(dx,dy)+1e-9:
                return False
        return True

    def reachable_mask(self, terrain, inventory, position=None, patch=None):
        c=self.config;p=self.position if position is None else np.asarray(position)
        patch=patch if patch is not None else np.s_[:,:]
        distance=np.hypot(terrain.x[patch]-p[0],terrain.y[patch]-p[1])
        density=max(m.density for m in terrain.soil.profile.materials) if terrain.soil is not None else terrain.config.terrain.density
        moment=(c.digging_force_n+c.bucket_volume_m3*c.bucket_fill_factor*density*9.80665)*distance
        restoring=(self.mass(inventory)-self.bucket_kg)*9.80665*c.support_half_width_m/c.tipping_margin
        return ((distance<=c.reach_m) & (distance>=c.support_half_width_m+terrain.cell/math.sqrt(2))
                & (terrain.height[patch]>p[2]-c.local_depth_m+1e-9)
                & (terrain.height[patch]<=p[2]+.5) & (moment<=restoring))

    def work_point(self, terrain, inventory):
        if not self.stable(self.position,terrain):return None
        patch=terrain.patch(*self.position[:2],self.config.reach_m)
        available=np.where(terrain.excavation[patch]&self.reachable_mask(terrain,inventory,patch=patch),
                           np.maximum(terrain.height[patch]-terrain.target[patch],0),0.)
        if terrain.front is not None:
            terrain.front.update()
            available=np.where(terrain.front.chainage[patch]<=terrain.front.position+terrain.front.settings['front_length_m'],available,0.)
        if float(available.sum()*terrain.area)<=min(.001,terrain.config.terrain.completion_volume/10):
            return None
        indices=np.flatnonzero(available.ravel()>1e-10)
        if not len(indices):return None
        # Stable tie-breaking, prefer deeper cells while retaining boom clearance.
        from .laser import visible
        for index in indices[np.argsort(-available.ravel()[indices],kind='stable')]:
            p=np.array([terrain.x[patch].ravel()[index],terrain.y[patch].ravel()[index],terrain.height[patch].ravel()[index]+.02])
            if visible(self.position+[0,0,.8],p,terrain):return tuple(p[:2])
        return None

    def find_site(self, terrain, inventory):
        indices=terrain.source_indices('cut')
        if not len(indices):return None
        # Choose a flat supporting footprint next to real remaining work.
        candidates=[]
        distances=(terrain.x.ravel()[indices]-self.position[0])**2+(terrain.y.ravel()[indices]-self.position[1])**2
        nearest=indices[np.argsort(distances,kind='stable')[:64]]
        sampled=indices[::max(1,len(indices)//64)]
        shortlist=np.unique(np.r_[nearest,sampled]) if terrain.front is not None else sampled
        for index in shortlist:
            x,y=terrain.x.ravel()[index],terrain.y.ravel()[index]
            for dx,dy in ((0,-1),(0,1),(-1,0),(1,0)):
                p=np.array([x+dx*self.config.reach_m*.75,y+dy*self.config.reach_m*.75,0.])
                p[2]=terrain.sample(*p[:2])
                if self.stable(p,terrain):
                    distance=float(np.linalg.norm(p-self.position))
                    if distance<=terrain.cell:continue
                    patch=terrain.patch(*p[:2],self.config.reach_m)
                    excess=np.where(terrain.excavation[patch],np.maximum(terrain.height[patch]-terrain.target[patch],0),0.)
                    if terrain.front is not None:
                        excess=np.where(terrain.front.chainage[patch]<=terrain.front.position+terrain.front.settings['front_length_m'],excess,0.)
                    mask=self.reachable_mask(terrain,inventory,p,patch)
                    volume=float(excess[mask].sum()*terrain.area)
                    footprint=(terrain.x[patch]-p[0])**2+(terrain.y[patch]-p[1])**2<=self.config.support_half_width_m**2
                    under_tracks=float(excess[footprint].sum()*terrain.area)
                    if volume>min(.001,terrain.config.terrain.completion_volume/10):
                        candidates.append(((distance+.5+50*under_tracks)/(volume+.01),tuple(p)))
        return np.array(min(candidates,key=lambda v:(v[0],v[1]))[1]) if candidates else None

    def park_clear_of_cut(self, terrain):
        """Vacate the last soil under the tracks so aircraft can finish it."""
        air_site=None
        for radius in (self.config.reach_m*1.5,self.config.reach_m*2.5):
            for angle in np.arange(8)*math.pi/4:
                p=self.position.copy();p[:2]+=radius*np.array([math.cos(angle),math.sin(angle)])
                patch=terrain.patch(*p[:2],self.config.support_half_width_m+terrain.cell)
                if terrain.height[patch].size==0:continue
                if np.any(terrain.excavation[patch]|terrain.dump_mask[patch]):continue
                if terrain.protected is not None and np.any(terrain.protected[patch]):continue
                p[2]=terrain.sample(*p[:2])
                if self.stable(p,terrain):
                    if self.can_drive(p,terrain):return p
                    if air_site is None:air_site=p
        return air_site

    def can_drive(self, destination, terrain):
        length=np.linalg.norm(destination[:2]-self.position[:2])
        samples=max(2,int(math.ceil(length/(terrain.cell/2)))+1)
        previous=self.position
        for p in np.linspace(self.position,destination,samples):
            if not self.stable(p,terrain):return False
            z=terrain.sample(*p[:2]);horizontal=np.linalg.norm(p[:2]-previous[:2])
            if horizontal>0 and abs(z-previous[2])>math.tan(math.radians(self.config.max_slope_deg))*horizontal+1e-9:
                return False
            previous=np.array([*p[:2],z])
        return True

    def tick(self, logistics):
        s,c,inv=logistics.sim,self.config,logistics.inventory;t=s.terrain
        if self.state in (ExcavatorState.AIR_TRANSPORT,ExcavatorState.RIGGED_FOR_LIFT):
            return
        self.position[2]=t.sample(*self.position[:2])
        if self.state==ExcavatorState.SWAPPING_BATTERIES:return
        self.elapsed+=1.
        if (self.battery_ids and self.buffer_j<c.buffer_kwh*KWH-1e-8
                and self.energy(inv)>self.capacity(inv)*c.excavator_reserve):
            energy=min(300.,(c.buffer_kwh*KWH-self.buffer_j)/.9)
            if self.consume(energy,inv):
                stored=energy*.9;self.buffer_j+=stored
                self.consumed_j-=stored;self.buffer_charge_losses_j+=energy-stored
        idle=c.sleep_power_w if not self.battery_ids or self.energy(inv)<self.capacity(inv)*c.sleep_soc else c.idle_power_w
        if not self.consume(idle,inv,control=True):
            self.state=ExcavatorState.LOW_BATTERY;self.reason='CONTROL_OFF_WAITING_DOCK_POWER';return
        if self.state==ExcavatorState.WAITING_RELOCATION:
            self.waiting_seconds+=1.;return
        if self.state==ExcavatorState.MOVING_LOCAL:
            if not self.can_drive(self.destination,t):
                self.state=ExcavatorState.WAITING_RELOCATION;self.reason='LOCAL_PATH_BLOCKED';return
            horizontal=float(np.linalg.norm(self.destination[:2]-self.position[:2]))
            grade=abs(self.destination[2]-self.position[2])/max(horizontal,t.cell)
            if not self.consume(c.moving_power_w*(1.2+5*grade),inv):
                self.state=ExcavatorState.LOW_BATTERY;self.reason='MOVEMENT_ENERGY';return
            self.supports=False;difference=self.destination-self.position
            length=np.linalg.norm(difference[:2]);fraction=min(1.,c.excavator_speed/max(length,1e-9))
            self.position[:2]+=difference[:2]*fraction;self.position[2]=t.sample(*self.position[:2])
            self.moving_seconds+=1.;logistics.rack=self.position.copy()
            if length<=c.excavator_speed:
                self.state=ExcavatorState.DEPLOYING_SUPPORTS;self.elapsed=0.;self.relocations+=1
            return
        if (self.reason=='DRONES_FINISH_LOCAL_REMAINDER' and
                s.time<getattr(self,'_site_retry_at',0)):
            self.waiting_seconds+=1.;return
        if self.state==ExcavatorState.DEPLOYING_SUPPORTS:
            if not self.stable(self.position,t):
                self.state=ExcavatorState.BLOCKED;self.reason='UNSTABLE_SUPPORT_FOOTPRINT';return
            if self.elapsed<c.support_seconds:return
            self.supports=True;self.state=ExcavatorState.IDLE
        if self.bucket_kg>0:
            self.working_seconds+=1.
            if not self.consume(max(0.,c.digging_power_w-c.idle_power_w),inv):
                self.reason='BUCKET_AWAITING_POWER';return
            self.state=ExcavatorState.FILLING_CONTAINER
            if self.elapsed<c.cycle_seconds*self.cycle_factor:return
            container=inv.items[self.active_container]
            container.soil_kg+=self.bucket_kg
            if self.soil is not None:
                if container.soil is None:container.soil=SoilLoad.empty(t.soil.profile)
                container.soil.add(self.soil);self.soil=None
            self.bucket_kg=0.
            container.state='PARTIALLY_FILLED';self.cycles+=1
            inv.release('excavator_cycle');self.active_container=None
            if container.full(s.config.terrain.density) or t.remaining_cut<=s.config.terrain.completion_volume:
                container.location=L.FILLED_QUEUE;container.state='FULL_READY'
            self.state=ExcavatorState.IDLE;self.elapsed=0.;logistics.progress_at=s.time
            return
        if t.remaining_cut<=s.config.terrain.completion_volume:
            inv.release('excavator_cycle');self.active_container=None
            for container in logistics.containers:
                if container.location==L.WORKSITE_RACK and container.soil_kg>0 and not container.reserved_by:
                    container.location=L.FILLED_QUEUE;container.state='FULL_READY'
            self.state=ExcavatorState.IDLE;self.reason='WORK_COMPLETE';return
        if s.landing_requested:
            inv.release('excavator_cycle');self.active_container=None;self.elapsed=0.
            for container in logistics.containers:
                if container.location==L.WORKSITE_RACK and container.soil_kg>0 and not container.reserved_by:
                    container.location=L.FILLED_QUEUE;container.state='FULL_READY'
            self.state=ExcavatorState.IDLE;self.reason='PAUSED_FOR_PLAN_OR_MAINTENANCE';return
        cycle_budget=c.digging_power_w*c.cycle_seconds*(max(t.soil.profile.factors) if t.soil is not None else 1.)
        stop_energy=(max(self.capacity(inv)*c.courier_low_soc,cycle_budget) if logistics.batch_enabled else
                     self.capacity(inv)*c.excavator_reserve+cycle_budget)
        if self.active_container is None and self.energy(inv)<=stop_energy:
            self.state=ExcavatorState.LOW_BATTERY;self.reason='WAITING_CHARGED_BATTERY_PAIR';self.waiting_seconds+=1.;return
        point=self.work_point(t,inv)
        if point is None:
            inv.release('excavator_cycle');self.active_container=None;self.target=None;self.elapsed=0.
            for container in logistics.containers:
                if container.location==L.WORKSITE_RACK and container.soil_kg>0 and not container.reserved_by:
                    container.location=L.FILLED_QUEUE;container.state='FULL_READY'
            if t.remaining_cut<=s.config.terrain.completion_volume:
                self.state=ExcavatorState.IDLE;self.reason='LOCAL_WORK_COMPLETE';return
            if any(j.status in ('ACTIVE','QUEUED') and (j.kind!='EXCAVATE_DIRECT' or not logistics.batch_enabled)
                   for j in logistics.jobs):
                self.state=ExcavatorState.WAITING_TRANSPORT;self.reason='CLEAR_WORKSITE_BEFORE_MOVE';self.waiting_seconds+=1.;return
            site=self.find_site(t,inv)
            if site is None:
                if self.config.direct_allowed or self.config.idle_digging:
                    parked=self.park_clear_of_cut(t)
                    self._site_retry_at=s.time+30
                    self.reason='DRONES_FINISH_LOCAL_REMAINDER'
                    if parked is not None:
                        self.destination=parked;self.supports=False
                        if self.can_drive(parked,t):self.state=ExcavatorState.MOVING_LOCAL
                        elif self.config.air_relocation:self.state=ExcavatorState.WAITING_RELOCATION
                        else:self.state=ExcavatorState.IDLE
                    else:self.state=ExcavatorState.IDLE
                    self.waiting_seconds+=1.;return
                self.state=ExcavatorState.BLOCKED;self.reason='NO_ACCESSIBLE_WORKSITE';return
            self.destination=site;self.supports=False
            if self.can_drive(site,t):self.state=ExcavatorState.MOVING_LOCAL
            else:self.state=ExcavatorState.WAITING_RELOCATION;self.reason='LOCAL_PATH_BLOCKED'
            return
        containers=[o for o in logistics.containers if o.location==L.WORKSITE_RACK and not o.reserved_by
                    and np.linalg.norm(np.array(o.position[:2])-self.position[:2])<=c.reach_m
                    and not o.full(s.config.terrain.density)]
        if self.active_container is not None:
            container=inv.items[self.active_container]
        elif not containers:
            self.state=ExcavatorState.WAITING_EMPTY_CONTAINER;self.reason='NO_EMPTY_CONTAINER';self.waiting_seconds+=1.;return
        else:
            container=min(containers,key=lambda o:(-o.soil_kg,o.id))
        if self.active_container is None:
            if not inv.reserve([container.id],'excavator_cycle'):return
            self.active_container=container.id;self.elapsed=0.;self.target=point
            if t.soil is not None:
                volume=min(c.bucket_volume_m3*c.bucket_fill_factor,max(0.,container.volume_m3-container.soil_volume(s.config.terrain.density)))
                preview=t.preview_load(*self.target,max(0.,container.assigned_limit_kg-container.soil_kg),radius=.5,
                    max_volume=volume,allowed_mask=self.reachable_mask(t,inv),minimum_height=self.position[2]-c.local_depth_m)
                self.cycle_factor=preview.factor
        self.state=ExcavatorState.DIGGING;self.reason='';self.working_seconds+=1.
        if not self.supports or sum(b.max_discharge_w for b in self.batteries(inv))<c.peak_power_w:
            self.state=ExcavatorState.BLOCKED;self.reason='INSUFFICIENT_STABILITY_OR_PEAK_POWER';return
        if not self.consume(max(0.,c.digging_power_w-c.idle_power_w),inv):return
        if self.elapsed>=c.cycle_seconds*self.cycle_factor/2:
            if t.soil is not None:
                volume=min(c.bucket_volume_m3*c.bucket_fill_factor,max(0.,container.volume_m3-container.soil_volume(s.config.terrain.density)))
                self.soil=t.scoop_load(*self.target,max(0.,container.assigned_limit_kg-container.soil_kg),radius=.5,
                    max_volume=volume,allowed_mask=self.reachable_mask(t,inv),minimum_height=self.position[2]-c.local_depth_m)
                self.bucket_kg=self.soil.mass;s.excavated_volume+=self.soil.volume
            else:
                max_mass=min(c.bucket_volume_m3*c.bucket_fill_factor*s.config.terrain.density,
                             container.assigned_limit_kg-container.soil_kg)
                self.bucket_kg=t.scoop(*self.target,max_mass,radius=.5,
                    allowed_mask=self.reachable_mask(t,inv),minimum_height=self.position[2]-c.local_depth_m)
            self.excavated_kg+=self.bucket_kg;s.excavated_mass+=self.bucket_kg
            self.productivity.record(s.time,int((s.config.start_hour*3600+s.time)//86400),
                                     self.bucket_kg,s.config.terrain.density,'excavated',volume=self.soil.volume if self.soil is not None else None)
            s._terrain_active=True
            s.dust.emit(1.,self.position,0.,.15)
            if self.bucket_kg<=1e-8:
                inv.release('excavator_cycle');self.active_container=None;self.elapsed=0.

    def report(self, inv, time=0.,day=0,density=1600.):
        production=self.productivity.report(time,day)
        return dict(state=self.state.value,position=self.position.tolist(),reason=self.reason,
                    excavated_m3=self.excavated_kg/density,activity=self.activity.report(),
                    today_excavated_kg=production['today_mass_kg'],today_excavated_m3=production['today_volume_m3'],
                    last24_excavated_kg=production['last24_mass_kg'],last24_excavated_m3=production['last24_volume_m3'],
                    daily_excavation=production['daily'],
                    battery_ids=list(self.battery_ids),energy_kwh=self.energy(inv)/KWH,
                    capacity_kwh=self.capacity(inv)/KWH,buffer_kwh=self.buffer_j/KWH,
                    buffer_charge_losses_kwh=self.buffer_charge_losses_j/KWH,
                    mass_kg=self.mass(inv),bucket_kg=self.bucket_kg,container=self.active_container,
                    excavated_kg=self.excavated_kg,consumed_kwh=self.consumed_j/KWH,
                    working_seconds=self.working_seconds,waiting_seconds=self.waiting_seconds,
                    moving_seconds=self.moving_seconds,cycles=self.cycles,swaps=self.swaps,
                    relocations=self.relocations,supports=self.supports)
