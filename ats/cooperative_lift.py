"""Quasi-static lift. Cable dynamics and gust control are outside this model."""
from dataclasses import dataclass, field
import math
import numpy as np
from .inventory import Location as L
from .drone import DroneState as DS
from .excavator import ExcavatorState as ES


def lift_check(excavator, inventory, drones, config, distance=0.):
    mass=excavator.mass(inventory)+config.rigging_mass_kg
    if not drones:return dict(feasible=False,reason='BLOCKED_INSUFFICIENT_LIFT',mass_kg=mass)
    equivalent=mass/(len(drones)*math.cos(math.radians(config.cable_angle_deg)))
    energies=[]
    for d in drones:
        from .equipment import physical_payload_limit
        limit=physical_payload_limit(d)
        if equivalent*config.lift_margin>limit:
            return dict(feasible=False,reason='BLOCKED_INSUFFICIENT_LIFT',mass_kg=mass,equivalent_kg=equivalent)
        seconds=math.ceil(distance/d.config.loaded_speed)+120.
        energy=d.power(moving=True,cargo=equivalent)*seconds*d.config.budget_margin
        energies.append(energy)
        if d.battery-energy<d.config.battery*d.config.reserve:
            return dict(feasible=False,reason='WAITING_GROUP_SOC',mass_kg=mass,equivalent_kg=equivalent)
    return dict(feasible=True,reason='',mass_kg=mass,equivalent_kg=equivalent,energies_j=energies)


@dataclass
class CollectiveLift:
    world: object
    destination: np.ndarray
    participants: list
    phase: str = 'REMOVE_BATTERIES'
    elapsed: float = 0.
    cargo_position: np.ndarray | None = None
    altitude: float = 0.
    removed_ids: list = field(default_factory=list)
    offsets: dict = field(default_factory=dict)

    def __post_init__(self):
        w=self.world;e=w.excavator
        self.cargo_position=e.position.copy()
        self.altitude=float(w.sim.terrain.height.max())+w.sim.config.terrain.dump_max_height+6.
        for i,d in enumerate(self.participants):
            angle=2*math.pi*i/len(self.participants)
            # Formation circumference maintains the existing two-metre separation.
            radius=max(2.,w.sim.config.fleet.separation/(2*math.sin(math.pi/len(self.participants))))
            height=radius/math.tan(math.radians(w.config.cable_angle_deg)) if w.config.cable_angle_deg>0 else 4.
            self.offsets[d.identifier]=np.array([radius*math.cos(angle),radius*math.sin(angle),height])
        e.state=ES.RIGGED_FOR_LIFT;e.supports=False;e.reason='REMOVE_POWER_BATTERIES_BEFORE_LIFT'
        w.inventory.reserve(e.battery_ids+['group_rig'],'collective_lift')

    def transition(self, phase):
        self.phase,self.elapsed=phase,0.

    def tick(self):
        w,s,e,c=self.world,self.world.sim,self.world.excavator,self.world.config
        inv=w.inventory;self.elapsed+=1.
        if self.phase=='REMOVE_BATTERIES':
            if not e.consume(c.idle_power_w,inv,control=True):s.block('CONTROL_BUFFER_EMPTY');return
            if self.elapsed>=c.battery_swap_seconds:
                if e.battery_ids:
                    id=e.battery_ids.pop(0);self.removed_ids.append(id)
                    inv.move(id,L.WORKSITE_RACK,'excavator',e.position,'STORED_READY','collective_lift')
                    self.elapsed=0.
                else:
                    for id in self.removed_ids:inv.items[id].reserved_by=None
                    self.transition('GATHER')
                    for d in self.participants:
                        s._clear_slot(d);d.transition(DS.TAKEOFF)
        elif self.phase=='GATHER':
            arrived=True
            for d in self.participants:
                if not s.consume_drone(d.power(moving=True),d):return
                d.airborne_seconds+=1.;d.flight_seconds+=1.
                # Each starts along its separate cruising layer, then joins the formation.
                target=e.position+self.offsets[d.identifier]
                if d.position[2]<s.cruise(d)-1e-7 and np.linalg.norm(d.position[:2]-target[:2])>1e-7:
                    target=np.array([*d.position[:2],s.cruise(d)])
                elif np.linalg.norm(d.position[:2]-target[:2])>1e-7:
                    target[2]=s.cruise(d)
                s._move(d,target,DS.TAKEOFF,1.)
                arrived &= np.linalg.norm(d.position-(e.position+self.offsets[d.identifier]))<1e-7
            if arrived:
                self.transition('RIG');inv.move('group_rig',L.WORKSITE_RACK,'excavator',e.position,'RIGGED','collective_lift')
        elif self.phase=='RIG':
            for d in self.participants:
                if not s.consume_drone(d.power(winching=True),d):return
                d.airborne_seconds+=1.;d.flight_seconds+=1.
            if self.elapsed>=20.:
                self.transition('LIFT');e.state=ES.AIR_TRANSPORT
                inv.move('group_rig',L.IN_TRANSFER,'lift_team',e.position,'AIR_TRANSPORT','collective_lift')
                load=lift_check(e,inv,self.participants,c)['equivalent_kg']
                physical=(e.mass(inv)+c.rigging_mass_kg)/len(self.participants)
                for d in self.participants:d.cargo=physical;d.equivalent_load=load
        elif self.phase in ('LIFT','FLY','LOWER'):
            for d in self.participants:
                if not s.consume_drone(d.power(moving=True),d):return
                d.airborne_seconds+=1.;d.flight_seconds+=1.
            previous=self.cargo_position.copy()
            if self.phase=='LIFT':
                self.cargo_position[2]=min(self.altitude,self.cargo_position[2]+min(d.config.vertical_speed for d in self.participants))
                if self.cargo_position[2]>=self.altitude:self.transition('FLY')
            elif self.phase=='FLY':
                delta=self.destination[:2]-self.cargo_position[:2];length=np.linalg.norm(delta)
                speed=min(d.config.loaded_speed for d in self.participants)
                self.cargo_position[:2]+=delta*min(1.,speed/max(length,1e-9))
                if length<=speed:self.transition('LOWER')
            else:
                self.cargo_position[2]=max(self.destination[2],self.cargo_position[2]-min(d.config.vertical_speed for d in self.participants))
                if self.cargo_position[2]<=self.destination[2]:self.transition('RELEASE')
            # Test the load's lower point throughout the swept segment.
            length=np.linalg.norm(self.cargo_position-previous)
            for p in np.linspace(previous,self.cargo_position,max(2,int(math.ceil(length/(s.terrain.cell/2)))+1)):
                if p[2]<s.terrain.sample(*p[:2])-1e-8:
                    self.cargo_position=previous;s.block('BLOCKED_LOAD_TERRAIN_COLLISION');return
            # Formation members move simultaneously. Sequential collision checks
            # would compare a moving member with a neighbour's old position.
            from .traffic import closest_separation
            targets={d.identifier:self.cargo_position+self.offsets[d.identifier] for d in self.participants}
            for i,d in enumerate(self.participants):
                target=targets[d.identifier]
                if (np.linalg.norm(target[:2]-d.position[:2])>d.config.loaded_speed+1e-7
                        or abs(target[2]-d.position[2])>d.config.vertical_speed+1e-7):
                    s.block('BLOCKED_GROUP_FORMATION_SPEED');return
                for other in self.participants[i+1:]:
                    if closest_separation(d.position,target,other.position,targets[other.identifier])<s.config.fleet.separation-1e-8:
                        s.block('BLOCKED_GROUP_FORMATION');return
                for p in np.linspace(d.position,target,max(2,int(math.ceil(np.linalg.norm(target-d.position)/(s.terrain.cell/2)))+1)):
                    ground=s.terrain.sample(*p[:2])
                    if np.linalg.norm(p[:2]-s.base.position[:2])<=s.base.config.radius:
                        ground=max(ground,s.base.position[2]+s.base.config.roof_height)
                    if p[2]<ground-1e-8:s.block('BLOCKED_GROUP_TERRAIN');return
            for d in self.participants:
                target=targets[d.identifier];d.velocity=target-d.position;d.position[:]=target
            e.position[:]=self.cargo_position
            inv.items['group_rig'].position=tuple(self.cargo_position)
        elif self.phase=='RELEASE':
            for d in self.participants:
                if not s.consume_drone(d.power(),d):return
                d.airborne_seconds+=1.;d.flight_seconds+=1.
            if self.elapsed>=20.:
                inv.move('group_rig',L.WORKSITE_RACK,'excavator',e.position,'STORED_READY','collective_lift')
                inv.release('collective_lift')
                w.rack=e.position.copy();e.relocations+=1;e.state=ES.SWAP_PREPARATION
                e.reason='WAITING_POWER_BATTERY_DELIVERY';e.elapsed=0.;w.progress_at=s.time
                for id in self.removed_ids:
                    b=inv.items[id]
                    if b.soc<.999999:
                        b.state='AWAITING_PICKUP'
                        w.new_job('RECOVER_EMPTY_EXCAVATOR_BATTERY',b,b.position,s.base.position,2)
                    else:w.new_job('DELIVER_EXCAVATOR_BATTERY',b,b.position,e.position,1)
                for d in self.participants:
                    d.cargo=0.;d.equivalent_load=None;d.transition(DS.RETURN);s._reserve_return(d)
                w.lift_count+=1;w.group=None

    def report(self):
        return dict(phase=self.phase,participants=[d.serial for d in self.participants],
                    cargo_position=self.cargo_position.tolist(),destination=self.destination.tolist(),
                    batteries_removed=list(self.removed_ids))
