"""Tracked optical power links: finite beam footprint, range and visibility.

Uniform circular beam; ideal pointing and receiver normal to the incident ray.
Beam geometry is sampled at the start of each fixed one-second energy step.
"""
from dataclasses import dataclass,field
import math
import numpy as np


@dataclass
class LaserLink:
    drone: object
    source: np.ndarray
    target: np.ndarray
    status: str="Нет передатчика"
    code: str="no_emitter"
    distance: float=0.0
    beam_radius: float=0.0
    capture: float=0.0
    efficiency: float=0.0
    request_w: float=0.0
    input_w: float=0.0
    received_w: float=0.0
    limit_w: float=0.0
    limits: list=field(default_factory=list)
    receiver_heat_w: float=0.
    receiver_efficiency: float=1.
    thermal_capacity_j_k: float=0.
    atmosphere: float=1.
    dust_factor: float=1.
    profile: str='existing_legacy'
    source_efficiency: float=1.
    source_heat_w: float=0.
    path_loss_w: float=0.

    def report(self):
        return {"serial":self.drone.serial,"position_number":self.drone.identifier+1,
                "status":self.status,"code":self.code,"limits":self.limits,
                "input_power_w":self.input_w,"received_power_w":self.received_w,
                "receiver_limit_w":self.limit_w,"distance_m":self.distance,
                "beam_radius_m":self.beam_radius,"capture_fraction":self.capture,
                "total_efficiency":self.efficiency,"source":self.source.tolist(),"target":self.target.tolist(),
                "receiver_heat_w":self.receiver_heat_w,"temperature_c":self.drone.receiver_temperature_c if self.profile=='realistic_reference' else None,
                "source_heat_w":self.source_heat_w,"path_loss_w":self.path_loss_w,
                "atmosphere_factor":self.atmosphere,"dust_factor":self.dust_factor,"profile":self.profile}


def visible(source,target,terrain,others=(),body_radius=.5,base=None):
    difference=target-source;length=float(np.linalg.norm(difference))
    if length<=1e-12:return True
    # At most half a cell between samples, including both end cells.
    count=max(2,int(math.ceil(length/(terrain.cell/2)))+1)
    points=source+np.linspace(0,1,count)[:,None]*difference
    ix=np.clip((points[:,0]/terrain.cell).astype(int),0,terrain.height.shape[1]-1)
    iy=np.clip((points[:,1]/terrain.cell).astype(int),0,terrain.height.shape[0]-1)
    if np.any(points[:,2]<=terrain.height[iy,ix]+1e-6):return False
    if base is not None:
        over_base=np.sum((points[:,:2]-base.position[:2])**2,axis=1)<base.config.radius**2
        if np.any(over_base & (points[:,2]<base.position[2]+base.config.roof_height-1e-6)):return False
    for other in others:
        fraction=float(np.dot(other.position-source,difference)/(length*length))
        if 0<fraction<1 and np.linalg.norm(other.position-(source+fraction*difference))<body_radius:
            return False
    return True


def prepare_links(base,drones,terrain,dt,flight_power,body_radius=.5,dust=None,obstacles=(),forbidden_zones=()):
    emitters={e.drone_number:e for e in base.config.equipment.emitters}
    source=base.position+np.array([0,0,base.config.roof_height+.5]);links=[]
    for d in drones:
        link=LaserLink(d,source.copy(),d.position.copy());links.append(link)
        d.laser_power_w=0.;d.last_flight_power_w=0.
        emitter=emitters.get(d.identifier+1);receiver=d.config.equipment.receiver
        if emitter is not None and emitter.profile=='realistic_reference':
            d.receiver_temperature_c=receiver.ambient_temperature_c+(d.receiver_temperature_c-receiver.ambient_temperature_c)*math.exp(-receiver.cooling_w_k*dt/receiver.thermal_capacity_j_k)
        if emitter is None:continue
        link.source_efficiency=emitter.efficiency
        link.receiver_efficiency=receiver.efficiency
        link.limit_w=receiver.max_power_kw*1000
        if not emitter.enabled or emitter.input_power_kw==0:
            link.code,link.status="disabled","Передатчик выключен";continue
        if not receiver.installed:
            link.code,link.status="no_receiver","Приёмник не установлен";continue
        if d.grounded:
            link.code,link.status="grounded","На земле: зарядка через слот";continue
        if base.destination is not None:
            link.code,link.status="base_moving","База движется: передатчик сложен";continue
        link.distance=float(np.linalg.norm(link.target-source))
        if link.distance>emitter.range_m:
            link.code,link.status="out_of_range","Вне дальности передатчика";continue
        others=[o for o in drones if o is not d]+list(obstacles)
        unsafe=any(np.linalg.norm(np.asarray(center)-source)<radius or
                   np.linalg.norm(np.asarray(center)-(source+np.clip(np.dot(np.asarray(center)-source,link.target-source)/max(link.distance**2,1e-12),0,1)*(link.target-source)))<radius
                   for center,radius in forbidden_zones)
        if unsafe or not visible(source,link.target,terrain,others,body_radius,base):
            link.code,link.status="occluded","Луч перекрыт рельефом, базой или дроном";continue
        link.beam_radius=emitter.beam_radius_m+emitter.divergence_mrad*link.distance/1000
        link.capture=min(1.,receiver.area_m2/(math.pi*link.beam_radius**2))
        link.efficiency=emitter.efficiency*link.capture*receiver.efficiency
        if emitter.profile=='realistic_reference':
            link.profile=emitter.profile
            link.atmosphere=math.exp(-emitter.attenuation_per_m*link.distance)
            if dust is not None and emitter.dust_attenuation:
                samples=max(2,int(math.ceil(link.distance/dust.cell))+1)
                points=np.linspace(source,link.target,samples)
                ix=np.clip((points[:,0]/dust.cell).astype(int),0,dust.n-1)
                iy=np.clip((points[:,1]/dust.cell).astype(int),0,dust.n-1)
                heights=np.array([terrain.sample(*p[:2]) for p in points])
                local=dust.field[iy,ix]*np.exp(-np.maximum(0.,points[:,2]-heights)/dust.config.height_scale)
                link.dust_factor=math.exp(-emitter.dust_attenuation*float(local.mean())*link.distance)
            link.capture*=emitter.reference_capture*receiver.orientation_cosine
            link.efficiency=emitter.efficiency*link.capture*link.atmosphere*link.dust_factor*receiver.efficiency
            optical_limit=receiver.optical_cap_kw*1000 if receiver.optical_cap_kw is not None else math.inf
            heat_budget=max(0.,receiver.max_temperature_c-d.receiver_temperature_c)*receiver.thermal_capacity_j_k/dt
            heat_limit=heat_budget*receiver.efficiency/max(1-receiver.efficiency,1e-12)
            link.limit_w=min(link.limit_w,optical_limit*receiver.efficiency,heat_limit)
            link.receiver_efficiency=receiver.efficiency
            link.thermal_capacity_j_k=receiver.thermal_capacity_j_k
            if heat_limit<min(receiver.max_power_kw*1000,optical_limit*receiver.efficiency):
                link.limits.append('thermal')
        power=flight_power(d);d.last_flight_power_w=power
        room=max(0.,d.config.battery-d.battery)/dt
        desired=min(link.limit_w,power+room)
        offered=emitter.input_power_kw*1000*link.efficiency
        if offered>link.limit_w+1e-9:link.limits.append("receiver")
        if offered>power+room+1e-9:link.limits.append("battery")
        if link.capture<1-1e-12:link.limits.append("beam_area")
        if link.efficiency<=0:
            link.code,link.status="no_capture","Световое пятно не попадает на приёмник";continue
        link.request_w=min(emitter.input_power_kw*1000,desired/link.efficiency)
        link.code,link.status="ready","Готов к передаче"
        if 'thermal' in link.limits and link.request_w<=1e-12:
            link.code,link.status='thermal_throttling','Приёмник перегрет: луч отключён'
    sources={e.shared_source for e in emitters.values() if e.shared_source}
    for source_id in sources:
        members=[link for link in links if emitters.get(link.drone.identifier+1) is not None
                 and emitters[link.drone.identifier+1].shared_source==source_id]
        demand=sum(link.request_w for link in members)
        limit=min(e.shared_limit_kw*1000 for e in emitters.values() if e.shared_source==source_id)
        if demand>limit:
            for link in members:
                link.request_w*=limit/demand;link.limits.append('shared_source')
    return links


def transfer(links,available,dt):
    """Share the remaining base energy proportionally; no duplicate bus power."""
    demand=sum(link.request_w*dt for link in links)
    fraction=min(1.,available/demand) if demand else 0.
    input_energy=received_energy=0.
    for link in links:
        if link.request_w<=0:link.drone.laser_status=link.status;continue
        link.input_w=link.request_w*fraction;link.received_w=link.input_w*link.efficiency
        if link.input_w<=1e-12:
            link.code,link.status="no_base_energy","Нет доступной энергии базы"
        else:
            link.code,link.status="active","Передача энергии"
            if fraction<1-1e-12:link.limits.append("base_energy")
            reasons={"receiver":"предел приёмника","battery":"заряд и потребление дрона","beam_area":"площадь светового пятна","base_energy":"энергия базы","thermal":"тепловое ограничение","shared_source":"общий источник"}
            if link.limits:link.status+=" · "+", ".join(reasons[v] for v in link.limits)
        delivered=link.received_w*dt
        link.receiver_heat_w=link.received_w*(1/link.receiver_efficiency-1)
        link.source_heat_w=link.input_w*(1-link.source_efficiency)
        link.path_loss_w=max(0.,link.input_w*link.source_efficiency-link.received_w-link.receiver_heat_w)
        heat=link.receiver_heat_w*dt;link.drone.receiver_heat_j+=heat
        if link.profile=='realistic_reference':
            link.drone.receiver_temperature_c+=heat/link.thermal_capacity_j_k
        link.drone.laser_step_energy+=delivered;link.drone.laser_received+=delivered
        link.drone.laser_power_w=link.received_w;link.drone.laser_status=link.status
        input_energy+=link.input_w*dt;received_energy+=delivered
    return input_energy,received_energy
