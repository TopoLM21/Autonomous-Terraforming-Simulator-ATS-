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

    def report(self):
        return {"serial":self.drone.serial,"position_number":self.drone.identifier+1,
                "status":self.status,"code":self.code,"limits":self.limits,
                "input_power_w":self.input_w,"received_power_w":self.received_w,
                "receiver_limit_w":self.limit_w,"distance_m":self.distance,
                "beam_radius_m":self.beam_radius,"capture_fraction":self.capture,
                "total_efficiency":self.efficiency,"source":self.source.tolist(),"target":self.target.tolist()}


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


def prepare_links(base,drones,terrain,dt,flight_power,body_radius=.5):
    emitters={e.drone_number:e for e in base.config.equipment.emitters}
    source=base.position+np.array([0,0,base.config.roof_height+.5]);links=[]
    for d in drones:
        link=LaserLink(d,source.copy(),d.position.copy());links.append(link)
        d.laser_power_w=0.;d.last_flight_power_w=0.
        emitter=emitters.get(d.identifier+1);receiver=d.config.equipment.receiver
        if emitter is None:continue
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
        if not visible(source,link.target,terrain,[o for o in drones if o is not d],body_radius,base):
            link.code,link.status="occluded","Луч перекрыт рельефом, базой или дроном";continue
        link.beam_radius=emitter.beam_radius_m+emitter.divergence_mrad*link.distance/1000
        link.capture=min(1.,receiver.area_m2/(math.pi*link.beam_radius**2))
        link.efficiency=emitter.efficiency*link.capture*receiver.efficiency
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
            reasons={"receiver":"предел приёмника","battery":"заряд и потребление дрона","beam_area":"площадь светового пятна","base_energy":"энергия базы"}
            if link.limits:link.status+=" · "+", ".join(reasons[v] for v in link.limits)
        delivered=link.received_w*dt
        link.drone.laser_step_energy+=delivered;link.drone.laser_received+=delivered
        link.drone.laser_power_w=link.received_w;link.drone.laser_status=link.status
        input_energy+=link.input_w*dt;received_energy+=delivered
    return input_energy,received_energy
