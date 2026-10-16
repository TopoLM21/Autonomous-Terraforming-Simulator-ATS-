"""Versioned configurations and reproducible S0–S9 experiments."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
from .config import Config,TrenchConfig,TerrainConfig,DroneConfig,BaseConfig,DustConfig,FleetConfig
from .logistics_config import LogisticsConfig
from .equipment import resolve_config,load_equipment_data,equipment_data
from .laser_profiles import with_laser_profile
from .plan import PlanConfig,PlanSegment


def experiment_config(config, mode='logistics',drones=4,containers=8,ports=4,
                      battery_modules=None,robot_slots=2,panel_area=None,laser='laser_off',landing_slots=None,idle_digging=True,
                      drone_spare_sets=None,excavator_spare_batteries=None):
    c=deepcopy(config);resolve_config(c)
    previous=c.logistics
    if drone_spare_sets is None:
        drone_spare_sets=previous.drone_spare_sets if previous else 1
    if excavator_spare_batteries is None:
        excavator_spare_batteries=(previous.excavator_battery_count-previous.excavator_battery_slots
                                   if previous else max(4,robot_slots*2)-robot_slots)
    for name,value,maximum in [('drone_spare_sets',drone_spare_sets,64),
                               ('excavator_spare_batteries',excavator_spare_batteries,128)]:
        if not isinstance(value,int) or isinstance(value,bool) or not 0<=value<=maximum:
            raise ValueError(f'{name}: integer in 0..{maximum} required')
    models=c.drone_models();c.fleet.count=drones
    c.fleet.models=[deepcopy(models[i%len(models)]) for i in range(drones)]
    for d in c.fleet.models:
        if battery_modules is not None and len(d.equipment.batteries)==1:d.equipment.batteries[0].quantity=battery_modules
    c.base.equipment.landing_slots=min(4,drones) if landing_slots is None else landing_slots
    if panel_area is not None and c.base.equipment.panels:
        old_area=c.base.equipment.panel_area
        for p in c.base.equipment.panels:
            factor=panel_area/old_area if old_area else 1.
            p.area_m2*=factor;p.mass_kg*=factor
    c.logistics=None if mode=='legacy' else LogisticsConfig(direct_allowed=mode=='hybrid',idle_digging=idle_digging,
        container_count=containers,charger_ports=ports,excavator_battery_slots=robot_slots,
        excavator_battery_count=robot_slots+excavator_spare_batteries,
        drone_spare_sets=drone_spare_sets,laser_profile=laser)
    c=with_laser_profile(c,laser);c.validate();return c


def save_configuration(path, config):
    clone=deepcopy(config);resolve_config(clone)
    data={'schema_version':2,'config':asdict(clone),'equipment':equipment_data(clone)}
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def load_configuration(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version')!=2:raise ValueError('Unknown experiment schema version')
    raw=dict(data['config'])
    for key,cls in [('trench',TrenchConfig),('terrain',TerrainConfig),('base',BaseConfig),('drone',DroneConfig),('dust',DustConfig)]:
        values=dict(raw[key])
        if 'equipment' in values:values['equipment']=None
        raw[key]=cls(**values)
    f=dict(raw['fleet']);f['models']=[];raw['fleet']=FleetConfig(**f)
    if raw.get('logistics') is not None:raw['logistics']=LogisticsConfig(**raw['logistics'])
    if raw.get('plan') is not None:raw['plan']=PlanConfig([PlanSegment(**s) for s in raw['plan']['segments']])
    return load_equipment_data(data['equipment'],Config(**raw))


def benchmark_configs(config):
    specs=[('S0',dict(mode='legacy',drones=1)),('S1',dict(mode='legacy',drones=4)),
           ('S2',dict(drones=2)),('S3',dict(drones=4)),('S4',dict(mode='hybrid',drones=4)),
           ('S5',dict(drones=4,laser='existing_legacy')),
           ('S6',dict(drones=4,laser='realistic_reference'))]
    specs += [(f'S7_ports_{n}',dict(drones=4,ports=n)) for n in (1,2,4)]
    specs += [(f'S8_containers_{n}',dict(drones=4,containers=n)) for n in (4,8,12)]
    specs += [(f'S9_panels_{n}',dict(drones=4,panel_area=n)) for n in (20,200)]
    return [(name,experiment_config(config,**values)) for name,values in specs]


def run_benchmarks(config,steps,output=None):
    from .simulation import Simulation
    import time
    results=[]
    cache={}
    for name,c in benchmark_configs(config):
        key=json.dumps(asdict(c),sort_keys=True)
        if key not in cache:
            start=time.perf_counter();s=Simulation(c);report=s.run(steps)
            report['wall_seconds']=time.perf_counter()-start;cache[key]=report
        report=deepcopy(cache[key]);report['experiment']=name
        results.append(report)
        if output:Path(output).write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return results
