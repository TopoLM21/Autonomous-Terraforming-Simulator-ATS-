"""Equipment for repeated 24-hour duty throughout a finite calendar season.

Same horizontal clear-sky physics, loading, charge limits and losses as winter
sizing. Stored electricity covers nights, not months of work without sunlight.
"""
from copy import deepcopy
from dataclasses import asdict
import datetime
import math
from .solar import sunlight
from .equipment import BASE_BATTERY_LIMIT,resolve_config
from .power_design import equipment_copy,calculate_power
from .base_layout import design_continuous,configure_layout
from .winter_power import default_request,audit_day,signature,ready_design

QATTARA_LATITUDE=31.078
START_DAY=60  # non-leap 1 March
END_DAY=273  # 30 September, included


def date_label(day):return (datetime.date(2025,1,1)+datetime.timedelta(days=day-1)).strftime('%d.%m')


def daily_input(latitude,day,peak=1000.):
    s=sunlight(latitude,day,12,peak);lat=math.radians(latitude);decl=math.radians(s.declination_deg)
    a=math.sin(lat)*math.sin(decl);b=math.cos(lat)*math.cos(decl)
    return max(0.,peak*(a*s.daylight_hours+24/math.pi*b*math.sin(math.pi*s.daylight_hours/24)))


def service_capacity(config,layout,reference):
    """Mean replenishment must fit real spare sockets and their carrier buses.

    Mounted working packs are excluded; a ready spare cannot charge in flight.
    Queue congestion and physical handling remain mission effects.
    """
    if not layout.get('battery_platforms'):return []
    c=config.logistics;loads={a['id']:dict(platform_id=a['id'],available_kw=0.,required_kw=0.) for a in layout['platforms'] if a.get('role')=='SERVICE'}
    groups=[]
    for i,d in enumerate(config.fleet.models):
        ids=[(key,min(c.port_power_w,c.drone_pack_charge_w)/1000) for key,e in layout['bindings'].items() if e['kind']=='BATTERY' and key.startswith(f'drone_{i+1}_spare_')]
        groups.append((ids,reference['drones'][i]['mean_charging_kw']))
    if c.excavator_enabled:
        ids=[(key,min(c.port_power_w,c.pack_charge_kw*1000)/1000) for key,e in layout['bindings'].items() if e['kind']=='BATTERY' and key.startswith('excavator_battery_') and int(key.rsplit('_',1)[1])>c.excavator_battery_slots]
        groups.append((ids,reference['robot_charging_kw']))
    for ids,required in groups:
        total=sum(power for key,power in ids)
        if required>total+1e-8:raise ValueError('Сезон: мощность зарядных гнёзд сменных комплектов меньше средней нагрузки')
        for key,power in ids:
            row=loads[layout['bindings'][key]['platform_id']];row['available_kw']+=power;row['required_kw']+=required*power/total if total else 0.
    for row in loads.values():
        row['available_kw']=min(row['available_kw'],layout['battery_platforms']['service_bus_kw'])
        if row['required_kw']>row['available_kw']+1e-8:raise ValueError('Сезон: зарядная мощность сервисной платформы недостаточна для оборота батарей')
    return list(loads.values())


def design_season(config,request=None):
    original=equipment_copy(config);c=equipment_copy(config)
    request=deepcopy(request or default_request());request.work_hours=24.;request.source='equipment';request.validate()
    days=range(START_DAY,END_DAY+1)
    control=min(days,key=lambda d:(daily_input(c.base.latitude_deg,d,c.base.peak_irradiance),d))
    night=min(days,key=lambda d:(sunlight(c.base.latitude_deg,d,12).daylight_hours,d))
    if daily_input(c.base.latitude_deg,control,c.base.peak_irradiance)<=1e-9:raise ValueError('В выбранном сезоне есть день без солнца: повторяемый солнечный суточный режим невозможен')
    previous=None
    for _ in range(16):
        c.base.start_day_of_year=control;c,result=design_continuous(c,request)
        # The longest night need not be the day with least daily solar energy.
        c.base.start_day_of_year=night;night_result=calculate_power(c,request)
        count=max(sum(b.quantity for b in c.base.equipment.batteries),night_result['base_battery_count'])
        if count>BASE_BATTERY_LIMIT:raise ValueError('Ночной запас сезона превышает предел аккумуляторов комплекса')
        battery=c.base.equipment.batteries[0];battery.quantity=count;resolve_config(c);layout=configure_layout(c)
        state=(sum(p.quantity for p in c.base.equipment.panels),count,layout['platform_count'],c.base.idle_power)
        if state==previous:break
        previous=state
    else:raise ValueError('Сезонный подбор не сошёлся')
    c.base.start_day_of_year=original.base.start_day_of_year;c.base.initial_charge=1.
    for d in c.fleet.models:d.initial_charge=1.
    c.logistics.drone_spare_initial_soc=c.logistics.excavator_initial_soc=1.
    reference=calculate_power(c,request);layout=configure_layout(c)
    audits=[audit_day(c,request,day,reference=reference,layout=layout) for day in days]
    rotation=all(r['max_utilization']+1e-9>=request.drone_utilization for r in reference['drones'])
    if reference['excavator'] is not None:rotation=rotation and reference['excavator']['max_utilization']+1e-9>=request.excavator_utilization
    charging=service_capacity(c,layout,reference)
    if not rotation or not all(a['repeatable'] for a in audits):raise ValueError('Проверка сезона: ограничение ночного запаса, зарядных мощностей или оборота сменных батарей; сезонный режим не подтверждён')
    limiting=next(a for a in audits if a['day_of_year']==control)
    dimensions=layout.get('grid',{}).get('deployed_dimensions_local_m',layout['dimensions_m'])
    design=dict(version=2,kind='season',mode='renewable',region='Qattara' if abs(c.base.latitude_deg-QATTARA_LATITUDE)<.01 else 'custom',latitude_deg=c.base.latitude_deg,
        season_start_day=START_DAY,season_end_day=END_DAY,season_label='01.03–30.09',checked_days=len(audits),
        limiting_day_of_year=control,limiting_date=date_label(control),longest_night_day=night,
        request=asdict(request),current=audit_day(c,request,c.base.start_day_of_year,reference=reference,layout=layout),
        limiting=limiting,season_checks=audits,rotation_feasible=rotation,service_charging=charging,reason=None,
        panels=sum(p.quantity for p in c.base.equipment.panels),panel_area_m2=c.base.panel_area,
        base_battery_count=sum(b.quantity for b in c.base.equipment.batteries),base_capacity_kwh=c.base.battery/3600000,
        platform_count=layout['platform_count'],equipment_mass_kg=layout['packed_gross_mass_kg'],
        required_map_side_m=max(64.,2*math.sqrt(c.base.panel_area)+32.,max(dimensions)+24.))
    design['signature']=signature(c);c.energy_design=design
    return c,design


def ready_season_simulation(config,request=None,*,allow_resize=True):
    c,design=design_season(config,request)
    return ready_design(c,design,allow_resize=allow_resize)
