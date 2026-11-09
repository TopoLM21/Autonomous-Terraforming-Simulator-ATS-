"""Annual work-calendar sizing using the existing electrical and carrier model."""
from copy import deepcopy
from dataclasses import asdict
import math
from .equipment import BASE_BATTERY_LIMIT,PANEL_QUANTITY_LIMIT,resolve_config
from .power_design import equipment_copy,calculate_power,storage_requirement
from .base_layout import design_layout,configure_layout
from .season_power import daily_input,service_capacity
from .winter_power import default_request,audit_day,day_profile,signature
from .work_schedule import WorkSchedule,day_label


def day_request(request,schedule,latitude,day):
    value=deepcopy(request);interval=schedule.interval(latitude,day)
    if interval is None:raise ValueError(f'{day_label(day)}: световой день короче заданной смены')
    value.work_hours=interval[1]-interval[0];value.start_hour=interval[0]
    value.source='equipment';value.validate();return value


def design_scheduled(config,schedule,request=None,*,cancelled=lambda:False):
    """Cover every selected day twice, not a single day funded by initial SOC."""
    if not isinstance(schedule,WorkSchedule):schedule=WorkSchedule.from_data(schedule)
    schedule.validate();c=equipment_copy(config);original_day=c.base.start_day_of_year
    c.energy_design=None
    c.operating_schedule=schedule.to_data();request=deepcopy(request or default_request())
    request.source='equipment';request.validate();days=schedule.days();summary=schedule.summary(c.base.latitude_deg)
    if summary['insufficient_daylight_days']:
        day=summary['insufficient_daylight_days'][0]
        raise ValueError(f'{day_label(day)}: световой день меньше {schedule.hours:g} ч; уменьшите смену или измените период')
    if any(daily_input(c.base.latitude_deg,d,c.base.peak_irradiance)<=1e-10 for d in days):
        raise ValueError('В периоде есть день без солнечной генерации: повторяемый суточный режим от одних панелей невозможен')
    control=min(days,key=lambda d:(daily_input(c.base.latitude_deg,d,c.base.peak_irradiance),d))
    min_panels=min_batteries=0;audits=[]
    for _ in range(12):
        if cancelled():raise InterruptedError('Расчёт комплектации отменён')
        c.base.start_day_of_year=control
        sizing_request=day_request(request,schedule,c.base.latitude_deg,control)
        c,_=design_layout(c,sizing_request,min_panels=min_panels,min_batteries=min_batteries)
        reference=calculate_power(c,sizing_request);layout=configure_layout(c)
        def ratio(day):
            shift=day_request(request,schedule,c.base.latitude_deg,day).work_hours
            demand=reference['idle_power_kw']*24+(reference['work_power_kw']-reference['idle_power_kw'])*shift
            return demand/daily_input(c.base.latitude_deg,day,c.base.peak_irradiance)
        next_control=max(days,key=lambda d:(ratio(d),-d))
        panels=sum(p.quantity for p in c.base.equipment.panels)
        batteries=sum(b.quantity for b in c.base.equipment.batteries)
        required_panels=panels;required_batteries=batteries;audits=[]
        for day in days:
            if cancelled():raise InterruptedError('Расчёт комплектации отменён')
            daily=day_request(request,schedule,c.base.latitude_deg,day)
            ref=dict(reference)
            ref['mean_power_kw']=ref['idle_power_kw']+(ref['work_power_kw']-ref['idle_power_kw'])*daily.work_hours/24
            profile=day_profile(c,daily,day,ref)
            required=storage_requirement(*profile,daily.reserve_soc)
            required_batteries=max(required_batteries,math.ceil(required/c.base.equipment.batteries[0].capacity_kwh-1e-12))
            audit=audit_day(c,daily,day,reference=ref,layout=layout,profile=profile)
            audit['work_hours']=daily.work_hours;audits.append(audit)
            required_panels=max(required_panels,math.ceil(panels*audit['demand_kwh']*(1+daily.energy_margin)/max(audit['solar_kwh'],1e-20)-1e-12))
        if (next_control==control and required_panels<=panels and required_batteries<=batteries
                and all(a['repeatable'] for a in audits)):
            break
        min_panels=max(panels,required_panels);min_batteries=max(batteries,required_batteries)
        if min_panels>PANEL_QUANTITY_LIMIT or min_batteries>BASE_BATTERY_LIMIT:
            raise ValueError('Выбранный график превышает допустимый размер солнечного комплекса')
        control=next_control
    else:raise ValueError('Календарный подбор не сошёлся: проверьте зарядные мощности и нагрузку')
    charging=service_capacity(c,layout,reference)
    if any(d['max_utilization']+1e-9<request.drone_utilization for d in reference['drones']):
        raise ValueError('Запрошенная доля работы превышает физический оборот батарей дронов')
    if reference['excavator'] is not None and reference['excavator']['max_utilization']+1e-9<request.excavator_utilization:
        raise ValueError('Запрошенная доля работы превышает физический оборот батарей Крота')
    c.base.start_day_of_year=original_day;c.base.initial_charge=1.
    for d in c.fleet.models:d.initial_charge=1.
    c.logistics.drone_spare_initial_soc=c.logistics.excavator_initial_soc=1.
    limiting=next(a for a in audits if a['day_of_year']==control)
    dimensions=layout.get('grid',{}).get('deployed_dimensions_local_m',layout['dimensions_m'])
    design=dict(version=3,kind='calendar',mode='renewable',latitude_deg=c.base.latitude_deg,
        schedule=schedule.to_data(),calendar=summary,request=asdict(request),checked_days=len(days),
        limiting_day_of_year=control,limiting_date=day_label(control),limiting=limiting,
        season_checks=audits,service_charging=charging,panels=panels,panel_area_m2=c.base.panel_area,
        base_battery_count=batteries,base_capacity_kwh=c.base.battery/3600000,
        platform_count=layout['platform_count'],equipment_mass_kg=layout['packed_gross_mass_kg'],
        required_map_side_m=max(64.,2*math.sqrt(c.base.panel_area)+32.,max(dimensions)+24.))
    design['signature']=signature(c);c.energy_design=design
    return c,design


def design_status(config,latitude=None,day_of_year=None):
    design=config.energy_design;schedule=WorkSchedule.from_data(design['schedule'])
    latitude=config.base.latitude_deg if latitude is None else latitude
    day=config.base.start_day_of_year if day_of_year is None else day_of_year
    if signature(config,latitude)!=design['signature']:return 'stale','Комплектация или широта изменена — нужен повторный подбор по календарю'
    if not schedule.contains(day):return 'outside','Вне рабочего периода '+design['calendar']['label']
    if schedule.interval(latitude,day) is None:return 'outside','Световой день короче заданной смены'
    hours='24 ч/сут' if schedule.mode=='continuous' else 'световой день' if schedule.mode=='daylight' else f'{schedule.hours:g} ч/сут без ночи'
    return 'ready',f"{design['calendar']['label']} · {hours} · {latitude:+.3f}° · контроль {design['limiting_date']}"
