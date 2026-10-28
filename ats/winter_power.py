"""Winter-checked startup recipes, reusing the existing bus and platform sizing.

An energy duty profile is a design assumption, not a prediction of soil output.
Stored energy can fund a single day without establishing a renewable daily cycle.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import numpy as np
from .solar import sunlight
from .equipment import KWH,BASE_BATTERY_LIMIT,equipment_data
from .power_design import PowerRequest,equipment_copy,calculate_power,sized_configuration,_buffer_test
from .base_layout import design_continuous,configure_layout


def default_request():
    # Full payload is conservative; the ground fraction allows battery service.
    return PowerRequest(drone_utilization=.8,excavator_utilization=.95,
                        payload_fraction=1.,energy_margin=.3)


def winter_day(latitude):
    # Use this simulator's 365-day terrestrial solar model, including its phase.
    # The idealised minimum is about one day before the calendar solstice.
    preferred=355 if latitude>=0 else 172
    return min(range(1,366),key=lambda day:(sunlight(latitude,day,12).irradiance,abs(day-preferred)))


def signature(config,latitude=None):
    # Canonical hardware is independent of a drone's current tool/SOC. Resolving
    # this detached copy must never reset the mounted tool mass in a live run.
    data=equipment_data(equipment_copy(config))
    data.update(latitude=float(config.base.latitude_deg if latitude is None else latitude),
                peak=float(config.base.peak_irradiance),idle=float(config.base.idle_power),
                logistics=asdict(config.logistics) if config.logistics else None)
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()


def audit_day(config,request,day):
    """Actual installed PV/storage, two worst-day cycles from the start hour."""
    c=equipment_copy(config);c.base.start_day_of_year=day
    reference=calculate_power(c,request)
    hours=c.start_hour+(np.arange(1440)+.5)/60
    watts_per_irradiance=sum(p.quantity*p.area_m2*p.efficiency for p in c.base.equipment.panels)
    generation=np.array([sunlight(c.base.latitude_deg,day,float(h),c.base.peak_irradiance).irradiance
                         for h in hours])*watts_per_irradiance*request.solar_factor/1000
    load=np.full(1440,reference['mean_power_kw'])
    capacity=c.base.battery/KWH
    check=_buffer_test(generation,load,capacity,request.reserve_soc,1/60,
                       capacity*2,capacity*10)
    # The discharge bus must also cover the bounded connected consumer peak.
    peak=reference['connected_peak_kw'];bus=c.logistics.inverter_power_w/1000
    solar=float(generation.sum()/60);demand=float(load.sum()/60)
    repeatable=(check['unserved_kwh']<1e-7 and check['daily_storage_drift_kwh']>=-1e-7
                and solar>=demand-1e-7 and bus>=peak-1e-7)
    one_day=check['first_day_final_kwh']>=capacity*request.reserve_soc-1e-7
    # _buffer_test totals both days, so explicitly measure the first day too.
    stored=capacity;unserved=0.
    for pv,wanted in zip(generation,load):
        balance=float(pv-wanted)/60
        if balance>=0:stored=min(capacity,stored+min(balance,capacity*2/60))
        else:
            supplied=min(-balance,capacity*10/60,max(0.,stored-capacity*request.reserve_soc))
            stored-=supplied;unserved+=-balance-supplied
    one_day=one_day and unserved<1e-7 and bus>=peak-1e-7
    autonomy=capacity*(1-request.reserve_soc)/max(reference['mean_power_kw'],1e-12)
    return dict(day_of_year=day,daylight_hours=sunlight(c.base.latitude_deg,day,12).daylight_hours,
                solar_kwh=solar,demand_kwh=demand,capacity_kwh=capacity,
                one_day_covered=bool(one_day),repeatable=bool(repeatable),
                first_day_final_kwh=float(stored),first_day_unserved_kwh=float(unserved),
                no_sun_base_autonomy_hours=float(autonomy),connected_peak_kw=peak,inverter_kw=bus,
                storage_check=check)


def design_winter(config,request=None):
    """Return new hardware only. Keep the user's season, terrain and live run."""
    request=deepcopy(request or default_request());request.work_hours=24.;request.source='equipment';request.validate()
    original=equipment_copy(config);c=equipment_copy(config)
    day=winter_day(c.base.latitude_deg);c.base.start_day_of_year=day
    reason=None
    try:
        c,result=design_continuous(c,request)
    except ValueError as error:
        reason=str(error)
        # Finite one-day storage is honest even when solar renewal is impossible
        # or the panel/platform limit is exceeded. No new power source is added.
        c=equipment_copy(original);c.base.start_day_of_year=day
        from .base_layout import parameters,plan_heading
        c.base.layout=parameters(c.base.layout);c.base.layout['heading_deg']=plan_heading(original)
        previous=None
        for _ in range(16):
            result=calculate_power(c,request)
            panel=deepcopy(next((p for p in original.base.equipment.panels if p.quantity),None))
            result['panel_count']=panel.quantity if panel else 0
            if panel:result['panel_type']=asdict(panel)
            unit=result['base_battery_type']['capacity_kwh']
            result['base_battery_count']=math.ceil(result['daily_energy_kwh']/((1-request.reserve_soc)*unit))
            if result['base_battery_count']>BASE_BATTERY_LIMIT:raise ValueError('Даже одни сутки требуют больше доступных аккумуляторов базы')
            c=sized_configuration(c,result);layout=configure_layout(c)
            current=(result['base_battery_count'],layout['platform_count'],c.base.idle_power)
            if current==previous:break
            previous=current
        else:raise ValueError('Подбор суточного запаса не сошёлся')
        # Recompute recommendations and mark them separately from installed data.
        result=calculate_power(c,request);result['base_layout']=layout
    c.base.start_day_of_year=original.base.start_day_of_year
    c.base.initial_charge=1.
    for model in c.fleet.models:model.initial_charge=1.
    c.logistics.drone_spare_initial_soc=c.logistics.excavator_initial_soc=1.
    current=audit_day(c,request,c.base.start_day_of_year);winter=audit_day(c,request,day)
    rotation=all(r['max_utilization']+1e-9>=request.drone_utilization for r in result['drones'])
    mode='renewable' if winter['repeatable'] and rotation else 'one_day_storage'
    design=dict(version=1,mode=mode,latitude_deg=c.base.latitude_deg,winter_day_of_year=day,
        calendar_solstice_day=355 if c.base.latitude_deg>=0 else 172,
        request=asdict(request),current=current,winter=winter,rotation_feasible=rotation,
        reason=reason,panels=sum(p.quantity for p in c.base.equipment.panels),panel_area_m2=c.base.panel_area,
        base_battery_count=sum(p.quantity for p in c.base.equipment.batteries),base_capacity_kwh=c.base.battery/KWH,
        platform_count=c.base.layout and configure_layout(c)['platform_count'],
        equipment_mass_kg=result['base_layout']['packed_gross_mass_kg'],
        required_map_side_m=max(64.,2*math.sqrt(c.base.panel_area)+32.,max(result['base_layout']['dimensions_m'])+24.))
    design['signature']=signature(c);c.energy_design=design
    return c,design


def design_status(config,latitude=None):
    design=getattr(config,'energy_design',None)
    if not design:return 'manual','Энергетика: ручная комплектация · проверить сутки и зиму'
    if signature(config,latitude)!=design['signature']:
        return 'stale','Энергетика: условия или оборудование изменились · требуется перепроверка'
    if design['mode']=='renewable':
        return 'ready',f"24 ч + зимний минимум проверены · {design['panel_area_m2']:,.0f} м² · {design['base_capacity_kwh']:,.0f} кВт·ч"
    hours=design['winter']['no_sun_base_autonomy_hours']
    return 'limited',f'Солнце не обеспечивает повторяемый зимний режим · запас базы без солнца {hours:.1f} ч'


def design_report(config,design):
    """Existing power-dialog fields, describing the actual installed recipe."""
    c=equipment_copy(config);c.base.start_day_of_year=design['winter_day_of_year']
    r=calculate_power(c,PowerRequest(**design['request']))
    recommended=r['panel_area_m2'];ratio=c.base.panel_area/recommended if recommended else 0.
    r['hourly_solar_kw']=[power*ratio for power in r['hourly_solar_kw']]
    mobile=sum(d.battery/KWH*(1+c.logistics.drone_spare_sets) for d in c.fleet.models)
    mobile+=c.logistics.excavator_battery_count*c.logistics.pack_capacity_kwh if c.logistics.excavator_enabled else 0.
    buffer=c.logistics.buffer_kwh if c.logistics.excavator_enabled else 0.
    r.update(panel_count=design['panels'],panel_area_m2=c.base.panel_area,
        base_battery_count=design['base_battery_count'],base_capacity_kwh=c.base.battery/KWH,
        initial_stored_kwh=c.base.battery/KWH+mobile+buffer,
        empty_to_full_input_kwh=c.base.battery/KWH+(mobile+buffer/.9)/c.base.charge_efficiency,
        solar_daily_kwh=design['winter']['solar_kwh'],solar_peak_kw=max(r['hourly_solar_kw']),
        storage_check=design['winter']['storage_check'],base_layout=configure_layout(c),winter_design=design)
    if design['mode']!='renewable':r['warnings'].append(design_status(config)[1])
    return r


def ready_simulation(config,request=None,*,allow_resize=True):
    """One new deployment, with a bounded site search before replacing a run."""
    from .maps import with_map_size
    from .simulation import Simulation
    c,design=design_winter(config,request)
    side=c.terrain.n*c.terrain.cell_size
    sizes=([side]+[v for v in (128,256,512,1024) if v>side and v>=design['required_map_side_m']]
           if allow_resize and c.reference_height is None else [side])
    # Skip a known undersized synthetic staging field, preserving explicit DEMs.
    if len(sizes)>1 and sizes[0]<design['required_map_side_m']:sizes=sizes[1:]
    reason=''
    for size in sizes:
        trial=with_map_size(c,size) if size!=side else equipment_copy(c)
        sim=Simulation(trial)
        if sim.state.value!='BLOCKED':return sim,design
        reason=sim.reason
    raise ValueError(f'Комплектация рассчитана, но не помещается на текущей площадке: {reason}')
