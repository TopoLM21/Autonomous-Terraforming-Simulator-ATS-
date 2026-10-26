"""Daily energy, pack rotation and solar/storage sizing for a complex template.

This is an explicit duty-cycle design calculation, not a throughput predictor.
The simulator remains the validation of actual routes, queues and weather-free
solar geometry. All energies are kWh and all powers kW in this module.
"""
from copy import copy,deepcopy
from dataclasses import dataclass,asdict
import math
import numpy as np
from .equipment import BatteryModule,PanelModule,KWH,resolve_config
from .solar import sunlight


@dataclass
class PowerRequest:
    work_hours: float = 24.
    start_hour: float | None = None
    drone_utilization: float = .5
    excavator_utilization: float = .8
    payload_fraction: float = .5
    solar_factor: float = .8
    reserve_soc: float = .2
    energy_margin: float = .2
    auxiliary_kw: float = .2
    courier_minutes: float = 5.
    laser_coverage: float = .5
    laser_distance_m: float = 25.
    source: str = 'equipment'

    def validate(self):
        if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Параметры энергоснабжения должны быть конечными')
        if not 0<self.work_hours<=24 or not 0<self.solar_factor<=1 or not 0<=self.reserve_soc<1:
            raise ValueError('Неверная длительность смены, солнечный коэффициент или резерв')
        if self.start_hour is not None and not 0<=self.start_hour<24:raise ValueError('Начало смены: от 0 до 24 часов')
        for value in (self.drone_utilization,self.excavator_utilization,self.payload_fraction,self.laser_coverage):
            if not 0<=value<=1:raise ValueError('Доли загрузки должны быть от 0 до 100%')
        if not 0<=self.energy_margin<=2 or min(self.auxiliary_kw,self.courier_minutes,self.laser_distance_m)<0:
            raise ValueError('Неверный запас или собственное потребление')
        if self.source not in ('equipment','measured'):raise ValueError('Неизвестный профиль нагрузки')


def equipment_copy(config):
    """Copy hardware without duplicating DEM arrays merely to size a power bus."""
    c=copy(config)
    for key in ('base','drone','fleet','logistics'):setattr(c,key,deepcopy(getattr(config,key)))
    resolve_config(c);return c


def _laser_reference(config,request,active_kw):
    """Reference geometry, receiver caps and shared sources, without terrain LOS."""
    models=config.fleet.models;channels=[];groups={}
    for e in config.base.equipment.emitters:
        if not e.enabled or not 1<=e.drone_number<=len(models):continue
        i=e.drone_number-1;r=models[i].equipment.receiver
        if not r.installed or request.laser_distance_m>e.range_m:continue
        group=e.shared_source or f'channel_{i}'
        groups.setdefault(group,[]).append((i,e,r))
    for group in groups.values():
        limit=min((e.shared_limit_kw for _,e,_ in group if e.shared_limit_kw>0),default=math.inf)
        available=min(limit,sum(e.input_power_kw for _,e,_ in group))
        # Equal allocation is conservative for unequal optical channels.
        for i,e,r in group:
            power=min(e.input_power_kw,available/len(group))
            radius=e.beam_radius_m+request.laser_distance_m*e.divergence_mrad/1000
            capture=min(1.,r.area_m2*r.orientation_cosine/(math.pi*radius**2))*e.reference_capture
            efficiency=e.efficiency*capture*math.exp(-e.attenuation_per_m*request.laser_distance_m)*r.efficiency
            limit_kw=r.max_power_kw
            if r.optical_cap_kw is not None:limit_kw=min(limit_kw,r.optical_cap_kw*r.efficiency)
            if r.efficiency<1:
                limit_kw=min(limit_kw,r.cooling_w_k*(r.max_temperature_c-r.ambient_temperature_c)/1000*r.efficiency/(1-r.efficiency))
            received=min(active_kw[i],limit_kw,power*efficiency)
            used=received/efficiency if efficiency>0 else 0.
            channels.append((i,used,received))
    return channels


def _rotation(capacity,low_soc,active_kw,duty,charge_kw,efficiency,lead_hours=0.,swap_seconds=0.):
    usable=capacity*(1-low_soc)
    work=usable/active_kw if active_kw>0 else math.inf
    recharge=usable/(charge_kw*efficiency) if charge_kw>0 else math.inf
    if duty<=0 or active_kw<=0:spare=0
    elif not math.isfinite(recharge):spare=None
    else:spare=max(1,math.ceil((recharge+lead_hours)/(work/duty)-1e-12))
    max_duty=work/(work+swap_seconds/3600) if math.isfinite(work) else 1.
    return dict(usable_kwh=usable,active_minutes=work*60 if math.isfinite(work) else None,
                recharge_minutes=recharge*60 if math.isfinite(recharge) else None,spare_sets=spare,
                recharge_input_kwh=usable/efficiency,empty_to_full_input_kwh=capacity/efficiency,
                total_sets=spare+1 if spare is not None else None,max_utilization=max_duty)


def _buffer_test(generation,load,capacity,reserve,step_hours,charge_limit_kw=math.inf,discharge_limit_kw=math.inf):
    stored=capacity;unserved=0.;curtailed=0.;minimum=stored
    first_day_final=stored
    for day in range(2):
        for solar,demand in zip(generation,load):
            balance=float(solar-demand)
            if balance>=0:
                added=min(balance,charge_limit_kw)*step_hours
                accepted=min(added,capacity-stored);stored+=accepted
                curtailed+=balance*step_hours-accepted
            else:
                requested=-balance*step_hours
                supplied=min(requested,discharge_limit_kw*step_hours,max(0.,stored-capacity*reserve))
                stored-=supplied;unserved+=requested-supplied
            minimum=min(minimum,stored)
        if day==0:first_day_final=stored
    solar=float(np.sum(generation)*step_hours*2);demand=float(np.sum(load)*step_hours*2)
    return dict(unserved_kwh=float(unserved),curtailed_kwh=float(curtailed/2),minimum_kwh=float(minimum),
                initial_kwh=float(capacity),final_kwh=float(stored),solar_kwh=solar,demand_kwh=demand,
                daily_storage_drift_kwh=float(stored-first_day_final),
                energy_error_kwh=float(capacity+solar-stored-(demand-unserved)-curtailed))


def calculate_power(config,request=None,observation=None):
    request=request or PowerRequest();request.validate();c=equipment_copy(config)
    b=c.base;log=c.logistics;eff=b.charge_efficiency
    if not 0<eff<=1:raise ValueError('Нужен положительный КПД зарядки')
    warnings=[];models=c.fleet.models
    panel=deepcopy(next((p for p in b.equipment.panels if p.quantity>0),PanelModule(quantity=1)))
    battery=deepcopy(next((p for p in b.equipment.batteries if p.quantity>0),BatteryModule(quantity=1)))
    active=[]
    for d in models:
        mass=min(d.equipment.payload_limit_kg,d.equipment.bucket.volume_m3*c.terrain.density)*request.payload_fraction
        active.append(d.hover_power*((d.empty_mass+mass)/d.empty_mass)**1.5*d.flight_factor/1000)
    channels=_laser_reference(c,request,active)
    laser_in=[0.]*len(models);laser_out=[0.]*len(models)
    for i,power,received in channels:laser_in[i]+=power*request.laser_coverage;laser_out[i]+=received*request.laser_coverage
    rows=[];drone_input=0.;laser_input=0.
    for i,d in enumerate(models):
        duty=request.drone_utilization
        # Individual module controllers are limited by both the slot and pack.
        packs=sum(m.quantity for m in d.equipment.batteries)
        charge=(sum(m.quantity*min(log.port_power_w,log.drone_pack_charge_w) for m in d.equipment.batteries)/1000
                if log is not None else b.charge_power/1000/max(1,len(models)))
        battery_kw=max(0.,active[i]-laser_out[i]);input_kw=battery_kw*duty/eff
        rotation=_rotation(d.battery/KWH,max(d.reserve,d.charge_threshold),battery_kw,duty,charge,eff,
                           swap_seconds=log.drone_swap_seconds if log else 0.)
        if rotation['spare_sets'] is None:warnings.append(f'{d.name}: зарядная мощность равна нулю')
        if duty>rotation['max_utilization']+1e-9:warnings.append(f'{d.name}: смена модулей ограничивает заданную долю полёта')
        rows.append(dict(number=i+1,name=d.name,modules_per_set=packs,flight_kw=active[i],battery_kw=battery_kw,
                         mean_charging_kw=input_kw,laser_input_kw=laser_in[i]*duty,
                         **rotation))
        drone_input+=input_kw;laser_input+=laser_in[i]*duty
    robot_input=0.;robot_idle=0.;robot=None
    if log is not None and log.excavator_enabled:
        robot_idle=log.idle_power_w/1000/eff
        robot_input=(log.digging_power_w-log.idle_power_w)/1000*request.excavator_utilization/eff
        capacity=log.excavator_battery_slots*log.pack_capacity_kwh
        robot_active=(log.idle_power_w+(log.digging_power_w-log.idle_power_w)*request.excavator_utilization)/1000
        charge=log.excavator_battery_slots*min(log.port_power_w,log.pack_charge_kw*1000)/1000
        low=log.courier_low_soc if log.batch_courier else log.excavator_reserve
        robot=_rotation(capacity,low,robot_active,1.,charge,eff,request.courier_minutes/60,log.battery_swap_seconds)
        robot['modules_per_set']=log.excavator_battery_slots
        robot['recommended_modules']=log.excavator_battery_slots*(robot['total_sets'] or 1)
        if robot['spare_sets'] is None:warnings.append('Крот: зарядная мощность равна нулю')
    idle=b.idle_power/1000+request.auxiliary_kw+robot_idle
    work=drone_input+laser_input+robot_input
    source='equipment'
    if request.source=='measured':
        if observation is None or observation.get('time_seconds',0)<60:raise ValueError('Для профиля по симуляции нужно хотя бы 60 секунд расчёта')
        hours=observation['time_seconds']/3600
        # Consumer energy already includes received laser energy; add its losses
        # once, and estimate contact losses for net battery-supplied consumption.
        consumed=observation['consumed_kwh']
        battery_use=sum(row.get('consumed_kwh',0.) for row in observation.get('drones',[])+observation.get('retired_drones',[]))
        battery_use+=observation.get('energy_excavator_kwh',0.)
        if not observation.get('drones'):battery_use=consumed
        station=max(0.,consumed-battery_use)
        received=observation.get('laser_received_kwh',0.)
        optical_losses=observation.get('laser_losses_kwh',0.)
        # Excavator consumed energy already includes its buffer charge losses.
        net_battery=max(0.,battery_use-received)
        measured=(station+net_battery/eff+received+optical_losses)/hours
        work=max(0.,measured-idle);source='measured'
        warnings.append('Измеренный профиль усреднён с начала прогона; остановки и короткий участок влияют на оценку')
    step=1/60;hours=(np.arange(1440)+.5)/60
    start_hour=c.start_hour if request.start_hour is None else request.start_hour
    working=(hours-start_hour)%24<request.work_hours
    # start_hour is solar local time. No longitude/time-zone correction is used.
    load=np.full(1440,idle)+working*work
    sun=np.array([sunlight(b.latitude_deg,b.start_day_of_year,float(h),b.peak_irradiance).irradiance for h in hours])
    per_area=sun*panel.efficiency*request.solar_factor/1000
    yield_per_m2=float(per_area.sum()*step);demand=float(load.sum()*step)
    area=(demand*(1+request.energy_margin)/yield_per_m2) if yield_per_m2>0 else None
    count=math.ceil(area/panel.area_m2-1e-12) if area is not None else None
    if count is None:
        warnings.append('Солнечная генерация равна нулю: повторяемый суточный режим не обеспечивается')
        generation=np.zeros_like(load);storage=None;storage_count=None;test=None
    else:
        generation=per_area*(count*panel.area_m2)
        deficit=(load-generation)*step;debt=0.;maximum=0.
        for value in np.tile(deficit,2):debt=max(0.,debt+float(value));maximum=max(maximum,debt)
        storage=maximum/(1-request.reserve_soc)
        if log is not None and count*panel.area_m2>20:
            storage=max(storage,200*2*log.field_deployment_seconds/KWH/(1-request.reserve_soc))
        storage_count=max(0,math.ceil(storage/battery.capacity_kwh-1e-12)) if demand>0 else 0
        # Verify capacity and the same 2C/10C limits as physical base modules.
        test=_buffer_test(generation,load,storage_count*battery.capacity_kwh,request.reserve_soc,step,
                          storage_count*battery.capacity_kwh*2,storage_count*battery.capacity_kwh*10)
        while (test['unserved_kwh']>1e-7 or test['daily_storage_drift_kwh']< -1e-7) and storage_count<=1000:
            storage_count+=1
            test=_buffer_test(generation,load,storage_count*battery.capacity_kwh,request.reserve_soc,step,
                              storage_count*battery.capacity_kwh*2,storage_count*battery.capacity_kwh*10)
    max_spare=max((row['spare_sets'] or 0 for row in rows),default=0)
    spare_modules=max_spare*sum(row['modules_per_set'] for row in rows)
    for row in rows:row['allocated_spare_sets']=max_spare
    robot_spares=(robot['spare_sets'] or 0)*robot['modules_per_set'] if robot else 0
    slot_kw=(log.port_power_w/1000 if log else 0.)
    connected=(spare_modules*slot_kw+robot_spares*slot_kw+b.charge_power/1000+
               sum(p for _,p,_ in channels)+b.idle_power/1000+request.auxiliary_kw)
    continuous=(drone_input+robot_input+robot_idle)*(1+request.energy_margin)
    inverter=max(connected,idle+work)*(1+request.energy_margin)
    if log and inverter>log.inverter_power_w/1000:warnings.append('Мощность преобразователя текущей базы меньше расчётной')
    if not log:warnings.append('В прежнем режиме нет сменных модулей: оценка запасов относится к комплексу ATS 2.0')
    elif not log.individual_chargers:warnings.append('Подбор нового комплекса выполняется для индивидуальных зарядных слотов')
    if count is not None and count>10000:warnings.append('Требуется больше 10 000 стандартных панелей; уменьшите нагрузку или измените тип панели')
    if storage_count is not None and storage_count>1000:warnings.append('Требуется больше 1000 аккумуляторов базы')
    if max_spare>64 or robot_spares>128:warnings.append('Запас сменных батарей превышает предел одного комплекса')
    mobile_capacity=sum(d.battery/KWH*(1+max_spare) for d in models)
    buffer_capacity=0.
    if log and log.excavator_enabled:
        mobile_capacity+=(log.excavator_battery_slots+robot_spares)*log.pack_capacity_kwh
        buffer_capacity=log.buffer_kwh
    base_capacity=storage_count*battery.capacity_kwh if storage_count is not None else None
    return dict(request=asdict(request),source=source,start_hour=start_hour,latitude_deg=b.latitude_deg,day_of_year=b.start_day_of_year,
        daily_energy_kwh=demand,mean_power_kw=demand/24,work_power_kw=idle+work,idle_power_kw=idle,
        drone_charging_kw=drone_input,laser_input_kw=laser_input,robot_charging_kw=robot_input+robot_idle,
        solar_kwh_per_m2=yield_per_m2,panel_area_required_m2=area,panel_count=count,
        panel_area_m2=count*panel.area_m2 if count is not None else None,panel_type=asdict(panel),
        solar_peak_kw=float(generation.max()),solar_daily_kwh=float(generation.sum()*step),
        storage_required_kwh=storage,base_battery_count=storage_count,base_battery_type=asdict(battery),
        base_capacity_kwh=base_capacity,
        initial_stored_kwh=base_capacity+mobile_capacity+buffer_capacity if base_capacity is not None else None,
        empty_to_full_input_kwh=base_capacity+(mobile_capacity+buffer_capacity/.9)/eff if base_capacity is not None else None,
        charging_continuous_kw=continuous,connected_peak_kw=connected,inverter_required_kw=inverter,
        drone_spare_sets=max_spare,drone_spare_modules=spare_modules,excavator_spare_modules=robot_spares,
        module_charging_slots=spare_modules+robot_spares+sum(r['modules_per_set'] for r in rows)+(robot['modules_per_set'] if robot else 0),
        drones=rows,excavator=robot,storage_check=test,warnings=warnings,
        hourly_load_kw=[float(load[i:i+60].mean()) for i in range(0,1440,60)],
        hourly_solar_kw=[float(generation[i:i+60].mean()) for i in range(0,1440,60)])


def sized_configuration(config,result):
    """New hardware recipe only; never mutate a live inventory or its energy."""
    if result['panel_count'] is None:raise ValueError('Для выбранного дня нет солнечного решения')
    if config.logistics is None:raise ValueError('Расчётный шаблон со сменными батареями требует режима ATS 2.0')
    if result['panel_count']>10000 or result['base_battery_count']>1000 or result['drone_spare_sets']>64 or result['excavator_spare_modules']>128:
        raise ValueError('Расчёт превышает пределы одного комплекса')
    if any(row['spare_sets'] is None for row in result['drones']):raise ValueError('Включите зарядные устройства модулей')
    c=equipment_copy(config);p=PanelModule(**result['panel_type']);p.quantity=result['panel_count']
    b=BatteryModule(**result['base_battery_type']);b.quantity=result['base_battery_count']
    c.base.equipment.panels=[p] if p.quantity else [];c.base.equipment.batteries=[b] if b.quantity else []
    if c.logistics is not None:
        c.logistics.individual_chargers=True;c.logistics.drone_spare_sets=result['drone_spare_sets']
        c.logistics.excavator_battery_count=c.logistics.excavator_battery_slots+result['excavator_spare_modules']
        c.logistics.inverter_power_w=max(c.logistics.inverter_power_w,result['inverter_required_kw']*1000)
    resolve_config(c);c.validate();return c
