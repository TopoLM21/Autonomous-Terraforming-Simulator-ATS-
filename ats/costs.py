"""Editable capital-cost assumptions for owned hardware; no operating ledger.

These starting numbers are model prices in USD, not vendor quotations. Aircraft,
batteries, panel modules, carrier chassis and deployment hardware are separate
lines. Soil, stored electricity and inactive copies of a template are not assets.
"""
from copy import deepcopy
import math


ITEMS={
    'airframe_kg':('Рамы, двигатели и управление дронов','кг',1000.),
    'grabber_kg':('Грейферы дронов','кг',400.),
    'mobile_battery_kwh':('Рабочие и сменные батареи техники','кВт·ч',450.),
    'base_battery_kwh':('Аккумуляторные модули базы','кВт·ч',250.),
    'solar_m2':('Солнечные модули','м²',60.),
    'crawler':('Гусеничные шасси платформ','шт.',75000.),
    'pv_mechanism':('Приводы и опоры самораскрытия PV','компл.',25000.),
    'bess_hardware':('Шкафы и опоры энергетических платформ','компл.',12000.),
    'service_hardware':('Механизмы сервисных платформ','компл.',20000.),
    'charger':('Индивидуальные зарядные контроллеры','шт.',500.),
    'landing_pad':('Посадочные площадки','шт.',1500.),
    'power_bus_kw':('Силовая коммутация и защита комплекса','кВт',150.),
    'excavator':('Кроты без сменных батарей','шт.',70000.),
    'container':('Грунтовые ёмкости','шт.',300.),
    'hook':('Сменные крюки','шт.',500.),
    'hoist':('Грузовые подвесы','шт.',1000.),
    'rig':('Комплект совместного переноса','компл.',1500.),
    'laser_input_kw':('Лазерные передатчики','кВт входа',2500.),
    'laser_receiver_kw':('Приёмники лазерной энергии','кВт выхода',2000.),
}


def default_catalog():
    return dict(version=1,currency='USD',basis='Условные цены модели, 2026; не коммерческое предложение',
                prices={key:value[2] for key,value in ITEMS.items()})


def validate_catalog(value):
    if (not isinstance(value,dict) or set(value)!={'version','currency','basis','prices'}
            or value['version']!=1 or value['currency'] not in ('USD','RUB','EUR','CU')
            or not isinstance(value['basis'],str) or len(value['basis'])>1000
            or not isinstance(value['prices'],dict) or set(value['prices'])!=set(ITEMS)):
        raise ValueError('Неверный каталог ориентировочных цен')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0
           for v in value['prices'].values()):
        raise ValueError('Цены оборудования должны быть конечными и неотрицательными')
    return value


def equipment_quantities(config,layout=None,world=None,stock=()):
    c=config;log=c.logistics;models=c.fleet.models or c.drone_models()
    if c.base.equipment is None or any(d.equipment is None for d in models):
        from .power_design import equipment_copy
        c=equipment_copy(c);log=c.logistics;models=c.fleet.models
    if layout is None and c.base.layout is not None:
        from .base_layout import configure_layout
        from .power_design import equipment_copy
        layout=configure_layout(equipment_copy(c))
    platforms=(layout or {}).get('platforms',())
    qty={key:0. for key in ITEMS}
    qty.update(airframe_kg=sum(d.equipment.frame_mass_kg for d in models),
        grabber_kg=sum(d.equipment.bucket.mass_kg for d in models),
        base_battery_kwh=sum(b.quantity*b.capacity_kwh for b in c.base.equipment.batteries),
        mobile_battery_kwh=sum(sum(b.quantity*b.capacity_kwh for b in d.equipment.batteries) for d in models)*(1+log.drone_spare_sets if log else 1),
        solar_m2=c.base.equipment.panel_area,crawler=len(platforms) if platforms else 1,
        pv_mechanism=sum(a.get('role')=='PV' for a in platforms),
        bess_hardware=sum(a.get('role')=='BESS' for a in platforms),
        service_hardware=sum(a.get('role')=='SERVICE' for a in platforms),
        landing_pad=c.base.equipment.landing_slots,
        power_bus_kw=(log.inverter_power_w/1000 if log else c.base.charge_power/1000),
        laser_input_kw=sum(e.input_power_kw for e in c.base.equipment.emitters),
        laser_receiver_kw=sum(d.equipment.receiver.max_power_kw for d in models if d.equipment.receiver.installed))
    if log:
        qty.update(excavator=log.robot_count,container=log.container_count,hook=len(models),hoist=len(models),rig=1)
        if log.excavator_enabled:qty['mobile_battery_kwh']+=log.excavator_battery_count*log.pack_capacity_kwh*log.robot_count
        modules=sum(sum(b.quantity for b in d.equipment.batteries) for d in models)*(1+log.drone_spare_sets)
        if log.excavator_enabled:modules+=log.excavator_battery_count*log.robot_count
        qty['charger']=modules if log.individual_chargers else log.charger_ports
    for row in stock:
        qty['mobile_battery_kwh']+=row['capacity_kwh']*row['count']
        if log and log.individual_chargers:qty['charger']+=row['count']
    if world is not None:
        from .inventory import BatteryModule,SoilContainer
        items=list(world.inventory.items.values());base=set(world.base_battery_ids)
        qty['base_battery_kwh']=sum(o.capacity_j/3600000 for o in items if isinstance(o,BatteryModule) and o.id in base)
        qty['mobile_battery_kwh']=sum(o.capacity_j/3600000 for o in items if isinstance(o,BatteryModule) and o.id not in base)
        qty['container']=sum(isinstance(o,SoilContainer) for o in items)
        qty['charger']=len(world.battery_service.slots) if log.individual_chargers else log.charger_ports
        qty['grabber_kg']=sum(o.mass_kg for o in items if o.kind=='GRABBER')
        qty['hook']=sum(o.kind=='CONTAINER_HOOK' for o in items)
        qty['hoist']=sum(o.kind=='HOIST' for o in items)
        qty['rig']=sum(o.kind=='HEAVY_LIFT_RIG' for o in items)
    return qty


def quote(config,*,catalog=None,layout=None,world=None,stock=()):
    if catalog is None:catalog=getattr(config,'cost_catalog',None)
    if catalog is None:return None
    validate_catalog(catalog);qty=equipment_quantities(config,layout,world,stock)
    rows=[dict(key=key,name=ITEMS[key][0],unit=ITEMS[key][1],quantity=value,
               unit_price=catalog['prices'][key],amount=value*catalog['prices'][key])
          for key,value in qty.items() if value>0]
    total=sum(row['amount'] for row in rows)
    if not math.isfinite(total):raise ValueError('Ориентировочная стоимость превышает допустимый диапазон')
    return dict(currency=catalog['currency'],basis=catalog['basis'],rows=rows,total=total,
                scope='Стоимость оборудования; эксплуатационные расходы не рассчитаны')


def project_quote(simulations):
    totals={};known=0;count=0
    for s in simulations:
        count+=1;r=quote(s.config,layout=getattr(s.base,'layout_spec',None),world=s.logistics)
        if r is None:continue
        known+=1;totals[r['currency']]=totals.get(r['currency'],0.)+r['total']
    return dict(totals=totals,priced_complexes=known,complexes=count)


def excavation_cost(simulation,estimate=None):
    """Capital equipment value per gross excavated m³, never haul volume.

    The quote is an editable model estimate, not a cash or operating ledger.
    Missing historical prices and zero production remain unknown, not zero.
    """
    s=simulation
    estimate=quote(s.config,layout=getattr(s.base,'layout_spec',None),world=s.logistics) if estimate is None else estimate
    volume=s.excavated_volume if s.terrain.soil is not None else s.excavated_mass/s.config.terrain.density
    return dict(currency=estimate['currency'] if estimate else None,capital_cost=estimate['total'] if estimate else None,
        excavated_m3=volume,capital_per_m3=estimate['total']/volume if estimate and volume>1e-10 else None,
        basis='Стоимость имеющегося оборудования / валовая выемка; без топлива, персонала, ремонта и амортизации')


def project_excavation_cost(simulations):
    sims=list(simulations);estimates=[quote(s.config,layout=getattr(s.base,'layout_spec',None),world=s.logistics) for s in sims]
    currencies={e['currency'] for e in estimates if e};known=bool(sims) and all(estimates) and len(currencies)==1
    volume=sum(s.excavated_volume if s.terrain.soil is not None else s.excavated_mass/s.config.terrain.density for s in sims)
    capital=sum(e['total'] for e in estimates) if known else None
    return dict(currency=next(iter(currencies)) if known else None,capital_cost=capital,excavated_m3=volume,
        capital_per_m3=capital/volume if known and volume>1e-10 else None,
        reason='' if known else 'Не у всех комплексов заданы цены в одной валюте')
