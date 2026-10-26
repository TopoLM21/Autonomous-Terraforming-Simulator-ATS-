import argparse
import json
import sys
from .scenarios import scenario, PRESETS
from .simulation import Simulation


def _write_report(report, path):
    result=json.dumps(report,ensure_ascii=False,indent=2)
    try:result.encode(sys.stdout.encoding or 'utf-8');stdout=result
    except UnicodeEncodeError:stdout=json.dumps(report,ensure_ascii=True,indent=2)
    print(stdout)
    if path:
        from pathlib import Path
        Path(path).write_text(result+'\n',encoding='utf-8')


def _finish_region(sim, region, report, save_path=None):
    if region is None:return
    if save_path:
        w=sim.logistics;e=w.excavator if w else None
        if (any(not d.grounded or d.cargo>0 for d in sim.drones) or w and
                (w.active or w.group is not None or any(c.soil_kg>0 for c in w.containers) or
                 e and (e.bucket_kg>0 or e.active_container is not None))):
            raise ValueError('Ландшафт не сохранён: завершите рейсы/выгрузку; для сохранения работающей техники используйте --save-state')
    region.commit_window(sim.config.georeference['origin_m'],sim.terrain.height,cell_size=sim.terrain.cell,soil=sim.terrain.soil)
    if region.channel_project:
        from .channel import ChannelProject
        project=ChannelProject.from_data(region.channel_project);project.record(report);region.channel_project=project.to_data()
    if save_path:region.save(save_path)
    report['landscape']=region.report()


def _run_project(project,steps,save_landscape=None):
    for _ in range(steps):
        if not project.running:break
        project.step()
    report=project.report()
    report['reports']=[dict(id=u.id,report=u.simulation.report()) for u in project.units]
    if project.landscape is not None:
        for unit in project.units:
            if save_landscape:project.commit(unit)
            else:
                s=unit.simulation
                project.landscape.commit_window(s.config.georeference['origin_m'],s.terrain.height,cell_size=s.terrain.cell,soil=s.terrain.soil)
        if save_landscape:project.landscape.save(save_landscape)
        report['landscape']=project.landscape.report()
    return report


def _design_outputs(args,config,observation=None):
    if not args.power_report and not args.save_complex_template:return
    from .power_design import PowerRequest,calculate_power
    from .complexes import ComplexTemplate
    from pathlib import Path
    request=PowerRequest(work_hours=args.work_hours,start_hour=args.work_start_hour,drone_utilization=args.drone_duty/100,
                         source='measured' if args.measured_power else 'equipment')
    report=calculate_power(config,request,observation)
    if args.power_report:Path(args.power_report).write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    if args.save_complex_template:ComplexTemplate.capture(config,'Рассчитанный комплекс',report).save(args.save_complex_template)


def _resume(args, parser):
    from .checkpoint import load_state,save_state
    allowed={'--load-state','--save-state','--headless','--steps','--json','--panel-area','--base-batteries',
             '--power-report','--save-complex-template','--work-hours','--work-start-hour','--drone-duty','--measured-power'}
    forbidden=[v.split('=')[0] for v in sys.argv[1:] if v.startswith('--') and v.split('=')[0] not in allowed]
    if forbidden:parser.error('Сохранение уже задаёт сценарий; несовместимые параметры: '+', '.join(forbidden))
    session=load_state(args.load_state);sim=session.simulation
    if args.panel_area is not None:sim.set_panel_area(args.panel_area)
    if args.base_batteries is not None:sim.set_base_battery_count(args.base_batteries)
    if not args.headless:
        from .gui import launch
        return launch(sim.config,session=session)
    if session.complex_project is not None:report=_run_project(session.complex_project,args.steps)
    else:report=sim.run(args.steps);_finish_region(sim,session.landscape,report)
    _design_outputs(args,sim.config,sim.report(full=False))
    if args.save_state:save_state(args.save_state,sim,session.landscape,session.ui,session.complex_project)
    _write_report(report,args.json_path)
    return 2 if report['state']=='BLOCKED' else 0


def main():
    parser = argparse.ArgumentParser(description="ATS 2.0 · Autonomous Terraforming Simulator")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", choices=PRESETS, default="demo")
    parser.add_argument("--start-hour", type=float, default=9)
    parser.add_argument("--roughness", type=float, default=0)
    parser.add_argument("--latitude", type=float, default=55, help="Latitude in degrees; positive = north")
    parser.add_argument("--day-of-year", type=int, default=172, help="Start day, 1..365; day 172 = 21 June")
    parser.add_argument("--panel-area", type=float, help="Horizontal solar panel area in m²; default 20, or equipment profile")
    parser.add_argument("--base-mass", type=float, default=1000, help="Base mass including panels/battery, kg")
    parser.add_argument("--drones", type=int, help="Fleet size, 1..6; default 1 legacy / 4 logistics")
    parser.add_argument("--charge-slots", type=int, help="Landing/charging sockets, 1..6; share total charging power")
    parser.add_argument("--wait-mode", choices=("ground","air"), default="ground")
    parser.add_argument("--charge-threshold", type=float, default=30, help="Lower charging trigger, percent; charging then continues to 100%%")
    parser.add_argument("--plan", help="Load a brush plan JSON, including its starting terrain")
    parser.add_argument("--equipment", help="Load an equipment inventory JSON exported from the equipment editor")
    parser.add_argument("--compare-fleet", help="Compare fleet sizes, e.g. 1,2,3, with the same scenario")
    parser.add_argument("--json", dest="json_path", help="Save the headless report")
    parser.add_argument('--no-idle-digging',action='store_true',help='Keep transport drones on container/battery service only')
    # The old engine remains an explicit compatibility switch, outside the GUI.
    parser.add_argument('--mode',choices=('legacy','logistics','hybrid'),metavar='{logistics,hybrid}',
                        help='ATS 2.0 mode; default logistics')
    parser.add_argument('--courier-strategy',choices=('batch','responsive'),help='Batch at 75%% full / Krot below 10%% SOC (default) or immediate transport')
    parser.add_argument('--containers',type=int,default=8)
    parser.add_argument('--battery-modules',type=int,choices=(1,2),help='Explicit drone battery variant; otherwise preserve equipment')
    parser.add_argument('--drone-spare-sets',type=int,
                        help='Initial spare battery sets per drone, 0..64; one set matches its installed modules')
    parser.add_argument('--excavator-spare-batteries',type=int,
                        help='Initial spare Krot battery modules, 0..128; installed modules are additional')
    parser.add_argument('--module-charge-ports',type=int,default=4)
    parser.add_argument('--module-charge-mode',choices=('individual','shared'),help='One physical slot per module (default) or shared FIFO ports')
    parser.add_argument('--map-size',type=float,help='Square field side in metres, 32..256; new scenarios default to 64 m')
    parser.add_argument('--laser-profile',choices=('laser_off','existing_legacy','realistic_reference'))
    parser.add_argument('--configuration',help='Load version 2 experiment JSON')
    parser.add_argument('--save-configuration',help='Save version 2 experiment JSON')
    parser.add_argument('--load-state',help='Resume a complete .atsstate snapshot, including working machinery')
    parser.add_argument('--save-state',help='Save complete state after a headless run; airborne cargo is supported')
    parser.add_argument('--base-batteries',type=int,help='Installed base packs, 0..1000; hot additions to loaded state are empty')
    parser.add_argument('--batch-experiments',action='store_true',help='Run S0–S9 with the same terrain/seed')
    dem_source=parser.add_mutually_exclusive_group()
    dem_source.add_argument('--dem',help='Offline DEM: NPZ, ESRI ASC or optional GeoTIFF')
    dem_source.add_argument('--qattara',action='store_true',help='Bundled real NOAA ETOPO1: Qattara and Mediterranean, no internet')
    dem_source.add_argument('--landscape',help='Resume terrain edits from an .atsregion file (local machinery starts a new run)')
    parser.add_argument('--dem-crs',help='ASC CRS: EPSG:4326 or LOCAL_METRES (or use .json sidecar)')
    parser.add_argument('--dem-datum',choices=('MSL',),help='Confirm that imported ASC/GeoTIFF elevations are metres above mean sea level')
    location=parser.add_mutually_exclusive_group()
    location.add_argument('--work-origin',nargs=2,type=float,metavar=('X','Y'),help='Fine work window southwest origin, metres from DEM southwest node')
    location.add_argument('--work-lon-lat',nargs=2,type=float,metavar=('LON','LAT'),help='Fine window southwest origin in WGS84; snapped to fine cell')
    parser.add_argument('--goal-msl',type=float,help='Absolute local trench bottom in metres MSL; DEM starts with no work if omitted')
    parser.add_argument('--save-landscape',help='Save DEM and sparse edits after headless run; requires unloaded and landed machinery')
    parser.add_argument('--dig-rate',type=float,help='Drone soil intake limit, kg/s; standard loose-sand estimate 6')
    parser.add_argument('--soil-profile',choices=('uniform','qattara'),help='Conditional layers; new --qattara runs default to qattara, old saves retain uniform soil')
    parser.add_argument('--power-report',help='Save a daily solar/storage and spare-pack sizing report JSON')
    parser.add_argument('--save-complex-template',help='Save the calculated hardware recipe as an .atstemplate')
    parser.add_argument('--complex-template',help='Start a fresh complex from an .atstemplate hardware recipe')
    parser.add_argument('--complexes',type=int,default=1,help='Independent complexes on separated sections of one regional channel, 1..8')
    parser.add_argument('--work-hours',type=float,default=24,help='Design assumption: operating hours per solar day')
    parser.add_argument('--work-start-hour',type=float,help='Design assumption: shift start in local solar hours')
    parser.add_argument('--drone-duty',type=float,default=50,help='Design assumption: percent of shift spent flying')
    parser.add_argument('--measured-power',action='store_true',help='Size from average energy use in the simulated run')
    parser.add_argument('--channel',action='store_true',help='Excavate the saved channel project in --landscape')
    parser.add_argument('--channel-route',help='Channel route JSON: points, coordinates=source/metres and optional dimensions')
    parser.add_argument('--channel-section',type=int,default=1,help='Prepared local channel section, starting at 1')
    parser.add_argument('--channel-width',type=float,help='Flat channel bottom width in metres')
    parser.add_argument('--channel-depth',type=float,help='Positive depth below MSL, e.g. 5 means bottom -5 m MSL')
    parser.add_argument('--shore-offset',type=float,help='Dry start offset along alignment from sea coastline, default 10 m')
    args = parser.parse_args()
    if not 1<=args.complexes<=8:parser.error('--complexes: от 1 до 8')
    if args.complex_template and (args.configuration or args.equipment):parser.error('Шаблон комплекса уже задаёт оборудование')
    if (args.channel_width is not None or args.channel_depth is not None or args.shore_offset is not None or args.channel_section!=1) and not (args.channel or args.channel_route):
        parser.error('Параметры канала требуют --channel или --channel-route')
    mode=args.mode or 'logistics'
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.save_state and (not args.headless or args.batch_experiments or args.compare_fleet):
        parser.error('--save-state требует один headless-прогон; в интерфейсе используйте меню «Файл»')
    if args.load_state:
        try:return _resume(args,parser)
        except (ValueError,OSError,KeyError,TypeError) as error:parser.error(str(error))
    config = scenario(args.preset, args.seed)
    config.start_hour = args.start_hour
    config.terrain.roughness = args.roughness
    config.base.latitude_deg = args.latitude
    config.base.start_day_of_year = args.day_of_year
    # Resolve the original hardware before resizing panels; extra area adds mass.
    config.base.panel_area = 20
    config.base.mass = args.base_mass
    config.fleet.count = args.drones if args.drones is not None else (1 if mode=='legacy' else 4)
    config.fleet.charge_slots = 1 if args.charge_slots is None else args.charge_slots
    config.fleet.wait_mode = args.wait_mode
    config.drone.charge_threshold = args.charge_threshold/100
    try:
        from .maps import with_map_size
        if args.complex_template:
            from .complexes import ComplexTemplate
            config=ComplexTemplate.load(args.complex_template).config
            config=with_map_size(config,args.map_size or config.terrain.n*config.terrain.cell_size)
            if args.mode is None:mode='legacy' if config.logistics is None else 'logistics'
        if not args.plan and not args.configuration and not args.complex_template:
            config=with_map_size(config,args.map_size if args.map_size is not None else 64.)
        if args.plan:
            from .plan import load_plan
            config = load_plan(args.plan, config)
        if args.equipment:
            from pathlib import Path
            from .equipment import load_equipment_data
            config=load_equipment_data(json.loads(Path(args.equipment).read_text(encoding="utf-8")),config)
        if args.configuration:
            from .experiments import load_configuration
            config=load_configuration(args.configuration)
        if mode!='legacy' and (not (args.configuration or args.complex_template) or config.logistics is None or args.mode is not None):
            from .experiments import experiment_config
            profile=args.laser_profile or ('existing_legacy' if config.base.equipment and any(e.enabled for e in config.base.equipment.emitters) else 'laser_off')
            landing=args.charge_slots if args.charge_slots is not None else (config.base.equipment.landing_slots if args.equipment else None)
            config=experiment_config(config,mode=mode,drones=config.fleet.count,containers=args.containers,
                ports=args.module_charge_ports,laser=profile,
                battery_modules=args.battery_modules,landing_slots=landing)
        elif mode=='legacy':
            config.logistics=None
        if args.laser_profile and mode=='legacy':
            from .laser_profiles import with_laser_profile
            config=with_laser_profile(config,args.laser_profile)
        if args.panel_area is not None:
            from .equipment import resolve_config,set_panel_area
            resolve_config(config);set_panel_area(config.base.equipment,args.panel_area,config.base.panel_efficiency)
            resolve_config(config)
        if args.map_size is not None:config=with_map_size(config,args.map_size)
        if args.module_charge_mode is not None:
            if config.logistics is None:raise ValueError('Module charging mode requires logistics configuration')
            config.logistics.individual_chargers=args.module_charge_mode=='individual'
        # Apply explicit inventory overrides after loading. Omitting these flags
        # preserves the saved physical inventory rather than replacing defaults.
        for flag,maximum in [('drone_spare_sets',64),('excavator_spare_batteries',128)]:
            value=getattr(args,flag)
            if value is not None:
                if not 0<=value<=maximum:raise ValueError(f'--{flag.replace("_","-")}: 0..{maximum} required')
                if config.logistics is None:raise ValueError('Spare battery inventory requires --mode logistics or a logistics configuration')
        if args.drone_spare_sets is not None:config.logistics.drone_spare_sets=args.drone_spare_sets
        if args.excavator_spare_batteries is not None:
            config.logistics.excavator_battery_count=config.logistics.excavator_battery_slots+args.excavator_spare_batteries
        if args.no_idle_digging and config.logistics is not None:config.logistics.idle_digging=False
        if args.courier_strategy is not None:
            if config.logistics is None:raise ValueError('Courier strategy requires ATS 2.0')
            config.logistics.batch_courier=args.courier_strategy=='batch'
        region=None
        if args.soil_profile is not None:
            from .soil import SoilProfile
            config.soil_profile=SoilProfile.qattara() if args.soil_profile=='qattara' else None
            config.initial_soil=None
        if args.dem or args.qattara or args.landscape:
            if args.plan or args.configuration or args.compare_fleet or args.batch_experiments:
                raise ValueError('DEM: начните отдельный локальный запуск без --plan/--configuration/сравнения')
            from .landscape import Landscape,load_dem,qattara_dem,workspace_config
            region=(Landscape.load(args.landscape) if args.landscape else
                    Landscape(qattara_dem() if args.qattara else load_dem(args.dem,crs=args.dem_crs,vertical_datum=args.dem_datum)))
            if args.soil_profile is not None:
                if region.tiles or region.soil_tiles:raise ValueError('Профиль грунта выбирается до земляных работ; существующий ландшафт сохранит свой состав')
                region.soil_profile=config.soil_profile
            elif args.qattara:
                from .soil import SoilProfile
                region.soil_profile=SoilProfile.qattara()
            if args.channel or args.channel_route:
                if args.work_origin or args.work_lon_lat or args.goal_msl is not None:
                    raise ValueError('Для проекта используйте --channel-section/--channel-depth вместо --work-origin/--goal-msl')
                from .channel import ChannelProject
                if args.channel_route:
                    from pathlib import Path
                    route=json.loads(Path(args.channel_route).read_text(encoding='utf-8'))
                    points=route['points']
                    if route.get('coordinates','source')=='source':
                        points=[[float(v) for v in region.raster.to_metres(*p)] for p in points]
                    elif route.get('coordinates')!='metres':raise ValueError('Channel coordinates: source or metres required')
                    parameters={key:route[key] for key in ('width_m','depth_m','shore_offset_m','angle_deg','side_m','spoil_height_m') if key in route}
                elif region.channel_project:
                    old=ChannelProject.from_data(region.channel_project);points=old.raw_points
                    parameters={key:getattr(old,key) for key in ('width_m','depth_m','shore_offset_m','angle_deg','side_m','spoil_height_m')}
                else:raise ValueError('Сначала сохраните размеченный канал или укажите --channel-route')
                for key,value in [('width_m',args.channel_width),('depth_m',args.channel_depth),('shore_offset_m',args.shore_offset),('side_m',args.map_size)]:
                    if value is not None:parameters[key]=value
                project=ChannelProject.create(region.raster,points,**parameters)
                if region.channel_project:
                    old=ChannelProject.from_data(region.channel_project)
                    if old.identity==project.identity:project=old
                project.active_section=args.channel_section-1;project.validate()
                config=project.configuration(region,config);region.channel_project=project.to_data()
            else:
                origin=args.work_origin
                if args.work_lon_lat:
                    if region.raster.metadata['crs']!='EPSG:4326':raise ValueError('--work-lon-lat requires a WGS84 DEM')
                    origin=region.raster.to_metres(*args.work_lon_lat)
                if origin is None:origin=[v/2 for v in region.raster.extent_m]
                origin=[round(v/region.cell_size)*region.cell_size for v in origin]
                config=workspace_config(region,config,origin,args.map_size or 64.)
                if args.goal_msl is not None:
                    from .plan import PlanConfig,PlanSegment
                    q=config.trench
                    config.plan=PlanConfig([PlanSegment([[q.x-q.length/2,q.y],[q.x+q.length/2,q.y]],'dig',q.width,args.goal_msl,'Канава MSL')])
        elif args.work_origin or args.work_lon_lat or args.goal_msl is not None or args.save_landscape or args.channel or args.channel_route:
            raise ValueError('Нужен --dem, --qattara или --landscape')
        if args.dig_rate is not None:
            config.drone.dig_rate_kg_s=args.dig_rate
            for model in config.fleet.models:model.dig_rate_kg_s=args.dig_rate
        if args.base_batteries is not None:
            from .equipment import set_base_battery_count,resolve_config
            set_base_battery_count(config.base.equipment,args.base_batteries);resolve_config(config)
        if args.save_configuration:
            from .experiments import save_configuration
            save_configuration(args.save_configuration,config)
        config.validate()
        if args.complexes>1 and (region is None or region.channel_project is None):raise ValueError('Для --complexes нужен общий региональный проект канала')
        if (args.power_report or args.save_complex_template or args.complexes>1) and (args.batch_experiments or args.compare_fleet):
            raise ValueError('Расчёт энергоснабжения и клонирование требуют одного проекта')
        if not args.headless:
            _design_outputs(args,config)
        if args.headless:
            if args.batch_experiments:
                from .experiments import run_benchmarks
                report=run_benchmarks(config,args.steps,args.json_path)
            elif args.compare_fleet:
                from .comparison import compare_fleets
                report = compare_fleets(config,[int(n) for n in args.compare_fleet.split(',')],args.steps)
            else:
                sim = Simulation(config)
                project=None
                if args.complexes>1:
                    from .complexes import ComplexProject,ComplexTemplate
                    project=ComplexProject(sim,region);template=ComplexTemplate.capture(config,'Комплекс');project.templates.append(template)
                    for _ in range(args.complexes-1):
                        section=project.find_free_section(template)
                        if section is None:raise ValueError('В проекте недостаточно свободных непересекающихся участков')
                        project.add(template,section=section)
                    report=_run_project(project,args.steps,args.save_landscape)
                else:
                    report = sim.run(args.steps)
                    _finish_region(sim,region,report,args.save_landscape)
                _design_outputs(args,config,sim.report(full=False))
                if args.save_state:
                    from .checkpoint import save_state
                    save_state(args.save_state,sim,region,complex_project=project)
            _write_report(report,args.json_path)
            blocked=isinstance(report,dict) and report['state']=='BLOCKED'
            if args.batch_experiments:blocked=any(row['state']=='BLOCKED' for row in report)
            return 2 if blocked else 0
        from .gui import launch
        if args.complexes>1:
            from .complexes import ComplexProject,ComplexTemplate
            from .checkpoint import Session
            sim=Simulation(config);project=ComplexProject(sim,region);template=ComplexTemplate.capture(config,'Комплекс');project.templates.append(template)
            for _ in range(args.complexes-1):
                section=project.find_free_section(template)
                if section is None:raise ValueError('В проекте недостаточно свободных непересекающихся участков')
                project.add(template,section=section)
            return launch(config,session=Session(sim,region,{},complex_project=project))
        return launch(config,landscape=region)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
