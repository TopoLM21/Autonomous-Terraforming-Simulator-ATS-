import argparse
import json
import sys
from .scenarios import scenario, PRESETS
from .simulation import Simulation


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
    parser.add_argument("--charge-threshold", type=float, default=30, help="Lower charging trigger, percent; charging then continues to 100%")
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
    parser.add_argument('--batch-experiments',action='store_true',help='Run S0–S9 with the same terrain/seed')
    args = parser.parse_args()
    mode=args.mode or 'logistics'
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
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
        if not args.plan and not args.configuration:
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
        if mode!='legacy' and (not args.configuration or config.logistics is None or args.mode is not None):
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
        if args.save_configuration:
            from .experiments import save_configuration
            save_configuration(args.save_configuration,config)
        config.validate()
        if args.headless:
            if args.batch_experiments:
                from .experiments import run_benchmarks
                report=run_benchmarks(config,args.steps,args.json_path)
            elif args.compare_fleet:
                from .comparison import compare_fleets
                report = compare_fleets(config,[int(n) for n in args.compare_fleet.split(',')],args.steps)
            else:
                sim = Simulation(config)
                report = sim.run(args.steps)
            result = json.dumps(report, ensure_ascii=False, indent=2)
            # Windows redirected consoles can use cp1252. Escape Unicode only
            # when the stream cannot encode it; JSON files always remain UTF-8.
            stdout_result = result
            try:
                result.encode(sys.stdout.encoding or "utf-8")
            except UnicodeEncodeError:
                stdout_result = json.dumps(report, ensure_ascii=True, indent=2)
            print(stdout_result)
            if args.json_path:
                from pathlib import Path
                Path(args.json_path).write_text(result+"\n", encoding="utf-8")
            blocked=isinstance(report,dict) and report['state']=='BLOCKED'
            if args.batch_experiments:blocked=any(row['state']=='BLOCKED' for row in report)
            return 2 if blocked else 0
        from .gui import launch
        return launch(config)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
