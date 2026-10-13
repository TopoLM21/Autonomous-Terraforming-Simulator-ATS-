import argparse
import json
import sys
from .scenarios import scenario, PRESETS
from .simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description="Autonomous Terraforming Simulator")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", choices=PRESETS, default="demo")
    parser.add_argument("--start-hour", type=float, default=9)
    parser.add_argument("--roughness", type=float, default=0)
    parser.add_argument("--latitude", type=float, default=55, help="Latitude in degrees; positive = north")
    parser.add_argument("--day-of-year", type=int, default=172, help="Start day, 1..365; day 172 = 21 June")
    parser.add_argument("--panel-area", type=float, default=20, help="Horizontal solar panel area in m²")
    parser.add_argument("--base-mass", type=float, default=1000, help="Base mass including panels/battery, kg")
    parser.add_argument("--drones", type=int, default=1, help="Fleet size, 1..6")
    parser.add_argument("--charge-slots", type=int, default=1, help="Charging sockets, 1..6; share total charging power")
    parser.add_argument("--wait-mode", choices=("ground","air"), default="ground")
    parser.add_argument("--charge-threshold", type=float, default=30, help="Lower charging trigger, percent; charging then continues to 100%")
    parser.add_argument("--plan", help="Load a brush plan JSON, including its starting terrain")
    parser.add_argument("--equipment", help="Load an equipment inventory JSON exported from the equipment editor")
    parser.add_argument("--compare-fleet", help="Compare fleet sizes, e.g. 1,2,3, with the same scenario")
    parser.add_argument("--json", dest="json_path", help="Save the headless report")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    config = scenario(args.preset, args.seed)
    config.start_hour = args.start_hour
    config.terrain.roughness = args.roughness
    config.base.latitude_deg = args.latitude
    config.base.start_day_of_year = args.day_of_year
    config.base.panel_area = args.panel_area
    config.base.mass = args.base_mass
    config.fleet.count = args.drones
    config.fleet.charge_slots = args.charge_slots
    config.fleet.wait_mode = args.wait_mode
    config.drone.charge_threshold = args.charge_threshold/100
    try:
        if args.plan:
            from .plan import load_plan
            config = load_plan(args.plan, config)
        if args.equipment:
            from pathlib import Path
            from .equipment import load_equipment_data
            config=load_equipment_data(json.loads(Path(args.equipment).read_text(encoding="utf-8")),config)
        config.validate()
        if args.headless:
            if args.compare_fleet:
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
            return 2 if isinstance(report,dict) and report["state"] == "BLOCKED" else 0
        from .gui import launch
        return launch(config)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
