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
    parser.add_argument("--json", dest="json_path", help="Save the headless report")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    config = scenario(args.preset, args.seed)
    config.start_hour = args.start_hour
    config.terrain.roughness = args.roughness
    try:
        config.validate()
        if args.headless:
            sim = Simulation(config)
            report = sim.run(args.steps)
            result = json.dumps(report, ensure_ascii=False, indent=2)
            print(result)
            if args.json_path:
                from pathlib import Path
                Path(args.json_path).write_text(result+"\n", encoding="utf-8")
            return 2 if report["state"] == "BLOCKED" else 0
        from .gui import launch
        return launch(config)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
