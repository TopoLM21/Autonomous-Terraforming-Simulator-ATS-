"""Independent reproducible scenarios for fleet-size comparisons."""
from copy import deepcopy
from .simulation import Simulation


def comparison_config(config,count):
    clone = deepcopy(config)
    models = config.drone_models()
    clone.fleet.count = count
    clone.fleet.models = [deepcopy(models[i % len(models)]) for i in range(count)]
    clone.validate()
    return clone


def compare_fleets(config,counts,steps):
    return [Simulation(comparison_config(config,count)).run(steps) for count in counts]
