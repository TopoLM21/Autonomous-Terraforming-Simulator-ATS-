from .config import Config

PRESETS = {"demo": (2.0, 1.0, 0.25), "medium": (5.0, 1.5, 0.5), "large": (10.0, 2.0, 1.0)}


def scenario(name="demo", seed=42):
    if name not in PRESETS:
        raise ValueError(f"Неизвестный сценарий: {name}")
    config = Config(seed=seed)
    config.trench.length, config.trench.width, config.trench.depth = PRESETS[name]
    return config
