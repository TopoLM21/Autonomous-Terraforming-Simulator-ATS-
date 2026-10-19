"""Change field extent without changing metres, cell size or project coordinates."""
from copy import deepcopy
import math


def with_map_size(config, side_m):
    side=float(side_m)
    if not math.isfinite(side) or not 32<=side<=256:
        raise ValueError('Сторона карты: от 32 до 256 м')
    cells=side/config.terrain.cell_size
    if abs(cells-round(cells))>1e-8 or not 16<=round(cells)<=1024:
        raise ValueError('Размер карты должен быть кратен ячейке; максимум 1024×1024 ячеек')
    result=deepcopy(config)
    if round(cells)!=config.terrain.n and config.initial_height is not None:
        raise ValueError('Загруженный рельеф фиксирует сетку. Для другого размера создайте новый сценарий.')
    result.terrain.n=round(cells)
    result.validate()
    return result
