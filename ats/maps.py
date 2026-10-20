"""Change field extent without changing metres, cell size or project coordinates."""
from copy import deepcopy
import math


def workspace_cell(side, minimum=.25):
    """Bound active arrays to 1024² cells; leave existing small maps unchanged."""
    cell=minimum
    while side/cell>1024:cell*=2
    return cell


def with_map_size(config, side_m):
    side=float(side_m)
    if not math.isfinite(side) or not 32<=side<=1024:
        raise ValueError('Сторона карты: от 32 до 1024 м')
    cell=workspace_cell(side,config.terrain.cell_size)
    cells=side/cell
    if abs(cells-round(cells))>1e-8 or not 16<=round(cells)<=1024:
        raise ValueError('Размер карты должен быть кратен ячейке; максимум 1024×1024 ячеек')
    result=deepcopy(config)
    if (round(cells)!=config.terrain.n or cell!=config.terrain.cell_size) and (config.initial_height is not None or config.reference_height is not None):
        raise ValueError('Загруженный рельеф фиксирует сетку. Для другого размера создайте новый сценарий.')
    result.terrain.n=round(cells)
    result.terrain.cell_size=cell
    result.terrain.grab_radius=max(result.terrain.grab_radius,cell)
    result.validate()
    return result
