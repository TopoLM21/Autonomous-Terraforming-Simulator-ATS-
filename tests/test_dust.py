import numpy as np
import pytest
from ats.config import Config
from ats.dust import Dust


def test_dust_advects_with_wind_and_decays_without_mass_coupling():
    c = Config()
    c.dust.wind_x = 1
    c.dust.wind_y = 0
    c.dust.diffusion = 0
    d = Dust(c)
    d.field[10, 10] = 1
    d.tick(1, (0, 0, 0), 6, 1, False)
    assert np.unravel_index(np.argmax(d.field), d.field.shape) == (10, 11)
    assert d.field.sum() == pytest.approx(np.exp(-1/c.dust.lifetime))
    assert np.all(d.field >= 0)


def test_emission_responds_to_strength_altitude_and_power():
    c = Config()
    low, high = Dust(c), Dust(c)
    low.tick(1, (16, 16, 0), 3, 2, True)
    high.tick(1, (16, 16, 0), 8, 1, True)
    assert low.field.sum() > high.field.sum() > 0
    c.dust.strength = 0
    silent = Dust(c)
    silent.tick(1, (16, 16, 0), 3, 2, True)
    assert silent.field.sum() == 0
