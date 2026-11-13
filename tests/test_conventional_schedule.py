"""Calendar-level regressions independent of ground planning and rendering."""
from collections import defaultdict
from types import SimpleNamespace

import pytest

from ats.conventional import ConventionalConfig, ConventionalFleet


def calendar(config):
    config.validate()
    fleet = ConventionalFleet.__new__(ConventionalFleet)
    fleet.config = config
    fleet.sim = SimpleNamespace(config=SimpleNamespace(start_hour=0.), time=0.)
    return fleet


@pytest.mark.parametrize('shifts,span', [(1, 9.), (2, 9.), (3, 8.)])
@pytest.mark.parametrize('start_hour', [6., 23.5])
def test_week_has_separate_crews_meal_breaks_and_paid_rest(shifts, span, start_hour):
    c = ConventionalConfig(shifts_per_day=shifts, shift_span_hours=span,
                           shift_start_hour=start_hour)
    fleet = calendar(c)
    active_hours = defaultdict(float)
    meal_hours = defaultdict(float)
    consecutive = defaultdict(float)
    max_consecutive = defaultdict(float)
    dt = 900
    start = int(start_hour * 3600)
    # One complete roster week measured from the first crew's shift start.
    for when in range(start, start + 7 * 86400, dt):
        row = fleet.shift(when)
        crew = row['crew']
        if row['active']:
            active_hours[crew] += dt / 3600
            consecutive[crew] += dt / 3600
            max_consecutive[crew] = max(max_consecutive[crew], consecutive[crew])
            assert row['key'][1] == crew - 1
            assert row['seconds_left'] > 0
        else:
            for other in list(consecutive):
                consecutive[other] = 0.
            if crew is not None:
                meal_hours[crew] += dt / 3600
        # Each operator belongs to one crew; other crews reset independently.
        for other in list(consecutive):
            if other != crew:
                consecutive[other] = 0.
    assert set(active_hours) == set(range(1, shifts + 1))
    for crew in range(1, shifts + 1):
        assert active_hours[crew] == pytest.approx(6 * (span - c.lunch_hours))
        assert active_hours[crew] <= 48
        assert meal_hours[crew] == pytest.approx(6 * c.lunch_hours)
        assert max_consecutive[crew] <= 5
    # Default Friday rest is a full 24 h, including for the overnight crew.
    rest_start = start + c.weekly_rest_day * 86400
    assert all(not fleet.shift(t)['active']
               for t in range(rest_start, rest_start + 86400, dt))


def test_lunch_and_shift_end_are_exact_half_open_boundaries():
    fleet = calendar(ConventionalConfig())
    start = 6 * 3600
    lunch = start + 4 * 3600
    end_lunch = lunch + 3600
    end = start + 9 * 3600
    assert not fleet.shift(start - 1)['active']
    assert fleet.shift(start)['active']
    assert fleet.shift(lunch - 1)['seconds_left'] == 1
    assert not fleet.shift(lunch)['active']
    assert not fleet.shift(end_lunch - 1)['active']
    assert fleet.shift(end_lunch)['active']
    assert fleet.shift(end - 1)['seconds_left'] == 1
    assert not fleet.shift(end)['active']


@pytest.mark.parametrize('changes', [
    {'shifts_per_day': 3, 'shift_span_hours': 9.},
    {'shift_span_hours': 10., 'lunch_hours': 1.},
    {'lunch_hours': .5},
    {'lunch_after_hours': 5.5},
    {'lunch_after_hours': 1., 'shift_span_hours': 9.},
])
def test_invalid_rosters_cannot_bypass_hours_and_break_limits(changes):
    with pytest.raises(ValueError):
        ConventionalConfig(**changes).validate()
