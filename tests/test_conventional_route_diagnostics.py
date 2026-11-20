"""A visible open centreline can conceal a blocked truck footprint."""
import numpy as np
import pytest

from tests.test_conventional import world


def test_route_identifies_another_truck_at_its_destination():
    _, fleet = world(trucks=2)
    truck, other = fleet.machines[1:]
    before = truck.position.copy()
    assert fleet.route(truck, other.position) is None
    assert 'Точка назначения' in fleet._route_failure and other.id in fleet._route_failure
    assert f'({other.position[0]:.1f}, {other.position[1]:.1f})' in fleet._route_failure
    np.testing.assert_array_equal(truck.position, before)


@pytest.mark.parametrize(('height', 'reason'), [(.35, 'ступень'), (1., 'перепад')])
def test_truck_reports_height_under_edge_of_body(height, reason):
    _, fleet = world(trucks=1)
    truck = fleet.machines[1]
    goal = (70., 15.)
    t = fleet.terrain
    assert fleet.passable(goal, truck.radius, truck)
    t.height[int(15./t.cell), int(73./t.cell)] += height
    assert not fleet.passable(goal, truck.radius, truck)
    assert fleet.route(truck, goal) is None
    assert '(70.0, 15.0)' in fleet._route_failure and reason in fleet._route_failure


def test_water_under_standing_truck_and_unreachable_mid_edge_are_reported():
    _, fleet = world(trucks=1)
    truck = fleet.machines[1]
    x, y = truck.position[:2]
    t = fleet.terrain
    water = fleet._marine_mask()
    water[int(y/t.cell), int((x+2.)/t.cell)] = True
    assert fleet.route(truck, (80., 35.)) is None
    assert 'Текущая стоянка' in fleet._route_failure and 'вода' in fleet._route_failure
    water.fill(False)
    water[int(17./t.cell), int(70./t.cell)] = True
    start, end = (60., 15.), (80., 15.)
    assert all(fleet.passable(p, truck.radius, truck) for p in (start, end))
    assert not fleet.segment_safe(start, end, truck.radius, truck)
    details = []
    assert not fleet.segment_safe(start, end, truck.radius, truck, reason=details)
    assert len(details) == 1 and 'вода' in details[0] and 'точке (' in details[0]
