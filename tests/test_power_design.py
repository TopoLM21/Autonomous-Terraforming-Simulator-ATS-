"""Power sizing uses a closed daily ledger and finite charging throughput."""
from copy import deepcopy
import json
import numpy as np
import pytest
from ats.config import Config
from ats.experiments import experiment_config
from ats.power_design import PowerRequest,calculate_power,sized_configuration
from ats.complexes import ComplexTemplate
from ats.simulation import Simulation


def config():
    c=experiment_config(Config());c.base.latitude_deg=31.
    return c


def test_daily_generation_and_two_day_storage_cover_the_requested_profile():
    r=calculate_power(config(),PowerRequest(start_hour=6))
    assert r['solar_daily_kwh']>=r['daily_energy_kwh']*1.2-1e-9
    assert r['panel_count']==int(r['panel_count']) and 0<r['panel_area_m2']<2000
    check=r['storage_check']
    assert check['unserved_kwh']<1e-7 and abs(check['energy_error_kwh'])<1e-7
    assert check['daily_storage_drift_kwh']>=-1e-7
    assert check['minimum_kwh']>=r['base_capacity_kwh']*.2-1e-7
    assert check['initial_kwh']+check['solar_kwh']==pytest.approx(check['final_kwh']+check['demand_kwh']-check['unserved_kwh']+2*check['curtailed_kwh'],abs=1e-7)
    json.dumps(r,allow_nan=False)


def test_night_shift_and_full_day_require_more_storage_than_daylight_work():
    c=config();day=calculate_power(c,PowerRequest(start_hour=6,work_hours=12))
    late=calculate_power(c,PowerRequest(start_hour=9,work_hours=12))
    full=calculate_power(c,PowerRequest(start_hour=0,work_hours=24))
    assert day['daily_energy_kwh']==pytest.approx(late['daily_energy_kwh'])
    assert day['base_capacity_kwh']<late['base_capacity_kwh']<full['base_capacity_kwh']
    assert full['daily_energy_kwh']==pytest.approx(2*day['daily_energy_kwh']-24*day['idle_power_kw'])


def test_weather_factor_and_winter_are_used_in_panel_count():
    c=config();sun=calculate_power(c,PowerRequest(solar_factor=.8))
    cloud=calculate_power(c,PowerRequest(solar_factor=.4))
    assert cloud['panel_area_required_m2']==pytest.approx(2*sun['panel_area_required_m2'])
    c.base.latitude_deg=55;c.base.start_day_of_year=355
    winter=calculate_power(c)
    assert winter['panel_area_required_m2']>sun['panel_area_required_m2']


def test_polar_night_has_no_solar_solution_and_cannot_make_a_sized_template():
    c=config();c.base.latitude_deg=90;c.base.start_day_of_year=355
    r=calculate_power(c)
    assert r['panel_count'] is None and r['solar_daily_kwh']==0
    with pytest.raises(ValueError,match='нет солнечного решения'):sized_configuration(c,r)
    json.dumps(r,allow_nan=False)


def test_slow_charge_requires_more_spares_and_disabled_charge_is_explicit():
    c=config();normal=calculate_power(c)
    c.logistics.port_power_w=500;slow=calculate_power(c)
    assert slow['drone_spare_sets']>normal['drone_spare_sets']
    for row in slow['drones']:
        interval=row['active_minutes']/.5
        assert row['spare_sets']*interval>=row['recharge_minutes']-1e-9
    c.logistics.port_power_w=0;disabled=calculate_power(c)
    assert any(row['spare_sets'] is None for row in disabled['drones'])
    with pytest.raises(ValueError,match='Включите зарядные'):sized_configuration(c,disabled)
    json.dumps(disabled,allow_nan=False)


def test_unequal_drone_models_allocate_the_uniform_physical_spare_recipe():
    c=config();c.fleet.models[0].equipment.batteries[0].capacity_kwh*=2
    r=calculate_power(c);s=sized_configuration(c,r)
    expected=sum(sum(p.quantity for p in d.equipment.batteries) for d in s.fleet.models)*s.logistics.drone_spare_sets
    assert r['drone_spare_modules']==expected
    assert r['module_charging_slots']>=expected


def test_zero_work_does_not_require_panels_or_buffers():
    c=config();c.logistics.excavator_enabled=False
    r=calculate_power(c,PowerRequest(drone_utilization=0,auxiliary_kw=0))
    assert r['daily_energy_kwh']==0 and r['panel_count']==0 and r['base_battery_count']==0
    assert r['drone_spare_sets']==0


def test_two_days_from_a_full_battery_do_not_prove_repeatable_supply():
    from ats.power_design import _buffer_test
    # Ample daily solar energy cannot replenish storage through a disabled
    # charge controller. Initial stored energy hides that for several days.
    r=_buffer_test([10.,0.],[1.,1.],100.,0.,1.,charge_limit_kw=0.)
    assert r['unserved_kwh']==0 and abs(r['energy_error_kwh'])<1e-9
    assert r['daily_storage_drift_kwh']==pytest.approx(-1.)


def test_shared_laser_power_is_counted_once_and_cannot_create_free_energy():
    from ats.laser_profiles import with_laser_profile
    c=with_laser_profile(config(),'realistic_reference')
    off=calculate_power(c,PowerRequest(laser_coverage=0))
    on=calculate_power(c,PowerRequest(laser_coverage=1))
    assert on['laser_input_kw']<=5*.5+1e-9
    assert on['daily_energy_kwh']>off['daily_energy_kwh']
    assert on['drone_charging_kw']<off['drone_charging_kw']
    assert abs(on['storage_check']['energy_error_kwh'])<1e-7


def test_auxiliary_load_raises_connected_peak_in_addition_to_charging_power():
    c=config();normal=calculate_power(c,PowerRequest(auxiliary_kw=0,energy_margin=0))
    extra=calculate_power(c,PowerRequest(auxiliary_kw=200,energy_margin=0))
    assert extra['daily_energy_kwh']-normal['daily_energy_kwh']==pytest.approx(200*24)
    assert extra['connected_peak_kw']-normal['connected_peak_kw']==pytest.approx(200)
    assert extra['inverter_required_kw']>=normal['connected_peak_kw']+200-1e-9


def test_actual_measurement_is_read_only_and_requires_a_nonempty_sample():
    s=Simulation(config());before=s.report()
    with pytest.raises(ValueError,match='60 секунд'):calculate_power(s.config,PowerRequest(source='measured'),before)
    s.run(120);before=s.report();r=calculate_power(s.config,PowerRequest(source='measured'),s.report(full=False))
    assert r['source']=='measured' and r['daily_energy_kwh']>0
    assert s.report()==before


def test_measured_profile_counts_contact_and_laser_losses_once():
    c=config();eff=c.base.charge_efficiency
    # A one-hour ledger: 10 drone kWh (including 2 received from laser),
    # 4 excavator kWh (including .1 buffer loss), 3 station kWh and 5 laser loss.
    observation=dict(time_seconds=3600,consumed_kwh=17.,drones=[dict(consumed_kwh=10.)],
        energy_excavator_kwh=4.,excavator=dict(buffer_charge_losses_kwh=.1),
        laser_received_kwh=2.,laser_losses_kwh=5.)
    r=calculate_power(c,PowerRequest(source='measured',auxiliary_kw=0),observation)
    expected=3.+(10.+4.-2.)/eff+2.+5.
    assert r['daily_energy_kwh']==pytest.approx(expected*24)


def test_template_mass_matches_every_physical_item_and_vehicle_frame():
    c=config();s=Simulation(c);template=ComplexTemplate.capture(c)
    mass=sum(o.mass_kg for o in s.logistics.inventory.items.values())
    mass+=sum(d.config.equipment.frame_mass_kg for d in s.drones)+c.logistics.excavator_dry_mass_kg
    assert template.equipment_mass()==pytest.approx(mass)


def test_calculated_template_has_the_reported_energy_slots_and_physical_equipment():
    c=config();r=calculate_power(c,PowerRequest(start_hour=6,work_hours=12))
    template=ComplexTemplate.capture(c,power_result=r);s=Simulation(template.config)
    assert s.state.value=='RUNNING',s.reason
    assert s.report()['initial_energy_kwh']==pytest.approx(r['initial_stored_kwh'])
    assert s.logistics.battery_service.capacity==r['module_charging_slots']
    assert r['empty_to_full_input_kwh']>r['initial_stored_kwh']
    for row in r['drones']:
        assert row['empty_to_full_input_kwh']>=row['recharge_input_kwh']>row['usable_kwh']


def test_calculated_shared_mode_template_still_sizes_actual_individual_controllers():
    c=config();c.logistics.individual_chargers=False;c.base.equipment.charge_power_kw=100
    r=calculate_power(c);template=ComplexTemplate.capture(c,power_result=r)
    assert template.config.logistics.individual_chargers
    assert r['drone_spare_sets']>=5


@pytest.mark.parametrize('changes',[dict(work_hours=0),dict(start_hour=24),dict(drone_utilization=2),dict(solar_factor=0),dict(reserve_soc=1),dict(auxiliary_kw=-1),dict(energy_margin=float('nan'))])
def test_invalid_operating_assumptions_are_rejected(changes):
    with pytest.raises(ValueError):calculate_power(config(),PowerRequest(**changes))
