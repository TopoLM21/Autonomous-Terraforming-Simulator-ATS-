"""Engineering assumptions for ATS 2.0. Existing presets never enable these."""
from dataclasses import dataclass
import math


@dataclass
class LogisticsConfig:
    version: int = 2
    excavator_enabled: bool = True
    direct_allowed: bool = False
    idle_digging: bool = True  # idle transport drones may fit their stored grabber
    idle_dig_delay_s: float = 30.
    container_count: int = 8
    container_tare_kg: float = 5.
    container_volume_m3: float = .055
    container_gross_kg: float = 90.
    hoist_mass_kg: float = 2.
    excavator_dry_mass_kg: float = 160.
    excavator_battery_slots: int = 2
    excavator_battery_count: int = 4
    pack_capacity_kwh: float = 1.5
    pack_mass_kg: float = 11.
    pack_charge_kw: float = 1.5
    buffer_kwh: float = .1
    excavator_initial_soc: float = 1.
    excavator_reserve: float = .2
    excavator_speed: float = .2
    max_slope_deg: float = 10.
    reach_m: float = 2.
    local_depth_m: float = 1.2
    bucket_volume_m3: float = .015
    bucket_fill_factor: float = .75
    cycle_seconds: float = 35.
    digging_power_w: float = 2500.
    peak_power_w: float = 6000.
    idle_power_w: float = 60.
    sleep_power_w: float = 1.  # model: receiver/controller only, drives powered off
    sleep_soc: float = .1
    moving_power_w: float = 500.
    support_seconds: float = 10.
    digging_force_n: float = 150.
    support_half_width_m: float = .6
    tipping_margin: float = 1.5
    battery_swap_seconds: float = 60.
    drone_swap_seconds: float = 60.
    attachment_seconds: float = 45.
    handling_seconds: float = 5.
    charger_ports: int = 4
    port_power_w: float = 1500.
    bus_power_w: float = 8000.
    inverter_power_w: float = 100000.
    swap_station: bool = True
    drone_spare_sets: int = 1
    drone_pack_discharge_w: float = 60_000.  # model, preserves existing heavy flight power
    drone_pack_charge_w: float = 4000.  # two installed packs preserve the original 8 kW dock input
    drone_spare_initial_soc: float = 1.
    rigging_mass_kg: float = 12.
    cable_angle_deg: float = 30.
    lift_margin: float = 1.2
    air_relocation: bool = True
    laser_profile: str = 'laser_off'
    stall_timeout_s: float = 1800.
    # Large solar field is a separate scenario, not a hidden default bonus.
    field_deployment_seconds: float = 300.
    individual_chargers: bool = True
    charger_slot_mass_kg: float = 1.  # model: socket, wiring and local charge controller
    batch_courier: bool = True
    courier_fill_fraction: float = .75
    courier_low_soc: float = .10

    def validate(self):
        from dataclasses import asdict
        values = asdict(self)
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in values.values()):
            raise ValueError('Non-finite logistics configuration')
        for name in ('container_count','excavator_battery_count','drone_spare_sets'):
            if not isinstance(values[name],int) or values[name] < 0:
                raise ValueError(f'{name}: nonnegative integer required')
        if self.version != 2 or self.excavator_battery_slots not in (2,4):
            raise ValueError('ATS 2 supports two slots or explicit four-slot experiment')
        if self.excavator_battery_count < self.excavator_battery_slots:
            raise ValueError('Fewer physical batteries than active excavator slots')
        if not 1 <= self.charger_ports <= 64:
            raise ValueError('Charging ports: 1..64')
        if not isinstance(self.individual_chargers,bool) or self.charger_slot_mass_kg<=0:
            raise ValueError('Invalid individual charging slot configuration')
        if (not isinstance(self.batch_courier,bool) or not 0<self.courier_fill_fraction<=1
                or not 0<self.courier_low_soc<1):
            raise ValueError('Invalid batch courier thresholds')
        for name in ('pack_capacity_kwh','pack_mass_kg','container_tare_kg','container_volume_m3',
                     'reach_m','local_depth_m','cycle_seconds','excavator_speed','battery_swap_seconds',
                     'drone_swap_seconds','attachment_seconds','handling_seconds','rigging_mass_kg',
                     'idle_dig_delay_s','support_half_width_m','tipping_margin','lift_margin','stall_timeout_s'):
            if values[name] <= 0:
                raise ValueError(f'{name}: positive value required')
        if not 0 < self.bucket_fill_factor <= 1 or not 0 <= self.excavator_initial_soc <= 1:
            raise ValueError('Invalid fill factor/SOC')
        if not 0 <= self.excavator_reserve < 1 or not 0 <= self.cable_angle_deg < 75:
            raise ValueError('Invalid reserve/cable angle')
        if not 0<=self.sleep_soc<=self.excavator_reserve or not 0<=self.sleep_power_w<=self.idle_power_w:
            raise ValueError('Sleep mode must consume nonnegative power below ordinary idle')
        if not 0<=self.drone_spare_initial_soc<=1 or min(self.drone_pack_charge_w,self.drone_pack_discharge_w)<0:
            raise ValueError('Invalid drone battery rating / initial spare SOC')
        if self.container_gross_kg <= self.container_tare_kg:
            raise ValueError('Container gross mass must exceed tare')
        if min(self.pack_charge_kw,self.port_power_w,self.bus_power_w,self.inverter_power_w,
               self.digging_power_w,self.moving_power_w,self.idle_power_w,self.buffer_kwh) < 0:
            raise ValueError('Negative power/energy rating')
        if self.laser_profile not in ('laser_off','existing_legacy','realistic_reference'):
            raise ValueError('Unknown laser profile')
