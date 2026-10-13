"""Reference figures: receiver 2.8 kg, 800 W peak/~600 W average (ARC).

Every transmitter, attenuation and thermal parameter below is a model assumption.
Legacy hardware values are preserved exactly when selecting existing_legacy.
"""
from copy import deepcopy
from dataclasses import asdict
from .equipment import LaserEmitter, LaserReceiver, resolve_config


def with_laser_profile(config, profile):
    c=deepcopy(config);resolve_config(c)
    if (c.legacy_laser_equipment is None or profile=='realistic_reference') and not any(e.profile=='realistic_reference' for e in c.base.equipment.emitters):
        c.legacy_laser_equipment={'emitters':[asdict(e) for e in c.base.equipment.emitters],
                                'receivers':[asdict(d.equipment.receiver) for d in c.fleet.models]}
    if profile=='existing_legacy':
        if any(e.profile=='realistic_reference' for e in c.base.equipment.emitters):
            c.base.equipment.emitters=[]
            for d in c.fleet.models:d.equipment.receiver=LaserReceiver(installed=True)
        if c.legacy_laser_equipment and c.legacy_laser_equipment['emitters']:
            c.base.equipment.emitters=[LaserEmitter(**e) for e in c.legacy_laser_equipment['emitters']
                                      if e['drone_number']<=c.fleet.count]
            receivers=c.legacy_laser_equipment['receivers']
            for i,d in enumerate(c.fleet.models):d.equipment.receiver=LaserReceiver(**receivers[i%len(receivers)])
        if not c.base.equipment.emitters:
            c.base.equipment.emitters=[LaserEmitter(drone_number=i+1) for i in range(c.fleet.count)]
            for d in c.fleet.models:d.equipment.receiver.installed=True
    elif profile=='realistic_reference':
        c.base.equipment.emitters=[LaserEmitter(drone_number=i+1,mass_kg=300/c.fleet.count,
            input_power_kw=5,efficiency=.4,range_m=300,profile=profile,reference_capture=.75,
            attenuation_per_m=.0001,dust_attenuation=.005,divergence_mrad=.3,shared_source='reference_station',
            shared_limit_kw=5.) for i in range(c.fleet.count)]
        # A shared 300 kg ground installation with per-drone channels. MVP shares
        # one 5 kW source; prepare_links distributes that input between channels.
        for d in c.fleet.models:
            d.equipment.receiver=LaserReceiver(installed=True,mass_kg=2.8,area_m2=.25,
                max_power_kw=.8,efficiency=.4,optical_cap_kw=1.5)
    elif profile=='laser_off':
        for e in c.base.equipment.emitters:e.enabled=False
    else:raise ValueError('Unknown laser profile')
    if c.logistics is not None:c.logistics.laser_profile=profile
    resolve_config(c);c.validate();return c
