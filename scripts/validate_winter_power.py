"""Bounded winter checks: real one-second SolarBase bus, equivalent input load.

Flight/contact/receiver losses are already included in the designed input load.
This checks the power circuit, not autonomous earthwork throughput for 24 h.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ats.config import Config
from ats.experiments import experiment_config
from ats.solar_base import SolarBase
from ats.solar import sunlight
from ats.equipment import KWH
from ats.winter_power import design_winter


def audit_bus(config,design,days=2):
    c=deepcopy(config.base);c.start_day_of_year=design['winter_day_of_year']
    # Explicit conservative solar derating; hardware area/mass is unchanged.
    c.panel_area*=design['request']['solar_factor']
    c.idle_power=design['winter']['demand_kwh']/24*1000
    b=SolarBase(c,np.zeros(3));capacity=c.battery;minimum=capacity
    demand=c.idle_power;missing=0.;first_missing=0.;first_final=None
    for tick in range(days*86400):
        hour=(config.start_hour+(tick%86400+.5)/3600)%24
        pv=sunlight(c.latitude_deg,c.start_day_of_year,hour,c.peak_irradiance).irradiance*c.panel_area*c.panel_efficiency
        available=min(b.battery+pv,config.logistics.inverter_power_w,pv+min(b.battery,capacity/KWH*10000))
        missing+=max(0.,demand-available)
        b.tick(1.,hour,[],[],c.start_day_of_year,bus_limit_w=config.logistics.inverter_power_w,
               storage_charge_w=capacity/KWH*2000,storage_discharge_w=capacity/KWH*10000)
        minimum=min(minimum,b.battery)
        if tick==86399:first_final=b.battery;first_missing=missing
    error=capacity+b.generated-b.curtailed-b.consumed-b.battery
    return dict(seconds=days*86400,physical_dt_seconds=1,initial_kwh=capacity/KWH,
        first_day_final_kwh=first_final/KWH,final_kwh=b.battery/KWH,
        minimum_soc=minimum/capacity,first_day_unserved_kwh=first_missing/KWH,
        unserved_kwh=missing/KWH,daily_storage_drift_kwh=(b.battery-first_final)/KWH,
        energy_error_j=error,generated_kwh=b.generated/KWH,consumed_kwh=b.consumed/KWH,
        curtailed_kwh=b.curtailed/KWH)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latitudes',default='0,31,-31,55,-55,70,-70')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rows=[]
    for latitude in map(float,args.latitudes.split(',')):
        source=experiment_config(Config());source.base.latitude_deg=latitude
        start=time.perf_counter();c,d=design_winter(source);audit=audit_bus(c,d)
        renewable=d['mode']=='renewable'
        assert audit['first_day_unserved_kwh']<1e-7
        assert audit['first_day_final_kwh']>=c.base.battery/KWH*.2-1e-6
        if renewable:
            assert audit['unserved_kwh']<1e-7 and audit['daily_storage_drift_kwh']>=-1e-6
            assert audit['minimum_soc']>=.2-1e-6
        else:assert audit['unserved_kwh']>0 and audit['daily_storage_drift_kwh']<0
        rows.append(dict(latitude=latitude,design=d,bus=audit,wall_seconds=time.perf_counter()-start))
        print(latitude,d['mode'],d['panel_area_m2'],d['base_capacity_kwh'],audit,flush=True)
    args.output.write_text(json.dumps(dict(cases=rows),ensure_ascii=False,allow_nan=False,indent=2)+'\n')


if __name__=='__main__':main()
