"""Actual deposited soil by solar-calendar day and a trailing 24-hour window."""
from collections import deque
from dataclasses import dataclass,field


@dataclass
class Productivity:
    days: dict=field(default_factory=dict)
    recent: deque=field(default_factory=deque)
    total_mass: float=0.0
    total_volume: float=0.0

    def record(self,time,day,mass,density,destination):
        if mass<=0:return
        volume=mass/density
        row=self.days.setdefault(day,{"mass_kg":0.,"volume_m3":0.,"fill_mass_kg":0.,"dump_mass_kg":0.,"deliveries":0})
        row["mass_kg"]+=mass;row["volume_m3"]+=volume
        key=destination+"_mass_kg";row[key]=row.get(key,0.)+mass;row["deliveries"]+=1
        self.recent.append((time,mass,volume));self.total_mass+=mass;self.total_volume+=volume
        while self.recent and self.recent[0][0]<=time-86400:self.recent.popleft()

    def report(self,time,day):
        while self.recent and self.recent[0][0]<=time-86400:self.recent.popleft()
        today=self.days.get(day,{"mass_kg":0.,"volume_m3":0.})
        return {"transported_mass_kg":self.total_mass,"transported_volume_m3":self.total_volume,"today_mass_kg":today["mass_kg"],"today_volume_m3":today["volume_m3"],"last24_mass_kg":sum(e[1] for e in self.recent),"last24_volume_m3":sum(e[2] for e in self.recent),"daily":[{"mission_day":number,**row} for number,row in sorted(self.days.items())]}
