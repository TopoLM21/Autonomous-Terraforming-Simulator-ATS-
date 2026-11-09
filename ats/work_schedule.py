"""Annual work windows in the simulator's 365-day, local-solar calendar.

Schedules gate new work; completing an existing flight, safe return, charging
and equipment service remain physical operations, not a freeze of the clock.
"""
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import math
from .solar import sunlight


def day_label(day):
    return (date(2025,1,1)+timedelta(days=day-1)).strftime('%d.%m')


@dataclass(frozen=True)
class WorkSchedule:
    mode: str = 'continuous'  # continuous, hours, daylight
    first_day: int = 60
    last_day: int = 273
    hours: float = 8.

    @classmethod
    def from_data(cls,data):
        value=cls(**data);value.validate();return value

    def to_data(self):return asdict(self)

    def validate(self):
        if self.mode not in ('continuous','hours','daylight'):raise ValueError('Неизвестный режим рабочего календаря')
        if any(type(v) is not int or not 1<=v<=365 for v in (self.first_day,self.last_day)):
            raise ValueError('Даты рабочего периода: дни 1–365')
        if isinstance(self.hours,bool) or not isinstance(self.hours,(int,float)) or not math.isfinite(self.hours) or not 0<self.hours<=24:
            raise ValueError('Длительность дневной работы: больше 0 и не более 24 часов')

    def days(self):
        if self.first_day<=self.last_day:return list(range(self.first_day,self.last_day+1))
        return list(range(self.first_day,366))+list(range(1,self.last_day+1))

    def contains(self,day):
        return self.first_day<=day<=self.last_day if self.first_day<=self.last_day else day>=self.first_day or day<=self.last_day

    def interval(self,latitude,day):
        if not self.contains(day):return None
        if self.mode=='continuous':return 0.,24.
        daylight=sunlight(latitude,day,12.).daylight_hours
        if daylight<=1e-10:return None
        if self.mode=='hours' and daylight+1e-9<self.hours:return None
        hours=daylight if self.mode=='daylight' else self.hours
        return max(0.,12.-hours/2),min(24.,12.+hours/2)

    def allows(self,latitude,day,hour):
        interval=self.interval(latitude,day)
        return interval is not None and interval[0]<=hour%24<interval[1]

    def reason(self,latitude,day,hour):
        if not self.contains(day):return f'Вне рабочего периода {day_label(self.first_day)}–{day_label(self.last_day)}'
        interval=self.interval(latitude,day)
        if interval is None:return 'Световой день короче заданного рабочего времени'
        if not self.allows(latitude,day,hour):return 'Ожидание дневной смены; зарядка и безопасное завершение заданий продолжаются'
        return ''

    def summary(self,latitude):
        self.validate();days=self.days()
        daylight=[(sunlight(latitude,d,12.).daylight_hours,d) for d in days]
        shortest,day=min(daylight,key=lambda row:(row[0],row[1]))
        impossible=[d for d in days if self.mode=='hours' and self.interval(latitude,d) is None]
        active=[self.interval(latitude,d) for d in days]
        hours=[(v[1]-v[0]) if v else 0. for v in active]
        return dict(first_day=self.first_day,last_day=self.last_day,days=len(days),
            label=f'{day_label(self.first_day)}–{day_label(self.last_day)}',
            min_daylight_hours=shortest,shortest_day=day,shortest_date=day_label(day),
            min_work_hours=min(hours),max_work_hours=max(hours),annual_work_hours=sum(hours),
            insufficient_daylight_days=impossible)


def state(config,day,hour,latitude=None):
    raw=getattr(config,'operating_schedule',None)
    if raw is None:return True,''
    schedule=WorkSchedule.from_data(raw)
    latitude=config.base.latitude_deg if latitude is None else latitude
    return schedule.allows(latitude,day,hour),schedule.reason(latitude,day,hour)
