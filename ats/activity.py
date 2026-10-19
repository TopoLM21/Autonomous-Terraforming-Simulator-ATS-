"""Mutually exclusive machine activities, sampled at the end of each fixed step."""
from dataclasses import dataclass,field
import math


LABELS={
    'digging':'Копание / рейсы копателя','transport':'Курьер / перевозки',
    'group_lift':'Совместный перенос Крота','charging':'Зарядка',
    'battery_swap':'Замена батарей','tool_change':'Смена инструмента',
    'flight':'Перелёт','moving':'Наземное движение','setup':'Подготовка / опоры',
    'waiting_charge':'Ожидание зарядки','waiting_transport':'Ожидание перевозки',
    'waiting_container':'Ожидание тары','waiting_power':'Ожидание энергии',
    'waiting_relocation':'Ожидание переноса','idle':'Простой / завершение',
}


@dataclass
class ActivityStats:
    seconds: dict = field(default_factory=dict)
    days: dict = field(default_factory=dict)

    def record(self, activity, start_seconds, duration=1.):
        if activity not in LABELS or not math.isfinite(start_seconds) or not math.isfinite(duration) or duration<0:
            raise ValueError('Invalid activity interval')
        self.seconds[activity]=self.seconds.get(activity,0.)+duration
        end=start_seconds+duration
        while start_seconds<end:
            day=math.floor(start_seconds/86400)
            stop=min(end,(day+1)*86400)
            row=self.days.setdefault(day,{})
            row[activity]=row.get(activity,0.)+stop-start_seconds
            start_seconds=stop

    def report(self, day=None):
        values=self.seconds if day is None else self.days.get(day,{})
        return dict(total_seconds=sum(values.values()),seconds=dict(values))


def drone_activity(drone, world=None):
    state=drone.state.value
    if state=='SWAPPING_BATTERIES':return 'battery_swap'
    if state=='CHANGING_TOOL':return 'tool_change'
    if state=='CHARGING':return 'charging'
    if drone.grounded and drone.charge_requested:return 'waiting_charge'
    if world is not None:
        if world.group and any(member is drone for member in world.group.participants):return 'group_lift'
        if drone.identifier in world.active:return 'transport'
        if drone.identifier in world.direct_ids:return 'digging'
    elif drone.trip is not None:return 'digging'
    return 'idle' if drone.grounded else 'flight'


def excavator_activity(excavator):
    state=excavator.state.value
    if state in ('DIGGING','FILLING_CONTAINER'):return 'digging'
    if state=='MOVING_LOCAL':return 'moving'
    if state in ('DEPLOYING_SUPPORTS','SWAP_PREPARATION'):return 'setup'
    if state=='SWAPPING_BATTERIES':return 'battery_swap'
    if state in ('AIR_TRANSPORT','RIGGED_FOR_LIFT'):return 'group_lift'
    return {'WAITING_EMPTY_CONTAINER':'waiting_container','WAITING_TRANSPORT':'waiting_transport',
            'LOW_BATTERY':'waiting_power','WAITING_RELOCATION':'waiting_relocation'}.get(state,'idle')
