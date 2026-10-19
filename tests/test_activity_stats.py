"""Activity time is a partition of simulated time; Krot production is measured soil."""
from copy import deepcopy
import os
from types import SimpleNamespace
import pytest
from ats.activity import ActivityStats
from ats.config import Config
from ats.experiments import experiment_config
from ats.scenarios import scenario
from ats.simulation import Simulation


def test_activity_splits_midnight_and_rejects_invalid_intervals_atomically():
    stats=ActivityStats();stats.record('digging',86399.5,1.)
    stats.record('charging',86400.5,2.)
    assert stats.report()['total_seconds']==3
    assert stats.report(0)['seconds']['digging']==.5
    assert stats.report(1)['total_seconds']==2.5
    before=deepcopy(stats)
    for kind,start,duration in [('invalid',0,1),('digging',float('nan'),1),('idle',0,-1)]:
        with pytest.raises(ValueError):stats.record(kind,start,duration)
        assert stats==before


def test_machine_times_sum_to_elapsed_seconds_and_krot_daily_mass_matches_soil():
    c=experiment_config(scenario('medium'));c.start_hour=23.999
    sim=Simulation(c);sim.run(300);r=sim.report();e=sim.logistics.excavator
    assert e.excavated_kg>0
    for actor in [*sim.drones,e]:
        assert actor.activity.report()['total_seconds']==sim.time
        assert sum(sum(day.values()) for day in actor.activity.days.values())==pytest.approx(sim.time)
        assert len(actor.activity.days)==2
    assert r['excavator']['excavated_m3']==pytest.approx(e.excavated_kg/c.terrain.density)
    assert sum(day['mass_kg'] for day in r['excavator']['daily_excavation'])==pytest.approx(e.excavated_kg)
    assert e.productivity.total_mass==pytest.approx(e.excavated_kg)
    assert abs(r['mass_error_kg'])<1e-7 and abs(r['energy_error_j'])<1e-4
    before=[a.activity.report() for a in [*sim.drones,e]]
    sim.state=type(sim.state).BLOCKED;sim.step()
    assert before==[a.activity.report() for a in [*sim.drones,e]]


def test_retired_drone_keeps_its_activity_and_new_drone_starts_at_delivery():
    s=Simulation();s.run(20)
    config=deepcopy(s.config);config.fleet.count=2;config.fleet.models=[deepcopy(s.drone.config),deepcopy(s.drone.config)]
    s.request_landing()
    for _ in range(200):
        if all(d.grounded for d in s.drones):break
        s.step()
    s.request_equipment(config);s.landing_requested=False;s.run(10)
    added=s.drones[-1];assert added.activity.report()['total_seconds']==s.time-added.created_at
    config=deepcopy(s.config);config.fleet.count=1;config.fleet.models=config.fleet.models[:1]
    s.request_landing()
    for _ in range(300):
        if all(d.grounded for d in s.drones):break
        s.step()
    s.request_equipment(config)
    total=added.activity.report()['total_seconds'];s.run(5)
    assert added in s.retired_drones and added.activity.report()['total_seconds']==total
    assert s.report()['retired_drones'][0]['activity']['total_seconds']==total


def test_stats_window_contains_krot_production_and_activity_breakdown():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from ats.gui import MainWindow
    from ats.equipment_ui import DailyStatsWindow
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(experiment_config(scenario('medium')));w.timer.stop();w.sim.run(240)
    stats=DailyStatsWindow(w);stats.timer.stop()
    assert 'Крот выкопал' in stats.summary.text()
    assert any(stats.table.item(i,0).text()=='Крот · выемка' for i in range(stats.table.rowCount()))
    assert any(stats.activity.item(i,0).text()=='Крот' for i in range(stats.activity.rowCount()))
    assert any('Копание' in stats.activity.item(i,1).text() for i in range(stats.activity.rowCount()))
    stats.close();w.close();app.processEvents()


def test_fast_compute_slices_run_more_steps_without_forcing_a_redraw(monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    import ats.gui as gui
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=gui.MainWindow(Config());w.timer.stop();w.speed.setCurrentIndex(w.speed.findData(2000))
    clock=SimpleNamespace(now=1.)
    monkeypatch.setattr(gui,'time',SimpleNamespace(perf_counter=lambda:clock.now))
    step=w.sim.step;calls=[];renders=[]
    def physical_step():
        step();calls.append(w.sim.time);clock.now+=.001
    monkeypatch.setattr(w.sim,'step',physical_step)
    monkeypatch.setattr(w,'refresh',lambda *args:renders.append(clock.now))
    w.running=True;w.credit=200;w.last_wall=clock.now
    w.performance_wall=clock.now;w.performance_sim=0
    w.on_frame();first=len(calls)
    assert 20<first<=32 and len(renders)==1 and w.config.dt==1
    w.on_frame();assert len(calls)>first and len(renders)==1
    assert w.sim.time==len(calls) and calls==list(range(1,len(calls)+1))
    assert w.sim.drone.activity.report()['total_seconds']==len(calls)
    w.running=False;before=w.sim.time;w.on_frame();assert w.sim.time==before
    w.close();app.processEvents()
