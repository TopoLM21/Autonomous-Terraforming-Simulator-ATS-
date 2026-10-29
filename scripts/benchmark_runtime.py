"""Profile a real saved run or the offline Qattara canal, with/without Qt drawing."""
import argparse
import cProfile
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time
from types import SimpleNamespace


def default_session():
    from ats.channel import ChannelProject
    from ats.config import Config
    from ats.experiments import experiment_config
    from ats.landscape import Landscape,qattara_dem
    from ats.simulation import Simulation
    raster=qattara_dem();region=Landscape(raster)
    points=[list(raster.to_metres(*point)) for point in ((28.3,31.2),(27.5,29.5))]
    project=ChannelProject.create(raster,points)
    return SimpleNamespace(simulation=Simulation(project.configuration(region,experiment_config(Config()))),landscape=region,ui={})


def session_factory(path):
    if path is None:return default_session()
    from ats.checkpoint import load_state
    return load_state(path)


def profile_rows(profile):
    stats=pstats.Stats(profile)
    rows=[dict(file=Path(key[0]).name,line=key[1],function=key[2],calls=value[1],
               self_seconds=value[2],cumulative_seconds=value[3]) for key,value in stats.stats.items()]
    return dict(self_top=sorted(rows,key=lambda r:r['self_seconds'],reverse=True)[:25],
                cumulative_top=sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:25])


def trial(args, mode, number):
    gc.collect();begin=time.perf_counter();session=session_factory(args.state);sim=session.simulation
    project=getattr(session,'complex_project',None)
    actor=project if project is not None else sim
    clock=lambda:project.time if project is not None else sim.time
    running=lambda:project.running if project is not None else sim.state.value=='RUNNING'
    if not running():raise ValueError('Для замера нужно сохранение незавершённой работающей симуляции')
    for _ in range(getattr(args,'warmup',0)):
        if not running():break
        actor.step()
    start=clock();target=start+args.steps
    robots=[u.simulation.logistics.excavator for u in project.units] if project is not None else [sim.logistics.excavator] if sim.logistics else []
    robots=[e for e in robots if e is not None]
    robot_before=dict(moving=sum(e.moving_seconds for e in robots),working=sum(e.working_seconds for e in robots),
        relocations=sum(e.relocations for e in robots))
    robot_states={state:sum(e.state.value==state for e in robots) for state in sorted({e.state.value for e in robots})}
    profile=cProfile.Profile() if args.profile and number==0 else None
    step_costs=[];physical_step=actor.step
    def timed_step():
        stamp=time.perf_counter();physical_step();step_costs.append(time.perf_counter()-stamp)
    actor.step=timed_step
    event_gaps=[];heartbeat=None;last_beat=[0.]
    window=None;render_costs=[];original_refresh=None
    if mode=='gui':
        if args.offscreen:os.environ['QT_QPA_PLATFORM']='offscreen'
        from PySide6 import QtCore,QtWidgets
        import pyqtgraph as pg
        if not getattr(pg.PlotWidget,'_benchmark_paint',False):
            class MeasuredPlot(pg.PlotWidget):
                _benchmark_paint=True
                measured_paint_seconds=0.;measured_paints=0
                def paintEvent(self,event):
                    tick=time.perf_counter();super().paintEvent(event)
                    self.measured_paint_seconds+=time.perf_counter()-tick;self.measured_paints+=1
            pg.PlotWidget=MeasuredPlot
        from ats.gui import MainWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        # Supports the previous GUI as well, for measurements against a checkout.
        window=(MainWindow(sim.config,session=session) if 'session' in inspect.signature(MainWindow).parameters
                else MainWindow(sim.config))
        window.timer.stop();sim=window.sim
        window.resize(getattr(args,'width',1280),getattr(args,'height',900));window.show();app.processEvents()
        window.speed.setCurrentIndex(window.speed.findData(2000))
        if hasattr(window,'render_mode'):
            window.render_mode.setCurrentIndex(window.render_mode.findData(args.render))
        actor=project if project is not None else sim
        original=actor.step
        def step():
            original()
            if clock()>=target or not running():
                window.running=False;window.credit=0;app.quit()
        actor.step=step
        original_refresh=window.refresh
        def refresh(*values,**kwargs):
            tick=time.perf_counter()
            try:return original_refresh(*values,**kwargs)
            finally:render_costs.append(time.perf_counter()-tick)
        window.refresh=refresh
        window.map.measured_paint_seconds=0.;window.map.measured_paints=0
        window.running=True;window.credit=0.;window.last_wall=time.perf_counter()
        window.timer.start(0)
        deadline=QtCore.QTimer();deadline.setSingleShot(True);deadline.timeout.connect(app.quit)
        deadline.start(round(args.max_wall_seconds*1000))
        heartbeat=QtCore.QTimer();heartbeat.setInterval(10)
        def beat():
            stamp=time.perf_counter();event_gaps.append(stamp-last_beat[0]);last_beat[0]=stamp
        heartbeat.timeout.connect(beat)
    ready=time.perf_counter()
    last_beat[0]=ready
    if heartbeat is not None:heartbeat.start()
    if profile:profile.enable()
    if window is None:
        for _ in range(args.steps):
            if not running() or time.perf_counter()-ready>=args.max_wall_seconds:break
            actor.step()
    else:app.exec()
    if profile:profile.disable()
    elapsed=time.perf_counter()-ready
    if window is not None:
        event_gaps.append(time.perf_counter()-last_beat[0]);heartbeat.stop()
        window.timer.stop();deadline.stop()
    actor.step=physical_step
    report=sim.report() if project is None else dict(project.report(),reports=[u.simulation.report() for u in project.units])
    advanced=clock()-start
    if advanced<=0:raise ValueError('Замер не продвинул модельное время')
    row=dict(mode=mode,render=args.render if mode=='gui' else None,profiled=profile is not None,
        initialization_seconds=ready-begin,wall_seconds=elapsed,model_seconds=advanced,actual_speed=advanced/elapsed,
        grid_cells=sim.config.terrain.n,drones=len(sim.drones) if project is None else report['drone_count'],
        inventory_items=(len(sim.logistics.inventory.items) if sim.logistics else 0) if project is None else sum(len(u.simulation.logistics.inventory.items) if u.simulation.logistics else 0 for u in project.units),
        state=report['state'],trips=report['trips'],mass_error_kg=report['mass_error_kg'],energy_error_j=report['energy_error_j'],
        physical_sha256=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest(),
        wall_limit_reached=clock()<target and running(),
        max_step_wall_seconds=max(step_costs,default=0.),
        p99_step_wall_seconds=sorted(step_costs)[min(len(step_costs)-1,int(len(step_costs)*.99))] if step_costs else 0.,
        max_gui_event_gap_seconds=max(event_gaps,default=0.) if window is not None else None,
        gui_prepare_seconds=sum(render_costs) if window is not None else None,
        gui_refreshes=len(render_costs) if window is not None else None,
        gui_paint_seconds=window.map.measured_paint_seconds if window is not None else None,
        gui_paints=window.map.measured_paints if window is not None else None,
        gui_viewport_px=[window.map.viewport().width(),window.map.viewport().height()] if window is not None else None,
        engine_path=str(Path(inspect.getfile(type(sim))).resolve()),initial_time=start,robot_states_at_start=robot_states,
        robot_moving_seconds=sum(e.moving_seconds for e in robots)-robot_before['moving'],
        robot_working_seconds=sum(e.working_seconds for e in robots)-robot_before['working'],
        robot_relocations=sum(e.relocations for e in robots)-robot_before['relocations'])
    if profile:
        prefix=Path(args.profile);prefix.parent.mkdir(parents=True,exist_ok=True)
        profile.dump_stats(str(prefix)+f'-{mode}.prof');row.update(profile_rows(profile))
    if window is not None:
        del window.refresh  # Restore the class method without a bound-method self-cycle.
        window.close();window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None,QtCore.QEvent.Type.DeferredDelete);app.processEvents()
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1],help='Engine checkout to compare on the same snapshot')
    parser.add_argument('--width',type=int,default=1280);parser.add_argument('--height',type=int,default=900)
    parser.add_argument('--state',type=Path,help='Profile your actual .atsstate instead of the reference canal')
    parser.add_argument('--mode',choices=('headless','gui','both'),default='headless')
    parser.add_argument('--render',choices=('auto','detail','off'),default='auto')
    parser.add_argument('--steps',type=int,default=1200);parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=0,help='Untimed model steps after each load; use the same value in comparisons')
    parser.add_argument('--max-wall-seconds',type=float,default=300.)
    parser.add_argument('--offscreen',action='store_true',help='Use Qt offscreen instead of a visible desktop window')
    parser.add_argument('--profile',help='Write first-trial cProfile files and timing tables; profiler affects speed')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.root.resolve()))
    if args.steps<1 or args.warmup<0 or not 1<=args.trials<=100 or args.max_wall_seconds<=0:parser.error('steps > 0; warmup >= 0; trials 1..100; max wall > 0')
    rows=[];modes=('headless','gui') if args.mode=='both' else (args.mode,)
    for number in range(args.trials):
        for mode in modes:
            row=trial(args,mode,number);rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if not k.endswith('_top')},ensure_ascii=False),flush=True)
    medians={}
    for mode in modes:
        # Do not mix profiled and unprofiled wall times in the same median.
        group=[row for row in rows if row['mode']==mode and not row['profiled']]
        if not group:continue
        medians[mode]={key:statistics.median(row[key] for row in group) for key in ('wall_seconds','model_seconds','actual_speed')}
    result=dict(source_root=str(args.root.resolve()),python=platform.python_version(),platform=platform.system(),cpu_count=os.cpu_count(),
        physical_dt_seconds=1,steps=args.steps,source=args.state.name if args.state else 'Offline Qattara reference canal',
        trials=rows,median=medians)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
