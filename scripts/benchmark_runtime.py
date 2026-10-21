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
    if sim.state.value!='RUNNING':raise ValueError('Для замера нужно сохранение незавершённой работающей симуляции')
    start=sim.time;target=start+args.steps
    profile=cProfile.Profile() if args.profile and number==0 else None
    window=None
    if mode=='gui':
        if args.offscreen:os.environ['QT_QPA_PLATFORM']='offscreen'
        from PySide6 import QtCore,QtWidgets
        from ats.gui import MainWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        # Supports the previous GUI as well, for measurements against a checkout.
        window=(MainWindow(sim.config,session=session) if 'session' in inspect.signature(MainWindow).parameters
                else MainWindow(sim.config))
        window.timer.stop();sim=window.sim;window.show();app.processEvents()
        window.speed.setCurrentIndex(window.speed.findData(2000))
        if hasattr(window,'render_mode'):
            window.render_mode.setCurrentIndex(window.render_mode.findData(args.render))
        original=sim.step
        def step():
            original()
            if sim.time>=target or sim.state.value!='RUNNING':
                window.running=False;window.credit=0;app.quit()
        sim.step=step
        window.running=True;window.credit=0.;window.last_wall=time.perf_counter()
        window.timer.start(0)
        deadline=QtCore.QTimer();deadline.setSingleShot(True);deadline.timeout.connect(app.quit)
        deadline.start(round(args.max_wall_seconds*1000))
    ready=time.perf_counter()
    if profile:profile.enable()
    if window is None:sim.run(args.steps)
    else:app.exec()
    if profile:profile.disable()
    elapsed=time.perf_counter()-ready
    if window is not None:
        window.timer.stop();deadline.stop();sim.step=original
    report=sim.report();advanced=sim.time-start
    if advanced<=0:raise ValueError('Замер не продвинул модельное время')
    row=dict(mode=mode,render=args.render if mode=='gui' else None,profiled=profile is not None,
        initialization_seconds=ready-begin,wall_seconds=elapsed,model_seconds=advanced,actual_speed=advanced/elapsed,
        grid_cells=sim.config.terrain.n,drones=len(sim.drones),inventory_items=len(sim.logistics.inventory.items) if sim.logistics else 0,
        state=sim.state.value,trips=report['trips'],mass_error_kg=report['mass_error_kg'],energy_error_j=report['energy_error_j'],
        physical_sha256=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest(),
        wall_limit_reached=mode=='gui' and sim.time<target and sim.state.value=='RUNNING')
    if profile:
        prefix=Path(args.profile);prefix.parent.mkdir(parents=True,exist_ok=True)
        profile.dump_stats(str(prefix)+f'-{mode}.prof');row.update(profile_rows(profile))
    if window is not None:
        window.close();window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None,QtCore.QEvent.Type.DeferredDelete);app.processEvents()
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,help='Profile your actual .atsstate instead of the reference canal')
    parser.add_argument('--mode',choices=('headless','gui','both'),default='headless')
    parser.add_argument('--render',choices=('auto','detail','off'),default='auto')
    parser.add_argument('--steps',type=int,default=1200);parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--max-wall-seconds',type=float,default=300.)
    parser.add_argument('--offscreen',action='store_true',help='Use Qt offscreen instead of a visible desktop window')
    parser.add_argument('--profile',help='Write first-trial cProfile files and timing tables; profiler affects speed')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.steps<1 or not 1<=args.trials<=100 or args.max_wall_seconds<=0:parser.error('steps > 0; trials 1..100; max wall > 0')
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
    result=dict(python=platform.python_version(),platform=platform.system(),cpu_count=os.cpu_count(),
        physical_dt_seconds=1,steps=args.steps,source=args.state.name if args.state else 'Offline Qattara reference canal',
        trials=rows,median=medians)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
