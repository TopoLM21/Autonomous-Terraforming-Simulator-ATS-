"""Bounded offscreen comparison on the SAME physical snapshot, including paint.

Use --root for a baseline checkout and --effects off/on. These are Qt timings,
not a measurement of a user's GPU, monitor FPS or a sustained 24-hour mission.
"""
import argparse,hashlib,inspect,json,os,platform,statistics,sys,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--state',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--effects',choices=('off','on'),default='on');parser.add_argument('--frames',type=int,default=20)
    parser.add_argument('--steps',type=int,default=60);parser.add_argument('--max-wall',type=float,default=45.)
    args=parser.parse_args()
    if args.frames<2 or args.steps<1 or args.max_wall<=0:parser.error('frames >= 2, steps/max-wall > 0 required')
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen');sys.path.insert(0,str(args.root.resolve()))
    import numpy as np
    from PySide6 import QtCore,QtWidgets
    from ats.checkpoint import load_state
    from ats.gui import MainWindow
    session=load_state(args.state);sim=session.simulation;p=session.complex_project
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(sim.config,session=session);w.timer.stop();w.resize(1280,900)
    for name in ('day_night','drone_lights'):
        if hasattr(w,name):getattr(w,name).setChecked(args.effects=='on')
    w.render_mode.setCurrentIndex(w.render_mode.findData('auto'));w.show();w.fit_map();app.processEvents()
    w.running=True
    for _ in range(3):w.refresh();app.processEvents()
    def static(dirty):
        update=[];paint=[]
        for _ in range(args.frames):
            if dirty:w.last_revision=-1;w.last_terrain_wall=-float('inf')
            now=time.perf_counter();w.refresh();update.append(time.perf_counter()-now)
            now=time.perf_counter();w.map.viewport().repaint();app.processEvents();paint.append(time.perf_counter()-now)
        return dict(update_median_ms=1000*statistics.median(update),paint_median_ms=1000*statistics.median(paint),
            update_max_ms=1000*max(update),paint_max_ms=1000*max(paint))
    result=dict(root=str(args.root.resolve()),renderer_path=inspect.getfile(MainWindow),python=platform.python_version(),numpy=np.__version__,qt=QtCore.qVersion(),
        latitude=sim.base.config.latitude_deg,day_of_year=sim.day_of_year,start_hour=sim.hour,physical_dt_s=sim.config.dt,
        snapshot=args.state.name,snapshot_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
        effects=args.effects,lighting_available=hasattr(w,'day_night'),frames=args.frames,map_items=len(w.map.items()),
        viewport_px=[w.map.viewport().width(),w.map.viewport().height()],n=sim.terrain.height.shape[0],
        complexes=len(p.units) if p else 1,drones=sum(len(u.simulation.drones) for u in p.units) if p else len(sim.drones),
        platforms=sum(u.simulation.base.layout_spec['platform_count'] for u in p.units) if p else sim.base.layout_spec['platform_count'],
        cached=static(False),dirty=static(True))
    actor=p or sim;clock=lambda:p.time if p else sim.time
    start=clock();target=start+args.steps;physical_step=actor.step;gaps=[];last=[time.perf_counter()]
    def step():
        physical_step()
        if clock()>=target:w.running=False;w.credit=0.;app.quit()
    actor.step=step
    heartbeat=QtCore.QTimer();heartbeat.setInterval(10)
    def beat():
        now=time.perf_counter();gaps.append(now-last[0]);last[0]=now
    heartbeat.timeout.connect(beat)
    limit=QtCore.QTimer();limit.setSingleShot(True);limit.timeout.connect(app.quit)
    w.speed.setCurrentIndex(w.speed.findData(2000));w.credit=0;w.last_wall=time.perf_counter();w.running=True
    tick=time.perf_counter();last[0]=tick;heartbeat.start();limit.start(int(args.max_wall*1000));w.timer.start(0);app.exec()
    wall=time.perf_counter()-tick;w.timer.stop();heartbeat.stop();limit.stop();actor.step=physical_step
    gaps.append(time.perf_counter()-last[0]);report=actor.report()
    if clock()!=target:raise RuntimeError(f'Benchmark advanced {clock()-start}/{args.steps} seconds; limit/state reached')
    result.update(model_seconds=clock()-start,wall_s=wall,factor=(clock()-start)/wall,max_heartbeat_gap_s=max(gaps),
        physical_sha256=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest(),
        mass_error_kg=report['mass_error_kg'],energy_error_j=report['energy_error_j'])
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True);w.running=False;w.close();app.processEvents()


if __name__=='__main__':main()
