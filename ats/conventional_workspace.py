"""A conventional-only worksite: no implicit drone purchase or autonomous base."""
from pathlib import Path
import time
import numpy as np
from PySide6 import QtCore,QtWidgets
import pyqtgraph as pg
from .config import Config
from .simulation import Simulation,elapsed_text
from .checkpoint import load_state,save_state
from .map_visuals import MapWidget,terrain_rgb
from .ui_windows import fit_window


class ConventionalWorkWindow(QtWidgets.QMainWindow):
    planning_only=False
    complex_project=None

    def __init__(self,config=None,landscape=None,session=None,parent=None):
        super().__init__(parent)
        self.sim=session.simulation if session is not None else Simulation.conventional_only(config or Config())
        if self.sim.drones or self.sim.logistics is not None:
            raise ValueError('Это окно предназначено для площадки без автономной техники')
        self.config=self.sim.config
        self.landscape=session.landscape if session is not None else landscape
        self.complex_project=session.complex_project if session is not None else None
        self.running=False;self.last_wall=time.perf_counter();self._credit=0.
        self._map_revision=None;self._last_refresh=-float('inf');self._label_items={}
        self.equipment_window=None
        self.setWindowTitle('ATS 2.0 · Обычная техника')
        central=QtWidgets.QWidget();self.setCentralWidget(central)
        layout=QtWidgets.QVBoxLayout(central);toolbar=QtWidgets.QHBoxLayout()
        self.equipment_button=QtWidgets.QPushButton('Обычная техника…')
        self.equipment_button.clicked.connect(self.show_conventional)
        self.autonomous_button=QtWidgets.QPushButton('＋ Автономный комплекс…')
        self.autonomous_button.clicked.connect(self.show_composer)
        self.play=QtWidgets.QPushButton('▶ Старт');self.play.clicked.connect(self.toggle_play)
        self.step_button=QtWidgets.QPushButton('Шаг 1 с');self.step_button.clicked.connect(self.single_step)
        self.speed=QtWidgets.QComboBox()
        for factor in (1,10,50,200,1000):self.speed.addItem(f'{factor}×',factor)
        self.speed.setCurrentIndex(2);self.speed.currentIndexChanged.connect(self.reset_clock)
        self.save_button=QtWidgets.QPushButton('Сохранить…');self.save_button.clicked.connect(self.save_session)
        self.load_button=QtWidgets.QPushButton('Открыть…');self.load_button.clicked.connect(self.load_session)
        for widget in (self.equipment_button,self.play,self.step_button,self.speed,self.save_button,self.load_button):toolbar.addWidget(widget)
        toolbar.addStretch();layout.addLayout(toolbar)
        delivery=QtWidgets.QHBoxLayout();delivery.addWidget(self.autonomous_button);delivery.addStretch();layout.addLayout(delivery)
        self.overview=QtWidgets.QLabel();self.overview.setWordWrap(True);layout.addWidget(self.overview)
        self.map=MapWidget(background='#122133');self.map.setAspectLocked(True)
        self.map.setLabel('bottom','X',units='м');self.map.setLabel('left','Y',units='м')
        self.image=pg.ImageItem(axisOrder='row-major');self.map.addItem(self.image)
        self.design_overlay=pg.ImageItem(axisOrder='row-major');self.map.addItem(self.design_overlay)
        self.work_outline=pg.PlotDataItem(pen=pg.mkPen('#46d6fc',width=2));self.map.addItem(self.work_outline)
        self.paths=pg.PlotDataItem(pen=pg.mkPen('#f8dd7c',width=1),connect='finite');self.map.addItem(self.paths)
        self.machines=pg.ScatterPlotItem(pxMode=True);self.map.addItem(self.machines)
        self.machines.setZValue(3.);self.paths.setZValue(2.)
        self.camp_marker=pg.ScatterPlotItem(pxMode=True,symbol='+',size=11,brush='#65dcb2',pen=pg.mkPen('#10232d',width=2));self.map.addItem(self.camp_marker);self.camp_marker.setZValue(3.)
        self.camp_label=pg.TextItem('Заправка',color='#a7f0c3',anchor=(0,0));self.map.addItem(self.camp_label)
        layout.addWidget(self.map,1)
        self.details=QtWidgets.QLabel();self.details.setWordWrap(True);layout.addWidget(self.details)
        note=QtWidgets.QLabel('Корпуса машин и топливная база показаны в масштабе, в метрах. Голубой контур — рабочая захватка; зелёный — проектный контур съезда, его готовность определяется грунтом; жёлтый — проверенные наземные маршруты. Смета обычной техники учитывается отдельно от автономных комплексов.')
        note.setWordWrap(True);layout.addWidget(note)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(20)
        self.refresh(True);self.reset_view();fit_window(self,1150,790)

    def reset_clock(self,*_):
        self.last_wall=time.perf_counter();self._credit=0.

    def toggle_play(self):
        if getattr(self.sim,'conventional',None) is None:return
        self.running=not self.running;self.reset_clock()
        self.play.setText('Ⅱ Пауза' if self.running else '▶ Старт')
        self.step_button.setEnabled(not self.running)

    def single_step(self):
        if self.running or getattr(self.sim,'conventional',None) is None:return
        if self.complex_project is not None:self.complex_project.step()
        else:self.sim.step()
        self.refresh(True)

    def tick(self):
        now=time.perf_counter();elapsed=max(0.,now-self.last_wall);self.last_wall=now
        if self.running:
            # UI deadlines bound work; they never change the one-second physics.
            self._credit+=min(elapsed,.25)*self.speed.currentData()
            deadline=now+.05;steps=0
            while self._credit>=1. and time.perf_counter()<deadline:
                previous=self.sim.time
                if self.complex_project is not None:self.complex_project.step()
                else:self.sim.step()
                if self.sim.time==previous:
                    self.running=False;self.play.setText('▶ Старт');break
                self._credit-=1.;steps+=1
            self._credit=min(self._credit,self.speed.currentData())
            if steps:self.statusBar().showMessage(f'Цель {self.speed.currentData()}× · последний пакет {steps} модельных секунд')
        if now-self._last_refresh>=.2:self.refresh()

    def install_conventional(self,config):
        self.sim.install_conventional(config)
        self.refresh(True)

    def install_simulation(self,simulation,landscape=None,complex_project=None):
        """Adopt a larger live ground window, retaining this window's widgets."""
        if simulation.drones or simulation.logistics is not None:
            raise ValueError('Эта площадка предназначена для обычной техники')
        self.sim=simulation;self.config=simulation.config
        self.landscape=landscape;self.complex_project=complex_project
        self._map_revision=None;self.reset_clock();self.refresh(True);self.reset_view()

    def show_conventional(self):
        from .conventional_ui import ConventionalWindow
        if self.equipment_window is None:self.equipment_window=ConventionalWindow(self)
        self.equipment_window.show();self.equipment_window.raise_();self.equipment_window.activateWindow()

    def show_composer(self):
        from .complex_setup_ui import ComplexSetupDialog
        if hasattr(self,'setup_window'):self.setup_window.close()
        self.setup_window=ComplexSetupDialog(self);self.setup_window.show()

    def _after_file_action(self,running):
        self.running=bool(running and self.sim.conventional is not None)
        self.reset_clock();self.play.setText('Ⅱ Пауза' if self.running else '▶ Старт')

    def open_mixed_session(self,session,start=False):
        from .gui import MainWindow
        self.running=False;self.timer.stop()
        if self.equipment_window:self.equipment_window.close()
        self.mixed_window=MainWindow(session.simulation.config,session=session)
        self.mixed_window.show()
        if start:self.mixed_window.toggle_play()
        self.hide()

    def reset_view(self):
        side=self.sim.terrain.cell*self.sim.terrain.height.shape[0]
        self.map.setRange(xRange=(0,side),yRange=(0,side),padding=.02)

    def refresh(self,force=False):
        self._last_refresh=time.perf_counter();t=self.sim.terrain
        key=(id(t),t.revision)
        if force or key!=self._map_revision:
            from .site_ground import marine_mask
            n=t.height.shape[0];stride=max(1,int(np.ceil(n/768)))
            rgb=terrain_rgb(t.height[::stride,::stride],t.cell*stride,self.config.trench.depth,
                self.config.terrain.dump_max_height,reference=t.reference[::stride,::stride],marine=marine_mask(t)[::stride,::stride])
            self.image.setImage(rgb,autoLevels=False);self.image.setRect(QtCore.QRectF(0,0,t.cell*n,t.cell*n))
            overlay=np.zeros((*t.height[::stride,::stride].shape,4),dtype=np.uint8)
            overlay[t.excavation[::stride,::stride]]=[65,180,250,45]
            overlay[t.fill_mask[::stride,::stride]]=[112,242,132,45]
            self.design_overlay.setImage(overlay,autoLevels=False)
            self.design_overlay.setRect(QtCore.QRectF(0,0,t.cell*n,t.cell*n))
            self._map_revision=key
        fleet=getattr(self.sim,'conventional',None)
        from .conventional_visuals import refresh_geometry
        refresh_geometry(self,self.map,fleet)
        self.play.setEnabled(fleet is not None);self.step_button.setEnabled(fleet is not None and not self.running)
        if fleet is None:
            self.overview.setText(f'Время {elapsed_text(self.sim.time)} · Обычная техника: не размещена · Автономная техника: отсутствует')
            self.details.setText('Откройте «Обычная техника…»: выберите машины, рабочую захватку и базу заправки.')
            self.machines.clear();self.camp_marker.clear();self.work_outline.clear();self.paths.clear()
            self.camp_label.hide()
            for label in self._label_items.values():label.hide()
            return
        report=fleet.report();machines=report.get('machines',[])
        self.overview.setText(f'Время {elapsed_text(self.sim.time)} · {len(machines)} машин · '+str(report.get('reason','')))
        rect=report.get('work_rect',fleet.config.work_rect)
        x,y,w,h=rect;self.work_outline.setData([x,x+w,x+w,x,x],[y,y,y+h,y+h,y])
        camp=report.get('base_position_m',[fleet.config.base_x,fleet.config.base_y])
        self.camp_marker.setData([camp[0]],[camp[1]])
        self.camp_label.setPos(camp[0]+.8,camp[1]-3.5);self.camp_label.show()
        spots=[];routes=[];rows=[];visible=set()
        for machine in machines:
            position=machine['position_m'];kind=machine['kind'];name=machine['id'];visible.add(name)
            spots.append(dict(pos=tuple(position[:2]),symbol='s' if kind=='excavator' else 't',size=6,
                brush='#f2b95e' if kind=='excavator' else '#d5e8f4',pen=pg.mkPen('#11232d',width=1.5)))
            label=self._label_items.get(name)
            if label is None:label=pg.TextItem(anchor=(0,1),color='#f4f6f0',fill=pg.mkBrush(10,22,35,210));self.map.addItem(label);self._label_items[name]=label
            title='Экскаватор' if kind=='excavator' else 'Самосвал'
            label.setText(f"{title} {name.rsplit('_',1)[-1]} · {machine['state']}");label.setPos(position[0]+.8,position[1]+.8);label.show()
            path=machine.get('path_m',[])
            if path:routes.extend([position[:2],*[p[:2] for p in path],(np.nan,np.nan)])
            rows.append(f"{title}: {machine['state']} · {machine.get('fuel_l',0.):.1f} л · {machine.get('cargo_kg',0.):.0f} кг"+
                (' · '+machine['reason'] if machine.get('reason') else ''))
        self.machines.setData(spots)
        for name,label in self._label_items.items():label.setVisible(name in visible)
        if routes:
            array=np.asarray(routes);self.paths.setData(array[:,0],array[:,1],connect='finite')
        else:self.paths.clear()
        summary=rows[:4]
        if len(rows)>4:summary.append(f'Ещё машин: {len(rows)-4}. Подробности — «Обычная техника…».')
        self.details.setText('\n'.join(summary))

    def save_session(self,path=None):
        if not isinstance(path,(str,Path)):
            path=QtWidgets.QFileDialog.getSaveFileName(self,'Сохранить площадку','','Состояние ATS (*.atsstate)')[0]
        if not path:return False
        try:
            save_state(path,self.sim,self.landscape,ui={'conventional_workspace':True,'conventional_speed':self.speed.currentData()},complex_project=self.complex_project)
            self.statusBar().showMessage(f'Сохранено: {Path(path).name}');return True
        except (OSError,ValueError,TypeError) as error:
            self.statusBar().showMessage(f'Ошибка сохранения: {error}');return False

    def load_session(self,path=None):
        if not isinstance(path,(str,Path)):
            path=QtWidgets.QFileDialog.getOpenFileName(self,'Открыть площадку','','Состояние ATS (*.atsstate)')[0]
        if not path:return False
        try:
            session=load_state(path)
            if session.simulation.drones or session.simulation.logistics is not None:
                raise ValueError('Откройте сохранение автономных комплексов через главное окно проекта')
            if self.equipment_window is not None:self.equipment_window.close();self.equipment_window=None
            self.running=False;self.play.setText('▶ Старт');self.sim=session.simulation;self.config=self.sim.config;self.landscape=session.landscape
            self.complex_project=session.complex_project
            choice=(session.ui or {}).get('conventional_speed',50);index=self.speed.findData(choice)
            if index>=0:self.speed.setCurrentIndex(index)
            self.reset_clock();self.refresh(True);self.reset_view()
            self.statusBar().showMessage(f'Загружено: {Path(path).name}');return True
        except (OSError,ValueError,TypeError) as error:
            self.statusBar().showMessage(f'Ошибка загрузки: {error}');return False

    def closeEvent(self,event):
        self.running=False;self.timer.stop()
        if hasattr(self,'setup_window'):self.setup_window.close()
        if self.equipment_window is not None:self.equipment_window.close()
        if self.landscape is not None:self.landscape.cancel_prefetch()
        super().closeEvent(event)
