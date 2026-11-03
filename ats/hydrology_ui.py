"""A global, explicitly counterfactual filling experiment beside earthworks."""
import math
import time
from PySide6 import QtCore,QtWidgets
import pyqtgraph as pg
import numpy as np
from .hydrology import BasinCurve,create_experiment


class FloodingDialog(QtWidgets.QDialog):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.region=owner.region;self.project=owner.project
        self.host=owner.window;self._resume=self.host.running
        self.running=False;self.credit=0.;self.last_wall=time.perf_counter();self.last_draw=0.;self._draw_key=None
        self.setWindowTitle('Наполнение впадины · готовый канал');self.resize(1120,830)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('Отдельный гидросценарий: «Мгновенный прокоп» считает канал готовым и соединённым с морем. '
            'Грунт, задания и энергия рабочего прогона не меняются. Впадина — единый сообщающийся резервуар; '
            'течения, эрозия, фильтрация и затопление техники здесь ещё не рассчитываются. '
            'Испарение задаётся условно, не измеренным рядом погоды.')
        note.setWordWrap(True);layout.addWidget(note)
        self.tabs=QtWidgets.QTabWidget();layout.addWidget(self.tabs,1)
        self.map=pg.PlotWidget();self.map.setAspectLocked(True);self.map.setLabel('bottom','X',units='m');self.map.setLabel('left','Y',units='m')
        self.base_image=pg.ImageItem(axisOrder='row-major');self.water_image=pg.ImageItem(axisOrder='row-major')
        self.map.addItem(self.base_image);self.map.addItem(self.water_image)
        self.route=self.map.plot(pen=pg.mkPen('#ffbf62',width=2));self.tabs.addTab(self.map,'Регион и затопление')
        self.profile=pg.PlotWidget();self.profile.setLabel('bottom','Время сценария',units='day');self.profile.setLabel('left','Уровень MSL',units='m')
        self.line=self.profile.plot(pen='#63c9ff');self.tabs.addTab(self.profile,'История уровня')
        actions=QtWidgets.QHBoxLayout();layout.addLayout(actions)
        self.instant=QtWidgets.QPushButton('Мгновенный прокоп → открыть канал');self.instant.clicked.connect(self.open_channel)
        self.close_gate=QtWidgets.QPushButton('Перекрыть приток');self.close_gate.clicked.connect(self.shut_channel)
        self.play=QtWidgets.QPushButton('▶ Расчёт воды');self.play.clicked.connect(self.toggle)
        self.speed=QtWidgets.QComboBox()
        for rate in (1,10,30,100,365):self.speed.addItem(f'{rate} суток/с',rate)
        self.speed.setCurrentIndex(2)
        self.day=QtWidgets.QPushButton('+ сутки');self.day.clicked.connect(lambda:self.advance(86400.))
        self.year=QtWidgets.QPushButton('+ год');self.year.clicked.connect(lambda:self.request_days(365.))
        self.reset=QtWidgets.QPushButton('Новый сухой сценарий');self.reset.clicked.connect(self.new_scenario)
        for w in (self.instant,self.close_gate,self.play,self.speed,self.day,self.year,self.reset):actions.addWidget(w)
        grid=QtWidgets.QGridLayout();layout.addLayout(grid);self.inputs={}
        specs=[('Ширина дна','width_m',1,10000,' м',1),('Глубина ниже моря','depth_m',.01,1000,' м',2),
            ('Маннинг n','roughness',.005,.3,'',4),('Открытие','opening',0,1,'',2),
            ('Лимит расхода','flow_limit_m3_s',0,1e6,' м³/с',2),('Среднее испарение','evaporation_mm_day',0,100,' мм/сут',2),
            ('Сезонная доля','season_amplitude',0,1,'',2),('Начальный день года','start_day',1,365,'',0)]
        for i,(label,name,lo,hi,suffix,decimals) in enumerate(specs):
            spin=QtWidgets.QDoubleSpinBox();spin.setRange(lo,hi);spin.setDecimals(decimals);spin.setSuffix(suffix)
            grid.addWidget(QtWidgets.QLabel(label),2*(i//4),i%4);grid.addWidget(spin,2*(i//4)+1,i%4)
            self.inputs[name]=spin
        self.status=QtWidgets.QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        self.error=QtWidgets.QLabel();self.error.setWordWrap(True);layout.addWidget(self.error)
        self.error.setText('Испарение — условная сезонная модель, не измеренный климатический ряд Каттары. '
            'Если приток перекрыт, испарение продолжает понижать уровень.')
        self.rebuild=QtWidgets.QPushButton('Обновить объём и площадь по текущему рельефу')
        self.rebuild.clicked.connect(self.rebuild_basin)
        self.save_button=QtWidgets.QPushButton('Сохранить регион и воду…');self.save_button.clicked.connect(self.save_region)
        storage=QtWidgets.QHBoxLayout();storage.addWidget(self.rebuild);storage.addWidget(self.save_button);layout.addLayout(storage)
        self.model=self.region.flooding
        if self.model is None:self.new_scenario()
        self._different_route=self.model.project_id!=self.project.identity
        if self._different_route:self.error.setText('Сохранённый расчёт относится к другому маршруту. Кнопка «Новый сухой сценарий» начнёт расчёт по выбранному проекту.')
        elif getattr(self.region,'_flooding_basis',None)!=self.basis_key():self.rebuild_basin()
        self.load_inputs();self.make_map()
        for spin in self.inputs.values():spin.valueChanged.connect(self.apply_inputs)
        self.host.running=False;self.host.credit=0.;self.host.play.setText('▶ Start')
        self.timer=QtCore.QTimer(self);self.timer.setInterval(30);self.timer.timeout.connect(self.tick);self.timer.start()
        self.finished.connect(self.finish);self.refresh()

    def view(self):
        land=self.region
        if land is getattr(self.host,'landscape',None) and self.host.config.georeference:
            view=land.snapshot();s=self.host.sim
            view.commit_window(s.config.georeference['origin_m'],s.terrain.height,cell_size=s.terrain.cell,soil=s.terrain.soil)
            return view
        return land

    def basis_key(self):
        land=self.region;key=(id(land.raster),land._tile_cache.writes,len(land.tiles),self.project.identity)
        if land is getattr(self.host,'landscape',None) and self.host.config.georeference:
            t=self.host.sim.terrain;key+=(*self.host.config.georeference['origin_m'],t.cell,t.height.shape[0],t.revision)
        return key

    def new_scenario(self):
        try:
            self.model=create_experiment(self.view(),self.project,latitude=self.host.config.base.latitude_deg)
            self.region.flooding=self.model;self._different_route=False;self._topography_error=False
            self.error.setText('Гидросценарий: синий слой — вода в едином резервуаре, оранжевая линия — ось условно готового канала.')
            self.region._flooding_basis=self.basis_key()
            self.running=False;self.credit=0.;self._draw_key=None
            if hasattr(self,'inputs'):self.load_inputs()
            if hasattr(self,'timer'):self.make_map();self.refresh()
        except (ValueError,OSError) as error:
            self.running=False;self.credit=0.;self._topography_error=True
            if hasattr(self,'error'):self.error.setText(str(error))
            if getattr(self,'model',None) is None:raise

    def rebuild_basin(self):
        try:
            curve=BasinCurve.from_landscape(self.view(),self.project.points[-1]);m=self.model
            excess=max(0.,m.volume_m3-curve.capacity);m.volume_m3-=excess;m.spilled_m3+=excess;m.curve=curve
            m.validate();self._draw_key=None
            self._topography_error=False
            self.region._flooding_basis=self.basis_key()
            if hasattr(self,'timer'):self.make_map();self.refresh()
        except (ValueError,OSError) as error:
            self.running=False;self.credit=0.;self._topography_error=True;self.error.setText(str(error))

    def save_region(self):
        try:
            path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Сохранить регион и гидросценарий','flooding.atsregion','ATS landscape (*.atsregion)')
            if not path:return
            if self.region is self.host.landscape:self.host.commit_landscape()
            self.region.save(path);self.error.setText('Регион, вода, настройки и история сохранены. Открытие: «Ландшафт / импорт DEM → Открыть ландшафт».')
        except (ValueError,OSError) as error:self.error.setText(str(error)+' Для сохранения работающей техники используйте .atsstate в меню «Файл».')

    def load_inputs(self):
        for name,spin in self.inputs.items():
            spin.blockSignals(True);spin.setValue(getattr(self.model.config,name));spin.blockSignals(False)

    def apply_inputs(self):
        for name,spin in self.inputs.items():setattr(self.model.config,name,spin.value())
        self.model.config.validate();self.refresh()

    def make_map(self):
        r=self.region.raster;rx,ry=r.resolution_m;w,h=r.extent_m
        step=max(1,math.ceil(max(r.height.shape)/1024));self._map_step=step
        data=r.height[::step,::step]
        from .channel import ocean_nodes
        cmap=pg.ColorMap([0,.5,1],['#967d55','#b7a47a','#f5e5bc'])
        rgba=cmap.map(np.clip((np.nan_to_num(data)+150)/300,0,1),mode='byte')
        rgba[ocean_nodes(r)[::step,::step]]=[25,65,94,255]
        rgba[~np.isfinite(data)]=[80,80,80,255]
        self.base_image.setImage(rgba)
        rect=QtCore.QRectF(-rx/2,-ry/2,(data.shape[1])*rx*step,(data.shape[0])*ry*step)
        self.base_image.setRect(rect);self._map_rect=rect
        self.map.setRange(xRange=(0,w),yRange=(0,h),padding=.01)
        self.route.setData(np.asarray([self.project.coast,*self.project.points]))
        self._water_bottom=np.full(data.shape,np.nan)
        indices=self.model.curve.indices;yy,xx=np.divmod(indices,r.height.shape[1]);selected=(yy%step==0)&(xx%step==0)
        self._water_bottom[yy[selected]//step,xx[selected]//step]=self.model.curve.bottom[selected]
        self._draw_key=None

    def open_channel(self):
        self.model.channel_open=True;self.refresh()

    def shut_channel(self):
        self.model.channel_open=False;self.refresh()

    def toggle(self):
        self.running=not (self.running or self.credit>0);self.last_wall=time.perf_counter();self.credit=0.;self.refresh()

    def request_days(self,days):
        self.credit+=days*86400.;self.refresh()

    def advance(self,seconds):
        self.model.advance(seconds);self.refresh()

    def tick(self):
        now=time.perf_counter();elapsed=min(.25,now-self.last_wall);self.last_wall=now
        if self.running:self.credit=min(365*86400.,self.credit+elapsed*self.speed.currentData()*86400)
        deadline=time.perf_counter()+.025
        while self.credit>1e-8 and time.perf_counter()<deadline:
            dt=min(self.credit,86400.);self.model.advance(dt);self.credit-=dt
        if now-self.last_draw>=.25:self.refresh();self.last_draw=now

    def refresh(self):
        m=getattr(self,'model',None)
        if m is None:return
        r=m.report();q,e=m.current_rates()
        level_label=f'уровень {m.level:+.3f} м MSL' if m.volume_m3>0 else f'сухо · дно {m.level:+.3f} м MSL'
        self.status.setText(f"Время сценария: {m.elapsed_seconds/86400:,.2f} суток · {level_label} · объём {m.volume_m3/1e9:.6f} км³\n"
            f"Затоплено {m.area/1e6:,.3f} км² · приток {q:,.3f} м³/с · испарение {e:,.3f} м³/с · баланс {m.balance_error:.3g} м³\n"
            f"Поступило {m.inflow_m3/1e9:.6f} км³ · испарилось {m.evaporated_m3/1e9:.6f} км³ · перелив {m.spilled_m3/1e9:.6f} км³ · длина канала {m.config.length_m/1000:.2f} км")
        different=getattr(self,'_different_route',False);invalid=getattr(self,'_topography_error',False)
        self.instant.setEnabled(not different and not invalid and not m.channel_open);self.close_gate.setEnabled(not different and not invalid and m.channel_open)
        for widget in (self.play,self.day,self.year,*self.inputs.values()):widget.setEnabled(not different and not invalid)
        self.rebuild.setEnabled(not different)
        self.play.setText('⏸ Пауза воды' if self.running or self.credit>0 else '▶ Расчёт воды')
        key=(id(m.curve),m.volume_m3)
        if self._draw_key!=key and hasattr(self,'_water_bottom') and self.tabs.currentWidget() is self.map:
            depth=np.where(np.isfinite(self._water_bottom),np.maximum(0,m.level-self._water_bottom),0) if m.volume_m3>0 else np.zeros_like(self._water_bottom)
            rgba=np.zeros((*depth.shape,4),dtype=np.uint8);rgba[:,:,0]=25;rgba[:,:,1]=115;rgba[:,:,2]=220;rgba[:,:,3]=np.where(depth>0,210,0)
            if m.volume_m3>0:
                lower=m.curve.area_at(np.nextafter(m.level,-math.inf));upper=m.curve.area_at(m.level)
                if upper>lower:
                    fraction=float(np.clip((m.area-lower)/(upper-lower),0,1))
                    rgba[:,:,3]=np.where(self._water_bottom==m.level,int(210*fraction),rgba[:,:,3])
            self.water_image.setImage(rgba);self.water_image.setRect(self._map_rect);self._draw_key=key
        rows=m.history
        self.line.setData([r['seconds']/86400 for r in rows]+[m.elapsed_seconds/86400],[r['level_msl'] for r in rows]+[m.level])

    def finish(self,_):
        if getattr(self,'_finished',False):return
        self._finished=True
        self.timer.stop();self.running=False;self.credit=0.
        self.host.running=self._resume;self.host.credit=0.;self.host.last_wall=time.perf_counter()
        self.host.play.setText('⏸ Pause' if self._resume else '▶ Start')

    def done(self,result):
        self.finish(result);super().done(result)

    def closeEvent(self,event):
        self.finish(0);super().closeEvent(event)
