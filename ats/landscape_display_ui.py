"""Regional MSL palette, horizontal levels and independent read-only profiles."""
import threading
import math
import numpy as np
from PySide6 import QtCore,QtWidgets
import pyqtgraph as pg
from .channel import ocean_nodes
from .landscape_display import overview,grid_key,marine_samples,colors,level_segments,profile,samples


class _MarineSignal(QtCore.QObject):
    ready=QtCore.Signal(int,object,object,str)


def _read_marine(raster,generation,signal,cancel):
    try:mask=ocean_nodes(raster);error=''
    except Exception as failure:mask=None;error=str(failure)
    if not cancel.is_set():
        try:signal.ready.emit(generation,raster,mask,error)
        except RuntimeError:pass  # the owning dialog has already been destroyed


class SectionWindow(QtWidgets.QDialog):
    def __init__(self,display):
        super().__init__(display.owner);self.display=display;self.setWindowTitle('Сечения ландшафта · отметки MSL')
        layout=QtWidgets.QVBoxLayout(self);tools=QtWidgets.QHBoxLayout();layout.addLayout(tools)
        self.direction=QtWidgets.QComboBox()
        for label,value in [('Вдоль нарисованной линии','route'),('Запад → восток · Y фиксирован','x'),('Юг → север · X фиксирован','y')]:self.direction.addItem(label,value)
        self.position=QtWidgets.QDoubleSpinBox();self.position.setDecimals(2);self.position.setSuffix(' м')
        self.slider=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal);self.slider.setRange(0,1000)
        for w in (self.direction,self.position,self.slider):tools.addWidget(w)
        self.plot=pg.PlotWidget();self.plot.setLabel('left','Высота над уровнем моря',units='m');self.plot.showGrid(x=True,y=True,alpha=.2)
        self.ground=self.plot.plot(pen=pg.mkPen('#e9c794',width=2),connect='finite')
        self.sea=pg.InfiniteLine(pos=0,angle=0,pen=pg.mkPen('#76d8ff',width=2));self.plot.addItem(self.sea)
        self.level=pg.InfiniteLine(pos=0,angle=0,pen=pg.mkPen('#9be7ab',style=QtCore.Qt.PenStyle.DashLine));self.plot.addItem(self.level)
        layout.addWidget(self.plot,1);self.info=QtWidgets.QLabel();self.info.setWordWrap(True);layout.addWidget(self.info)
        self.direction.currentIndexChanged.connect(self.mode_changed);self.position.valueChanged.connect(self.position_changed)
        self.slider.valueChanged.connect(self.slider_changed)
        self.mode_changed()
        from .ui_windows import fit_window
        fit_window(self,1050,650)

    def mode_changed(self):
        mode=self.direction.currentData();r=self.display.raster
        maximum=r.extent_m[1 if mode=='x' else 0] if r is not None else 1.
        self.position.blockSignals(True);self.position.setRange(0,maximum);self.position.setSingleStep(maximum/1000)
        self.position.setValue(maximum/2);self.position.blockSignals(False)
        self.position.setEnabled(mode!='route');self.slider.setEnabled(mode!='route')
        self.position.setVisible(mode!='route');self.slider.setVisible(mode!='route');self.position_changed()

    def position_changed(self):
        self.slider.blockSignals(True);self.slider.setValue(round(self.position.value()/max(self.position.maximum(),1e-9)*1000));self.slider.blockSignals(False)
        self.refresh()

    def slider_changed(self,value):self.position.setValue(self.position.maximum()*value/1000)

    def refresh(self):
        d=self.display;r=d.raster
        if r is None:return
        mode=self.direction.currentData();x,y=d.owner.route.getData()
        points=np.column_stack((x,y)) if x is not None else None
        axis,z=profile(r,mode,self.position.value(),points=points)
        self.ground.setData(axis,z,connect='finite');self.sea.setVisible(d.zero_line.isChecked())
        self.level.setValue(d.level.value());self.level.setVisible(d.mode.currentData() in ('below','band'))
        self.plot.setLabel('bottom','Вдоль линии' if mode=='route' else 'X от запада' if mode=='x' else 'Y от юга',units='m')
        self.plot.enableAutoRange()
        finite=z[np.isfinite(z)]
        description=(f'Отметки {finite.min():+.2f}…{finite.max():+.2f} м MSL.' if finite.size else 'Нет высот: нарисуйте линию или выберите сечение; NoData показано разрывом.')
        rx,ry=r.resolution_m
        self.info.setText(description+f' Голубая линия — 0 м (уровень моря). Исходный DEM: {rx:g} × {ry:g} м. '
            'Зелёный пунктир — выбранная отметка среза. Shift+ЛКМ на карте переносит сечение X/Y. Это исходный рельеф, без проектного прокопа.')
        d.draw_section()

    def closeEvent(self,event):
        self.display.section_line.clear();super().closeEvent(event)


class ElevationDisplay(QtCore.QObject):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.raster=None;self.marine=None;self.marine_error='';self.generation=0
        self.cancel=threading.Event();self.closed=False;self.section_window=None;self._grid=None;self._grid_key=None;self._contour_key=None
        self.signal=_MarineSignal(self);self.signal.ready.connect(self.marine_ready)
        self.range_timer=QtCore.QTimer(self);self.range_timer.setSingleShot(True);self.range_timer.setInterval(100)
        self.range_timer.timeout.connect(self.draw)
        owner.map.getViewBox().sigRangeChanged.connect(lambda *_:self.range_timer.start())
        owner.finished.connect(self.stop);owner.destroyed.connect(self.stop)
        self.toolbar=QtWidgets.QWidget();layout=QtWidgets.QVBoxLayout(self.toolbar);layout.setContentsMargins(0,0,0,0)
        selection=QtWidgets.QHBoxLayout();layout.addLayout(selection)
        self.mode=QtWidgets.QComboBox()
        for label,value in [('Высоты относительно моря','msl'),('Общая шкала высот','height'),('Срез · ниже отметки','below'),('Слой · диапазон высот','band')]:self.mode.addItem(label,value)
        selection.addWidget(self.mode,1)
        self.parameters_button=QtWidgets.QPushButton('Параметры');self.parameters_button.setCheckable(True);self.parameters_button.setChecked(True)
        self.parameters_button.toggled.connect(owner.controls_scroll.setVisible);selection.addWidget(self.parameters_button)
        top=QtWidgets.QHBoxLayout();layout.addLayout(top)
        self.level_tools=QtWidgets.QWidget();level_layout=QtWidgets.QHBoxLayout(self.level_tools);level_layout.setContentsMargins(0,0,0,0)
        self.level=QtWidgets.QDoubleSpinBox();self.level.setDecimals(2);self.level.setSuffix(' м MSL');self.level.setRange(-11000,10000);self.level.setSingleStep(1.)
        self.slider=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal);self.slider.setRange(0,1000);self.slider.setMinimumWidth(80)
        self.zero=QtWidgets.QPushButton('0 м');self.zero.clicked.connect(lambda:self.level.setValue(0.))
        self.thickness=QtWidgets.QDoubleSpinBox();self.thickness.setRange(.01,22000);self.thickness.setValue(10.);self.thickness.setSuffix(' м слой');self.thickness.setToolTip('Полная толщина слоя, симметрично выбранной отметке')
        for w in (self.level,self.slider,self.zero):level_layout.addWidget(w)
        layout.addWidget(self.level_tools)
        layout.addWidget(self.thickness)
        self.zero_line=QtWidgets.QCheckBox('Линия 0 м');self.zero_line.setChecked(True);top.addWidget(self.zero_line)
        self.sections_button=QtWidgets.QPushButton('Сечения…');self.sections_button.clicked.connect(self.show_sections);top.addWidget(self.sections_button)
        self.overview_button=QtWidgets.QPushButton('Обзор');self.overview_button.setToolTip('Показать весь исходный DEM')
        self.overview_button.clicked.connect(self.fit_view);top.addWidget(self.overview_button)
        self.legend=QtWidgets.QLabel();self.legend.setWordWrap(True);layout.addWidget(self.legend)
        self.probe=QtWidgets.QLabel('Наведите курсор на карту: здесь появится отметка относительно моря.');self.probe.setWordWrap(True);layout.addWidget(self.probe)
        self.contour=pg.PlotDataItem(pen=pg.mkPen('#f3f7ff',width=1.5),connect='pairs');self.contour.setZValue(3);owner.map.addItem(self.contour)
        self.slice_contour=pg.PlotDataItem(pen=pg.mkPen('#9be7ab',width=2),connect='pairs');self.slice_contour.setZValue(4);owner.map.addItem(self.slice_contour)
        self.section_line=pg.PlotDataItem(pen=pg.mkPen('#63e6ef',width=2,style=QtCore.Qt.PenStyle.DashLine));self.section_line.setZValue(5);owner.map.addItem(self.section_line)
        owner.route.setZValue(6);owner.endpoints.setZValue(7);owner.marker.setZValue(8)
        self.profile_zero=pg.InfiniteLine(pos=0,angle=0,pen=pg.mkPen('#76d8ff',width=1));owner.profile.addItem(self.profile_zero)
        owner.route.sigPlotChanged.connect(self.route_changed)
        self.mouse=pg.SignalProxy(owner.map.scene().sigMouseMoved,rateLimit=20,slot=self.hover)
        self.mode.currentIndexChanged.connect(self.mode_changed);self.zero_line.toggled.connect(self.changed)
        self.level.valueChanged.connect(self.level_changed);self.slider.valueChanged.connect(self.slider_changed);self.thickness.valueChanged.connect(self.changed)
        state=getattr(owner.window,'landscape_display',{})
        if isinstance(state,dict):
            self.mode.setCurrentIndex(max(0,self.mode.findData(state.get('mode','msl'))))
            for name,default in (('level',0.),('thickness',10.)):
                value=state.get(name,default)
                if type(value) in (float,int) and math.isfinite(value):getattr(self,name).setValue(value)
            self.zero_line.setChecked(bool(state.get('zero_line',True)))
        self.mode_changed()

    def stop(self,*_):
        if self.closed:return
        self.closed=True;self.cancel.set();self.range_timer.stop()
        if self.section_window:self.section_window.close()

    def fit_view(self):
        if self.raster is None or self.closed:return
        width,height=self.raster.extent_m
        self.owner.map.setRange(xRange=(0,width),yRange=(0,height),padding=.01)

    def set_region(self,raster):
        self.cancel.set();self.cancel=threading.Event();self.generation+=1;self.raster=raster;self.marine=None;self.marine_error=''
        self._grid_key=None;self._contour_key=None
        lo=min(0.,float(np.nanmin(raster.height)));hi=max(0.,float(np.nanmax(raster.height)))
        if hi-lo<.01:hi=lo+.01
        self.level.blockSignals(True);self.level.setRange(lo,hi);self.level.blockSignals(False)
        cached=getattr(raster,'_ocean_nodes',None)
        if cached is not None:self.marine=cached
        elif raster.height.size<=262144:self.marine=ocean_nodes(raster)
        else:threading.Thread(target=_read_marine,args=(raster,self.generation,self.signal,self.cancel),daemon=True,name='ats-overview-sea').start()
        if self.section_window:self.section_window.mode_changed()
        self.level_changed();self.draw()

    @QtCore.Slot(int,object,object,str)
    def marine_ready(self,generation,raster,mask,error):
        if self.closed or generation!=self.generation or raster is not self.raster:return
        self.marine=mask;self.marine_error=error;self.draw()

    def persist(self):
        self.owner.window.landscape_display=dict(mode=self.mode.currentData(),level=self.level.value(),thickness=self.thickness.value(),zero_line=self.zero_line.isChecked())

    def mode_changed(self,*_):
        mode=self.mode.currentData();self.level_tools.setVisible(mode in ('below','band'));self.thickness.setVisible(mode=='band');self.changed()

    def level_changed(self,*_):
        self.slider.blockSignals(True)
        fraction=(self.level.value()-self.level.minimum())/max(1e-9,self.level.maximum()-self.level.minimum())
        self.slider.setValue(round(fraction*1000));self.slider.blockSignals(False);self.changed()

    def slider_changed(self,value):self.level.setValue(self.level.minimum()+(self.level.maximum()-self.level.minimum())*value/1000)

    def changed(self,*_):
        self.persist();self.range_timer.start(30)
        if self.section_window and self.section_window.isVisible():self.section_window.refresh()

    def draw(self,*_):
        if self.raster is None or self.closed:return
        key=grid_key(self.raster,self.owner.map.getViewBox().viewRange())
        if self._grid_key!=key:
            self._grid=overview(self.raster,key);self._grid_key=key;self._contour_key=None
        data,frame=self._grid;x0,y0,dx,dy=frame
        sea=marine_samples(self.raster,self.marine,data,frame) if self.marine is not None else None
        mode=self.mode.currentData();self.owner.image.setImage(colors(data,sea,mode=mode,level=self.level.value(),thickness=self.thickness.value()),autoLevels=False)
        self.owner.image.setRect(QtCore.QRectF(x0-dx/2,y0-dy/2,data.shape[1]*dx,data.shape[0]*dy))
        contour_key=(key,self.level.value(),self.thickness.value(),mode)
        if self._contour_key!=contour_key:
            points=level_segments(data,0.,frame);self.contour.setData(points,connect='pairs')
            levels=([self.level.value()-self.thickness.value()/2,self.level.value()+self.thickness.value()/2] if mode=='band' else [self.level.value()])
            parts=[level_segments(data,h,frame) for h in levels] if mode in ('below','band') else []
            self.slice_contour.setData(np.concatenate(parts) if parts else np.empty((0,2)),connect='pairs');self._contour_key=contour_key
        self.contour.setVisible(self.zero_line.isChecked());self.profile_zero.setVisible(self.zero_line.isChecked())
        text='<span style="color:#f3f7ff">━━</span> 0 м MSL — уровень моря. '
        if mode=='height':text+='Общая шкала от минимальной к максимальной высоте; цвет не определяет наличие воды. '
        elif mode=='msl':
            text+=('<span style="color:#ae8dc7">■</span> отметки ниже 0; тип области уточняется. ' if self.marine is None else
                '<span style="color:#2783c5">■</span> море · <span style="color:#ae8dc7">■</span> сухая впадина ниже 0. ')
            text+='<span style="color:#ddc389">■</span> суша выше 0. '
        else:text+='<span style="color:#2783c5">■</span> море · серый — остальной рельеф. '
        if mode=='below':text+=f'<span style="color:#62c885">■</span> рельеф ≤ {self.level.value():+.2f} м. '
        elif mode=='band':text+=f'<span style="color:#62c885">■</span> слой {self.level.value()-self.thickness.value()/2:+.2f}…{self.level.value()+self.thickness.value()/2:+.2f} м. '
        if mode in ('below','band'):text+='Срез по высоте, не расчёт затопления. '
        if self.marine is None:text+='Морская маска ещё вычисляется. ' if not self.marine_error else 'Ошибка морской маски: '+self.marine_error+'. '
        text+=f'Исходный обзорный DEM: {self.raster.resolution_m[0]:g} × {self.raster.resolution_m[1]:g} м; тёмно-серое — NoData.'
        self.legend.setText(text)

    def hover(self,events):
        if self.raster is None or self.closed:return
        pos=events[0]
        if not self.owner.map.sceneBoundingRect().contains(pos):return
        point=self.owner.map.getViewBox().mapSceneToView(pos);x,y=point.x(),point.y()
        h=float(samples(self.raster,x,y))
        if not math.isfinite(h):self.probe.setText(f'X {x:.1f} / Y {y:.1f} м · NoData или вне DEM');return
        category='выше моря' if h>0 else 'ниже моря' if h<0 else 'уровень моря'
        if self.marine is not None:
            sea=bool(marine_samples(self.raster,self.marine,np.array([[h,h],[h,h]]),(x,y,0.,0.))[0,0])
            if sea:category='морское дно'
            elif h<0:category='сухая впадина ниже моря'
        sx,sy=self.raster.to_source(x,y)
        self.probe.setText(f'X {x:,.1f} / Y {y:,.1f} м · {h:+.2f} м MSL · {category} · координаты DEM {sx:.6f}, {sy:.6f}')

    def show_sections(self):
        if self.section_window is None:self.section_window=SectionWindow(self)
        self.section_window.show();self.section_window.raise_();self.section_window.refresh()

    def route_changed(self,*_):
        if self.section_window and self.section_window.isVisible() and self.raster is self.owner.region.raster:self.section_window.refresh()

    def draw_section(self):
        window=self.section_window
        if window is None or not window.isVisible() or self.raster is None:return
        mode=window.direction.currentData();value=window.position.value();width,height=self.raster.extent_m
        points=[[0,value],[width,value]] if mode=='x' else [[value,0],[value,height]] if mode=='y' else []
        self.section_line.setData(np.asarray(points).reshape(-1,2))

    def shift_click(self,point,modifiers):
        window=self.section_window
        if not (window and window.isVisible() and modifiers&QtCore.Qt.KeyboardModifier.ShiftModifier):return False
        mode=window.direction.currentData()
        if mode=='route':return False
        window.position.setValue(point[1 if mode=='x' else 0]);return True
