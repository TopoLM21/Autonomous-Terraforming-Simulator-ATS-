"""Regional DEM overview, offline import and a bounded local work window."""
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
from .landscape import Landscape,load_dem,qattara_dem,workspace_config,corridor_profile


class LandscapeDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window=window;self.region=getattr(window,'landscape',None) or Landscape(qattara_dem())
        self.points=[];self.setWindowTitle('Реальный ландшафт · офлайн DEM · высоты MSL');self.resize(1150,850)
        layout=QtWidgets.QVBoxLayout(self)
        buttons=QtWidgets.QHBoxLayout()
        for text,handler in [('Каттара / NOAA',self.use_qattara),('Импорт DEM…',self.import_dem),('Открыть ландшафт…',self.load_region),('Сохранить ландшафт…',self.save_region)]:
            button=QtWidgets.QPushButton(text);button.clicked.connect(handler);buttons.addWidget(button)
        self.crs=QtWidgets.QComboBox();self.crs.addItem('ASC: WGS84 (градусы)','EPSG:4326');self.crs.addItem('ASC: локальные метры','LOCAL_METRES')
        self.msl=QtWidgets.QCheckBox('Импорт: метры над MSL');buttons.addWidget(self.crs);buttons.addWidget(self.msl)
        layout.addLayout(buttons)
        self.info=QtWidgets.QLabel();self.info.setWordWrap(True);layout.addWidget(self.info)
        self.map=pg.PlotWidget();self.map.setAspectLocked(True);self.map.setLabel('bottom','X от западного края',units='m');self.map.setLabel('left','Y от южного края',units='m')
        self.image=pg.ImageItem(axisOrder='row-major');self.map.addItem(self.image)
        self.marker=pg.PlotDataItem(pen=pg.mkPen('#ffca63',width=2),symbol='+',symbolSize=12);self.map.addItem(self.marker)
        self.route=pg.PlotDataItem(pen=pg.mkPen('#ff8ecb',width=2),symbol='o',symbolSize=5);self.map.addItem(self.route)
        layout.addWidget(self.map,4);self.map.scene().sigMouseClicked.connect(self.map_clicked)
        form=QtWidgets.QHBoxLayout()
        self.x=QtWidgets.QDoubleSpinBox();self.y=QtWidgets.QDoubleSpinBox()
        for s in (self.x,self.y):s.setRange(0,100_000_000);s.setDecimals(2);s.setSingleStep(64);s.setSuffix(' м')
        self.side=QtWidgets.QComboBox()
        for size in (32,64,128,256):self.side.addItem(f'{size} м',size)
        self.side.setCurrentIndex(1)
        self.route_mode=QtWidgets.QCheckBox('Рисовать коридор (ЛКМ)')
        self.target=QtWidgets.QDoubleSpinBox();self.target.setRange(-11000,10000);self.target.setDecimals(2);self.target.setValue(-5);self.target.setSuffix(' м MSL')
        self.width=QtWidgets.QDoubleSpinBox();self.width.setRange(1,1000);self.width.setValue(20);self.width.setSuffix(' м')
        for item in (QtWidgets.QLabel('Начало X/Y'),self.x,self.y,self.side,self.route_mode,QtWidgets.QLabel('Дно'),self.target,QtWidgets.QLabel('Ширина'),self.width):form.addWidget(item)
        layout.addLayout(form)
        self.anchor=QtWidgets.QLabel();layout.addWidget(self.anchor)
        actions=QtWidgets.QHBoxLayout()
        for text,handler in [('Профиль коридора',self.preview_corridor),('Пример: море → Каттара',self.example_corridor),('Очистить линию',self.clear_route),('Открыть участок · новый локальный запуск',self.open_workspace)]:
            button=QtWidgets.QPushButton(text);button.clicked.connect(handler);actions.addWidget(button)
        layout.addLayout(actions)
        self.profile=pg.PlotWidget();self.profile.setLabel('left','Высота MSL',units='m');self.profile.setLabel('bottom','Вдоль коридора',units='m')
        self.ground=self.profile.plot(pen='#efce8b');self.design=self.profile.plot(pen=pg.mkPen('#59cde4',style=QtCore.Qt.PenStyle.DashLine));layout.addWidget(self.profile,1)
        self.result=QtWidgets.QLabel('ЛКМ выбирает участок; в режиме коридора добавляет вершины линии. Оценка коридора не запускает земляные работы и не выбирает оптимальный маршрут.');self.result.setWordWrap(True);layout.addWidget(self.result)
        for s in (self.x,self.y):s.valueChanged.connect(self.update_selection)
        self.side.currentIndexChanged.connect(self.update_selection)
        self.show_region()

    def show_region(self):
        r=self.region.raster;h=r.height
        finite=h[np.isfinite(h)];lo,hi=min(-150.,float(finite.min())),max(150.,float(finite.max()))
        colors=pg.ColorMap([0.,max(.001,min(.999,-lo/(hi-lo))),1.],['#103b63','#aaab79','#fff0c1'])
        self.image.setImage(h,levels=(lo,hi),lut=colors.getLookupTable())
        # Pixel centres correspond to the source nodes; extend half a DEM cell.
        rx,ry=r.resolution_m;w,t=r.extent_m
        self.image.setRect(QtCore.QRectF(-rx/2,-ry/2,w+rx,t+ry))
        self.map.setRange(xRange=(0,w),yRange=(0,t),padding=.01)
        active=self.window.config.georeference
        origin=active['origin_m'] if getattr(self.window,'landscape',None) is self.region and active else (w/2,t/2)
        cell=self.region.cell_size
        self.x.setValue(round(origin[0]/cell)*cell);self.y.setValue(round(origin[1]/cell)*cell);self.clear_route(forget=False);self.update_selection()
        if self.region.corridors:
            route=self.region.corridors[-1]
            self.points=route['points'];self.target.setValue(route['target_msl']);self.width.setValue(route['width_m'])
            self.route.setData(np.array(self.points));self.preview_corridor()
        s=self.region.report()
        self.info.setText(f"{r.metadata.get('source','DEM')} · {h.shape[1]}×{h.shape[0]} · шаг {rx:.0f}×{ry:.0f} м · MSL: {finite.min():.0f}…{finite.max():.0f} м\nИзменённых блоков: {s['modified_tiles']} · подробные данные: {s['fine_storage_bytes']/1024:.1f} КиБ. Мелкая сетка интерполирует DEM, точность источника остаётся прежней. Жёлтый участок намного меньше региона; увеличьте масштаб колесом.")

    def update_selection(self):
        x,y=self.x.value(),self.y.value();side=self.side.currentData()
        self.marker.setData([x,x+side,x+side,x,x],[y,y,y+side,y+side,y])
        a,b=self.region.raster.to_source(x,y)
        self.anchor.setText(f"Привязка начала: X/долгота {a:.6f}, Y/широта {b:.6f} · {self.region.raster.metadata['crs']} · 0 м = уровень моря")

    def map_clicked(self,event):
        if event.button()!=QtCore.Qt.MouseButton.LeftButton or not self.map.sceneBoundingRect().contains(event.scenePos()):return
        p=self.map.getViewBox().mapSceneToView(event.scenePos());point=[p.x(),p.y()]
        if self.route_mode.isChecked():
            if len(self.points)>=4096:return
            self.points.append(point);self.route.setData(np.array(self.points));return
        c=self.region.cell_size;side=self.side.currentData()
        self.x.setValue(max(0,round((point[0]-side/2)/c)*c));self.y.setValue(max(0,round((point[1]-side/2)/c)*c))

    def clear_route(self,checked=False,*,forget=True):
        self.points=[];self.route.clear();self.ground.clear();self.design.clear()
        if forget:self.region.corridors=[]

    def preview_corridor(self):
        try:
            p=corridor_profile(self.region.raster,self.points,self.target.value(),self.width.value())
            self.region.corridors=[dict(points=[list(v) for v in self.points],target_msl=self.target.value(),width_m=self.width.value())]
            self.ground.setData(p['distance_m'],p['ground_msl'])
            self.design.setData(p['distance_m'],[p['target_msl']]*len(p['distance_m']))
            self.result.setText(f"Линия: {p['length_m']/1000:.1f} км · ориентировочная выемка {p['estimated_cut_m3']/1e6:.2f} млн м³, дно {p['target_msl']:+.2f} м MSL. Откос 35°, поперечный рельеф принят постоянным. Это обзорная оценка по исходному DEM.")
        except (ValueError,TypeError) as error:self.result.setText(str(error))

    def example_corridor(self):
        r=self.region.raster
        if 'ETOPO1' not in r.metadata.get('source','') or r.metadata['crs']!='EPSG:4326':
            self.result.setText('Этот пример предназначен для встроенного DEM Каттары.');return
        # An illustrative straight line, not a proposed/surveyed canal alignment.
        self.points=[[float(v) for v in r.to_metres(lon,lat)] for lon,lat in [(28.3,31.2),(27.5,29.5)]]
        self.route.setData(np.array(self.points));self.preview_corridor()

    def use_qattara(self):
        self.region=Landscape(qattara_dem());self.show_region()

    def import_dem(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Импорт локального DEM','','DEM (*.npz *.asc *.tif *.tiff)')
        if not path:return
        try:
            self.region=Landscape(load_dem(path,crs=self.crs.currentData(),vertical_datum='MSL' if self.msl.isChecked() else None))
            self.show_region()
        except (OSError,ValueError,KeyError,TypeError) as error:self.result.setText(str(error))

    def load_region(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Открыть ландшафт','','ATS landscape (*.atsregion)')
        if not path:return
        try:self.region=Landscape.load(path);self.show_region()
        except (OSError,ValueError,KeyError,TypeError) as error:self.result.setText(str(error))

    def save_region(self):
        try:
            if getattr(self.window,'landscape',None) is self.region:self.window.commit_landscape()
            path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Сохранить DEM и изменения','landscape.atsregion','ATS landscape (*.atsregion)')
            if path:self.region.save(path);self.show_region();self.result.setText('Ландшафт сохранён: исходный DEM и изменённые блоки. Конфигурация/статистика локального запуска сохраняется отдельно.')
        except (OSError,ValueError) as error:self.result.setText(str(error))

    def open_workspace(self):
        try:
            # Validate the new field before touching the current run.
            cell=self.region.cell_size
            origin=[round(v/cell)*cell for v in (self.x.value(),self.y.value())]
            c=workspace_config(self.region,self.window.config,origin,self.side.currentData())
            self.window.commit_landscape()
            # The same region may have received edits from the field just closed.
            c=workspace_config(self.region,c,origin,self.side.currentData())
            self.window.replace_simulation(c)
            self.window.landscape=self.region;self.window.plan_editor.height.setValue(self.target.value())
            self.window.brush_button.setChecked(True);self.accept()
        except (ValueError,OSError) as error:self.result.setText(str(error))
