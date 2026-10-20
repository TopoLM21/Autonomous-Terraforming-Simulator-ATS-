"""Regional DEM overview, offline import and a bounded local work window."""
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
from .landscape import Landscape,load_dem,qattara_dem,workspace_config,corridor_profile
from .channel import ChannelProject


class LandscapeDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window=window;self.region=getattr(window,'landscape',None) or Landscape(qattara_dem())
        self.points=[];self.project=None;self.setWindowTitle('Маршрут и проект канала · офлайн DEM');self.resize(1250,950)
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
        self.endpoints=pg.ScatterPlotItem(size=14,symbol='o');self.map.addItem(self.endpoints)
        layout.addWidget(self.map,4);self.map.scene().sigMouseClicked.connect(self.map_clicked)
        form=QtWidgets.QHBoxLayout()
        self.x=QtWidgets.QDoubleSpinBox();self.y=QtWidgets.QDoubleSpinBox()
        for s in (self.x,self.y):s.setRange(0,100_000_000);s.setDecimals(2);s.setSingleStep(64);s.setSuffix(' м')
        self.side=QtWidgets.QComboBox()
        for size in (32,64,128,256,512,1024):self.side.addItem(f'{size} м',size)
        self.side.setCurrentIndex(1)
        self.route_mode=QtWidgets.QCheckBox('Рисовать маршрут (ЛКМ)');self.route_mode.setChecked(True)
        self.target=QtWidgets.QDoubleSpinBox();self.target.setRange(-11000,10000);self.target.setDecimals(2);self.target.setValue(-5);self.target.setSuffix(' м MSL')
        self.width=QtWidgets.QDoubleSpinBox();self.width.setRange(1,1000);self.width.setValue(20);self.width.setSuffix(' м')
        for item in (QtWidgets.QLabel('Окно X/Y'),self.x,self.y,self.side,self.route_mode):form.addWidget(item)
        layout.addLayout(form)
        channel=QtWidgets.QGridLayout()
        self.depth=QtWidgets.QDoubleSpinBox();self.depth.setRange(.01,500);self.depth.setValue(5);self.depth.setSuffix(' м ниже моря')
        self.offset=QtWidgets.QDoubleSpinBox();self.offset.setRange(.01,10000);self.offset.setValue(10);self.offset.setSuffix(' м')
        self.angle=QtWidgets.QDoubleSpinBox();self.angle.setRange(5,55);self.angle.setValue(35);self.angle.setSuffix('°')
        self.project_side=QtWidgets.QComboBox()
        for size in (32,64,128,256,512,1024):self.project_side.addItem(f'{size} м',size)
        self.project_side.setCurrentIndex(self.project_side.findData(512))
        self.spoil=QtWidgets.QDoubleSpinBox();self.spoil.setRange(.1,200);self.spoil.setValue(20);self.spoil.setSuffix(' м')
        self.section=QtWidgets.QSpinBox();self.section.setRange(1,1)
        for i,(label,widget) in enumerate([('Ширина дна',self.width),('Глубина дна',self.depth),('Отступ от берега',self.offset),('Откос',self.angle),('Карта рабочего участка',self.project_side),('Высота отвала',self.spoil),('Участок проекта',self.section)]):
            channel.addWidget(QtWidgets.QLabel(label),i//4*2,i%4);channel.addWidget(widget,i//4*2+1,i%4)
        layout.addLayout(channel)
        self.depth.valueChanged.connect(lambda v:self.target.setValue(-v))
        self.target.valueChanged.connect(lambda v:self.depth.setValue(-v) if v<0 else None)
        self.section.valueChanged.connect(self.select_section)
        self.anchor=QtWidgets.QLabel();layout.addWidget(self.anchor)
        actions=QtWidgets.QHBoxLayout()
        for text,handler in [('Профиль коридора',self.preview_corridor),('Пример: море → Каттара',self.example_corridor),('Очистить линию',self.clear_route),('Открыть участок · новый локальный запуск',self.open_workspace)]:
            button=QtWidgets.QPushButton(text);button.clicked.connect(handler);actions.addWidget(button)
        layout.addLayout(actions)
        launch=QtWidgets.QHBoxLayout()
        self.prepare_button=QtWidgets.QPushButton('Разметить канал');self.prepare_button.clicked.connect(self.prepare_channel)
        self.start_button=QtWidgets.QPushButton('Начать копание выбранного участка');self.start_button.clicked.connect(self.start_channel)
        self.next_button=QtWidgets.QPushButton('Следующий участок · новая расстановка');self.next_button.clicked.connect(self.next_section)
        for b in (self.prepare_button,self.start_button,self.next_button):launch.addWidget(b)
        layout.addLayout(launch)
        self.profile=pg.PlotWidget();self.profile.setLabel('left','Высота MSL',units='m');self.profile.setLabel('bottom','Вдоль коридора',units='m')
        self.ground=self.profile.plot(pen='#efce8b');self.design=self.profile.plot(pen=pg.mkPen('#59cde4',style=QtCore.Qt.PenStyle.DashLine));layout.addWidget(self.profile,1)
        self.result=QtWidgets.QLabel('ЛКМ выбирает участок; в режиме коридора добавляет вершины линии. Оценка коридора не запускает земляные работы и не выбирает оптимальный маршрут.');self.result.setWordWrap(True);layout.addWidget(self.result)
        self.project_info=QtWidgets.QLabel('Нарисуйте линию от морского берега в сухую впадину. «Разметить канал» установит береговой отступ и конец по уровню моря.');self.project_info.setWordWrap(True);layout.addWidget(self.project_info)
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
        if self.region.channel_project:
            try:
                self.project=ChannelProject.from_data(self.region.channel_project)
                p=self.project;self.points=p.raw_points
                self.width.setValue(p.width_m);self.depth.setValue(p.depth_m);self.offset.setValue(p.shore_offset_m)
                self.angle.setValue(p.angle_deg);self.spoil.setValue(p.spoil_height_m)
                self.project_side.setCurrentIndex(self.project_side.findData(p.side_m))
                self.show_project()
            except (ValueError,KeyError,TypeError) as error:self.result.setText(str(error))
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
        self.endpoints.clear();self.project=None
        self.project_info.setText('Нарисуйте маршрут от моря в сухую впадину и нажмите «Разметить канал».')
        if forget:self.region.corridors=[];self.region.channel_project=None

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
        self.prepare_channel()

    def parameters(self):
        return dict(width_m=self.width.value(),depth_m=self.depth.value(),shore_offset_m=self.offset.value(),
            angle_deg=self.angle.value(),side_m=self.project_side.currentData(),spoil_height_m=self.spoil.value())

    def prepare_channel(self):
        try:
            candidate=ChannelProject.create(self.region.raster,self.points,**self.parameters())
            if self.project and self.project.identity==candidate.identity:
                candidate=self.project
            self.project=candidate;self.region.channel_project=candidate.to_data();self.show_project()
            return True
        except (ValueError,TypeError) as error:self.result.setText(str(error));return False

    def show_project(self):
        p=self.project;r=self.region.raster
        self.section.blockSignals(True);self.section.setRange(1,p.section_count);self.section.setValue(p.active_section+1);self.section.blockSignals(False)
        self.route.setData(np.asarray(p.points))
        self.endpoints.setData([dict(pos=p.points[0],brush='#63e5b1'),dict(pos=p.points[-1],brush='#ffbd69')])
        profile=p.profile(r);self.ground.setData(profile['distance_m'],profile['ground_msl'])
        self.design.setData(profile['distance_m'],[-p.depth_m]*len(profile['distance_m']))
        start=r.to_source(*p.points[0]);end=r.to_source(*p.points[-1])
        start_z=float(r.sample(*p.points[0]));end_z=float(r.sample(*p.points[-1]))
        self.project_info.setText(f'Канал: {p.length/1000:.2f} км · дно шириной {p.width_m:g} м, отметка −{p.depth_m:g} м MSL · участков {p.section_count}, завершено {len(p.completed)}.\nНачало: {start[0]:.6f}, {start[1]:.6f}, {start_z:+.3f} м MSL, отступ по линии {p.shore_offset_m:g} м от берега DEM. Конец: {end[0]:.6f}, {end[1]:.6f}, {end_z:+.3f} м MSL — первый вход в выбранную сухую впадину.\nПлан выемки создаётся автоматически. После завершения участка сохраните прогресс и выберите следующий; новая расстановка начинает отдельный локальный запуск оборудования.')
        self.result.setText(f"Готовый проект: ориентировочная выемка {profile['estimated_cut_m3']/1e6:.2f} млн м³. Точность берега определяется DEM ({r.resolution_m[0]:.0f}×{r.resolution_m[1]:.0f} м).")
        self.select_section()

    def select_section(self):
        if self.project is None:return
        self.project.active_section=self.section.value()-1
        origin=self.project.origin(self.region.raster)
        self.x.setValue(origin[0]);self.y.setValue(origin[1]);self.side.setCurrentIndex(self.side.findData(self.project.side_m))
        self.region.channel_project=self.project.to_data()

    def start_channel(self):
        if not self.prepare_channel():return
        try:
            self.window.commit_landscape()
            if self.region.channel_project and self.region.channel_project.get('points')==self.project.points:
                # A finished field may just have contributed its production history.
                stored=ChannelProject.from_data(self.region.channel_project)
                if stored.identity==self.project.identity:self.project=stored
            self.project.active_section=self.section.value()-1
            c=self.project.configuration(self.region,self.window.config)
            self.window.replace_simulation(c)
            self.window.landscape=self.region;self.region.channel_project=self.project.to_data()
            self.window.plan_editor.height.setValue(-self.project.depth_m)
            if self.window.sim.state.value=='RUNNING':self.window.toggle_play()
            self.accept()
        except (ValueError,OSError,KeyError) as error:self.result.setText(str(error))

    def next_section(self):
        try:
            self.window.commit_landscape()
            if self.region.channel_project:self.project=ChannelProject.from_data(self.region.channel_project)
            if self.project is None:
                if not self.prepare_channel():return
            index=self.project.active_section
            if index not in self.project.completed:
                raise ValueError('Сначала завершите текущий участок и выгрузите грунт; можно выбрать другой участок вручную')
            if index+1>=self.project.section_count:
                self.result.setText('Достигнут последний участок проекта.');return
            self.project.active_section=index+1;self.region.channel_project=self.project.to_data()
            self.section.setValue(index+2);self.start_channel()
        except (ValueError,OSError) as error:self.result.setText(str(error))

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
