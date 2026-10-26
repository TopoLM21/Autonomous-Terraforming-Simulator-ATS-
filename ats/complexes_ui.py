"""Regional overview and cloning of self-contained working complexes."""
from pathlib import Path
import math
from PySide6 import QtCore,QtWidgets
import pyqtgraph as pg
import numpy as np
from .complexes import ComplexTemplate


class ComplexesWindow(QtWidgets.QDialog):
    def __init__(self,host):
        super().__init__(host);self.host=host;self.project=host.complex_project
        self.setWindowTitle('Комплексы общего проекта');self.resize(1160,780)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('До 20 комплексов. В режиме общего участка все копают один рельеф: диспетчер распределяет ячейки плана и согласует маршруты. У каждого комплекса свои реальные батареи и оборудование. Новый сценарий начинает новый проект.')
        note.setWordWrap(True);layout.addWidget(note)
        self.status=QtWidgets.QLabel();layout.addWidget(self.status)
        self.tabs=QtWidgets.QTabWidget();layout.addWidget(self.tabs,1)
        overview=QtWidgets.QWidget();overview_layout=QtWidgets.QVBoxLayout(overview);self.tabs.addTab(overview,'Комплексы и общая карта')
        provision=QtWidgets.QWidget();provision_layout=QtWidgets.QVBoxLayout(provision);self.tabs.addTab(provision,'Добавить / настроить')
        self.map=pg.PlotWidget();self.map.setAspectLocked(True);self.map.setLabel('bottom','X региона',units='м');self.map.setLabel('left','Y региона',units='м')
        self.image=pg.ImageItem(axisOrder='row-major');self.map.addItem(self.image)
        self.bases=pg.ScatterPlotItem(size=15,symbol='s',pen='w');self.map.addItem(self.bases)
        self.drones=pg.ScatterPlotItem(size=7,symbol='+',pen='#94ffe0');self.map.addItem(self.drones)
        self.outlines=[];self.labels=[];self._count=0;overview_layout.addWidget(self.map,3)
        if self.project.landscape is not None:
            r=self.project.landscape.raster;lo=float(np.nanmin(r.height));hi=float(np.nanmax(r.height))
            self.image.setImage(r.height,levels=(lo,hi if hi>lo else lo+1),lut=pg.ColorMap([0,1],['#234d6a','#d7c190']).getLookupTable())
            rx,ry=r.resolution_m;w,h=r.extent_m;self.image.setRect(QtCore.QRectF(-rx/2,-ry/2,w+rx,h+ry))
        self.table=QtWidgets.QTableWidget(0,8);self.table.setHorizontalHeaderLabels(['Комплекс','Участок','Статус','Дроны','Панели, м²','База, кВт·ч','Выемка, м³','Расход, кВт·ч'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers);self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(lambda row,col:self.open_unit(row));overview_layout.addWidget(self.table,2)
        controls=QtWidgets.QGridLayout();self.template=QtWidgets.QComboBox();self.template.setMinimumWidth(150)
        controls.addWidget(QtWidgets.QLabel('Шаблон'),0,0);controls.addWidget(self.template,0,1,1,3)
        self.shared=QtWidgets.QCheckBox('Общий грунт и общий план');self.shared.setChecked(self.project.shared_site is not None or len(self.project.units)==1)
        self.shared.setEnabled(len(self.project.units)==1 and self.project.shared_site is None);controls.addWidget(self.shared,1,0,1,2)
        self.section=QtWidgets.QSpinBox();self.section.setRange(1,100000);controls.addWidget(QtWidgets.QLabel('Участок'),2,0);controls.addWidget(self.section,2,1)
        self.auto=QtWidgets.QCheckBox('Автоподбор места / участка');self.auto.setChecked(True);controls.addWidget(self.auto,2,2,1,2)
        self.x=QtWidgets.QDoubleSpinBox();self.y=QtWidgets.QDoubleSpinBox()
        for s in (self.x,self.y):s.setRange(0,100_000_000);s.setDecimals(2)
        bounds=self.project.active.bounds();self.x.setValue(bounds[0]+bounds[2]+10);self.y.setValue(bounds[1])
        controls.addWidget(QtWidgets.QLabel('X/Y базы / окна'),3,0);controls.addWidget(self.x,3,1);controls.addWidget(self.y,3,2)
        self.target_count=QtWidgets.QSpinBox();self.target_count.setRange(1,20);self.target_count.setValue(min(20,len(self.project.units)+1))
        controls.addWidget(QtWidgets.QLabel('Всего комплексов'),4,0);controls.addWidget(self.target_count,4,1)
        self.batch_button=QtWidgets.QPushButton('Добавить до указанного количества');self.batch_button.clicked.connect(self.clone_batch);controls.addWidget(self.batch_button,4,2,1,2)
        self.map_side=QtWidgets.QSpinBox();self.map_side.setRange(32,1024);self.map_side.setSingleStep(32)
        self.map_side.setValue(round(self.project.active.bounds()[2]));controls.addWidget(QtWidgets.QLabel('Сторона общей карты, м'),5,0);controls.addWidget(self.map_side,5,1)
        expand=QtWidgets.QPushButton('Увеличить без сброса');expand.clicked.connect(self.enlarge_map);controls.addWidget(expand,5,2,1,2)
        provision_layout.addLayout(controls)
        actions=QtWidgets.QGridLayout()
        for i,(text,handler) in enumerate([('Энергоснабжение…',host.show_power_design),('Загрузить шаблон…',self.load_template),('Сохранить комплектацию…',self.save_current_template),('Показать выбранный',lambda:self.open_unit(self.table.currentRow()))]):
            button=QtWidgets.QPushButton(text);button.clicked.connect(handler);actions.addWidget(button,i//2,i%2)
        self.clone_button=QtWidgets.QPushButton('Добавить одну копию');self.clone_button.clicked.connect(self.clone_unit);actions.addWidget(self.clone_button,2,0,1,2)
        provision_layout.addLayout(actions);provision_layout.addStretch()
        self.info=QtWidgets.QLabel('Двойной щелчок по строке открывает локальную карту комплекса. Start/Pause главного окна управляет общим временем.');self.info.setWordWrap(True);layout.addWidget(self.info)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(700)
        self._pending_batch=False
        self._resume_after_growth=None
        self.shared.toggled.connect(self.refresh)
        self.refresh()
        from .ui_windows import fit_window
        fit_window(self,1100,800)

    def refresh(self):
        p=self.project;r=p.report()
        keys=[(t.name,id(t)) for t in p.templates]
        if getattr(self,'_templates',None)!=keys:
            index=self.template.currentIndex();self.template.clear()
            for t in p.templates:self.template.addItem(t.name)
            if index>=0:self.template.setCurrentIndex(min(index,len(p.templates)-1))
            self._templates=keys
        shared=self.shared.isChecked() or p.shared_site is not None
        self.clone_button.setEnabled(len(p.units)<20 and (shared or p.landscape is not None) and bool(p.templates))
        self.batch_button.setEnabled(self.clone_button.isEnabled())
        channel=p.landscape is not None and p.landscape.channel_project is not None
        self.section.setEnabled(channel and not shared);self.auto.setEnabled(channel or shared)
        self.x.setEnabled(shared and not self.auto.isChecked() or p.landscape is not None and not channel and not shared);self.y.setEnabled(self.x.isEnabled())
        self.status.setText(f"Общее время: {r['time_seconds']/3600:.2f} ч · комплексов {len(p.units)} · выемка {r['excavated_m3']:.3f} м³ · расход {r['consumed_kwh']:.2f} кВт·ч · энергия новых поставок {r['delivered_energy_kwh']:.2f} кВт·ч")
        self.table.setRowCount(len(p.units));points=[];air=[]
        colors=['#ffd166','#79d8b2','#7fb7ff','#ef8bbd','#b9dc7e','#c3a0f2','#ffad76','#77d5dd']
        if self._count!=len(p.units):
            for item in self.outlines+self.labels:self.map.removeItem(item)
            self.outlines=[];self.labels=[]
            for i,u in enumerate(p.units):
                color=pg.intColor(i,hues=20)
                outline=pg.PlotDataItem(pen=pg.mkPen(color,width=2));self.map.addItem(outline);self.outlines.append(outline)
                label=pg.TextItem(color=color,anchor=(0,1));self.map.addItem(label);self.labels.append(label)
            self._count=len(p.units)
            bounds=[u.bounds() for u in p.units]
            self.map.setRange(xRange=(min(b[0] for b in bounds),max(b[0]+b[2] for b in bounds)),
                              yRange=(min(b[1] for b in bounds),max(b[1]+b[3] for b in bounds)),padding=.1)
        for row,(u,data) in enumerate(zip(p.units,r['complexes'])):
            s=u.simulation;origin=np.asarray(u.bounds()[:2]);x,y,w,h=u.bounds()
            values=[u.name,str(u.section+1) if u.section is not None else '—',data['state'],str(data['drones']),f"{data['panel_area_m2']:g}",f"{data['base_capacity_kwh']:g}",f"{data['excavated_m3']:.3f}",f"{data['consumed_kwh']:.2f}"]
            for col,value in enumerate(values):
                item=self.table.item(row,col)
                if item is None:item=QtWidgets.QTableWidgetItem();self.table.setItem(row,col,item)
                item.setText(value);item.setToolTip(data['reason'])
            if p.shared_site is None:self.outlines[row].setData([x,x+w,x+w,x,x],[y,y,y+h,y+h,y])
            else:
                radius=s.base.config.radius;px,py=s.base.position[:2]+origin
                self.outlines[row].setData([px-radius,px+radius,px+radius,px-radius,px-radius],[py-radius,py-radius,py+radius,py+radius,py-radius])
            point=s.base.position[:2]+origin;points.append(dict(pos=point,brush=pg.intColor(row,hues=20)))
            self.labels[row].setPos(*point);self.labels[row].setText(u.name)
            air.extend(d.position[:2]+origin for d in s.drones)
        self.bases.setData(points);self.drones.setData(pos=np.array(air) if air else np.empty((0,2)))
        if p.shared_site is not None:
            core=p.shared_site.core;revision=(core.revision,core.height.shape)
            if getattr(self,'_ground_revision',None)!=revision:
                from .gui import terrain_rgb
                stride=max(1,core.height.shape[0]//512)
                self.image.setImage(terrain_rgb(core.height[::stride,::stride],core.cell*stride,p.active.simulation.config.trench.depth,p.active.simulation.config.terrain.dump_max_height),autoLevels=False)
                side=core.height.shape[0]*core.cell;ox,oy=p.active.bounds()[:2]
                self.image.setRect(QtCore.QRectF(ox,oy,side,side));self._ground_revision=revision
        if self._pending_batch and all(d.grounded for u in p.units for d in u.simulation.drones):
            count=self._pending_batch;self._pending_batch=False;self.target_count.setValue(count);QtCore.QTimer.singleShot(0,self.clone_batch)

    def load_template(self,path=None):
        try:
            if not isinstance(path,(str,Path)):path=QtWidgets.QFileDialog.getOpenFileName(self,'Загрузить шаблон','','Комплекс ATS (*.atstemplate)')[0]
            if not path:return
            self.project.templates.append(ComplexTemplate.load(path));self.refresh();self.template.setCurrentIndex(len(self.project.templates)-1)
        except (ValueError,OSError,KeyError,TypeError) as error:self.info.setText(str(error))

    def save_current_template(self,path=None):
        try:
            template=ComplexTemplate.from_simulation(self.host.sim,'Текущая комплектация')
            if not isinstance(path,(str,Path)):path=QtWidgets.QFileDialog.getSaveFileName(self,'Сохранить комплектацию','complex.atstemplate','Комплекс ATS (*.atstemplate)')[0]
            if not path:return
            template.save(path);self.project.templates.append(template);self.refresh()
        except (ValueError,OSError) as error:self.info.setText(str(error))

    def clone_unit(self):
        try:
            i=self.template.currentIndex()
            if i<0:raise ValueError('Сохраните или загрузите шаблон комплекса')
            template=self.project.templates[i]
            channel=self.project.landscape is not None and self.project.landscape.channel_project is not None
            if self.shared.isChecked() or self.project.shared_site is not None:
                if self.auto.isChecked() and not self.ensure_space(template,len(self.project.units)+1):return False
                unit=self.project.add_shared(template,position=None if self.auto.isChecked() else [self.x.value(),self.y.value()])
                self.shared.setChecked(True);self.shared.setEnabled(False)
                self.host.setup_map()
            elif channel:
                section=self.project.find_free_section(template,self.section.value()-1) if self.auto.isChecked() else self.section.value()-1
                if section is None:raise ValueError('В проекте нет свободного непересекающегося рабочего окна')
                unit=self.project.add(template,section=section);self.section.setValue(section+2)
            else:unit=self.project.add(template,origin=[self.x.value(),self.y.value()])
            self.info.setText(f'Добавлен {unit.name}. Собственный инвентарь и начальная энергия учтены как новая поставка.');self.refresh()
            return True
        except (ValueError,OSError,KeyError) as error:self.info.setText(str(error));return False

    def clone_batch(self):
        if self.shared.isChecked() or self.project.shared_site is not None:
            try:
                i=self.template.currentIndex()
                if i<0:raise ValueError('Сначала выберите шаблон')
                template=self.project.templates[i];count=self.target_count.value()
                if not self.ensure_space(template,count):return
            except ValueError as error:self.info.setText(str(error));return
        while len(self.project.units)<self.target_count.value():
            if not self.clone_unit():break

    def ensure_space(self,template,count):
        required=min(1024,max(self.project.active.bounds()[2],math.ceil(math.sqrt(count*(template.config.base.panel_area+150))*2/32)*32))
        site=self.project.enable_shared()
        if required<=site.core.height.shape[0]*site.core.cell:return True
        previous=self.host.running if self._resume_after_growth is None else self._resume_after_growth
        try:site.enlarge(required)
        except ValueError as error:
            if 'посадите' not in str(error):raise
            self._resume_after_growth=previous
            for u in self.project.units:u.simulation.request_landing()
            self._pending_batch=count;self.host.running=True;self.host.last_wall=__import__('time').perf_counter()
            self.info.setText('Комплексы завершают перевозки и садятся. Затем карта увеличится и копии будут добавлены автоматически.');return False
        path=self.host._session_path
        self.host.install_simulation(self.project.active.simulation,self.project.landscape,self.project)
        self.host._session_path=path
        for u in self.project.units:u.simulation.landing_requested=False
        self.host._after_file_action(previous);self._resume_after_growth=None
        self.map_side.setValue(round(required));return True

    def enlarge_map(self):
        try:
            self.project.enable_shared().enlarge(self.map_side.value())
            self.host.install_simulation(self.project.active.simulation,self.project.landscape,self.project)
            self._count=0;self.refresh();self.info.setText('Общая карта увеличена; техника, грунт, заряды и время сохранены.')
        except ValueError as error:self.info.setText(str(error))

    def open_unit(self,row):
        if 0<=row<len(self.project.units):self.host.select_complex(self.project.units[row].id)

    def closeEvent(self,event):self.timer.stop();super().closeEvent(event)
