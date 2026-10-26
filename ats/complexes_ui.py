"""Regional overview and cloning of self-contained working complexes."""
from pathlib import Path
from PySide6 import QtCore,QtWidgets
import pyqtgraph as pg
import numpy as np
from .complexes import ComplexTemplate


class ComplexesWindow(QtWidgets.QDialog):
    def __init__(self,host):
        super().__init__(host);self.host=host;self.project=host.complex_project
        self.setWindowTitle('Комплексы общего проекта');self.resize(1160,780)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('Каждый комплекс имеет собственную базу, технику, батареи и задания. Общая региональная карта и время; рабочие окна с защитным промежутком не пересекаются. Перевозки между комплексами не выполняются. Новый сценарий начинает новый проект.')
        note.setWordWrap(True);layout.addWidget(note)
        self.status=QtWidgets.QLabel();layout.addWidget(self.status)
        self.map=pg.PlotWidget();self.map.setAspectLocked(True);self.map.setLabel('bottom','X региона',units='м');self.map.setLabel('left','Y региона',units='м')
        self.image=pg.ImageItem(axisOrder='row-major');self.map.addItem(self.image)
        self.bases=pg.ScatterPlotItem(size=15,symbol='s',pen='w');self.map.addItem(self.bases)
        self.drones=pg.ScatterPlotItem(size=7,symbol='+',pen='#94ffe0');self.map.addItem(self.drones)
        self.outlines=[];self.labels=[];self._count=0;layout.addWidget(self.map,3)
        if self.project.landscape is not None:
            r=self.project.landscape.raster;lo=float(np.nanmin(r.height));hi=float(np.nanmax(r.height))
            self.image.setImage(r.height,levels=(lo,hi if hi>lo else lo+1),lut=pg.ColorMap([0,1],['#234d6a','#d7c190']).getLookupTable())
            rx,ry=r.resolution_m;w,h=r.extent_m;self.image.setRect(QtCore.QRectF(-rx/2,-ry/2,w+rx,h+ry))
        self.table=QtWidgets.QTableWidget(0,8);self.table.setHorizontalHeaderLabels(['Комплекс','Участок','Статус','Дроны','Панели, м²','База, кВт·ч','Выемка, м³','Расход, кВт·ч'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers);self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(lambda row,col:self.open_unit(row));layout.addWidget(self.table,1)
        controls=QtWidgets.QHBoxLayout();self.template=QtWidgets.QComboBox();self.template.setMinimumWidth(250)
        controls.addWidget(QtWidgets.QLabel('Шаблон'));controls.addWidget(self.template,1)
        self.section=QtWidgets.QSpinBox();self.section.setRange(1,100000);controls.addWidget(QtWidgets.QLabel('Участок'));controls.addWidget(self.section)
        self.auto=QtWidgets.QCheckBox('Следующий свободный');self.auto.setChecked(True);controls.addWidget(self.auto)
        self.x=QtWidgets.QDoubleSpinBox();self.y=QtWidgets.QDoubleSpinBox()
        for s in (self.x,self.y):s.setRange(0,100_000_000);s.setDecimals(2)
        bounds=self.project.active.bounds();self.x.setValue(bounds[0]+bounds[2]+10);self.y.setValue(bounds[1])
        controls.addWidget(QtWidgets.QLabel('X/Y'));controls.addWidget(self.x);controls.addWidget(self.y)
        layout.addLayout(controls)
        actions=QtWidgets.QHBoxLayout()
        for text,handler in [('Рассчитать энергоснабжение…',host.show_power_design),('Загрузить шаблон…',self.load_template),('Сохранить текущую комплектацию…',self.save_current_template),('Показать выбранный',lambda:self.open_unit(self.table.currentRow()))]:
            button=QtWidgets.QPushButton(text);button.clicked.connect(handler);actions.addWidget(button)
        self.clone_button=QtWidgets.QPushButton('Добавить копию комплекса');self.clone_button.clicked.connect(self.clone_unit);actions.addWidget(self.clone_button)
        layout.addLayout(actions)
        self.info=QtWidgets.QLabel('Двойной щелчок по строке открывает локальную карту комплекса. Start/Pause главного окна управляет общим временем.');self.info.setWordWrap(True);layout.addWidget(self.info)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(700)
        self.refresh()

    def refresh(self):
        p=self.project;r=p.report()
        keys=[(t.name,id(t)) for t in p.templates]
        if getattr(self,'_templates',None)!=keys:
            index=self.template.currentIndex();self.template.clear()
            for t in p.templates:self.template.addItem(t.name)
            if index>=0:self.template.setCurrentIndex(min(index,len(p.templates)-1))
            self._templates=keys
        self.clone_button.setEnabled(p.landscape is not None and bool(p.templates))
        channel=p.landscape is not None and p.landscape.channel_project is not None
        self.section.setEnabled(channel);self.auto.setEnabled(channel)
        self.x.setEnabled(p.landscape is not None and not channel);self.y.setEnabled(p.landscape is not None and not channel)
        self.status.setText(f"Общее время: {r['time_seconds']/3600:.2f} ч · комплексов {len(p.units)} · выемка {r['excavated_m3']:.3f} м³ · расход {r['consumed_kwh']:.2f} кВт·ч · энергия новых поставок {r['delivered_energy_kwh']:.2f} кВт·ч")
        self.table.setRowCount(len(p.units));points=[];air=[]
        colors=['#ffd166','#79d8b2','#7fb7ff','#ef8bbd','#b9dc7e','#c3a0f2','#ffad76','#77d5dd']
        if self._count!=len(p.units):
            for item in self.outlines+self.labels:self.map.removeItem(item)
            self.outlines=[];self.labels=[]
            for i,u in enumerate(p.units):
                outline=pg.PlotDataItem(pen=pg.mkPen(colors[i],width=2));self.map.addItem(outline);self.outlines.append(outline)
                label=pg.TextItem(color=colors[i],anchor=(0,1));self.map.addItem(label);self.labels.append(label)
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
            self.outlines[row].setData([x,x+w,x+w,x,x],[y,y,y+h,y+h,y])
            point=s.base.position[:2]+origin;points.append(dict(pos=point,brush=colors[row]))
            self.labels[row].setPos(*point);self.labels[row].setText(u.name)
            air.extend(d.position[:2]+origin for d in s.drones)
        self.bases.setData(points);self.drones.setData(pos=np.array(air) if air else np.empty((0,2)))

    def load_template(self,path=None):
        try:
            if not isinstance(path,(str,Path)):path=QtWidgets.QFileDialog.getOpenFileName(self,'Загрузить шаблон','','Комплекс ATS (*.atstemplate)')[0]
            if not path:return
            self.project.templates.append(ComplexTemplate.load(path));self.refresh();self.template.setCurrentIndex(len(self.project.templates)-1)
        except (ValueError,OSError,KeyError,TypeError) as error:self.info.setText(str(error))

    def save_current_template(self,path=None):
        try:
            template=ComplexTemplate.capture(self.host.config,'Текущая комплектация')
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
            if channel:
                section=self.project.find_free_section(template,self.section.value()-1) if self.auto.isChecked() else self.section.value()-1
                if section is None:raise ValueError('В проекте нет свободного непересекающегося рабочего окна')
                unit=self.project.add(template,section=section);self.section.setValue(section+2)
            else:unit=self.project.add(template,origin=[self.x.value(),self.y.value()])
            self.info.setText(f'Добавлен {unit.name}. Собственный инвентарь и начальная энергия учтены как новая поставка.');self.refresh()
        except (ValueError,OSError,KeyError) as error:self.info.setText(str(error))

    def open_unit(self,row):
        if 0<=row<len(self.project.units):self.host.select_complex(self.project.units[row].id)

    def closeEvent(self,event):self.timer.stop();super().closeEvent(event)
