"""Equipment inventories and actual daily production, independent of terrain editing."""
from copy import deepcopy
from dataclasses import fields
from datetime import date,timedelta
import json
from pathlib import Path
from PySide6 import QtCore,QtWidgets
from .config import DroneConfig
from .equipment import (BatteryModule,PanelModule,LaserEmitter,LaserReceiver,BucketModule,
                        DroneEquipment,resolve_config,equipment_data,load_equipment_data)


def spin(value,minimum=0,maximum=100000,decimals=3):
    widget=QtWidgets.QDoubleSpinBox();widget.setDecimals(decimals)
    widget.setRange(minimum,maximum);widget.setValue(value);return widget


class ModuleTable(QtWidgets.QWidget):
    changed=QtCore.Signal()

    def __init__(self,kind,columns,modules,parent=None):
        super().__init__(parent);self.kind=kind;self.columns=columns
        layout=QtWidgets.QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.table=QtWidgets.QTableWidget(0,len(columns))
        self.table.setHorizontalHeaderLabels([label for _,label in columns])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(115);layout.addWidget(self.table)
        buttons=QtWidgets.QHBoxLayout()
        add=QtWidgets.QPushButton("Добавить тип");remove=QtWidgets.QPushButton("Удалить выбранный")
        add.clicked.connect(lambda:self.append(self.kind()));remove.clicked.connect(self.remove)
        buttons.addWidget(add);buttons.addWidget(remove);buttons.addStretch();layout.addLayout(buttons)
        for module in modules:self.append(module)
        self.table.itemChanged.connect(lambda _:self.changed.emit())

    def append(self,module):
        limit=6 if self.kind is LaserEmitter else 12
        if self.table.rowCount()>=limit:return
        row=self.table.rowCount();self.table.insertRow(row)
        for col,(key,_) in enumerate(self.columns):
            value=getattr(module,key)
            if key=="efficiency":value*=100
            self.table.setItem(row,col,QtWidgets.QTableWidgetItem(str(value)))
        self.changed.emit()

    def remove(self):
        row=self.table.currentRow()
        if row>=0:self.table.removeRow(row);self.changed.emit()

    def modules(self):
        result=[]
        types={f.name:f.type for f in fields(self.kind)}
        for row in range(self.table.rowCount()):
            values={}
            for col,(key,_) in enumerate(self.columns):
                item=self.table.item(row,col)
                if item is None:raise ValueError("Заполните все ячейки оборудования")
                raw=item.text().strip()
                value=raw if types[key] is str else float(raw.replace(",","."))
                if types[key] is int:
                    if not value.is_integer():raise ValueError("Количество и номер должны быть целыми")
                    value=int(value)
                if key=="efficiency":value/=100
                values[key]=value
            module=self.kind(**values);module.validate();result.append(module)
        return result


BATTERY_COLUMNS=[("name","Тип батареи"),("quantity","Количество"),("capacity_kwh","кВт·ч / шт."),("mass_kg","кг / шт.")]


class BaseEditor(QtWidgets.QWidget):
    def __init__(self,equipment):
        super().__init__();self.original=deepcopy(equipment)
        layout=QtWidgets.QVBoxLayout(self);form=QtWidgets.QFormLayout();layout.addLayout(form)
        self.chassis=spin(equipment.chassis_mass_kg,.001)
        self.slots=QtWidgets.QSpinBox();self.slots.setRange(1,6);self.slots.setValue(equipment.landing_slots)
        self.slot_mass=spin(equipment.slot_mass_kg,.001)
        self.charger_mass=spin(equipment.charger_mass_kg,.001)
        self.power=spin(equipment.charge_power_kw)
        self.efficiency=spin(equipment.charge_efficiency*100,.001,100)
        for label,widget in [("Шасси, привод и управление, кг",self.chassis),("Посадочных мест",self.slots),("Масса одного места, кг",self.slot_mass),("Зарядное устройство, кг",self.charger_mass),("Общая входная мощность зарядки, кВт",self.power),("КПД зарядки, %",self.efficiency)]:form.addRow(label,widget)
        self.summary=QtWidgets.QLabel();self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.batteries=ModuleTable(BatteryModule,BATTERY_COLUMNS,equipment.batteries)
        self.panels=ModuleTable(PanelModule,[("name","Тип панели"),("quantity","Количество"),("area_m2","м² / шт."),("mass_kg","кг / шт."),("efficiency","КПД, %")],equipment.panels)
        self.emitters=ModuleTable(LaserEmitter,[("drone_number","Позиция дрона"),("mass_kg","Масса, кг"),("input_power_kw","Вход, кВт"),("efficiency","КПД, %"),("range_m","Дальность, м")],equipment.emitters)
        self.module_tabs=QtWidgets.QTabWidget();layout.addWidget(self.module_tabs)
        for title,table in [("Общий аккумулятор",self.batteries),("Солнечные панели",self.panels),("Лазерные передатчики",self.emitters)]:self.module_tabs.addTab(table,title)
        for widget in [self.chassis,self.slots,self.slot_mass,self.charger_mass,self.power,self.efficiency]:widget.valueChanged.connect(self.update_summary)
        for table in [self.batteries,self.panels,self.emitters]:table.changed.connect(self.update_summary)
        self.update_summary()

    def value(self):
        e=deepcopy(self.original);e.chassis_mass_kg=self.chassis.value();e.landing_slots=self.slots.value()
        e.slot_mass_kg=self.slot_mass.value();e.charger_mass_kg=self.charger_mass.value()
        e.charge_power_kw=self.power.value();e.charge_efficiency=self.efficiency.value()/100
        e.batteries=self.batteries.modules();e.panels=self.panels.modules();e.emitters=self.emitters.modules()
        e.validate();return e

    def update_summary(self):
        try:
            e=self.value();self.summary.setText(f"Масса базы: {e.mass:.1f} кг · аккумулятор: {e.capacity/3_600_000:.2f} кВт·ч · панели: {e.panel_area:.2f} м² · средний КПД: {e.panel_efficiency*100:.1f}%")
        except ValueError as error:self.summary.setText(str(error))


class DroneEditor(QtWidgets.QWidget):
    def __init__(self,model,density):
        super().__init__();self.original=deepcopy(model);self.density=density;e=model.equipment
        layout=QtWidgets.QVBoxLayout(self);form=QtWidgets.QFormLayout();layout.addLayout(form)
        self.name=QtWidgets.QLineEdit(model.name)
        self.frame=spin(e.frame_mass_kg,.001);self.payload_limit=spin(e.payload_limit_kg,.001)
        self.reference_mass=spin(e.reference_mass_kg,.001);self.reference_power=spin(e.reference_hover_power_kw,.001)
        self.empty_speed=spin(model.empty_speed,.001,50);self.loaded_speed=spin(model.loaded_speed,.001,50)
        self.threshold=spin(model.charge_threshold*100,model.reserve*100,100)
        self.initial=spin(model.initial_charge*100,0,100)
        for label,widget in [("Имя",self.name),("Рама, двигатели и управление, кг",self.frame),("Предел груза рамы, кг",self.payload_limit),("Опорная масса для оценки мощности, кг",self.reference_mass),("Висение при опорной массе, кВт",self.reference_power),("Скорость без груза, м/с",self.empty_speed),("Скорость с грузом, м/с",self.loaded_speed),("Заряжаться ниже, % → до 100%",self.threshold),("Заряд при доставке нового дрона, %",self.initial)]:form.addRow(label,widget)
        self.summary=QtWidgets.QLabel();self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.batteries=ModuleTable(BatteryModule,BATTERY_COLUMNS,e.batteries)
        bucket=QtWidgets.QGroupBox("Ковш / грейфер");f=QtWidgets.QFormLayout(bucket)
        self.bucket_mass=spin(e.bucket.mass_kg,.001);self.volume=spin(e.bucket.volume_m3,.000001,100,6)
        self.radius=spin(e.bucket.radius_m,.001,1)
        for label,widget in [("Масса, кг",self.bucket_mass),("Объём, м³",self.volume),("Радиус захвата, м",self.radius)]:f.addRow(label,widget)
        receiver=QtWidgets.QGroupBox("Лазерный приёмник — оборудование будущего этапа");f=QtWidgets.QFormLayout(receiver)
        self.receiver=QtWidgets.QCheckBox("Установлен");self.receiver.setChecked(e.receiver.installed);f.addRow(self.receiver)
        self.receiver_mass=spin(e.receiver.mass_kg,.001);self.receiver_area=spin(e.receiver.area_m2,.001)
        self.receiver_power=spin(e.receiver.max_power_kw,.001);self.receiver_efficiency=spin(e.receiver.efficiency*100,.001,100)
        for label,widget in [("Масса, кг",self.receiver_mass),("Площадь, м²",self.receiver_area),("Предельная мощность, кВт",self.receiver_power),("КПД, %",self.receiver_efficiency)]:f.addRow(label,widget)
        self.module_tabs=QtWidgets.QTabWidget();self.module_tabs.setMaximumHeight(300);layout.addWidget(self.module_tabs)
        self.module_tabs.addTab(self.batteries,"Батареи");self.module_tabs.addTab(bucket,"Ковш / грейфер");self.module_tabs.addTab(receiver,"Лазерный приёмник")
        for widget in self.findChildren(QtWidgets.QDoubleSpinBox):widget.valueChanged.connect(self.update_summary)
        self.receiver.toggled.connect(self.update_summary);self.batteries.changed.connect(self.update_summary);self.update_summary()

    def value(self):
        model=deepcopy(self.original);e=model.equipment
        model.name=self.name.text();model.empty_speed=self.empty_speed.value();model.loaded_speed=self.loaded_speed.value()
        model.charge_threshold=self.threshold.value()/100;model.initial_charge=self.initial.value()/100
        e.frame_mass_kg=self.frame.value();e.payload_limit_kg=self.payload_limit.value()
        e.reference_mass_kg=self.reference_mass.value();e.reference_hover_power_kw=self.reference_power.value()
        e.batteries=self.batteries.modules();e.bucket=BucketModule(self.bucket_mass.value(),self.volume.value(),self.radius.value())
        e.receiver=LaserReceiver(self.receiver.isChecked(),self.receiver_mass.value(),self.receiver_area.value(),self.receiver_power.value(),self.receiver_efficiency.value()/100)
        e.resolve(model,self.density);return model

    def update_summary(self):
        try:
            m=self.value();self.summary.setText(f"Без груза: {m.empty_mass:.1f} кг · батарея: {m.battery/3_600_000:.2f} кВт·ч · груз: {m.payload:.1f} кг · висение: {m.hover_power/1000:.2f} кВт")
        except ValueError as error:self.summary.setText(str(error))


class EquipmentDialog(QtWidgets.QDialog):
    def __init__(self,config,parent=None):
        super().__init__(parent);self.config=resolve_config(deepcopy(config))
        self.setWindowTitle("Оборудование базы и дронов · без перезапуска");self.resize(1000,850)
        layout=QtWidgets.QVBoxLayout(self);form=QtWidgets.QFormLayout()
        self.count=QtWidgets.QSpinBox();self.count.setRange(1,6);self.count.setValue(config.fleet.count)
        self.wait_mode=QtWidgets.QComboBox();self.wait_mode.addItem("На земле","ground");self.wait_mode.addItem("В воздухе с посадкой по резерву","air")
        self.wait_mode.setCurrentIndex(0 if config.fleet.wait_mode=="ground" else 1)
        form.addRow("Дронов (удаляются последние позиции)",self.count);form.addRow("Очередь зарядки",self.wait_mode);layout.addLayout(form)
        info=QtWidgets.QLabel("База обновляется сразу на стоянке. Для изменения дронов флот завершит перенос груза и приземлится. Новые батарейные модули пустые; снятые уносят свой заряд. Новый дрон доставляется с указанным зарядом. Лазерное оборудование пока добавляет массу; передача энергии будет реализована отдельно.")
        info.setWordWrap(True);layout.addWidget(info)
        self.tabs=QtWidgets.QTabWidget();layout.addWidget(self.tabs,1)
        self.base=BaseEditor(self.config.base.equipment);self.slots=self.base.slots;self.add_tab(self.base,"База")
        self.editors=[];self.rows(self.count.value());self.count.valueChanged.connect(self.rows)
        actions=QtWidgets.QHBoxLayout()
        for label,callback in [("Тяжёлый 60",lambda:self.preset(False)),("Лёгкий 30",lambda:self.preset(True)),("Сохранить комплектацию",self.export_profile),("Загрузить комплектацию",self.import_profile)]:
            button=QtWidgets.QPushButton(label);button.clicked.connect(callback);actions.addWidget(button)
        layout.addLayout(actions)
        self.error=QtWidgets.QLabel();self.error.setWordWrap(True);layout.addWidget(self.error)
        self.buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok|QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Применить без перезапуска")
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        self.buttons.accepted.connect(self.save);self.buttons.rejected.connect(self.reject);layout.addWidget(self.buttons)

    def add_tab(self,widget,title,index=None):
        scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(widget)
        if index is None:self.tabs.addTab(scroll,title)
        else:self.tabs.insertTab(index,scroll,title)

    def rows(self,count):
        while len(self.editors)>count:
            self.editors.pop();widget=self.tabs.widget(self.tabs.count()-1);self.tabs.removeTab(self.tabs.count()-1);widget.deleteLater()
        models=self.config.drone_models()
        while len(self.editors)<count:
            index=len(self.editors);model=deepcopy(models[index] if index<len(models) else self.config.drone)
            editor=DroneEditor(model,self.config.terrain.density);self.editors.append(editor);self.add_tab(editor,f"Дрон: позиция {index+1}")

    def preset(self,light):
        index=max(0,self.tabs.currentIndex()-1);model=DroneConfig()
        if light:model.name="Лёгкий 30";model.empty_mass=75;model.payload=30;model.battery=3*3_600_000;model.hover_power=12000
        model.equipment=DroneEquipment.from_legacy(model,self.config.terrain.density,self.config.terrain.grab_radius)
        editor=DroneEditor(model,self.config.terrain.density);self.editors[index]=editor
        old=self.tabs.widget(index+1);self.tabs.removeTab(index+1);old.deleteLater()
        self.add_tab(editor,f"Дрон: позиция {index+1}",index+1);self.tabs.setCurrentIndex(index+1)

    def value(self):
        clone=deepcopy(self.config);clone.base.equipment=self.base.value()
        clone.fleet.count=self.count.value();clone.fleet.models=[editor.value() for editor in self.editors]
        clone.fleet.wait_mode=self.wait_mode.currentData();resolve_config(clone);clone.validate();return clone

    def save(self):
        try:self.result_config=self.value();self.accept()
        except (ValueError,TypeError,AttributeError) as error:self.error.setText(f"Проверьте параметры: {error}")

    def export_profile(self):
        try:config=self.value()
        except ValueError as error:self.error.setText(str(error));return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,"Сохранить комплектацию","equipment.json","JSON (*.json)")
        if not path:return
        try:Path(path).write_text(json.dumps(equipment_data(config),ensure_ascii=False,indent=2),encoding="utf-8")
        except OSError as error:self.error.setText(str(error))

    def import_profile(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,"Загрузить комплектацию","","JSON (*.json)")
        if not path:return
        try:config=load_equipment_data(json.loads(Path(path).read_text(encoding="utf-8")),self.config)
        except (OSError,ValueError,TypeError,KeyError,AttributeError) as error:self.error.setText(f"Не удалось загрузить: {error}");return
        self.config=config
        while self.tabs.count():
            widget=self.tabs.widget(0);self.tabs.removeTab(0);widget.deleteLater()
        self.base=BaseEditor(config.base.equipment);self.slots=self.base.slots;self.add_tab(self.base,"База")
        self.editors=[];self.count.blockSignals(True);self.count.setValue(config.fleet.count);self.count.blockSignals(False)
        self.wait_mode.setCurrentIndex(0 if config.fleet.wait_mode=="ground" else 1);self.rows(config.fleet.count)


class DailyStatsWindow(QtWidgets.QDialog):
    def __init__(self,window):
        super().__init__(window);self.window=window;self.setWindowTitle("Перемещённый грунт по суткам");self.resize(1000,500)
        layout=QtWidgets.QVBoxLayout(self)
        info=QtWidgets.QLabel("Учитывается реально выгруженный грунт: в отвал и в засыпку. Груз в полёте ещё не засчитан. Сутки — от полуночи до полуночи местного солнечного времени; текущие сутки могут быть неполными.")
        info.setWordWrap(True);layout.addWidget(info)
        self.day=QtWidgets.QComboBox();self.day.addItem("Все сутки",None);layout.addWidget(self.day)
        self.table=QtWidgets.QTableWidget(0,8);self.table.setHorizontalHeaderLabels(["Дрон","Сутки / дата","Грунт, кг","Грунт, м³","Засыпка, кг","Отвал, кг","Выгрузок","В составе"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers);layout.addWidget(self.table)
        self.day.currentIndexChanged.connect(self.refresh)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(1000);self.refresh()

    def day_text(self,number):
        start=self.window.sim.config.base.start_day_of_year
        d=date(2025,1,1)+timedelta(days=(start-1+number)%365)
        return f"{number+1} · {d:%d.%m}"

    def refresh(self):
        s=self.window.sim;current=int((s.config.start_hour*3600+s.time)//86400)
        selected=self.day.currentData();self.day.blockSignals(True)
        self.day.clear();self.day.addItem("Все сутки",None)
        for number in range(current+1):self.day.addItem(self.day_text(number),number)
        self.day.setCurrentIndex(max(0,self.day.findData(selected)));self.day.blockSignals(False)
        selected=self.day.currentData();rows=[]
        for d in [*s.drones,*s.retired_drones]:
            report=d.productivity.report(s.time,current);daily={row['mission_day']:row for row in report['daily']}
            first=int((s.config.start_hour*3600+d.created_at)//86400)
            last=int((s.config.start_hour*3600+(d.retired_at if d.retired_at is not None else s.time))//86400)
            for number in range(first,last+1):
                if selected is not None and number!=selected:continue
                r=daily.get(number,{"mass_kg":0,"volume_m3":0,"fill_mass_kg":0,"dump_mass_kg":0,"deliveries":0})
                rows.append([f"{d.serial} · {d.config.name}",self.day_text(number),f"{r['mass_kg']:.2f}",f"{r['volume_m3']:.4f}",f"{r['fill_mass_kg']:.2f}",f"{r['dump_mass_kg']:.2f}",str(r['deliveries']),"Снят" if d.retired_at is not None else "Работает"])
        self.table.setRowCount(len(rows))
        for row,values in enumerate(rows):
            for col,value in enumerate(values):self.table.setItem(row,col,QtWidgets.QTableWidgetItem(value))

    def closeEvent(self,event):
        self.timer.stop();super().closeEvent(event)
