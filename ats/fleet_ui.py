"""Small fleet editor and responsive independent scenario comparison."""
from copy import deepcopy
import time
from PySide6 import QtCore,QtWidgets
from .config import DroneConfig,KWH
from .comparison import comparison_config
from .simulation import Simulation,elapsed_text


class FleetDialog(QtWidgets.QDialog):
    def __init__(self,config,parent=None):
        super().__init__(parent)
        self.config = deepcopy(config)
        self.setWindowTitle("Дроны и зарядка")
        self.resize(980,480)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.count = QtWidgets.QSpinBox(); self.count.setRange(1,6); self.count.setValue(config.fleet.count)
        self.slots = QtWidgets.QSpinBox(); self.slots.setRange(1,6); self.slots.setValue(config.fleet.charge_slots)
        self.wait_mode = QtWidgets.QComboBox()
        self.wait_mode.addItem("Посадка рядом и ожидание на земле","ground")
        self.wait_mode.addItem("Ожидание в воздухе; аварийная посадка по резерву","air")
        self.wait_mode.setCurrentIndex(0 if config.fleet.wait_mode == "ground" else 1)
        self.power = QtWidgets.QDoubleSpinBox(); self.power.setRange(0,100); self.power.setValue(config.base.charge_power/1000); self.power.setSuffix(" кВт")
        for name,widget in [("Количество дронов",self.count),("Зарядных слотов",self.slots),("Общая мощность зарядки",self.power),("Очередь",self.wait_mode)]:
            form.addRow(name,widget)
        layout.addLayout(form)
        info=QtWidgets.QLabel("Каждый рабочий вылет — после зарядки до 100%. Слоты делят общую входную мощность, КПД зарядки 90%. Параметры моделей — инженерные оценки. Изменения создадут новый сценарий.")
        info.setWordWrap(True); layout.addWidget(info)
        self.table=QtWidgets.QTableWidget(0,7)
        self.table.setHorizontalHeaderLabels(["Имя","Масса, кг","Груз, кг","Батарея, кВт·ч","Висение, кВт","Без груза, м/с","С грузом, м/с"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        self.models=config.drone_models()
        self.rows(self.count.value()); self.count.valueChanged.connect(self.rows)
        presets=QtWidgets.QHBoxLayout()
        heavy=QtWidgets.QPushButton("Тяжёлый 60 для выбранной строки")
        light=QtWidgets.QPushButton("Лёгкий 30 для выбранной строки")
        heavy.clicked.connect(lambda:self.preset(False)); light.clicked.connect(lambda:self.preset(True))
        presets.addWidget(heavy);presets.addWidget(light);layout.addLayout(presets)
        self.error=QtWidgets.QLabel(); self.error.setWordWrap(True); layout.addWidget(self.error)
        buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok|QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.save);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def rows(self,count):
        old=self.table.rowCount(); self.table.setRowCount(count)
        for i in range(old,count):
            self.fill(i,deepcopy(self.models[i] if i<len(self.models) else self.config.drone))

    def fill(self,row,model):
        values=[model.name,model.empty_mass,model.payload,model.battery/KWH,model.hover_power/1000,model.empty_speed,model.loaded_speed]
        for column,value in enumerate(values):
            self.table.setItem(row,column,QtWidgets.QTableWidgetItem(str(value)))

    def preset(self,light):
        row=max(0,self.table.currentRow())
        model=deepcopy(self.config.drone)
        if light:
            model.name="Лёгкий 30";model.empty_mass=75;model.payload=30;model.battery=3*KWH;model.hover_power=12000
        self.fill(row,model)

    def save(self):
        try:
            clone=deepcopy(self.config); clone.fleet.count=self.count.value(); clone.fleet.charge_slots=self.slots.value()
            clone.fleet.wait_mode=self.wait_mode.currentData(); clone.base.charge_power=self.power.value()*1000
            models=[]
            for row in range(self.table.rowCount()):
                model=deepcopy(self.models[row] if row<len(self.models) else self.config.drone)
                values=[self.table.item(row,c).text() for c in range(7)]
                model.name=values[0]
                model.empty_mass,model.payload,battery,power,model.empty_speed,model.loaded_speed=map(float,values[1:])
                model.battery=battery*KWH;model.hover_power=power*1000;models.append(model)
            clone.fleet.models=models;clone.validate();self.result_config=clone;self.accept()
        except (ValueError,AttributeError) as error:
            self.error.setText(f"Проверьте параметры: {error}")


class ComparisonWindow(QtWidgets.QDialog):
    def __init__(self,config,parent=None):
        super().__init__(parent)
        self.config=deepcopy(config);self.count=1;self.sim=None;self.results=[]
        self.setWindowTitle("Сравнение независимых флотов 1…N")
        self.resize(1000,430)
        layout=QtWidgets.QVBoxLayout(self)
        info=QtWidgets.QLabel("Одинаковые участок, seed, панели, база и зарядные слоты. Составы используют первые 1…N моделей. Начальный запас энергии меняется с числом батарей и показан отдельно. Расчёт идёт без отрисовки карты; его можно остановить.")
        info.setWordWrap(True);layout.addWidget(info)
        self.status=QtWidgets.QLabel();layout.addWidget(self.status)
        self.table=QtWidgets.QTableWidget(0,7)
        self.table.setHorizontalHeaderLabels(["Дронов","Статус","Время","Рейсы","Расход, кВт·ч","Старт, кВт·ч","Очередь, мин"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        self.stop=QtWidgets.QPushButton("Остановить расчёт");self.stop.clicked.connect(self.cancel);layout.addWidget(self.stop)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(config.display_interval_ms)

    def tick(self):
        deadline=time.perf_counter()+self.config.frame_budget_ms/1000
        while time.perf_counter()<deadline:
            if self.sim is None:
                if self.count>self.config.fleet.count:
                    self.timer.stop();self.status.setText("Сравнение завершено");return
                self.sim=Simulation(comparison_config(self.config,self.count))
            self.sim.step()
            self.status.setText(f"Флот {self.count}/{self.config.fleet.count} · {elapsed_text(self.sim.time)} · {self.sim.terrain.remaining:.3f} м³ осталось")
            if self.sim.state.value!='RUNNING' or self.sim.time>=self.config.fleet.comparison_step_limit:
                r=self.sim.report();self.results.append(r);row=self.table.rowCount();self.table.insertRow(row)
                values=[r['drone_count'],r['state'],r['elapsed_display'],r['trips'],f"{r['consumed_kwh']:.3f}",f"{r['initial_energy_kwh']:.1f}",f"{sum(d['waiting_seconds'] for d in r['drones'])/60:.1f}"]
                for col,value in enumerate(values):self.table.setItem(row,col,QtWidgets.QTableWidgetItem(str(value)))
                self.table.item(row,1).setToolTip(r['reason'] or "Достигнут предел шагов" if r['state']=='RUNNING' else r['reason'])
                self.count+=1;self.sim=None

    def cancel(self):
        self.timer.stop();self.status.setText("Расчёт остановлен")

    def closeEvent(self,event):
        self.timer.stop();super().closeEvent(event)
