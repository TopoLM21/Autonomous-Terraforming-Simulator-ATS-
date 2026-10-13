"""Small fleet editor and responsive independent scenario comparison."""
from copy import deepcopy
import time
from PySide6 import QtCore,QtWidgets
from .comparison import comparison_config
from .simulation import Simulation,elapsed_text


from .equipment_ui import EquipmentDialog as FleetDialog


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
