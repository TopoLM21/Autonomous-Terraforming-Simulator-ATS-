"""Additional ATS 2.0 windows; the existing map and controls remain in place."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from PySide6 import QtCore,QtWidgets
from .logistics_config import LogisticsConfig
from .laser_profiles import with_laser_profile


class LogisticsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent);self.host=parent
        self.setWindowTitle('ATS 2.0 · Склад и диспетчер');self.resize(1180,720)
        layout=QtWidgets.QVBoxLayout(self)
        tabs=QtWidgets.QTabWidget();layout.addWidget(tabs)
        self.status=QtWidgets.QLabel();self.status.setWordWrap(True)
        tabs.addTab(self.status,'Крот / энергетика')
        inventory_page=QtWidgets.QWidget();box=QtWidgets.QVBoxLayout(inventory_page)
        self.filter=QtWidgets.QLineEdit();self.filter.setPlaceholderText('Фильтр: ID, место, состояние, исполнитель');box.addWidget(self.filter)
        self.inventory=QtWidgets.QTableWidget(0,9);box.addWidget(self.inventory)
        self.inventory.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.inventory.setHorizontalHeaderLabels(['ID','Тип','Место','Состояние','Держатель','Масса, кг','Грунт, кг','SOC / энергия','Резерв'])
        self.inventory.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        tabs.addTab(inventory_page,'Инвентарь')
        self.jobs=QtWidgets.QTableWidget(0,8)
        self.jobs.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.jobs.setHorizontalHeaderLabels(['ID','Задание','Приоритет','Статус / этап','Ресурсы','Дрон','ETA, с','Причина'])
        self.jobs.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        tabs.addTab(self.jobs,'Диспетчер')
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(500)
        self.filter.textChanged.connect(self.refresh);self.refresh()

    def closeEvent(self,event):
        self.timer.stop();super().closeEvent(event)

    def refresh(self):
        s=self.host.sim;l=s.logistics
        if l is None:
            self.status.setText('Старый режим: физический склад ATS 2.0 отключён. Выберите «Эксперименты ATS 2.0».')
            self.inventory.setRowCount(0);self.jobs.setRowCount(0);return
        r=s.report();e=r.get('excavator');service=r['battery_service']
        robot=(f"Крот: {e['state']} · {e['reason']}\nПозиция: {e['position']} · масса {e['mass_kg']:.1f} кг\n"
               f"Батареи: {', '.join(e['battery_ids'])} · {e['energy_kwh']:.3f}/{e['capacity_kwh']:.3f} кВт·ч\n"
               f"Ковш: {e['bucket_kg']:.2f} кг · контейнер: {e['container']}\n"
               f"Циклы: {e['cycles']} · смены батарей: {e['swaps']} · ожидание {e['waiting_seconds']/60:.1f} мин\n") if e else 'Крот отключён\n'
        self.status.setText(robot+f"\nКонтейнерные рейсы: {r['container_cycles']} · групповые переносы: {r['collective_lift_count']}\n"
            f"Группа: {r['collective_lift'] or 'нет'}\n"
            f"Зарядные порты: {len(service['ports'])}/{service['capacity']} · {', '.join(service['ports'])}\n"
            f"Доковая зарядка дронов: {service['dock_charging_drones']}\n"
            f"Очередь: {', '.join(service['queue']) or 'пусто'}\n"
            f"Солнце: {r['solar_power_w']/1000:.2f} кВт · выработка {r['solar_generated_kwh']:.3f} кВт·ч\n"
            f"База: {r['base_kwh']:.2f} кВт·ч · расход {r['consumed_kwh']:.3f} кВт·ч\n"
            f"Лазер: {r['laser_input_power_w']/1000:.2f} → {r['laser_received_power_w']/1000:.2f} кВт\n"
            f"Баланс грунта {r['mass_error_kg']:+.2e} кг · энергии {r['energy_error_j']:+.2e} Дж")
        query=self.filter.text().casefold()
        rows=[i for i in r['inventory'] if query in str(i).casefold()]
        self.inventory.setRowCount(len(rows))
        for row,item in enumerate(rows):
            energy=f"{100*item['energy_j']/item['capacity_j']:.1f}% / {item['energy_j']/3600000:.3f}" if 'energy_j' in item else ''
            values=[item['id'],item['kind'] if item['type']=='Item' else item['type'],item['location'],item['state'],item['holder_id'],
                    f"{item['gross_mass_kg']:.2f}",f"{item.get('soil_kg',0):.2f}",energy,item['reserved_by'] or '']
            for col,value in enumerate(values):self.inventory.setItem(row,col,QtWidgets.QTableWidgetItem(str(value)))
        jobs=r['jobs'][-200:];self.jobs.setRowCount(len(jobs))
        for row,j in enumerate(jobs):
            values=[j['id'],j['kind'],j['priority'],j['status']+' / '+j['phase'],', '.join(j['resource_ids']),
                    j['executor'] or '',round(j['eta_seconds']),j['reason']]
            for col,value in enumerate(values):self.jobs.setItem(row,col,QtWidgets.QTableWidgetItem(str(value)))


class ExperimentDialog(QtWidgets.QDialog):
    def __init__(self,parent):
        super().__init__(parent);self.host=parent
        self.setWindowTitle('ATS 2.0 · Новый эксперимент');self.resize(520,520)
        layout=QtWidgets.QFormLayout(self)
        self.mode=QtWidgets.QComboBox();self.mode.addItems(['Старый грейфер','Крот и контейнеры','Смешанная команда'])
        self.mode.setCurrentIndex(2 if parent.config.logistics is not None and parent.config.logistics.direct_allowed else 1)
        self.laser=QtWidgets.QComboBox();self.laser.addItems(['laser_off','existing_legacy','realistic_reference'])
        self.drones=QtWidgets.QSpinBox();self.drones.setRange(1,6);self.drones.setValue(4)
        self.containers=QtWidgets.QSpinBox();self.containers.setRange(0,64);self.containers.setValue(8)
        self.ports=QtWidgets.QSpinBox();self.ports.setRange(1,16);self.ports.setValue(4)
        self.packs=QtWidgets.QComboBox();self.packs.addItems(['two_batteries','one_battery'])
        self.robot_slots=QtWidgets.QComboBox();self.robot_slots.addItems(['two_slot_krot_160','four_slot_experimental'])
        self.solar=QtWidgets.QComboBox();self.solar.addItems(['20 м² · исходная база','200 м² · solar_field_large'])
        for label,widget in [('Режим',self.mode),('Лазер',self.laser),('Дроны',self.drones),('Контейнеры',self.containers),
                             ('Зарядные порты модулей',self.ports),('Батареи дрона',self.packs),('Слоты Крота',self.robot_slots),('Панели',self.solar)]:
            layout.addRow(label,widget)
        note=QtWidgets.QLabel('Создаёт новый прогон с текущими настройками рельефа и проекта.\nПараметры Крота и наземного лазера — модельные допущения.\nБольшое поле имеет дополнительную массу и время развёртывания.')
        note.setWordWrap(True);layout.addRow(note)
        button=QtWidgets.QPushButton('Начать новый эксперимент');button.clicked.connect(self.apply);layout.addRow(button)
        save=QtWidgets.QPushButton('Сохранить конфигурацию');save.clicked.connect(self.save);layout.addRow(save)
        load=QtWidgets.QPushButton('Загрузить конфигурацию');load.clicked.connect(self.load);layout.addRow(load)

    def candidate(self):
        from .experiments import experiment_config
        return experiment_config(self.host.config,mode=('legacy','logistics','hybrid')[self.mode.currentIndex()],
            drones=self.drones.value(),containers=self.containers.value(),ports=self.ports.value(),
            battery_modules=2-self.packs.currentIndex(),robot_slots=2 if self.robot_slots.currentIndex()==0 else 4,
            panel_area=20 if self.solar.currentIndex()==0 else 200,laser=self.laser.currentText())

    def apply(self):
        try:self.host.replace_simulation(self.candidate());self.accept()
        except ValueError as error:QtWidgets.QMessageBox.warning(self,'Конфигурация',str(error))

    def save(self):
        from .experiments import save_configuration
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Сохранить эксперимент','','JSON (*.json)')
        if path:
            try:save_configuration(path,self.candidate())
            except (OSError,ValueError) as error:QtWidgets.QMessageBox.warning(self,'Конфигурация',str(error))

    def load(self):
        from .experiments import load_configuration
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Загрузить эксперимент','','JSON (*.json)')
        if path:
            try:self.host.replace_simulation(load_configuration(path));self.accept()
            except (OSError,ValueError,TypeError,KeyError) as error:QtWidgets.QMessageBox.warning(self,'Конфигурация',str(error))
