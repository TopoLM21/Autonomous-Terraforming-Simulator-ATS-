"""Additional ATS 2.0 windows; the existing map and controls remain in place."""
from PySide6 import QtCore,QtWidgets


class LogisticsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent);self.host=parent
        self.setWindowTitle('ATS 2.0 · Склад и диспетчер');self.resize(1180,720)
        layout=QtWidgets.QVBoxLayout(self)
        self.tabs=tabs=QtWidgets.QTabWidget();layout.addWidget(tabs)
        self.status=QtWidgets.QLabel();self.status.setWordWrap(True)
        tabs.addTab(self.status,'Крот / энергетика')
        inventory_page=QtWidgets.QWidget();box=QtWidgets.QVBoxLayout(inventory_page)
        self.filter=QtWidgets.QLineEdit();self.filter.setPlaceholderText('Фильтр: ID, место, состояние, исполнитель');box.addWidget(self.filter)
        self.inventory=QtWidgets.QTableWidget(0,10);box.addWidget(self.inventory)
        self.inventory.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.inventory.setHorizontalHeaderLabels(['ID','Тип','Место','Состояние','Держатель','Масса, кг','Грунт, кг','SOC / энергия','Резерв','Слот зарядки'])
        self.inventory.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        for col,width in enumerate((190,150,150,170,100,90,90,150,130)):self.inventory.setColumnWidth(col,width)
        tabs.addTab(inventory_page,'Инвентарь')
        self.jobs=QtWidgets.QTableWidget(0,8)
        self.jobs.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.jobs.setHorizontalHeaderLabels(['ID','Задание','Приоритет','Статус / этап','Ресурсы','Дрон','ETA, с','Причина'])
        self.jobs.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        for col,width in enumerate((100,230,85,200,250,60,80,180)):self.jobs.setColumnWidth(col,width)
        tabs.addTab(self.jobs,'Диспетчер')
        self.charging=QtWidgets.QTableWidget(0,7)
        self.charging.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.charging.setHorizontalHeaderLabels(['Слот','Модуль','Состояние слота','Место модуля','SOC','Вход, Вт','Предел, Вт'])
        for col,width in enumerate((260,210,150,155,80,95,100)):self.charging.setColumnWidth(col,width)
        tabs.addTab(self.charging,'Зарядные слоты')
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(500)
        self.filter.textChanged.connect(self.refresh)
        self.tabs.currentChanged.connect(self.refresh);self.refresh()

    def showEvent(self,event):
        self.timer.start(500);self.refresh();super().showEvent(event)

    def closeEvent(self,event):
        self.timer.stop();super().closeEvent(event)

    def refresh(self):
        s=self.host.sim;l=s.logistics
        if l is None:
            self.status.setText('Старый режим: физический склад ATS 2.0 отключён. Выберите «Эксперименты ATS 2.0».')
            self.inventory.setRowCount(0);self.jobs.setRowCount(0);self.charging.setRowCount(0);return
        r=s.report(full=False);e=r.get('excavator');service=r['battery_service']
        swaps='; '.join(f"дрон {serial}: {swap['remaining_seconds']:.0f} с · {', '.join(swap['old_ids'])} → {', '.join(swap['ids'])}"
                       for serial,swap in service['drone_swaps'].items()) or 'нет'
        robot=(f"Крот: {e['state']} · {e['reason']}\nПозиция: {e['position']} · масса {e['mass_kg']:.1f} кг\n"
               f"Батареи: {', '.join(e['battery_ids'])} · {e['energy_kwh']:.3f}/{e['capacity_kwh']:.3f} кВт·ч\n"
               f"Ковш: {e['bucket_kg']:.2f} кг · контейнер: {e['container']}\n"
               f"Циклы: {e['cycles']} · смены батарей: {e['swaps']} · ожидание {e['waiting_seconds']/60:.1f} мин\n") if e else 'Крот отключён\n'
        self.status.setText(robot+f"\nКонтейнерные рейсы: {r['container_cycles']} · групповые переносы: {r['collective_lift_count']}\n"
            f"Группа: {r['collective_lift'] or 'нет'}\n"
            f"Зарядные слоты ({service['mode']}): {len(service['ports'])}/{service['capacity']} · вход {service['module_input_power_w']/1000:.2f} кВт\n"
            f"Замена батарей дронов: {swaps} · завершено {service['drone_swap_count']}\n"
            f"Доковая зарядка дронов: {service['dock_charging_drones']}\n"
            f"Очередь: {', '.join(service['queue']) or 'пусто'}\n"
            f"Солнце: {r['solar_power_w']/1000:.2f} кВт · выработка {r['solar_generated_kwh']:.3f} кВт·ч\n"
            f"База: {r['base_kwh']:.2f} кВт·ч · расход {r['consumed_kwh']:.3f} кВт·ч\n"
            f"Лазер: {r['laser_input_power_w']/1000:.2f} → {r['laser_received_power_w']/1000:.2f} кВт\n"
            f"Баланс грунта {r['mass_error_kg']:+.2e} кг · энергии {r['energy_error_j']:+.2e} Дж")
        # Build only the visible table. Telemetry never serializes the archive.
        if self.tabs.currentIndex()==1:
            query=self.filter.text().casefold()
            rows=[i for i in l.inventory.snapshot() if query in str(i).casefold()]
            values=[]
            for item in rows:
                energy=f"{100*item['energy_j']/item['capacity_j']:.1f}% / {item['energy_j']/3600000:.3f}" if 'energy_j' in item else ''
                values.append([item['id'],item['kind'] if item['type']=='Item' else item['type'],item['location'].value,
                    item['state'],item['holder_id'],f"{item['gross_mass_kg']:.2f}",f"{item.get('soil_kg',0):.2f}",energy,item['reserved_by'] or '',
                    l.battery_service.slots[item['id']].id if item['id'] in l.battery_service.slots else item['id'] if item['type']=='ChargingSlot' else ''])
            self.update_table(self.inventory,values)
        elif self.tabs.currentIndex()==2:
            # Slice before conversion, so opening is independent of archive length.
            values=[[j.id,j.kind,j.priority,j.status+' / '+j.phase,', '.join(j.resource_ids),
                     j.executor or '',round(j.eta_seconds),j.reason] for j in l.jobs[-200:]]
            self.update_table(self.jobs,values)
        elif self.tabs.currentIndex()==3:
            rows=l.battery_service.slot_snapshot()
            self.update_table(self.charging,[[p['id'],p['module_id'],p['state'],p['location'],
                f"{100*p['soc']:.1f}%",f"{p['input_power_w']:.1f}",f"{p['rated_w']:.0f}"] for p in rows])

    @staticmethod
    def update_table(table,rows):
        table.setUpdatesEnabled(False)
        try:
            table.setRowCount(len(rows))
            for row,values in enumerate(rows):
                for col,value in enumerate(values):
                    value=str(value);item=table.item(row,col)
                    if item is None:table.setItem(row,col,QtWidgets.QTableWidgetItem(value))
                    elif item.text()!=value:item.setText(value)
        finally:table.setUpdatesEnabled(True)



class ExperimentDialog(QtWidgets.QDialog):
    def __init__(self,parent):
        super().__init__(parent);self.host=parent
        self.setWindowTitle('ATS 2.0 · Новый эксперимент');self.resize(680,780)
        config=parent.config;logistics=config.logistics
        layout=QtWidgets.QFormLayout(self)
        self.mode=QtWidgets.QComboBox();self.mode.addItems(['Старый грейфер','Крот и контейнеры','Смешанная команда'])
        self.mode.setCurrentIndex(2 if parent.config.logistics is not None and parent.config.logistics.direct_allowed else 1)
        self.laser=QtWidgets.QComboBox();self.laser.addItems(['laser_off','existing_legacy','realistic_reference'])
        if logistics:self.laser.setCurrentText(logistics.laser_profile)
        self.drones=QtWidgets.QSpinBox();self.drones.setRange(1,6);self.drones.setValue(config.fleet.count if logistics else 4)
        self.containers=QtWidgets.QSpinBox();self.containers.setRange(0,64);self.containers.setValue(logistics.container_count if logistics else 8)
        self.ports=QtWidgets.QSpinBox();self.ports.setRange(1,64);self.ports.setValue(logistics.charger_ports if logistics else 4)
        self.charge_mode=QtWidgets.QComboBox()
        self.charge_mode.addItem('Индивидуальный слот каждого модуля',True)
        self.charge_mode.addItem('Общая очередь зарядных портов',False)
        self.charge_mode.setCurrentIndex(0 if logistics is None or logistics.individual_chargers else 1)
        self.charge_mode.currentIndexChanged.connect(lambda:self.ports.setEnabled(not self.charge_mode.currentData()))
        self.ports.setEnabled(not self.charge_mode.currentData())
        self.map_size=QtWidgets.QDoubleSpinBox();self.map_size.setRange(32,256);self.map_size.setDecimals(2)
        self.map_size.setSingleStep(16);self.map_size.setValue(config.terrain.n*config.terrain.cell_size);self.map_size.setSuffix(' м')
        self.packs=QtWidgets.QComboBox();self.packs.addItems(['2 установленных модуля','1 установленный модуль'])
        models=config.drone_models()
        if models[0].equipment and sum(b.quantity for b in models[0].equipment.batteries)==1:self.packs.setCurrentIndex(1)
        self.drone_spares=QtWidgets.QSpinBox();self.drone_spares.setRange(0,64)
        self.drone_spares.setValue(logistics.drone_spare_sets if logistics else 1)
        self.drone_spares.setToolTip('На каждого дрона. Один запасной комплект содержит столько же модулей, сколько установлено на дроне.')
        self.robot_slots=QtWidgets.QComboBox();self.robot_slots.addItems(['two_slot_krot_160','four_slot_experimental'])
        if logistics and logistics.excavator_battery_slots==4:self.robot_slots.setCurrentIndex(1)
        self.robot_spares=QtWidgets.QSpinBox();self.robot_spares.setRange(0,128)
        self.robot_spares.setValue(logistics.excavator_battery_count-logistics.excavator_battery_slots if logistics else 2)
        self.robot_spares.setToolTip('Только запасные модули на складе; установленные батареи Крота считаются отдельно.')
        self.idle_digging=QtWidgets.QCheckBox('Свободные транспортные дроны копают')
        self.idle_digging.setChecked(parent.config.logistics.idle_digging if parent.config.logistics else True)
        layout.addRow(self.idle_digging)
        self.solar=QtWidgets.QComboBox();self.solar.addItems(['20 м² · исходная база','200 м² · solar_field_large'])
        if logistics and config.base.panel_area>=200:self.solar.setCurrentIndex(1)
        for label,widget in [('Режим',self.mode),('Лазер',self.laser),('Дроны',self.drones),('Контейнеры',self.containers),
                             ('Сторона карты',self.map_size),('Зарядка модулей',self.charge_mode),
                             ('Портов общей очереди',self.ports),('Батареи на дроне',self.packs),
                             ('Запасных комплектов на дрона',self.drone_spares),('Слоты Крота',self.robot_slots),
                             ('Запасных модулей Крота',self.robot_spares),('Панели',self.solar)]:
            layout.addRow(label,widget)
        note=QtWidgets.QLabel('Создаёт новый прогон с текущими настройками рельефа и проекта.\nЗапасные батареи начинают на складе заряженными; их масса и начальная энергия учитываются в балансе.\nПараметры Крота и наземного лазера — модельные допущения.\nБольшое поле имеет дополнительную массу и время развёртывания.')
        note.setWordWrap(True);layout.addRow(note)
        button=QtWidgets.QPushButton('Начать новый эксперимент');button.clicked.connect(self.apply);layout.addRow(button)
        save=QtWidgets.QPushButton('Сохранить конфигурацию');save.clicked.connect(self.save);layout.addRow(save)
        load=QtWidgets.QPushButton('Загрузить конфигурацию');load.clicked.connect(self.load);layout.addRow(load)

    def candidate(self):
        from .experiments import experiment_config
        from .maps import with_map_size
        config=with_map_size(self.host.config,self.map_size.value())
        return experiment_config(config,mode=('legacy','logistics','hybrid')[self.mode.currentIndex()],
            drones=self.drones.value(),containers=self.containers.value(),ports=self.ports.value(),
            battery_modules=2-self.packs.currentIndex(),robot_slots=2 if self.robot_slots.currentIndex()==0 else 4,
            panel_area=20 if self.solar.currentIndex()==0 else 200,laser=self.laser.currentText(),
            idle_digging=self.idle_digging.isChecked(),drone_spare_sets=self.drone_spares.value(),
            excavator_spare_batteries=self.robot_spares.value(),individual_chargers=self.charge_mode.currentData())

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
