"""One table row per carrier; individual rack mounts are inspected on demand."""
from PySide6 import QtCore,QtWidgets

STATES=dict(STOWED='Упакована',DEPLOY_SUPPORTS='Установка опор',OPEN_APRON='Открытие площадки',
    CONNECTING='Подключение',WORKING='Работает',WAIT_SERVICE='Завершает обслуживание',
    DISCONNECTING='Отключение сервиса',CLOSE_APRON='Сборка площадки',RETRACT_SUPPORTS='Уборка опор')
ROLES=dict(BESS='Энергетическая',SERVICE='Сервисная')


class BatteryPlatformsWindow(QtWidgets.QDialog):
    def __init__(self,parent):
        super().__init__(parent);self.host=parent;self.setWindowTitle('Батарейные платформы');self.resize(980,600)
        box=QtWidgets.QVBoxLayout(self);self.summary=QtWidgets.QLabel();self.summary.setWordWrap(True);box.addWidget(self.summary)
        self.table=QtWidgets.QTableWidget(0,8);self.table.setHorizontalHeaderLabels(['ID','Назначение','Состояние','Масса, т','Модули','Ёмкость, кВт·ч','В шкафах, кВт·ч','Мощность, кВт'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        box.addWidget(self.table)
        row=QtWidgets.QHBoxLayout();self.scope=QtWidgets.QCheckBox('Только выбранная платформа');row.addWidget(self.scope)
        self.deploy=QtWidgets.QPushButton('Развернуть / отменить упаковку');self.pack=QtWidgets.QPushButton('Упаковать')
        row.addWidget(self.deploy);row.addWidget(self.pack);box.addLayout(row)
        self.deploy.clicked.connect(lambda:self.action('DEPLOY'));self.pack.clicked.connect(lambda:self.action('PACK'))
        self.show_mounts=QtWidgets.QCheckBox('Показать крепления и состав выбранной платформы');box.addWidget(self.show_mounts)
        self.details=QtWidgets.QPlainTextEdit();self.details.setReadOnly(True);self.details.setMaximumHeight(190);box.addWidget(self.details)
        self.message=QtWidgets.QLabel();self.message.setWordWrap(True);box.addWidget(self.message)
        self.table.itemSelectionChanged.connect(self.detail);self.show_mounts.toggled.connect(self.detail)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(750);self.refresh()

    def manager(self):
        w=self.host.sim.logistics
        return w.battery_platforms if w else None

    def closeEvent(self,event):
        self.timer.stop();super().closeEvent(event)

    def action(self,mode):
        p=self.manager()
        try:
            if p is None:raise ValueError('В этой миссии сохранена прежняя компоновка без физических батарейных платформ')
            if getattr(self.host.sim,'regional_stage',None):raise ValueError('Сначала завершите или отмените переезд группы')
            index=self.table.currentRow();id=self.rows[index]['id'] if self.scope.isChecked() and 0<=index<len(self.rows) else None
            if self.scope.isChecked() and id is None:raise ValueError('Выберите платформу в таблице')
            p.request(mode,carrier_id=id)
            self.message.setText('Команда принята. Нужен ход модельного времени и питание. Занятый обмен завершится перед упаковкой; BESS продолжает питать общую шину.')
            self.host.refresh(True)
        except ValueError as error:self.message.setText(str(error))
        self.refresh()

    def refresh(self):
        p=self.manager();self.rows=p.report() if p else []
        index=self.table.currentRow();self.table.blockSignals(True);self.table.setRowCount(len(self.rows))
        for i,r in enumerate(self.rows):
            values=[f"P{r['id']:02}",ROLES[r['role']],STATES.get(r['state'],r['state']),f"{r['actual_mass_kg']/1000:.3f}",str(r['modules']),f"{r['capacity_kwh']:.1f}",f"{r['energy_kwh']:.2f}",f"{r['power_kw']:.1f}"]
            for j,value in enumerate(values):
                item=self.table.item(i,j)
                if item is None:item=QtWidgets.QTableWidgetItem();self.table.setItem(i,j,item)
                item.setText(value);item.setToolTip(r['reason'])
        self.table.blockSignals(False)
        if self.rows:
            if index<0:self.table.selectRow(0)
            self.summary.setText(f"{len(self.rows)} батарейных платформ · закреплённые шкафы BESS и отдельный оборотный запас. "
                f"Общая мощность BESS {p.converter_power_w/1000:g} кВт. Ячейки остаются в шкафах; складываются опоры и проходы. Модельное гусеничное шасси, не серийная модель производителя. {p.issue}")
        else:self.summary.setText('Сохранена прежняя агрегатная компоновка этой миссии. Оборудование и заряд не изменены. Физические батарейные платформы входят в новые комплексы с автокомпоновкой шасси.')
        self.deploy.setEnabled(bool(p));self.pack.setEnabled(bool(p));self.detail()

    def detail(self,*_):
        self.details.setVisible(self.show_mounts.isChecked())
        if not self.show_mounts.isChecked():return
        index=self.table.currentRow()
        if not 0<=index<len(self.rows):self.details.clear();return
        r=self.rows[index];p=self.manager();a=p.world.sim.base.layout_spec['platforms'][r['id']-1]
        vector=lambda values:'('+', '.join(f'{v:.3f}' for v in values)+')'
        lines=[f"P{r['id']:02}: транспортные габариты {vector(r['dimensions_m'])} м; предельный груз {p.world.sim.base.layout_spec['parameters']['platform_payload_kg']:g} кг",
            f"Текущая масса {r['actual_mass_kg']:.1f} кг; максимальная при сборе {r['mass_kg']:.1f} кг; центр тяжести относительно носителя {vector(r['actual_centre_of_gravity_m'])} м",
            f"Опоры {r['supports']:.0%}; площадка {r['apron']:.0%}; подключение {r['connection']:.0%}. {r['reason']}",
            'ID оборудования | крепление | кг | габариты, м | положение на платформе, м | фиксаторы']
        for e in a['loads']:
            lines.append(f"{e['id']} | {e['mount_id']} | {e['mass_kg']:g} | {vector(e['dimensions_m'])} | {vector(e['offset_m'])} | {e['fasteners']}")
        text='\n'.join(lines)
        if self.details.toPlainText()!=text:self.details.setPlainText(text)
