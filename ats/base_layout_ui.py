"""Editable conditional engineering assumptions and the live crawler passport."""
from copy import deepcopy
from PySide6 import QtWidgets
from .base_layout import parameters,solve_layout
from .ui_windows import fit_window

LABELS=dict(platform_length_m='Длина платформы, м',platform_width_m='Ширина платформы, м',
    platform_height_m='Высота платформы, м',platform_payload_kg='Полезная нагрузка платформы, кг',
    frame_base_kg='Рама: постоянная масса, кг',frame_kg_m2='Рама: масса на площадь, кг/м²',drive_mass_kg='Масса привода и гусениц, кг',
    drive_kw='Мощность привода платформы, кВт',platform_gap_m='Промежуток платформ, м',
    track_contact_fraction='Доля опорной длины гусеницы',max_pressure_kpa='Проектное давление, кПа',
    min_track_width_m='Минимальная ширина гусеницы, м',battery_litres_kwh='Объём батарей, л/кВт·ч',
    panel_folded_thickness_m='Толщина сложенных панелей, м',packing_fraction='Заполнение полезного объёма',
    other_equipment_kg_m3='Средняя плотность прочего оборудования, кг/м³',drone_stowed_m3='Транспортный объём дрона, м³',
    landing_pad_m2='Площадь посадочного места, м²',drive_efficiency='КПД привода',rolling_resistance='Сопротивление качению: однородный грунт',
    traction_coefficient='Сцепление: однородный грунт',bearing_capacity_kpa='Несущая способность: однородный грунт, кПа',control_power_w='Потребление управления платформы, Вт',heading_deg='Поворот состава от оси Y, °',auto_heading='Ориентировать вдоль плана')


class BaseLayoutDialog(QtWidgets.QDialog):
    def __init__(self,host):
        super().__init__(host);self.host=host;self.setWindowTitle('Паспорт базы и автокомпоновка платформ')
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('Условная инженерная модель: оборудование, объём, полезная нагрузка и давление определяют число платформ. Для слоистого грунта параметры опоры берутся из материала поверхности. Изменение шасси требует остановки и посадки флота; энергия и время сохраняются.')
        note.setWordWrap(True);layout.addWidget(note)
        tabs=QtWidgets.QTabWidget();layout.addWidget(tabs,1)
        passport=QtWidgets.QWidget();box=QtWidgets.QVBoxLayout(passport);tabs.addTab(passport,'Паспорт и платформы')
        self.summary=QtWidgets.QTextBrowser();box.addWidget(self.summary)
        self.table=QtWidgets.QTableWidget(0,7);self.table.setHorizontalHeaderLabels(['Платформа','Рама, кг','Груз, кг','Полная масса, кг','Гусеницы, м²','Давление, кПа','Привод, кВт'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch);self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers);box.addWidget(self.table)
        scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);tabs.addTab(scroll,'Параметры компоновки')
        widget=QtWidgets.QWidget();form=QtWidgets.QFormLayout(widget);scroll.setWidget(widget);self.controls={}
        for key,value in parameters(host.config.base.layout).items():
            if key=='auto_heading':
                control=QtWidgets.QCheckBox();control.setChecked(bool(value));self.controls[key]=control;form.addRow(LABELS[key],control);continue
            control=QtWidgets.QDoubleSpinBox();control.setDecimals(4);control.setRange(.0001,1_000_000);control.setValue(value)
            if key=='heading_deg':control.setRange(-360,360);control.setValue(value)
            self.controls[key]=control;form.addRow(LABELS[key],control)
        actions=QtWidgets.QHBoxLayout();layout.addLayout(actions)
        preview=QtWidgets.QPushButton('Рассчитать');preview.clicked.connect(self.refresh);actions.addWidget(preview)
        apply=QtWidgets.QPushButton('Применить шасси на текущей базе');apply.clicked.connect(self.apply);actions.addWidget(apply)
        self.info=QtWidgets.QLabel();self.info.setWordWrap(True);layout.addWidget(self.info)
        self.refresh();fit_window(self,1000,680)

    def values(self):return parameters({k:int(v.isChecked()) if isinstance(v,QtWidgets.QCheckBox) else v.value() for k,v in self.controls.items()})

    def refresh(self):
        try:
            from .power_design import equipment_copy
            from .base_layout import plan_heading
            c=equipment_copy(self.host.config);c.base.layout=self.values()
            if c.base.layout['auto_heading']:c.base.layout['heading_deg']=plan_heading(c)
            r=solve_layout(c,self.host.sim.logistics)
            dims=' × '.join(f'{v:.2f}' for v in r['dimensions_m'])
            cog=' / '.join(f'{v:.2f}' for v in r['centre_of_gravity_m'])
            self.summary.setHtml(f"<p>Платформ: <b>{r['platform_count']}</b>; транспортные габариты <b>{dims} м</b></p>"
                f"<p>Оборудование и техника: {r['packed_payload_kg']:.1f} кг; шасси: {r['chassis_mass_kg']:.1f} кг; полностью собранный комплекс: <b>{r['packed_gross_mass_kg']:.1f} кг</b></p>"
                f"<p>Упаковочный объём: {r['packed_volume_m3']:.2f} м³; опорная площадь: {r['contact_area_m2']:.2f} м²; максимум {r['max_pressure_kpa']:.2f} кПа</p>"
                f"<p>Центр тяжести X/Y/Z от базы: {cog} м; приводы: {r['drive_power_kw']:.1f} кВт</p>"
                '<p>Оценка включает всю технику при транспортировке. Фактическая масса базы меняется при вылетах и возвращении оборудования.</p>')
            self.table.setRowCount(r['platform_count'])
            for row,m in enumerate(r['platforms']):
                for col,key in enumerate(('id','dry_mass_kg','payload_kg','gross_mass_kg','contact_area_m2','pressure_kpa','drive_power_kw')):
                    self.table.setItem(row,col,QtWidgets.QTableWidgetItem(f'{m[key]:.2f}'))
            self.info.setText('Параметры условные и редактируемые. Модель проверяет габариты, давление, сцепление, уклоны и энергию движения.')
        except (ValueError,TypeError) as error:self.info.setText(str(error))

    def apply(self):
        try:
            self.host.sim.configure_platforms(self.values());self.host.sync_equipment();self.host.refresh(True);self.refresh()
            self.info.setText('Шасси применено. Новые секции учтены как поставка оборудования; заряды сохранены.')
        except ValueError as error:self.info.setText(str(error))
