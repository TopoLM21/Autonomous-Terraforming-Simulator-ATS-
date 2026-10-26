"""Energy sizing and portable equipment templates; current machinery is untouched."""
from dataclasses import asdict
from html import escape
from pathlib import Path
from PySide6 import QtCore,QtWidgets
import pyqtgraph as pg
import numpy as np
from .power_design import PowerRequest,calculate_power,equipment_copy
from .complexes import ComplexTemplate


def _number(value,digits=1):return f'{value:,.{digits}f}'.replace(',',' ') if value is not None else '—'


class PowerDesignDialog(QtWidgets.QDialog):
    def __init__(self,host):
        super().__init__(host);self.host=host;self.result=None;self.design_source=None
        self.setWindowTitle('Энергоснабжение и шаблон комплекса');self.resize(1180,790)
        saved=getattr(host,'power_request_state',None) or {};request=PowerRequest(**saved)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('Подбор по заданной загрузке или среднему расходу текущего прогона. Часы и доли работы — условия расчёта, а не команды диспетчеру. Шаблон сохраняет комплектацию; действующая симуляция и её заряд сохраняются.')
        note.setWordWrap(True);layout.addWidget(note)
        tabs=QtWidgets.QTabWidget();layout.addWidget(tabs,1)
        sizing=QtWidgets.QWidget();tabs.addTab(sizing,'Подбор комплектации')
        top=QtWidgets.QHBoxLayout(sizing)
        settings=QtWidgets.QWidget();form=QtWidgets.QFormLayout(settings);top.addWidget(settings,1)
        self.controls={}
        specs=[('work_hours','Продолжительность работы, ч/сутки',.1,24,request.work_hours),
            ('start_hour','Начало смены, солнечные часы',0,23.99,host.config.start_hour if request.start_hour is None else request.start_hour),
            ('drone_utilization','Дроны в воздухе, %',0,100,request.drone_utilization*100),
            ('excavator_utilization','Крот: рабочая загрузка, %',0,100,request.excavator_utilization*100),
            ('payload_fraction','Среднее заполнение ковша, %',0,100,request.payload_fraction*100),
            ('solar_factor','Расчётный коэффициент солнца, %',1,100,request.solar_factor*100),
            ('reserve_soc','Остаток заряда базы, %',0,95,request.reserve_soc*100),
            ('energy_margin','Запас генерации, %',0,200,request.energy_margin*100),
            ('auxiliary_kw','Дополнительная нагрузка, кВт',0,10000,request.auxiliary_kw),
            ('courier_minutes','Доставка батарей Крота, мин',0,600,request.courier_minutes),
            ('laser_coverage','Лазер: доля доступного полёта, %',0,100,request.laser_coverage*100),
            ('laser_distance_m','Лазер: расчётная дистанция, м',0,100000,request.laser_distance_m)]
        for key,label,lo,hi,value in specs:
            s=QtWidgets.QDoubleSpinBox();s.setRange(lo,hi);s.setDecimals(2);s.setValue(value)
            form.addRow(label,s);self.controls[key]=s
        self.source=QtWidgets.QComboBox();self.source.addItem('По установленному оборудованию','equipment');self.source.addItem('По расходу текущей симуляции','measured')
        self.source.setCurrentIndex(max(0,self.source.findData(request.source)));form.addRow('Профиль нагрузки',self.source)
        self.name=QtWidgets.QLineEdit('Рабочий комплекс');form.addRow('Название шаблона',self.name)
        self.calculate_button=QtWidgets.QPushButton('Рассчитать');self.calculate_button.clicked.connect(self.calculate);form.addRow(self.calculate_button)
        self.save_button=QtWidgets.QPushButton('Сохранить рассчитанный шаблон…');self.save_button.clicked.connect(self.save_template);form.addRow(self.save_button)
        self.summary=QtWidgets.QTextBrowser();top.addWidget(self.summary,2)
        self.chart=pg.PlotWidget();self.chart.setLabel('bottom','Местное солнечное время',units='ч');self.chart.setLabel('left','Мощность',units='кВт')
        self.chart.addLegend();self.load_curve=self.chart.plot(pen=pg.mkPen('#efce8b',width=2),name='Потребление')
        self.solar_curve=self.chart.plot(pen=pg.mkPen('#7edca2',width=2),name='Расчётная генерация');self.chart.showGrid(x=True,y=True,alpha=.2)
        profile=QtWidgets.QWidget();tabs.addTab(profile,'Суточный профиль и сменные батареи')
        profile_layout=QtWidgets.QVBoxLayout(profile);profile_layout.addWidget(self.chart,1)
        self.table=QtWidgets.QTableWidget(0,6);self.table.setHorizontalHeaderLabels(['Дрон','В полёте, кВт','Работа комплекта, мин','Зарядка, мин','Запасных комплектов','Батарей в комплекте'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers);self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(150);self.table.setMaximumHeight(225);profile_layout.addWidget(self.table)
        self.info=QtWidgets.QLabel();self.info.setWordWrap(True);layout.addWidget(self.info)
        self.calculate()

    def request(self):
        values={k:s.value() for k,s in self.controls.items()}
        for key in ('drone_utilization','excavator_utilization','payload_fraction','solar_factor','reserve_soc','energy_margin','laser_coverage'):values[key]/=100
        return PowerRequest(**values,source=self.source.currentData())

    def calculate(self):
        try:
            request=self.request();self.design_source=equipment_copy(self.host.config)
            observation=self.host.sim.report(full=False) if request.source=='measured' else None
            result=calculate_power(self.design_source,request,observation);self.result=result
            self.host.power_request_state=asdict(request)
            rows=[('Широта / день года',f"{result['latitude_deg']:.3f}° / {result['day_of_year']}"),
                ('Панель для подбора',f"{result['panel_type']['name']} · {result['panel_type']['area_m2']:g} м²/шт. · КПД {result['panel_type']['efficiency']*100:g}%"),
                ('Потребление за сутки',_number(result['daily_energy_kwh'])+' кВт·ч'),
                ('Средняя мощность за сутки',_number(result['mean_power_kw'])+' кВт'),
                ('Панели',f"{result['panel_count'] if result['panel_count'] is not None else 'нет солнечного решения'} шт. / {_number(result['panel_area_m2'])} м²"),
                ('Солнечная энергия за сутки',_number(result['solar_daily_kwh'])+' кВт·ч'),
                ('Аккумуляторы базы',f"{result['base_battery_count']} шт. / {_number(result['base_capacity_kwh'])} кВт·ч"),
                ('Запас энергии во всех полных батареях',_number(result['initial_stored_kwh'])+' кВт·ч'),
                ('Для зарядки всего комплекса с 0 до 100%',_number(result['empty_to_full_input_kwh'])+' кВт·ч на входе'),
                ('Сменные батареи дронов',f"{result['drone_spare_modules']} шт. / {result['drone_spare_sets']} запасных комплектов на дрон"),
                ('Сменные батареи Крота',str(result['excavator_spare_modules'])+' шт.'),
                ('Индивидуальные зарядные слоты',str(result['module_charging_slots'])),
                ('Средняя требуемая мощность зарядки с запасом',_number(result['charging_continuous_kw'])+' кВт'),
                ('Расчётная мощность преобразователя',_number(result['inverter_required_kw'])+' кВт')]
            template=None
            try:
                template=self.make_template();rows.append(('Масса оборудования всего комплекса',_number(template.equipment_mass())+' кг'))
            except ValueError:pass
            self.summary.setHtml('<table cellspacing="9">'+''.join(f'<tr><td>{escape(k)}</td><td><b>{escape(v)}</b></td></tr>' for k,v in rows)+'</table>')
            self.chart.setTitle('Расчётный суточный профиль · ясное небо × выбранный коэффициент')
            self.load_curve.setData(np.arange(24)+.5,result['hourly_load_kw']);self.solar_curve.setData(np.arange(24)+.5,result['hourly_solar_kw'])
            self.table.setRowCount(len(result['drones']))
            for row,d in enumerate(result['drones']):
                values=[d['name'],_number(d['flight_kw']),_number(d['active_minutes']),_number(d['recharge_minutes']),str(d['allocated_spare_sets']),str(d['modules_per_set'])]
                for col,value in enumerate(values):self.table.setItem(row,col,QtWidgets.QTableWidgetItem(value))
            notes=list(result['warnings'])
            if result['storage_check'] is not None:
                notes.append(f"Двухсуточная проверка: недопоставка {result['storage_check']['unserved_kwh']:.6f} кВт·ч; баланс {result['storage_check']['energy_error_kwh']:.2g} кВт·ч.")
            notes.append('Расчёт сменных комплектов предполагает отдельный зарядный слот для каждого модуля. Фактическая работа проверяется симуляцией маршрутов и очередей.')
            self.info.setText('\n'.join(notes));self.save_button.setEnabled(template is not None)
        except (ValueError,TypeError,KeyError) as error:
            self.result=None;self.info.setText(str(error));self.save_button.setEnabled(False)

    def make_template(self):
        if self.result is None:raise ValueError('Сначала выполните расчёт')
        return ComplexTemplate.capture(self.design_source,self.name.text(),self.result)

    def save_template(self,path=None):
        try:
            template=self.make_template()
            if not isinstance(path,(str,Path)):
                path=QtWidgets.QFileDialog.getSaveFileName(self,'Сохранить комплектацию','complex.atstemplate','Комплекс ATS (*.atstemplate)')[0]
            if not path:return
            template.save(path);self.host.template_library.append(template)
            self.info.setText(f'Шаблон сохранён: {path}. Копии создаются через «Комплексы проекта».')
        except (ValueError,OSError) as error:self.info.setText(str(error))

    def closeEvent(self,event):
        self.design_source=None;super().closeEvent(event)
