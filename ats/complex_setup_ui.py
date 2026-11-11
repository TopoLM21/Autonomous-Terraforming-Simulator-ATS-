"""Review equipment, annual work calendar and costs before delivering a complex."""
from copy import deepcopy
from dataclasses import asdict
from html import escape
import math
from pathlib import Path
import threading
import time
import numpy as np
from PySide6 import QtCore,QtWidgets
import pyqtgraph as pg
from .complex_setup import SetupRequest,calculate,place,recipe_config
from .power_design import equipment_copy
from .work_schedule import WorkSchedule
from .winter_power import default_request
from .costs import ITEMS,default_catalog,validate_catalog,quote


STYLE='''
QWidget { background:#101c2b; color:#e1e9f2; font-size:13px; }
QLabel#heading { font-size:23px; font-weight:600; }
QLabel#muted { color:#9badc0; }
QPushButton { background:#253c51; border:1px solid #415d76; border-radius:5px; padding:9px 14px; }
QPushButton:hover { background:#35536d; }
QPushButton:disabled { color:#78899b; background:#1b2a39; }
QPushButton#primary { background:#217b6c; font-weight:600; }
QComboBox,QSpinBox,QDoubleSpinBox,QLineEdit,QDateEdit { background:#203247; border:1px solid #41546b; border-radius:4px; padding:5px; }
QTextBrowser,QTableWidget { background:#122133; border:1px solid #344a60; }
QTabWidget::pane { border:1px solid #344a60; }
QTabBar::tab { background:#1d3045; padding:10px 17px; }
QTabBar::tab:selected { background:#31536b; }
'''


def number(value):return f'{value:,.1f}'.replace(',',' ')


def spin(lo,hi,value,decimals=1):
    w=QtWidgets.QDoubleSpinBox();w.setRange(lo,hi);w.setDecimals(decimals);w.setValue(value);return w


class PriceDialog(QtWidgets.QDialog):
    def __init__(self,catalog,parent=None):
        super().__init__(parent);self.setWindowTitle('Ориентировочные цены оборудования');self.setStyleSheet(STYLE)
        self.original=deepcopy(catalog);layout=QtWidgets.QVBoxLayout(self)
        self.currency=QtWidgets.QComboBox()
        for value in ('USD','RUB','EUR','CU'):self.currency.addItem(value,value)
        self.currency.setCurrentIndex(self.currency.findData(catalog['currency']));layout.addWidget(self.currency)
        note=QtWidgets.QLabel('Начальные цены условные, не котировки производителя. Смена валюты не конвертирует числа: введите цены в выбранной валюте.');note.setWordWrap(True);layout.addWidget(note)
        self.table=QtWidgets.QTableWidget(len(ITEMS),3);self.table.setHorizontalHeaderLabels(['Оборудование','Единица','Цена единицы'])
        self.controls={}
        for row,(key,(name,unit,_)) in enumerate(ITEMS.items()):
            for col,text in enumerate((name,unit)):
                item=QtWidgets.QTableWidgetItem(text);item.setFlags(item.flags()&~QtCore.Qt.ItemFlag.ItemIsEditable);self.table.setItem(row,col,item)
            value=spin(0,1e9,catalog['prices'][key],2);self.table.setCellWidget(row,2,value);self.controls[key]=value
        self.table.horizontalHeader().setSectionResizeMode(0,QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table,1)
        buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok|QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        from .ui_windows import fit_window
        fit_window(self,850,700)

    def value(self):
        value=dict(version=1,currency=self.currency.currentData(),basis='Условные цены модели, редактируемая смета',
                   prices={key:widget.value() for key,widget in self.controls.items()})
        validate_catalog(value);return value


class CalendarReviewDialog(QtWidgets.QDialog):
    """Review the accepted annual design without proposing different hardware."""
    def __init__(self,host):
        super().__init__(host);self.setWindowTitle('Рабочий календарь и энергокомплект');self.setStyleSheet(STYLE)
        c=host.config;d=c.energy_design;s=WorkSchedule.from_data(c.operating_schedule)
        from .winter_power import design_status
        _,status=design_status(c,day_of_year=host.sim.day_of_year)
        layout=QtWidgets.QVBoxLayout(self);self.summary=QtWidgets.QTextBrowser();layout.addWidget(self.summary)
        rows=[('Принятый расчёт',status),('Рабочий период',d['calendar']['label']),
            ('График','Круглосуточно' if s.mode=='continuous' else 'Весь световой день' if s.mode=='daylight' else f'{s.hours:g} ч днём'),
            ('Часы за период',number(d['calendar']['annual_work_hours'])),
            ('Текущая широта',f'{c.base.latitude_deg:+.4f}°'),('Ограничивающий день',d['limiting_date']),
            ('Проверено дней',str(d['checked_days'])),('Панели',number(c.base.panel_area)+' м²'),
            ('Накопители базы',number(c.base.battery/3600000)+' кВт·ч'),
            ('Расчётный расход в контрольные сутки',number(d['limiting']['demand_kwh'])+' кВт·ч'),
            ('Расчётный солнечный приход',number(d['limiting']['solar_kwh'])+' кВт·ч'),
            ('Сменных комплектов на дрон',str(c.logistics.drone_spare_sets))]
        estimate=quote(c,layout=host.sim.base.layout_spec,world=host.sim.logistics)
        if estimate:rows.append(('Текущее оборудование',number(estimate['total'])+' '+estimate['currency']))
        self.summary.setHtml('<h2>Принятая комплектация</h2><table cellspacing="9">'+''.join(
            '<tr><td>'+escape(k)+'</td><td><b>'+escape(v)+'</b></td></tr>' for k,v in rows)+
            '</table><p>Диспетчер выдаёт новые задания в рабочую смену. Завершение рейса, безопасный возврат, зарядка и обслуживание продолжаются вне смены.</p>'+
            '<p>Расчёт использует принятые доли нагрузки, потери и солнечную модель. Это не гарантия работы в любую погоду. Ручные изменения оборудования не пересчитывают комплектацию автоматически.</p>')
        close=QtWidgets.QPushButton('Закрыть');close.clicked.connect(self.accept);layout.addWidget(close)
        from .ui_windows import fit_window
        fit_window(self,850,700)


class _Signals(QtCore.QObject):
    ready=QtCore.Signal(int,object)
    failed=QtCore.Signal(int,str)
    done=QtCore.Signal()


class _Calculation(QtCore.QRunnable):
    def __init__(self,generation,source,request,schedule,power,catalog,cancel):
        super().__init__();self.signals=_Signals();self.arguments=(source,request,schedule,power,catalog)
        self.generation=generation;self.cancel=cancel

    def run(self):
        try:
            result=calculate(*self.arguments,cancelled=self.cancel.is_set)
            self.signals.ready.emit(self.generation,result)
        except Exception as error:self.signals.failed.emit(self.generation,str(error))
        finally:self.signals.done.emit()


class ComplexSetupDialog(QtWidgets.QDialog):
    def __init__(self,host):
        super().__init__(host);self.host=host;self.source=equipment_copy(host.config)
        self.catalog=deepcopy(getattr(host.config,'cost_catalog',None) or default_catalog())
        self.result=None;self.generation=0;self.job=None;self._pending=False
        self.setWindowTitle('Собрать и разместить комплекс');self.setStyleSheet(STYLE)
        layout=QtWidgets.QVBoxLayout(self)
        title=QtWidgets.QLabel('Состав → график → размещение');title.setObjectName('heading');layout.addWidget(title)
        intro=QtWidgets.QLabel('Расчёт не создаёт технику. Комплекс появится на карте только после размещения.');intro.setObjectName('muted');intro.setWordWrap(True);layout.addWidget(intro)
        self.tabs=QtWidgets.QTabWidget();layout.addWidget(self.tabs,1)
        equipment=QtWidgets.QWidget();form=QtWidgets.QFormLayout(equipment);self.scroll_tab(equipment,'1 · Техника')
        self.name=QtWidgets.QLineEdit('Рабочий комплекс');form.addRow('Название',self.name)
        self.drones=QtWidgets.QSpinBox();self.drones.setRange(0,16);self.drones.setValue(4);form.addRow('Дроны-копатели',self.drones)
        self.krot=QtWidgets.QSpinBox();self.krot.setRange(0,10);self.krot.setToolTip('Независимые машины с общим инвентарём, зарядкой и курьером');form.addRow('Кроты',self.krot)
        self.containers=QtWidgets.QSpinBox();self.containers.setRange(0,128);self.containers.setValue(8);self.containers.setEnabled(False);form.addRow('Ёмкости для Кротов, всего',self.containers)
        self.model_info=QtWidgets.QLabel();self.model_info.setWordWrap(True);form.addRow(self.model_info)
        details=QtWidgets.QPushButton('Подробные параметры дронов и модулей…');details.clicked.connect(self.edit_equipment);form.addRow(details)
        note=QtWidgets.QLabel('Панели, накопители, сменные батареи, зарядные места и носители подбираются по графику. При работе с Кротом дроны смогут временно становиться курьерами.');note.setWordWrap(True);form.addRow(note)
        location=QtWidgets.QWidget();calendar=QtWidgets.QFormLayout(location);self.scroll_tab(location,'2 · График и широта')
        self.latitude=spin(-90,90,self.source.base.latitude_deg,4)
        self.latitude.setEnabled(not bool(self.source.georeference and self.source.georeference['source'].get('crs')=='EPSG:4326'))
        calendar.addRow('Широта участка, °',self.latitude)
        self.period=QtWidgets.QComboBox()
        for label,value in [('Март–сентябрь','season'),('Полгода от начальной даты периода','half'),('Весь год','year'),('Свой период','custom')]:self.period.addItem(label,value)
        calendar.addRow('Рабочая часть года',self.period)
        self.first=QtWidgets.QDateEdit(QtCore.QDate(2025,3,1));self.last=QtWidgets.QDateEdit(QtCore.QDate(2025,9,30))
        for widget in (self.first,self.last):
            widget.setDisplayFormat('dd.MM');widget.setCalendarPopup(True)
            widget.setDateRange(QtCore.QDate(2025,1,1),QtCore.QDate(2025,12,31))
        dates=QtWidgets.QHBoxLayout();dates.addWidget(self.first);dates.addWidget(QtWidgets.QLabel('—'));dates.addWidget(self.last)
        calendar.addRow('Включительно; возможен переход через Новый год',dates)
        self.mode=QtWidgets.QComboBox()
        for label,value in [('Круглосуточно, включая ночь','continuous'),('Заданное число часов днём','hours'),('Весь доступный световой день','daylight')]:self.mode.addItem(label,value)
        calendar.addRow('Режим работы',self.mode)
        self.hours=spin(.25,24,6.,2);self.hours.setEnabled(False);calendar.addRow('Минимальная дневная смена, ч',self.hours)
        self.daylight_info=QtWidgets.QLabel();self.daylight_info.setWordWrap(True);calendar.addRow(self.daylight_info)
        sim=getattr(host,'sim',None)
        day=sim.day_of_year if sim else self.source.base.start_day_of_year
        hour=sim.hour if sim else self.source.start_hour
        self.date=QtWidgets.QDateEdit(QtCore.QDate(2025,1,1).addDays(day-1));self.date.setCalendarPopup(True)
        self.date.setDateRange(QtCore.QDate(2025,1,1),QtCore.QDate(2025,12,31))
        self.start_hour=spin(0,23.999,hour,2)
        self.date.setEnabled(sim is None);self.start_hour.setEnabled(sim is None)
        calendar.addRow('Дата начала миссии',self.date);calendar.addRow('Начальное солнечное время, ч',self.start_hour)
        self.auto_place=QtWidgets.QCheckBox('Автоматически выбрать безопасную стоянку');self.auto_place.setChecked(True);calendar.addRow(self.auto_place)
        self.x=spin(0,1e8,self.source.base.x,2);self.y=spin(0,1e8,self.source.base.y,2)
        xy=QtWidgets.QHBoxLayout();xy.addWidget(self.x);xy.addWidget(self.y);calendar.addRow('X / Y базы на локальной карте, м',xy)
        self.x.setEnabled(False);self.y.setEnabled(False)
        self.advanced=QtWidgets.QCheckBox('Параметры расчётной нагрузки');calendar.addRow(self.advanced)
        advanced=QtWidgets.QWidget();advanced_form=QtWidgets.QFormLayout(advanced);advanced.hide();self.advanced.toggled.connect(advanced.setVisible);calendar.addRow(advanced)
        defaults=default_request();self.power_controls={}
        for key,label,value,lo,hi in [('drone_utilization','Доля рабочего времени дрона в полёте, %',defaults.drone_utilization*100,1,99),
              ('excavator_utilization','Загрузка Крота, %',defaults.excavator_utilization*100,1,99),
              ('payload_fraction','Расчётное заполнение грейфера, %',100,0,100),
              ('solar_factor','Коэффициент солнечного прихода, %',defaults.solar_factor*100,1,100),
              ('energy_margin','Запас суточной энергии, %',defaults.energy_margin*100,0,200),
              ('reserve_soc','Резерв накопителей базы, %',defaults.reserve_soc*100,0,90)]:
            w=spin(lo,hi,value);self.power_controls[key]=w;advanced_form.addRow(label,w)
        review=QtWidgets.QWidget();review_layout=QtWidgets.QVBoxLayout(review);self.tabs.addTab(review,'3 · Комплектация и смета')
        self.review=QtWidgets.QTextBrowser();review_layout.addWidget(self.review,1)
        self.chart=pg.PlotWidget(background='#122133');self.chart.addLegend();self.chart.setLabel('bottom','Местное солнечное время',units='ч');self.chart.setLabel('left','Мощность',units='кВт');self.chart.setMinimumHeight(160);self.chart.setMaximumHeight(230)
        self.solar_line=self.chart.plot(pen=pg.mkPen('#71dcb0',width=2),name='Генерация');self.load_line=self.chart.plot(pen=pg.mkPen('#ffca7b',width=2),name='Нагрузка');review_layout.addWidget(self.chart)
        self.status=QtWidgets.QLabel('Выберите состав и график, затем выполните расчёт.');self.status.setWordWrap(True);layout.addWidget(self.status)
        footer=QtWidgets.QGridLayout();self.prices_button=QtWidgets.QPushButton('Цены…');self.prices_button.clicked.connect(self.edit_prices);footer.addWidget(self.prices_button,0,0)
        self.calculate_button=QtWidgets.QPushButton('Рассчитать комплектацию');self.calculate_button.clicked.connect(self.calculate);footer.addWidget(self.calculate_button,0,1)
        self.cancel_button=QtWidgets.QPushButton('Отменить расчёт');self.cancel_button.setEnabled(False);self.cancel_button.clicked.connect(self.cancel_calculation);footer.addWidget(self.cancel_button,0,2)
        self.save_button=QtWidgets.QPushButton('Сохранить комплектацию…');self.save_button.setEnabled(False);self.save_button.clicked.connect(self.save_template);footer.addWidget(self.save_button,1,0,1,2)
        self.place_button=QtWidgets.QPushButton('Разместить комплекс');self.place_button.setObjectName('primary');self.place_button.setEnabled(False);self.place_button.clicked.connect(self.place);footer.addWidget(self.place_button,1,2);layout.addLayout(footer)
        self.start_after=QtWidgets.QCheckBox('Запустить симуляцию после размещения');self.start_after.setChecked(True);layout.addWidget(self.start_after)
        for widget in (self.drones,self.containers,self.hours,self.latitude,self.start_hour,self.x,self.y,*self.power_controls.values()):widget.valueChanged.connect(self.dirty)
        self.name.textChanged.connect(self.dirty)
        self.first.dateChanged.connect(self.first_changed)
        for widget in (self.last,self.date):widget.dateChanged.connect(self.dirty)
        self.mode.currentIndexChanged.connect(self.mode_changed);self.period.currentIndexChanged.connect(self.period_changed)
        self.krot.valueChanged.connect(self.robot_count_changed)
        self.auto_place.toggled.connect(lambda v:(self.x.setEnabled(not v),self.y.setEnabled(not v),self.dirty()))
        self.delivery_timer=QtCore.QTimer(self);self.delivery_timer.setInterval(250);self.delivery_timer.timeout.connect(self.finish_delivery)
        self.refresh_preview()
        from .ui_windows import fit_window
        fit_window(self,1120,850)

    def scroll_tab(self,content,title):
        scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(content);self.tabs.addTab(scroll,title)

    def schedule(self):return WorkSchedule(self.mode.currentData(),self.first.date().dayOfYear(),self.last.date().dayOfYear(),self.hours.value())

    def dirty(self,*args):
        self.cancel_delivery()
        self.generation+=1
        if self.job:self.job.cancel.set()
        self.result=None;self.place_button.setEnabled(False);self.save_button.setEnabled(False)
        self.status.setText('Параметры изменены — требуется расчёт.');self.refresh_preview()

    def refresh_preview(self):
        self.model_info.setText(f'В проекте: дроны — {self.drones.value()}, Кроты — {self.krot.value()}. Техника ещё не размещена.')
        try:
            info=self.schedule().summary(self.latitude.value())
            self.daylight_info.setText(f"Период {info['label']}: {info['days']} дней. Самый короткий световой день — {info['min_daylight_hours']:.2f} ч ({info['shortest_date']}).\nРабота: {info['min_work_hours']:.2f}–{info['max_work_hours']:.2f} ч/сут; {info['annual_work_hours']:.0f} ч за период.")
        except ValueError as error:self.daylight_info.setText(str(error))

    def robot_count_changed(self,value):
        self.containers.setEnabled(value>0)
        if value:self.containers.setValue(max(self.containers.value(),8*value))
        self.dirty()

    def mode_changed(self):self.hours.setEnabled(self.mode.currentData()=='hours');self.dirty()

    def period_changed(self):
        value=self.period.currentData()
        if value=='season':self.first.setDate(QtCore.QDate(2025,3,1));self.last.setDate(QtCore.QDate(2025,9,30))
        elif value=='year':self.first.setDate(QtCore.QDate(2025,1,1));self.last.setDate(QtCore.QDate(2025,12,31))
        elif value=='half':
            end=self.first.date().addMonths(6).addDays(-1);self.last.setDate(QtCore.QDate(2025,end.month(),end.day()))
        self.dirty()

    def first_changed(self):
        if self.period.currentData()=='half':
            end=self.first.date().addMonths(6).addDays(-1);self.last.setDate(QtCore.QDate(2025,end.month(),end.day()))
        self.dirty()

    def edit_equipment(self):
        from .equipment_ui import EquipmentDialog
        try:
            c=recipe_config(self.source,SetupRequest(max(1,self.drones.value()),bool(self.krot.value()),self.containers.value(),excavators=self.krot.value()),self.catalog)
            dialog=EquipmentDialog(c,self)
            if dialog.exec()==QtWidgets.QDialog.DialogCode.Accepted:
                c=dialog.value();self.source=equipment_copy(c);self.drones.setValue(c.fleet.count);self.dirty()
        except ValueError as error:self.status.setText(str(error))

    def edit_prices(self):
        dialog=PriceDialog(self.catalog,self)
        if dialog.exec()==QtWidgets.QDialog.DialogCode.Accepted:
            self.catalog=dialog.value()
            if self.result:
                self.result.config.cost_catalog=deepcopy(self.catalog);self.result.estimate=quote(self.result.config);self.show_result()

    def calculate(self):
        if self.job:return
        try:
            request=SetupRequest(self.drones.value(),bool(self.krot.value()),self.containers.value(),self.name.text(),self.krot.value());request.validate()
            schedule=self.schedule();schedule.validate();source=equipment_copy(self.source)
            source.base.latitude_deg=self.latitude.value();source.base.start_day_of_year=self.date.date().dayOfYear();source.start_hour=self.start_hour.value()
            power=default_request()
            for key,w in self.power_controls.items():setattr(power,key,w.value()/100)
            self.result=None;self.place_button.setEnabled(False);self.save_button.setEnabled(False)
            self.job=_Calculation(self.generation,source,request,schedule,power,deepcopy(self.catalog),threading.Event())
            self.job.signals.ready.connect(self.calculated);self.job.signals.failed.connect(self.failed);self.job.signals.done.connect(self.calculation_finished)
            self.calculate_button.setEnabled(False);self.cancel_button.setEnabled(True);self.status.setText('Проверка каждого дня периода, ночного запаса, зарядки и носителей…')
            QtCore.QThreadPool.globalInstance().start(self.job)
        except ValueError as error:self.status.setText(str(error))

    def calculated(self,generation,result):
        if generation!=self.generation or self.job is None or self.job.cancel.is_set():return
        self.result=result;result.config.cost_catalog=deepcopy(self.catalog);result.estimate=quote(result.config)
        self.show_result();self.tabs.setCurrentIndex(2);self.place_button.setEnabled(True);self.save_button.setEnabled(True)

    def failed(self,generation,message):
        if generation==self.generation:self.status.setText(message)

    def calculation_finished(self):
        self.job=None;self.calculate_button.setEnabled(True);self.cancel_button.setEnabled(False)

    def cancel_calculation(self):
        if self.job:self.job.cancel.set();self.status.setText('Расчёт отменяется; техника не создавалась.')

    def show_result(self):
        from .base_layout import configure_layout
        from .scheduled_power import day_request
        from .power_design import calculate_power,PowerRequest
        from .winter_power import day_profile
        r=self.result;c=r.config;d=r.design;cost=r.estimate;spec=configure_layout(c)
        dimensions=spec.get('grid',{}).get('deployed_dimensions_local_m',spec['dimensions_m'])
        rows=[('Техника',f'Дроны: {c.fleet.count} · Кроты: {c.logistics.robot_count}'),
              ('Рабочий период',f"{d['calendar']['label']} · {d['checked_days']} дней · {c.base.latitude_deg:+.4f}°"),
              ('Часы работы',f"{d['calendar']['min_work_hours']:.2f}–{d['calendar']['max_work_hours']:.2f} ч/сут"),
              ('Ограничивающий день',d['limiting_date']),('Солнечные панели',f'{c.base.panel_area:,.0f} м²'),
              ('Накопители базы',f'{c.base.battery/3600000:,.0f} кВт·ч'),
              ('Запасных комплектов на дрон',str(c.logistics.drone_spare_sets)),('Посадочных мест',str(c.fleet.charge_slots)),
              ('Самоходные носители',str(spec['platform_count'])),('Раскрытый габарит', ' × '.join(number(v) for v in dimensions)+' м'),
              ('Масса оборудования',number(spec['packed_gross_mass_kg'])+' кг'),
              ('Ориентировочная стоимость',number(cost['total'])+' '+cost['currency'])]
        html='<h2>'+escape(r.request.name)+'</h2><table cellspacing="7">'+''.join('<tr><td>'+escape(k)+'</td><td><b>'+escape(v)+'</b></td></tr>' for k,v in rows)+'</table>'
        html+='<p>Проверено восполнение энергии на повторных сутках всех дней периода. Солнечная модель и доли загрузки — расчётные предположения, погода не гарантируется.</p><h3>Смета оборудования</h3><table cellspacing="6">'
        for row in cost['rows']:html+=f"<tr><td>{escape(row['name'])}</td><td>{number(row['quantity'])} {escape(row['unit'])}</td><td>{number(row['amount'])}</td></tr>"
        html+='</table><p>'+escape(cost['basis'])+'. Эксплуатационные расходы, персонал и топливо в эту смету не входят.</p>'
        self.review.setHtml(html)
        request=day_request(PowerRequest(**d['request']),WorkSchedule.from_data(d['schedule']),c.base.latitude_deg,d['limiting_day_of_year'])
        ref=calculate_power(c,request);generation,load=day_profile(c,request,d['limiting_day_of_year'],ref)
        hours=(c.start_hour+(np.arange(1440)+.5)/60)%24;order=np.argsort(hours)
        self.solar_line.setData(hours[order],generation[order]);self.load_line.setData(hours[order],load[order])
        self.status.setText(f"Готово к размещению · {number(cost['total'])} {cost['currency']}. На карте пока ничего не добавлено.")

    def save_template(self,path=None):
        if self.result is None:return
        try:
            if not isinstance(path,(str,Path)):path=QtWidgets.QFileDialog.getSaveFileName(self,'Сохранить комплектацию','complex.atstemplate','Комплекс ATS (*.atstemplate)')[0]
            if path:self.result.template().save(path);self.status.setText('Комплектация сохранена; техника не размещена.')
        except (ValueError,OSError) as error:self.status.setText(str(error))

    def place(self):
        if self.result is None:return
        sims=[u.simulation for u in self.host.complex_project.units] if self.host.complex_project else [self.host.sim] if getattr(self.host,'sim',None) else []
        self._delivery_previous=(getattr(self.host,'running',False),[(s,s.landing_requested) for s in sims])
        self._pending=True;self.place_button.setEnabled(False);self.finish_delivery()

    def cancel_delivery(self):
        if not self._pending:return
        self._pending=False;self.delivery_timer.stop()
        previous=getattr(self,'_delivery_previous',None)
        if previous and not getattr(self.host,'planning_only',False):
            for sim,landing in previous[1]:sim.landing_requested=landing
            self.host._after_file_action(previous[0])

    def finish_delivery(self):
        if not self._pending or self.result is None:return
        try:
            position=None if self.auto_place.isChecked() else [self.x.value(),self.y.value()]
            if getattr(self.host,'planning_only',False):
                session=place(self.result,self.host.landscape,position=position)
                self._pending=False;self.host.open_session(session,self.start_after.isChecked());self.accept();return
            from .complexes import ComplexProject
            project=self.host.complex_project
            if project is None:
                project=ComplexProject(self.host.sim,self.host.landscape);self.host.complex_project=project
            template=self.result.template();site=project.enable_shared();side=site.core.height.shape[0]*site.core.cell
            area=sum(u.simulation.config.base.panel_area+150 for u in project.units)+self.result.config.base.panel_area+150
            required=max(side,math.ceil(max(2*math.sqrt(area),self.result.design['required_map_side_m'])/32)*32)
            if required>1024:raise ValueError('Для ещё одного комплекса нужна карта больше 1024 м; уменьшите комплектацию или число комплексов')
            if required>side:
                if not all(d.grounded and abs(d.cargo)<=1e-8 for u in project.units for d in u.simulation.drones):
                    for u in project.units:u.simulation.request_landing()
                    self.host.running=True;self.host.last_wall=time.perf_counter();self.host.play.setText('⏸ Pause')
                    self.status.setText('Техника завершает перевозки и садится перед увеличением общей карты. Время и прогресс сохраняются.')
                    self.delivery_timer.start();return
                site.enlarge(required)
            unit=project.add_shared(template,position=position);project.templates.append(template)
            path=self.host._session_path;self.host.install_simulation(project.active.simulation,project.landscape,project);self.host._session_path=path
            for sim,landing in self._delivery_previous[1]:sim.landing_requested=landing
            self.host._after_file_action(self.start_after.isChecked());self.host.statusBar().showMessage(f'Размещён {unit.name}; инвентарь и стоимость учтены.',10000)
            self._pending=False;self.delivery_timer.stop();self.accept()
        except (ValueError,OSError,KeyError) as error:
            self.cancel_delivery();self.place_button.setEnabled(True);self.status.setText(str(error))

    def closeEvent(self,event):
        self.cancel_calculation();self.cancel_delivery();super().closeEvent(event)

    def reject(self):
        self.cancel_calculation();self.cancel_delivery();super().reject()


class WorkspaceWindow(QtWidgets.QMainWindow):
    """An actual empty site: terrain/configuration exist, vehicles do not."""
    planning_only=True
    sim=None
    complex_project=None

    def __init__(self,config,landscape=None):
        super().__init__();self.config=equipment_copy(config);self.landscape=landscape
        self.template_library=[];self.running=False;self.preview=None;self.builder=None
        self.setWindowTitle('ATS 2.0 · Новый проект');self.setStyleSheet(STYLE)
        central=QtWidgets.QWidget();self.setCentralWidget(central);layout=QtWidgets.QVBoxLayout(central)
        title=QtWidgets.QLabel('Новый проект');title.setObjectName('heading');layout.addWidget(title)
        subtitle=QtWidgets.QLabel('На площадке пока нет техники. Выберите участок, соберите комплекс и проверьте график работы.');subtitle.setObjectName('muted');subtitle.setWordWrap(True);layout.addWidget(subtitle)
        controls=QtWidgets.QHBoxLayout()
        self.build_button=QtWidgets.QPushButton('＋ Собрать комплекс');self.build_button.setObjectName('primary');self.build_button.clicked.connect(self.show_composer)
        self.landscape_button=QtWidgets.QPushButton('Ландшафт и маршрут…');self.landscape_button.clicked.connect(self.show_landscape)
        self.load_button=QtWidgets.QPushButton('Открыть сохранение…');self.load_button.clicked.connect(self.load_session)
        self.template_button=QtWidgets.QPushButton('Открыть комплектацию…');self.template_button.clicked.connect(self.load_template)
        for w in (self.build_button,self.landscape_button,self.template_button,self.load_button):controls.addWidget(w)
        controls.addStretch();layout.addLayout(controls)
        self.inventory_status=QtWidgets.QLabel('На карте: 0 комплексов · 0 дронов · 0 Кротов · стоимость оборудования 0')
        layout.addWidget(self.inventory_status)
        self.map=pg.PlotWidget(background='#122133');self.map.setAspectLocked(True);self.map.setLabel('bottom','X',units='м');self.map.setLabel('left','Y',units='м')
        self.image=pg.ImageItem(axisOrder='row-major');self.overlay=pg.ImageItem(axisOrder='row-major');self.map.addItem(self.image);self.map.addItem(self.overlay)
        layout.addWidget(self.map,1)
        self.site_info=QtWidgets.QLabel();self.site_info.setWordWrap(True);layout.addWidget(self.site_info)
        self.preview=self.make_preview();self.refresh_site()
        from .ui_windows import fit_window
        fit_window(self,1180,820)

    def make_preview(self):
        from .terrain import Terrain
        return Terrain(self.config)

    def refresh_site(self):
        from .map_visuals import terrain_rgb
        from .site_ground import marine_mask
        t=self.preview;side=t.cell*t.height.shape[0];stride=max(1,t.height.shape[0]//768)
        rgb=terrain_rgb(t.height[::stride,::stride],t.cell*stride,self.config.trench.depth,
            self.config.terrain.dump_max_height,reference=t.reference[::stride,::stride],marine=marine_mask(t)[::stride,::stride])
        self.image.setImage(rgb,autoLevels=False);rect=QtCore.QRectF(0,0,side,side);self.image.setRect(rect)
        overlay=np.zeros((*t.height.shape,4),dtype=np.uint8);overlay[t.excavation]=[65,210,250,80];overlay[t.fill_mask]=[112,242,132,80];overlay[t.dump_mask]=[222,150,70,50]
        self.overlay.setImage(overlay[::stride,::stride],autoLevels=False);self.overlay.setRect(rect)
        self.map.setRange(xRange=(0,side),yRange=(0,side),padding=.02)
        self.site_info.setText(f'Участок {side:g} × {side:g} м · ячейка {t.cell:g} м · широта {self.config.base.latitude_deg:+.4f}°. Синим показана выемка, зелёным — засыпка, оранжевым — область отвала. Предпросмотр не выполняет земляные работы.')

    def replace_simulation(self,config):
        # LandscapeDialog uses the same callback to prepare a site. There is no
        # Simulation in this window and no hidden base or battery inventory.
        from .terrain import Terrain
        preview=Terrain(config)
        self.config=config;self.preview=preview
        if self.builder:self.builder.close();self.builder=None
        self.refresh_site()

    def commit_landscape(self):
        pass  # preview is read-only; no second terrain ledger is committed

    def show_composer(self):
        if self.builder:self.builder.close()
        self.builder=ComplexSetupDialog(self);self.builder.show();self.builder.raise_()

    def show_landscape(self):
        from .landscape_ui import LandscapeDialog
        self.landscape_window=LandscapeDialog(self);self.landscape_window.show()

    def load_template(self,path=None):
        from .complexes import ComplexTemplate
        try:
            if not isinstance(path,(str,Path)):path=QtWidgets.QFileDialog.getOpenFileName(self,'Открыть комплектацию','','Комплекс ATS (*.atstemplate)')[0]
            if not path:return
            template=ComplexTemplate.load(path)
            if template.stock:
                raise ValueError('В шаблоне есть дополнительный физический запас батарей. Для его полного сохранения используйте «Комплексы · общая карта» → импорт шаблона в работающем проекте. Сборщик не заменил этот запас стандартной рецептурой.')
            self.show_composer();b=self.builder
            # Hardware is a recipe: loading it still does not deliver equipment.
            b.source=equipment_copy(self.config)
            for key in ('base','drone','fleet','logistics'):setattr(b.source,key,deepcopy(getattr(template.config,key)))
            b.name.setText(template.name);b.drones.setValue(template.config.fleet.count)
            b.krot.setValue(template.config.logistics.robot_count if template.config.logistics else 0)
            if template.config.logistics:b.containers.setValue(template.config.logistics.container_count)
            if template.config.cost_catalog:b.catalog=deepcopy(template.config.cost_catalog)
            schedule=getattr(template.config,'operating_schedule',None)
            if schedule:
                s=WorkSchedule.from_data(schedule);b.period.setCurrentIndex(b.period.findData('custom'))
                b.first.setDate(QtCore.QDate(2025,1,1).addDays(s.first_day-1));b.last.setDate(QtCore.QDate(2025,1,1).addDays(s.last_day-1))
                b.mode.setCurrentIndex(b.mode.findData(s.mode));b.hours.setValue(s.hours)
            b.dirty()
        except (ValueError,OSError,TypeError) as error:self.site_info.setText(str(error))

    def load_session(self,path=None):
        try:
            if not isinstance(path,(str,Path)):path=QtWidgets.QFileDialog.getOpenFileName(self,'Открыть сохранение','','Состояние ATS (*.atsstate)')[0]
            if not path:return
            from .checkpoint import load_state
            self.open_session(load_state(path),False)
        except (ValueError,OSError,TypeError) as error:self.site_info.setText(str(error))

    def open_session(self,session,start=False):
        from .gui import MainWindow
        if hasattr(self,'landscape_window'):self.landscape_window.close()
        self.work_window=MainWindow(session.simulation.config,session=session)
        self.work_window.show()
        if start:self.work_window.toggle_play()
        self.hide()

    def closeEvent(self,event):
        if self.builder:self.builder.close()
        if hasattr(self,'landscape_window'):self.landscape_window.close()
        if self.landscape is not None:self.landscape.cancel_prefetch()
        super().closeEvent(event)
