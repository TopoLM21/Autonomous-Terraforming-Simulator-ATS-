"""Conventional diesel fleet: configuration and its separate operating account."""
from PySide6 import QtCore,QtWidgets
from .conventional_integration import fleet_for

STATES={'PARKED':'Стоит','REFUELING':'Заправка','WAIT_FUEL':'Очередь на заправку','DRIVING':'В пути',
        'WAIT_TRUCK':'Ждёт самосвал','DIGGING':'Копает','WAIT_LOAD':'Ждёт погрузку','UNLOADING':'Разгрузка'}


def _number(value,decimals=2):
    return f'{float(value):,.{decimals}f}'.replace(',',' ')


class ConventionalWindow(QtWidgets.QDialog):
    """This window submits commands; only Simulation advances the equipment."""
    def __init__(self,host):
        super().__init__(host);self.host=host;self._world=None;self._simulation=host.sim
        from .conventional import ConventionalConfig
        existing=fleet_for(host.sim)
        self.defaults=existing.config if existing is not None else ConventionalConfig.for_simulation(host.sim)
        self.controls={};self.setWindowTitle('Обычная техника · экскаватор и самосвалы')
        layout=QtWidgets.QVBoxLayout(self)
        intro=QtWidgets.QLabel('Одна обычная бригада на общей физической карте. Её рабочий участок отделён от заданий автономной техники; топливная база и смета собственные. Выбор другого автономного комплекса не создаёт новую бригаду.')
        intro.setWordWrap(True);layout.addWidget(intro)
        self.patch_status=QtWidgets.QLabel();self.patch_status.setWordWrap(True);layout.addWidget(self.patch_status)
        self.tabs=QtWidgets.QTabWidget();layout.addWidget(self.tabs,1)
        self.setup_pages=[]
        machines=self._form_tab('1 · Машины и участок')
        fixed=QtWidgets.QLabel('1 гусеничный экскаватор · отдельные водители и машинисты каждой смены')
        fixed.setWordWrap(True);machines.addRow(fixed)
        self._integer(machines,'truck_count','Самосвалы',1,16)
        self.dimension_controls={}
        for key,label in (('excavator_dimensions_m','Экскаватор: Д × Ш × В, м'),('truck_dimensions_m','Самосвал: Д × Ш × В, м')):
            row=QtWidgets.QHBoxLayout();widgets=[]
            for value in getattr(self.defaults,key):
                widget=self._spin(.5,30.,value);row.addWidget(widget);widgets.append(widget)
            self.dimension_controls[key]=widgets;machines.addRow(label,row)
        self._real(machines,'ramp_width_m','Ширина грунтового съезда, м',2.,100.)
        self.auto_area=QtWidgets.QCheckBox('Автоматически выбрать участок канала и топливную базу')
        self.auto_area.setChecked(bool(getattr(self.defaults,'auto_work_area',False)))
        self.auto_area.setEnabled(bool(getattr(host.sim.config.plan,'channel',None)))
        machines.addRow(self.auto_area)
        extent=host.sim.terrain.height.shape[0]*host.sim.terrain.cell
        self._real(machines,'base_x','База X, м',0,float(extent))
        self._real(machines,'base_y','База Y, м',0,float(extent))
        self.rect=[]
        for label,value,minimum in zip(('X участка, м','Y участка, м','Ширина участка, м','Высота участка, м'),self.defaults.work_rect,(0.,0.,6.,6.)):
            control=self._spin(minimum,float(extent),value);self.rect.append(control);machines.addRow(label,control)
        planned=QtWidgets.QLabel('Прямоугольник вместе с защитным отступом резервируется для обычной бригады в общем плане выемки; дроны получают задания за его границей. Задайте профиль кистью или планом канала. Подъезды и сохраняемый выезд проверяются до начала работ.')
        planned.setWordWrap(True);machines.addRow(planned)
        self.model_info=QtWidgets.QLabel();self.model_info.setWordWrap(True);machines.addRow(self.model_info)
        calendar=self._form_tab('2 · Смены и люди')
        self._integer(calendar,'shifts_per_day','Независимые сменные бригады в сутки',1,3)
        self._real(calendar,'shift_start_hour','Начало первой смены, ч',0,23.99)
        self._real(calendar,'shift_span_hours','Присутствие одной бригады, ч',6,9)
        self._real(calendar,'lunch_after_hours','Обед после начала смены, ч',1,5)
        self._real(calendar,'lunch_hours','Перерыв на обед, ч',1,2)
        days=('Понедельник','Вторник','Среда','Четверг','Пятница','Суббота','Воскресенье')
        for key,label in (('weekly_rest_day','Еженедельный выходной'),('start_weekday','День недели в начале миссии')):
            widget=QtWidgets.QComboBox()
            for i,day in enumerate(days):widget.addItem(day,i)
            widget.setCurrentIndex(getattr(self.defaults,key));self.controls[key]=widget;calendar.addRow(label,widget)
        self.shift_note=QtWidgets.QLabel();self.shift_note.setWordWrap(True);calendar.addRow(self.shift_note)
        law=QtWidgets.QLabel('Модель обычной рабочей недели Египта: не более 8 ч фактического труда в день и 48 ч в неделю, перерыв не менее 1 ч, не более 5 ч непрерывного труда. Сверхурочные и специальные исключения в этой комплектации не используются.')
        law.setWordWrap(True);calendar.addRow(law)
        costs=self._form_tab('3 · Смета и запасы')
        self._real(costs,'egp_per_usd','Курс бюджета, EGP за 1 USD',.01,10000)
        self._real(costs,'excavator_wage_egp_month','Машинист, EGP/месяц',0,1e8)
        self._real(costs,'driver_wage_egp_month','Водитель самосвала, EGP/месяц',0,1e8)
        self._real(costs,'diesel_egp_l','Дизель, EGP/л',0,1e5)
        self._real(costs,'food_egp_unit','Питание, EGP/порция',0,1e5)
        self._real(costs,'water_egp_l','Питьевая вода, EGP/л',0,1e5)
        self._real(costs,'water_l_per_person_hour','Вода, л на человека за час',0,100)
        self._real(costs,'food_units_per_person_shift','Порций на человека за смену',0,20)
        self._real(costs,'fuel_stock_l','Начальный склад топлива, л',0,1e9)
        self._real(costs,'water_stock_l','Начальный склад питьевой воды, л',0,1e9)
        self._real(costs,'food_stock_units','Начальный склад питания, порций',0,1e9)
        self._real(costs,'excavator_purchase_usd','Покупка экскаватора, USD',0,1e9)
        self._real(costs,'truck_purchase_usd','Покупка одного самосвала, USD',0,1e9)
        self._real(costs,'base_purchase_usd','Покупка топливной базы, USD',0,1e9)
        self._real(costs,'maintenance_usd_per_engine_hour','Обслуживание на машино-час, USD',0,1e5)
        basis=QtWidgets.QLabel('Зарплаты — редактируемые ориентиры обследования рынка, а не гарантированная ставка конкретной бригады. Цены машин и расходы лагеря модельные. Курс фиксируется для сметы, не обновляется из интернета. Расходники включаются в стоимость по фактическому использованию. Источники и ограничения: docs/ats2/conventional-sources.md.')
        basis.setWordWrap(True);costs.addRow(basis)
        live=QtWidgets.QWidget();live_layout=QtWidgets.QVBoxLayout(live);self.tabs.addTab(live,'4 · Работа и расходы')
        self.summary=QtWidgets.QLabel('Обычная техника ещё не создана.');self.summary.setWordWrap(True);live_layout.addWidget(self.summary)
        self.table=QtWidgets.QTableWidget(0,8)
        self.table.setHorizontalHeaderLabels(['Машина','Тип','Состояние','Бак, л','Грунт, кг','X / Y, м','Д × Ш × В, м','Причина ожидания'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(7,QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col,width in enumerate((90,90,100,85,85,100)):self.table.setColumnWidth(col,width)
        live_layout.addWidget(self.table,1)
        supplies=QtWidgets.QGroupBox('Внешняя поставка расходников');supply_layout=QtWidgets.QGridLayout(supplies)
        self.supply_fuel=self._spin(0,1e9,1000);self.supply_water=self._spin(0,1e9,200);self.supply_food=self._spin(0,1e9,20)
        for col,(label,widget) in enumerate((('Дизель, л',self.supply_fuel),('Вода, л',self.supply_water),('Питание, порций',self.supply_food))):
            supply_layout.addWidget(QtWidgets.QLabel(label),0,col);supply_layout.addWidget(widget,1,col)
        self.replenish_button=QtWidgets.QPushButton('Принять поставку');self.replenish_button.clicked.connect(self.replenish);supply_layout.addWidget(self.replenish_button,1,3)
        live_layout.addWidget(supplies)
        self.preview=QtWidgets.QLabel();self.preview.setWordWrap(True);layout.addWidget(self.preview)
        actions=QtWidgets.QHBoxLayout();layout.addLayout(actions)
        self.create_button=QtWidgets.QPushButton('Разместить обычную технику');self.create_button.clicked.connect(self.create);actions.addWidget(self.create_button)
        self.pause_button=QtWidgets.QPushButton('Приостановить и вернуться');self.pause_button.clicked.connect(self.toggle);actions.addWidget(self.pause_button)
        self.pause_button.setToolTip('Безопасный возврат парка. Зарплата по действующему договору продолжает начисляться.')
        close=QtWidgets.QPushButton('Закрыть');close.clicked.connect(self.close);actions.addWidget(close)
        self.message=QtWidgets.QLabel('Покупка будет записана после успешного размещения.');self.message.setWordWrap(True);layout.addWidget(self.message)
        self.controls['shifts_per_day'].valueChanged.connect(self.shift_changed)
        for widget in self.controls.values():
            signal=widget.currentIndexChanged if isinstance(widget,QtWidgets.QComboBox) else widget.valueChanged
            signal.connect(self.update_preview)
        for widget in self.rect:widget.valueChanged.connect(self.update_preview)
        for widgets in self.dimension_controls.values():
            for widget in widgets:widget.valueChanged.connect(self.update_preview)
        self.auto_area.toggled.connect(self.placement_mode_changed)
        self.placement_mode_changed()
        self.shift_changed()
        self.timer=QtCore.QTimer(self);self.timer.setInterval(500);self.timer.timeout.connect(self.refresh);self.timer.start()
        from .ui_windows import fit_window
        fit_window(self,1080,800);self.refresh()

    def _form_tab(self,title):
        content=QtWidgets.QWidget();form=QtWidgets.QFormLayout(content)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(content)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame);self.tabs.addTab(scroll,title);self.setup_pages.append(content)
        return form

    @staticmethod
    def _spin(lo,hi,value,decimals=2):
        widget=QtWidgets.QDoubleSpinBox();widget.setRange(lo,hi);widget.setDecimals(decimals);widget.setValue(value);widget.setKeyboardTracking(False)
        return widget

    def _real(self,form,key,label,lo,hi):
        widget=self._spin(lo,hi,getattr(self.defaults,key));self.controls[key]=widget;form.addRow(label,widget)

    def _integer(self,form,key,label,lo,hi):
        widget=QtWidgets.QSpinBox();widget.setRange(lo,hi);widget.setValue(getattr(self.defaults,key));self.controls[key]=widget;form.addRow(label,widget)

    def config_value(self):
        from dataclasses import replace
        values={key:(widget.currentData() if isinstance(widget,QtWidgets.QComboBox) else widget.value()) for key,widget in self.controls.items()}
        values['work_rect']=tuple(widget.value() for widget in self.rect)
        values['auto_work_area']=self.auto_area.isChecked()
        for key,widgets in self.dimension_controls.items():values[key]=tuple(widget.value() for widget in widgets)
        return replace(self.defaults,**values)

    def placement_mode_changed(self,*_):
        manual=not self.auto_area.isChecked()
        for widget in (*self.rect,self.controls['base_x'],self.controls['base_y']):widget.setEnabled(manual)
        self.update_preview()

    def shift_changed(self,*_):
        count=self.controls['shifts_per_day'].value();span=self.controls['shift_span_hours']
        span.setMaximum(8 if count==3 else 9)
        self.update_preview()

    def update_preview(self,*_):
        c=self.config_value();people=(1+c.truck_count)*c.shifts_per_day
        excavator=' × '.join(f'{v:g}' for v in c.excavator_dimensions_m);truck=' × '.join(f'{v:g}' for v in c.truck_dimensions_m)
        self.model_info.setText(f'Габариты Д × Ш × В: экскаватор {excavator} м; самосвал {truck} м; топливная база 8 × 8 × 3 м. На карте корпуса показаны в масштабе; безопасный габарит движения учитывает запас при повороте. Модельные параметры: ковш {c.bucket_m3:g} м³, базовый цикл {c.bucket_cycle_s:g} с; самосвал до {c.truck_payload_kg/1000:g} т / {c.truck_body_m3:g} м³; съезд до {100*c.ramp_gradient:g} %. Начальные баки пусты, заправка {c.refuel_l_min:g} л/мин занимает модельное время.')
        productive=c.shift_span_hours-c.lunch_hours
        self.shift_note.setText(f'{c.shifts_per_day} бригады · {people} работников всего. Одна смена: {c.shift_span_hours:g} ч присутствия, {productive:g} ч труда, {c.lunch_hours:g} ч обеда. Для трёх смен присутствие ограничено 8 ч; это до 7 ч труда на человека при часовом обеде.')
        capital=c.excavator_purchase_usd+c.truck_count*c.truck_purchase_usd+c.base_purchase_usd
        wages=(c.excavator_wage_egp_month+c.truck_count*c.driver_wage_egp_month)*c.shifts_per_day/c.egp_per_usd
        phase='Приобретённый состав' if self.fleet() is not None else 'Перед размещением'
        self.preview.setText(f'{phase}: техника и база ≈ {_number(capital)} USD · персонал {_number(wages)} USD/месяц. Расходы обычной бригады учитываются отдельно от автономного комплекса.')
        self.refresh_patch_status(c)

    def refresh_patch_status(self,config):
        fleet=self.fleet();x,y,w,h=config.work_rect
        geometry=f'X={x:g}, Y={y:g} · {w:g} × {h:g} м'
        if fleet is None:
            if config.auto_work_area:
                self.patch_status.setText('Автоматический участок канала: при размещении будут проверены реальная незавершённая выемка, размеры машин, съезд, безопасная стоянка и пути. До успешной проверки участок и база не приобретаются.')
                return
            self.patch_status.setText(f'Участок к размещению: {geometry}. После размещения дроны будут получать задания за его границей с защитным отступом.')
            return
        terrain=getattr(fleet.terrain,'core',fleet.terrain);reserved=getattr(terrain,'_conventional_mask',None)
        # Reservations cover the whole rectangle and are released as a whole.
        # Sample its centre instead of scanning a potentially million-cell map
        # on every UI refresh; in-place clearing is also seen immediately.
        row=int((y+h/2)/terrain.cell);col=int((x+w/2)/terrain.cell)
        active=(reserved is not None and 0<=row<reserved.shape[0] and
                0<=col<reserved.shape[1] and bool(reserved[row,col]))
        if active:
            text=f'Зарезервированный участок: {geometry}; для дронов — с защитным отступом. Общий грунт и время; дроны работают на других участках, сметы раздельные.'
        elif fleet.ground_stage_drained and fleet.sim.shared_site is not None:
            text=f'Участок передан дронам: {geometry}. Наземный этап завершён; рампа и защитная полоса доступны для завершения выемки. Парк и отдельная смета сохранены.'
        else:
            text=f'Рабочий участок обычной бригады: {geometry}. Общий грунт и время, сметы раздельные.'
        self.patch_status.setText(text)

    def fleet(self):return fleet_for(self.host.sim)

    def _notify_host(self):
        self.host.refresh(True);self.refresh()

    def create(self):
        try:
            self.host.install_conventional(self.config_value())
            self.message.setText('Техника размещена. Для движения и работы запустите модельное время; планирование подъездов выполняется по участкам.')
            self.tabs.setCurrentIndex(3);self._notify_host()
        except ValueError as error:self.message.setText(str(error))

    def toggle(self):
        fleet=self.fleet()
        if fleet is None:return
        enabled=fleet.report().get('enabled',fleet.config.enabled)
        try:
            fleet.set_enabled(not enabled)
            self.message.setText('Новые работы разрешены в рабочие смены.' if not enabled else 'Новые задания остановлены. Техника завершает безопасные операции и возвращается на базу.')
            self._notify_host()
        except ValueError as error:self.message.setText(str(error))

    def replenish(self):
        fleet=self.fleet()
        if fleet is None:return
        try:
            fleet.replenish(self.supply_fuel.value(),self.supply_water.value(),self.supply_food.value())
            self.message.setText('Расходники приняты на склад обычной бригады. Их стоимость включается в смету по фактическому использованию.')
            self._notify_host()
        except ValueError as error:self.message.setText(str(error))

    def refresh(self):
        fleet=self.fleet();active=fleet is not None
        self.create_button.setEnabled(not active);self.pause_button.setEnabled(active);self.replenish_button.setEnabled(active)
        for page in self.setup_pages:page.setEnabled(not active)
        if self._simulation is not self.host.sim:
            self._simulation=self.host.sim
            if not active:
                from .conventional import ConventionalConfig
                self._load_controls(ConventionalConfig.for_simulation(self.host.sim));self._world=None
        if not active:
            self.summary.setText('Обычная техника ещё не создана. Настройте состав, участок и смены, затем разместите бригаду.');self.table.setRowCount(0);return
        if self._world is not fleet:
            self._world=fleet;self._load_controls(fleet.config)
            self.message.setText('Состав приобретён. Остановка запрашивает безопасный возврат; расход топлива, питания и воды входит в отдельную смету.')
        self.refresh_patch_status(fleet.config)
        r=fleet.report();self.pause_button.setText('Приостановить и вернуться' if r.get('enabled',fleet.config.enabled) else 'Возобновить работу')
        self._render_report(r)

    def _load_controls(self,config):
        self.defaults=config;extent=self.host.sim.terrain.height.shape[0]*self.host.sim.terrain.cell
        self.controls['shift_span_hours'].setMaximum(8 if config.shifts_per_day==3 else 9)
        for key in ('base_x','base_y'):self.controls[key].setMaximum(extent)
        for widget in self.rect:widget.setMaximum(extent)
        for key,widget in self.controls.items():
            value=getattr(config,key);widget.blockSignals(True)
            if isinstance(widget,QtWidgets.QComboBox):widget.setCurrentIndex(widget.findData(value))
            else:widget.setValue(value)
            widget.blockSignals(False)
        for widget,value in zip(self.rect,config.work_rect):widget.setValue(value)
        self.auto_area.blockSignals(True);self.auto_area.setChecked(bool(getattr(config,'auto_work_area',False)));self.auto_area.blockSignals(False)
        for key,widgets in self.dimension_controls.items():
            for widget,value in zip(widgets,getattr(config,key)):
                widget.blockSignals(True);widget.setValue(value);widget.blockSignals(False)
        self.placement_mode_changed()
        self.update_preview()

    def _render_report(self,r):
        shift=r['shift'];crew=shift.get('crew')
        crew_text='нет активной бригады' if crew is None else f'бригада {crew}'
        cost=r['cost_per_m3_usd'];unit_cost='— до начала добычи' if cost is None else _number(cost)+' USD/м³'
        self.summary.setText((
            f"{r['reason']}\nСмена: {crew_text} · {shift['reason']}\n"
            f"Добыто {_number(r['excavated_m3'],3)} м³ · доставлено в отвал {_number(r['delivered_m3'],3)} м³ · в машинах {_number(r['carried_mass_kg'])} кг\n"
            f"Покупка техники {_number(r['capital_cost_usd'])} USD + эксплуатация {_number(r['operating_cost_usd'])} USD = {_number(r['total_cost_usd'])} USD · {unit_cost}\n"
            f"Зарплата {_number(r['payroll_usd'])} · топливо {_number(r['fuel_cost_usd'])} · питание {_number(r['food_cost_usd'])} · вода {_number(r['water_cost_usd'])} · обслуживание {_number(r['maintenance_usd'])} USD\n"
            f"Израсходовано: дизель {_number(r['fuel_used_l'])} л · вода {_number(r['water_used_l'])} л · питание {_number(r['food_used_units'])} порций\n"
            f"Остаток склада: дизель {_number(r['fuel_stock_l'])} л · вода {_number(r['water_stock_l'])} л · питание {_number(r['food_stock_units'])} порций\n"
            'Стоимость добычи включает покупку техники; автономный комплекс в эту смету не входит.').strip())
        rows=r.get('machines',[]);self.table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            kind={'excavator':'Экскаватор','truck':'Самосвал'}.get(row['kind'],row['kind'])
            capacity=self.fleet().config.excavator_tank_l if row['kind']=='excavator' else self.fleet().config.truck_tank_l
            reason=row['reason']
            if row['state']=='REFUELING' and not reason:reason=f"Заправка из склада: до полного бака ещё {max(0.,capacity-row['fuel_l']):.1f} л"
            dimensions=row.get('dimensions_m',(6.,3.,3.) if row['kind']=='excavator' else (7.,2.8,3.2))
            size=' × '.join(f'{v:g}' for v in dimensions)
            position=row['position_m'];values=[row['id'],kind,STATES.get(row['state'],row['state']),f"{row['fuel_l']:.1f} / {capacity:g}",_number(row['cargo_kg']),f'{position[0]:.1f} / {position[1]:.1f}',size,reason]
            for j,value in enumerate(values):
                item=self.table.item(i,j)
                if item is None:item=QtWidgets.QTableWidgetItem();self.table.setItem(i,j,item)
                item.setText(str(value));item.setToolTip(reason)

    def showEvent(self,event):
        self.timer.start();self.refresh();super().showEvent(event)

    def closeEvent(self,event):
        self.timer.stop();super().closeEvent(event)
