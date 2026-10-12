"""Qt presentation. Fixed engine steps are bounded by a per-frame time budget."""
from copy import deepcopy
import time
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from .config import KWH
from .scenarios import PRESETS
from .simulation import Simulation, MissionState
from .drone import DroneState
from .fleet_ui import FleetDialog,ComparisonWindow

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="#111d2b", foreground="#c7d7e5")


def terrain_rgb(height, cell, depth, maximum):
    stops = [-max(depth, 0.1), 0.0, max(maximum, 0.1)]
    colors = np.array([[48, 121, 161], [202, 181, 122], [240, 225, 186]])
    rgb = np.stack([np.interp(height, stops, colors[:, i]) for i in range(3)], axis=-1)
    gy, gx = np.gradient(height, cell)
    shade = (0.9+0.25*(-gx+gy)/np.sqrt(1+gx*gx+gy*gy)).clip(0.55, 1.1)
    return (rgb*shade[..., None]).clip(0, 255).astype(np.uint8)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = deepcopy(config)
        self.sim = Simulation(self.config)
        self.running = False
        self.credit = 0.0
        self.last_wall = time.perf_counter()
        self.last_terrain_wall = -1.0
        self.last_revision = -1
        self.last_event = 0
        self.setWindowTitle("ATS · Автономные земляные работы")
        self.resize(1450, 940)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #101a27; color: #dbe6ef; font-size: 13px; }
            QGroupBox { border: 1px solid #314457; border-radius: 6px; margin-top: 10px; padding-top: 12px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #283d50; border: 1px solid #435c72; border-radius: 5px; padding: 8px 12px; }
            QPushButton:hover { background: #38536b; }
            QPushButton#play { background: #207d6e; font-weight: bold; }
            QComboBox, QDoubleSpinBox, QSpinBox { background: #203142; border: 1px solid #41566b; border-radius: 3px; padding: 4px; }
            QPlainTextEdit { background: #0c1520; border: 1px solid #314457; font-family: monospace; }
            QProgressBar { border: 1px solid #41566b; text-align: center; border-radius: 3px; }
            QProgressBar::chunk { background: #207d6e; }
        """)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("<b style='font-size:22px'>ATS</b>  /  Autonomous Terraforming Simulator")
        header.addWidget(title, 1)
        self.play = QtWidgets.QPushButton("▶ Start")
        self.play.setObjectName("play")
        self.play.clicked.connect(self.toggle_play)
        header.addWidget(self.play)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        header.addWidget(self.reset_button)
        fleet_button=QtWidgets.QPushButton("Дроны / зарядка")
        fleet_button.clicked.connect(self.edit_fleet);header.addWidget(fleet_button)
        compare_button=QtWidgets.QPushButton("Сравнить 1…N")
        compare_button.clicked.connect(self.compare_fleet);header.addWidget(compare_button)
        self.speed = QtWidgets.QComboBox()
        for factor in [1, 10, 50, 200]:
            self.speed.addItem(f"{factor}×", factor)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(lambda: setattr(self, "credit", 0.0))
        header.addWidget(self.speed)
        layout.addLayout(header)
        split = QtWidgets.QSplitter()
        layout.addWidget(split, 1)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        legend = QtWidgets.QLabel("Рельеф: синий — выемка, песочный — исходный уровень, светлый — насыпь  |  голубой — проект  |  оранжевый — отвал")
        legend.setWordWrap(True)
        left_layout.addWidget(legend)
        self.map = pg.PlotWidget()
        self.map.setAspectLocked(True)
        self.map.setLabel("bottom", "X", units="m")
        self.map.setLabel("left", "Y", units="m")
        self.map.showGrid(x=True, y=True, alpha=0.12)
        left_layout.addWidget(self.map, 5)
        self.image = pg.ImageItem()
        self.map.addItem(self.image)
        self.design_overlay = pg.ImageItem()
        self.map.addItem(self.design_overlay)
        self.dust_image = pg.ImageItem()
        self.dust_image.setZValue(5)
        self.map.addItem(self.dust_image)
        self.base_marker = pg.ScatterPlotItem(size=22, symbol="s", brush="#f4b651", pen=pg.mkPen("#fff0c5", width=2))
        self.drone_marker = pg.ScatterPlotItem(size=20, symbol="+", pen=pg.mkPen("#82f5db", width=3))
        self.base_marker.setZValue(10)
        self.drone_marker.setZValue(12)
        self.map.addItem(self.base_marker)
        self.map.addItem(self.drone_marker)
        self.route = pg.PlotDataItem(pen=pg.mkPen("#75d8d1", width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.map.addItem(self.route)
        self.drone_label = pg.TextItem(color="#a4ffe8", anchor=(0, 1))
        self.base_label = pg.TextItem("База", color="#ffdb8a", anchor=(0, 0))
        self.map.addItem(self.drone_label)
        self.map.addItem(self.base_label)
        self.profile = pg.PlotWidget(title="Поперечный профиль через центр канавы")
        self.profile.setLabel("bottom", "Y", units="m")
        self.profile.setLabel("left", "Высота", units="m")
        self.profile.getAxis("left").enableAutoSIPrefix(False)
        self.profile.addLegend()
        self.current_curve = self.profile.plot(pen=pg.mkPen("#efce8b", width=2), name="Текущий рельеф")
        self.target_curve = self.profile.plot(pen=pg.mkPen("#59cde4", width=2, style=QtCore.Qt.PenStyle.DashLine), name="Проект")
        self.reference_curve = self.profile.plot(pen=pg.mkPen("#627789", width=1), name="Исходный")
        self.profile.showGrid(x=True, y=True, alpha=0.2)
        left_layout.addWidget(self.profile, 2)
        self.fleet_table=QtWidgets.QTableWidget(0,10)
        self.fleet_table.setHorizontalHeaderLabels(["Дрон","Состояние","Заряд","Груз, кг","Рейсы","Грунт, кг","Расход, кВт·ч","Очередь, мин","Зарядка, мин","Уступил"])
        self.fleet_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.fleet_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.fleet_table.setMaximumHeight(170);self.fleet_table.setMinimumHeight(85)
        left_layout.addWidget(self.fleet_table)
        self.palette=["#82f5db","#ffd166","#68b6ff","#ff89c1","#b0f083","#c4a6ff"]
        self.drone_labels=[self.drone_label]
        self.slot_markers=pg.ScatterPlotItem(size=10,symbol="s",pen="#f4b651")
        self.map.addItem(self.slot_markers)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        right.setMinimumWidth(335)
        right.setMaximumWidth(440)
        right_layout = QtWidgets.QVBoxLayout(right)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        right_layout.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        right_layout.addWidget(self.progress)
        self.telemetry = QtWidgets.QLabel()
        self.telemetry.setTextFormat(QtCore.Qt.TextFormat.RichText)
        right_layout.addWidget(self.telemetry)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        control_widget = QtWidgets.QWidget()
        controls = QtWidgets.QVBoxLayout(control_widget)
        controls.setContentsMargins(2, 2, 2, 2)
        project = QtWidgets.QGroupBox("Новый сценарий · размеры в метрах")
        form = QtWidgets.QFormLayout(project)
        self.preset = QtWidgets.QComboBox()
        for name, key in [("Демо", "demo"), ("Средний", "medium"), ("Большой", "large")]:
            self.preset.addItem(name, key)
        form.addRow("Пресет", self.preset)
        self.length = self.spin(0.25, 25, config.trench.length, 0.25)
        self.width = self.spin(0.25, 15, config.trench.width, 0.25)
        self.depth = self.spin(0.05, 2, config.trench.depth, 0.05)
        self.cx = self.spin(0.5, 31.5, config.trench.x, 0.25)
        self.cy = self.spin(0.5, 31.5, config.trench.y, 0.25)
        self.angle = self.spin(5, 55, config.trench.angle_deg, 1)
        self.start_hour = self.spin(0, 23.99, config.start_hour, 0.5)
        self.roughness = self.spin(0, 0.03, config.terrain.roughness, 0.005, 3)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(config.seed)
        for label, widget in [("Длина дна", self.length), ("Ширина дна", self.width), ("Глубина", self.depth), ("Центр X", self.cx), ("Центр Y", self.cy), ("Откос, °", self.angle), ("Стартовый час", self.start_hour), ("Неровности, ±м", self.roughness), ("Seed", self.seed)]:
            form.addRow(label, widget)
        solar_group = QtWidgets.QGroupBox("База и солнце · для нового сценария")
        solar_form = QtWidgets.QFormLayout(solar_group)
        self.panels = self.spin(0, 200, config.base.panel_area, 1)
        self.base_mass = self.spin(100, 20000, config.base.mass, 100, 0)
        self.latitude = self.spin(-90, 90, config.base.latitude_deg, 1)
        self.season = QtWidgets.QComboBox()
        for label, day in [("21 марта · равноденствие", 80), ("21 июня · солнцестояние", 172), ("21 сентября · равноденствие", 264), ("21 декабря · солнцестояние", 355)]:
            self.season.addItem(label, day)
        self.season.setPlaceholderText("Выбрать сезонную дату")
        self.season.setCurrentIndex(1)
        self.calendar = QtWidgets.QDateEdit(QtCore.QDate(2025, 1, 1).addDays(config.base.start_day_of_year-1))
        self.calendar.setDisplayFormat("dd.MM")
        self.calendar.setCalendarPopup(True)
        self.calendar.setMinimumDate(QtCore.QDate(2025, 1, 1))
        self.calendar.setMaximumDate(QtCore.QDate(2025, 12, 31))
        for label, widget in [("Площадь панелей, м²", self.panels), ("Масса базы, кг", self.base_mass), ("Широта, ° (+север)", self.latitude), ("Сезонная дата", self.season), ("Стартовая дата", self.calendar)]:
            solar_form.addRow(label, widget)
        self.panel_info = QtWidgets.QLabel()
        self.panel_info.setWordWrap(True)
        solar_form.addRow(self.panel_info)
        self.season.currentIndexChanged.connect(lambda index: self.calendar.setDate(QtCore.QDate(2025, 1, 1).addDays(self.season.currentData()-1)) if index >= 0 else None)
        for widget in [self.panels, self.latitude]:
            widget.valueChanged.connect(self.update_solar_preview)
        self.calendar.dateChanged.connect(self.update_solar_preview)
        self.update_solar_preview()
        self.preset.currentIndexChanged.connect(self.apply_preset)
        new = QtWidgets.QPushButton("Создать новый сценарий")
        new.clicked.connect(self.new_scenario)
        form.addRow(new)
        controls.addWidget(project)
        solar_apply = QtWidgets.QPushButton("Создать сценарий с этими настройками")
        solar_apply.clicked.connect(self.new_scenario)
        solar_form.addRow(solar_apply)
        controls.addWidget(solar_group)
        dust_group = QtWidgets.QGroupBox("Пыль и ветер · применяется сразу")
        dust_form = QtWidgets.QFormLayout(dust_group)
        self.show_dust = QtWidgets.QCheckBox("Показывать пылевое поле")
        self.show_dust.setChecked(True)
        dust_form.addRow(self.show_dust)
        self.wind_x = self.spin(-8, 8, config.dust.wind_x, 0.2)
        self.wind_y = self.spin(-8, 8, config.dust.wind_y, 0.2)
        self.dust_strength = self.spin(0, 10, config.dust.strength, 0.1)
        for label, widget in [("Ветер X, м/с", self.wind_x), ("Ветер Y, м/с", self.wind_y), ("Пылеобразование", self.dust_strength)]:
            dust_form.addRow(label, widget)
            widget.valueChanged.connect(self.change_dust)
        controls.addWidget(dust_group)
        controls.addStretch()
        scroll.setWidget(control_widget)
        right_layout.addWidget(scroll, 2)
        right_layout.addWidget(QtWidgets.QLabel("Журнал событий"))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(config.event_limit)
        right_layout.addWidget(self.log, 1)
        split.addWidget(right)
        split.setSizes([1030, 370])
        self.statusBar().showMessage("Колесо мыши — масштаб; перетаскивание — панорама. Размеры сценария применяются кнопкой создания.")
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_frame)
        self.timer.start(config.display_interval_ms)
        self.setup_map()
        self.refresh(True)

    @staticmethod
    def spin(minimum, maximum, value, step, decimals=2):
        result = QtWidgets.QDoubleSpinBox()
        result.setRange(minimum, maximum)
        result.setDecimals(decimals)
        result.setSingleStep(step)
        result.setValue(value)
        return result

    def setup_map(self):
        t = self.sim.terrain
        side = t.cell*t.height.shape[0]
        rect = QtCore.QRectF(0, 0, side, side)
        self.image.setRect(rect)
        overlay = np.zeros((*t.height.shape, 4), dtype=np.uint8)
        overlay[t.excavation] = [57, 200, 237, 24]
        overlay[t.bottom] = [57, 200, 237, 50]
        # Outlines remain visible without obscuring the actual terrain.
        for mask, color in [(t.excavation, [65, 210, 250, 190]), (t.bottom, [90, 225, 255, 200]), (t.dump_mask, [255, 164, 60, 190])]:
            padded = np.pad(mask, 1)
            border = mask & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
            overlay[border] = color
        self.design_overlay.setImage(overlay, autoLevels=False)
        self.design_overlay.setRect(rect)
        self.dust_image.setRect(rect)
        self.map.setRange(xRange=(0, side), yRange=(0, side), padding=0.015)
        q = self.config.trench
        self.profile.setXRange(q.y-q.width/2-q.depth/t.slope-1, q.y+q.width/2+q.depth/t.slope+1)
        self.profile.setYRange(-q.depth-0.08, 0.15)
        self.last_revision = -1
        while len(self.drone_labels)<len(self.sim.drones):
            label=pg.TextItem(color=self.palette[len(self.drone_labels)],anchor=(0,1));self.map.addItem(label);self.drone_labels.append(label)
        for index,label in enumerate(self.drone_labels):label.setVisible(index<len(self.sim.drones))
        self.fleet_table.setRowCount(len(self.sim.drones))

    def edit_fleet(self):
        dialog=FleetDialog(self.config,self)
        if dialog.exec()==QtWidgets.QDialog.DialogCode.Accepted:
            try:self.replace_simulation(dialog.result_config)
            except ValueError as error:QtWidgets.QMessageBox.warning(self,"Неверный флот",str(error))

    def compare_fleet(self):
        self.running=False;self.play.setText("▶ Start")
        self.comparison=ComparisonWindow(self.config,self);self.comparison.show()

    def apply_preset(self):
        length, width, depth = PRESETS[self.preset.currentData()]
        self.length.setValue(length)
        self.width.setValue(width)
        self.depth.setValue(depth)

    def change_dust(self):
        c = self.sim.config.dust
        c.wind_x, c.wind_y, c.strength = self.wind_x.value(), self.wind_y.value(), self.dust_strength.value()

    def update_solar_preview(self):
        from .solar import sunlight
        sun = sunlight(self.latitude.value(), self.calendar.date().dayOfYear(), 12, self.config.base.peak_irradiance)
        self.season.blockSignals(True)
        self.season.setCurrentIndex(self.season.findData(self.calendar.date().dayOfYear()))
        self.season.blockSignals(False)
        power = sun.irradiance*self.panels.value()*self.config.base.panel_efficiency/1000
        self.panel_info.setText(f"Ясное небо, горизонтальные панели. Полдень: {sun.irradiance:.0f} Вт/м² → {power:.2f} кВт; световой день {sun.daylight_hours:.1f} ч.\nВремя — местное солнечное; календарь 365 дней.")

    def replace_simulation(self, config):
        sim = Simulation(config)  # validate before discarding the current state
        self.running = False
        self.play.setText("▶ Start")
        self.credit = 0
        self.config = config
        self.sim = sim
        self.last_event = 0
        self.log.clear()
        self.setup_map()
        self.change_dust()
        self.refresh(True)

    def new_scenario(self):
        config = deepcopy(self.config)
        q = config.trench
        q.length, q.width, q.depth = self.length.value(), self.width.value(), self.depth.value()
        q.x, q.y, q.angle_deg = self.cx.value(), self.cy.value(), self.angle.value()
        config.seed = self.seed.value()
        config.start_hour = self.start_hour.value()
        config.terrain.roughness = self.roughness.value()
        config.base.panel_area = self.panels.value()
        config.base.mass = self.base_mass.value()
        config.base.latitude_deg = self.latitude.value()
        config.base.start_day_of_year = self.calendar.date().dayOfYear()
        try:
            self.replace_simulation(config)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Неверный сценарий", str(error))

    def reset(self):
        self.replace_simulation(deepcopy(self.config))

    def toggle_play(self):
        if self.sim.state != MissionState.RUNNING:
            return
        self.running = not self.running
        self.credit = 0
        self.last_wall = time.perf_counter()
        self.play.setText("⏸ Pause" if self.running else "▶ Start")

    def on_frame(self):
        now = time.perf_counter()
        elapsed = min(now-self.last_wall, 0.25)
        self.last_wall = now
        if self.running:
            self.credit = min(self.credit+elapsed*self.speed.currentData(), self.config.max_frame_steps)
            deadline = now+self.config.frame_budget_ms/1000
            steps = 0
            while self.credit >= 1 and steps < self.config.max_frame_steps and time.perf_counter() < deadline:
                self.sim.step()
                self.credit -= 1
                steps += 1
                if self.sim.state != MissionState.RUNNING:
                    self.running = False
                    self.play.setText("Завершено" if self.sim.state == MissionState.COMPLETED else "Заблокировано")
                    break
        self.refresh()

    def refresh(self, force=False):
        s, t, d, b = self.sim, self.sim.terrain, self.sim.drone, self.sim.base
        now = time.perf_counter()
        if force or (t.revision != self.last_revision and now-self.last_terrain_wall >= self.config.terrain_display_seconds):
            self.image.setImage(terrain_rgb(t.height, t.cell, self.config.trench.depth, max(0.5, float(t.height.max()))), autoLevels=False)
            self.image.setRect(QtCore.QRectF(0, 0, t.cell*t.height.shape[0], t.cell*t.height.shape[0]))
            index = int(np.clip(self.config.trench.x/t.cell, 0, t.height.shape[1]-1))
            self.current_curve.setData(t.axis, t.height[:, index])
            self.target_curve.setData(t.axis, t.target[:, index])
            self.reference_curve.setData(t.axis, t.reference[:, index])
            self.last_revision = t.revision
            self.last_terrain_wall = now
        concentration = 1-np.exp(-s.dust.field*self.config.dust.concentration_scale)
        cloud = np.zeros((*concentration.shape, 4), dtype=np.uint8)
        cloud[..., :3] = [211, 187, 146]
        cloud[..., 3] = (concentration*170).clip(0, 170).astype(np.uint8)
        self.dust_image.setImage(cloud, autoLevels=False)
        self.dust_image.setRect(QtCore.QRectF(0, 0, s.dust.side, s.dust.side))
        self.dust_image.setVisible(self.show_dust.isChecked())
        self.base_marker.setData([b.position[0]], [b.position[1]])
        self.drone_marker.setData([{'pos':tuple(o.position[:2]),'pen':pg.mkPen(self.palette[o.identifier],width=3),'symbol':'+','size':20} for o in s.drones])
        self.slot_markers.setData([{'pos':tuple(s.slot_position(i)[:2]),'brush':'#f4b651' if occupant is not None else '#41566b','symbol':'s','size':10} for i,occupant in enumerate(s.slots)])
        for o,label in zip(s.drones,self.drone_labels):
            operation=" · грейфер вниз" if o.state==DroneState.DUMP and not o.dump_released else " · грейфер вверх" if o.state==DroneState.DUMP else ""
            label.setText(f"Д{o.identifier+1} · {o.cargo:.0f} кг · z {o.position[2]:.1f} м{operation}")
            label.setPos(o.position[0]+.4,o.position[1]+.4+o.identifier*.3)
        self.base_label.setPos(b.position[0]+0.6, b.position[1]-0.4)
        if any(o.trip for o in s.drones):
            points = np.array([p for o in s.drones if o.trip for p in [b.position[:2],o.trip.dig,o.trip.dump,b.position[:2],(np.nan,np.nan)]])
            self.route.setData(points[:, 0], points[:, 1])
        else:
            self.route.setData([], [])
        report = s.report()
        self.fleet_table.setUpdatesEnabled(False)
        for row,o in enumerate(report['drones']):
            values=[f"{o['id']} · {o['name']}",o['state'],f"{o['battery_percent']:.0f}%",f"{o['cargo_kg']:.1f}",str(o['trips']),f"{o['excavated_mass_kg']:.0f}",f"{o['consumed_kwh']:.3f}",f"{o['waiting_seconds']/60:.1f}",f"{o['charging_seconds']/60:.1f}",str(o['traffic_yields'])]
            for col,value in enumerate(values):self.fleet_table.setItem(row,col,QtWidgets.QTableWidgetItem(value))
        self.fleet_table.setUpdatesEnabled(True)
        status = {MissionState.RUNNING: "Автономная работа" if self.running else "Готов / пауза", MissionState.COMPLETED: "✓ Миссия завершена", MissionState.BLOCKED: "⚠ Работа заблокирована"}[s.state]
        self.status.setText(f"<b style='font-size:17px'>{status}</b><br>{s.reason}")
        self.progress.setValue(round(report["completion_percent"]*10))
        self.progress.setFormat(f"{report['completion_percent']:.1f}% · допуск {self.config.terrain.completion_volume:.3f} м³")
        rows = [
            ("Прошло с запуска", report["elapsed_display"]),
            ("Дата / солнечное время", f"{s.date_text} · {s.clock_text}"),
            ("Флот / очередь",f"{len(s.drones)} / {len(s.queue)}"), ("База", b.state.value),
            ("Зарядные слоты",f"{sum(v is not None for v in s.slots)} / {len(s.slots)} · {b.config.charge_power/1000:.1f} кВт всего"),
            ("Проект / остаток", f"{report['initial_excavation_m3']:.3f} / {report['remaining_m3']:.3f} м³"),
            ("Вынуто / рейсы", f"{report['excavated_m3']:.3f} м³ / {s.trips}"),
            ("Общий груз", f"{report['cargo_kg']:.1f} кг"),
            ("База / панели", f"{b.config.mass:.0f} кг / {b.config.panel_area:.1f} м²"),
            ("Широта / световой день", f"{b.config.latitude_deg:+.1f}° / {report['daylight_hours']:.1f} ч"),
            ("Солнце: высота / поток", f"{report['sun_elevation_deg']:.1f}° / {report['irradiance_w_m2']:.0f} Вт/м²"),
            ("Батареи флота", f"{report['drone_kwh']:.2f} / {sum(o.config.battery for o in s.drones)/KWH:.2f} кВт·ч"),
            ("Батарея базы", f"{b.battery/KWH:.2f} кВт·ч · {100*b.battery/b.config.battery:.0f}%"),
            ("Солнечная мощность", f"{b.solar_power/1000:.2f} кВт"),
            ("Солнце / расход", f"{report['solar_generated_kwh']:.3f} / {report['consumed_kwh']:.3f} кВт·ч"),
            ("Потери зарядки", f"{report['charging_losses_kwh']:.3f} кВт·ч"),
            ("Баланс массы", f"{report['mass_error_kg']:+.2e} кг"),
            ("Баланс энергии", f"{report['energy_error_j']:+.2e} Дж"),
        ]
        self.telemetry.setText("<table cellspacing='4'>"+"".join(f"<tr><td style='color:#94abbf'>{label}</td><td>{value}</td></tr>" for label, value in rows)+"</table>")
        for serial, timestamp, message in s.events:
            if serial > self.last_event:
                self.log.appendPlainText(f"[{timestamp/60:6.1f} мин] {message}")
                self.last_event = serial
        if s.state != MissionState.RUNNING:
            self.play.setEnabled(False)
        else:
            self.play.setEnabled(True)


def launch(config):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("ATS")
    window = MainWindow(config)
    window.show()
    return app.exec()
