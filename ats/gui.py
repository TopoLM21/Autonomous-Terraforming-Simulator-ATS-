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
        self.panels = self.spin(0, 50, config.base.panel_area, 1)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(config.seed)
        for label, widget in [("Длина дна", self.length), ("Ширина дна", self.width), ("Глубина", self.depth), ("Центр X", self.cx), ("Центр Y", self.cy), ("Откос, °", self.angle), ("Стартовый час", self.start_hour), ("Неровности, ±м", self.roughness), ("Панели, м²", self.panels), ("Seed", self.seed)]:
            form.addRow(label, widget)
        self.panel_info = QtWidgets.QLabel("При 1000 Вт/м²: 4,0 кВт")
        form.addRow(self.panel_info)
        self.panels.valueChanged.connect(lambda value: self.panel_info.setText(f"Пик генерации: {value*self.config.base.panel_efficiency*self.config.base.peak_irradiance/1000:.1f} кВт"))
        self.preset.currentIndexChanged.connect(self.apply_preset)
        new = QtWidgets.QPushButton("Создать новый сценарий")
        new.clicked.connect(self.new_scenario)
        form.addRow(new)
        controls.addWidget(project)
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

    def apply_preset(self):
        length, width, depth = PRESETS[self.preset.currentData()]
        self.length.setValue(length)
        self.width.setValue(width)
        self.depth.setValue(depth)

    def change_dust(self):
        c = self.sim.config.dust
        c.wind_x, c.wind_y, c.strength = self.wind_x.value(), self.wind_y.value(), self.dust_strength.value()

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
        self.drone_marker.setData([d.position[0]], [d.position[1]])
        self.drone_label.setText(f"Дрон · {d.cargo:.0f} кг · z {d.position[2]:.1f} м")
        self.drone_label.setPos(d.position[0]+0.4, d.position[1]+0.4)
        self.base_label.setPos(b.position[0]+0.6, b.position[1]-0.4)
        if s.trip is not None:
            points = np.array([b.position[:2], s.trip.dig, s.trip.dump, b.position[:2]])
            self.route.setData(points[:, 0], points[:, 1])
        else:
            self.route.setData([], [])
        report = s.report()
        days, remainder = divmod(int(s.time), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        status = {MissionState.RUNNING: "Автономная работа" if self.running else "Готов / пауза", MissionState.COMPLETED: "✓ Миссия завершена", MissionState.BLOCKED: "⚠ Работа заблокирована"}[s.state]
        self.status.setText(f"<b style='font-size:17px'>{status}</b><br>{s.reason}")
        self.progress.setValue(round(report["completion_percent"]*10))
        self.progress.setFormat(f"{report['completion_percent']:.1f}% · допуск {self.config.terrain.completion_volume:.3f} м³")
        rows = [
            ("Время", f"{days}д {hours:02}:{minutes:02}:{seconds:02} · сутки {s.hour:05.2f}ч"),
            ("Дрон", d.state.value), ("База", b.state.value),
            ("Проект / остаток", f"{report['initial_excavation_m3']:.3f} / {report['remaining_m3']:.3f} м³"),
            ("Вынуто / рейсы", f"{report['excavated_m3']:.3f} м³ / {s.trips}"),
            ("Груз / грейфер", f"{d.cargo:.1f} кг / трос {d.grabber_extension*max(0, d.position[2]-t.sample(*d.position[:2])):.1f} м"),
            ("Батарея дрона", f"{d.battery/KWH:.2f} кВт·ч · {100*d.battery/d.config.battery:.0f}%"),
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
