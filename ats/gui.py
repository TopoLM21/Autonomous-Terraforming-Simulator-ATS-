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
from .equipment_ui import DailyStatsWindow
from .plan_ui import BrushViewBox,PlanEditor
from .plan import load_plan,save_plan,target_surface
from .profiles import profiles,segments_for

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
        self._equipment_resume = None
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
        fleet_button=QtWidgets.QPushButton("Оборудование")
        fleet_button.clicked.connect(self.edit_fleet);header.addWidget(fleet_button)
        daily_button=QtWidgets.QPushButton("По суткам")
        daily_button.clicked.connect(self.show_daily_stats);header.addWidget(daily_button)
        self.cancel_equipment_button=QtWidgets.QPushButton("Отменить установку")
        self.cancel_equipment_button.clicked.connect(self.cancel_equipment);header.addWidget(self.cancel_equipment_button)
        self.cancel_equipment_button.hide()
        compare_button=QtWidgets.QPushButton("Сравнить 1…N")
        compare_button.clicked.connect(self.compare_fleet);header.addWidget(compare_button)
        logistics_button=QtWidgets.QPushButton('Склад / диспетчер')
        logistics_button.clicked.connect(self.show_logistics);header.addWidget(logistics_button)
        experiments_button=QtWidgets.QPushButton('ATS 2.0')
        experiments_button.clicked.connect(self.show_experiments);header.addWidget(experiments_button)
        self.brush_button=QtWidgets.QPushButton("Кисть / план")
        self.brush_button.setCheckable(True);header.addWidget(self.brush_button)
        self.speed = QtWidgets.QComboBox()
        for factor in [1, 10, 50, 200, 1000, 2000]:
            self.speed.addItem(f"{factor}×", factor)
        self.speed.setCurrentIndex(2)
        self.speed.setToolTip("Целевая скорость виртуального времени. Физический шаг остаётся 1 с; доступное ускорение зависит от производительности компьютера.")
        self.speed.currentIndexChanged.connect(lambda: setattr(self, "credit", 0.0))
        header.addWidget(self.speed)
        layout.addLayout(header)
        split = QtWidgets.QSplitter()
        layout.addWidget(split, 1)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        legend = QtWidgets.QLabel("Рельеф: синий — выемка, песочный — исходный уровень, светлый — насыпь | План: голубой — выемка, зелёный — засыпка, оранжевый — отвал; розовый — профиль")
        legend.setWordWrap(True)
        left_layout.addWidget(legend)
        self.plan_editor=PlanEditor(config,self)
        left_layout.addWidget(self.plan_editor);self.plan_editor.hide()
        self.view_box=BrushViewBox()
        self.map = pg.PlotWidget(viewBox=self.view_box)
        self.map.setAspectLocked(True)
        self.map.setLabel("bottom", "X", units="m")
        self.map.setLabel("left", "Y", units="m")
        self.map.showGrid(x=True, y=True, alpha=0.12)
        left_layout.addWidget(self.map, 5)
        self.image = pg.ImageItem()
        self.map.addItem(self.image)
        self.design_overlay = pg.ImageItem()
        self.map.addItem(self.design_overlay)
        self.brush_overlay=pg.ImageItem();self.brush_overlay.setZValue(4);self.map.addItem(self.brush_overlay)
        self.brush_cursor=QtWidgets.QGraphicsEllipseItem();self.brush_cursor.setPen(pg.mkPen("#94ffe2",width=1));self.brush_cursor.setZValue(15);self.brush_cursor.hide();self.map.addItem(self.brush_cursor)
        self.brush_cursor.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.profile_line=pg.PlotDataItem(pen=pg.mkPen("#e7a3f0",width=2,style=QtCore.Qt.PenStyle.DotLine));self.map.addItem(self.profile_line)
        self.dust_image = pg.ImageItem()
        self.dust_image.setZValue(5)
        self.map.addItem(self.dust_image)
        self.base_marker = pg.ScatterPlotItem(size=22, symbol="s", brush="#f4b651", pen=pg.mkPen("#fff0c5", width=2))
        self.drone_marker = pg.ScatterPlotItem(size=20, symbol="+", pen=pg.mkPen("#82f5db", width=3))
        self.base_marker.setZValue(10)
        self.drone_marker.setZValue(12)
        self.map.addItem(self.base_marker)
        self.map.addItem(self.drone_marker)
        self.excavator_marker=pg.ScatterPlotItem(size=20,symbol='t',brush='#ffc16e');self.map.addItem(self.excavator_marker)
        self.container_markers=pg.ScatterPlotItem(size=9,symbol='s');self.map.addItem(self.container_markers)
        self.excavator_reach=pg.PlotDataItem(pen=pg.mkPen('#ffca82',width=1,style=QtCore.Qt.PenStyle.DotLine));self.map.addItem(self.excavator_reach)
        self.solar_field_outline=pg.PlotDataItem(pen=pg.mkPen('#83b9f8',width=2));self.map.addItem(self.solar_field_outline)
        self.excavator_label=pg.TextItem(color='#ffe0a1',anchor=(0,1));self.map.addItem(self.excavator_label)
        self.route = pg.PlotDataItem(pen=pg.mkPen("#75d8d1", width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.map.addItem(self.route)
        self.laser_beams=pg.PlotDataItem(pen=pg.mkPen("#ff7850",width=2));self.laser_beams.setZValue(9);self.map.addItem(self.laser_beams)
        self.drone_label = pg.TextItem(color="#a4ffe8", anchor=(0, 1))
        self.base_label = pg.TextItem("База", color="#ffdb8a", anchor=(0, 0))
        self.map.addItem(self.drone_label)
        self.map.addItem(self.base_label)
        profile_controls=QtWidgets.QHBoxLayout()
        self.profile_segment=QtWidgets.QComboBox()
        self.profile_direction=QtWidgets.QComboBox();self.profile_direction.addItem("Поперечный","cross");self.profile_direction.addItem("Сквозной вдоль линий","along")
        self.profile_position=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal);self.profile_position.setRange(0,100);self.profile_position.setValue(50);self.profile_position.setMaximumWidth(180)
        self.profile_position_label=QtWidgets.QLabel("Срез: 50%")
        for widget in [QtWidgets.QLabel("Профиль"),self.profile_segment,self.profile_direction,self.profile_position_label,self.profile_position]:profile_controls.addWidget(widget)
        profile_controls.addStretch();left_layout.addLayout(profile_controls)
        self.profile = pg.PlotWidget(title="Поперечный профиль")
        self.profile.setLabel("bottom", "Y", units="m")
        self.profile.setLabel("left", "Высота", units="m")
        self.profile.getAxis("left").enableAutoSIPrefix(False)
        self.profile.addLegend(offset=(5,5),colCount=3)
        self.current_curve = self.profile.plot(pen=pg.mkPen("#efce8b", width=2), name="Текущий рельеф")
        self.target_curve = self.profile.plot(pen=pg.mkPen("#59cde4", width=2, style=QtCore.Qt.PenStyle.DashLine), name="Проект")
        self.reference_curve = self.profile.plot(pen=pg.mkPen("#627789", width=1), name="Исходный")
        self.profile.showGrid(x=True, y=True, alpha=0.2)
        self.profile_extra=[];self.profile_boundaries=[]
        left_layout.addWidget(self.profile, 2)
        self.fleet_table=QtWidgets.QTableWidget(0,12)
        self.fleet_table.setHorizontalHeaderLabels(["Дрон","Состояние","Заряд","Груз, кг","Рейсы","Грунт, кг\nвыемка / засыпка","Расход, кВт·ч","Очередь, мин","Зарядка, мин","Уступил","24 ч, м³","Лазер, кВт"])
        self.fleet_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.fleet_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.fleet_table.setMaximumHeight(170);self.fleet_table.setMinimumHeight(85)
        left_layout.addWidget(self.fleet_table)
        self.palette=["#82f5db","#ffd166","#68b6ff","#ff89c1","#b0f083","#c4a6ff"]
        self.drone_labels=[self.drone_label]
        self.slot_markers=pg.ScatterPlotItem(size=10,symbol="s",pen="#f4b651")
        self.map.addItem(self.slot_markers)
        self.brush_button.toggled.connect(self.toggle_brush_panel)
        self.plan_editor.enabled.toggled.connect(self.toggle_brush)
        self.view_box.strokeFinished.connect(self.add_brush_stroke)
        self.view_box.brushMoved.connect(self.move_brush_cursor)
        self.map.scene().sigMouseMoved.connect(lambda p:self.move_brush_cursor(self.view_box.point(p)))
        self.plan_editor.planChanged.connect(self.preview_brush)
        self.plan_editor.applyRequested.connect(self.apply_brush_plan)
        self.plan_editor.landingRequested.connect(self.land_for_plan)
        self.plan_editor.saveRequested.connect(self.save_brush_plan)
        self.plan_editor.loadRequested.connect(self.load_brush_plan)
        self.profile_segment.currentIndexChanged.connect(self.update_profile)
        self.profile_direction.currentIndexChanged.connect(self.update_profile)
        self.profile_position.valueChanged.connect(self.update_profile)
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
        self.panels = self.spin(0, 10000000, self.config.base.panel_area, 1)
        self.base_mass = self.spin(0, 10000000, self.config.base.mass, 100, 1)
        for widget in [self.panels,self.base_mass]:
            widget.setReadOnly(True);widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            widget.setToolTip("Рассчитывается по установленным модулям. Измените состав кнопкой «Оборудование».")
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
        overlay[t.fill_mask] = [112, 242, 132, 65]
        # Outlines remain visible without obscuring the actual terrain.
        for mask, color in [(t.excavation, [65, 210, 250, 190]), (t.bottom & ~t.fill_mask, [90, 225, 255, 200]), (t.fill_mask,[112,242,132,220]), (t.dump_mask, [255, 164, 60, 190])]:
            padded = np.pad(mask, 1)
            border = mask & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
            overlay[border] = color
        self.design_overlay.setImage(overlay, autoLevels=False)
        self.design_overlay.setRect(rect)
        self.dust_image.setRect(rect)
        self.brush_overlay.clear()
        self.brush_overlay.setRect(rect)
        self.map.setRange(xRange=(0, side), yRange=(0, side), padding=0.015)
        self.profile_segment.blockSignals(True)
        self.profile_segment.clear();self.profile_segment.addItem("Все сегменты",None)
        for index,segment in enumerate(segments_for(self.config)):self.profile_segment.addItem(segment.name,index)
        self.profile_segment.blockSignals(False)
        self.update_profile()
        self.last_revision = -1
        while len(self.drone_labels)<len(self.sim.drones):
            label=pg.TextItem(color=self.palette[len(self.drone_labels)],anchor=(0,1));self.map.addItem(label);self.drone_labels.append(label)
        for index,label in enumerate(self.drone_labels):label.setVisible(index<len(self.sim.drones))
        self.fleet_table.setRowCount(len(self.sim.drones))

    def toggle_brush_panel(self,visible):
        self.plan_editor.setVisible(visible)
        self.plan_editor.enabled.setChecked(visible)

    def toggle_brush(self,enabled):
        self.view_box.brush_enabled=enabled
        self.brush_cursor.setVisible(enabled)
        if enabled:
            self.running=False;self.credit=0;self.play.setText("▶ Start")
        self.statusBar().showMessage("Кисть: зажмите левую кнопку и рисуйте. Колесо — масштаб. Примените план после посадки флота." if enabled else "Колесо — масштаб; перетаскивание — панорама.")

    def move_brush_cursor(self,point):
        if not self.view_box.brush_enabled:return
        width=self.plan_editor.width.value()
        self.brush_cursor.setRect(point[0]-width/2,point[1]-width/2,width,width)

    def add_brush_stroke(self,points):
        from .plan import PlanSegment
        try:
            PlanSegment(points,self.plan_editor.mode.currentData(),self.plan_editor.width.value(),self.plan_editor.height.value()).validate(self.config)
            if len(self.plan_editor.segments)>=100:raise ValueError("План ограничен 100 сегментами")
            self.plan_editor.add_stroke(points)
        except ValueError as error:self.plan_editor.info.setText(str(error))

    def preview_brush(self):
        t=self.sim.terrain
        target,_=target_surface(t,self.plan_editor.plan())
        overlay=np.zeros((*t.height.shape,4),dtype=np.uint8)
        overlay[target<t.height-1e-12]=[65,210,250,80]
        overlay[target>t.height+1e-12]=[112,242,132,95]
        self.brush_overlay.setImage(overlay,autoLevels=False)
        self.brush_overlay.setRect(QtCore.QRectF(0,0,t.cell*t.height.shape[0],t.cell*t.height.shape[0]))
        self.plan_editor.info.setText(f"Рисунок не применён: {len(self.plan_editor.segments)} сегментов. Голубой — выемка, зелёный — засыпка. «Применить план» сохранит текущий рельеф.")

    def apply_brush_plan(self):
        self.running=False;self.credit=0;self.play.setText("▶ Start")
        try:self.sim.apply_plan(self.plan_editor.plan())
        except ValueError as error:
            self.plan_editor.info.setText(str(error));return False
        self.config=self.sim.config;self.plan_editor.dirty=False
        view=self.map.viewRange()
        self.setup_map();self.map.setRange(xRange=view[0],yRange=view[1],padding=0)
        self.plan_editor.info.setText("План применён. Нажмите Start. Для засыпки используются отвал и запланированная выемка; при нехватке грунта расчёт остановится с причиной.")
        self.refresh(True);return True

    def land_for_plan(self):
        if self.sim.state!=MissionState.RUNNING:return
        self.sim.request_landing();self.running=True;self.last_wall=time.perf_counter();self.play.setText("Посадка…")
        self.plan_editor.enabled.setChecked(False)

    def save_brush_plan(self):
        if self.plan_editor.dirty or self.config.plan is None:
            self.plan_editor.info.setText("Перед сохранением примените нарисованный план.");return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,"Сохранить план","plan.json","План ATS (*.json)")
        if path:
            try:save_plan(path,self.config)
            except OSError as error:self.plan_editor.info.setText(str(error))

    def load_brush_plan(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,"Открыть план","","План ATS (*.json)")
        if path:
            try:self.replace_simulation(load_plan(path,self.config))
            except (ValueError,OSError,KeyError,TypeError) as error:self.plan_editor.info.setText(f"Не удалось открыть план: {error}")

    def update_profile(self):
        if not hasattr(self,"profile_extra"):return
        direction=self.profile_direction.currentData()
        fraction=self.profile_position.value()/100
        self.profile_position.setEnabled(direction=="cross")
        self.profile_position_label.setText(f"Срез: {fraction*100:.0f}%")
        segments=segments_for(self.config)
        selection=self.profile_segment.currentData()
        if selection is not None and selection>=len(segments):selection=None
        data=profiles(self.sim.terrain,segments,selection,direction,fraction)
        for item in self.profile_extra+self.profile_boundaries:self.profile.removeItem(item)
        self.profile_extra=[];self.profile_boundaries=[]
        for curve in [self.current_curve,self.target_curve,self.reference_curve]:curve.clear()
        points=[]
        for index,p in enumerate(data):
            if index==0:
                curves=[self.current_curve,self.target_curve,self.reference_curve]
            else:
                color=self.palette[index%len(self.palette)]
                curves=[self.profile.plot(pen=pg.mkPen(color,width=2),name=p.name if len(data)<=6 else None),self.profile.plot(pen=pg.mkPen(color,width=1,style=QtCore.Qt.PenStyle.DashLine)),self.profile.plot(pen=pg.mkPen("#627789",width=1))]
                self.profile_extra.extend(curves)
            for curve,values in zip(curves,[p.current,p.target,p.reference]):curve.setData(p.axis,values,connect="finite")
            points.extend([*p.points,(np.nan,np.nan)])
            for boundary in p.boundaries:
                line=pg.InfiniteLine(pos=boundary,angle=90,pen=pg.mkPen("#627789",style=QtCore.Qt.PenStyle.DotLine));self.profile.addItem(line);self.profile_boundaries.append(line)
        if points:
            points=np.asarray(points);self.profile_line.setData(points[:,0],points[:,1],connect="finite")
        else:self.profile_line.clear()
        self.profile.setLabel("bottom","Поперечное расстояние от оси" if direction=="cross" else "Суммарная длина сегментов",units="m")
        title="Поперечные профили всех сегментов" if direction=="cross" and selection is None else "Поперечный профиль" if direction=="cross" else "Сквозной профиль — сегменты разделены пунктиром"
        self.profile.setTitle(title)
        self.profile.enableAutoRange()

    def closeEvent(self,event):
        self.timer.stop();self.running=False
        if hasattr(self,'logistics_window'):self.logistics_window.close()
        super().closeEvent(event)

    def show_logistics(self):
        from .logistics_ui import LogisticsWindow
        if not hasattr(self,'logistics_window'):
            self.logistics_window=LogisticsWindow(self)
        self.logistics_window.show();self.logistics_window.raise_();self.logistics_window.activateWindow()

    def show_experiments(self):
        from .logistics_ui import ExperimentDialog
        dialog=ExperimentDialog(self);dialog.exec()

    def edit_fleet(self):
        previous=self.running;self.running=False;self.credit=0
        dialog=FleetDialog(self.sim.pending_equipment or self.config,self)
        if dialog.exec()==QtWidgets.QDialog.DialogCode.Accepted:
            try:self.apply_equipment(dialog.result_config,previous)
            except ValueError as error:
                self.running=previous;QtWidgets.QMessageBox.warning(self,"Неверная комплектация",str(error))
        else:self.running=previous
        self.last_wall=time.perf_counter();self.play.setText("⏸ Pause" if self.running else "▶ Start")

    def apply_equipment(self,config,resume=None):
        previous=self.running if resume is None else resume
        result=self.sim.request_equipment(config)
        if result=="pending":
            if self._equipment_resume is None:self._equipment_resume=previous
            self.plan_editor.enabled.setChecked(False);self.running=True
        else:self.running=previous;self.sync_equipment()
        self.play.setText("⏸ Pause" if self.running else "▶ Start");self.credit=0
        self.refresh(True)
        return result

    def sync_equipment(self):
        self.config=self.sim.config
        self.panels.setValue(self.config.base.panel_area);self.base_mass.setValue(self.config.base.mass)
        self.update_solar_preview()
        while len(self.drone_labels)<len(self.sim.drones):
            label=pg.TextItem(color=self.palette[len(self.drone_labels)],anchor=(0,1));self.map.addItem(label);self.drone_labels.append(label)
        for index,label in enumerate(self.drone_labels):label.setVisible(index<len(self.sim.drones))
        self.fleet_table.setRowCount(len(self.sim.drones))

    def cancel_equipment(self):
        self.sim.cancel_equipment()
        if self._equipment_resume is not None:self.running=self._equipment_resume
        self._equipment_resume=None;self.credit=0
        self.play.setText("⏸ Pause" if self.running else "▶ Start");self.refresh()

    def show_daily_stats(self):
        self.daily_stats=DailyStatsWindow(self);self.daily_stats.show()

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
        self._equipment_resume = None
        self.play.setText("▶ Start")
        self.credit = 0
        self.config = config
        self.sim = sim
        self.sync_equipment()
        self.plan_editor.segments=deepcopy(config.plan.segments) if config.plan is not None else []
        self.plan_editor.dirty=False;self.plan_editor.rebuild()
        self.plan_editor.default_depth=config.trench.depth
        self.last_event = 0
        self.log.clear()
        self.setup_map()
        self.change_dust()
        self.refresh(True)

    def new_scenario(self):
        config = deepcopy(self.config)
        config.plan=None;config.initial_height=None
        q = config.trench
        q.length, q.width, q.depth = self.length.value(), self.width.value(), self.depth.value()
        q.x, q.y, q.angle_deg = self.cx.value(), self.cy.value(), self.angle.value()
        config.seed = self.seed.value()
        config.start_hour = self.start_hour.value()
        config.terrain.roughness = self.roughness.value()
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
        if self.running:
            if self.sim.pending_equipment is None:self.sim.landing_requested=False
            self.plan_editor.enabled.setChecked(False)
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
                if self.sim.config is not self.config:self.sync_equipment()
                self.credit -= 1
                steps += 1
                if self._equipment_resume is not None and self.sim.pending_equipment is None:
                    self.running=self._equipment_resume;self._equipment_resume=None
                    self.play.setText("⏸ Pause" if self.running else "▶ Start")
                    if not self.running:self.credit=0;break
                logistics=self.sim.logistics
                robot=logistics.excavator if logistics is not None else None
                robot_ready=(robot is None or robot.bucket_kg==0 and robot.active_container is None)
                if logistics is not None:robot_ready=robot_ready and all(c.soil_kg==0 for c in logistics.containers)
                if self.sim.landing_requested and robot_ready and all(d.grounded and d.cargo==0 for d in self.sim.drones):
                    self.running=False;self.credit=0;self.play.setText("▶ Start")
                    self.plan_editor.info.setText("Флот приземлён. Можно применить нарисованный план.")
                    break
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
            self.update_profile()
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
        if s.logistics is not None:
            l=s.logistics;e=l.excavator
            if l.field_rect is not None:
                x,y,w,h=l.field_rect;self.solar_field_outline.setData([x,x+w,x+w,x,x],[y,y,y+h,y+h,y])
            else:self.solar_field_outline.clear()
            self.container_markers.setData([dict(pos=o.position[:2],brush='#a8e3ae' if o.soil_kg==0 else '#e9a560') for o in l.containers])
            if e is not None:
                self.excavator_marker.setData([e.position[0]],[e.position[1]])
                empty=sum(c.soil_kg==0 for c in l.containers);filled=len(l.containers)-empty
                self.excavator_label.setText(f'Крот · {e.state.value}\nТара: пустых {empty}, с грунтом {filled}')
                self.excavator_label.setPos(e.position[0]+1.,e.position[1]-1.)
                phi=np.linspace(0,2*np.pi,80)
                self.excavator_reach.setData(e.position[0]+l.config.reach_m*np.cos(phi),e.position[1]+l.config.reach_m*np.sin(phi))
            else:self.excavator_marker.clear();self.excavator_reach.clear();self.excavator_label.setText('')
        else:
            self.excavator_marker.clear();self.container_markers.clear();self.excavator_reach.clear();self.solar_field_outline.clear();self.excavator_label.setText('')
        self.drone_marker.setData([{'pos':tuple(o.position[:2]),'pen':pg.mkPen(self.palette[o.identifier],width=3),'symbol':'+','size':20} for o in s.drones])
        self.slot_markers.setData([{'pos':tuple(s.slot_position(i)[:2]),'brush':'#f4b651' if occupant is not None else '#41566b','symbol':'s','size':10} for i,occupant in enumerate(s.slots)])
        for o,label in zip(s.drones,self.drone_labels):
            operation=" · грейфер вниз" if o.state==DroneState.DUMP and not o.dump_released else " · грейфер вверх" if o.state==DroneState.DUMP else ""
            laser=f" · лазер {o.laser_power_w/1000:.1f} кВт" if o.laser_power_w>0 else ""
            label.setText(f"Д{o.serial} · {o.cargo:.0f} кг · z {o.position[2]:.1f} м{operation}{laser}")
            label.setPos(o.position[0]+.4,o.position[1]+.4+o.identifier*.3)
        self.base_label.setPos(b.position[0]+0.6, b.position[1]-0.4)
        routes=[p for o in s.drones if o.trip for p in [b.position[:2],o.trip.dig,o.trip.dump,b.position[:2],(np.nan,np.nan)]]
        if s.logistics is not None:
            routes.extend(p for j in s.logistics.active.values() for p in [j.origin[:2],j.destination[:2],(np.nan,np.nan)])
        if routes:
            points = np.array(routes)
            self.route.setData(points[:, 0], points[:, 1])
        else:
            self.route.setData([], [])
        report = s.report(full=False)
        beams=[]
        for link in s.laser_links:
            if link.received_w>0:beams.extend([link.source[:2],link.drone.position[:2],(np.nan,np.nan)])
        if beams:
            points=np.asarray(beams);self.laser_beams.setData(points[:,0],points[:,1],connect="finite")
        else:self.laser_beams.clear()
        self.fleet_table.setUpdatesEnabled(False)
        for row,o in enumerate(report['drones']):
            values=[f"{o['serial']} · {o['name']}",o['state'],f"{o['battery_percent']:.0f}%",f"{o['cargo_kg']:.1f}",str(o['trips']),f"{o['excavated_mass_kg']:.0f} / {o['filled_mass_kg']:.0f}",f"{o['consumed_kwh']:.3f}",f"{o['waiting_seconds']/60:.1f}",f"{o['charging_seconds']/60:.1f}",str(o['traffic_yields']),f"{o['last24_volume_m3']:.4f}",f"{o['laser_power_w']/1000:.2f}"]
            for col,value in enumerate(values):self.fleet_table.setItem(row,col,QtWidgets.QTableWidgetItem(value))
            if 'task' in o:self.fleet_table.item(row,1).setToolTip(f"{o['task']} · {o['attachment']}\nБатареи: {', '.join(o['battery_ids'])}\nЗамен комплектов: {o['battery_swaps']}; до завершения замены: {o['battery_swap_remaining_seconds']:.0f} с\nГрузовой лимит: {o['transport_payload_kg']:.1f} кг")
            self.fleet_table.item(row,2).setToolTip(f"Порог зарядки {o['charge_threshold_percent']:.0f}%; полных зарядок: {o['charge_cycles']}. Полёт сейчас: {o['current_flight_seconds']:.0f} с; самый длинный: {o['longest_flight_seconds']:.0f} с; в воздухе всего: {o['airborne_seconds']:.0f} с.")
            self.fleet_table.item(row,10).setToolTip(f"За текущие солнечные сутки: {o['today_volume_m3']:.4f} м³ / {o['today_mass_kg']:.2f} кг. Всего доставлено: {o['transported_volume_m3']:.4f} м³. Подробности — «По суткам».")
            self.fleet_table.item(row,11).setToolTip(f"{o['laser_status']}\nПолучено за миссию: {o['laser_received_kwh']:.3f} кВт·ч. Потребление двигателей и лебёдки за последний шаг: {o['flight_power_w']/1000:.2f} кВт. Показана фактическая мощность за последнюю физическую секунду.")
        self.fleet_table.setUpdatesEnabled(True)
        status = {MissionState.RUNNING: "Автономная работа" if self.running else "Готов / пауза", MissionState.COMPLETED: "✓ Миссия завершена", MissionState.BLOCKED: "⚠ Работа заблокирована"}[s.state]
        if s.pending_equipment is not None:status="Посадка для установки оборудования"
        self.cancel_equipment_button.setVisible(s.pending_equipment is not None)
        self.status.setText(f"<b style='font-size:17px'>{status}</b><br>{s.reason}")
        self.progress.setValue(round(report["completion_percent"]*10))
        self.progress.setFormat(f"{report['completion_percent']:.1f}% · допуск {self.config.terrain.completion_volume:.3f} м³")
        rows = [
            ("Прошло с запуска", report["elapsed_display"]),
            ("Дата / солнечное время", f"{s.date_text} · {s.clock_text}"),
            ("Флот / очередь",f"{len(s.drones)} / {len(s.queue)}"), ("База", b.state.value),
            ("Зарядные слоты",f"{sum(v is not None for v in s.slots)} / {len(s.slots)} · {b.config.charge_power/1000:.1f} кВт всего"),
            ("Проект / остаток", f"{report['initial_excavation_m3']:.3f} / {report['remaining_m3']:.3f} м³"),
            ("Выемка / засыпка осталось",f"{report['remaining_cut_m3']:.3f} / {report['remaining_fill_m3']:.3f} м³"),
            ("Вынуто / рейсы", f"{report['excavated_m3']:.3f} м³ / {s.trips}"),
            ("Общий груз", f"{report['cargo_kg']:.1f} кг"),
            ("База / панели", f"{b.config.mass:.0f} кг / {b.config.panel_area:.1f} м²"),
            ("Широта / световой день", f"{b.config.latitude_deg:+.1f}° / {report['daylight_hours']:.1f} ч"),
            ("Солнце: высота / поток", f"{report['sun_elevation_deg']:.1f}° / {report['irradiance_w_m2']:.0f} Вт/м²"),
            ("Батареи флота", f"{report['drone_kwh']:.2f} / {sum(o.config.battery for o in s.drones)/KWH:.2f} кВт·ч"),
            ("Батарея базы", f"{b.battery/KWH:.2f} / {b.config.battery/KWH:.2f} кВт·ч" if b.config.battery else "Без аккумулятора · напрямую от солнца"),
            ("Солнечная мощность", f"{b.solar_power/1000:.2f} кВт"),
            ("Солнце / расход", f"{report['solar_generated_kwh']:.3f} / {report['consumed_kwh']:.3f} кВт·ч"),
            ("Потери зарядки", f"{report['charging_losses_kwh']:.3f} кВт·ч"),
            ("Лазер: база → дроны",f"{report['laser_input_power_w']/1000:.2f} → {report['laser_received_power_w']/1000:.2f} кВт"),
            ("Потери лазера",f"{report['laser_losses_kwh']:.3f} кВт·ч"),
            ("Ввоз / вывоз энергии",f"{report['equipment_imported_kwh']:.3f} / {report['equipment_exported_kwh']:.3f} кВт·ч"),
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
