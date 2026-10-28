"""Qt presentation. Fixed engine steps are bounded by a per-frame time budget."""
from copy import deepcopy
from pathlib import Path
import time
import math
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


def terrain_view(height, cell, bounds, pixels=512):
    """Crop and average display pixels only; the physical float64 grid is untouched."""
    n=height.shape[0]
    x0=max(0,math.floor(bounds[0][0]/cell)-1);x1=min(n,math.ceil(bounds[0][1]/cell)+1)
    y0=max(0,math.floor(bounds[1][0]/cell)-1);y1=min(n,math.ceil(bounds[1][1]/cell)+1)
    if x1-x0<2 or y1-y0<2:x0=y0=0;x1=y1=n
    stride=2**max(0,math.ceil(math.log2(max(x1-x0,y1-y0)/pixels)))
    x0=x0//stride*stride;y0=y0//stride*stride
    x1=min(n,math.ceil(x1/stride)*stride);y1=min(n,math.ceil(y1/stride)*stride)
    view=height[y0:y1,x0:x1]
    if stride>1:
        view=np.pad(view,((0,(-view.shape[0])%stride),(0,(-view.shape[1])%stride)),mode='edge')
        view=view.reshape(view.shape[0]//stride,stride,view.shape[1]//stride,stride).mean(axis=(1,3))
    rect=(x0*cell,y0*cell,view.shape[1]*stride*cell,view.shape[0]*stride*cell)
    return view,rect,stride


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config, session=None):
        super().__init__()
        self.config = session.simulation.config if session else deepcopy(config)
        self.sim = session.simulation if session else Simulation(self.config)
        config=self.config
        self.landscape = session.landscape if session else None
        self.complex_project=getattr(session,'complex_project',None) if session else None
        self.template_library=(self.complex_project.templates if self.complex_project is not None else
                               list((session.ui or {}).get('complex_templates',[])) if session else [])
        self.power_request_state=((session.ui or {}).get('power_request') if session else None) or (
            (self.config.energy_design or {}).get('request'))
        self._session_path=None
        self.running = False
        self._equipment_resume = None
        self.credit = 0.0
        self.last_wall = time.perf_counter()
        self.last_render_wall=-float('inf')
        self.actual_speed=0.;self.actual_render_hz=0.
        self.performance_wall=self.last_wall;self.performance_sim=self.model_time();self.performance_renders=0
        self._render_cost=.02;self._last_view=None
        self.last_terrain_wall = -1.0
        self.last_revision = -1
        self.last_event = 0
        self.setWindowTitle("ATS 2.0 · Автономные земляные работы")
        files=self.menuBar().addMenu('Файл')
        self.save_action=files.addAction('Сохранить симуляцию…');self.save_action.setShortcut('Ctrl+S')
        self.save_action.triggered.connect(self.save_session)
        self.save_as_action=files.addAction('Сохранить симуляцию как…');self.save_as_action.setShortcut('Ctrl+Shift+S')
        self.save_as_action.triggered.connect(lambda:self.save_session(force_dialog=True))
        self.load_action=files.addAction('Загрузить симуляцию…');self.load_action.setShortcut('Ctrl+O')
        self.load_action.triggered.connect(self.load_session)
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
        title = QtWidgets.QLabel("<b style='font-size:22px'>ATS 2.0</b>")
        header.addWidget(title, 1)
        self.play = QtWidgets.QPushButton("▶ Start")
        self.play.setObjectName("play")
        self.play.clicked.connect(self.toggle_play)
        header.addWidget(self.play)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        header.addWidget(self.reset_button)
        from .ui_windows import menu_button,hidden_action
        actions_menu=menu_button(header,self,'Управление ▾')
        fleet_button=QtWidgets.QPushButton("Оборудование")
        fleet_button.clicked.connect(self.edit_fleet);hidden_action(actions_menu,self,fleet_button)
        daily_button=QtWidgets.QPushButton("Статистика")
        daily_button.clicked.connect(self.show_daily_stats);hidden_action(actions_menu,self,daily_button)
        self.cancel_equipment_button=QtWidgets.QPushButton("Отменить установку")
        self.cancel_equipment_button.clicked.connect(self.cancel_equipment);header.addWidget(self.cancel_equipment_button)
        self.cancel_equipment_button.hide()
        compare_button=QtWidgets.QPushButton("Сравнить 1…N")
        compare_button.clicked.connect(self.compare_fleet);hidden_action(actions_menu,self,compare_button)
        logistics_button=QtWidgets.QPushButton('Склад / диспетчер')
        logistics_button.clicked.connect(self.show_logistics);hidden_action(actions_menu,self,logistics_button)
        experiments_button=QtWidgets.QPushButton('Новый эксперимент')
        experiments_button.clicked.connect(self.show_experiments);hidden_action(actions_menu,self,experiments_button)
        self.brush_button=QtWidgets.QPushButton("Кисть / план")
        self.brush_button.setCheckable(True);header.addWidget(self.brush_button)
        self.speed = QtWidgets.QComboBox()
        for factor in [1, 10, 50, 200, 1000, 2000]:
            self.speed.addItem(f"{factor}×", factor)
        self.speed.setCurrentIndex(2)
        self.speed.setToolTip("Целевая скорость виртуального времени. Физический шаг остаётся 1 с; доступное ускорение зависит от производительности компьютера.")
        self.speed.currentIndexChanged.connect(self.speed_changed)
        header.addWidget(self.speed)
        layout.addLayout(header)
        split = QtWidgets.QSplitter()
        layout.addWidget(split, 1)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        self.left_layout=left_layout
        left_layout.setContentsMargins(0, 0, 4, 0)
        legend = QtWidgets.QLabel("Рельеф: синий — выемка, песочный — исходный уровень, светлый — насыпь | План: голубой — выемка, зелёный — засыпка, оранжевый — отвал; розовый — профиль")
        legend.setWordWrap(True)
        left_layout.addWidget(legend)
        self.energy_status=QtWidgets.QLabel();self.energy_status.setWordWrap(True)
        self.energy_status.linkActivated.connect(lambda _:self.show_power_design())
        left_layout.addWidget(self.energy_status)
        self.plan_editor=PlanEditor(config,self)
        left_layout.addWidget(self.plan_editor);self.plan_editor.hide()
        self.view_box=BrushViewBox()
        self.map_frame=QtWidgets.QWidget()
        map_layout=QtWidgets.QVBoxLayout(self.map_frame);map_layout.setContentsMargins(0,0,0,0)
        map_tools=QtWidgets.QHBoxLayout()
        self.zoom_in=QtWidgets.QPushButton('+');self.zoom_out=QtWidgets.QPushButton('−')
        self.zoom_fit=QtWidgets.QPushButton('Вся карта')
        self.map_fullscreen=QtWidgets.QPushButton('Карта ↗ · F11');self.map_fullscreen.setCheckable(True)
        self.map_shortcut=QtGui.QShortcut(QtGui.QKeySequence('F11'),self)
        self.map_shortcut.activated.connect(lambda:self.map_fullscreen.toggle())
        self.zoom_label=QtWidgets.QLabel('100%')
        for widget in (self.zoom_out,self.zoom_label,self.zoom_in,self.zoom_fit,self.map_fullscreen):map_tools.addWidget(widget)
        self.soil_view=QtWidgets.QCheckBox('Грунты');self.soil_view.setToolTip('Цвет материала на поверхности; условные слои задаются в окне ландшафта до начала работ')
        self.soil_view.toggled.connect(lambda _:self.refresh(True));map_tools.addWidget(self.soil_view)
        self.labels_view=QtWidgets.QCheckBox('Подписи');self.labels_view.toggled.connect(lambda _:self.refresh(True));map_tools.addWidget(self.labels_view)
        map_tools.addStretch();map_layout.addLayout(map_tools)
        self.soil_legend=QtWidgets.QLabel();self.soil_legend.setWordWrap(True);self.soil_legend.hide();map_layout.addWidget(self.soil_legend)
        self.landscape_button=QtWidgets.QPushButton('Ландшафт / импорт DEM…')
        left_layout.addWidget(self.landscape_button);self.landscape_button.clicked.connect(self.show_landscape)
        self.transition_status=QtWidgets.QLabel();self.transition_status.setWordWrap(True);self.transition_status.hide()
        left_layout.addWidget(self.transition_status)
        self.map_dialog=None
        self.map = pg.PlotWidget(viewBox=self.view_box)
        self.map.setAspectLocked(True)
        self.map.setLabel("bottom", "X", units="m")
        self.map.setLabel("left", "Y", units="m")
        self.map.showGrid(x=True, y=True, alpha=0.12)
        map_layout.addWidget(self.map,1);left_layout.addWidget(self.map_frame,7)
        self.zoom_in.clicked.connect(lambda:self.zoom_map(1.25))
        self.zoom_out.clicked.connect(lambda:self.zoom_map(1/1.25))
        self.zoom_fit.clicked.connect(self.fit_map)
        self.map_fullscreen.toggled.connect(self.toggle_map_fullscreen)
        self.view_box.sigRangeChanged.connect(self.update_zoom_label)
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
        self.base_outline=pg.PlotDataItem(pen=pg.mkPen('#ffdb8a',width=2))
        self.map.addItem(self.base_outline)
        self.workfront_outline=pg.PlotDataItem(pen=pg.mkPen('#92f3a2',width=2,style=QtCore.Qt.PenStyle.DashLine))
        self.map.addItem(self.workfront_outline)
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
        profile_controls.addStretch();self.profile_controls_widget=QtWidgets.QWidget();self.profile_controls_widget.setLayout(profile_controls)
        left_layout.addWidget(self.profile_controls_widget)
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
        from .shared_ui import initialize
        initialize(self)
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
        self.channel_status=QtWidgets.QLabel();self.channel_status.setWordWrap(True)
        self.channel_status.setStyleSheet('color: #83e3dc; padding: 5px;')
        right_layout.addWidget(self.channel_status)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        right_layout.addWidget(self.progress)
        self.telemetry = QtWidgets.QLabel()
        self.telemetry.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.telemetry.setWordWrap(True);self.telemetry.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        telemetry_scroll=QtWidgets.QScrollArea();telemetry_scroll.setWidgetResizable(True)
        telemetry_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame);telemetry_scroll.setMinimumHeight(180)
        telemetry_scroll.setWidget(self.telemetry);right_layout.addWidget(telemetry_scroll,3)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True);scroll.setMinimumHeight(180)
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
        side=config.terrain.n*config.terrain.cell_size
        self.map_size=self.spin(32,1024,side,16,2);self.map_size.setSuffix(' м')
        self.grid_info=QtWidgets.QLabel()
        self.cx = self.spin(0.5, side-.5, config.trench.x, 0.25)
        self.cy = self.spin(0.5, side-.5, config.trench.y, 0.25)
        self.angle = self.spin(0, 75, config.trench.angle_deg or 0, 1)
        self.angle.setSpecialValueText('Авто · минимальная ширина')
        self.angle.setToolTip('0: наиболее крутой устойчивый профиль по слоям; число: проектный угол к горизонту. Устойчивость грунта задаётся отдельно.')
        self.start_hour = self.spin(0, 23.99, config.start_hour, 0.5)
        self.roughness = self.spin(0, 0.03, config.terrain.roughness, 0.005, 3)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(config.seed)
        for label, widget in [("Длина дна", self.length), ("Ширина дна", self.width), ("Глубина", self.depth), ("Центр X", self.cx), ("Центр Y", self.cy), ("Проектный откос, °", self.angle), ("Стартовый час", self.start_hour), ("Неровности, ±м", self.roughness), ("Seed", self.seed)]:
            form.addRow(label, widget)
        form.addRow('Сторона новой карты',self.map_size);form.addRow(self.grid_info)
        self.map_size.valueChanged.connect(self.update_map_size_controls);self.update_map_size_controls()
        solar_group = QtWidgets.QGroupBox("База и солнце")
        solar_form = QtWidgets.QFormLayout(solar_group)
        solar_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.panels = self.spin(0, 10000000, self.config.base.panel_area, 1)
        self.base_mass = self.spin(0, 10000000, self.config.base.mass, 100, 1)
        for widget in [self.base_mass]:
            widget.setReadOnly(True);widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            widget.setToolTip("Установленное оборудование и текущие запасы базы, без пристыкованных дронов. Обновляется во время работы.")
        self.latitude = self.spin(-90, 90, config.base.latitude_deg, 1,4)
        self.latitude.setEnabled(not bool(config.georeference and config.georeference['source'].get('crs')=='EPSG:4326'))
        self.latitude.setToolTip('На географической карте широта берётся из координат базы и участвует в расчёте солнечной генерации.')
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
        self.base_dimensions=QtWidgets.QLabel();self.base_dimensions.setWordWrap(True)
        self.base_dimensions.setToolTip('Габариты корпуса: 2 радиуса × 2 радиуса × высота крыши из геометрии модели. Посадочная зона охватывает установленные площадки. Поле панелей показано отдельно; размеры корпуса не масштабируются по массе оборудования.')
        solar_form.addRow(self.base_dimensions)
        self.panel_info = QtWidgets.QLabel()
        self.panel_info.setWordWrap(True)
        solar_form.addRow(self.panel_info)
        self.apply_panels_button=QtWidgets.QPushButton('Применить площадь панелей сейчас')
        self.apply_panels_button.clicked.connect(self.apply_panel_area);solar_form.addRow(self.apply_panels_button)
        self.base_batteries=QtWidgets.QSpinBox();self.base_batteries.setRange(0,1000)
        self.base_batteries.setValue(sum(b.quantity for b in config.base.equipment.batteries))
        solar_form.addRow('Аккумуляторов базы, шт.',self.base_batteries)
        self.battery_type_info=QtWidgets.QLabel();self.battery_type_info.setWordWrap(True)
        solar_form.addRow(self.battery_type_info)
        self.apply_base_batteries_button=QtWidgets.QPushButton('Применить аккумуляторы сейчас')
        self.apply_base_batteries_button.clicked.connect(self.apply_base_battery_count)
        solar_form.addRow(self.apply_base_batteries_button)
        self.power_design_button=QtWidgets.QPushButton('Энергоснабжение и шаблон…');self.power_design_button.clicked.connect(self.show_power_design)
        solar_form.addRow(self.power_design_button)
        self.complexes_button=QtWidgets.QPushButton('Комплексы общего проекта…');self.complexes_button.clicked.connect(self.show_complexes)
        solar_form.addRow(self.complexes_button)
        self.layout_button=QtWidgets.QPushButton('Паспорт и платформы…');self.layout_button.clicked.connect(self.show_base_layout);solar_form.addRow(self.layout_button)
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
        from .ui_windows import compact_workspace
        compact_workspace(self,split,right,right_layout,telemetry_scroll,scroll,
            [('Новый сценарий',project),('База и солнце',solar_group),('Пыль и ветер',dust_group)],self.profile_controls_widget)
        self.statusBar().showMessage("Колесо мыши — масштаб; перетаскивание — панорама. Размеры сценария применяются кнопкой создания.")
        self.render_mode=QtWidgets.QComboBox()
        for label,value in [('Карта: авто','auto'),('Карта: подробно','detail'),('Без карты','off')]:self.render_mode.addItem(label,value)
        self.render_mode.setToolTip('Авто уменьшает только отображение большой карты и частоту перерисовки. Без карты расчёт и телеметрия продолжаются. Физический шаг и сетка сохраняются.')
        self.render_mode.currentIndexChanged.connect(self.render_changed)
        self.statusBar().addPermanentWidget(self.render_mode)
        self.speed_status=QtWidgets.QLabel('Факт: 0×')
        self.speed_status.setToolTip('Фактически прошедшие секунды симуляции за одну реальную секунду. Физический шаг всегда 1 с. Высокая цель может ограничиваться скоростью расчёта.')
        self.statusBar().addPermanentWidget(self.speed_status)
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.on_frame)
        self.timer.start(config.display_interval_ms)
        self.setup_map()
        self.sync_equipment()
        self.sync_wind_controls()
        if session:self.restore_session_ui(session.ui)
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
        self._display_origin=deepcopy((self.sim.config.georeference or {}).get('origin_m'))
        t = self.sim.terrain
        if self.complex_project is not None and self.complex_project.shared_site is not None:t=self.complex_project.shared_site.core
        self._assignment_revision=None
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
        self.fit_map()
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

    def update_map_size_controls(self):
        side=self.map_size.value();cell=self.config.terrain.cell_size
        from .maps import workspace_cell
        cell=workspace_cell(side,cell)
        self.cx.setMaximum(side-.5);self.cy.setMaximum(side-.5)
        self.length.setMaximum(side-2*cell);self.width.setMaximum(side-2*cell)
        n=round(side/cell);self.grid_info.setText(f'Сетка {n}×{n} · ячейка {cell:g} м')

    def fit_map(self):
        side=self.sim.config.terrain.n*self.sim.config.terrain.cell_size
        self.map.setRange(xRange=(0,side),yRange=(0,side),padding=0)

    def update_zoom_label(self,*args):
        side=self.sim.config.terrain.n*self.sim.config.terrain.cell_size
        ranges=self.map.viewRange();span=min(ranges[0][1]-ranges[0][0],ranges[1][1]-ranges[1][0])
        self.zoom_label.setText(f'{100*side/max(span,1e-9):.0f}%')

    def zoom_map(self,factor):
        ranges=self.map.viewRange();side=self.sim.config.terrain.n*self.sim.config.terrain.cell_size
        span=min(b-a for a,b in ranges)
        target=max(2*self.sim.terrain.cell,min(4*side,span/factor))
        self.view_box.scaleBy((target/span,target/span))

    def toggle_map_fullscreen(self,enabled):
        if not enabled:
            if self.map_dialog is not None:self.map_dialog.close()
            return
        self._map_home_index=self.left_layout.indexOf(self.map_frame)
        dialog=QtWidgets.QDialog(self);dialog.setWindowTitle('ATS · Карта · Esc / F11 — вернуть')
        box=QtWidgets.QVBoxLayout(dialog);box.setContentsMargins(0,0,0,0);box.addWidget(self.map_frame)
        self.map_dialog=dialog;dialog.finished.connect(self.restore_map)
        dialog.shortcut=QtGui.QShortcut(QtGui.QKeySequence('F11'),dialog)
        dialog.shortcut.activated.connect(dialog.close)
        self.map_fullscreen.setText('Вернуть карту · F11 / Esc');dialog.showFullScreen()

    def restore_map(self,*args):
        if self.map_dialog is None:return
        dialog=self.map_dialog;self.map_dialog=None
        self.left_layout.insertWidget(self._map_home_index,self.map_frame,7);self.map_frame.show()
        self.map_fullscreen.blockSignals(True);self.map_fullscreen.setChecked(False);self.map_fullscreen.blockSignals(False)
        self.map_fullscreen.setText('Развернуть карту · F11');dialog.deleteLater()

    def toggle_brush_panel(self,visible):
        self.plan_editor.setVisible(visible)
        if hasattr(self,'plan_window'):
            self.plan_window.setVisible(visible)
            if visible:self.plan_window.raise_()
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
        group=self.complex_project if self.complex_project is not None and self.complex_project.shared_site is not None else None
        if group is None and self.sim.state!=MissionState.RUNNING:return
        for s in ([u.simulation for u in group.units] if group is not None else [self.sim]):s.request_landing()
        self.running=True;self.last_wall=time.perf_counter();self.play.setText("Посадка…")
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
            if self.sim.terrain.soil is not None:
                soil=self.sim.terrain.soil.profile
                for depth,material in zip(soil.boundaries,soil.materials[1:]):
                    contact=p.reference-depth
                    contact=np.where(contact<=p.current+1e-9,contact,np.nan)
                    curve=self.profile.plot(p.axis,contact,pen=pg.mkPen(material.color,width=1,style=QtCore.Qt.PenStyle.DotLine),
                        name=f'Граница: {material.name}' if index==0 else None,connect='finite')
                    self.profile_extra.append(curve)
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
        if self.map_dialog is not None:self.map_dialog.close()
        if hasattr(self,'logistics_window'):self.logistics_window.close()
        if hasattr(self,'power_window'):self.power_window.close()
        if hasattr(self,'complex_window'):self.complex_window.close()
        if hasattr(self,'base_layout_window'):self.base_layout_window.close()
        for window in getattr(self,'detail_windows',[]):window.close()
        super().closeEvent(event)

    def show_logistics(self):
        from .logistics_ui import LogisticsWindow
        if not hasattr(self,'logistics_window'):
            self.logistics_window=LogisticsWindow(self)
        self.logistics_window.show();self.logistics_window.raise_();self.logistics_window.activateWindow()

    def show_experiments(self):
        from .logistics_ui import ExperimentDialog
        dialog=ExperimentDialog(self);dialog.exec()

    def show_landscape(self):
        from .landscape_ui import LandscapeDialog
        self.running=False;self.credit=0;self.play.setText('▶ Start')
        dialog=LandscapeDialog(self);dialog.exec()

    def commit_landscape(self):
        if self.complex_project is not None and self.complex_project.shared_site is not None:
            self.complex_project.persist_window();return
        s=self.sim
        if not all(d.grounded and d.cargo==0 for d in s.drones):
            raise ValueError('Сначала посадите флот и выгрузите грунт через «Кисть / план → Посадить флот»')
        if s.logistics is not None:
            w=s.logistics;e=w.excavator
            if w.group is not None or w.active or any(c.soil_kg>0 for c in w.containers) or e is not None and (e.bucket_kg>0 or e.active_container is not None):
                raise ValueError('Завершите переносы и выгрузите контейнеры командой «Посадить флот»')
        if self.landscape is not None and self.config.georeference:
            self.landscape.commit_window(self.config.georeference['origin_m'],s.terrain.height,cell_size=s.terrain.cell,soil=s.terrain.soil)
            if self.landscape.channel_project:
                from .channel import ChannelProject
                project=ChannelProject.from_data(self.landscape.channel_project);project.record(s.report())
                self.landscape.channel_project=project.to_data()

    def edit_fleet(self):
        previous=self.running;self.running=False;self.credit=0
        dialog=FleetDialog(self.sim.pending_equipment or self.config,self)
        if dialog.exec()==QtWidgets.QDialog.DialogCode.Accepted:
            try:
                if dialog.start_new_run:self.replace_simulation(dialog.result_config)
                else:self.apply_equipment(dialog.result_config,previous)
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
        self.base_batteries.setValue(sum(b.quantity for b in self.config.base.equipment.batteries))
        first=next(iter(self.config.base.equipment.batteries),None)
        capacity,mass=(first.capacity_kwh,first.mass_kg) if first else (5.,50.)
        self.battery_type_info.setText(f'Добавление: {capacity:g} кВт·ч / {mass:g} кг, пустые. Другие типы — «Оборудование».')
        self.update_solar_preview()
        while len(self.drone_labels)<len(self.sim.drones):
            label=pg.TextItem(color=self.palette[len(self.drone_labels)],anchor=(0,1));self.map.addItem(label);self.drone_labels.append(label)
        for index,label in enumerate(self.drone_labels):label.setVisible(index<len(self.sim.drones))
        self.fleet_table.setRowCount(len(self.sim.drones))

    def apply_panel_area(self):
        try:
            self.sim.set_panel_area(self.panels.value())
            self.sync_equipment();self.refresh(True)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self,'Площадь панелей',str(error))

    def apply_base_battery_count(self):
        try:
            self.sim.set_base_battery_count(self.base_batteries.value())
            self.sync_equipment();self.refresh(True)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self,'Аккумуляторы базы',str(error))

    def session_ui(self):
        return dict(speed=self.speed.currentData(),view_range=self.view_box.viewRange(),
                    brush_segments=self.plan_editor.segments,brush_dirty=self.plan_editor.dirty,
                    profile_direction=self.profile_direction.currentData(),profile_segment=self.profile_segment.currentData(),
                    profile_position=self.profile_position.value(),equipment_resume=self._equipment_resume,
                    render_mode=self.render_mode.currentData() if hasattr(self,'render_mode') else 'auto',
                    soil_view=self.soil_view.isChecked(),power_request=self.power_request_state,
                    complex_templates=self.template_library)

    def restore_session_ui(self, ui):
        self.power_request_state=ui.get('power_request')
        self.template_library=self.complex_project.templates if self.complex_project is not None else list(ui.get('complex_templates',[]))
        self.speed.setCurrentIndex(max(0,self.speed.findData(ui.get('speed',50))))
        if hasattr(self,'render_mode'):self.render_mode.setCurrentIndex(max(0,self.render_mode.findData(ui.get('render_mode','auto'))))
        self.soil_view.setChecked(bool(ui.get('soil_view',False)))
        if ui.get('view_range') is not None:self.view_box.setRange(xRange=ui['view_range'][0],yRange=ui['view_range'][1],padding=0)
        self.plan_editor.segments=deepcopy(ui.get('brush_segments',self.plan_editor.segments))
        self.plan_editor.dirty=bool(ui.get('brush_dirty',False));self.plan_editor.rebuild()
        for combo,key in ((self.profile_direction,'profile_direction'),(self.profile_segment,'profile_segment')):
            if key in ui:combo.setCurrentIndex(max(0,combo.findData(ui[key])))
        self.profile_position.setValue(ui.get('profile_position',50));self._equipment_resume=ui.get('equipment_resume')

    def _after_file_action(self, running):
        self.running=running;self.credit=0.;self.last_wall=time.perf_counter();self.reset_performance()
        self.play.setText('⏸ Pause' if running else '▶ Start')
        self.timer.start(0 if running else self.config.display_interval_ms)

    def save_session(self, path=None, force_dialog=False):
        from .checkpoint import save_state
        previous=self.running;self.running=False
        try:
            if not isinstance(path,(str,Path)):
                path=self._session_path if self._session_path and not force_dialog else QtWidgets.QFileDialog.getSaveFileName(
                    self,'Сохранить текущую симуляцию',str(self._session_path or 'simulation.atsstate'),'Состояние ATS (*.atsstate)')[0]
            if not path:return
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:save_state(path,self.sim,self.landscape,self.session_ui(),complex_project=self.complex_project)
            finally:QtWidgets.QApplication.restoreOverrideCursor()
            self._session_path=Path(path);self.statusBar().showMessage(f'Симуляция сохранена: {path}',7000)
        except (ValueError,OSError) as error:QtWidgets.QMessageBox.warning(self,'Сохранение симуляции',str(error))
        finally:self._after_file_action(previous)

    def load_session(self, path=None):
        from .checkpoint import load_state
        previous=self.running;self.running=False;loaded=False
        try:
            if not isinstance(path,(str,Path)):
                path=QtWidgets.QFileDialog.getOpenFileName(self,'Загрузить симуляцию','','Состояние ATS (*.atsstate)')[0]
            if not path:return
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:session=load_state(path)
            finally:QtWidgets.QApplication.restoreOverrideCursor()
            self.install_simulation(session.simulation,session.landscape,session.complex_project);self.restore_session_ui(session.ui)
            self._session_path=Path(path);loaded=True
            self.refresh(True);self.statusBar().showMessage(f'Загружено: {path}. Симуляция на паузе.',10000)
        except (ValueError,OSError) as error:QtWidgets.QMessageBox.warning(self,'Загрузка симуляции',str(error))
        finally:self._after_file_action(False if loaded else previous)

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

    def show_power_design(self):
        from .power_ui import PowerDesignDialog
        if hasattr(self,'power_window'):self.power_window.close()
        self.power_window=PowerDesignDialog(self);self.power_window.show()

    def show_base_layout(self):
        from .base_layout_ui import BaseLayoutDialog
        if hasattr(self,'base_layout_window'):self.base_layout_window.close()
        self.base_layout_window=BaseLayoutDialog(self);self.base_layout_window.show()

    def show_complexes(self):
        from .complexes import ComplexProject,ComplexTemplate
        from .complexes_ui import ComplexesWindow
        if self.complex_project is None:
            self.complex_project=ComplexProject(self.sim,self.landscape)
            self.complex_project.templates.extend(self.template_library)
        if not self.complex_project.templates:self.complex_project.templates.append(ComplexTemplate.from_simulation(self.sim,'Текущая комплектация'))
        self.template_library=self.complex_project.templates
        if not hasattr(self,'complex_window') or self.complex_window.project is not self.complex_project:
            self.complex_window=ComplexesWindow(self)
        self.complex_window.timer.start(700);self.complex_window.refresh();self.complex_window.show();self.complex_window.raise_()

    def select_complex(self,id):
        project=self.complex_project
        if project is None:return
        previous=self.running;sim=project.select(id)
        self.install_simulation(sim,project.landscape,project);self._after_file_action(previous)
        self.statusBar().showMessage(f'Открыт {project.active.name}. Start/Pause управляет всеми комплексами.',7000)

    def model_time(self):return self.complex_project.time if self.complex_project is not None else self.sim.time

    def apply_preset(self):
        length, width, depth = PRESETS[self.preset.currentData()]
        self.length.setValue(length)
        self.width.setValue(width)
        self.depth.setValue(depth)

    def change_dust(self):
        c = self.sim.config.dust
        c.wind_x, c.wind_y, c.strength = self.wind_x.value(), self.wind_y.value(), self.dust_strength.value()

    def sync_wind_controls(self):
        c=self.sim.config.dust
        for widget,value in ((self.wind_x,c.wind_x),(self.wind_y,c.wind_y),(self.dust_strength,c.strength)):
            widget.blockSignals(True)
            widget.setRange(min(widget.minimum(),value),max(widget.maximum(),value));widget.setValue(value)
            widget.blockSignals(False)

    def update_solar_preview(self):
        from .solar import sunlight
        sun = sunlight(self.latitude.value(), self.calendar.date().dayOfYear(), 12, self.config.base.peak_irradiance)
        self.season.blockSignals(True)
        self.season.setCurrentIndex(self.season.findData(self.calendar.date().dayOfYear()))
        self.season.blockSignals(False)
        power = sun.irradiance*self.panels.value()*self.config.base.panel_efficiency/1000
        self.panel_info.setText(f"Ясное небо, горизонтальные панели. Полдень: {sun.irradiance:.0f} Вт/м² → {power:.2f} кВт; световой день {sun.daylight_hours:.1f} ч.\nВремя — местное солнечное; календарь 365 дней.")
        self.update_energy_status()

    def update_energy_status(self):
        from .winter_power import design_status
        state,message=design_status(self.config,self.latitude.value())
        if abs(self.panels.value()-self.config.base.panel_area)>1e-6:
            state='stale';message='Площадь в предпросмотре изменена · требуется перепроверка энергетики'
        self.energy_status.setText(f'<a href="power">{message}</a>')
        self.energy_status.setProperty('designState',state)
        self.energy_status.setToolTip('Открыть расчёт. Широта и ручные настройки не заменяют оборудование автоматически. Зимний статус относится к расчётному профилю нагрузки, не к запасу заряда в эту секунду.')

    def replace_simulation(self, config):
        sim = Simulation(config)  # validate before discarding the current state
        region=self.landscape
        if region is not None and (not config.georeference or config.georeference.get('source')!=region.raster.metadata):region=None
        self.install_simulation(sim,region)

    def install_simulation(self, sim, landscape=None,complex_project=None):
        if hasattr(self,'complex_window') and self.complex_window.project is not complex_project:
            self.complex_window.close();self.complex_window.project=None
        self.complex_project=complex_project
        if complex_project is not None:self.template_library=complex_project.templates
        config=sim.config;self.landscape=landscape;self._session_path=None
        self.running = False
        self._equipment_resume = None
        self.play.setText("▶ Start")
        self.credit = 0
        self.config = config
        self.sim = sim
        self.map_size.setValue(config.terrain.n*config.terrain.cell_size)
        self.update_map_size_controls()
        self.cx.setValue(config.trench.x);self.cy.setValue(config.trench.y)
        self.length.setValue(config.trench.length);self.width.setValue(config.trench.width)
        self.depth.setValue(config.trench.depth);self.angle.setValue(config.trench.angle_deg or 0)
        self.seed.setValue(config.seed);self.roughness.setValue(config.terrain.roughness)
        self.start_hour.setValue(config.start_hour)
        self.calendar.setDate(QtCore.QDate(2025,1,1).addDays(config.base.start_day_of_year-1))
        self.latitude.setValue(config.base.latitude_deg)
        self.latitude.setEnabled(not bool(config.georeference and config.georeference['source'].get('crs')=='EPSG:4326'))
        self.sync_equipment()
        self.plan_editor.segments=deepcopy(config.plan.segments) if config.plan is not None else []
        self.plan_editor.dirty=False;self.plan_editor.rebuild()
        self.plan_editor.default_depth=config.trench.depth
        self.plan_editor.configure_elevation(config)
        self.profile.setLabel('left','Отметка MSL' if config.georeference else 'Высота',units='m')
        self.last_event = 0
        self.log.clear()
        self.setup_map()
        self.sync_wind_controls()
        self.refresh(True)

    def new_scenario(self):
        config = deepcopy(self.config)
        config.plan=None;config.initial_height=None
        config.initial_soil=None
        config.reference_height=None;config.georeference=None;config.terrain.floor=-3.
        q = config.trench
        q.length, q.width, q.depth = self.length.value(), self.width.value(), self.depth.value()
        q.x, q.y, q.angle_deg = self.cx.value(), self.cy.value(), self.angle.value() or None
        config.seed = self.seed.value()
        config.start_hour = self.start_hour.value()
        config.terrain.roughness = self.roughness.value()
        config.base.latitude_deg = self.latitude.value()
        config.base.start_day_of_year = self.calendar.date().dayOfYear()
        try:
            from .maps import with_map_size
            from .equipment import set_panel_area
            set_panel_area(config.base.equipment,self.panels.value(),config.base.panel_efficiency)
            config=with_map_size(config,self.map_size.value())
            self.replace_simulation(config)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Неверный сценарий", str(error))

    def reset(self):
        self.replace_simulation(deepcopy(self.config))

    def toggle_play(self):
        if not (self.complex_project.running if self.complex_project is not None else self.sim.state==MissionState.RUNNING):
            return
        self.running = not self.running
        if self.running:
            if self.sim.pending_equipment is None:self.sim.landing_requested=False
            self.plan_editor.enabled.setChecked(False)
        self.credit = 0
        self.last_wall = time.perf_counter()
        self.reset_performance()
        self.timer.setInterval(0 if self.running else self.config.display_interval_ms)
        self.play.setText("⏸ Pause" if self.running else "▶ Start")

    def reset_performance(self):
        self.performance_wall=time.perf_counter();self.performance_sim=self.model_time()
        self.performance_renders=0;self.actual_speed=0.;self.actual_render_hz=0.

    def render_changed(self):
        off=self.render_mode.currentData()=='off'
        self.map_frame.setVisible(not off);self.profile.setVisible(not off)
        self.last_revision=-1;self._last_view=None;self.last_render_wall=-float('inf')
        self._render_cost=.02;self.reset_performance()
        self.statusBar().showMessage('Карта и профиль приостановлены; расчёт и телеметрия продолжаются.' if off else 'Отображение карты включено.',7000)
        self.refresh(True)

    def speed_changed(self):
        self.credit=0.;self.last_wall=time.perf_counter();self.reset_performance()
        if hasattr(self,'timer'):self.timer.setInterval(0 if self.running else self.config.display_interval_ms)

    def on_frame(self):
        was_running=self.running
        now = time.perf_counter()
        elapsed = min(now-self.last_wall, 0.25)
        self.last_wall = now
        if self.running:
            self.credit = min(self.credit+elapsed*self.speed.currentData(), self.config.max_frame_steps)
            # Long but bounded compute slices run independently of rendering.
            # All edits remain on the same Qt thread, between physical steps.
            budget=max(self.config.frame_budget_ms,50 if self.speed.currentData()>=200 else 0)
            deadline = now+budget/1000
            steps = 0
            while self.credit >= 1 and steps < self.config.max_frame_steps and time.perf_counter() < deadline:
                if self.complex_project is not None:self.complex_project.step()
                else:self.sim.step()
                if getattr(self,'_display_origin',None)!=(self.sim.config.georeference or {}).get('origin_m'):
                    running=self.running;path=self._session_path
                    self.install_simulation(self.sim,self.landscape,self.complex_project)
                    self._session_path=path;self.running=running;self.play.setText('⏸ Pause' if running else '▶ Start')
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
                if self.sim.landing_requested and logistics is not None:robot_ready=robot_ready and all(c.soil_kg==0 for c in logistics.containers)
                shared=self.complex_project is not None and self.complex_project.shared_site is not None
                grounded=(all(d.grounded and abs(d.cargo)<=1e-8 for u in self.complex_project.units for d in u.simulation.drones) if shared else all(d.grounded and abs(d.cargo)<=1e-8 for d in self.sim.drones))
                if shared:
                    robot_ready=all(u.simulation.logistics is None or (not u.simulation.logistics.active and not u.simulation.logistics.group and
                        all(c.soil_kg==0 for c in u.simulation.logistics.containers) and (u.simulation.logistics.excavator is None or u.simulation.logistics.excavator.bucket_kg==0)) for u in self.complex_project.units)
                requested=any(u.simulation.landing_requested for u in self.complex_project.units) if shared else self.sim.landing_requested
                if requested and robot_ready and grounded and not (self.complex_project is not None and self.complex_project.transition is not None):
                    self.running=False;self.credit=0;self.play.setText("▶ Start")
                    self.plan_editor.info.setText("Флот приземлён. Можно применить нарисованный план.")
                    break
                done=not self.complex_project.running if self.complex_project is not None else self.sim.state!=MissionState.RUNNING
                if done:
                    if self.complex_project is None and self.sim.state==MissionState.COMPLETED and self.landscape is not None and self.config.georeference.get('channel'):
                        self.commit_landscape()
                    self.running = False
                    completed=not self.complex_project.blocked if self.complex_project is not None else self.sim.state==MissionState.COMPLETED
                    self.play.setText('Завершено' if completed else 'Заблокировано')
                    break
        stamp=time.perf_counter()
        interval=(.2 if self.speed.currentData()>=1000 else .1 if self.speed.currentData()>=200
                  else self.config.display_interval_ms/1000) if self.running else .25
        if self.running and self.render_mode.currentData()=='auto' and self.speed.currentData()>=200:
            interval=max(interval,self._render_cost*9)
        elif self.running and self.render_mode.currentData()=='off':interval=.5
        if stamp-self.last_render_wall>=interval or was_running and not self.running:
            begin=time.perf_counter();self.refresh();self.last_render_wall=time.perf_counter();self.performance_renders+=1
            self._render_cost=.75*self._render_cost+.25*(self.last_render_wall-begin)
        stamp=time.perf_counter();span=stamp-self.performance_wall
        if span>=1.:
            sample=max(0.,(self.model_time()-self.performance_sim)/span)
            self.actual_speed=sample if self.actual_speed==0 else .5*self.actual_speed+.5*sample
            self.actual_render_hz=self.performance_renders/span
            self.speed_status.setText(f'Цель: {self.speed.currentData()}× · факт: {self.actual_speed:.1f}× · обновление: {self.actual_render_hz:.1f}/с')
            self.performance_wall=stamp;self.performance_sim=self.model_time();self.performance_renders=0
        delay=(min(self.config.display_interval_ms,max(0,math.ceil((1-self.credit)/self.speed.currentData()*1000)))
               if self.running else self.config.display_interval_ms)
        self.timer.setInterval(delay)

    def refresh(self, force=False):
        moving=self.complex_project is not None and self.complex_project.transition is not None
        self.transition_status.setVisible(moving)
        if moving:
            move=self.complex_project.transition
            progress=f'{move.actor_index}/{len(move.actors)} наземных маршрутов · ' if move.actors else ''
            self.transition_status.setText(f'Переезд всей группы\n{move.reason}\n{progress}{move.project.time-move.started_at:.0f} с')
        s, t, d, b = self.sim, self.sim.terrain, self.sim.drone, self.sim.base
        channel=(self.config.georeference or {}).get('channel')
        self.channel_status.setVisible(channel is not None)
        if channel:
            self.channel_status.setText(f"Канал: дно {channel['width_m']:g} м × −{channel['depth_m']:g} м MSL\nУчасток {channel['section']+1}/{channel['sections']} · маршрут {channel['length_m']/1000:.2f} км · береговой отступ {channel['shore_offset_m']:g} м\nКарта {t.cell*t.height.shape[0]:g} м · ячейка {t.cell:g} м")
        if t.front is not None:
            front=t.front;front.update();indices=front.active_indices
            if len(indices):
                xx,yy=t.x.ravel()[indices],t.y.ravel()[indices]
                x0,x1,y0,y1=float(xx.min()-t.cell/2),float(xx.max()+t.cell/2),float(yy.min()-t.cell/2),float(yy.max()+t.cell/2)
                self.workfront_outline.setData([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0])
            else:self.workfront_outline.clear()
            if channel:
                state=s.base_move_status
                self.channel_status.setText(self.channel_status.text()+f"\nФронт {front.position/1000:.3f} км · рабочая зона {front.settings['front_length_m']:g} м · база {state}")
        else:self.workfront_outline.clear()
        if self.config.georeference:
            self.latitude.blockSignals(True);self.latitude.setValue(b.config.latitude_deg);self.latitude.blockSignals(False)
        if self.render_mode.currentData()=='off':
            self.refresh_telemetry(s.report(full=False));return
        self.soil_view.setEnabled(t.soil is not None)
        self.soil_legend.setVisible(t.soil is not None and self.soil_view.isChecked())
        if t.soil is not None:
            self.soil_legend.setText('Условные грунты · '+ ' · '.join(f'<span style="color:{m.color}">■ {m.name}: {m.density:g} кг/м³</span>' for m in t.soil.profile.materials))
        now = time.perf_counter()
        view_range=self.view_box.viewRange();view_key=tuple(tuple(p) for p in view_range)
        if force or (t.revision != self.last_revision or view_key!=self._last_view) and now-self.last_terrain_wall >= self.config.terrain_display_seconds:
            if self.render_mode.currentData()=='auto' and self.running:
                height,rect,stride=terrain_view(t.height,t.cell,view_range)
            else:height,rect,stride=t.height,(0,0,t.cell*t.height.shape[0],t.cell*t.height.shape[0]),1
            if t.soil is not None and self.soil_view.isChecked():
                x0=round(rect[0]/t.cell);y0=round(rect[1]/t.cell);n=t.height.shape[0]
                yy,xx=np.indices(height.shape);ids=np.minimum(y0+yy*stride+stride//2,n-1)*n+np.minimum(x0+xx*stride+stride//2,n-1)
                materials=t.soil.surface_ids(ids.ravel()).reshape(height.shape)
                colors=np.array([[int(m.color[i:i+2],16) for i in (1,3,5)] for m in t.soil.profile.materials])
                gy,gx=np.gradient(height,t.cell*stride);shade=(.9+.25*(-gx+gy)/np.sqrt(1+gx*gx+gy*gy)).clip(.55,1.1)
                rgb=(colors[materials]*shade[...,None]).clip(0,255).astype(np.uint8)
            else:rgb=terrain_rgb(height,t.cell*stride,self.config.trench.depth,max(.5,float(height.max())))
            self.image.setImage(rgb, autoLevels=False)
            self.image.setRect(QtCore.QRectF(*rect))
            self.update_profile()
            self.last_revision = t.revision
            self.last_terrain_wall = now
            self._last_view=view_key
        concentration = 1-np.exp(-s.dust.field*self.config.dust.concentration_scale)
        cloud = np.zeros((*concentration.shape, 4), dtype=np.uint8)
        cloud[..., :3] = [211, 187, 146]
        cloud[..., 3] = (concentration*170).clip(0, 170).astype(np.uint8)
        self.dust_image.setImage(cloud, autoLevels=False)
        self.dust_image.setRect(QtCore.QRectF(0, 0, s.dust.side, s.dust.side))
        self.dust_image.setVisible(self.show_dust.isChecked())
        self.base_marker.setData([b.position[0]], [b.position[1]])
        x,y=b.position[:2];r=b.config.radius
        if b.config.layout is None:self.base_outline.setData([x-r,x+r,x+r,x-r,x-r],[y-r,y-r,y+r,y+r,y-r])
        else:
            from .base_layout import platform_polygons
            points=[]
            for polygon in platform_polygons(b.layout_spec,b.position):points.extend([*polygon,(np.nan,np.nan)])
            points=np.asarray(points);self.base_outline.setData(points[:,0],points[:,1],connect='finite')
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
            waiting='\n'+s.logistics.idle_reason(o) if s.logistics is not None and o.grounded else ''
            role=(' · курьер' if s.logistics.courier_id==o.identifier else ' · копатель' if o.attachment=='GRABBER' else '') if s.logistics is not None else ''
            label.setText(f"Д{o.serial}{role} · {o.cargo:.0f} кг · z {o.position[2]:.1f} м{operation}{laser}{waiting}")
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
        for label in [*self.drone_labels,self.base_label,self.excavator_label]:label.setVisible(self.labels_view.isChecked())
        from .shared_ui import refresh
        refresh(self)
        self.refresh_telemetry(report)

    def refresh_telemetry(self, report):
        self.update_energy_status()
        s,t,d,b=self.sim,self.sim.terrain,self.sim.drone,self.sim.base
        spec=report['base_specification']
        body=' × '.join(f'{value:.2f}' for value in spec['body_dimensions_m'])
        landing=' × '.join(f'{value:.2f}' for value in spec['landing_zone_dimensions_m'])
        field=spec['solar_field_dimensions_m']
        field_text=(' × '.join(f'{value:.2f}' for value in field)+' м' if field else
                    'компактная установка' if b.config.panel_area<=20 else 'не размещено / сложено')
        self.base_mass.setValue(report['base_mass_kg'])
        self.base_dimensions.setText(f"Паспорт базы\nКорпус: {body} м\nПосадочная зона: {landing} м\nПоле панелей: {field_text}\nС пристыкованными дронами: {spec['gross_mass_kg']:.1f} кг")
        if spec.get('layout'):
            self.base_dimensions.setText(self.base_dimensions.text()+f"\nПлатформ: {spec['layout']['platform_count']} · давление {spec['ground_support']['max_pressure_kpa']:.1f} кПа")
        self.base_label.setText(f"База · {spec['gross_mass_kg']:.0f} кг с дронами\nКорпус {body} м")
        self.base_label.setToolTip(f"Оборудование: {spec['installed_equipment_mass_kg']:.1f} кг; запасы: {spec['stored_items_mass_kg']:.1f} кг; пристыкованные дроны: {spec['docked_drones_mass_kg']:.1f} кг.\nПоле панелей: {field_text}.")
        self.fleet_table.setUpdatesEnabled(False)
        for row,o in enumerate(report['drones']):
            values=[f"{o['serial']} · {o['name']}",o['state'],f"{o['battery_percent']:.0f}%",f"{o['cargo_kg']:.1f}",str(o['trips']),f"{o['excavated_mass_kg']:.0f} / {o['filled_mass_kg']:.0f}",f"{o['consumed_kwh']:.3f}",f"{o['waiting_seconds']/60:.1f}",f"{o['charging_seconds']/60:.1f}",str(o['traffic_yields']),f"{o['last24_volume_m3']:.4f}",f"{o['laser_power_w']/1000:.2f}"]
            for col,value in enumerate(values):self.fleet_table.setItem(row,col,QtWidgets.QTableWidgetItem(value))
            if 'task' in o:self.fleet_table.item(row,1).setToolTip(f"{o['task']} · {o['attachment']}\n{o['wait_reason']}\nБатареи: {', '.join(o['battery_ids'])}\nЗамен комплектов: {o['battery_swaps']}; до завершения замены: {o['battery_swap_remaining_seconds']:.0f} с\nДо завершения смены инструмента: {o['tool_change_remaining_seconds']:.0f} с · возврат по отмене: {'да' if o['recall_requested'] else 'нет'}\nГрузовой лимит: {o['transport_payload_kg']:.1f} кг")
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
        project=self.complex_project
        if project is not None and project.shared_site is not None:
            site=project.shared_site;percent=max(0.,min(100.,100*(1-site.core.remaining/max(site.core.initial_excavation,1e-12))))
            self.progress.setValue(round(percent*10));self.progress.setFormat(f'Общий план {percent:.1f}%')
            self.progress.setToolTip(f'Допуск общего плана: {site.completion_volume:.3f} м³')
            state='Группа работает' if project.running and self.running else 'Группа на паузе' if project.running else 'Группа остановлена' if project.blocked else 'Общий план завершён'
            self.status.setText(f'<b>{state}</b><br>Комплексов: {len(project.units)} · выбран {project.active_id[8:]}')
        rows = [
            ("Прошло с запуска", report["elapsed_display"]),
            ("Дата / солнечное время", f"{s.date_text} · {s.clock_text}"),
            ("Флот / очередь",f"{len(s.drones)} / {len(s.queue)}"), ("База", b.state.value),
            ("Зарядные слоты",f"{sum(v is not None for v in s.slots)} / {len(s.slots)} · {b.config.charge_power/1000:.1f} кВт всего"),
            ("Проект / остаток", f"{report['initial_excavation_m3']:.3f} / {report['remaining_m3']:.3f} м³"),
            ("Выемка / засыпка осталось",f"{report['remaining_cut_m3']:.3f} / {report['remaining_fill_m3']:.3f} м³"),
            ("Проектный откос",'Авто · минимальная устойчивая ширина' if s.config.trench.angle_deg is None else f'{s.config.trench.angle_deg:g}°'),
            ("Устойчивость грунта",'По слоям и составу насыпи' if s.config.slope_model=='materials' and t.soil else 'Однородный грунт · физический угол' if s.config.slope_model=='materials' else 'Прежний общий угол · сохранённый расчёт'),
            ("Вынуто / рейсы", f"{report['excavated_m3']:.3f} м³ / {s.trips}"),
            ("Общий груз", f"{report['cargo_kg']:.1f} кг"),
            ("База / панели", f"{b.config.mass:.0f} кг / {b.config.panel_area:.1f} м²"),
            ("Корпус, X × Y × В",body+' м'),
            ("Посадочная зона",landing+' м'),
            ("Поле панелей",field_text),
            ("Масса с дронами",f"{spec['gross_mass_kg']:.1f} кг"),
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
        if s.logistics is not None:
            courier=report['courier'];robot=report.get('excavator')
            if robot:
                rows.insert(3,('Крот',f"{robot['state']} · {robot['energy_kwh']:.2f}/{robot['capacity_kwh']:.2f} кВт·ч"))
                rows.insert(4,('Добыто Кротом',f"{robot['excavated_kg']:.1f} кг / {robot['excavated_m3']:.3f} м³"))
            rows.insert(4,('Партия / курьер',f"{courier['filled_containers']}/{courier['threshold']} · "+
                ((f"Д{courier['drone_serial']} · {courier['reason']}" if courier['drone_serial'] is not None
                  else 'ожидание готового дрона') if courier['active'] else 'все копают')))
            rows.extend([('Энергетика базы',report['base_energy_status']),
                ('Генерация / нагрузка',f"{report['solar_power_w']/1000:.2f} / {report['base_load_power_w']/1000:.2f} кВт"),
                ('Поток в батарею базы',f"{report['base_battery_power_w']/1000:+.2f} кВт · плюс: заряд, минус: разряд"),
                ('Развёрнуто панелей',f"{report['solar_generating_area_m2']:.1f} / {report['panel_area_m2']:.1f} м²")])
        self.telemetry.setText("<table width='100%' cellspacing='4'>"+"".join(f"<tr><td width='48%' style='color:#94abbf'>{label}</td><td width='52%'>{value}</td></tr>" for label, value in rows)+"</table>")
        self.compact_telemetry.setText(f"{s.clock_text}\nБаза: {b.battery/KWH:.1f} / {b.config.battery/KWH:.1f} кВт·ч\n"
            f"Панели: {b.config.panel_area:g} м²\nВыемка: {report['excavated_m3']:.2f} м³\nМасса базы: {b.config.mass:.0f} кг")
        if project is not None and project.shared_site is not None:
            volume=sum(u.simulation.excavated_volume if u.simulation.terrain.soil is not None else u.simulation.excavated_mass/u.simulation.config.terrain.density for u in project.units)
            self.compact_telemetry.setText(self.compact_telemetry.text()+f'\nОбщая выемка: {volume:.2f} м³')
        for serial, timestamp, message in s.events:
            if serial > self.last_event:
                self.log.appendPlainText(f"[{timestamp/60:6.1f} мин] {message}")
                self.last_event = serial
        self.play.setEnabled(self.complex_project.running if self.complex_project is not None else s.state==MissionState.RUNNING)


def launch(config,landscape=None,session=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("ATS")
    window = MainWindow(config,session=session)
    if session is None:window.landscape=landscape
    window.show()
    return app.exec()
