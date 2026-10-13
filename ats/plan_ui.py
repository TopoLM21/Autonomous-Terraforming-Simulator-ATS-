"""Map brush interaction and compact plan controls; no physical edits here."""
from copy import deepcopy
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore,QtWidgets
from .plan import PlanConfig,PlanSegment


class BrushViewBox(pg.ViewBox):
    strokeFinished = QtCore.Signal(object)
    brushMoved = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.brush_enabled = False
        self.spacing = .125
        self.points = []

    def point(self, scene_position):
        p = self.mapSceneToView(scene_position)
        return [p.x(),p.y()]

    def mouseDragEvent(self,event,axis=None):
        if not self.brush_enabled or event.button()!=QtCore.Qt.MouseButton.LeftButton:
            return super().mouseDragEvent(event,axis)
        event.accept()
        if event.isStart():
            self.points=[self.point(event.buttonDownScenePos())]
        point=self.point(event.scenePos())
        self.brushMoved.emit(point)
        if not self.points or np.linalg.norm(np.asarray(point)-self.points[-1])>=self.spacing:
            self.points.append(point)
        if event.isFinish():
            self.strokeFinished.emit(self.points[:4096]);self.points=[]

    def mouseClickEvent(self,event):
        if self.brush_enabled and event.button()==QtCore.Qt.MouseButton.LeftButton:
            event.accept();self.strokeFinished.emit([self.point(event.scenePos())])
        else:
            super().mouseClickEvent(event)


class PlanEditor(QtWidgets.QWidget):
    planChanged=QtCore.Signal()
    applyRequested=QtCore.Signal()
    landingRequested=QtCore.Signal()
    saveRequested=QtCore.Signal()
    loadRequested=QtCore.Signal()

    def __init__(self,config,parent=None):
        super().__init__(parent)
        self.segments=deepcopy(config.plan.segments) if config.plan is not None else []
        self.dirty=False
        self.default_depth=config.trench.depth
        layout=QtWidgets.QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        tools=QtWidgets.QHBoxLayout()
        self.enabled=QtWidgets.QCheckBox("Кисть (ЛКМ)")
        self.mode=QtWidgets.QComboBox();self.mode.addItem("Выемка","dig");self.mode.addItem("Засыпка","fill")
        self.width=QtWidgets.QDoubleSpinBox();self.width.setRange(config.terrain.cell_size,8);self.width.setSingleStep(.25);self.width.setValue(1);self.width.setSuffix(" м")
        self.height=QtWidgets.QDoubleSpinBox();self.height.setRange(config.terrain.floor+.05,config.terrain.dump_max_height);self.height.setSingleStep(.05);self.height.setValue(-config.trench.depth);self.height.setSuffix(" м")
        for widget in [self.enabled,self.mode,QtWidgets.QLabel("Ширина"),self.width,QtWidgets.QLabel("Высота цели"),self.height]:tools.addWidget(widget)
        self.selection=QtWidgets.QComboBox();tools.addWidget(self.selection,1)
        layout.addLayout(tools)
        actions=QtWidgets.QHBoxLayout()
        for label,handler in [("Удалить мазок",self.remove),("Очистить рисунок",self.clear),("Посадить флот",self.landingRequested.emit),("Применить план",self.applyRequested.emit),("Сохранить…",self.saveRequested.emit),("Открыть…",self.loadRequested.emit)]:
            button=QtWidgets.QPushButton(label)
            def invoke(checked=False,action=handler):
                action()  # discard the bool returned by Signal.emit()
            button.clicked.connect(invoke);actions.addWidget(button)
        layout.addLayout(actions)
        self.info=QtWidgets.QLabel("Кисть задаёт высоту поверхности, не меняя грунт сразу. Для применения флот должен приземлиться. Высота 0 м — исходный уровень; засыпка берёт грунт из отвала или выемки.")
        self.info.setWordWrap(True);layout.addWidget(self.info)
        self.mode.currentIndexChanged.connect(lambda:self.height.setValue(0 if self.mode.currentData()=="fill" else -self.default_depth))
        self.rebuild()

    def rebuild(self):
        self.selection.clear()
        for s in self.segments:self.selection.addItem(f"{s.name}: {'выемка' if s.mode=='dig' else 'засыпка'}, {s.height:+.2f} м")
        self.selection.setCurrentIndex(len(self.segments)-1)

    def add_stroke(self,points):
        if not points:return
        segment=PlanSegment(points,self.mode.currentData(),self.width.value(),self.height.value(),f"Сегмент {len(self.segments)+1}")
        self.segments.append(segment);self.dirty=True;self.rebuild();self.planChanged.emit()

    def remove(self):
        index=self.selection.currentIndex()
        if index>=0:
            self.segments.pop(index);self.dirty=True;self.rebuild();self.planChanged.emit()

    def clear(self):
        self.segments=[];self.dirty=True;self.rebuild();self.planChanged.emit()

    def plan(self):
        return PlanConfig(deepcopy(self.segments))
