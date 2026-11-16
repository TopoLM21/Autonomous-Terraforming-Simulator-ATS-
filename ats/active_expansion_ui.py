"""Explicit growth of a live regional window without resetting its mission."""
from copy import deepcopy
import math
import time
import numpy as np
from PySide6 import QtWidgets


def expansion_proposal(simulation, landscape, side_m):
    """Describe a containing, cell-aligned square; never read or mutate terrain."""
    from .maps import MAX_ACTIVE_CELLS,MAX_ACTIVE_SIDE_M
    terrain=getattr(simulation.terrain,'core',simulation.terrain)
    cell=float(terrain.cell);old_side=terrain.height.shape[0]*cell
    geo=simulation.config.georeference
    if landscape is None or not geo or 'origin_m' not in geo:
        raise ValueError('Для подгрузки нужен открытый региональный ландшафт; новая карта не заменяет текущую миссию')
    origin=np.asarray(geo['origin_m'],dtype=float)
    extent=np.asarray(landscape.raster.extent_m,dtype=float)
    side=float(side_m)
    limit=min(float(MAX_ACTIVE_SIDE_M),MAX_ACTIVE_CELLS*cell,float(extent.min()))
    limit=math.floor((limit+1e-8)/cell)*cell
    if not math.isfinite(side) or side<=old_side+1e-8:
        raise ValueError('Выберите размер больше текущего рабочего окна')
    if side>limit+1e-8:
        raise ValueError(f'При ячейке {cell:g} м доступно не более {limit:g} м: разрешение не будет уменьшено')
    n=round(side/cell)
    if abs(n*cell-side)>1e-7:
        raise ValueError('Размер должен быть кратен текущей рабочей ячейке')
    if np.any(origin<0) or np.any(origin+old_side>extent+1e-7):
        raise ValueError('Текущее окно не помещается в исходный DEM')
    # Anchor alignment to the old grid, including nonzero imported origins.
    lower=np.maximum(0.,origin+old_side-side)
    upper=np.minimum(origin,extent-side)
    first=np.ceil((lower-origin)/cell-1e-9)
    last=np.floor((upper-origin)/cell+1e-9)
    if np.any(first>last):raise ValueError('Не удалось совместить соседние области с текущей сеткой и границами DEM')
    steps=np.clip(np.rint((old_side-side)/(2*cell)),first,last)
    target=origin+steps*cell
    # Include the dense relaxation pairing tables, but not sparse soil tiles or
    # Python lists. This is an estimate of resident NumPy storage, not a RAM cap.
    seen=set()
    def arrays(value):
        if isinstance(value,np.ndarray):
            if id(value) in seen:return 0
            seen.add(id(value));return value.nbytes
        if isinstance(value,(list,tuple)):
            return sum(arrays(v) for v in value)
        return 0
    dense=sum(arrays(v) for v in vars(terrain).values())
    estimate=max(64.,dense/terrain.height.size)*n*n
    return dict(old_origin_m=origin.tolist(),origin_m=target.tolist(),old_side_m=old_side,
                side_m=side,cell_m=cell,n=n,max_side_m=limit,
                added_area_m2=side*side-old_side*old_side,dense_bytes=int(estimate),
                offset_m=(origin-target).tolist())


def _translated_ui(ui,offset):
    result=deepcopy(ui)
    if result.get('view_range') is not None:
        result['view_range']=[[float(v)+offset[i] for v in values]
                              for i,values in enumerate(result['view_range'])]
    for segment in result.get('brush_segments',[]):
        points=segment.get('points') if isinstance(segment,dict) else segment.points
        shifted=[[float(p[0])+offset[0],float(p[1])+offset[1]] for p in points]
        if isinstance(segment,dict):segment['points']=shifted
        else:segment.points=shifted
    return result


class ActiveExpansionDialog(QtWidgets.QDialog):
    """A user command applied synchronously between simulation steps."""
    def __init__(self,host):
        super().__init__(host)
        self.host=host;self.offset_m=(0.,0.);self.proposal=None;self._landing_before={}
        self.setWindowTitle('Подгрузить соседние области')
        layout=QtWidgets.QVBoxLayout(self)
        intro=QtWidgets.QLabel('Расширяется общая расчётная область, а не только изображение. Соседний грунт читается из того же ландшафта; техника, изменения грунта, энергия и модельное время сохраняются. Ячейка не укрупняется.')
        intro.setWordWrap(True);layout.addWidget(intro)
        terrain=getattr(host.sim.terrain,'core',host.sim.terrain)
        self.current_side=terrain.height.shape[0]*terrain.cell
        self.current=QtWidgets.QLabel(f'Сейчас: {self.current_side:g} × {self.current_side:g} м · ячейка {terrain.cell:g} м')
        layout.addWidget(self.current)
        self.side=QtWidgets.QDoubleSpinBox();self.side.setDecimals(6);self.side.setSuffix(' м');self.side.setSingleStep(terrain.cell)
        from .maps import MAX_ACTIVE_CELLS,MAX_ACTIVE_SIDE_M
        limit=min(MAX_ACTIVE_SIDE_M,MAX_ACTIVE_CELLS*terrain.cell)
        if getattr(host,'landscape',None) is not None:limit=min(limit,*host.landscape.raster.extent_m)
        limit=math.floor((limit+1e-8)/terrain.cell)*terrain.cell
        self.side.setRange(min(self.current_side+terrain.cell,limit),limit)
        preferred=min(limit,math.ceil(self.current_side*1.5/terrain.cell)*terrain.cell)
        self.side.setValue(preferred)
        form=QtWidgets.QFormLayout();form.addRow('Новая сторона общей карты',self.side);layout.addLayout(form)
        self.preview=QtWidgets.QLabel();self.preview.setWordWrap(True);layout.addWidget(self.preview)
        self.message=QtWidgets.QLabel();self.message.setWordWrap(True);layout.addWidget(self.message)
        note=QtWidgets.QLabel('При изменении начала координат сначала завершите перевозки и посадите дроны; Кроты и обычная техника должны остановиться без груза. Подгрузка сама не перемещает машины и не создаёт технику. Для обычной бригады завершите её безопасный возврат на стоянку.')
        note.setWordWrap(True);layout.addWidget(note)
        actions=QtWidgets.QHBoxLayout();layout.addLayout(actions)
        self.landing_button=QtWidgets.QPushButton('Посадить комплексы');self.landing_button.clicked.connect(self.request_landing);actions.addWidget(self.landing_button)
        self.apply_button=QtWidgets.QPushButton('Подгрузить соседние области');self.apply_button.clicked.connect(self.apply_expansion);actions.addWidget(self.apply_button)
        cancel=QtWidgets.QPushButton('Закрыть');cancel.clicked.connect(self.reject);actions.addWidget(cancel)
        self.side.valueChanged.connect(self.update_preview);self.update_preview()
        from .ui_windows import fit_window
        fit_window(self,660,480)

    def update_preview(self,*_):
        try:
            p=expansion_proposal(self.host.sim,getattr(self.host,'landscape',None),self.side.value())
            self.proposal=p
            x,y=p['origin_m'];self.preview.setText(
                f"Начало в регионе: X={x:g}, Y={y:g} м. Карта {p['side_m']:g} × {p['side_m']:g} м; "
                f"{p['n']} × {p['n']} ячеек по {p['cell_m']:g} м. Добавится {p['added_area_m2']:,.0f} м².\n"
                f"Оценка плотных NumPy-массивов: {p['dense_bytes']/2**20:.1f} МиБ; Python-копии, материалы и временные буферы потребуют дополнительную память.")
            self.apply_button.setEnabled(hasattr(self.host,'install_simulation'))
            if not self.apply_button.isEnabled():self.message.setText('Откройте общую карту с автономным комплексом для расширения текущего окна')
        except ValueError as error:
            self.proposal=None;self.preview.setText(str(error));self.apply_button.setEnabled(False)

    def _units(self):
        project=getattr(self.host,'complex_project',None)
        return [u.simulation for u in project.units] if project is not None else [self.host.sim]

    def _resume(self,running):
        if hasattr(self.host,'_after_file_action'):self.host._after_file_action(running)
        else:
            self.host.running=running
            if hasattr(self.host,'last_wall'):self.host.last_wall=time.perf_counter()

    def request_landing(self):
        try:
            for sim in self._units():
                self._landing_before.setdefault(id(sim),sim.landing_requested)
                sim.request_landing()
            self._resume(True)
            self.message.setText('Посадка запрошена. Время продолжает идти; дождитесь завершения грузовых операций и остановки техники, затем снова нажмите «Подгрузить соседние области».')
        except ValueError as error:self.message.setText(str(error))

    def apply_expansion(self):
        host=self.host;previous=bool(host.running);host.running=False
        project=getattr(host,'complex_project',None);installing_site=False;states=[]
        try:
            p=expansion_proposal(host.sim,getattr(host,'landscape',None),self.side.value())
            if project is None:
                from .complexes import ComplexProject
                project=ComplexProject(host.sim,host.landscape)
                project.templates.extend(getattr(host,'template_library',[]))
            installing_site=project.shared_site is None
            if installing_site:
                states=[(u,vars(u.simulation).copy(),u.simulation.config.terrain.completion_volume,u.committed) for u in project.units]
            path=getattr(host,'_session_path',None)
            ui=host.session_ui() if hasattr(host,'session_ui') else None
            project.enable_shared().enlarge(p['side_m'],origin_m=p['origin_m'],preserve_resolution=True)
        except (ValueError,OSError) as error:
            if installing_site:
                project.shared_site=None
                for unit,state,tolerance,committed in states:
                    vars(unit.simulation).clear();vars(unit.simulation).update(state)
                    unit.simulation.config.terrain.completion_volume=tolerance;unit.committed=committed
            self._resume(previous);self.message.setText(str(error));return
        self.proposal=p;self.offset_m=tuple(p['offset_m'])
        for unit in project.units:
            if self._landing_before.get(id(unit.simulation)) is False:unit.simulation.landing_requested=False
        host.install_simulation(project.active.simulation,project.landscape,project)
        host._session_path=path
        if ui is not None and hasattr(host,'restore_session_ui'):host.restore_session_ui(_translated_ui(ui,self.offset_m))
        self._resume(previous)
        self.accept()
