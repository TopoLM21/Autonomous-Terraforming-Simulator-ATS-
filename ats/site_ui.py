"""Compact site controls and batched drawing (no Qt item per panel)."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore,QtWidgets


def initialize(host):
    host.spoil_overlay=pg.ImageItem();host.spoil_overlay.setZValue(2);host.map.addItem(host.spoil_overlay)
    host.solar_planned=pg.PlotDataItem(pen=pg.mkPen('#7791a3',width=1,style=QtCore.Qt.PenStyle.DotLine))
    host.solar_deployed=pg.PlotDataItem(pen=pg.mkPen('#64b8ff',width=2))
    host.solar_active=pg.PlotDataItem(pen=pg.mkPen('#ffd166',width=2))
    host.solar_installers=pg.ScatterPlotItem(size=8,symbol='s',brush='#ffd166')
    host.site_access=pg.PlotDataItem(pen=pg.mkPen('#75c8a4',width=1,style=QtCore.Qt.PenStyle.DashLine))
    host.battery_platform_contours=pg.PlotDataItem(pen=pg.mkPen('#b09bff',width=2))
    host.battery_platform_reserves=pg.PlotDataItem(pen=pg.mkPen('#b09bff',width=1,style=QtCore.Qt.PenStyle.DotLine))
    for item in (host.solar_planned,host.solar_deployed,host.solar_active,host.solar_installers,host.site_access,host.battery_platform_contours,host.battery_platform_reserves):host.map.addItem(item)


def _rect(points,rect,supports=False):
    x,y,w,h=rect
    points.extend([(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y),(np.nan,np.nan)])
    if supports:points.extend([(x,y),(x+w,y+h),(np.nan,np.nan),(x+w,y),(x,y+h),(np.nan,np.nan)])


def refresh(host,terrain):
    project=host.complex_project
    simulations=[u.simulation for u in project.units] if project and project.shared_site else [host.sim]
    planned=[];deployed=[];active=[];installers=[];battery=[];reserves=[];access=[]
    for s in simulations:
        w=s.logistics;a=getattr(w,'solar_array',None) if w else None
        from .compact_layout import active_polygons
        for polygon in active_polygons(s):access.extend([*map(tuple,polygon),tuple(polygon[0]),(np.nan,np.nan)])
        if w and w.battery_platforms:
            p=w.battery_platforms
            for c,rect in zip(p.carriers.values(),p.rectangles(planned=True)):
                if c.mode=='DEPLOY' or c.state!='STOWED':_rect(reserves,rect)
                fraction=max(c.supports,c.apron)
                if fraction:
                    x,y,ww,hh=rect;_rect(battery,(x+(1-fraction)*ww/2,y+(1-fraction)*hh/2,ww*fraction,hh*fraction),True)
        if a is None:continue
        if w.solar_ground_rects():
            for polygon in w.solar_ground_polygons():planned.extend([*map(tuple,polygon),tuple(polygon[0]),(np.nan,np.nan)])
        for q in a.sections:
            if getattr(a,'mechanism',None)=='self_deploying':
                target=deployed if q.state=='DEPLOYED' else active
                for polygon in a.leaf_polygons(w,q):target.extend([*map(tuple,polygon),(np.nan,np.nan)])
                if q.supports:
                    from .self_solar import support_centres
                    r=np.sqrt(a.settings['support_pad_m2'])/2*q.supports
                    for point in support_centres(s.base.layout_spec,a.module(w,q),s.base.position,q.open_fraction):
                        _rect(target,(point[0]-r,point[1]-r,2*r,2*r),True)
                continue
            if q.state=='DEPLOYED':_rect(deployed,q.rect,True)
            elif q.state in ('ASSEMBLING','FOLDING'):_rect(active,q.rect,True)
        installers.extend(tuple(c.position[:2]) for c in a.crews)
    for item,points in ((host.solar_planned,planned),(host.solar_deployed,deployed),(host.solar_active,active),(host.battery_platform_contours,battery),(host.battery_platform_reserves,reserves),(host.site_access,access)):
        if points:
            values=np.asarray(points);item.setData(values[:,0],values[:,1],connect='finite')
        else:item.clear()
    host.solar_installers.setData(pos=np.asarray(installers) if installers else np.empty((0,2)))
    key=(id(terrain.height),getattr(terrain,'_spoil_reserve_revision',0))
    if getattr(host,'_spoil_render_key',None)!=key:
        stride=max(1,terrain.height.shape[0]//512);mask=terrain.dump_mask[::stride,::stride]
        allowed=getattr(terrain,'dump_allowed_mask',terrain.dump_mask)[::stride,::stride]
        rgba=np.zeros((*mask.shape,4),dtype=np.uint8)
        rgba[allowed]=[133,153,165,22];rgba[mask]=[255,164,60,65]
        padded=np.pad(mask,1);border=mask&~(padded[1:-1,:-2]&padded[1:-1,2:]&padded[:-2,1:-1]&padded[2:,1:-1])
        rgba[border]=[255,164,60,185]
        host.spoil_overlay.setImage(rgba,autoLevels=False)
        side=terrain.height.shape[0]*terrain.cell;host.spoil_overlay.setRect(QtCore.QRectF(0,0,side,side))
        host._spoil_render_key=key


class SiteWindow(QtWidgets.QDialog):
    def __init__(self,parent):
        super().__init__(parent);self.host=parent;self.setWindowTitle('Солнечные платформы и отвал');self.resize(760,650)
        layout=QtWidgets.QVBoxLayout(self);scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);layout.addWidget(scroll)
        page=QtWidgets.QWidget();box=QtWidgets.QVBoxLayout(page);scroll.setWidget(page)
        self.details=QtWidgets.QLabel();self.details.setWordWrap(True);self.details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse);box.addWidget(self.details)
        self.fold=QtWidgets.QPushButton('Сложить секции и вернуть на носители');box.addWidget(self.fold);self.fold.clicked.connect(lambda:self.action('fold'))
        self.deploy=QtWidgets.QPushButton('Развернуть / продолжить монтаж');box.addWidget(self.deploy);self.deploy.clicked.connect(lambda:self.action('deploy'))
        self.replan=QtWidgets.QPushButton('Перепланировать площадку после полного складывания');box.addWidget(self.replan);self.replan.clicked.connect(lambda:self.action('replan'))
        self.retry=QtWidgets.QPushButton('Повторить проверку после устранения препятствия');box.addWidget(self.retry);self.retry.clicked.connect(lambda:self.action('retry'))
        self.repack=QtWidgets.QPushButton('Сократить свободный резерв отвала');box.addWidget(self.repack);self.repack.clicked.connect(lambda:self.action('repack'))
        self.message=QtWidgets.QLabel();self.message.setWordWrap(True);box.addWidget(self.message);box.addStretch()
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(500);self.refresh()

    def action(self,kind):
        s=self.host.sim;w=s.logistics;a=getattr(w,'solar_array',None) if w else None
        try:
            if getattr(s,'regional_stage',None) or s.base.destination is not None:raise ValueError('Дождитесь завершения переезда')
            if kind=='repack':
                s.terrain.refresh_dump_reserve(force=True,repack=True)
            elif a is None:raise ValueError('Этот сценарий использует прежнюю агрегатную модель панелей')
            elif kind=='fold':s.landing_requested=True;a.request_fold()
            elif kind=='deploy':a.request_deploy()
            elif kind=='replan':w.plan_solar_field()
            elif kind=='retry':a.issue='';w.audit_solar_site()
            if kind!='repack':w.wake_for_solar_service()
            if s.shared_site:s.shared_site.invalidate_geometry()
            self.message.setText('Команда принята. Приводы работают при ходе модельного времени и расходуют энергию; панели остаются на носителях.' if getattr(a,'mechanism',None)=='self_deploying' else 'Команда принята. Монтаж и перевозка требуют модельного времени и энергии.')
            self.host.refresh(True)
        except ValueError as error:self.message.setText(str(error))
        self.refresh()

    def refresh(self):
        s=self.host.sim;w=s.logistics;a=getattr(w,'solar_array',None) if w else None;t=s.terrain
        lines=['<b>Солнечные секции</b>']
        grid=getattr(s.base,'layout_spec',{}).get('grid')
        attached=getattr(a,'mechanism',None)=='self_deploying'
        self.fold.setText('Сложить собственные крылья и убрать опоры' if attached else 'Сложить секции и вернуть на носители')
        self.deploy.setText('Раскрыть / возобновить раскрытие' if attached else 'Развернуть / продолжить монтаж')
        self.replan.setVisible(not attached)
        if attached:
            p=a.settings;spec=s.base.layout_spec
            lines=['<b>Самораскрывающиеся солнечные платформы</b>',
                f'Введено {a.installed_area:g} / {s.base.config.panel_area:g} м². Собственных носителей {len(a.sections)}; монтажники не используются.',
                f"Два крыла по {p['leaf_pairs']} листов {p['span_m']:g} × {p['leaf_width_m']:g} м; максимум {2*p['leaf_pairs']*p['span_m']*p['leaf_width_m']:g} м² на носитель. Панели и опоры постоянно закреплены.",
                f"Транспортная высота {p['hinge_height_m']+p['leaf_width_m']+p['transport_clearance_m']:g} м. Опоры {p['support_seconds']:g} с, раскрытие {p['unfold_seconds']:g} с, подключение {p['connect_seconds']:g} с; приводы {p['motor_w']:g} Вт плюс механическая работа.",
                'Серый пунктир — весь ход раскрытия; жёлтые листья — движение, голубые — подключённая платформа. Под крыльями и в зоне опор проезд запрещён.',
                'Параметры условные: ветровой расчёт и прочность шарниров не моделируются.']
            for q in a.sections:
                module=a.module(w,q)
                lines.append(f"Носитель {q.platform_id} · {q.id}: {q.state}; опоры {q.supports:.0%}, раскрытие {q.open_fraction:.0%}, подключение {q.connection:.0%}; {q.area_m2:g} м², полная масса {module['gross_mass_kg']:,.0f} кг; привод {q.actuation_energy_j/3600000:.3f} кВт·ч. {q.reason}")
            if a.issue:lines.append(a.issue)
            if grid:
                ww,hh=grid['deployed_dimensions_local_m']
                lines.append(f"Компактный массив: {grid['columns']} столбцов × {grid['rows']} рядов; полный развёрнутый габарит {ww:.1f} × {hh:.1f} м. Проходы {grid['column_access_m']:g} / {grid['row_access_m']:g} м показаны зелёным пунктиром; сервисный блок и посадки учтены.")
        elif a:
            p=a.settings;states={state:sum(q.state==state for q in a.sections) for state in ('STOWED','DEPLOYED','IN_TRANSIT','ASSEMBLING','FOLDING')}
            mass=sum(w.inventory.items[id].mass_kg for id in a.support_ids)
            from .solar_array import recipe
            _,_,_,volume=recipe(s.base.config.equipment.panels,p)
            lines += [f"Панели: {a.installed_area:g} / {s.base.config.panel_area:g} м² введено в работу; секций {len(a.sections)}.",
                f"Развёрнуто {states['DEPLOYED']}; сложено {states['STOWED']}; в перевозке {states['IN_TRANSIT']}; монтаж/демонтаж {states['ASSEMBLING']+states['FOLDING']}.",
                f"Стандартная секция: {p['section_width_m']:g} × {p['section_area_m2']/p['section_width_m']:g} м, опоры {p['support_height_m']:g} м; сложена {p['folded_length_m']:g} × {p['folded_width_m']:g} × {p['folded_height_m']:g} м.",
                f"Упакованный объём: {volume:g} м³. Опоры и монтажники: {mass:g} кг. На носителях: {a.available_installers} монтажников, доступно у проездов {len(a.crews)}.",
                f"Проезды: {p['column_lane_m']:g} / {p['row_lane_m']:g} м. Площадка с проездами ≈ {sum(r[2]*r[3] for r in a.reserved_rects):,.0f} м².",
                f"Полный цикл монтажа или возврата: оценка {a.duration()/3600:.2f} ч при непрерывном питании. Скорость {p['speed_m_s']:g} м/с; сборка/разборка секции {p['install_seconds']:g} / {p['fold_seconds']:g} с.",
                'До окончания первоначального монтажа энергетический паспорт описывает установленное оборудование, а не готовность всего комплекса.',
                a.issue or 'Модельные параметры оборудования; не спецификация производителя.']
        else:lines.append('До 20 м²: встроенная установка на крыше. Старые сохранения сохраняют прежнюю агрегатную модель поля.')
        if w and getattr(w,'solar_site_issue',''):lines.append(w.solar_site_issue)
        info=getattr(t,'dump_reserve_info',None)
        lines.append('<br><b>Отвал</b> · серая область — допустимые границы; оранжевая — резерв. Фактическая насыпь видна по рельефу.')
        if info:
            occupied=float(np.count_nonzero(~t.design_work&(t.height>t.reference+1e-10))*t.area)
            lines += [f"Допустимо {info['allowed_area_m2']:,.0f} м²; резерв {info['reserved_area_m2']:,.0f} м²; занято насыпью {occupied:,.1f} м².",
                f"Для текущих фронтов нужно {info['required_free_m3']:,.1f} м³, включая запас {info['margin']:.0%}; свободная вместимость при подборе {info['available_free_m3']:,.1f} м³.",
                'Вместимость достаточна.' if info['sufficient'] else 'НЕДОСТАТОЧНО МЕСТА: исчерпаны допустимые границы. Расширьте участок или измените план.']
        else:lines.append('Прежняя фиксированная область отвала.')
        self.details.setText('<br><br>'.join(lines))
        moving=bool(getattr(s,'regional_stage',None) or s.base.destination is not None)
        for button in (self.fold,self.deploy,self.retry):button.setEnabled(a is not None and not moving)
        self.replan.setEnabled(a is not None and a.stowed and not moving)
        self.repack.setEnabled(bool(info) and not moving)

    def closeEvent(self,event):self.timer.stop();super().closeEvent(event)
