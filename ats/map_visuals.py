"""Top-down presentation only: solar colour, cached contours and batched beacons.

No values in this module feed simulation power, timing, navigation or soil.
"""
import math
import time
import numpy as np
from PySide6 import QtCore,QtGui,QtWidgets
import pyqtgraph as pg
from .solar import sunlight


class MapWidget(pg.PlotWidget):
    """Account for actual Qt paint as well as Python's scene preparation."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.paint_seconds=0.

    def paintEvent(self,event):
        begin=time.perf_counter();super().paintEvent(event)
        self.paint_seconds=.75*getattr(self,'paint_seconds',0.)+.25*(time.perf_counter()-begin)


def smooth(lo,hi,value):
    x=min(1.,max(0.,(value-lo)/(hi-lo)))
    return x*x*(3-2*x)


def sky(latitude,day,hour):
    sun=sunlight(latitude,day,hour)
    night=1-smooth(-6.,3.,sun.elevation_deg)
    warm=smooth(-6.,-1.,sun.elevation_deg)*(1-smooth(2.,10.,sun.elevation_deg))
    return sun.elevation_deg,night,warm


def direction(latitude,day,hour):
    lat=math.radians(latitude);decl=math.radians(sunlight(latitude,day,hour).declination_deg)
    angle=math.pi*(hour-12)/12
    return np.array([-math.cos(decl)*math.sin(angle),math.cos(lat)*math.sin(decl)-math.sin(lat)*math.cos(decl)*math.cos(angle),
        math.sin(lat)*math.sin(decl)+math.cos(lat)*math.cos(decl)*math.cos(angle)])


NEUTRAL_LIGHT=np.array([.35,-.45,.82]);NEUTRAL_LIGHT/=np.linalg.norm(NEUTRAL_LIGHT)


def relief_light(sim,enabled):
    if not enabled:return ('fixed',),NEUTRAL_LIGHT
    # Quantise MODEL minutes; never replay skipped intervals at high speed.
    absolute=sim.time+sim.config.start_hour*3600;slot=math.floor(absolute/900)
    day=(sim.config.base.start_day_of_year-1+slot//96)%365+1
    latitude=sim.base.config.latitude_deg;hour=(slot%96)/4
    vector=direction(latitude,day,hour)
    weight=smooth(0.,.25,float(vector[2]));vector=NEUTRAL_LIGHT*(1-weight)+vector*weight
    vector/=np.linalg.norm(vector)
    return (latitude,day,slot),vector


def terrain_rgb(height,cell,depth,maximum,*,reference=None,marine=None,light=None,materials=None):
    """Sea is connectivity data, never the sign of height in a dry depression."""
    sea=np.zeros(height.shape,dtype=bool) if marine is None else np.asarray(marine,dtype=bool)
    land=(height if reference is None else reference)[~sea]
    lo=float(land.min()) if land.size else 0.;hi=float(land.max()) if land.size else 1.
    if hi-lo<1:lo-=.5;hi=lo+1.
    if materials is None:
        colors=np.array([[164,148,112],[203,185,141],[232,218,180]])
        rgb=np.stack([np.interp(height,[lo,(lo+hi)/2,hi],colors[:,i]) for i in range(3)],axis=-1)
    else:rgb=np.asarray(materials,dtype=float).copy()
    if reference is not None:
        delta=height-reference;cut=(delta<-1e-6)&~sea;fill=(delta>1e-6)&~sea
        for mask,color in ((cut,[105,145,154]),(fill,[217,160,91])):
            amount=np.clip(np.abs(delta[mask])/max(.1,depth),.25,.8)*(1. if materials is None else .3)
            rgb[mask]=rgb[mask]*(1-amount[:,None])+np.asarray(color)*amount[:,None]
    rgb[sea]=[43,105,150]
    gy,gx=np.gradient(height,cell);sx,sy,sz=NEUTRAL_LIGHT if light is None else light
    shade=.72+.28*np.maximum(0.,(-gx*sx-gy*sy+sz)/np.sqrt(1+gx*gx+gy*gy))
    return (rgb*shade[...,None]).clip(0,255).astype(np.uint8)


def shape(kind):
    p=QtGui.QPainterPath()
    if kind=='drone':
        for x,y in ((-.32,-.32),(.32,-.32),(.32,.32),(-.32,.32)):
            p.addEllipse(QtCore.QPointF(x,y),.15,.15)
            p.moveTo(0,0);p.lineTo(x,y)
        p.addRoundedRect(QtCore.QRectF(-.15,-.2,.3,.4),.06,.06)
    else:
        p.addRoundedRect(QtCore.QRectF(-.28,-.32,.56,.55),.06,.06)
        p.addRect(QtCore.QRectF(-.46,-.36,.14,.7));p.addRect(QtCore.QRectF(.32,-.36,.14,.7))
        p.moveTo(-.1,.2);p.lineTo(-.16,.46);p.lineTo(.16,.46);p.lineTo(.1,.2);p.closeSubpath()
    return p


def initialize(host):
    host.view_options=QtWidgets.QToolButton();host.view_options.setText('Вид ▾')
    host.view_options.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    menu=QtWidgets.QMenu(host.view_options);host.view_options.setMenu(menu)
    host.day_night=menu.addAction('Суточное освещение');host.day_night.setCheckable(True);host.day_night.setChecked(True)
    host.drone_lights=menu.addAction('Огоньки дронов');host.drone_lights.setCheckable(True);host.drone_lights.setChecked(True)
    host.day_night.setToolTip('Только вид карты. Выключено: постоянный читаемый дневной свет. Энергетика не меняется.')
    host.drone_lights.setToolTip('Навигационные огни ночью, независимо от затемнения карты. Декоративные маркеры без прожекторов и теней.')
    host.day_night.toggled.connect(lambda _:host.display_style_changed())
    host.drone_lights.toggled.connect(lambda _:host.display_style_changed())
    host.night_overlay=QtWidgets.QGraphicsRectItem();host.night_overlay.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    host.night_overlay.setZValue(6);host.night_overlay.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
    host.map.addItem(host.night_overlay,ignoreBounds=True)
    glow=QtGui.QRadialGradient(0,0,.5)
    for stop,color in ((0,QtGui.QColor(145,218,255,105)),(.45,QtGui.QColor(145,218,255,45)),(1,QtGui.QColor(145,218,255,0))):glow.setColorAt(stop,color)
    host.light_halos=pg.ScatterPlotItem(pxMode=True,pen=None,symbol='o',size=15,brush=QtGui.QBrush(glow))
    host.light_points=pg.ScatterPlotItem(pxMode=True,pen=None,symbol='o',size=4,brush='#e4f9ff')
    for item,z in ((host.light_halos,13),(host.light_points,14)):
        item.setZValue(z);item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton);host.map.addItem(item,ignoreBounds=True)
    host.drone_symbol=shape('drone');host.crawler_symbol=shape('crawler')
    host.drone_pen=pg.mkPen('#12212c',width=1.2);host.drone_brushes=[pg.mkBrush(c) for c in host.palette]
    host._site_render_cache={};host._site_paths_key=None
    for name in ('design_overlay','brush_overlay','profile_line','route','workfront_outline','site_access','spoil_overlay'):
        getattr(host,name).setZValue(8)
    for item in [*host.drone_labels,host.base_label,host.excavator_label]:
        item.setZValue(16);item.fill=pg.mkBrush(10,22,35,210)
    host.excavator_marker.setZValue(10)
    host.excavator_marker.setSymbol(host.crawler_symbol);host.excavator_marker.setPen(pg.mkPen('#15212c',width=1.5))


def lighting(host):
    if not hasattr(host,'night_overlay'):return
    s=host.sim;side=s.terrain.height.shape[0]*s.terrain.cell
    if host.night_overlay.rect()!=QtCore.QRectF(0,0,side,side):host.night_overlay.setRect(0,0,side,side)
    elevation,night,warm=sky(s.base.config.latitude_deg,s.day_of_year,s.hour)
    host.visual_sun_elevation=elevation;host.visual_night=night
    visible=host.render_mode.currentData()!='off'
    host.night_overlay.setVisible(visible and host.day_night.isChecked())
    if host.day_night.isChecked():
        a=145*night;b=24*warm;alpha=a+b
        color=((np.array([12,24,52])*a+np.array([244,148,75])*b)/alpha) if alpha else np.zeros(3)
        rgba=(*map(int,color),int(alpha))
        if getattr(host,'_tint_rgba',None)!=rgba:
            host.night_overlay.setBrush(QtGui.QColor(*rgba));host._tint_rgba=rgba
    show=visible and host.drone_lights.isChecked() and night>.01
    host.light_halos.setVisible(show);host.light_points.setVisible(show)
    if show:
        opacity=round(night,2)
        if getattr(host,'_beacon_opacity',None)!=opacity:
            host.light_halos.setOpacity(opacity);host.light_points.setOpacity(opacity);host._beacon_opacity=opacity


def beacons(host):
    lighting(host)
    if not host.drone_lights.isChecked() or host.visual_night<=.01:return
    p=host.complex_project
    sims=[u.simulation for u in p.units] if p and p.shared_site else [host.sim]
    size=marker_size(host)
    key=(size,tuple((id(s),tuple((tuple(d.position[:2]),d.grounded) for d in s.drones)) for s in sims))
    if getattr(host,'_beacon_key',None)==key:return
    host._beacon_key=key
    drones=[d for s in sims for d in s.drones];points=np.array([d.position[:2] for d in drones])
    host.light_halos.setData(pos=points,size=[size+6 if not d.grounded else size+2 for d in drones])
    host.light_points.setData(pos=points,size=[4 if not d.grounded else 3 for d in drones])


def marker_size(host):
    dx,dy=host.view_box.viewPixelSize()
    return round(min(23.,max(9.,2./max(abs(dx),abs(dy),1e-6))),1)


def cache_paths(host):
    """Reuse Qt's rasterisation of static vectors, with a bounded device extent.

    setData invalidates the item cache; Qt handles pan/scale/device changes.
    A close zoom disables it before a world-size contour becomes a huge bitmap.
    """
    dx,dy=host.view_box.viewPixelSize();ratio=host.map.devicePixelRatioF()
    for name in ('solar_planned','solar_deployed','solar_active','battery_platform_contours','battery_platform_reserves','site_access','base_outline','peer_outlines'):
        curve=getattr(host,name).curve;rect=curve.boundingRect()
        fits=max(rect.width()/max(abs(dx),1e-9),rect.height()/max(abs(dy),1e-9))*ratio<=2048
        mode=QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache if fits else QtWidgets.QGraphicsItem.CacheMode.NoCache
        if curve.cacheMode()!=mode:curve.setCacheMode(mode)


def place_labels(host):
    """A small screen-space placement pass for the selected fleet's labels."""
    if not host.labels_view.isChecked():return
    occupied=[host.base_label.sceneBoundingRect(),host.excavator_label.sceneBoundingRect()]
    occupied.extend(label.sceneBoundingRect() for label in host.peer_labels if label.isVisible())
    bounds=host.view_box.sceneBoundingRect()
    for drone,label in zip(host.sim.drones,host.drone_labels):
        anchor=host.view_box.mapViewToScene(QtCore.QPointF(*map(float,drone.position[:2])))
        rect=label.sceneBoundingRect();width,height=rect.width(),rect.height()
        candidates=[(x,y) for y in (-8.,-height-16.,height+8.,-2*height-24.,2*height+16.) for x in (8.,-width-8.)]
        for x,y in candidates:
            point=host.view_box.mapSceneToView(anchor+QtCore.QPointF(x,y));label.setPos(point)
            rect=label.sceneBoundingRect()
            if bounds.contains(rect) and not any(rect.intersects(old.adjusted(-2,-2,2,2)) for old in occupied):break
        occupied.append(rect)


def layout_key(sim):
    b=sim.base;source=getattr(b,'layout_spec',None);spec=source or {}
    return (id(sim),id(source),tuple(b.position),tuple((a['id'],tuple(a['offset_m']),tuple(a['dimensions_m']),a.get('role')) for a in spec.get('platforms',())),
        spec.get('parameters',{}).get('heading_deg'),b.config.radius,tuple(map(tuple,spec.get('access_lanes_local',()))),
        tuple(sorted(spec.get('battery_platforms',{}).items())))


def telemetry_due(host,force=False):
    import time
    now=time.perf_counter();context=(id(host.sim),id(host.sim.config))
    if force or not host.running or context!=getattr(host,'_telemetry_context',None) or now-getattr(host,'_telemetry_wall',0) >= .5:
        host._telemetry_wall=now;host._telemetry_context=context;return True
    return False
