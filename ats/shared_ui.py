"""One map for every base, crawler, solar field and drone in a shared site."""
import numpy as np
import pyqtgraph as pg


def initialize(host):
    host.peer_bases=pg.ScatterPlotItem(size=15,symbol='s',pen='w')
    host.peer_drones=pg.ScatterPlotItem(size=10,symbol='+')
    host.peer_excavators=pg.ScatterPlotItem(size=12,symbol='t')
    host.peer_outlines=pg.PlotDataItem(pen=pg.mkPen('#91acbf',width=1))
    host.peer_labels=[];host._assignment_revision=None
    for item in (host.peer_outlines,host.peer_bases,host.peer_drones,host.peer_excavators):host.map.addItem(item)
    for item,z in ((host.peer_bases,10),(host.peer_drones,12),(host.peer_excavators,10)):item.setZValue(z)
    host.peer_bases.sigClicked.connect(lambda item,points,event:host.select_complex(points[0].data()) if points else None)


def refresh(host):
    p=host.complex_project;site=p.shared_site if p else None
    if site is None:
        for item in (host.peer_outlines,host.peer_bases,host.peer_drones,host.peer_excavators):item.clear()
        for item in host.peer_labels:item.hide()
        host._peer_outline_cache={};host._peer_paths_key=None
        return
    from .base_layout import platform_polygons
    from .map_visuals import layout_key,marker_size
    from .site_ground import rectangle_polygon
    bases=[];drones=[];robots=[];keys=[];size=max(8.,marker_size(host)*.85)
    cache=getattr(host,'_peer_outline_cache',{});live={u.id for u in p.units}
    cache={id:v for id,v in cache.items() if id in live}
    styles=getattr(host,'_peer_styles',{})
    for i,u in enumerate(p.units):
        s=u.simulation;b=s.base
        if i not in styles:
            color=pg.intColor(i,hues=20);styles[i]=(color,pg.mkPen('#15212c',width=1.2),pg.mkBrush(color))
        color,pen,brush=styles[i]
        if i>=len(host.peer_labels):
            label=pg.TextItem(f'К{i+1}',color=color,anchor=(0,1),fill=(10,22,35,210));label.setZValue(16);host.map.addItem(label);host.peer_labels.append(label)
        label=host.peer_labels[i];label.show();label.setPos(*b.position[:2])
        if s is host.sim:continue
        bases.append(dict(pos=b.position[:2],brush=brush,data=u.id))
        drones.extend(dict(pos=d.position[:2],pen=pen,brush=brush,symbol=host.drone_symbol,size=size) for d in s.drones)
        if s.logistics and s.logistics.excavator:robots.append(dict(pos=s.logistics.excavator.position[:2],pen=pen,brush=brush,symbol=host.crawler_symbol,size=18))
        field=tuple(s.logistics.solar_ground_rects()) if s.logistics and getattr(s.logistics,'solar_array',None) is None else ()
        key=(layout_key(s),field);keys.append((u.id,key))
        if u.id not in cache or cache[u.id][0]!=key:
            r=b.config.radius
            polygons=list(platform_polygons(b.layout_spec,b.position)) if b.config.layout is not None else [rectangle_polygon((b.position[0]-r,b.position[1]-r,2*r,2*r))]
            polygons.extend(rectangle_polygon(rect) for rect in field)
            points=np.array([point for polygon in polygons for point in [*polygon,polygon[0],(np.nan,np.nan)]])
            cache[u.id]=(key,points)
    host._peer_outline_cache=cache;host._peer_styles=styles
    for label in host.peer_labels[len(p.units):]:label.hide()
    host.peer_bases.setData(bases);host.peer_drones.setData(drones);host.peer_excavators.setData(robots)
    key=tuple(keys)
    if getattr(host,'_peer_paths_key',None)!=key:
        if keys:
            points=np.concatenate([cache[id][1] for id,_ in keys]);host.peer_outlines.setData(points[:,0],points[:,1],connect='finite')
        else:host.peer_outlines.clear()
        host._peer_paths_key=key
    revision=(id(site),id(site.core.height),site.assignments,site.core.height.shape,site.core.cell)
    if host._assignment_revision!=revision:
        lookup=np.zeros((21,4),dtype=np.uint8)
        for u in p.units:
            rgb=pg.intColor(int(u.id[8:])-1,hues=20).getRgb();lookup[int(u.id[8:])]=[*rgb[:3],95]
        overlay=np.zeros((*site.core.height.shape,4),dtype=np.uint8)
        overlay[site.core.work_mask]=lookup[site.owners[site.core.work_mask]]
        host.design_overlay.setImage(overlay,autoLevels=False);side=site.core.height.shape[0]*site.core.cell
        from PySide6 import QtCore
        host.design_overlay.setRect(QtCore.QRectF(0,0,side,side));host._assignment_revision=revision
