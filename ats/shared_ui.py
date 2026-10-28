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
    host.peer_bases.sigClicked.connect(lambda item,points,event:host.select_complex(points[0].data()) if points else None)


def refresh(host):
    p=host.complex_project;site=p.shared_site if p else None
    if site is None:
        for item in (host.peer_outlines,host.peer_bases,host.peer_drones,host.peer_excavators):item.clear()
        for item in host.peer_labels:item.hide()
        return
    from .base_layout import layout_footprints
    bases=[];drones=[];robots=[];outlines=[]
    for i,u in enumerate(p.units):
        s=u.simulation;b=s.base;color=pg.intColor(i,hues=20)
        if i>=len(host.peer_labels):
            label=pg.TextItem(color=color,anchor=(0,1));host.map.addItem(label);host.peer_labels.append(label)
        label=host.peer_labels[i];label.show();label.setPos(*b.position[:2]);label.setText(f'К{i+1}')
        if s is host.sim:continue
        bases.append(dict(pos=b.position[:2],brush=color,data=u.id))
        drones.extend(dict(pos=d.position[:2],pen=pg.mkPen(color,width=2)) for d in s.drones)
        if s.logistics and s.logistics.excavator:robots.append(dict(pos=s.logistics.excavator.position[:2],brush=color))
        r=b.config.radius
        boxes=list(layout_footprints(b.layout_spec,b.position)) if b.config.layout is not None else [(b.position[0]-r,b.position[1]-r,2*r,2*r)]
        if s.logistics and getattr(s.logistics,'solar_array',None) is None:boxes.extend(s.logistics.solar_ground_rects())
        for x,y,w,h in boxes:outlines.extend([(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y),(np.nan,np.nan)])
    host.peer_bases.setData(bases);host.peer_drones.setData(drones);host.peer_excavators.setData(robots)
    if outlines:
        points=np.asarray(outlines);host.peer_outlines.setData(points[:,0],points[:,1],connect='finite')
    else:host.peer_outlines.clear()
    revision=(site.assignments,site.core.height.shape)
    if host._assignment_revision!=revision:
        lookup=np.zeros((21,4),dtype=np.uint8)
        for u in p.units:
            rgb=pg.intColor(int(u.id[8:])-1,hues=20).getRgb();lookup[int(u.id[8:])]=[*rgb[:3],95]
        overlay=np.zeros((*site.core.height.shape,4),dtype=np.uint8)
        overlay[site.core.work_mask]=lookup[site.owners[site.core.work_mask]]
        host.design_overlay.setImage(overlay,autoLevels=False);side=site.core.height.shape[0]*site.core.cell
        from PySide6 import QtCore
        host.design_overlay.setRect(QtCore.QRectF(0,0,side,side));host._assignment_revision=revision
