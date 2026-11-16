"""Meter-scaled human-operated equipment bodies shared by both map windows."""
import math
import numpy as np
from PySide6 import QtCore,QtGui
import pyqtgraph as pg


def body_polygons(fleet):
    """Return physical body rectangles; the router's support disc is larger."""
    if fleet is None:return []
    bx,by=map(float,fleet.base_position[:2])
    rows=[dict(id='conventional_base',kind='base',dimensions_m=(8.,8.,3.),heading_rad=0.,
               polygon=np.array([[bx-4,by-4],[bx+4,by-4],[bx+4,by+4],[bx-4,by+4]]))]
    for machine in fleet.machines:
        dimensions=tuple(map(float,machine.dimensions_m));length,width,_=dimensions
        heading=vars(machine).get('heading_rad')
        if heading is None:
            delta=(np.asarray(machine.path[0])[:2]-machine.position[:2]) if machine.path else np.array([1.,0.])
            heading=math.atan2(float(delta[1]),float(delta[0]))
        c,s=math.cos(heading),math.sin(heading)
        local=np.array([[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2]])
        polygon=local@np.array([[c,s],[-s,c]])+machine.position[:2]
        rows.append(dict(id=machine.id,kind=machine.kind,dimensions_m=dimensions,
                         heading_rad=float(heading),polygon=polygon))
    return rows


def ramp_polygon(fleet):
    """Planned access corridor, not a claim that its soil has been excavated."""
    ramp=getattr(fleet,'ramp',None) if fleet is not None else None
    if not ramp or ramp.get('entry_m') is None or ramp.get('end_m') is None:return np.empty((0,2))
    start,end=np.asarray(ramp['entry_m'][:2],dtype=float),np.asarray(ramp['end_m'][:2],dtype=float)
    delta=end-start;length=float(np.linalg.norm(delta))
    if length<=1e-9 or ramp.get('width_m',0.)<=0:return np.empty((0,2))
    side=np.array([-delta[1],delta[0]])/length*ramp['width_m']/2
    return np.asarray([start+side,end+side,end-side,start-side])


class GroundEquipmentItem(pg.GraphicsObject):
    """One cached paint picture, with a stable scene item count during work."""
    def __init__(self):
        super().__init__();self.polygons=[];self.ramp_polygon=np.empty((0,2));self._key=None
        self._picture=QtGui.QPicture();self._bounds=QtCore.QRectF()
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton);self.setZValue(1.)

    def set_fleet(self,fleet):
        ramp=getattr(fleet,'ramp',None) if fleet is not None else None
        ramp_key=(tuple(ramp['entry_m']) if ramp.get('entry_m') is not None else (),
                  tuple(ramp['end_m']) if ramp.get('end_m') is not None else (),ramp.get('width_m')) if ramp else ()
        key=() if fleet is None else (tuple(fleet.base_position[:2]),tuple(
            (m.id,m.kind,tuple(m.dimensions_m),tuple(m.position[:2]),vars(m).get('heading_rad'),
             tuple(m.path[0][:2]) if vars(m).get('heading_rad') is None and m.path else ())
            for m in fleet.machines),ramp_key)
        if key==self._key:return
        rows=body_polygons(fleet)
        self.prepareGeometryChange();self._key=key;self.polygons=rows;self.ramp_polygon=ramp_polygon(fleet)
        self._picture=QtGui.QPicture();painter=QtGui.QPainter(self._picture)
        if len(self.ramp_polygon):
            pen=QtGui.QPen(QtGui.QColor('#9df3a4'));pen.setCosmetic(True);pen.setWidthF(1.3);pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            painter.setPen(pen);painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(float(x),float(y)) for x,y in self.ramp_polygon]))
        colors={'base':'#54c89f','excavator':'#f4bf5f','truck':'#d8e7f0'}
        for row in rows:
            points=row['polygon'];polygon=QtGui.QPolygonF([QtCore.QPointF(float(x),float(y)) for x,y in points])
            color=QtGui.QColor(colors[row['kind']]);color.setAlpha(210)
            pen=QtGui.QPen(QtGui.QColor('#142d3d'));pen.setCosmetic(True);pen.setWidthF(1.5)
            painter.setPen(pen);painter.setBrush(color);painter.drawPolygon(polygon)
            if row['kind']!='base':
                # Front stripe identifies heading without expanding the body.
                front=points[1:3];pen=QtGui.QPen(QtGui.QColor('#fff4cf'));pen.setCosmetic(True);pen.setWidthF(2.)
                painter.setPen(pen);painter.drawLine(QtCore.QPointF(*map(float,front[0])),QtCore.QPointF(*map(float,front[1])))
        painter.end()
        if rows:
            points=np.concatenate([*[row['polygon'] for row in rows],self.ramp_polygon]);lo=points.min(axis=0);hi=points.max(axis=0)
            self._bounds=QtCore.QRectF(float(lo[0]-.2),float(lo[1]-.2),float(hi[0]-lo[0]+.4),float(hi[1]-lo[1]+.4))
        else:self._bounds=QtCore.QRectF()
        self.setVisible(bool(rows));self.update()

    def boundingRect(self):return self._bounds

    def paint(self,painter,*_):painter.drawPicture(0,0,self._picture)


def refresh_geometry(host,container,fleet):
    """Create once; coordinates remain in metres when the camera zooms."""
    item=getattr(host,'_ground_geometry_item',None)
    if item is None:
        if fleet is None:return
        item=GroundEquipmentItem();host._ground_geometry_item=item;container.addItem(item)
    item.set_fleet(fleet)
    label=getattr(host,'_ground_ramp_label',None)
    if len(item.ramp_polygon):
        if label is None:
            label=pg.TextItem('Проектный съезд',color='#9df3a4',anchor=(0,1));label.setZValue(3.)
            host._ground_ramp_label=label;container.addItem(label)
        label.setPos(*map(float,item.ramp_polygon[0]));label.show()
    elif label is not None:label.hide()
