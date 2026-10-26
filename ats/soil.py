"""Conditional stratigraphy and conservative, sparse material transport.

Depths follow the original surface, not sea level. Each edited fine-grid tile
stores native cut depth plus loose thickness for each material. Untouched soil
is implicit. Loose soil is mixed within a column; compaction/swell are omitted.
"""
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class SoilMaterial:
    name: str
    density: float
    digging_factor: float = 1.
    color: str = '#c9b57a'
    bearing_kpa: float = 60.
    traction: float = .55
    rolling_resistance: float = .10


@dataclass
class SoilProfile:
    materials: list[SoilMaterial] = field(default_factory=list)
    thicknesses: list[float] = field(default_factory=list)
    name: str = 'Условные слои Каттары'

    @classmethod
    def qattara(cls):
        return cls([SoilMaterial('Песчаные отложения',1600.,1.,'#dbc18a',60.,.55,.10),
                    SoilMaterial('Песчано-глинистые отложения',1850.,1.5,'#ae8162',100.,.6,.07),
                    SoilMaterial('Выветрелая известняковая порода',2200.,3.,'#a9b6bc',500.,.7,.03)], [2.,6.])

    @classmethod
    def from_data(cls, data):
        result=cls([SoilMaterial(**m) for m in data['materials']],list(data['thicknesses']),
                   data.get('name','Условные слои'))
        result.validate();return result

    def validate(self):
        if not 1<=len(self.materials)<=3 or len(self.thicknesses)!=len(self.materials)-1:
            raise ValueError('Грунт: требуется 1–3 материала и толщины верхних слоёв')
        if not isinstance(self.name,str) or not self.name.strip():raise ValueError('Нужно название профиля грунта')
        for m in self.materials:
            if (not all(math.isfinite(v) for v in (m.bearing_kpa,m.traction,m.rolling_resistance)) or
                    m.bearing_kpa<=0 or not 0<m.traction<=1 or not 0<m.rolling_resistance<1):
                raise ValueError('Неверная несущая способность или параметры сцепления грунта')
            if (not isinstance(m.name,str) or not m.name.strip() or
                    not math.isfinite(m.density) or not 100<=m.density<=5000 or
                    not math.isfinite(m.digging_factor) or not 1<=m.digging_factor<=100 or
                    not isinstance(m.color,str) or len(m.color)!=7 or m.color[0]!='#'):
                raise ValueError('Неверные свойства грунта')
            try:int(m.color[1:],16)
            except ValueError as error:raise ValueError('Неверный цвет грунта') from error
        if any(not math.isfinite(v) or v<=0 for v in self.thicknesses):
            raise ValueError('Толщины слоёв должны быть положительными')

    @property
    def densities(self):return np.array([m.density for m in self.materials])

    @property
    def factors(self):return np.array([m.digging_factor for m in self.materials])

    @property
    def boundaries(self):return np.cumsum(self.thicknesses)


@dataclass
class SoilLoad:
    profile: SoilProfile
    masses: np.ndarray

    @classmethod
    def empty(cls,profile):return cls(profile,np.zeros(len(profile.materials)))

    @property
    def mass(self):return float(self.masses.sum())

    @property
    def volume(self):return float((self.masses/self.profile.densities).sum())

    @property
    def effort_volume(self):return float((self.masses/self.profile.densities*self.profile.factors).sum())

    @property
    def factor(self):return self.effort_volume/self.volume if self.volume>0 else 1.

    def scaled(self,fraction):return SoilLoad(self.profile,self.masses*max(0.,min(1.,fraction)))

    def add(self,other):
        if self.profile!=other.profile:raise ValueError('Несовместимые каталоги грунта')
        self.masses+=other.masses

    def remove(self,mass):
        total=self.mass
        if not math.isfinite(mass) or mass<0 or mass>total+1e-7:raise ValueError('Неверная масса грунта')
        removed=self.scaled(min(1.,mass/total) if total>0 else 0.)
        self.masses=np.maximum(0.,self.masses-removed.masses)
        return removed

    def validate(self,mass=None):
        if (not isinstance(self.masses,np.ndarray) or self.masses.shape!=(len(self.profile.materials),) or
                not np.all(np.isfinite(self.masses)) or np.any(self.masses < -1e-9)):
            raise ValueError('Повреждён состав груза')
        if mass is not None and abs(self.mass-mass)>1e-7:raise ValueError('Масса груза не совпадает с его составом')


class SoilField:
    """Tile coordinates are global at the persistent fine resolution.

    A 0.5/1 m workspace retains the individual 0.25 m columns under its cells,
    avoiding material creation when changing window size or resolution.
    """
    def __init__(self,profile,n,cell,initial=None):
        profile.validate();self.profile=profile;self.n=n;self.cell=cell
        initial=initial or {}
        self.fine_cell=float(initial.get('fine_cell',cell))
        if not math.isfinite(self.fine_cell) or self.fine_cell<=0:raise ValueError('Неверный шаг грунта')
        self.factor=round(cell/self.fine_cell)
        if self.factor not in (1,2,4) or abs(self.factor*self.fine_cell-cell)>1e-9:
            raise ValueError('Неверное разрешение материалов')
        self.tile_cells=int(initial.get('tile_cells',64))
        if not 8<=self.tile_cells<=256:raise ValueError('Неверный блок материалов')
        origin=np.asarray(initial.get('origin_m',[0.,0.]),dtype=float)
        if origin.shape!=(2,) or not np.all(np.isfinite(origin)) or np.any(origin<0):raise ValueError('Неверное начало материалов')
        self.origin=np.rint(origin/self.fine_cell).astype(np.int64)
        if np.any(np.abs(self.origin*self.fine_cell-origin)>1e-7):raise ValueError('Начало материалов вне сетки')
        self.tiles={(int(x),int(y)):np.array(a,dtype=np.float64,copy=True) for x,y,a in initial.get('tiles',[])}
        self.revision=0;self._audit_revision=-1
        self.validate()

    def validate(self):
        self.profile.validate()
        if (type(self.n) is not int or not 16<=self.n<=1024 or
                not math.isfinite(self.cell) or self.cell<=0 or
                not math.isfinite(self.fine_cell) or self.fine_cell<=0 or
                type(self.factor) is not int or self.factor not in (1,2,4) or
                abs(self.factor*self.fine_cell-self.cell)>1e-9 or
                type(self.tile_cells) is not int or not 8<=self.tile_cells<=256 or
                not isinstance(self.origin,np.ndarray) or self.origin.shape!=(2,) or
                self.origin.dtype.kind not in 'iu' or np.any(self.origin<0) or
                np.any(self.origin+self.n*self.factor >= 1<<31)):
            raise ValueError('Повреждена сетка грунта')
        shape=(len(self.profile.materials)+1,self.tile_cells,self.tile_cells)
        for key,tile in self.tiles.items():
            if (not isinstance(key,tuple) or len(key)!=2 or any(type(v) is not int or not 0<=v<1<<31 for v in key) or
                    not isinstance(tile,np.ndarray) or tile.shape!=shape or
                    not np.all(np.isfinite(tile)) or np.any(tile< -1e-9)):
                raise ValueError('Повреждён блок состава грунта')
        self._rho=self.profile.densities;self._contacts=self.profile.boundaries
        self._limits=np.r_[0.,self._contacts,np.inf];self._audit_revision=-1

    def _coordinates(self,indices,center=False):
        indices=np.asarray(indices,dtype=np.int64).ravel()
        if np.any(indices<0) or np.any(indices>=self.n**2):raise ValueError('Ячейка грунта вне участка')
        if self.factor==1 or center:
            offset=self.factor//2 if center else 0
            return self.origin[0]+indices%self.n*self.factor+offset,self.origin[1]+indices//self.n*self.factor+offset
        if center:
            offsets=np.array([self.factor//2]);ox=oy=offsets
        else:
            oy,ox=np.indices((self.factor,self.factor));ox=ox.ravel();oy=oy.ravel()
        xx=(self.origin[0]+(indices%self.n)[:,None]*self.factor+ox).ravel()
        yy=(self.origin[1]+(indices//self.n)[:,None]*self.factor+oy).ravel()
        return xx,yy

    def _read(self,xx,yy):
        values=np.zeros((len(self.profile.materials)+1,len(xx)))
        if not len(xx):return values
        size=self.tile_cells;keys=(yy//size)*(1<<32)+xx//size
        for packed in np.unique(keys):
            key=int(packed%(1<<32)),int(packed//(1<<32));tile=self.tiles.get(key)
            if tile is not None:
                mask=keys==packed;values[:,mask]=tile[:,yy[mask]%size,xx[mask]%size]
        return values

    def _write(self,xx,yy,values):
        if not len(xx):return
        size=self.tile_cells;keys=(yy//size)*(1<<32)+xx//size
        for packed in np.unique(keys):
            key=int(packed%(1<<32)),int(packed//(1<<32));mask=keys==packed
            tile=self.tiles.get(key)
            if tile is None:
                tile=np.zeros((len(self.profile.materials)+1,size,size));self.tiles[key]=tile
            tile[:,yy[mask]%size,xx[mask]%size]=values[:,mask]
        self.revision+=1

    def _native(self,depth,thickness):
        shape=(-1,)+(1,)*depth.ndim
        return np.maximum(0.,np.minimum(depth+thickness,self._limits[1:].reshape(shape))-
                          np.maximum(depth,self._limits[:-1].reshape(shape)))

    def _removed(self,values,dh):
        loose=values[1:];total=loose.sum(axis=0)
        take=np.minimum(dh,total)
        fraction=np.divide(take,total,out=np.zeros_like(take),where=total>0)
        loose_removed=loose*fraction
        native_dh=np.maximum(dh-take,0.)
        removed=loose_removed+self._native(values[0],native_dh)
        after=values.copy();after[0]+=native_dh;after[1:]=np.maximum(0.,loose-loose_removed)
        return after,removed

    def _fraction(self,values,heights,max_mass,max_volume):
        """Exact inverse of the piecewise linear removed-mass curve."""
        area=self.fine_cell**2;volume=float(heights.sum()*area)
        if volume<=0 or max_mass<=0 or max_volume<=0:return 0.
        maximum=min(1.,max_volume/volume)
        depth=values[0];loose=values[1:].sum(axis=0);rho=self._rho
        native_id=np.searchsorted(self._contacts,depth,side='right')
        top=np.divide((values[1:]*rho[:,None]).sum(axis=0),loose,
                      out=rho[native_id].copy(),where=loose>0)
        slope=float((heights*top).sum()*area)
        events=[];jumps=[]
        # Crossing the mixed loose layer, then each still-buried native contact.
        contact=np.divide(loose,heights,out=np.full_like(loose,np.inf),where=heights>0)
        valid=(loose>0)&(contact<maximum)
        events.append(contact[valid]);jumps.append((heights*(rho[native_id]-top)*area)[valid])
        for i,bound in enumerate(self._contacts):
            contact=np.divide(loose+bound-depth,heights,out=np.full_like(loose,np.inf),where=heights>0)
            valid=(depth<bound)&(contact>=0)&(contact<maximum)
            events.append(contact[valid]);jumps.append((heights*(rho[i+1]-rho[i])*area)[valid])
        positions=np.concatenate(events);changes=np.concatenate(jumps)
        if not len(positions):return min(maximum,max_mass/slope) if slope>0 else 0.
        order=np.argsort(positions,kind='stable');positions=positions[order];changes=changes[order]
        starts=np.r_[0.,positions];slopes=np.r_[slope,slope+np.cumsum(changes)]
        masses=np.r_[0.,np.cumsum(np.diff(starts)*slopes[:-1])]
        i=min(len(starts)-1,int(np.searchsorted(masses,max_mass,side='right')-1))
        return min(maximum,float(starts[i]+max(0.,max_mass-masses[i])/slopes[i]))

    def extract(self,indices,heights,max_mass=math.inf,max_volume=math.inf,commit=False):
        indices=np.asarray(indices).ravel();heights=np.asarray(heights,dtype=float).ravel()
        active=heights>0;ids=indices[active];dh=np.repeat(heights[active],self.factor**2)
        if not len(ids) or max_mass<=0 or max_volume<=0:return heights*0.,SoilLoad.empty(self.profile)
        xx,yy=self._coordinates(ids);values=self._read(xx,yy)
        # Most footprints stay in one layer. Their exact solution is linear,
        # including loose material of the same kind lying above the native soil.
        volume=float(dh.sum()*self.fine_cell**2);maximum=min(1.,max_volume/volume)
        native=np.searchsorted(self._contacts,values[0],side='right');material=int(native[0])
        same=(np.all(native==material) and
              np.all(values[0]+np.maximum(dh*maximum-values[1:].sum(axis=0),0.)<=self._limits[material+1]) and
              all(not np.any(values[1+i]) for i in range(len(self._rho)) if i!=material))
        if same:
            fraction=min(maximum,max_mass/(volume*self._rho[material]));actual=dh*fraction
            after=values.copy();loose=values[1+material]
            after[0]+=np.maximum(actual-loose,0.);after[1+material]=np.maximum(loose-actual,0.)
            masses=np.zeros(len(self._rho));masses[material]=float(actual.sum()*self.fine_cell**2)*self._rho[material]
            load=SoilLoad(self.profile,masses)
        else:
            fraction=self._fraction(values,dh,max_mass,max_volume)
            after,removed=self._removed(values,dh*fraction)
            load=SoilLoad(self.profile,removed.sum(axis=1)*self.fine_cell**2*self._rho)
        if commit and load.mass>0:self._write(xx,yy,after)
        return heights*fraction,load

    def add(self,indices,heights,load):
        """The supplied heights sum to the parcel volume at workspace resolution."""
        ids=np.asarray(indices).ravel();dh=np.asarray(heights,dtype=float).ravel();active=dh>0
        if load.profile!=self.profile:raise ValueError('Состав груза не соответствует профилю грунта')
        if load.volume<=0:return
        xx,yy=self._coordinates(ids[active]);values=self._read(xx,yy)
        portions=(load.masses/self._rho)/load.volume
        values[1:]+=portions[:,None]*np.repeat(dh[active],self.factor**2)
        self._write(xx,yy,values)

    def avalanche(self,source,destination,thickness):
        """Edges in a matching are disjoint, so no column can spend soil twice."""
        source=np.asarray(source);destination=np.asarray(destination)
        xx,yy=self._coordinates(source);values=self._read(xx,yy)
        after,removed=self._removed(values,np.repeat(thickness,self.factor**2))
        dx,dy=self._coordinates(destination);receiving=self._read(dx,dy)
        receiving[1:]+=removed
        self._write(np.r_[xx,dx],np.r_[yy,dy],np.concatenate((after,receiving),axis=1))

    @property
    def volume_delta_by_material(self):
        if self._audit_revision!=self.revision:
            volume=np.zeros(len(self.profile.materials))
            for (tx,ty),stored in self.tiles.items():
                x0=max(0,int(self.origin[0])-tx*self.tile_cells);y0=max(0,int(self.origin[1])-ty*self.tile_cells)
                x1=min(self.tile_cells,int(self.origin[0])+self.n*self.factor-tx*self.tile_cells)
                y1=min(self.tile_cells,int(self.origin[1])+self.n*self.factor-ty*self.tile_cells)
                if x1<=x0 or y1<=y0:continue
                tile=stored[:,y0:y1,x0:x1]
                volume+=(tile[1:]-self._native(np.zeros_like(tile[0]),tile[0])).sum(axis=(1,2))*self.fine_cell**2
            self._audit_volume=volume;self._audit_revision=self.revision
        return self._audit_volume.copy()

    @property
    def mass_delta(self):return float((self.volume_delta_by_material*self._rho).sum())

    def height_delta(self,indices):
        xx,yy=self._coordinates(indices);v=self._read(xx,yy)
        return (v[1:].sum(axis=0)-v[0]).reshape(-1,self.factor**2).mean(axis=1)

    def surface_ids(self,indices):
        xx,yy=self._coordinates(indices,center=True);v=self._read(xx,yy)
        return np.where(v[1:].sum(axis=0)>1e-12,np.argmax(v[1:],axis=0),
                        np.searchsorted(self._contacts,v[0],side='right'))

    def export(self):
        return dict(fine_cell=self.fine_cell,tile_cells=self.tile_cells,
                    origin_m=(self.origin*self.fine_cell).tolist(),
                    tiles=[[x,y,a.copy()] for (x,y),a in self.tiles.items()])


def cargo_volume(mass,load,density):
    return load.volume if load is not None else mass/density


def json_soil_state(state):
    if state is None:return None
    return {**state,'tiles':[[x,y,np.asarray(a).tolist()] for x,y,a in state.get('tiles',[])]}
