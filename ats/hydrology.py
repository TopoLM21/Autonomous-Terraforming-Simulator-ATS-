"""Regional reservoir experiment: conservative storage, not a flow solver.

The selected depression is a single communicating reservoir. Channel capacity
is a quasi-steady rating, without waves, sediment, infiltration or erosion.
No routine in this module changes the earthworks/material inventory.
"""
from dataclasses import dataclass,field
from collections import deque
from copy import deepcopy
import math
import numpy as np

YEAR=365*86400.


def create_experiment(land,project,*,latitude=31.078):
    from .channel import latitude_at
    latitude=latitude_at(dict(source=land.raster.metadata,origin_m=[0.,0.]),*project.points[-1],latitude)
    curve=BasinCurve.from_landscape(land,project.points[-1])
    angle=project.angle_deg or (min(m.native_angle_deg for m in land.soil_profile.materials) if land.soil_profile else 35.)
    config=FloodConfig(project.width_m,project.depth_m,project.length+project.shore_offset_m,
        side_slope=1/math.tan(math.radians(angle)),latitude=latitude)
    return Flooding(curve,config,project.identity)


def regional_bed(land):
    """Volume-conservative coarse quadrature of persisted fine height changes."""
    bed=land.raster.height.copy();rows,cols=bed.shape;rx,ry=land.raster.resolution_m
    weights=np.ones_like(bed);weights[[0,-1],:]*=.5;weights[:,[0,-1]]*=.5
    areas=weights*rx*ry
    delta=np.zeros_like(bed)
    for (tx,ty) in land.tiles:
        lo=(np.array([tx,ty])*land.tile_cells+.5)*land.cell_size
        hi=lo+(land.tile_cells-1)*land.cell_size
        node_lo=np.floor(lo/[rx,ry]+.5).astype(int);node_hi=np.floor(hi/[rx,ry]+.5).astype(int)
        if np.array_equal(node_lo,node_hi) and np.all(hi<=land.raster.extent_m):
            record=land.tiles.token((tx,ty))
            total=record.total if hasattr(record,'total') else float(land.tiles[tx,ty].sum(dtype=np.float64))
            delta[min(rows-1,node_lo[1]),min(cols-1,node_lo[0])]+=total*land.cell_size**2
            continue
        a=land.tiles[tx,ty];yy,xx=np.indices(a.shape)
        x=(tx*land.tile_cells+xx+.5)*land.cell_size;y=(ty*land.tile_cells+yy+.5)*land.cell_size
        outside=(x>land.raster.extent_m[0])|(y>land.raster.extent_m[1])
        if np.any(outside&(a!=0)):raise ValueError('Изменённый грунт за границами DEM')
        ix=np.clip(np.floor(x/rx+.5).astype(int),0,cols-1)
        iy=np.clip(np.floor(y/ry+.5).astype(int),0,rows-1)
        np.add.at(delta,(iy.ravel(),ix.ravel()),a.ravel()*land.cell_size**2)
    bed+=delta/areas
    return bed,areas


@dataclass
class BasinCurve:
    shape: tuple
    indices: np.ndarray
    bottom: np.ndarray
    areas: np.ndarray
    order: np.ndarray
    levels: np.ndarray
    prefix_area: np.ndarray
    prefix_moment: np.ndarray
    break_volume: np.ndarray
    maximum: float=0.

    @classmethod
    def from_landscape(cls,land,point):
        from .channel import ocean_nodes
        if land.raster.height.size>1_000_000:raise ValueError('Гидросценарий: нужен обзорный DEM до 1 млн узлов; укрупните исходную карту')
        bed,areas=regional_bed(land);rx,ry=land.raster.resolution_m
        rows,cols=bed.shape
        if len(point)!=2 or not all(math.isfinite(v) for v in point):raise ValueError('Нужна точка впадины X/Y')
        x,y=point
        if not 0<=x<=land.raster.extent_m[0] or not 0<=y<=land.raster.extent_m[1]:raise ValueError('Впадина вне DEM')
        possible=np.isfinite(bed)&(bed<0)&~ocean_nodes(land.raster)
        ix,iy=min(cols-2,int(x/rx)),min(rows-2,int(y/ry))
        seeds=[(j,i) for j in (iy,iy+1) for i in (ix,ix+1) if possible[j,i]]
        if not seeds:raise ValueError('Конец канала должен находиться в сухой впадине ниже моря')
        seed=min(seeds,key=lambda p:(p[1]*rx-x)**2+(p[0]*ry-y)**2)
        mask=np.zeros_like(possible);mask[seed]=True;queue=deque([seed])
        while queue:
            j,i=queue.popleft()
            if j in (0,rows-1) or i in (0,cols-1):raise ValueError('Впадина выходит за границу DEM; нужен больший регион')
            for jj,ii in ((j-1,i),(j+1,i),(j,i-1),(j,i+1)):
                if not np.isfinite(bed[jj,ii]):raise ValueError('У впадины есть NoData; нужен полный DEM')
                if possible[jj,ii] and not mask[jj,ii]:mask[jj,ii]=True;queue.append((jj,ii))
        ids=np.flatnonzero(mask);return cls.create(bed.shape,ids,bed.ravel()[ids],areas.ravel()[ids])

    @classmethod
    def create(cls,shape,indices,bottom,areas,maximum=0.):
        bottom=np.asarray(bottom,dtype=float);areas=np.asarray(areas,dtype=float)
        order=np.argsort(bottom,kind='stable');z=bottom[order];a=areas[order]
        pa=np.cumsum(a);pm=np.cumsum(a*z);vol=z*pa-pm
        result=cls(tuple(shape),np.asarray(indices,dtype=np.int64),bottom.copy(),areas.copy(),order,z,pa,pm,np.maximum.accumulate(vol),float(maximum))
        result.validate();return result

    def validate(self):
        n=len(self.bottom)
        if (len(self.shape)!=2 or any(type(v) is not int or v<2 for v in self.shape) or math.prod(self.shape)>16_000_000 or
                not n or self.indices.shape!=(n,) or self.indices.dtype.kind not in 'iu' or
                np.any(self.indices<0) or np.any(self.indices>=math.prod(self.shape)) or len(np.unique(self.indices))!=n or
                self.areas.shape!=(n,) or not np.all(np.isfinite(self.bottom)) or not np.all(np.isfinite(self.areas)) or
                np.any(self.areas<=0) or not math.isfinite(self.maximum) or np.max(self.bottom)>=self.maximum):
            raise ValueError('Повреждена кривая площади и объёма впадины')
        order=np.argsort(self.bottom,kind='stable');a=self.areas[order];z=self.bottom[order]
        if not (np.array_equal(self.order,order) and np.array_equal(self.levels,z) and
                np.array_equal(self.prefix_area,np.cumsum(a)) and np.array_equal(self.prefix_moment,np.cumsum(a*z)) and
                np.array_equal(self.break_volume,np.maximum.accumulate(z*np.cumsum(a)-np.cumsum(a*z)))):
            raise ValueError('Повреждены производные объёмы впадины')
        if not math.isfinite(self.capacity) or self.capacity<=0:raise ValueError('Впадина должна иметь конечную положительную вместимость')

    @property
    def capacity(self):return self.volume_at(self.maximum)

    def volume_at(self,level):
        if not math.isfinite(level):raise ValueError('Нужен конечный уровень воды')
        level=min(float(level),self.maximum);n=int(np.searchsorted(self.levels,level,side='left'))
        return max(0.,float(level*self.prefix_area[n-1]-self.prefix_moment[n-1])) if n else 0.

    def level_at(self,volume):
        if not math.isfinite(volume) or not 0<=volume<=self.capacity+max(1e-8,self.capacity*1e-13):raise ValueError('Объём воды вне впадины')
        i=min(len(self.levels)-1,max(0,int(np.searchsorted(self.break_volume,volume,side='right'))-1))
        if volume==self.break_volume[i]:return float(self.levels[i])
        return min(self.maximum,float((volume+self.prefix_moment[i])/self.prefix_area[i]))

    def area_at(self,level,*,dry=False):
        n=int(np.searchsorted(self.levels,level,side='right'))
        if dry and level<=self.levels[0]:return 0.
        return float(self.prefix_area[n-1]) if n else 0.

    def water_depth(self,volume):
        result=np.zeros(self.shape)
        if volume>0:result.ravel()[self.indices]=np.maximum(0,self.level_at(volume)-self.bottom)
        return result


@dataclass
class FloodConfig:
    width_m: float
    depth_m: float
    length_m: float
    side_slope: float=0.
    roughness: float=.03
    opening: float=1.
    flow_limit_m3_s: float=1e6
    evaporation_mm_day: float=5.
    season_amplitude: float=.5
    start_day: float=172.
    latitude: float=31.078
    inflow_salinity_g_l: float=38.
    solubility_g_l: float=300.
    exchange_factor: float=.25

    def validate(self):
        values=vars(self)
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in values.values()):raise ValueError('Нужны конечные параметры водоёма')
        if not (0<self.width_m<=10000 and 0<self.depth_m<=1000 and 0<self.length_m<=1e7 and 0<=self.side_slope<=20 and
                .005<=self.roughness<=.3 and 0<=self.opening<=1 and 0<=self.flow_limit_m3_s<=1e6 and
                0<=self.evaporation_mm_day<=100 and 0<=self.season_amplitude<=1 and 1<=self.start_day<=365 and -90<=self.latitude<=90 and
                0<=self.inflow_salinity_g_l<=1000 and 0<self.solubility_g_l<=1000 and 0<=self.exchange_factor<=1):
            raise ValueError('Проверьте размеры канала, шероховатость, затвор, испарение и солёность')

    def flow(self,level):
        """Approximate open-channel rating; no pump or arbitrarily large drop.

        Constant wetted section at sea-level depth; friction head is capped at
        the outlet bed for free discharge. A critical-inlet bound caps short
        channels. This is an estimate, not a Saint-Venant/backwater solution.
        """
        depth=self.depth_m;area=depth*(self.width_m+self.side_slope*depth)
        perimeter=self.width_m+2*depth*math.hypot(1,self.side_slope)
        head=max(0.,-max(float(level),-depth))
        manning=area/self.roughness*(area/perimeter)**(2/3)*math.sqrt(head/self.length_m)
        # Rectangular energy-limited inlet is conservative for the trapezoid.
        critical=self.width_m*math.sqrt(9.81)*(2*depth/3)**1.5
        return self.opening*min(manning,critical,self.flow_limit_m3_s)

    def evaporation(self,elapsed):
        day=(self.start_day-1+elapsed/86400)%365+1
        peak=172 if self.latitude>=0 else 355
        return self.evaporation_mm_day*.001/86400*(1+self.season_amplitude*math.cos(2*math.pi*(day-peak)/365))

    def exchange_geometry(self,level,net_flow,available_depth=math.inf):
        """Conditional two-layer friction bound, not a resolved estuary model.

        Return flow needs a submerged outlet AND density head exceeding the
        opposing free-surface head. Equal counterflows leave net storage alone.
        Half-section/interface drag is deliberately conservative. No tidal or
        wind forcing is invented. Density contrast is an approximate 0.7 kg/m³
        per g/L; it affects this rating only, not the water volume equation.
        """
        wet=min(self.depth_m,max(0.,self.depth_m+level),max(0.,available_depth))
        if wet<=0 or self.exchange_factor==0 or self.opening==0:return (0.,0.,0.,0.)
        area=wet*(self.width_m+self.side_slope*wet)/2
        perimeter=self.width_m+wet*math.hypot(1,self.side_slope)+2*self.side_slope*wet
        conductance=self.opening*self.exchange_factor*area/self.roughness*(area/perimeter)**(2/3)/math.sqrt(self.length_m)
        capacity=max(0.,(self.flow(-self.depth_m)-net_flow)/2)
        critical=self.opening*area*math.sqrt(9.81*wet/2)
        return conductance,max(0.,-level),.7*wet/(1000.+.7*self.inflow_salinity_g_l),min(capacity,critical)


@dataclass
class Flooding:
    curve: BasinCurve
    config: FloodConfig
    project_id: str
    volume_m3: float=0.
    initial_volume_m3: float=0.
    elapsed_seconds: float=0.
    inflow_m3: float=0.
    evaporated_m3: float=0.
    spilled_m3: float=0.
    channel_open: bool=False
    calculations: int=0
    skipped_seconds: float=0.
    last_inflow_m3_s: float=0.
    last_evaporation_m3_s: float=0.
    history: list=field(default_factory=list)
    version: int=3
    salt_mass_kg: float=0.
    initial_salt_kg: float=0.
    imported_salt_kg: float=0.
    exported_salt_kg: float=0.
    salt_baseline_seconds: float=0.
    salt_known: bool=True
    salt_origin: str='dry_start'
    exchanged_m3: float=0.
    exchange_calculations: int=0

    def restore_legacy(self):
        """Migrate both JSON regions and old graph-decoded mission objects.

        Old water histories contain no salinity observations. Only an initially
        dry, non-spilling run permits an estimate from the recorded sea inflow.
        Otherwise keep the water and explicitly leave its salt inventory unknown.
        """
        if vars(self).get('version') not in (1,2):return
        if self.version==1:
            vars(self.config).setdefault('inflow_salinity_g_l',38.)
            vars(self.config).setdefault('solubility_g_l',300.)
            known=self.initial_volume_m3==0 and self.spilled_m3==0
            mass=self.inflow_m3*self.config.inflow_salinity_g_l if known else 0.
            self.salt_mass_kg=mass;self.initial_salt_kg=0.;self.imported_salt_kg=mass;self.exported_salt_kg=0.
            self.salt_baseline_seconds=0. if known else self.elapsed_seconds
            self.salt_known=known;self.salt_origin='legacy_assumed_inflow' if known else 'legacy_unknown'
        # Existing experiments keep their one-way boundary until explicitly enabled.
        self.config.exchange_factor=0.;self.exchanged_m3=0.;self.exchange_calculations=0;self.version=3

    def initialize_salt(self,salinity_g_l):
        """Explicit current-state assumption; never reinterpret past evaporation."""
        if (type(salinity_g_l) not in (int,float) or not math.isfinite(salinity_g_l) or
                not 0<=salinity_g_l<=self.config.solubility_g_l):
            raise ValueError('Начальная солёность: от 0 до условного предела растворимости, г/л')
        self.salt_mass_kg=self.initial_salt_kg=self.volume_m3*salinity_g_l
        self.imported_salt_kg=self.exported_salt_kg=0.;self.salt_baseline_seconds=self.elapsed_seconds
        self.salt_known=True;self.salt_origin='manual'
        self._record_history()

    @property
    def dissolved_salt_kg(self):
        return min(self.salt_mass_kg,self.volume_m3*self.config.solubility_g_l) if self.salt_known else None
    @property
    def precipitated_salt_kg(self):
        return max(0.,self.salt_mass_kg-self.dissolved_salt_kg) if self.salt_known else None
    @property
    def salinity_g_l(self):
        # 1 g/L = 1 kg/m³. This is a mass/volume concentration, not PSU or g/kg.
        return self.dissolved_salt_kg/self.volume_m3 if self.salt_known and self.volume_m3>0 else None
    @property
    def salt_balance_error(self):
        return math.fsum((self.salt_mass_kg,-self.initial_salt_kg,-self.imported_salt_kg,self.exported_salt_kg)) if self.salt_known else None

    def _salt_flux(self,incoming,volume_before_spill,spilled=0.):
        if not self.salt_known:return
        imported=incoming*self.config.inflow_salinity_g_l
        self.salt_mass_kg+=imported;self.imported_salt_kg+=imported
        if spilled>0 and volume_before_spill>0:
            concentration=min(self.config.solubility_g_l,self.salt_mass_kg/volume_before_spill)
            exported=min(self.salt_mass_kg,concentration*spilled)
            self.salt_mass_kg-=exported;self.exported_salt_kg+=exported

    def _exchange_geometry(self):
        if not self.channel_open or not self.salt_known or self.volume_m3<=0:return (0.,0.,0.,0.)
        level=self.level
        return self.config.exchange_geometry(level,self.config.flow(level),level-float(self.curve.levels[0]))

    def _exchange_rate(self,concentration,geometry):
        conductance,head,density,capacity=geometry
        return min(capacity,conductance*math.sqrt(max(0.,abs(concentration-self.config.inflow_salinity_g_l)*density-head)))

    @property
    def exchange_rate(self):
        return self._exchange_rate(self.salinity_g_l,self._exchange_geometry()) if self.salinity_g_l is not None else 0.

    def _exchange_salt(self,seconds):
        """Implicit scalar salt exchange; stiff tiny reservoirs cannot go negative.

        Net inlet salt has already arrived. Dissolved outflow is bounded by
        solubility; a solid reserve can dissolve, but never leaves as sediment.
        Gross counterflows are recorded separately and cancel in water storage.
        """
        geometry=self._exchange_geometry()
        if not self.salt_known or not geometry[0] or not geometry[3]:return
        v=self.volume_m3;old=self.salt_mass_kg;source=self.config.inflow_salinity_g_l;limit=self.config.solubility_g_l
        concentration=min(limit,old/v)
        if self._exchange_rate(concentration,geometry)==0:return
        q=self._exchange_rate(limit,geometry)
        saturated=old+(source-limit)*q*seconds>=limit*v
        if saturated:new=old+(source-limit)*q*seconds
        else:
            low,high=0.,limit
            for _ in range(64):
                c=(low+high)/2;q=self._exchange_rate(c,geometry)
                residual=c*v-old-(source-c)*q*seconds
                if residual>0:high=c
                else:low=c
                if high-low<=max(1e-12,c*2e-15):break
            q=self._exchange_rate((low+high)/2,geometry)
            fraction=(q*seconds)/(v+q*seconds)
            new=old*(1-fraction)+source*v*fraction
        exchanged=q*seconds;imported=source*exchanged
        exported=max(0.,math.fsum((old,imported,-new)))
        self.salt_mass_kg=new;self.imported_salt_kg+=imported;self.exported_salt_kg+=exported
        self.exchanged_m3+=exchanged;self.exchange_calculations+=1

    def replace_curve(self,curve):
        """Terrain changes preserve water and salt; excess leaves as mixed brine."""
        curve.validate()
        if curve.shape!=self.curve.shape:raise ValueError('Водоём относится к другой сетке DEM')
        spilled=max(0.,self.volume_m3-curve.capacity)
        self._salt_flux(0.,self.volume_m3,spilled)
        self.volume_m3-=spilled;self.spilled_m3+=spilled;self.curve=curve
        self.validate()

    def _record_history(self):
        self.history.append(dict(seconds=self.elapsed_seconds,level_msl=self.level,volume_m3=self.volume_m3,
            area_m2=self.area,salinity_g_l=self.salinity_g_l,salt_mass_kg=self.salt_mass_kg if self.salt_known else None))
        if len(self.history)>2000:self.history=self.history[::2]

    def validate(self):
        self.curve.validate();self.config.validate()
        if self.version!=3 or type(self.project_id) is not str or not self.project_id or type(self.channel_open) is not bool:
            raise ValueError('Повреждён гидрологический сценарий')
        for name in ('volume_m3','initial_volume_m3','elapsed_seconds','inflow_m3','evaporated_m3','spilled_m3','skipped_seconds','last_inflow_m3_s','last_evaporation_m3_s','exchanged_m3'):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Повреждён баланс воды')
        if self.volume_m3>self.curve.capacity+1e-6 or abs(self.balance_error)>max(1e-6,(self.initial_volume_m3+self.inflow_m3)*1e-11):
            raise ValueError('Нарушен баланс воды')
        if type(self.salt_known) is not bool or self.salt_origin not in ('dry_start','manual','legacy_assumed_inflow','legacy_unknown'):
            raise ValueError('Повреждён источник солевого баланса')
        for name in ('salt_mass_kg','initial_salt_kg','imported_salt_kg','exported_salt_kg','salt_baseline_seconds'):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Повреждён баланс соли')
        if self.salt_baseline_seconds>self.elapsed_seconds+1e-6:raise ValueError('Начало солевого баланса позже текущего времени')
        if self.salt_known and abs(self.salt_balance_error)>max(1e-6,(self.initial_salt_kg+self.imported_salt_kg)*1e-11):
            raise ValueError('Нарушен баланс соли')
        if len(self.history)>2000:raise ValueError('Слишком длинная история воды')
        if type(self.calculations) is not int or self.calculations<0:raise ValueError('Повреждён счётчик расчётов воды')
        if type(self.exchange_calculations) is not int or self.exchange_calculations<0:raise ValueError('Повреждён счётчик обмена с морем')
        previous=-1.
        for row in self.history:
            keys={'seconds','level_msl','volume_m3','area_m2'}
            if (not isinstance(row,dict) or set(row) not in (keys,keys|{'salinity_g_l','salt_mass_kg'}) or
                    any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) for k in keys) or
                    any(v is not None and (type(v) not in (int,float) or not math.isfinite(v) or v<0)
                        for k,v in row.items() if k not in keys) or
                    not previous<=row['seconds']<=self.elapsed_seconds+1e-6 or row['volume_m3']<0 or row['area_m2']<0):
                raise ValueError('Повреждена история водоёма')
            previous=row['seconds']

    @property
    def level(self):return self.curve.level_at(self.volume_m3)
    @property
    def area(self):
        if self.volume_m3==0:return 0.
        level=self.level;upper=self.curve.area_at(level);lower=self.curve.area_at(np.nextafter(level,-math.inf))
        rate=self.config.evaporation(self.elapsed_seconds);q=self.config.flow(level) if self.channel_open else 0.
        if rate>0 and lower*rate<=q<=upper*rate:return q/rate
        return upper
    @property
    def balance_error(self):return math.fsum((self.volume_m3,-self.initial_volume_m3,-self.inflow_m3,self.evaporated_m3,self.spilled_m3))

    def rates(self,volume,elapsed):
        level=self.curve.level_at(volume)
        q=self.config.flow(level) if self.channel_open else 0.
        evap=self.config.evaporation(elapsed)*self.curve.area_at(level)
        return q,evap

    def current_rates(self):
        q,e=self.rates(self.volume_m3,self.elapsed_seconds)
        lower=self.config.evaporation(self.elapsed_seconds)*self.curve.area_at(np.nextafter(self.level,-math.inf))
        # At a newly wetted, zero-depth DEM terrace the effective evaporating
        # area may lie between the two one-sided areas. This prevents artificial
        # filling/drying chatter around an otherwise stationary shoreline.
        if lower<=q<=e:e=q
        if self.volume_m3==0:e=min(q,e)
        return q,e

    def advance(self,seconds,*,max_step=21600.):
        if not math.isfinite(seconds) or seconds<0:raise ValueError('Время воды должно быть неотрицательным')
        if not math.isfinite(max_step) or not 0<max_step<=86400:raise ValueError('Шаг воды: 0…86400 секунд')
        self.config.validate()
        end=self.elapsed_seconds+seconds
        if not self.history:self._record_history()
        remaining=float(seconds)
        while remaining>1e-8:
            # Seasonal forcing changes even when the surface is momentarily at
            # equilibrium. Constant, truly idle states need no numeric solves.
            q,e=self.current_rates()
            minimum_evap=(self.config.evaporation_mm_day*.001/86400*(1-self.config.season_amplitude)*
                          self.curve.area_at(self.level))
            if (q==0 and (self.volume_m3==0 or self.config.evaporation_mm_day==0) or
                    self.volume_m3==0 and q<=minimum_evap or
                    self.config.season_amplitude==0 and q==e):
                exchange_possible=bool(self._exchange_geometry()[0])
                salt_changes=self.exchange_rate>0 or q*self.config.inflow_salinity_g_l>0
                dt=min(remaining,max_step,86400.-self.elapsed_seconds%86400) if exchange_possible and salt_changes else remaining
                self.skipped_seconds+=dt;self.elapsed_seconds+=dt
                self.last_inflow_m3_s=q;self.last_evaporation_m3_s=min(q,e) if self.volume_m3==0 else e
                if q>0:
                    self._salt_flux(q*dt,self.volume_m3)
                    self.inflow_m3+=q*dt;self.evaporated_m3+=q*dt
                self._exchange_salt(dt);remaining-=dt
                if self.elapsed_seconds-self.history[-1]['seconds']>=86400:self._record_history()
                continue
            change_limit=.02*self.curve.capacity/abs(q-e) if q!=e else max_step
            dt=min(remaining,max_step,86400.-self.elapsed_seconds%86400,change_limit)
            mid=self.elapsed_seconds+dt/2;old=self.volume_m3
            q0,e0=self.rates(0.,mid)
            if old+dt*(q0-e0)<=0:
                new=0.;incoming=q0*dt;evap=old+incoming
            else:
                low,high=0.,self.curve.capacity
                for _ in range(64):
                    trial=(low+high)/2;q,e=self.rates(trial,mid)
                    residual=trial-old-dt*(q-e)
                    if residual>0:high=trial
                    else:low=trial
                    if high-low<=max(1e-9,abs(trial)*2e-15):break
                root=(low+high)/2
                snapped=False
                j=int(np.searchsorted(self.curve.break_volume,root))
                tolerance=max(1e-9,abs(root)*3e-15)
                for k in (j-1,j):
                    if 0<=k<len(self.curve.break_volume) and low-tolerance<=self.curve.break_volume[k]<=high+tolerance:
                        root=float(self.curve.break_volume[k]);snapped=True;break
                # Interpolate the bracketing monotone fluxes. This resolves a
                # partially wetted DEM terrace and the last floating-point
                # increment to sea level without inventing an overflow.
                left=min(low,np.nextafter(root,-math.inf));right=max(high,np.nextafter(root,math.inf))
                left=max(0.,left);right=min(self.curve.capacity,right)
                ql,el=self.rates(left,mid);qr,er=self.rates(right,mid)
                span=(ql-el)-(qr-er);needed=(root-old)/dt
                balanced=span>0 and qr-er<=needed<=ql-el
                fraction=float(np.clip((needed-(qr-er))/span,0,1)) if span>0 else 0.
                incoming=(qr+fraction*(ql-qr))*dt
                evap=min((er+fraction*(el-er))*dt,old+incoming)
                new=math.fsum((old,incoming,-evap))
                if snapped and balanced and abs(new-root)<=max(1e-12,abs(root)*3e-15):new=root
            spilled=max(0.,new-self.curve.capacity)
            self._salt_flux(incoming,max(0.,new),spilled)
            new=min(self.curve.capacity,max(0.,new))
            self.volume_m3=new;self.inflow_m3+=incoming;self.evaporated_m3+=evap;self.spilled_m3+=spilled
            self._exchange_salt(dt)
            self.last_inflow_m3_s=incoming/dt;self.last_evaporation_m3_s=evap/dt
            self.elapsed_seconds+=dt;remaining-=dt;self.calculations+=1
            if not self.history or self.elapsed_seconds-self.history[-1]['seconds']>=86400:
                self._record_history()
        self.elapsed_seconds=end
        if self.elapsed_seconds-self.history[-1]['seconds']>=86400:self._record_history()
        return self.report()

    def report(self):
        q,e=self.current_rates()
        return dict(mode='regional_reservoir',version=self.version,channel_open=self.channel_open,project_id=self.project_id,
            elapsed_seconds=self.elapsed_seconds,level_msl=self.level,volume_m3=self.volume_m3,area_m2=self.area,
            capacity_m3=self.curve.capacity,inflow_m3=self.inflow_m3,evaporated_m3=self.evaporated_m3,spilled_m3=self.spilled_m3,
            inflow_m3_s=q,evaporation_m3_s=e,
            water_balance_error_m3=self.balance_error,calculations=self.calculations,skipped_seconds=self.skipped_seconds,
            salt_known=self.salt_known,salt_origin=self.salt_origin,salt_baseline_seconds=self.salt_baseline_seconds,
            salinity_g_l=self.salinity_g_l,inflow_salinity_g_l=self.config.inflow_salinity_g_l,
            solubility_g_l=self.config.solubility_g_l,salt_mass_kg=self.salt_mass_kg if self.salt_known else None,
            dissolved_salt_kg=self.dissolved_salt_kg,precipitated_salt_kg=self.precipitated_salt_kg,
            initial_salt_kg=self.initial_salt_kg if self.salt_known else None,
            imported_salt_kg=self.imported_salt_kg if self.salt_known else None,
            exported_salt_kg=self.exported_salt_kg if self.salt_known else None,salt_balance_error_kg=self.salt_balance_error,
            exchange_factor=self.config.exchange_factor,exchange_m3_s=self.exchange_rate,exchanged_m3=self.exchanged_m3,
            gross_inflow_m3=self.inflow_m3+self.exchanged_m3,channel_outflow_m3=self.exchanged_m3,
            exchange_calculations=self.exchange_calculations)

    def to_data(self):
        from dataclasses import asdict
        state={k:deepcopy(v) for k,v in vars(self).items() if k not in ('curve','config')}
        curve=dict(shape=list(self.curve.shape),indices=self.curve.indices.tolist(),bottom=self.curve.bottom.tolist(),
                   areas=self.curve.areas.tolist(),maximum=self.curve.maximum)
        return dict(format='ATS_FLOODING',version=self.version,state=state,config=asdict(self.config),curve=curve)

    @classmethod
    def from_data(cls,data):
        if data.get('format')!='ATS_FLOODING' or data.get('version') not in (1,2,3):raise ValueError('Неизвестный формат водоёма')
        curve=BasinCurve.create(**data['curve']);config=FloodConfig(**data['config'])
        state=deepcopy(data['state'])
        if state.get('version',data['version'])!=data['version']:raise ValueError('Несогласованные версии водоёма')
        state['version']=data['version']
        result=cls(curve,config,**state);result.restore_legacy();result.validate();return result
