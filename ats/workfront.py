"""Local canal operations: an advancing face and two separated spoil banks."""
import math
import numpy as np


def coordinates(plan, x, y):
    shape = np.broadcast_shapes(np.shape(x), np.shape(y))
    distance = np.full(shape, np.inf)
    chainage = np.zeros(shape)
    side = np.zeros(shape)
    starts = plan.channel.get('chainage_starts', [])
    offset = 0.
    for number, segment in enumerate(plan.segments):
        for a, b in zip(segment.points, segment.points[1:]):
            a, b = np.asarray(a), np.asarray(b)
            v = b-a; length = float(np.linalg.norm(v))
            if length < 1e-12:continue
            along = ((x-a[0])*v[0]+(y-a[1])*v[1])/length
            clipped = np.clip(along, 0, length)
            d = np.hypot(x-a[0]-clipped*v[0]/length, y-a[1]-clipped*v[1]/length)
            nearer = d < distance
            chainage[nearer] = (clipped+(starts[number] if starts else offset))[nearer]
            signed = ((y-a[1])*v[0]-(x-a[0])*v[1])/length
            side[nearer] = np.broadcast_to(signed, shape)[nearer]
            distance = np.minimum(distance, d)
            offset += length
    return distance, chainage, side


def expand(mask):
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
                                 for dy in range(3) for dx in range(3)])


def bank_layout(terrain, plan, work, protected,target=None):
    """Chebyshev setbacks are conservative Euclidean setbacks, even at bends."""
    cell = terrain.cell; c = terrain.config
    settings = plan.channel['operations']
    distance, along, signed = coordinates(plan, terrain.x, terrain.y)
    bin_size = max(4., cell)
    bins = np.maximum(0, (along/bin_size).astype(np.int32))
    radii = np.full(int(bins.max())+1, max(s.width for s in plan.segments)/2)
    np.maximum.at(radii, bins[work], distance[work])
    # Adjacent cross sections share the wider opening; no piles on cut slopes.
    radii = np.maximum.reduce([radii, np.r_[radii[1:], radii[-1]], np.r_[radii[0], radii[:-1]]])
    blocked = work.copy()
    for _ in range(math.ceil(settings['setback_m']/cell)):
        blocked = expand(blocked)
    if protected is not None:blocked |= protected
    edge = np.minimum(np.minimum(terrain.x, terrain.y),
                      np.minimum(c.terrain.n*cell-terrain.x, c.terrain.n*cell-terrain.y))
    mask = (~blocked & (edge > 2*cell) & (np.abs(signed)>=distance*.95) &
            (distance < radii[bins]+settings['setback_m']+settings['bank_width_m']))
    if c.terrain.compact_spoil:
        from .spoil import reserve
        from .site_ground import marine_mask
        from types import SimpleNamespace
        if target is None:
            from .plan import target_surface
            target,_=target_surface(terrain,plan)
        view=SimpleNamespace(config=c,cell=cell,area=cell**2,height=terrain.height,
            reference=terrain.reference,target=target,design_work=work,work_mask=work,
            dump_slope=getattr(terrain,'dump_slope',terrain.slope))
        mask &= ~marine_mask(terrain)
        geometry=dict(allowed=mask.copy(),along=along,signed=signed,opening=radii[bins],settings=settings)
        reserved,cap,info=reserve(view,**geometry)
        terrain.dump_allowed_mask=mask.copy();terrain._spoil_geometry=geometry
        terrain.dump_reserve_info=info
        return reserved,cap,along,signed
    # Erode each bank to form stable, bounded heaps with zero height at its rim.
    inside = mask.copy(); depth = np.zeros(mask.shape)
    dump_slope=getattr(terrain,'dump_slope',terrain.slope)
    limit = math.ceil(c.terrain.dump_max_height/(dump_slope*c.terrain.dump_slope_factor*cell))+2
    for layer in range(limit):
        inside = ~expand(~inside)
        inside[[0,-1],:] = False; inside[:,[0,-1]] = False
        if not np.any(inside):break
        depth[inside] = (layer+1)*cell
    cap = terrain.reference+np.minimum(c.terrain.dump_max_height,
        depth*dump_slope*c.terrain.dump_slope_factor)
    return mask, cap, along, signed


class Workfront:
    def __init__(self, terrain):
        self.terrain = terrain
        plan = terrain.config.plan
        self.settings = plan.channel['operations']
        geometry=getattr(terrain,'_spoil_geometry',None)
        if geometry:self.chainage,self.signed=geometry['along'],geometry['signed']
        else:_, self.chainage, self.signed = coordinates(plan, terrain.x, terrain.y)
        self.indices = np.flatnonzero(terrain.excavation)
        self.along = self.chainage.ravel()[self.indices]
        order=np.argsort(self.along,kind='stable')
        self.indices,self.along=self.indices[order],self.along[order]
        self.start = float(self.along.min()) if len(self.indices) else 0.
        self.end = float(self.along.max()) if len(self.indices) else self.start
        self.position = self.start
        self.revision = -1
        self.active_indices = self.indices
        self.stationed=True
        self.moving=False
        self.followers_launched=False
        self.legacy_pickups=[]
        if self.sectors:
            excess=terrain.height.ravel()[self.indices]-terrain.target.ravel()[self.indices]
            useful=np.flatnonzero(excess>1e-10)
            self._set_sector(float(self.along[useful[0]]) if len(useful) else self.start)

    @property
    def sectors(self):return self.settings.get('sector_length_m',0)>0

    def _set_sector(self,along):
        length=self.settings['sector_length_m']
        self.sector_id=max(0,math.floor((along+1e-9)/length))
        self.sector_start=max(self.start-self.terrain.cell/2,self.sector_id*length)
        self.sector_end=min(self.end+self.terrain.cell/2,(self.sector_id+1)*length)
        self.sector_centre=(self.sector_start+self.sector_end)/2
        count=max(1,math.ceil((self.end+self.terrain.cell/2)/length)-math.floor(max(0.,self.start-self.terrain.cell/2)/length))
        self.sector_tolerance=self.terrain.config.terrain.completion_volume*.1/count
        self.revision=-1

    def cut_mask(self,patch):
        along=self.chainage[patch]
        return (along>=self.sector_start-1e-9)&(along<self.sector_end-1e-9)&self.stationed

    @property
    def complete(self):
        self.update()
        return self.sectors and self.sector_remaining<=self.sector_tolerance

    def advance_sector(self):
        """Called only after the controller has drained every carried load."""
        if not self.complete:return False
        excess=self.terrain.height.ravel()[self.indices]-self.terrain.target.ravel()[self.indices]
        useful=(excess>1e-10)&(self.along>=self.sector_end-1e-9)
        if not np.any(useful):return False
        self._set_sector(float(self.along[np.flatnonzero(useful)[0]]))
        self.legacy_pickups=[]
        self.stationed=False;self.moving=False;self.followers_launched=False
        self.update()
        return True

    def update(self):
        t = self.terrain
        if self.revision == t.revision:return
        excess = t.height.ravel()[self.indices]-t.target.ravel()[self.indices]
        remaining = excess > 1e-10
        if self.sectors:
            local=(self.along>=self.sector_start-1e-9)&(self.along<self.sector_end-1e-9)
            self.sector_remaining=float(np.maximum(excess[local],0).sum()*t.area)
            useful=remaining&local
            self.position=float(self.along[np.flatnonzero(useful)[0]]) if np.any(useful) else self.sector_end
            self.active_indices=self.indices[useful] if self.stationed else self.indices[:0]
            self.revision=t.revision
            return
        # A fixed global fraction of the existing completion tolerance prevents
        # numerical crumbs under tracks from pinning the entire work front.
        # These cells stay in the plan, mass ledger and available dig candidates.
        cumulative=np.cumsum(np.where(remaining,excess,0))*t.area
        skip=t.config.terrain.completion_volume*.1
        i=int(np.searchsorted(cumulative,skip,side='right'))
        self.position=float(self.along[i]) if i<len(self.along) else self.end
        self.active_indices = self.indices[remaining &
            (self.along <= self.position+self.settings['front_length_m'])]
        self.revision = t.revision

    def report(self):
        self.update()
        report=dict(chainage_m=self.position, start_m=self.start, end_m=self.end,
                    length_m=self.settings['front_length_m'], setback_m=self.settings['setback_m'])
        if self.sectors:report.update(mode='fixed_sectors',sector_id=self.sector_id,
            length_m=self.sector_end-self.sector_start,
            sector_start_m=self.sector_start,sector_end_m=self.sector_end,
            sector_centre_m=self.sector_centre,sector_length_m=self.settings['sector_length_m'],
            sector_remaining_m3=self.sector_remaining,sector_complete=self.complete,stationed=self.stationed)
        return report
