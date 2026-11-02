"""One coalescing background reader; no actor, clock or live terrain writes."""
from dataclasses import dataclass
from collections import OrderedDict
import threading
import time

PREFETCH_BYTES=64*1024**2


class PrefetchCancelled(Exception):pass


@dataclass
class PreparedWindow:
    origin: tuple
    n: int
    cell: float
    baseline: object
    current: object
    soil: object
    height_tokens: dict
    soil_tokens: dict
    fine_cell: float
    tile_cells: int
    soil_profile: object
    raster: object
    generation: object = None

    @property
    def key(self):return (*self.origin,self.n,self.cell)

    @property
    def nbytes(self):
        return self.baseline.nbytes+self.current.nbytes+sum(a.nbytes for _,_,a in (self.soil or {}).get('tiles',[]))


class WindowPrefetch:
    """One reader, a bounded set of ready windows and one replacement request.

    A worker exits when idle, so closed projects don't leave permanent threads.
    Cancellation never waits on disk/NumPy and cannot publish a late result.
    """
    def __init__(self):
        self._lock=threading.RLock();self._thread=None;self._pending=None
        self._generation=0;self._cancel=threading.Event();self._done=threading.Event()
        self.state='IDLE';self.error='';self.result=None;self.key=None
        self.elapsed=0.;self.progress=0.
        self._ready=OrderedDict();self._bytes=0;self.total=0;self.completed=0

    def request(self,snapshot,key,budget=PREFETCH_BYTES):
        self.request_many(snapshot,[key],budget,retain=False)

    def request_many(self,snapshot,keys,budget=PREFETCH_BYTES,*,retain=True):
        keys=list(dict.fromkeys(tuple(k) for k in keys))
        if not 1<=len(keys)<=8:raise ValueError('Предзагрузка: требуется 1…8 окон')
        with self._lock:
            self._cancel.set();self._done.set();self._generation+=1
            self._cancel=threading.Event();self._done=threading.Event()
            if not retain:self._ready.clear();self._bytes=0
            for key in list(self._ready):
                p=self._ready[key]
                if key not in keys or p.raster is not snapshot.raster or p.generation!=snapshot.generation_signature():
                    self._bytes-=self._ready.pop(key).nbytes
            self.result=None
            self.key=keys[0];self.state='QUEUED';self.error=''
            self.elapsed=0.;self.progress=0.
            self.total=len(keys);self.completed=sum(k in self._ready for k in keys)
            self._pending=(self._generation,snapshot,keys,budget,self._cancel,self._done)
            if self._thread is None:
                self._thread=threading.Thread(target=self._run,name='ats-window-read',daemon=True)
                try:self._thread.start()
                except (RuntimeError,OSError) as failure:
                    self._thread=None;self._pending=None;self.state='ERROR'
                    self.error='Не удалось запустить чтение: '+str(failure);self._done.set()

    def _run(self):
        while True:
            with self._lock:
                job=self._pending;self._pending=None
                if job is None:self._thread=None;return
            generation,region,keys,budget,cancel,done=job;start=time.perf_counter()
            def check(progress=None):
                if cancel.is_set():raise PrefetchCancelled()
                with self._lock:
                    if generation==self._generation:
                        self.state='READING'
                        if progress is not None:self.progress=(self.completed+progress)/self.total
            result=None;error='';state='READY'
            try:
                check(0.)
                for key in reversed(keys):
                    check()
                    with self._lock:
                        if generation!=self._generation:raise PrefetchCancelled()
                        if key in self._ready:continue
                    estimate=region.window_bytes(key[:2],key[2],key[3])
                    if estimate>budget:raise ValueError('Окно превышает лимит памяти предзагрузки')
                    with self._lock:
                        while self._ready and self._bytes+estimate>budget:
                            _,old=self._ready.popitem(last=False);self._bytes-=old.nbytes
                            old=None
                        self.result=None
                    result=region.prepare_window(key[:2],key[2],key[3],budget=budget,check=check)
                    check()
                    with self._lock:
                        if generation!=self._generation:raise PrefetchCancelled()
                        self._ready[key]=result;self._bytes+=result.nbytes;self.completed+=1
                        self.result=result;self.progress=min(1.,self.completed/self.total)
                    result=None
                check()
            except PrefetchCancelled:state='CANCELLED'
            except Exception as failure:
                state='ERROR';error=f'{type(failure).__name__}: {failure}'
            with self._lock:
                if generation==self._generation and not cancel.is_set():
                    self.result=self._ready.get(self.key);self.state=state;self.error=error;self.elapsed=time.perf_counter()-start
                    if state=='READY':self.progress=1.
                done.set()
            # Release the old immutable tile references before the next read.
            result=region=job=None

    def cancel(self):
        with self._lock:
            self._cancel.set();self._done.set();self._generation+=1
            self._pending=None;self.result=None;self._ready.clear();self._bytes=0;self.state='CANCELLED';self.error=''

    def prepared(self,key):
        with self._lock:return self._ready.get(tuple(key))

    def wait(self,timeout=30.):
        if not self._done.wait(timeout):raise TimeoutError('Предзагрузка ещё выполняется')
        return self.report()

    def report(self):
        with self._lock:
            return dict(state=self.state,key=self.key,error=self.error,progress=self.progress,
                read_seconds=self.elapsed,prepared_bytes=self._bytes,
                ready_windows=len(self._ready),windows_total=self.total,windows_done=self.completed)
