"""One coalescing background reader; no actor, clock or live terrain writes."""
from dataclasses import dataclass
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

    @property
    def key(self):return (*self.origin,self.n,self.cell)

    @property
    def nbytes(self):
        return self.baseline.nbytes+self.current.nbytes+sum(a.nbytes for _,_,a in (self.soil or {}).get('tiles',[]))


class WindowPrefetch:
    """At most one reader and one replacement request; obsolete reads are dropped.

    A worker exits when idle, so closed projects don't leave permanent threads.
    Cancellation never waits on disk/NumPy and cannot publish a late result.
    """
    def __init__(self):
        self._lock=threading.RLock();self._thread=None;self._pending=None
        self._generation=0;self._cancel=threading.Event();self._done=threading.Event()
        self.state='IDLE';self.error='';self.result=None;self.key=None
        self.elapsed=0.;self.progress=0.

    def request(self,snapshot,key,budget=PREFETCH_BYTES):
        with self._lock:
            self._cancel.set();self._done.set();self._generation+=1
            self._cancel=threading.Event();self._done=threading.Event()
            self.key=tuple(key);self.state='QUEUED';self.error='';self.result=None
            self.elapsed=0.;self.progress=0.
            self._pending=(self._generation,snapshot,self.key,budget,self._cancel,self._done)
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
            generation,region,key,budget,cancel,done=job;start=time.perf_counter()
            def check(progress=None):
                if cancel.is_set():raise PrefetchCancelled()
                with self._lock:
                    if generation==self._generation:
                        self.state='READING'
                        if progress is not None:self.progress=progress
            result=None;error='';state='READY'
            try:
                check(0.)
                result=region.prepare_window(key[:2],key[2],key[3],budget=budget,check=check)
                check(1.)
            except PrefetchCancelled:state='CANCELLED'
            except Exception as failure:
                state='ERROR';error=f'{type(failure).__name__}: {failure}'
            with self._lock:
                if generation==self._generation and not cancel.is_set():
                    self.result=result;self.state=state;self.error=error;self.elapsed=time.perf_counter()-start
                done.set()
            # Release the old immutable tile references before the next read.
            result=region=job=None

    def cancel(self):
        with self._lock:
            self._cancel.set();self._done.set();self._generation+=1
            self._pending=None;self.result=None;self.state='CANCELLED';self.error=''

    def prepared(self,key):
        with self._lock:return self.result if self.state=='READY' and self.key==tuple(key) else None

    def wait(self,timeout=30.):
        if not self._done.wait(timeout):raise TimeoutError('Предзагрузка ещё выполняется')
        return self.report()

    def report(self):
        with self._lock:
            return dict(state=self.state,key=self.key,error=self.error,progress=self.progress,
                read_seconds=self.elapsed,prepared_bytes=self.result.nbytes if self.result is not None else 0)
