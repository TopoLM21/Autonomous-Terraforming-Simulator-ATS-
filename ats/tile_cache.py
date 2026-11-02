"""Immutable tile versions on owned temporary disk, with a bounded RAM LRU.

The files are backing storage, not a second terrain ledger or an autosave.
Snapshots retain old versions while a reader is running. Saved missions embed
the arrays themselves and never depend on these temporary paths.
"""
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from pathlib import Path
import hashlib
import os
import tempfile
import threading
import weakref
import numpy as np

DEFAULT_TILE_RAM = 64 * 1024**2


def _unlink(path):
    try:Path(path).unlink(missing_ok=True)
    except OSError:pass  # the owned directory also cleans up at final release


class TileFiles:
    def __init__(self):
        self.directory=tempfile.TemporaryDirectory(prefix='ats-tiles-')
        self.serial=0

    def write(self,array):
        self.serial+=1
        path=Path(self.directory.name)/f'{self.serial}.npy'
        pending=path.with_suffix('.tmp')
        try:
            with pending.open('wb') as stream:np.save(stream,array,allow_pickle=False)
            os.replace(pending,path)
        finally:pending.unlink(missing_ok=True)
        return path


class TileVersion:
    def __init__(self,files,array):
        self.files=files;self.path=files.write(array)
        self.shape=array.shape;self.nbytes=array.nbytes
        self.total=float(array.sum(dtype=np.float64))
        self.digest=hashlib.sha256(array).digest()
        weakref.finalize(self,_unlink,str(self.path))

    def read(self):
        try:
            with self.path.open('rb') as stream:array=np.load(stream,allow_pickle=False)
            if (array.dtype!=np.float64 or array.shape!=self.shape or not np.all(np.isfinite(array)) or
                    hashlib.sha256(array).digest()!=self.digest):
                raise ValueError('Неверный блок')
        except (OSError,ValueError,EOFError) as error:
            raise OSError('Не удалось прочитать тайл файлового кэша') from error
        array.flags.writeable=False
        return array


class TileCache:
    def __init__(self,limit_bytes=DEFAULT_TILE_RAM):
        self._files=TileFiles();self._records={};self._ram=OrderedDict();self._bytes=0
        self._lock=threading.RLock();self._pending=None
        self.hits=self.misses=self.evictions=self.writes=0
        self.set_limit(limit_bytes)

    def set_limit(self,value):
        if type(value) is not int or not 0<=value<=4*1024**3:
            raise ValueError('Лимит кэша тайлов: 0…4096 МиБ')
        with self._lock:
            self.limit_bytes=value
            self._trim()

    def _trim(self):
        while self._bytes>self.limit_bytes:
            _,array=self._ram.popitem(last=False);self._bytes-=array.nbytes;self.evictions+=1

    def records(self):
        result=self._records.copy()
        if self._pending is not None:
            for key,value in self._pending.items():
                if value is None:result.pop(key,None)
                else:result[key]=value
        return result

    def version(self,key):
        if self._pending is not None and key in self._pending:
            value=self._pending[key]
            if value is None:raise KeyError(key)
            return value
        return self._records[key]

    def read(self,key):
        with self._lock:
            record=self.version(key)
            if key in self._ram and self._records.get(key) is record:
                self.hits+=1;self._ram.move_to_end(key);return self._ram[key]
            self.misses+=1;array=record.read()
            if array.nbytes<=self.limit_bytes and self._records.get(key) is record:
                previous=self._ram.pop(key,None)
                if previous is not None:self._bytes-=previous.nbytes
                self._ram[key]=array;self._bytes+=array.nbytes;self._trim()
            return array

    @staticmethod
    def validate_key(key):
        if (not isinstance(key,tuple) or len(key)!=3 or key[0] not in ('height','soil') or
                any(type(v) is not int or not 0<=v<1<<31 for v in key[1:])):
            raise ValueError('Неверный адрес тайла')

    def write(self,key,value):
        self.validate_key(key)
        array=np.array(value,dtype=np.float64,copy=True,order='C')
        if (array.ndim!=(2 if key[0]=='height' else 3) or array.shape[-1]!=array.shape[-2] or
                not 8<=array.shape[-1]<=256 or array.ndim==3 and not 1<=array.shape[0]<=32 or
                not np.all(np.isfinite(array))):raise ValueError('Неверный массив тайла')
        array.flags.writeable=False
        with self._lock:
            record=TileVersion(self._files,array);self.writes+=1
            if self._pending is not None:self._pending[key]=record
            else:self._apply({key:record})

    def delete(self,key):
        self.version(key)
        if self._pending is not None:self._pending[key]=None
        else:self._apply({key:None})

    def _apply(self,changes):
        for key,value in changes.items():
            array=self._ram.pop(key,None)
            if array is not None:self._bytes-=array.nbytes
            if value is None:self._records.pop(key,None)
            else:self._records[key]=value

    @contextmanager
    def transaction(self):
        """Publish all height/material blocks together, or retain the old index."""
        with self._lock:
            if self._pending is not None:raise RuntimeError('Вложенная запись тайлов')
            self._pending={}
            try:
                yield
                self._apply(self._pending)
            finally:self._pending=None

    def snapshot(self,kind):
        with self._lock:return FrozenTiles({(x,y):v for (k,x,y),v in self.records().items() if k==kind})

    def report(self):
        return dict(limit_bytes=self.limit_bytes,resident_bytes=self._bytes,resident_tiles=len(self._ram),
            tile_count=len(self._records),logical_bytes=sum(v.nbytes for v in self._records.values()),
            hits=self.hits,misses=self.misses,evictions=self.evictions,writes=self.writes)

    def __deepcopy__(self,memo):
        other=TileCache(self.limit_bytes);memo[id(self)]=other
        for key,value in self._records.items():other.write(key,value.read())
        return other


class TileMap(MutableMapping):
    def __init__(self,cache,kind):self.cache=cache;self.kind=kind
    def __getitem__(self,key):return self.cache.read((self.kind,*key))
    def __setitem__(self,key,value):self.cache.write((self.kind,*key),value)
    def __delitem__(self,key):self.cache.delete((self.kind,*key))
    def __iter__(self):return iter((x,y) for kind,x,y in self.cache.records() if kind==self.kind)
    def __len__(self):return sum(k[0]==self.kind for k in self.cache.records())
    def __contains__(self,key):
        try:self.cache.version((self.kind,*key));return True
        except KeyError:return False
    def token(self,key):
        try:return self.cache.version((self.kind,*key))
        except KeyError:return None
    def totals(self):
        rows=[v for k,v in self.cache.records().items() if k[0]==self.kind]
        return sum(v.nbytes for v in rows),[v.total for v in rows]


class FrozenTiles(Mapping):
    def __init__(self,records):self.records=records
    def __getitem__(self,key):return self.records[key].read()
    def __iter__(self):return iter(self.records)
    def __len__(self):return len(self.records)
    def token(self,key):return self.records.get(key)
    def totals(self):return sum(v.nbytes for v in self.records.values()),[v.total for v in self.records.values()]


class OverlayTiles(MutableMapping):
    """A private preview: only changed blocks allocate arrays; base stays frozen."""
    def __init__(self,base):self.base=base;self.changed={};self.removed=set()
    def __getitem__(self,key):
        if key in self.removed:raise KeyError(key)
        return self.changed[key] if key in self.changed else self.base[key]
    def __setitem__(self,key,value):
        array=np.array(value,dtype=np.float64,copy=True);array.flags.writeable=False
        self.changed[key]=array;self.removed.discard(key)
    def __delitem__(self,key):
        if key not in self:raise KeyError(key)
        self.changed.pop(key,None);self.removed.add(key)
    def __iter__(self):return iter(dict.fromkeys([*(k for k in self.base if k not in self.removed),*self.changed]))
    def __len__(self):return sum(1 for _ in self)
    def __contains__(self,key):return key not in self.removed and (key in self.changed or key in self.base)
    def token(self,key):
        if key in self.removed:return None
        return id(self.changed[key]) if key in self.changed else self.base.token(key)
    def totals(self):
        arrays=[self[k] for k in self.changed]
        records=[self.base.token(k) for k in self.base if k not in self.removed and k not in self.changed]
        return (sum(a.nbytes for a in arrays)+sum(v.nbytes for v in records),
            [float(a.sum(dtype=np.float64)) for a in arrays]+[v.total for v in records])
