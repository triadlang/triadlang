from __future__ import annotations

import builtins as _builtins
import cmath
import ctypes
import math
import random as _py_random
import struct as _struct

_bsum = _builtins.sum
_bmax = _builtins.max
_bmin = _builtins.min
_ball = _builtins.all
_bany = _builtins.any
_babs = _builtins.abs
_bround = _builtins.round


class _DType:
    __slots__ = ('name', 'kind', 'itemsize', 'type')

    def __init__(self, name, kind, itemsize, type):
        self.name = name
        self.kind = kind
        self.itemsize = itemsize
        self.type = type

    def __repr__(self):
        return f'dtype({self.name!r})'

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, _DType):
            return self.name == other.name
        if isinstance(other, str):
            try:
                return self is dtype(other)
            except (TypeError, ValueError):
                return False
        if isinstance(other, type):
            return self.type is other
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def __call__(self, v=0):
        if isinstance(v, ndarray):
            return v.astype(self)
        return _cast_value(v, self)


bool_ = _DType('bool', 'b', 1, bool)
int8 = _DType('int8', 'i', 1, int)
int16 = _DType('int16', 'i', 2, int)
int32 = _DType('int32', 'i', 4, int)
int64 = _DType('int64', 'i', 8, int)
uint8 = _DType('uint8', 'u', 1, int)
uint16 = _DType('uint16', 'u', 2, int)
uint32 = _DType('uint32', 'u', 4, int)
uint64 = _DType('uint64', 'u', 8, int)
float16 = _DType('float16', 'f', 2, float)
float32 = _DType('float32', 'f', 4, float)
float64 = _DType('float64', 'f', 8, float)
complex64 = _DType('complex64', 'c', 8, complex)
complex128 = _DType('complex128', 'c', 16, complex)

int_ = int64
float_ = float64
complex_ = complex128
single = float32
double = float64
byte = int8
short = int16
longlong = int64
ubyte = uint8

_DTYPE_BY_NAME = {
    'bool': bool_, '?': bool_,
    'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64,
    'uint8': uint8, 'uint16': uint16, 'uint32': uint32, 'uint64': uint64,
    'float16': float16, 'float32': float32, 'float64': float64,
    'complex64': complex64, 'complex128': complex128,
    'f4': float32, 'f8': float64, 'c8': complex64, 'c16': complex128,
    'i1': int8, 'i2': int16, 'i4': int32, 'i8': int64,
    'u1': uint8, 'u2': uint16, 'u4': uint32, 'u8': uint64,
}


def dtype(d=None):
    if d is None:
        return float64
    if isinstance(d, _DType):
        return d
    if d is bool:
        return bool_
    if d is int:
        return int64
    if d is float:
        return float64
    if d is complex:
        return complex128
    if isinstance(d, str):
        key = d.strip('<>=|')
        if key in _DTYPE_BY_NAME:
            return _DTYPE_BY_NAME[key]
        raise TypeError(f'data type {d!r} not understood')
    raise TypeError(f'data type {d!r} not understood')


def _resolve_dtype(given, default):
    if given is None:
        return default
    if isinstance(given, _DType):
        return given
    return dtype(given)


def _pop_dtype(dt, kw):
    if dt is None:
        dt = kw.get('dtype', None)
    return dt


def _scalar_dtype(v):
    if isinstance(v, bool):
        return bool_
    if isinstance(v, int):
        return int64
    if isinstance(v, float):
        return float64
    if isinstance(v, complex):
        return complex128
    return None


def _promote(a, b):
    if a is None:
        return b
    if b is None:
        return a
    order = (bool_, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
             float16, float32, float64, complex64, complex128)
    ia = order.index(a) if a in order else 0
    ib = order.index(b) if b in order else 0
    if a.kind == 'c' or b.kind == 'c':
        return complex128 if (a is complex128 or b is complex128) else complex64
    if a.kind == 'f' or b.kind == 'f':
        if a is float16 and b is float16:
            return float16
        if a in (float16, float32) and b in (float16, float32):
            return float32
        return float64
    if a.kind == 'u' and b.kind == 'u':
        return order[_bmax(ia, ib)]
    if a.kind == 'b':
        return b
    if b.kind == 'b':
        return a
    return int64 if (a.kind in 'iu' or b.kind in 'iu') else order[_bmax(ia, ib)]


def _wrap_int(v, dt):
    bits = dt.itemsize * 8
    try:
        iv = int(v.real) if isinstance(v, complex) else int(v)
    except (OverflowError, ValueError):
        iv = 0
    if dt.kind == 'u':
        return iv % (1 << bits)
    mod = iv % (1 << bits)
    if mod >= (1 << (bits - 1)):
        mod -= 1 << bits
    return mod


def _cast_value(v, dt):
    if dt.kind == 'b':
        if isinstance(v, complex):
            return v.real != 0 or v.imag != 0
        if isinstance(v, str):
            return v != '' and v != '0'
        return bool(v)
    if dt.kind == 'c':
        return complex(v)
    if dt.kind == 'f':
        if isinstance(v, complex):
            return float(v.real)
        try:
            return float(v)
        except (OverflowError, ValueError):
            return float('inf') if (v or 0) > 0 else float('-inf')
    if dt.kind in 'iu':
        return _wrap_int(v, dt)
    if isinstance(v, bytes):
        return v
    return v


def _unwrap(v):
    if isinstance(v, ndarray):
        if v.size == 1:
            return v._data[0]
        raise TypeError('only size-1 arrays can be converted to Python scalars')
    return v


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _c_strides(shape):
    strides = []
    acc = 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= s
    return tuple(reversed(strides))


def _broadcast_shapes(sa, sb):
    ra = list(reversed(sa))
    rb = list(reversed(sb))
    out = []
    for i in range(_bmax(len(ra), len(rb))):
        da = ra[i] if i < len(ra) else 1
        db = rb[i] if i < len(rb) else 1
        if da == db:
            out.append(da)
        elif da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        else:
            raise ValueError(f'operands could not be broadcast together with shapes {sa} {sb}')
    return tuple(reversed(out))


def _broadcast_to_data(data, shape, target):
    if shape == target:
        return list(data)
    off_a = len(target) - len(shape)
    strides = _c_strides(shape)
    out = []
    for idx in range(_numel(target)):
        rem = idx
        src = 0
        for dim in range(len(target) - 1, -1, -1):
            pos = rem % target[dim]
            rem //= target[dim]
            ai = dim - off_a
            if ai >= 0 and shape[ai] != 1:
                src += pos * strides[ai]
        out.append(data[src])
    return out


def _normalize_index(key, shape):
    if not isinstance(key, tuple):
        key = (key,)
    ei = -1
    for i, k in enumerate(key):
        if k is Ellipsis:
            ei = i
            break
    if ei >= 0:
        consuming = 0
        for k in key:
            if k is not Ellipsis and k is not None:
                consuming += 1
        fill = len(shape) - consuming
        if fill < 0:
            fill = 0
        key = key[:ei] + (slice(None),) * fill + key[ei + 1:]
    insert = []
    kept = 0
    stripped = []
    for k in key:
        if k is None:
            insert.append(kept)
        else:
            stripped.append(k)
            if not isinstance(k, int):
                kept += 1
    out = list(stripped)
    while len(out) < len(shape):
        out.append(slice(None))
    return tuple(out), tuple(insert)


def _apply_newaxis(arr, positions, ndim_before):
    for k, p in enumerate(sorted(positions)):
        q = p + k
        arr = _expand_dims_impl(arr, q if q <= arr.ndim else arr.ndim)
    return arr


def _adv_shape(k, n):
    if isinstance(k, ndarray):
        if k.dtype.kind == 'b':
            return None
        return k._shape
    if isinstance(k, list):
        return (len(k),)
    return None


def _is_paired(key):
    shapes = []
    for k in key:
        if isinstance(k, (int, slice)):
            continue
        if isinstance(k, ndarray) and k.dtype.kind == 'b':
            return False
        s = _adv_shape(k, 0)
        shapes.append(s)
    return len(shapes) >= 2 and _ball(s == shapes[0] for s in shapes)


class ndarray:
    def __init__(self, data, shape, dtype):
        self._data = data
        self._shape = tuple(shape)
        self.dtype = dtype

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return _numel(self._shape)

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def strides(self):
        st = _c_strides(self._shape)
        return tuple(s * self.dtype.itemsize for s in st)

    @property
    def T(self):
        if self.ndim < 2:
            return self
        return self.transpose()

    @property
    def real(self):
        if self.dtype.kind == 'c':
            return ndarray([complex(v).real for v in self._data], self._shape, float64)
        return self.copy()

    @property
    def imag(self):
        if self.dtype.kind == 'c':
            return ndarray([complex(v).imag for v in self._data], self._shape, float64)
        return zeros(self._shape, dtype=float64)

    def __len__(self):
        if not self._shape:
            raise TypeError('len() of unsized object')
        return self._shape[0]

    def __iter__(self):
        if not self._shape:
            raise TypeError('iteration over a 0-d array')
        for i in range(self._shape[0]):
            yield self[i]

    def __bool__(self):
        if self.size == 1:
            return bool(self._data[0])
        raise ValueError('truth value of an array with more than one element is ambiguous')

    def _select(self, key):
        norm, newaxis_pos = _normalize_index(key, self._shape)
        if _bany(isinstance(k, ndarray) and k.dtype.kind == 'b' for k in norm):
            res = self._mask_select(norm)
        elif _bany(isinstance(k, (ndarray, list)) for k in norm):
            res = self._fancy_select(norm)
        else:
            res = self._basic_select(norm)
        if newaxis_pos:
            if not isinstance(res, ndarray):
                res = ndarray([res], (), self.dtype)
            res = _apply_newaxis(res, newaxis_pos, len(norm))
        return res

    def _basic_select(self, key):
        shape = self._shape
        strides = _c_strides(shape)
        out_shape = []
        for dim, k in enumerate(key):
            n = shape[dim]
            if isinstance(k, int):
                idx = k + n if k < 0 else k
                if idx < 0 or idx >= n:
                    raise IndexError('index out of bounds')
            elif isinstance(k, slice):
                start, stop, step = k.indices(n)
                ln = len(range(start, stop, step))
                out_shape.append(ln)
            else:
                raise IndexError(f'invalid index type {type(k)}')
        def rec(dim, base):
            if dim == len(shape):
                return base, True
            k = key[dim]
            n = shape[dim]
            if isinstance(k, int):
                idx = k + n if k < 0 else k
                return rec(dim + 1, base + idx * strides[dim])
            start, stop, step = k.indices(n)
            vals = []
            for pos in range(start, stop, step):
                v, _ = rec(dim + 1, base + pos * strides[dim])
                vals.append(v)
            return vals, False
        vals, leaf = rec(0, 0)
        if leaf:
            return self._data[vals]
        idxs = []

        def gather(v):
            if isinstance(v, list):
                for x in v:
                    gather(x)
            else:
                idxs.append(v)
        gather(vals)
        if not out_shape:
            return self._data[idxs[0]]
        return ndarray([self._data[i] for i in idxs],
                       tuple(out_shape), self.dtype)

    def _mask_select(self, key):
        masks = [k if isinstance(k, ndarray) and k.dtype.kind == 'b' else None
                 for k in key]
        if len(key) == 1 and masks[0] is not None and masks[0].shape == self._shape:
            return ndarray([v for v, m in zip(self._data, masks[0]._data) if m],
                           (_bsum(1 for m in masks[0]._data if m),), self.dtype)
        base = self
        for dim, m in enumerate(masks):
            if m is None:
                continue
            if m.shape != (base._shape[dim],):
                raise IndexError('boolean index did not match indexed array')
            idx = [i for i, f in enumerate(m._data) if f]
            sub = [base._take_axis(dim, i) for i in idx]
            if not sub:
                new_shape = base._shape[:dim] + (0,) + base._shape[dim + 1:]
                return ndarray([], new_shape, base.dtype)
            base = _concat_impl(sub, axis=dim)
        return base

    def _take_axis(self, axis, i):
        sub = self.take([i], axis=axis)
        return _squeeze_axis(sub, axis)

    def _fancy_select(self, key):
        adv = [(i, k) for i, k in enumerate(key)
               if isinstance(k, ndarray) and k.dtype.kind != 'b']
        if (len(adv) == 1 and adv[0][0] == 0
                and not _bany(isinstance(k, list) for k in key)):
            idx = adv[0][1]
            rest_full = True
            for j in range(1, len(key)):
                k = key[j]
                if not isinstance(k, slice) or k.indices(self._shape[j]) != (0, self._shape[j], 1):
                    rest_full = False
                    break
            if rest_full:
                n = self._shape[0]
                flat = []
                for v in idx._data:
                    flat.extend(self._take_axis(0, int(v) + n if int(v) < 0 else int(v))._data)
                return ndarray(flat, idx._shape + self._shape[1:], self.dtype)
        if _is_paired(key) and not _bany(isinstance(k, slice) for k in key):
            return self._paired_select(key)
        idx_lists = []
        for dim, k in enumerate(key):
            n = self._shape[dim]
            if isinstance(k, ndarray):
                if k.dtype.kind == 'b':
                    idx_lists.append([i for i, f in enumerate(k._data) if f])
                else:
                    idx_lists.append([int(v) + n if int(v) < 0 else int(v)
                                      for v in k._data])
            elif isinstance(k, list):
                idx_lists.append([int(v) + n if int(v) < 0 else int(v) for v in k])
            elif isinstance(k, int):
                idx_lists.append(k + n if k < 0 else k)
            elif isinstance(k, slice):
                start, stop, step = k.indices(n)
                idx_lists.append(list(range(start, stop, step)))
            else:
                raise IndexError(f'invalid index type {type(k)}')
        return self._orthogonal_select(idx_lists)

    def _orthogonal_select(self, idx_lists):
        result = []
        strides = _c_strides(self._shape)

        def rec(dim, base):
            if dim == len(self._shape):
                result.append(self._data[base])
                return
            sel = idx_lists[dim]
            if isinstance(sel, int):
                rec(dim + 1, base + sel * strides[dim])
            else:
                for i in sel:
                    rec(dim + 1, base + i * strides[dim])
        rec(0, 0)
        shape = tuple(len(v) for v in idx_lists if isinstance(v, list))
        if not shape:
            return result[0]
        return ndarray(result, shape, self.dtype)

    def __getitem__(self, key):
        if key is None:
            return self.expand_dims(0)
        return self._select(key)

    def __setitem__(self, key, value):
        if key is None:
            raise IndexError('cannot setitem with None')
        norm, _ = _normalize_index(key, self._shape)
        if _bany(isinstance(k, ndarray) and k.dtype.kind == 'b' for k in norm):
            self._mask_setitem(norm, value)
            return
        if _bany(isinstance(k, (ndarray, list)) for k in norm):
            self._fancy_setitem(norm, value)
            return
        shape = self._shape
        strides = _c_strides(shape)
        if isinstance(value, ndarray):
            vdata = _broadcast_to_data(value._data, value._shape,
                                       tuple(len(range(*k.indices(shape[d])))
                                             if isinstance(k, slice) else 1
                                             for d, k in enumerate(norm)))
        else:
            vdata = None

        def rec(dim, base, vi):
            if dim == len(shape):
                if vdata is not None:
                    self._data[base] = _cast_value(vdata[vi[0]], self.dtype)
                    vi[0] += 1
                else:
                    self._data[base] = _cast_value(value, self.dtype)
                return
            k = norm[dim]
            n = shape[dim]
            if isinstance(k, int):
                idx = k + n if k < 0 else k
                rec(dim + 1, base + idx * strides[dim], vi)
            else:
                start, stop, step = k.indices(n)
                for pos in range(start, stop, step):
                    rec(dim + 1, base + pos * strides[dim], vi)
        rec(0, 0, [0])

    def _mask_setitem(self, key, value):
        if len(key) == 1 and isinstance(key[0], ndarray) and key[0].dtype.kind == 'b':
            m = key[0]
            if m.shape != self._shape:
                raise IndexError('boolean index did not match')
            if isinstance(value, ndarray):
                vals = value._data
            else:
                vals = [value] * self.size
            vi = 0
            for i, f in enumerate(m._data):
                if f:
                    v = vals[vi] if vi < len(vals) else vals[-1]
                    self._data[i] = _cast_value(v, self.dtype)
                    vi += 1
            return
        for dim, m in enumerate(key):
            if isinstance(m, ndarray) and m.dtype.kind == 'b':
                idx = [i for i, f in enumerate(m._data) if f]
                break
        else:
            raise IndexError('boolean index did not match')
        if isinstance(value, ndarray):
            vals = value._data
        else:
            vals = [value] * len(idx)
        strides = _c_strides(self._shape)
        for ii, i in enumerate(idx):
            v = vals[ii] if ii < len(vals) else vals[-1]
            self._data[i * strides[dim]] = _cast_value(v, self.dtype)

    def _paired_select(self, key):
        pairs = []
        for dim, k in enumerate(key):
            n = self._shape[dim]
            if isinstance(k, ndarray):
                pairs.append([int(v) + n if int(v) < 0 else int(v) for v in k._data])
            elif isinstance(k, list):
                pairs.append([int(v) + n if int(v) < 0 else int(v) for v in k])
            else:
                pairs.append(None)
        shape = _adv_shape(key[[isinstance(k, (ndarray, list)) for k in key].index(True)], 0)
        strides = _c_strides(self._shape)
        out = []
        for p in range(_numel(shape)):
            off = 0
            for dim, k in enumerate(key):
                if pairs[dim] is None:
                    if isinstance(k, slice):
                        raise IndexError('paired indexing with slices is not supported')
                    if isinstance(k, int):
                        kk = k + self._shape[dim] if k < 0 else k
                        off += kk * strides[dim]
                else:
                    off += pairs[dim][p] * strides[dim]
            out.append(self._data[off])
        return ndarray(out, shape, self.dtype)

    def _paired_setitem(self, key, value):
        pairs = []
        for dim, k in enumerate(key):
            n = self._shape[dim]
            if isinstance(k, ndarray):
                pairs.append([int(v) + n if int(v) < 0 else int(v) for v in k._data])
            elif isinstance(k, list):
                pairs.append([int(v) + n if int(v) < 0 else int(v) for v in k])
            else:
                pairs.append(None)
        shape = _adv_shape(key[[isinstance(k, (ndarray, list)) for k in key].index(True)], 0)
        strides = _c_strides(self._shape)
        if isinstance(value, ndarray):
            vals = value._data
        else:
            vals = [value] * _numel(shape)
        for p in range(_numel(shape)):
            off = 0
            for dim, k in enumerate(key):
                if pairs[dim] is None:
                    kk = k + self._shape[dim] if isinstance(k, int) and k < 0 else k
                    off += kk * strides[dim]
                else:
                    off += pairs[dim][p] * strides[dim]
            v = vals[p] if p < len(vals) else vals[-1]
            self._data[off] = _cast_value(v, self.dtype)

    def _fancy_setitem(self, key, value):
        if _is_paired(key):
            self._paired_setitem(key, value)
            return
        idx_lists = []
        for dim, k in enumerate(key):
            n = self._shape[dim]
            if isinstance(k, ndarray):
                if k.dtype.kind == 'b':
                    idx_lists.append([i for i, f in enumerate(k._data) if f])
                else:
                    idx_lists.append([int(v) + n if int(v) < 0 else int(v)
                                      for v in k._data])
            elif isinstance(k, list):
                idx_lists.append([int(v) + n if int(v) < 0 else int(v) for v in k])
            elif isinstance(k, int):
                idx_lists.append(k + n if k < 0 else k)
            elif isinstance(k, slice):
                start, stop, step = k.indices(n)
                idx_lists.append(list(range(start, stop, step)))
        if isinstance(value, ndarray):
            vals = value._data
        else:
            vals = [value]
        strides = _c_strides(self._shape)
        vi = [0]

        def rec(dim, base):
            if dim == len(self._shape):
                v = vals[vi[0]] if vi[0] < len(vals) else vals[-1]
                self._data[base] = _cast_value(v, self.dtype)
                vi[0] += 1
                return
            sel = idx_lists[dim]
            if isinstance(sel, int):
                rec(dim + 1, base + sel * strides[dim])
            else:
                for i in sel:
                    rec(dim + 1, base + i * strides[dim])
        rec(0, 0)

    def copy(self):
        return ndarray(list(self._data), self._shape, self.dtype)

    def astype(self, dt, copy=True):
        dt = dtype(dt)
        return ndarray([_cast_value(v, dt) for v in self._data], self._shape, dt)

    def view(self, dt):
        dt = dtype(dt)
        if dt is self.dtype or dt.name == self.dtype.name:
            return ndarray(list(self._data), self._shape, dt)
        raw = self.tobytes()
        if len(raw) % dt.itemsize:
            raise ValueError(f'cannot view {self.dtype} as {dt}: size mismatch')
        arr = frombuffer(raw, dtype=dt)
        if not self._shape:
            return arr.reshape(())
        last = self._shape[-1] * self.dtype.itemsize // dt.itemsize
        return arr.reshape(self._shape[:-1] + (last,))

    def tolist(self):
        def nest(data, shape):
            if not shape:
                return data[0]
            n = shape[0]
            step = _numel(shape[1:]) if len(shape) > 1 else 1
            return [nest(data[i * step:(i + 1) * step], shape[1:])
                    for i in range(n)]
        if not self._shape:
            return self._data[0]
        return nest(self._data, self._shape)

    def item(self, *args):
        if args:
            return self[args if len(args) > 1 else args[0]]
        if self.size == 1:
            return self._data[0]
        raise ValueError('can only convert an array of size 1 to a Python scalar')

    def fill(self, value):
        v = _cast_value(value, self.dtype)
        for i in range(len(self._data)):
            self._data[i] = v

    def flatten(self):
        return ndarray(list(self._data), (self.size,), self.dtype)

    def ravel(self):
        return self.flatten()

    def reshape(self, *shape):
        return _reshape_impl(self, shape)

    def transpose(self, *axes):
        return _transpose_impl(self, axes)

    def swapaxes(self, axis1, axis2):
        ndim = self.ndim
        a1 = axis1 + ndim if axis1 < 0 else axis1
        a2 = axis2 + ndim if axis2 < 0 else axis2
        axes = list(range(ndim))
        axes[a1], axes[a2] = axes[a2], axes[a1]
        return self.transpose(*axes)

    def squeeze(self, axis=None):
        return _squeeze_impl(self, axis)

    def expand_dims(self, axis):
        return _expand_dims_impl(self, axis)

    def take(self, indices, axis=None):
        return _take_impl(self, indices, axis)

    def sum(self, axis=None, keepdims=False, dtype=None):
        return _reduce_impl(self, 'sum', axis, keepdims, dtype)

    def mean(self, axis=None, keepdims=False, dtype=None):
        return _reduce_impl(self, 'mean', axis, keepdims, dtype)

    def var(self, axis=None, keepdims=False, ddof=0):
        return var(self, axis=axis, keepdims=keepdims, ddof=ddof)

    def std(self, axis=None, dtype=None, ddof=0, keepdims=False):
        out = std(self, axis=axis, keepdims=keepdims, ddof=ddof)
        if dtype is not None and isinstance(out, ndarray):
            return out.astype(dtype)
        return out

    def max(self, axis=None, keepdims=False):
        return _reduce_impl(self, 'max', axis, keepdims, None)

    def min(self, axis=None, keepdims=False):
        return _reduce_impl(self, 'min', axis, keepdims, None)

    def prod(self, axis=None, keepdims=False, dtype=None):
        return _reduce_impl(self, 'prod', axis, keepdims, dtype)

    def any(self, axis=None, keepdims=False):
        return _reduce_impl(self, 'any', axis, keepdims, None)

    def all(self, axis=None, keepdims=False):
        return _reduce_impl(self, 'all', axis, keepdims, None)

    def argmax(self, axis=None):
        return _arg_impl(self, 'max', axis)

    def argmin(self, axis=None):
        return _arg_impl(self, 'min', axis)

    def conj(self):
        if self.dtype.kind == 'c':
            return ndarray([complex(v).conjugate() for v in self._data],
                           self._shape, self.dtype)
        return self.copy()

    conjugate = conj

    def clip(self, lo=None, hi=None):
        return _clip_impl(self, lo, hi)

    def round(self, decimals=0):
        return _round_impl(self, decimals)

    def sort(self, axis=-1):
        return _sort_impl(self, axis)

    def argsort(self, axis=-1):
        return _argsort_impl(self, axis)

    def cumsum(self, axis=None):
        return _cumsum_impl(self, axis)

    def tobytes(self):
        if self.dtype.kind == 'c':
            comp = 'f' if self.dtype is complex64 else 'd'
            flat = []
            for v in self._data:
                c = complex(v)
                flat.extend([c.real, c.imag])
            return _struct.pack(f'<{len(flat)}{comp}', *flat)
        fmt = _struct_fmt(self.dtype)
        vals = [float(v) if self.dtype.kind == 'f' else (bool(v) if self.dtype.kind == 'b' else v)
                for v in self._data]
        return _struct.pack(f'<{len(vals)}{fmt}', *vals)

    def __repr__(self):
        return f'array({self.tolist()}, dtype={self.dtype.name})'

    def __float__(self):
        if self.size == 1:
            v = self._data[0]
            return float(v.real) if isinstance(v, complex) else float(v)
        raise TypeError('only size-1 arrays can be converted to Python scalars')

    def __int__(self):
        if self.size == 1:
            v = self._data[0]
            return int(v.real) if isinstance(v, complex) else int(v)
        raise TypeError('only size-1 arrays can be converted to Python scalars')

    def __complex__(self):
        if self.size == 1:
            return complex(self._data[0])
        raise TypeError('only size-1 arrays can be converted to Python scalars')

    def __format__(self, spec):
        if self._shape == ():
            v = self._data[0]
            return format(float(v.real) if isinstance(v, complex) else float(v), spec)
        if spec:
            raise TypeError('unsupported format string passed to ndarray.__format__')
        return str(self)

    def _binop(self, other, f, rop=None):
        if isinstance(other, ndarray):
            shape = _broadcast_shapes(self._shape, other._shape)
            a = _broadcast_to_data(self._data, self._shape, shape)
            b = _broadcast_to_data(other._data, other._shape, shape)
            dt = _promote(self.dtype, other.dtype)
            return ndarray([_cast_value(f(x, y), dt) for x, y in zip(a, b)],
                           shape, dt)
        dt = _promote(self.dtype, _scalar_dtype(other))
        return ndarray([_cast_value(f(x, other), dt) for x in self._data],
                       self._shape, dt)

    def __add__(self, o):
        return self._binop(o, lambda x, y: x + y)

    def __radd__(self, o):
        return self._binop(o, lambda x, y: y + x)

    def __sub__(self, o):
        return self._binop(o, lambda x, y: x - y)

    def __rsub__(self, o):
        return self._binop(o, lambda x, y: y - x)

    def __mul__(self, o):
        return self._binop(o, lambda x, y: x * y)

    def __rmul__(self, o):
        return self._binop(o, lambda x, y: y * x)

    def __truediv__(self, o):
        def div(x, y):
            if isinstance(x, complex) or isinstance(y, complex):
                return complex(x) / complex(y)
            if isinstance(x, float) or isinstance(y, float):
                return float(x) / float(y)
            return float(x) / float(y)
        r = self._binop(o, div)
        if r.dtype.kind not in 'fc':
            r = r.astype(float64)
        return r

    def __rtruediv__(self, o):
        return self._binop(o, lambda x, y: float(y) / float(x) if not isinstance(x, complex) and not isinstance(y, complex) else complex(y) / complex(x)).astype(complex128 if self.dtype.kind == 'c' else float64)

    def __floordiv__(self, o):
        return self._binop(o, lambda x, y: x // y)

    def __mod__(self, o):
        return self._binop(o, lambda x, y: x % y)

    def __pow__(self, o):
        return self._binop(o, lambda x, y: x ** y)

    def __rpow__(self, o):
        return self._binop(o, lambda x, y: y ** x)

    def __neg__(self):
        return ndarray([-v for v in self._data], self._shape, self.dtype)

    def __pos__(self):
        return self.copy()

    def __abs__(self):
        return _ufunc_impl(self, 'abs')

    def _cmp(self, o, f):
        if isinstance(o, ndarray):
            shape = _broadcast_shapes(self._shape, o._shape)
            a = _broadcast_to_data(self._data, self._shape, shape)
            b = _broadcast_to_data(o._data, o._shape, shape)
            return ndarray([bool(f(x, y)) for x, y in zip(a, b)], shape, bool_)
        return ndarray([bool(f(x, o)) for x in self._data], self._shape, bool_)

    def __eq__(self, o):
        return self._cmp(o, lambda x, y: x == y)

    def __ne__(self, o):
        return self._cmp(o, lambda x, y: x != y)

    def __lt__(self, o):
        return self._cmp(o, lambda x, y: x < y)

    def __le__(self, o):
        return self._cmp(o, lambda x, y: x <= y)

    def __gt__(self, o):
        return self._cmp(o, lambda x, y: x > y)

    def __ge__(self, o):
        return self._cmp(o, lambda x, y: x >= y)

    def __matmul__(self, o):
        return _matmul_impl(self, o)

    def __rmatmul__(self, o):
        return _matmul_impl(asarray(o), self)

    def _logic(self, o, f):
        if isinstance(o, ndarray):
            shape = _broadcast_shapes(self._shape, o._shape)
            a = _broadcast_to_data(self._data, self._shape, shape)
            b = _broadcast_to_data(o._data, o._shape, shape)
            return ndarray([bool(f(bool(x), bool(y))) for x, y in zip(a, b)],
                           shape, bool_)
        return ndarray([bool(f(bool(x), bool(o))) for x in self._data],
                       self._shape, bool_)

    def _bitwise(self, o, f):
        if isinstance(o, ndarray):
            if self.dtype.kind == 'b' and o.dtype.kind == 'b':
                return self._logic(o, f)
            if self.dtype.kind in 'iu' and o.dtype.kind in 'iu':
                shape = _broadcast_shapes(self._shape, o._shape)
                a = _broadcast_to_data(self._data, self._shape, shape)
                b = _broadcast_to_data(o._data, o._shape, shape)
                dt = _promote(self.dtype, o.dtype)
                return ndarray([_cast_value(f(int(x), int(y)), dt) for x, y in zip(a, b)],
                               shape, dt)
            raise TypeError(f'bitwise op not supported for dtypes {self.dtype} and {o.dtype}')
        if isinstance(o, bool) or self.dtype.kind == 'b':
            return self._logic(o, f)
        if self.dtype.kind in 'iu' and isinstance(o, int):
            return ndarray([_cast_value(f(int(x), o), self.dtype) for x in self._data],
                           self._shape, self.dtype)
        raise TypeError(f'bitwise op not supported for dtype {self.dtype}')

    def __and__(self, o):
        return self._bitwise(o, lambda x, y: x & y)

    def __rand__(self, o):
        return self._bitwise(o, lambda x, y: x & y)

    def __or__(self, o):
        return self._bitwise(o, lambda x, y: x | y)

    def __ror__(self, o):
        return self._bitwise(o, lambda x, y: x | y)

    def __xor__(self, o):
        return self._bitwise(o, lambda x, y: x ^ y)

    def __rxor__(self, o):
        return self._bitwise(o, lambda x, y: x ^ y)

    def __invert__(self):
        if self.dtype.kind == 'b':
            return ndarray([not bool(v) for v in self._data], self._shape, bool_)
        if self.dtype.kind in 'iu':
            return ndarray([_cast_value(~int(v), self.dtype) for v in self._data],
                           self._shape, self.dtype)
        raise TypeError(f'bitwise invert not supported for dtype {self.dtype}')

    def _shift(self, o, f):
        if isinstance(o, ndarray):
            if o.size != 1:
                raise ValueError('shift amount must be a scalar')
            o = o.item()
        if self.dtype.kind not in 'iu' or not isinstance(o, int):
            raise TypeError(f'shift not supported for dtype {self.dtype}')
        if isinstance(o, bool):
            raise TypeError(f'shift not supported for dtype {self.dtype}')
        return ndarray([_cast_value(f(int(x), o), self.dtype) for x in self._data],
                       self._shape, self.dtype)

    def __lshift__(self, o):
        return self._shift(o, lambda x, y: x << y)

    def __rlshift__(self, o):
        return self._shift(o, lambda x, y: x << y)

    def __rshift__(self, o):
        return self._shift(o, lambda x, y: x >> y)

    def __rrshift__(self, o):
        return self._shift(o, lambda x, y: x >> y)

    def __bytes__(self):
        return self.tobytes()


def _reshape_impl(a, shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    else:
        shape = tuple(int(s) for s in shape)
    if -1 in shape:
        known = 1
        for s in shape:
            if s != -1:
                known *= s
        missing = a.size // known
        shape = tuple(missing if s == -1 else s for s in shape)
    if _numel(shape) != a.size:
        raise ValueError(f'cannot reshape array of size {a.size} into shape {shape}')
    return ndarray(list(a._data), shape, a.dtype)


def _transpose_impl(a, axes):
    if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        axes = tuple(axes[0])
    else:
        axes = tuple(axes)
    if not axes:
        axes = tuple(reversed(range(a.ndim)))
    axes = tuple(ax + a.ndim if ax < 0 else ax for ax in axes)
    new_shape = tuple(a._shape[ax] for ax in axes)
    old_strides = _c_strides(a._shape)
    out = []
    for idx in range(_numel(new_shape)):
        rem = idx
        src = 0
        for dim in range(len(new_shape) - 1, -1, -1):
            pos = rem % new_shape[dim]
            rem //= new_shape[dim]
            src += pos * old_strides[axes[dim]]
        out.append(a._data[src])
    return ndarray(out, new_shape, a.dtype)


def _squeeze_impl(a, axis=None):
    if axis is None:
        shape = tuple(s for s in a._shape if s != 1)
    else:
        ax = axis + a.ndim if axis < 0 else axis
        if a._shape[ax] != 1:
            raise ValueError('cannot select an axis to squeeze out which has size not equal to one')
        shape = tuple(s for i, s in enumerate(a._shape) if i != ax)
    return ndarray(list(a._data), shape, a.dtype)


def _squeeze_axis(a, axis):
    ax = axis + a.ndim if axis < 0 else axis
    shape = tuple(s for i, s in enumerate(a._shape) if i != ax)
    return ndarray(list(a._data), shape, a.dtype)


def _expand_dims_impl(a, axis):
    ax = axis + a.ndim + 1 if axis < 0 else axis
    shape = list(a._shape)
    shape.insert(ax, 1)
    return ndarray(list(a._data), tuple(shape), a.dtype)


def _take_impl(a, indices, axis=None):
    single = False
    if isinstance(indices, ndarray):
        idx = [int(v) for v in indices._data]
    elif isinstance(indices, (list, tuple)):
        idx = [int(v) for v in indices]
    else:
        idx = [int(indices)]
        single = True
    if axis is None:
        flat = a._data
        n = len(flat)
        out = [flat[i + n if i < 0 else i] for i in idx]
        if single:
            return out[0]
        return ndarray(out, (len(out),), a.dtype)
    ax = axis + a.ndim if axis < 0 else axis
    n = a._shape[ax]
    idx = [i + n if i < 0 else i for i in idx]
    inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
    outer = _numel(a._shape[:ax])
    step = n * inner
    out = []
    for o in range(outer):
        base = o * step
        for i in idx:
            out.extend(a._data[base + i * inner:base + (i + 1) * inner])
    shape = a._shape[:ax] + (len(idx),) + a._shape[ax + 1:]
    res = ndarray(out, shape, a.dtype)
    if single:
        return _squeeze_axis(res, ax)
    return res


def _reduce_impl(a, op, axis=None, keepdims=False, dt=None):
    dt = dtype(dt) if dt is not None else None
    if axis is None:
        vals = list(a._data)
        r = _reduce_list(vals, op, a.dtype)
        rdt = dt or _reduce_dtype(op, a.dtype)
        v = _cast_value(r, rdt)
        return ndarray([v], (), rdt)
    if isinstance(axis, int):
        axes = (axis + a.ndim if axis < 0 else axis,)
    else:
        axes = tuple(ax + a.ndim if ax < 0 else ax for ax in axis)
    axes = tuple(sorted(axes))
    out_shape = []
    for i, s in enumerate(a._shape):
        if i in axes:
            if keepdims:
                out_shape.append(1)
        else:
            out_shape.append(s)
    out_shape = tuple(out_shape)
    inner_axes = set(axes)
    res = {}

    def rec(dim, base, key):
        if dim == a.ndim:
            res.setdefault(key, []).append(a._data[base])
            return
        strides = _c_strides(a._shape)
        for pos in range(a._shape[dim]):
            k2 = key if dim in inner_axes else key + (pos,)
            rec(dim + 1, base + pos * strides[dim], k2)
    rec(0, 0, ())
    rdt = dt or _reduce_dtype(op, a.dtype)
    order = sorted(res.keys())
    out = [_cast_value(_reduce_list(res[k], op, a.dtype), rdt) for k in order]
    if not out_shape:
        return ndarray(out, (), rdt)
    return ndarray(out, out_shape, rdt)


def _reduce_list(vals, op, dt):
    if op == 'sum':
        return _bsum(vals, 0)
    if op == 'prod':
        p = 1
        for v in vals:
            p *= v
        return p
    if op == 'mean':
        return _bsum(vals, 0) / len(vals)
    if op == 'max':
        m = vals[0]
        for v in vals[1:]:
            if v > m:
                m = v
        return m
    if op == 'min':
        m = vals[0]
        for v in vals[1:]:
            if v < m:
                m = v
        return m
    if op == 'any':
        return _bany(bool(v) for v in vals)
    if op == 'all':
        return _ball(bool(v) for v in vals)
    raise ValueError(f'unknown reduction {op}')


def _reduce_dtype(op, dt):
    if op in ('any', 'all'):
        return bool_
    if op in ('sum', 'prod', 'mean'):
        if dt.kind == 'c':
            return complex128
        if dt.kind == 'f':
            return float64
        if dt.kind == 'b':
            return int64 if op != 'mean' else float64
        return int64 if op != 'mean' else float64
    return dt


def _arg_impl(a, op, axis):
    if a.ndim == 0:
        return 0
    if axis is None:
        vals = list(a._data)
        best = 0
        for i in range(1, len(vals)):
            if (vals[i] > vals[best]) if op == 'max' else (vals[i] < vals[best]):
                best = i
        return best
    ax = axis + a.ndim if axis < 0 else axis
    inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
    outer = _numel(a._shape[:ax])
    n = a._shape[ax]
    step = n * inner
    out = []
    for o in range(outer):
        base = o * step
        for j in range(inner):
            best = 0
            bv = a._data[base + j]
            for i in range(1, n):
                v = a._data[base + i * inner + j]
                if (v > bv) if op == 'max' else (v < bv):
                    bv = v
                    best = i
            out.append(best)
    shape = a._shape[:ax] + a._shape[ax + 1:]
    return ndarray(out, shape, int64)


def _clip_impl(a, lo=None, hi=None):
    if isinstance(lo, ndarray):
        lo = _broadcast_to_data(lo._data, lo._shape, a._shape)
    if isinstance(hi, ndarray):
        hi = _broadcast_to_data(hi._data, hi._shape, a._shape)
    out = []
    for i, v in enumerate(a._data):
        l = lo[i] if isinstance(lo, list) else lo
        h = hi[i] if isinstance(hi, list) else hi
        if l is not None and v < l:
            v = l
        if h is not None and v > h:
            v = h
        out.append(_cast_value(v, a.dtype))
    return ndarray(out, a._shape, a.dtype)


def _round_impl(a, decimals=0):
    f = 10 ** decimals
    return ndarray([_cast_value(_bround(float(v), decimals) if a.dtype.kind == 'f' else _bround(float(v) / f) * f, a.dtype) for v in a._data],
                   a._shape, a.dtype)


def round(a, decimals=0):
    return _round_impl(asarray(a), decimals)


around = round


def _sort_impl(a, axis=-1):
    if a.ndim == 0:
        return a.copy()
    ax = axis + a.ndim if axis < 0 else axis
    inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
    outer = _numel(a._shape[:ax])
    n = a._shape[ax]
    step = n * inner
    out = list(a._data)
    for o in range(outer):
        base = o * step
        for j in range(inner):
            col = sorted(out[base + j + i * inner] for i in range(n))
            for i in range(n):
                out[base + j + i * inner] = col[i]
    return ndarray(out, a._shape, a.dtype)


def _argsort_impl(a, axis=-1):
    if a.ndim == 0:
        return zeros((), dtype=int64)
    ax = axis + a.ndim if axis < 0 else axis
    inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
    outer = _numel(a._shape[:ax])
    n = a._shape[ax]
    step = n * inner
    out = [0] * a.size
    for o in range(outer):
        base = o * step
        for j in range(inner):
            col = sorted(range(n), key=lambda i: a._data[base + j + i * inner])
            for i in range(n):
                out[base + j + i * inner] = col[i]
    return ndarray(out, a._shape, int64)


def _cumsum_impl(a, axis=None):
    if a.ndim == 0:
        return ndarray(list(a._data), (1,), a.dtype)
    if axis is None:
        acc = 0
        out = []
        for v in a._data:
            acc += v
            out.append(acc)
        return ndarray(out, (a.size,), _promote(a.dtype, int64 if a.dtype.kind in 'biu' else a.dtype))
    ax = axis + a.ndim if axis < 0 else axis
    inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
    outer = _numel(a._shape[:ax])
    n = a._shape[ax]
    step = n * inner
    out = list(a._data)
    for o in range(outer):
        base = o * step
        for j in range(inner):
            acc = 0
            for i in range(n):
                acc += out[base + j + i * inner]
                out[base + j + i * inner] = acc
    return ndarray(out, a._shape, _promote(a.dtype, int64 if a.dtype.kind in 'biu' else a.dtype))


def _nested_info(obj):
    if isinstance(obj, ndarray):
        return obj._shape, obj.dtype, True
    if isinstance(obj, (list, tuple)):
        if not obj:
            return (0,), None, False
        shapes = []
        dts = []
        for x in obj:
            s, d, _ = _nested_info(x)
            shapes.append(s)
            dts.append(d)
        first = shapes[0]
        if _bany(s != first for s in shapes):
            raise ValueError('inhomogeneous shape after conversion')
        dt = None
        for d in dts:
            dt = _promote(dt, d)
        return (len(obj),) + first, dt, False
    return (), _scalar_dtype(obj), False


def _nested_flat(obj, dt):
    if isinstance(obj, ndarray):
        return [_cast_value(v, dt) for v in
                _broadcast_to_data(obj._data, obj._shape, obj._shape)]
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(_nested_flat(x, dt))
        return out
    return [_cast_value(obj, dt)]


def array(obj, dt=None, copy=True, **kw):
    dt = _pop_dtype(dt, kw)
    gd = getattr(obj, '_data', None)
    if isinstance(gd, ndarray) and not isinstance(obj, ndarray):
        obj = gd
    if isinstance(obj, ndarray) and (dt is None or dt is obj.dtype
                                     or dt == obj.dtype):
        return obj.copy() if copy else obj
    shape, inferred, _ = _nested_info(obj)
    dt = _resolve_dtype(dt, inferred or float64)
    return ndarray(_nested_flat(obj, dt), shape, dt)


def asarray(obj, dtype=None):
    return array(obj, dtype=dtype, copy=False)


def ascontiguousarray(obj, dtype=None):
    return array(obj, dtype=dtype, copy=False)


def asanyarray(obj, dtype=None):
    return array(obj, dtype=dtype, copy=False)


def _full_impl(shape, value, dt):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)
    if isinstance(value, ndarray):
        data = _broadcast_to_data(value._data, value._shape, shape)
        return ndarray([_cast_value(v, dt) for v in data], shape, dt)
    if isinstance(value, (list, tuple)):
        return _full_impl(shape, asarray(value), dt)
    v = _cast_value(_unwrap(value), dt)
    return ndarray([v] * _numel(shape), shape, dt)


def zeros(shape, dt=None, **kw):
    return _full_impl(shape, 0, _resolve_dtype(_pop_dtype(dt, kw), float64))


def ones(shape, dt=None, **kw):
    return _full_impl(shape, 1, _resolve_dtype(_pop_dtype(dt, kw), float64))


def empty(shape, dt=None, **kw):
    return _full_impl(shape, 0, _resolve_dtype(_pop_dtype(dt, kw), float64))


def full(shape, value, dt=None, **kw):
    dt = _pop_dtype(dt, kw)
    if dt is None:
        if isinstance(value, ndarray):
            dt = value.dtype
        elif isinstance(value, (list, tuple)):
            dt = asarray(value).dtype
        else:
            dt = _scalar_dtype(_unwrap(value)) or float64
    else:
        dt = _resolve_dtype(dt, float64)
    return _full_impl(shape, value, dt)


triad = full


def zeros_like(a, dtype=None):
    a = asarray(a)
    return zeros(a._shape, dtype=dtype or a.dtype)


def ones_like(a, dtype=None):
    a = asarray(a)
    return ones(a._shape, dtype=dtype or a.dtype)


def empty_like(a, dtype=None):
    a = asarray(a)
    return zeros(a._shape, dtype=dtype or a.dtype)


def full_like(a, value, dtype=None):
    a = asarray(a)
    return full(a._shape, value, dtype=dtype or a.dtype)


triad_like = full_like


def arange(*args, dtype=None):
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise TypeError('arange expected 1-3 arguments')
    vals = []
    v = start
    if step > 0:
        while v < stop:
            vals.append(v)
            v += step
    elif step < 0:
        while v > stop:
            vals.append(v)
            v += step
    else:
        raise ValueError('arange step cannot be zero')
    if _bany(isinstance(x, float) for x in (start, stop, step)):
        dt = float64
    else:
        dt = int64
    dt = _resolve_dtype(dtype, dt)
    return ndarray([_cast_value(x, dt) for x in vals], (len(vals),), dt)


def linspace(start, stop, num=50, endpoint=True, dt=None, **kw):
    dt = _resolve_dtype(_pop_dtype(dt, kw), float64)
    num = int(num)
    if num <= 0:
        return ndarray([], (0,), dt)
    if num == 1:
        vals = [float(start)]
    else:
        if endpoint:
            step = (float(stop) - float(start)) / (num - 1)
        else:
            step = (float(stop) - float(start)) / num
        vals = [float(start) + step * i for i in range(num)]
    return ndarray([_cast_value(v, dt) for v in vals], (num,), dt)


def logspace(start, stop, num=50, endpoint=True, base=10.0, dt=None, **kw):
    e = linspace(start, stop, num, endpoint=endpoint)
    rdt = _resolve_dtype(_pop_dtype(dt, kw), float64)
    return ndarray([_cast_value(base ** float(v), rdt) for v in e._data],
                   e._shape, rdt)


def eye(n, m=None, k=0, dt=None, **kw):
    n = int(n)
    m = int(m) if m is not None else n
    dt = _resolve_dtype(_pop_dtype(dt, kw), float64)
    out = []
    for i in range(n):
        for j in range(m):
            out.append(_cast_value(1 if j - i == k else 0, dt))
    return ndarray(out, (n, m), dt)


identity = eye


def meshgrid(*xis, indexing='xy'):
    arrays = [asarray(x).flatten() for x in xis]
    if indexing == 'xy' and len(arrays) >= 2:
        arrays[0], arrays[1] = arrays[1], arrays[0]
        swapped = True
    else:
        swapped = False
    shape = tuple(a.size for a in arrays)
    outs = []
    for ax, a in enumerate(arrays):
        reps = list(shape)
        reps[ax] = 1
        tiled = a._data * 1
        grid = _tile_to(tiled, (a.size,), shape, ax)
        outs.append(ndarray(grid, shape, a.dtype))
    if swapped:
        outs[0], outs[1] = outs[1], outs[0]
    return outs


def _tile_to(data, src_shape, dst_shape, ax):
    strides = _c_strides(dst_shape)
    out = []
    for idx in range(_numel(dst_shape)):
        rem = idx
        pos = 0
        for dim in range(len(dst_shape) - 1, -1, -1):
            p = rem % dst_shape[dim]
            rem //= dst_shape[dim]
            if dim == ax:
                pos = p
        out.append(data[pos])
    return out


def _ufunc_impl(a, name, out_dt=None):
    a = asarray(a)
    _m = math
    _cm = cmath
    if name == 'abs':
        def f(v):
            return _babs(v)
        dt = a.dtype if a.dtype.kind in 'iuf' else (
            float64 if a.dtype.kind == 'c' else a.dtype)
    elif name == 'exp':
        def f(v):
            if isinstance(v, complex):
                return _cexp(v)
            return _m.exp(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'log':
        def f(v):
            return _cm.log(v) if isinstance(v, complex) else _m.log(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'log10':
        def f(v):
            return _cm.log10(v) if isinstance(v, complex) else _m.log10(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'sqrt':
        def f(v):
            if isinstance(v, complex):
                return v ** 0.5
            if v < 0:
                return complex(v) ** 0.5
            return _m.sqrt(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'square':
        def f(v):
            return v * v
        dt = a.dtype
    elif name == 'sin':
        def f(v):
            return _cm.sin(v) if isinstance(v, complex) else _m.sin(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'cos':
        def f(v):
            return _cm.cos(v) if isinstance(v, complex) else _m.cos(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'tan':
        def f(v):
            return _cm.tan(v) if isinstance(v, complex) else _m.tan(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'arcsin':
        def f(v):
            return _cm.asin(v) if isinstance(v, complex) else _m.asin(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'arccos':
        def f(v):
            return _cm.acos(v) if isinstance(v, complex) else _m.acos(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'arctan':
        def f(v):
            return _cm.atan(v) if isinstance(v, complex) else _m.atan(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'sinh':
        def f(v):
            return _cm.sinh(v) if isinstance(v, complex) else _m.sinh(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'cosh':
        def f(v):
            return _cm.cosh(v) if isinstance(v, complex) else _m.cosh(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'tanh':
        def f(v):
            return _cm.tanh(v) if isinstance(v, complex) else _m.tanh(v)
        dt = complex128 if a.dtype.kind == 'c' else float64
    elif name == 'sign':
        def f(v):
            r = v.real if isinstance(v, complex) else v
            return (1 if r > 0 else (-1 if r < 0 else 0))
        dt = a.dtype if a.dtype.kind in 'iuf' else float64
    elif name == 'conj':
        def f(v):
            return v.conjugate() if isinstance(v, complex) else v
        dt = a.dtype
    elif name == 'real':
        def f(v):
            return v.real if isinstance(v, complex) else v
        dt = float64 if a.dtype.kind == 'c' else a.dtype
    elif name == 'imag':
        def f(v):
            return v.imag if isinstance(v, complex) else 0.0
        dt = float64
    elif name == 'angle':
        def f(v):
            return _m.atan2(v.imag, v.real) if isinstance(v, complex) else (_m.pi if v < 0 else 0.0)
        dt = float64
    elif name == 'isnan':
        def f(v):
            return v != v
        dt = bool_
    elif name == 'isinf':
        def f(v):
            return v in (inf, -inf)
        dt = bool_
    elif name == 'isfinite':
        def f(v):
            return v == v and v not in (inf, -inf)
        dt = bool_
    elif name == 'ceil':
        def f(v):
            return _m.ceil(v)
        dt = a.dtype
    elif name == 'floor':
        def f(v):
            return _m.floor(v)
        dt = a.dtype
    else:
        raise ValueError(f'unknown ufunc {name}')
    if out_dt is not None:
        dt = out_dt
    return ndarray([_cast_value(f(v), dt) for v in a._data], a._shape, dt)


def _cexp(v):
    import cmath as _cm
    return _cm.exp(v)


def abs(a):
    return _ufunc_impl(a, 'abs')


def exp(a):
    return _ufunc_impl(a, 'exp')


def log(a):
    return _ufunc_impl(a, 'log')


def log10(a):
    return _ufunc_impl(a, 'log10')


def sqrt(a):
    return _ufunc_impl(a, 'sqrt')


def square(a):
    return _ufunc_impl(a, 'square')


def sin(a):
    return _ufunc_impl(a, 'sin')


def cos(a):
    return _ufunc_impl(a, 'cos')


def tan(a):
    return _ufunc_impl(a, 'tan')


def arcsin(a):
    return _ufunc_impl(a, 'arcsin')


def arcsinh(a):
    a = asarray(a)
    return ndarray([_cast_value(math.log(v + math.sqrt(v * v + 1)), float64) for v in a._data],
                   a._shape, float64)


def arccos(a):
    return _ufunc_impl(a, 'arccos')


def arctan(a):
    return _ufunc_impl(a, 'arctan')


def sinh(a):
    return _ufunc_impl(a, 'sinh')


def cosh(a):
    return _ufunc_impl(a, 'cosh')


def tanh(a):
    return _ufunc_impl(a, 'tanh')


def sign(a):
    return _ufunc_impl(a, 'sign')


def conj(a):
    return _ufunc_impl(a, 'conj')


conjugate = conj


def real(a):
    return _ufunc_impl(a, 'real')


def imag(a):
    return _ufunc_impl(a, 'imag')


def angle(a):
    return _ufunc_impl(a, 'angle')


def isnan(a):
    return _ufunc_impl(a, 'isnan')


def isinf(a):
    return _ufunc_impl(a, 'isinf')


def isfinite(a):
    return _ufunc_impl(a, 'isfinite')


def ceil(a):
    return _ufunc_impl(a, 'ceil')


def floor(a):
    return _ufunc_impl(a, 'floor')


class _AddUfunc:
    def __call__(self, a, b):
        a, b = asarray(a), asarray(b)
        shape = _broadcast_shapes(a._shape, b._shape)
        da = _broadcast_to_data(a._data, a._shape, shape)
        db = _broadcast_to_data(b._data, b._shape, shape)
        dt = _promote(a.dtype, b.dtype)
        return ndarray([_cast_value(x + y, dt) for x, y in zip(da, db)], shape, dt)

    def at(self, a, indices, values):
        if isinstance(indices, ndarray):
            idx = [int(v) for v in indices._data]
        elif isinstance(indices, (list, tuple)):
            idx = [int(v) for v in indices]
        else:
            idx = [int(indices)]
        if a.ndim > 1 and len(idx) > 0:
            n0 = a._shape[0]
            row = _numel(a._shape[1:])
            if isinstance(values, ndarray) and values._shape == (len(idx), row):
                vrows = [values._data[i * row:(i + 1) * row] for i in range(len(idx))]
            elif isinstance(values, ndarray) and values._shape == (row,):
                vrows = [values._data] * len(idx)
            else:
                vrows = [[values]] * len(idx)
            for ii, i in enumerate(idx):
                r = i + n0 if i < 0 else i
                base = r * row
                for j in range(row):
                    v = vrows[ii][j] if j < len(vrows[ii]) else vrows[ii][-1]
                    a._data[base + j] = _cast_value(a._data[base + j] + v, a.dtype)
            return
        if isinstance(values, ndarray):
            vals = values._data
        else:
            vals = [values] * len(idx)
        n = a.size
        for ii, i in enumerate(idx):
            v = vals[ii] if ii < len(vals) else vals[-1]
            pos = i + n if i < 0 else i
            a._data[pos] = _cast_value(a._data[pos] + v, a.dtype)


add = _AddUfunc()


def maximum(a, b):
    a, b = asarray(a), asarray(b)
    shape = _broadcast_shapes(a._shape, b._shape)
    da = _broadcast_to_data(a._data, a._shape, shape)
    db = _broadcast_to_data(b._data, b._shape, shape)
    dt = _promote(a.dtype, b.dtype)
    return ndarray([_cast_value(x if x >= y else y, dt) for x, y in zip(da, db)],
                   shape, dt)


def minimum(a, b):
    a, b = asarray(a), asarray(b)
    shape = _broadcast_shapes(a._shape, b._shape)
    da = _broadcast_to_data(a._data, a._shape, shape)
    db = _broadcast_to_data(b._data, b._shape, shape)
    dt = _promote(a.dtype, b.dtype)
    return ndarray([_cast_value(x if x <= y else y, dt) for x, y in zip(da, db)],
                   shape, dt)


def clip(a, lo=None, hi=None, a_min=None, a_max=None):
    if lo is None:
        lo = a_min
    if hi is None:
        hi = a_max
    return _clip_impl(asarray(a), lo, hi)


def where(cond, x=None, y=None):
    cond = asarray(cond)
    if x is None and y is None:
        idx = tuple([i for i, v in enumerate(cond._data) if v])
        return (ndarray(list(idx), (len(idx),), int64),)
    xs = _broadcast_to_data(asarray(x)._data, asarray(x)._shape, cond._shape) if not isinstance(x, (int, float, complex)) else [x] * cond.size
    ys = _broadcast_to_data(asarray(y)._data, asarray(y)._shape, cond._shape) if not isinstance(y, (int, float, complex)) else [y] * cond.size
    dt = _promote(_scalar_dtype(x) if not isinstance(x, ndarray) else asarray(x).dtype,
                  _scalar_dtype(y) if not isinstance(y, ndarray) else asarray(y).dtype)
    return ndarray([_cast_value(xx if c else yy, dt) for c, xx, yy in zip(cond._data, xs, ys)],
                   cond._shape, dt)


def sum(a, axis=None, keepdims=False, dtype=None):
    return _reduce_impl(asarray(a), 'sum', axis, keepdims, dtype)


def mean(a, axis=None, keepdims=False, dtype=None):
    return _reduce_impl(asarray(a), 'mean', axis, keepdims, dtype)


def prod(a, axis=None, keepdims=False, dtype=None):
    return _reduce_impl(asarray(a), 'prod', axis, keepdims, dtype)


def max(a, axis=None, keepdims=False):
    return _reduce_impl(asarray(a), 'max', axis, keepdims, None)


amax = max


def min(a, axis=None, keepdims=False):
    return _reduce_impl(asarray(a), 'min', axis, keepdims, None)


amin = min


def any(a, axis=None, keepdims=False):
    return _reduce_impl(asarray(a), 'any', axis, keepdims, None)


def all(a, axis=None, keepdims=False):
    return _reduce_impl(asarray(a), 'all', axis, keepdims, None)


def argmax(a, axis=None):
    return _arg_impl(asarray(a), 'max', axis)


def argmin(a, axis=None):
    return _arg_impl(asarray(a), 'min', axis)


def _var_ddof_scale(a, axis, ddof):
    if not ddof:
        return 1.0
    if axis is None:
        n = a.size
    else:
        ax = (axis,) if isinstance(axis, int) else tuple(axis)
        ax = tuple(x + a.ndim if x < 0 else x for x in ax)
        n = 1
        for x in ax:
            n *= a.shape[x]
    return n / (n - ddof) if n - ddof > 0 else float('nan')


def var(a, axis=None, keepdims=False, ddof=0):
    a = asarray(a)
    m = _reduce_impl(a, 'mean', axis, keepdims=True, dt=None)
    d = a - m if keepdims or axis is None else a - _reduce_impl(a, 'mean', axis, keepdims=False)
    if axis is None:
        vals = [(float(v.real) if isinstance(v, complex) else float(v)) ** 2 for v in d._data]
        base = ndarray([_bsum(vals, 0) / len(vals)], (), float64)
    else:
        base = _reduce_impl(d * d.conj() if d.dtype.kind == 'c' else d * d, 'mean', axis, keepdims, float64)
    return base * _var_ddof_scale(a, axis, ddof)


def std(a, axis=None, keepdims=False, ddof=0):
    v = var(a, axis=axis, keepdims=keepdims, ddof=ddof)
    if isinstance(v, ndarray):
        return ndarray([math.sqrt(float(x)) for x in v._data], v._shape, float64)
    return math.sqrt(float(v))


def _median_of_sorted(col):
    n = len(col)
    if n == 0:
        return 0.0
    return col[n // 2] if n % 2 else (col[n // 2 - 1] + col[n // 2]) / 2

def median(a, axis=None, keepdims=False):
    a = asarray(a)
    if axis is None:
        return ndarray([_cast_value(_median_of_sorted(sorted(a._data)), float64)], (), float64)
    ax = int(axis)
    if ax < 0:
        ax += a.ndim
    if not 0 <= ax < a.ndim:
        raise ValueError(f'median axis {axis} out of bounds')
    n_ax = a._shape[ax]
    outer_shape = a._shape[:ax] + a._shape[ax + 1:]
    strides = _c_strides(a._shape)
    out = []
    total = _numel(outer_shape) if outer_shape else 1
    for idx in range(total):
        rem = idx
        base = 0
        odims = list(outer_shape)
        for dim in range(len(odims) - 1, -1, -1):
            p = rem % odims[dim]
            rem //= odims[dim]
            adim = dim if dim < ax else dim + 1
            base += p * strides[adim]
        col = sorted(a._data[base + i * strides[ax]] for i in range(n_ax))
        out.append(_cast_value(_median_of_sorted(col), float64))
    if keepdims:
        rshape = a._shape[:ax] + (1,) + a._shape[ax + 1:]
    else:
        rshape = outer_shape
    return ndarray(out, rshape, float64)


def percentile(a, q, axis=None):
    a = asarray(a)
    s = sorted(float(v) for v in a._data)
    n = len(s)
    qs = [q] if not isinstance(q, (list, tuple, ndarray)) else list(q)
    out = []
    for qi in qs:
        pos = float(qi) / 100 * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        out.append(s[lo] + (s[hi] - s[lo]) * (pos - lo))
    if isinstance(q, (list, tuple, ndarray)):
        return ndarray(out, (len(out),), float64)
    return ndarray(out, (), float64)


def sort(a, axis=-1):
    return _sort_impl(asarray(a), axis)


def argsort(a, axis=-1):
    return _argsort_impl(asarray(a), axis)


def searchsorted(a, v, side='left'):
    a = asarray(a).flatten()._data
    if isinstance(v, ndarray):
        vs = v._data
        vshape = v._shape
    elif isinstance(v, (list, tuple)):
        vs = list(v)
        vshape = (len(vs),)
    else:
        vs = [v]
        vshape = ()
    out = []
    for x in vs:
        lo, hi = 0, len(a)
        if side == 'left':
            while lo < hi:
                mid = (lo + hi) // 2
                if a[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if a[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
        out.append(lo)
    if vshape == ():
        return out[0]
    return ndarray(out, vshape, int64)


def cumsum(a, axis=None):
    return _cumsum_impl(asarray(a), axis)


def allclose(a, b, rtol=1e-05, atol=1e-08):
    a, b = asarray(a), asarray(b)
    shape = _broadcast_shapes(a._shape, b._shape)
    da = _broadcast_to_data(a._data, a._shape, shape)
    db = _broadcast_to_data(b._data, b._shape, shape)
    for x, y in zip(da, db):
        if _babs(complex(x) - complex(y)) > atol + rtol * _babs(complex(y)):
            return False
    return True


def isclose(a, b, rtol=1e-05, atol=1e-08):
    a, b = asarray(a), asarray(b)
    shape = _broadcast_shapes(a._shape, b._shape)
    da = _broadcast_to_data(a._data, a._shape, shape)
    db = _broadcast_to_data(b._data, b._shape, shape)
    return ndarray([_babs(complex(x) - complex(y)) <= atol + rtol * _babs(complex(y))
                    for x, y in zip(da, db)], shape, bool_)


def array_equal(a, b):
    a, b = asarray(a), asarray(b)
    return a._shape == b._shape and _ball(x == y for x, y in zip(a._data, b._data))


def _matmul_impl(a, b):
    a, b = asarray(a), asarray(b)
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError('matmul: inputs must have at least 1 dimension')
    if a.ndim == 1 and b.ndim == 1:
        if a.size != b.size:
            raise ValueError('matmul: mismatched lengths')
        return ndarray([_bsum((x * y for x, y in zip(a._data, b._data)), 0)], (),
                       _promote(a.dtype, b.dtype))
    if a.ndim == 1:
        r = _matmul_impl(a.reshape(1, a.size), b)
        return r.reshape(r._shape[1:])
    if b.ndim == 1:
        r = _matmul_impl(a, b.reshape(b.size, 1))
        return r.reshape(r._shape[:-1])
    if a._shape[-1] != b._shape[-2]:
        raise ValueError(f'matmul: shapes {a._shape} and {b._shape} not aligned')
    batch = _broadcast_shapes(a._shape[:-2], b._shape[:-2])
    m, k, n = a._shape[-2], a._shape[-1], b._shape[-1]
    da = _broadcast_to_data(a._data, a._shape[:-2], batch)
    db = _broadcast_to_data(b._data, b._shape[:-2], batch)
    sa, sb = _numel(a._shape[-2:]), _numel(b._shape[-2:])
    nb = _numel(batch)
    ca = [da[i * sa:(i + 1) * sa] for i in range(nb)]
    cb = [db[i * sb:(i + 1) * sb] for i in range(nb)]
    dt = _promote(a.dtype, b.dtype)
    out = []
    for ba, bb in zip(ca, cb):
        for i in range(m):
            for j in range(n):
                s = 0
                for p in range(k):
                    s += ba[i * k + p] * bb[p * n + j]
                out.append(_cast_value(s, dt))
    return ndarray(out, batch + (m, n), dt)


def matmul(a, b):
    return _matmul_impl(asarray(a), asarray(b))


def dot(a, b):
    a, b = asarray(a), asarray(b)
    if a.ndim == 0 or b.ndim == 0:
        return (a * b).reshape(())
    if a.ndim == 1 and b.ndim == 1:
        return _matmul_impl(a, b)
    if a.ndim == 2 and b.ndim == 2:
        return _matmul_impl(a, b)
    if a.ndim >= 1 and b.ndim >= 1 and a.ndim + b.ndim > 2:
        return tensordot(a, b, axes=1)
    return _matmul_impl(a, b)


def tensordot(a, b, axes=2):
    a, b = asarray(a), asarray(b)
    if isinstance(axes, int):
        ax_a = list(range(a.ndim - axes, a.ndim))
        ax_b = list(range(axes))
    else:
        ax_a, ax_b = axes
        ax_a = [x + a.ndim if x < 0 else x for x in ax_a]
        ax_b = [x + b.ndim if x < 0 else x for x in ax_b]
    rest_a = [d for d in range(a.ndim) if d not in ax_a]
    rest_b = [d for d in range(b.ndim) if d not in ax_b]
    pa = rest_a + ax_a
    pb = ax_b + rest_b
    ta = _transpose_impl(a, tuple(pa))
    tb = _transpose_impl(b, tuple(pb))
    na = _numel(tuple(ta._shape[d] for d in range(len(rest_a))))
    ka = _numel(tuple(ta._shape[d] for d in range(len(rest_a), ta.ndim)))
    kb = _numel(tuple(tb._shape[d] for d in range(len(ax_b))))
    nb = _numel(tuple(tb._shape[d] for d in range(len(ax_b), tb.ndim)))
    if ka != kb:
        raise ValueError('tensordot: contracted dimensions must match')
    dt = _promote(a.dtype, b.dtype)
    out = []
    for i in range(na):
        for j in range(nb):
            s = 0
            for p in range(ka):
                s += ta._data[i * ka + p] * tb._data[p * nb + j]
            out.append(_cast_value(s, dt))
    shape = tuple(ta._shape[d] for d in range(len(rest_a))) + tuple(tb._shape[d] for d in range(len(ax_b), tb.ndim))
    return ndarray(out, shape, dt)


def outer(a, b):
    a = asarray(a).flatten()
    b = asarray(b).flatten()
    dt = _promote(a.dtype, b.dtype)
    return ndarray([_cast_value(x * y, dt) for x in a._data for y in b._data],
                   (a.size, b.size), dt)


def inner(a, b):
    a, b = asarray(a), asarray(b)
    return tensordot(a, b, axes=[[-1], [-1]])


def vdot(a, b):
    a = asarray(a).flatten()
    b = asarray(b).flatten()
    s = 0
    for x, y in zip(a._data, b._data):
        s += (x.conjugate() if isinstance(x, complex) else x) * y
    return s


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    a = asarray(a)
    b = asarray(b)
    if axis is not None:
        axisa = axisc = axisb = axis
    ta = _moveaxis_impl(a, axisa, -1)
    tb = _moveaxis_impl(b, axisb, -1)
    if ta._shape[-1] not in (2, 3) or tb._shape[-1] not in (2, 3):
        raise ValueError('cross: last dimension must be 2 or 3')
    n = _numel(ta._shape[:-1])
    dt = _promote(a.dtype, b.dtype)
    out = []
    for i in range(n):
        x = ta._data[i * 3:(i + 1) * 3] if ta._shape[-1] == 3 else ta._data[i * 2:(i + 1) * 2] + [0]
        y = tb._data[i * 3:(i + 1) * 3] if tb._shape[-1] == 3 else tb._data[i * 2:(i + 1) * 2] + [0]
        out.extend([_cast_value(x[1] * y[2] - x[2] * y[1], dt),
                    _cast_value(x[2] * y[0] - x[0] * y[2], dt),
                    _cast_value(x[0] * y[1] - x[1] * y[0], dt)])
    r = ndarray(out, ta._shape[:-1] + (3,), dt)
    return _moveaxis_impl(r, -1, axisc)


def _moveaxis_impl(a, src, dst):
    src = src + a.ndim if src < 0 else src
    dst = dst + a.ndim if dst < 0 else dst
    order = list(range(a.ndim))
    order.insert(dst, order.pop(src))
    return _transpose_impl(a, tuple(order))


def moveaxis(a, src, dst):
    return _moveaxis_impl(asarray(a), src, dst)


def swapaxes(a, axis1, axis2):
    a = asarray(a)
    ax1 = axis1 + a.ndim if axis1 < 0 else axis1
    ax2 = axis2 + a.ndim if axis2 < 0 else axis2
    order = list(range(a.ndim))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return _transpose_impl(a, tuple(order))


def transpose(a, axes=None):
    a = asarray(a)
    if axes is None:
        return _transpose_impl(a, ())
    return _transpose_impl(a, tuple(axes))


def reshape(a, shape):
    a = asarray(a)
    if isinstance(shape, int):
        shape = (shape,)
    return _reshape_impl(a, tuple(shape))


def squeeze(a, axis=None):
    return _squeeze_impl(asarray(a), axis)


def expand_dims(a, axis):
    return _expand_dims_impl(asarray(a), axis)


def ravel(a):
    return asarray(a).flatten()


def flatten(a):
    return asarray(a).flatten()


def _concat_impl(arrays, axis=0):
    arrays = [asarray(x) for x in arrays]
    if not arrays:
        raise ValueError('concatenate of empty list')
    ax = axis + arrays[0].ndim if axis < 0 else axis
    base = list(arrays[0]._shape)
    for x in arrays[1:]:
        if len(x._shape) != len(base) or _bany(s != b for i, (s, b) in enumerate(zip(x._shape, base)) if i != ax):
            raise ValueError('all input arrays must have the same shape except in the concatenation axis')
        base[ax] += x._shape[ax]
    inner = _numel(base[ax + 1:]) if ax + 1 < len(base) else 1
    outer = _numel(base[:ax])
    dt = arrays[0].dtype
    for x in arrays[1:]:
        dt = _promote(dt, x.dtype)
    out = []
    for o in range(outer):
        for x in arrays:
            n = x._shape[ax]
            xinner = _numel(x._shape[ax + 1:]) if ax + 1 < x.ndim else 1
            src = x._data[o * n * xinner:(o + 1) * n * xinner]
            out.extend([_cast_value(v, dt) for v in src])
    return ndarray(out, tuple(base), dt)


def concatenate(arrays, axis=0):
    return _concat_impl(list(arrays), axis)


def stack(arrays, axis=0):
    arrays = [asarray(x) for x in arrays]
    if not arrays:
        raise ValueError('stack of empty list')
    shape = arrays[0]._shape
    for x in arrays[1:]:
        if x._shape != shape:
            raise ValueError('all input arrays must have the same shape to stack')
    ax = axis + len(shape) + 1 if axis < 0 else axis
    new_shape = list(shape)
    new_shape.insert(ax, len(arrays))
    dt = arrays[0].dtype
    for x in arrays[1:]:
        dt = _promote(dt, x.dtype)
    inner = _numel(shape[ax:]) if ax < len(shape) else 1
    outer = _numel(shape[:ax])
    out = []
    for o in range(outer):
        for x in arrays:
            out.extend(x._data[o * inner:(o + 1) * inner])
    return ndarray([_cast_value(v, dt) for v in out], tuple(new_shape), dt)


def vstack(arrays):
    arrays = [asarray(x) for x in arrays]
    if arrays[0].ndim == 1:
        return _concat_impl([x.reshape(1, x.size) for x in arrays], axis=0)
    return _concat_impl(arrays, axis=0)


def hstack(arrays):
    arrays = [asarray(x) for x in arrays]
    if arrays[0].ndim == 1:
        return _concat_impl(arrays, axis=0)
    return _concat_impl(arrays, axis=1)


def dstack(arrays):
    arrays = [asarray(x) for x in arrays]
    reshaped = []
    for x in arrays:
        if x.ndim == 1:
            reshaped.append(x.reshape(1, x.size, 1))
        elif x.ndim == 2:
            reshaped.append(x.reshape(x._shape + (1,)))
        else:
            reshaped.append(x)
    return _concat_impl(reshaped, axis=2)


def array_split(a, sections, axis=0):
    a = asarray(a)
    ax = axis + a.ndim if axis < 0 else axis
    n = a._shape[ax]
    if isinstance(sections, int):
        q, r = divmod(n, sections)
        bounds = [0]
        for i in range(sections):
            bounds.append(bounds[-1] + q + (1 if i < r else 0))
    else:
        bounds = [0] + [int(v) for v in sections] + [n]
    out = []
    for i in range(len(bounds) - 1):
        sl = [slice(None)] * a.ndim
        sl[ax] = slice(bounds[i], bounds[i + 1])
        out.append(a[tuple(sl)])
    return out


def histogram(a, bins=10, range=None):
    a = asarray(a).flatten()._data
    if isinstance(bins, int):
        if not a:
            edges = linspace(0, 1, bins + 1)._data
        else:
            lo = _bmin(a) if range is None else range[0]
            hi = _bmax(a) if range is None else range[1]
            edges = linspace(lo, hi, bins + 1)._data
    else:
        edges = list(asarray(bins).flatten()._data)
        bins = len(edges) - 1
    counts = [0] * bins
    for v in a:
        if v < edges[0] or v > edges[-1]:
            continue
        lo, hi = 0, bins
        while lo < hi:
            mid = (lo + hi) // 2
            if edges[mid + 1] <= v:
                lo = mid + 1
            else:
                hi = mid
        if lo >= bins:
            lo = bins - 1
        counts[lo] += 1
    return (ndarray(counts, (bins,), int64),
            ndarray(list(edges), (len(edges),), float64))


def interp(x, xp, fp, left=None, right=None):
    scalar = not isinstance(x, (ndarray, list, tuple))
    xs = [x] if scalar else (x._data if isinstance(x, ndarray) else list(x))
    px = list(asarray(xp).flatten()._data)
    pf = list(asarray(fp).flatten()._data)
    out = []
    for v in xs:
        if v <= px[0]:
            out.append(pf[0] if left is None else left)
            continue
        if v >= px[-1]:
            out.append(pf[-1] if right is None else right)
            continue
        lo, hi = 0, len(px) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if px[mid] <= v:
                lo = mid
            else:
                hi = mid
        t = (v - px[lo]) / (px[hi] - px[lo]) if px[hi] != px[lo] else 0.0
        out.append(pf[lo] + t * (pf[hi] - pf[lo]))
    if scalar:
        return out[0]
    return ndarray(out, (len(out),), float64)


def log2(a):
    a = asarray(a)
    return ndarray([_cast_value(math.log(float(v), 2), float64) for v in a._data],
                   a._shape, float64)


def log1p(a):
    a = asarray(a)
    return ndarray([_cast_value(math.log1p(float(v)), float64) for v in a._data],
                   a._shape, float64)


def arctan2(y, x):
    y, x = asarray(y), asarray(x)
    shape = _broadcast_shapes(y._shape, x._shape)
    dy = _broadcast_to_data(y._data, y._shape, shape)
    dx = _broadcast_to_data(x._data, x._shape, shape)
    return ndarray([math.atan2(float(b), float(a)) for a, b in zip(dx, dy)],
                   shape, float64)


def isneginf(a):
    a = asarray(a)
    return ndarray([v == float('-inf') for v in a._data], a._shape, bool_)


def isposinf(a):
    a = asarray(a)
    return ndarray([v == float('inf') for v in a._data], a._shape, bool_)


def iscomplexobj(x):
    if isinstance(x, ndarray):
        return x.dtype.kind == 'c'
    if isinstance(x, (list, tuple)):
        return _bany(isinstance(v, complex) for v in x)
    return isinstance(x, complex)


def argwhere(a):
    a = asarray(a)
    idx = [i for i, v in enumerate(a._data) if v]
    strides = _c_strides(a._shape)
    out = []
    for flat in idx:
        rem = flat
        multi = []
        for dim in range(a.ndim - 1, -1, -1):
            multi.append(rem % a._shape[dim])
            rem //= a._shape[dim]
        out.extend(reversed(multi))
    n = len(idx)
    return ndarray(out, (n, a.ndim), int64)


def flatnonzero(a):
    a = asarray(a)
    return ndarray([i for i, v in enumerate(a._data) if v],
                   (_bsum(1 for v in a._data if v),), int64)


def unravel_index(indices, shape):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)
    single = not isinstance(indices, (ndarray, list, tuple))
    idx = [indices] if single else (list(indices._data) if isinstance(indices, ndarray) else list(indices))
    out = []
    for flat in idx:
        rem = int(flat)
        multi = []
        for dim in range(len(shape) - 1, -1, -1):
            multi.append(rem % shape[dim])
            rem //= shape[dim]
        out.extend(reversed(multi))
    if single:
        return tuple(out)
    n = len(idx)
    cols = []
    for d in range(len(shape)):
        cols.append(ndarray([out[i * len(shape) + d] for i in range(n)], (n,), int64))
    return tuple(cols)


def unique(a, return_index=False, return_inverse=False, return_counts=False):
    a = asarray(a)
    seen = []
    first = {}
    for i, v in enumerate(a._data):
        key = (v, type(v).__name__)
        if key not in first:
            first[key] = i
            seen.append(v)
    try:
        order = sorted(range(len(seen)), key=lambda i: seen[i])
    except TypeError:
        order = sorted(range(len(seen)),
                       key=lambda i: (seen[i].real, seen[i].imag)
                       if isinstance(seen[i], complex) else (float(seen[i]), 0.0))
    vals = [seen[i] for i in order]
    dt = a.dtype
    res = (ndarray(list(vals), (len(vals),), dt),)
    if return_index:
        res += (ndarray([first[(seen[i], type(seen[i]).__name__)] for i in order],
                        (len(order),), int64),)
    if return_inverse:
        inv = {id_: pos for pos, id_ in enumerate(order)}
        pos_of = {}
        for i, v in enumerate(seen):
            pos_of[(v, type(v).__name__)] = inv[i]
        res += (ndarray([pos_of[(v, type(v).__name__)] for v in a._data],
                        a._shape, int64),)
    if return_counts:
        counts = [0] * len(seen)
        for v in a._data:
            for i, s in enumerate(seen):
                if s == v and type(s) is type(v):
                    counts[order.index(i)] += 1
                    break
        res += (ndarray(counts, (len(counts),), int64),)
    if len(res) == 1:
        return res[0]
    return res


def digitize(x, bins, right=False):
    scalar = not isinstance(x, (ndarray, list, tuple))
    xs = [x] if scalar else (list(x._data) if isinstance(x, ndarray) else list(x))
    edges = list(asarray(bins).flatten()._data)
    out = []
    for v in xs:
        lo, hi = 0, len(edges)
        if right:
            while lo < hi:
                mid = (lo + hi) // 2
                if edges[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if edges[mid] <= v:
                    lo = mid + 1
                else:
                    hi = mid
        out.append(lo)
    if scalar:
        return out[0]
    shape = x._shape if isinstance(x, ndarray) else (len(out),)
    return ndarray(out, shape, int64)


def gradient(f, *varargs):
    a = asarray(f)
    if a.ndim == 0:
        raise ValueError('gradient of 0-d array')
    if len(varargs) == 0:
        hs = [1.0] * a.ndim
    elif len(varargs) == 1 and not isinstance(varargs[0], (ndarray, list, tuple)):
        hs = [float(varargs[0])] * a.ndim
    else:
        hs = []
        for v in varargs:
            if isinstance(v, (ndarray, list, tuple)):
                vv = list(v._data) if isinstance(v, ndarray) else list(v)
                hs.append(vv)
            else:
                hs.append(float(v))
        while len(hs) < a.ndim:
            hs.append(hs[-1])
    grads = []
    for ax in range(a.ndim):
        n = a._shape[ax]
        inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
        outer = _numel(a._shape[:ax])
        step = n * inner
        out = [0.0] * a.size
        h = hs[ax] if ax < len(hs) else 1.0
        for o in range(outer):
            base = o * step
            for j in range(inner):
                col = [float(a._data[base + j + i * inner]) for i in range(n)]
                if isinstance(h, list):
                    hh = h
                    g = [(col[1] - col[0]) / (hh[1] - hh[0])]
                    for i in range(1, n - 1):
                        g.append((col[i + 1] - col[i - 1]) / (hh[i + 1] - hh[i - 1]))
                    g.append((col[-1] - col[-2]) / (hh[-1] - hh[-2]))
                else:
                    if n == 1:
                        g = [0.0]
                    else:
                        g = [(col[1] - col[0]) / h]
                        for i in range(1, n - 1):
                            g.append((col[i + 1] - col[i - 1]) / (2 * h))
                        g.append((col[-1] - col[-2]) / h)
                for i in range(n):
                    out[base + j + i * inner] = g[i]
        grads.append(ndarray(out, a._shape, float64))
    if a.ndim == 1:
        return grads[0]
    return grads


def gcd(a, b):
    a, b = asarray(a), asarray(b)
    shape = _broadcast_shapes(a._shape, b._shape)
    da = _broadcast_to_data(a._data, a._shape, shape)
    db = _broadcast_to_data(b._data, b._shape, shape)
    return ndarray([math.gcd(int(x), int(y)) for x, y in zip(da, db)], shape, int64)


def corrcoef(x, y=None):
    if y is None:
        m = asarray(x)
    else:
        x = asarray(x).flatten()
        y = asarray(y).flatten()
        m = stack([x, y], axis=0)
    if m.ndim == 1:
        m = m.reshape(1, m.size)
    n = m._shape[0]
    out = zeros((n, n))
    for i in range(n):
        for j in range(n):
            xi = [float(v) for v in m._data[i * m._shape[1]:(i + 1) * m._shape[1]]]
            xj = [float(v) for v in m._data[j * m._shape[1]:(j + 1) * m._shape[1]]]
            mi = _bsum(xi) / len(xi)
            mj = _bsum(xj) / len(xj)
            cov = _bsum((a - mi) * (b - mj) for a, b in zip(xi, xj))
            si = math.sqrt(_bsum((a - mi) ** 2 for a in xi))
            sj = math.sqrt(_bsum((b - mj) ** 2 for b in xj))
            out._data[i * n + j] = cov / (si * sj) if si and sj else 0.0
    return out


def argpartition(a, kth, axis=-1):
    a = asarray(a)
    if a.ndim <= 1:
        order = sorted(range(a.size), key=lambda i: a._data[i])
        return ndarray(order, (a.size,), int64)
    ax = axis + a.ndim if axis < 0 else axis
    inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
    outer = _numel(a._shape[:ax])
    n = a._shape[ax]
    step = n * inner
    out = [0] * a.size
    for o in range(outer):
        base = o * step
        for j in range(inner):
            col = sorted(range(n), key=lambda i: a._data[base + j + i * inner])
            for i in range(n):
                out[base + j + i * inner] = col[i]
    return ndarray(out, a._shape, int64)


def memmap(path, dt=None, mode='r', offset=0, shape=None, **kw):
    dt = _resolve_dtype(_pop_dtype(dt, kw), uint8)
    if mode in ('w+', 'w'):
        if shape is None:
            raise ValueError('memmap write mode requires shape')
        if isinstance(shape, int):
            shape = (shape,)
        return zeros(tuple(shape), dtype=dt)
    with open(path, 'rb') as f:
        f.seek(int(offset))
        raw = f.read()
    arr = frombuffer(raw, dtype=dt)
    if shape is not None:
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(int(s) for s in shape)
        return arr.reshape(shape).copy()
    return arr


class NpzFile(dict):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass


def _npy_blob(arr):
    arr = asarray(arr)
    header = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (
        _npy_descr(arr.dtype), repr(arr._shape))
    header += ' ' * (64 - (len(header) + 10) % 64 - 1) + '\n'
    return (b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little')
            + header.encode('latin1') + arr.tobytes())


def savez(file, *args, **kw):
    import zipfile as _zf
    if args:
        kw.update({f'arr_{i}': v for i, v in enumerate(args)})
    if hasattr(file, 'write'):
        zf = _zf.ZipFile(file, 'w', _zf.ZIP_STORED)
        own = False
    else:
        zf = _zf.ZipFile(file, 'w', _zf.ZIP_STORED)
        own = True
    try:
        for k, v in kw.items():
            zf.writestr(f'{k}.npy', _npy_blob(v))
    finally:
        if own:
            zf.close()
    return file if not own else None


def savez_compressed(file, *args, **kw):
    import zipfile as _zf
    if args:
        kw.update({f'arr_{i}': v for i, v in enumerate(args)})
    if hasattr(file, 'write'):
        zf = _zf.ZipFile(file, 'w', _zf.ZIP_DEFLATED)
        own = False
    else:
        zf = _zf.ZipFile(file, 'w', _zf.ZIP_DEFLATED)
        own = True
    try:
        for k, v in kw.items():
            zf.writestr(f'{k}.npy', _npy_blob(v))
    finally:
        if own:
            zf.close()
    return file if not own else None


def _load_npz(blob):
    import io as _io
    import zipfile as _zf
    zf = _zf.ZipFile(_io.BytesIO(blob))
    out = NpzFile()
    for name in zf.namelist():
        if not name.endswith('.npy'):
            continue
        raw = zf.read(name)
        hlen = int.from_bytes(raw[8:10], 'little')
        header = raw[10:10 + hlen].decode('latin1')
        info = {}
        for part in _split_top_level(header.strip().strip('{}'), ','):
            part = part.strip()
            if not part or ':' not in part:
                continue
            k, v = part.split(':', 1)
            info[k.strip().strip("'\"")] = v.strip()
        dt = _npy_from_descr(info['descr'].strip("'\""))
        shape = _parse_shape(info['shape'])
        data = raw[10 + hlen:]
        n = _numel(shape)
        if dt in (complex64, complex128):
            comp = 'f' if dt is complex64 else 'd'
            sz = _struct.calcsize(comp)
            vals = _struct.unpack(f'<{n * 2}{comp}', data[:n * 2 * sz])
            arr = ndarray([complex(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)],
                          tuple(shape), dt)
        else:
            fmt = _struct_fmt(dt)
            sz = _struct.calcsize(fmt)
            vals = _struct.unpack(f'<{n}{fmt}', data[:n * sz])
            arr = ndarray([_cast_value(v, dt) for v in vals], tuple(shape), dt)
        out[name[:-4]] = arr
    return out


def split(a, indices_or_sections, axis=0):
    a = asarray(a)
    ax = axis + a.ndim if axis < 0 else axis
    n = a._shape[ax]
    if isinstance(indices_or_sections, int):
        q = n // indices_or_sections
        bounds = [i * q for i in range(indices_or_sections + 1)]
        bounds[-1] = n
    else:
        bounds = [0] + [int(v) for v in indices_or_sections] + [n]
    out = []
    for i in range(len(bounds) - 1):
        sl = [slice(None)] * a.ndim
        sl[ax] = slice(bounds[i], bounds[i + 1])
        out.append(a[tuple(sl)])
    return out


def take(a, indices, axis=None):
    return _take_impl(asarray(a), indices, axis)


def triu(a, k=0):
    a = asarray(a)
    if a.ndim < 2:
        raise ValueError('triu requires at least 2 dimensions')
    m, n = a._shape[-2], a._shape[-1]
    inner = m * n
    outer = a.size // inner
    out = []
    for o in range(outer):
        base = o * inner
        for i in range(m):
            for j in range(n):
                out.append(a._data[base + i * n + j] if j - i >= k else _cast_value(0, a.dtype))
    return ndarray(out, a._shape, a.dtype)


def tril(a, k=0):
    a = asarray(a)
    if a.ndim < 2:
        raise ValueError('tril requires at least 2 dimensions')
    m, n = a._shape[-2], a._shape[-1]
    inner = m * n
    outer = a.size // inner
    out = []
    for o in range(outer):
        base = o * inner
        for i in range(m):
            for j in range(n):
                out.append(a._data[base + i * n + j] if j - i <= k else _cast_value(0, a.dtype))
    return ndarray(out, a._shape, a.dtype)


def triu_indices(n, k=0, m=None):
    n = int(n)
    m = int(m) if m is not None else n
    ri, ci = [], []
    for i in range(n):
        for j in range(m):
            if j - i >= k:
                ri.append(i)
                ci.append(j)
    return (ndarray(ri, (len(ri),), int64), ndarray(ci, (len(ci),), int64))


def tril_indices(n, k=0, m=None):
    n = int(n)
    m = int(m) if m is not None else n
    ri, ci = [], []
    for i in range(n):
        for j in range(m):
            if j - i <= k:
                ri.append(i)
                ci.append(j)
    return (ndarray(ri, (len(ri),), int64), ndarray(ci, (len(ci),), int64))


def diag(v, k=0):
    v = asarray(v)
    if v.ndim == 1:
        n = v.size + _babs(k)
        dt = v.dtype
        out = [_cast_value(0, dt)] * (n * n)
        for i, val in enumerate(v._data):
            r = i if k >= 0 else i - k
            c = i + k if k >= 0 else i
            out[r * n + c] = val
        return ndarray(out, (n, n), dt)
    m = _bmin(v._shape[0], v._shape[1])
    out = []
    for i in range(m):
        r, c = (i, i + k) if k >= 0 else (i - k, i)
        if 0 <= r < v._shape[0] and 0 <= c < v._shape[1]:
            strides = _c_strides(v._shape)
            out.append(v._data[r * strides[0] + c * strides[1]])
    return ndarray(out, (len(out),), v.dtype)


def trace(a):
    a = asarray(a)
    return diag(a).sum().item() if diag(a).size else 0


def _pad_stat(a, pad_width, mode):
    out = a
    for axis, (before, after) in enumerate(pad_width):
        if before == 0 and after == 0:
            continue
        if mode == 'maximum':
            stat = out.max(axis=axis, keepdims=True)
        elif mode == 'minimum':
            stat = out.min(axis=axis, keepdims=True)
        elif mode == 'mean':
            stat = mean(out, axis=axis, keepdims=True)
        elif mode == 'median':
            stat = median(out, axis=axis, keepdims=True)
        else:
            stat = None
        if stat is None:
            shape = out._shape
            parts = [out]
            if before:
                parts.insert(0, full(shape[:axis] + (before,) + shape[axis + 1:], 0, out.dtype))
            if after:
                parts.append(full(shape[:axis] + (after,) + shape[axis + 1:], 0, out.dtype))
        else:
            parts = [stat] * before + [out] + [stat] * after
        out = concatenate(parts, axis=axis)
    return out


def _pad_ramp(a, pad_width, end_values):
    if isinstance(end_values, (int, float, complex)):
        ends = [(end_values, end_values)] * a.ndim
    elif (isinstance(end_values, (tuple, list)) and len(end_values) == 2
          and all(isinstance(v, (int, float, complex)) for v in end_values)):
        ends = [tuple(end_values)] * a.ndim
    else:
        ends = [tuple(p) for p in end_values]
    out = a
    for axis, (before, after) in enumerate(pad_width):
        if before == 0 and after == 0:
            continue
        end_b, end_a = ends[axis]
        parts = [out]
        if before:
            edge = out.take([0], axis=axis)
            ramps = [end_b * (1.0 - i / before) + edge * (i / before) for i in range(before)]
            parts = ramps + parts
        if after:
            edge = out.take([out._shape[axis] - 1], axis=axis)
            ramps = [edge * (1.0 - (i + 1) / after) + end_a * ((i + 1) / after) for i in range(after)]
            parts = parts + ramps
        out = concatenate(parts, axis=axis)
    return out


def pad(a, pad_width, mode='constant', constant_values=0, end_values=0, **kwargs):
    a = asarray(a)
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)] * a.ndim
    elif isinstance(pad_width, tuple) and pad_width and isinstance(pad_width[0], int):
        pad_width = [tuple(pad_width)] * a.ndim
    pad_width = [tuple(p) for p in pad_width]
    if callable(mode):
        out = a
        for axis, (before, after) in enumerate(pad_width):
            if before == 0 and after == 0:
                continue
            shape = out._shape
            buf = zeros(shape[:axis] + (before + shape[axis] + after,) + shape[axis + 1:], dtype=out.dtype)
            mid = tuple(slice(None) if d != axis else slice(before, before + shape[axis]) for d in range(out.ndim))
            buf[mid] = out
            mode(buf, (before, after), axis, kwargs)
            out = buf
        return out
    if mode == 'linear_ramp':
        return _pad_ramp(a, pad_width, end_values)
    if mode in ('maximum', 'minimum', 'mean', 'median', 'empty'):
        return _pad_stat(a, pad_width, mode)
    if mode not in ('constant', 'edge', 'reflect', 'symmetric', 'wrap'):
        raise NotImplementedError(f'pad mode {mode!r}')
    new_shape = tuple(s + p[0] + p[1] for s, p in zip(a._shape, pad_width))
    const = _cast_value(constant_values, a.dtype)
    nstr = _c_strides(new_shape)
    ostr = _c_strides(a._shape)

    def _map(q, s):
        if mode == 'edge':
            return 0 if q < 0 else (s - 1 if q >= s else q)
        if mode == 'wrap':
            return q % s
        if mode == 'reflect':
            if s == 1:
                return 0
            while q < 0 or q >= s:
                if q < 0:
                    q = -q
                else:
                    q = 2 * (s - 1) - q
            return q
        if mode == 'symmetric':
            if s == 1:
                return 0
            while q < 0 or q >= s:
                if q < 0:
                    q = -q - 1
                else:
                    q = 2 * s - 1 - q
            return q
        return q

    out = []
    for idx in range(_numel(new_shape)):
        rem = idx
        pos = []
        for dim in range(a.ndim - 1, -1, -1):
            p = rem % new_shape[dim]
            rem //= new_shape[dim]
            pos.append(p)
        pos.reverse()
        if mode == 'constant':
            fill = False
            src = 0
            for dim in range(a.ndim):
                q = pos[dim] - pad_width[dim][0]
                if q < 0 or q >= a._shape[dim]:
                    fill = True
                    break
                src += q * ostr[dim]
            out.append(const if fill else a._data[src])
        else:
            src = 0
            empty = False
            for dim in range(a.ndim):
                if a._shape[dim] <= 0:
                    empty = True
                    break
                src += _map(pos[dim] - pad_width[dim][0], a._shape[dim]) * ostr[dim]
            out.append(const if empty else a._data[src])
    return ndarray(out, new_shape, a.dtype)


def tile(a, reps):
    a = asarray(a)
    if isinstance(reps, int):
        reps = (reps,)
    reps = tuple(int(r) for r in reps)
    while len(reps) < a.ndim:
        reps = (1,) + reps
    shape = a._shape
    while len(shape) < len(reps):
        shape = (1,) + shape
        a = a.reshape(shape)
    new_shape = tuple(s * r for s, r in zip(shape, reps))
    out = []
    for idx in range(_numel(new_shape)):
        rem = idx
        src = 0
        st = _c_strides(shape)
        for dim in range(len(new_shape) - 1, -1, -1):
            p = rem % new_shape[dim]
            rem //= new_shape[dim]
            src += (p % shape[dim]) * st[dim]
        out.append(a._data[src])
    return ndarray(out, new_shape, a.dtype)


def repeat(a, repeats, axis=None):
    a = asarray(a)
    if axis is None:
        flat = a._data
        if isinstance(repeats, int):
            out = []
            for v in flat:
                out.extend([v] * repeats)
            return ndarray(out, (len(out),), a.dtype)
        out = []
        for v, r in zip(flat, repeats):
            out.extend([v] * int(r))
        return ndarray(out, (len(out),), a.dtype)
    ax = axis + a.ndim if axis < 0 else axis
    n = a._shape[ax]
    if isinstance(repeats, int):
        reps = [repeats] * n
    else:
        reps = [int(r) for r in repeats]
    inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
    outer = _numel(a._shape[:ax])
    step = n * inner
    out = []
    for o in range(outer):
        base = o * step
        for i in range(n):
            out.extend(a._data[base + i * inner:base + (i + 1) * inner] * reps[i])
    shape = list(a._shape)
    shape[ax] = _bsum(reps)
    return ndarray(out, tuple(shape), a.dtype)


def roll(a, shift, axis=None):
    a = asarray(a)
    if axis is None:
        n = a.size
        s = shift % n
        return ndarray(a._data[n - s:] + a._data[:n - s], a._shape, a.dtype)
    ax = axis + a.ndim if axis < 0 else axis
    n = a._shape[ax]
    s = shift % n
    inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
    outer = _numel(a._shape[:ax])
    step = n * inner
    out = list(a._data)
    for o in range(outer):
        base = o * step
        block = out[base:base + step]
        moved = block[(n - s) * inner:] + block[:(n - s) * inner]
        out[base:base + step] = moved
    return ndarray(out, a._shape, a.dtype)


def flip(a, axis=None):
    a = asarray(a)
    if axis is None:
        return ndarray(list(reversed(a._data)), a._shape, a.dtype)
    if isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)
    out = a
    for ax in axes:
        n = out._shape[ax]
        inner = _numel(out._shape[ax + 1:]) if ax + 1 < out.ndim else 1
        outer = _numel(out._shape[:ax])
        step = n * inner
        res = list(out._data)
        for o in range(outer):
            base = o * step
            chunks = [res[base + i * inner:base + (i + 1) * inner] for i in range(n)]
            chunks.reverse()
            flat = []
            for c in chunks:
                flat.extend(c)
            res[base:base + step] = flat
        out = ndarray(res, out._shape, out.dtype)
    return out


def correlate(a, v, mode='valid'):
    a = asarray(a).flatten()._data
    v = asarray(v).flatten()._data
    n, m = len(a), len(v)
    full = []
    for i in range(n + m - 1):
        s = 0
        for j in range(m):
            k = i - j
            if 0 <= k < n:
                s += a[k] * v[j]
        full.append(s)
    if mode == 'full':
        out = full
    elif mode == 'valid':
        out = full[m - 1:n] if n >= m else full[n - 1:m]
    elif mode == 'same':
        start = (len(full) - _bmax(n, m)) // 2
        out = full[start:start + _bmax(n, m)]
    else:
        raise ValueError(f'unknown mode {mode!r}')
    return ndarray(out, (len(out),), float64)


def _parse_einsum(eq, n_ops):
    eq = eq.replace(' ', '')
    if '->' in eq:
        lhs, rhs = eq.split('->')
    else:
        lhs, rhs = eq, None
    inputs = lhs.split(',')
    if len(inputs) != n_ops:
        raise ValueError('einsum: operand count does not match equation')
    if rhs is None:
        seen = []
        for s in inputs:
            for c in s:
                if c not in seen:
                    seen.append(c)
        counts = {}
        for s in inputs:
            for c in s:
                counts[c] = counts.get(c, 0) + 1
        rhs = ''.join(c for c in seen if counts[c] == 1)
    return inputs, rhs


def einsum(eq, *operands):
    ops = [asarray(o) for o in operands]
    inputs, out_sub = _parse_einsum(eq, len(ops))
    for sub, op in zip(inputs, ops):
        if len(sub) != op.ndim:
            raise ValueError(f'einsum: subscript {sub!r} does not match {op.ndim} dimensions')
    labels = []
    for s in inputs:
        labels.extend(list(s))
    labels.extend(list(out_sub))
    dims = {}
    for sub, op in zip(inputs, ops):
        for c, s in zip(sub, op._shape):
            if c in dims and dims[c] != s:
                raise ValueError(f'einsum: dimension mismatch for label {c!r}')
            dims[c] = s
    for c in out_sub:
        if c not in dims:
            raise ValueError(f'einsum: output subscript {c!r} never appears in input')
    dt = ops[0].dtype
    for op in ops[1:]:
        dt = _promote(dt, op.dtype)
    if dt.kind == 'b':
        dt = int64
    out_shape = tuple(dims[c] for c in out_sub)
    in_strides = [_c_strides(op._shape) for op in ops]
    result = []
    for out_idx in range(_numel(out_shape)):
        rem = out_idx
        assign = {}
        for dim in range(len(out_shape) - 1, -1, -1):
            c = out_sub[dim]
            assign[c] = rem % out_shape[dim]
            rem //= out_shape[dim]
        total = 0
        sum_labels = sorted(set(labels) - set(out_sub))

        def rec(li, acc):
            nonlocal total
            if li == len(sum_labels):
                prod = 1
                for sub, op, st in zip(inputs, ops, in_strides):
                    off = 0
                    for pos, c in enumerate(sub):
                        off += assign[c] * st[pos]
                    prod *= op._data[off]
                total += prod
                return
            c = sum_labels[li]
            for v in range(dims[c]):
                assign[c] = v
                rec(li + 1, acc)
        rec(0, None)
        result.append(_cast_value(total, dt))
    return ndarray(result, out_shape, dt)


def _dft(x, inverse=False):
    n = len(x)
    sign = 1 if inverse else -1
    out = []
    for k in range(n):
        s = 0j
        for t in range(n):
            ang = sign * 2 * math.pi * k * t / n
            s += complex(x[t]) * complex(math.cos(ang), math.sin(ang))
        if inverse:
            s /= n
        out.append(s)
    return out


class _FFTModule:
    def fft(self, a, n=None, axis=-1):
        a = asarray(a)
        if a.ndim == 0:
            return ndarray([complex(a._data[0])], (), complex128)
        ax = axis + a.ndim if axis < 0 else axis
        inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
        outer = _numel(a._shape[:ax])
        m = a._shape[ax]
        n = m if n is None else int(n)
        step = m * inner
        nshape = a._shape[:ax] + (n,) + a._shape[ax + 1:]
        out = [0j] * (outer * inner * n)
        for o in range(outer):
            base = o * step
            for j in range(inner):
                col = [a._data[base + j + i * inner] for i in range(m)]
                if n > m:
                    col = col + [0j] * (n - m)
                else:
                    col = col[:n]
                vals = _dft(col)
                for i in range(n):
                    out[(o * n + i) * inner + j] = vals[i]
        return ndarray(out, nshape, complex128)

    def ifft(self, a, n=None, axis=-1):
        a = asarray(a)
        if a.ndim == 0:
            return ndarray([complex(a._data[0])], (), complex128)
        ax = axis + a.ndim if axis < 0 else axis
        inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
        outer = _numel(a._shape[:ax])
        m = a._shape[ax]
        n = m if n is None else int(n)
        step = m * inner
        nshape = a._shape[:ax] + (n,) + a._shape[ax + 1:]
        out = [0j] * (outer * inner * n)
        for o in range(outer):
            base = o * step
            for j in range(inner):
                col = [a._data[base + j + i * inner] for i in range(m)]
                if n > m:
                    col = col + [0j] * (n - m)
                else:
                    col = col[:n]
                vals = _dft(col, inverse=True)
                for i in range(n):
                    out[(o * n + i) * inner + j] = vals[i]
        return ndarray(out, nshape, complex128)

    def rfft(self, a, n=None, axis=-1):
        f = self.fft(a, n=n, axis=axis)
        ax = axis + f.ndim if axis < 0 else axis
        m = f._shape[ax]
        sl = [slice(None)] * f.ndim
        sl[ax] = slice(0, m // 2 + 1)
        return f[tuple(sl)]

    def irfft(self, a, n=None, axis=-1):
        a = asarray(a)
        ax = axis + a.ndim if axis < 0 else axis
        m = a._shape[ax]
        n = (m - 1) * 2 if n is None else int(n)
        full_sl = [slice(None)] * a.ndim
        full_sl[ax] = slice(0, m)
        buf = zeros(a._shape[:ax] + (n,) + a._shape[ax + 1:], dtype=complex128)
        buf[tuple(full_sl)] = a
        return self.ifft(buf, axis=ax).real

    def _nd_axes(self, a, axes):
        if axes is None:
            return tuple(range(a.ndim))
        return tuple(ax + a.ndim if ax < 0 else ax for ax in axes)

    def fftn(self, a, s=None, axes=None):
        if s is not None:
            raise ValueError('fftn com s= exige remodelagem explícita')
        out = asarray(a)
        for ax in self._nd_axes(out, axes):
            out = self.fft(out, axis=ax)
        return out

    def ifftn(self, a, s=None, axes=None):
        if s is not None:
            raise ValueError('ifftn com s= exige remodelagem explícita')
        out = asarray(a)
        for ax in self._nd_axes(out, axes):
            out = self.ifft(out, axis=ax)
        return out

    def fft2(self, a, s=None, axes=(-2, -1)):
        return self.fftn(a, s=s, axes=axes)

    def ifft2(self, a, s=None, axes=(-2, -1)):
        return self.ifftn(a, s=s, axes=axes)

    def fftfreq(self, n, d=1.0):
        n = int(n)
        val = 1.0 / (n * d)
        out = []
        for i in range(n):
            out.append((i if i < (n + 1) // 2 else i - n) * val)
        return ndarray(out, (n,), float64)

    def rfftfreq(self, n, d=1.0):
        n = int(n)
        val = 1.0 / (n * d)
        m = n // 2 + 1
        return ndarray([i * val for i in range(m)], (m,), float64)

    def fftshift(self, a, axes=None):
        a = asarray(a)
        if axes is None:
            axes = tuple(range(a.ndim))
        elif isinstance(axes, int):
            axes = (axes,)
        out = a
        for ax in axes:
            n = out._shape[ax]
            p = (n + 1) // 2
            out = concatenate([out.take(list(range(p, n)), axis=ax),
                               out.take(list(range(p)), axis=ax)], axis=ax)
        return out

    def ifftshift(self, a, axes=None):
        a = asarray(a)
        if axes is None:
            axes = tuple(range(a.ndim))
        elif isinstance(axes, int):
            axes = (axes,)
        out = a
        for ax in axes:
            n = out._shape[ax]
            p = n - (n + 1) // 2
            out = concatenate([out.take(list(range(p, n)), axis=ax),
                               out.take(list(range(p)), axis=ax)], axis=ax)
        return out


fft = _FFTModule()


class LinAlgError(Exception):
    pass


def _as_2d(a):
    a = asarray(a)
    if a.ndim != 2:
        raise LinAlgError('expected 2-d array')
    return a


class _LinalgModule:
    LinAlgError = LinAlgError

    def norm(self, x, ord=None, axis=None):
        x = asarray(x)
        if axis is None:
            vals = [_babs(complex(v)) for v in x._data]
            if ord is None or ord == 2:
                return math.sqrt(_bsum(v * v for v in vals))
            if ord == 1:
                return _bsum(vals)
            if ord == float('inf') or ord == 'inf':
                return _bmax(vals)
            if ord == 0:
                return _bsum(1 for v in vals if v != 0)
            raise LinAlgError(f'norm order {ord!r} not supported')
        if axis == -1 and x.ndim == 2:
            return array([self.norm(row) for row in x.tolist()])
        return self.norm(x.flatten())

    def det(self, a):
        a = _as_2d(a)
        n = a._shape[0]
        if a._shape[1] != n:
            raise LinAlgError('det: not square')
        m = [[complex(v) for v in row] for row in a.tolist()]
        det = 1 + 0j
        for col in range(n):
            piv = _bmax(range(col, n), key=lambda r: _babs(m[r][col]))
            if _babs(m[piv][col]) == 0:
                return 0.0
            if piv != col:
                m[col], m[piv] = m[piv], m[col]
                det = -det
            det *= m[col][col]
            for row in range(col + 1, n):
                f = m[row][col] / m[col][col]
                for k in range(col, n):
                    m[row][k] -= f * m[col][k]
        return det.real if det.imag == 0 else det

    def solve(self, a, b):
        a = _as_2d(a)
        n = a._shape[0]
        if a._shape[1] != n:
            raise LinAlgError('solve: not square')
        b = asarray(b)
        nrhs = b._shape[0] if b.ndim == 2 else 1
        if b.ndim == 1:
            rhs = [[complex(v)] for v in b._data]
        elif b.ndim == 2 and b._shape[0] == n:
            rhs = [[complex(v)] for v in b._data]
            nrhs = b._shape[1]
            rhs = [[b._data[i * nrhs + j] for i in range(n)] for j in range(nrhs)]
            rhs = [[complex(rhs[j][i]) for j in range(nrhs)] for i in range(n)]
        else:
            raise LinAlgError('solve: bad rhs shape')
        m = [[complex(v) for v in row] for row in a.tolist()]
        for i in range(n):
            m[i].extend(rhs[i])
        for col in range(n):
            piv = _bmax(range(col, n), key=lambda r: _babs(m[r][col]))
            if _babs(m[piv][col]) == 0:
                raise LinAlgError('singular matrix')
            m[col], m[piv] = m[piv], m[col]
            pivv = m[col][col]
            for k in range(col, n + nrhs):
                m[col][k] /= pivv
            for row in range(n):
                if row != col:
                    f = m[row][col]
                    for k in range(col, n + nrhs):
                        m[row][k] -= f * m[col][k]
        sol = [[m[i][n + j] for j in range(nrhs)] for i in range(n)]
        flat = [v for row in sol for v in row]
        if _ball(v.imag == 0 for v in flat):
            flat = [v.real for v in flat]
            sol = [[v.real for v in row] for row in sol]
            dt = float64
        else:
            dt = complex128
        if b.ndim == 1:
            return ndarray([_cast_value(s[0], dt) for s in sol], (n,), dt)
        out = []
        for j in range(nrhs):
            for i in range(n):
                out.append(sol[i][j])
        return ndarray([_cast_value(v, dt) for v in out], (n, nrhs), dt)

    def inv(self, a):
        a = _as_2d(a)
        n = a._shape[0]
        e = eye(n)
        cols = []
        for j in range(n):
            cols.append(self.solve(a, e[:, j]))
        return stack(cols, axis=1)

    def eigvalsh(self, a):
        a = _as_2d(a)
        n = a._shape[0]
        if a._shape[1] != n:
            raise LinAlgError('eigvalsh: not square')
        m = [[float(v.real) if isinstance(v, complex) else float(v) for v in row]
             for row in a.tolist()]
        for _ in range(100):
            off = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    off += m[i][j] * m[i][j]
            if off < 1e-20:
                break
            p, q = 0, 1
            best = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    if _babs(m[i][j]) > best:
                        best = _babs(m[i][j])
                        p, q = i, j
            if best == 0:
                break
            theta = (m[q][q] - m[p][p]) / (2 * m[p][q])
            t = (1 if theta >= 0 else -1) / (_babs(theta) + math.sqrt(theta * theta + 1))
            c = 1 / math.sqrt(t * t + 1)
            s = t * c
            for k in range(n):
                if k != p and k != q:
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = m[p][k] = c * mkp - s * mkq
                    m[k][q] = m[q][k] = s * mkp + c * mkq
            mpp, mqq, mpq = m[p][p], m[q][q], m[p][q]
            m[p][p] = c * c * mpp - 2 * s * c * mpq + s * s * mqq
            m[q][q] = s * s * mpp + 2 * s * c * mpq + c * c * mqq
            m[p][q] = m[q][p] = 0.0
        return ndarray([m[i][i] for i in range(n)], (n,), float64)

    def eig(self, a):
        w = self.eigvalsh(a)
        return w, eye(a._shape[0])

    def eigvals(self, a):
        a = _as_2d(a)
        n = a._shape[0]
        if a._shape[1] != n:
            raise LinAlgError('eigvals: not square')
        src = a.tolist()
        h = []
        for row in src:
            hr = []
            for v in row:
                if isinstance(v, complex):
                    raise LinAlgError('eigvals: complex input not supported')
                hr.append(float(v))
            h.append(hr)
        for k in range(n - 2):
            x = [h[i][k] for i in range(k + 1, n)]
            nx = math.sqrt(_bsum(v * v for v in x))
            if nx == 0.0:
                continue
            alpha = -math.copysign(nx, x[0])
            u = list(x)
            u[0] -= alpha
            nu = math.sqrt(_bsum(v * v for v in u))
            if nu == 0.0:
                continue
            v = [t / nu for t in u]
            m = n - k - 1
            for j in range(k, n):
                d = _bsum(v[i] * h[k + 1 + i][j] for i in range(m))
                for i in range(m):
                    h[k + 1 + i][j] -= 2.0 * d * v[i]
            for i in range(n):
                d = _bsum(h[i][k + 1 + j] * v[j] for j in range(m))
                for j in range(m):
                    h[i][k + 1 + j] -= 2.0 * d * v[j]
        out = []
        mm = n
        sweeps = 0
        while mm > 2:
            scale = _babs(h[mm - 1][mm - 1]) + _babs(h[mm - 2][mm - 2])
            if _babs(h[mm - 1][mm - 2]) <= 1e-12 * scale:
                out.append(complex(h[mm - 1][mm - 1], 0.0))
                mm -= 1
                continue
            if _babs(h[mm - 2][mm - 3]) <= 1e-12 * scale:
                aa, bb = h[mm - 2][mm - 2], h[mm - 2][mm - 1]
                cc, dd = h[mm - 1][mm - 2], h[mm - 1][mm - 1]
                tr = aa + dd
                disc = tr * tr - 4.0 * (aa * dd - bb * cc)
                if disc >= 0.0:
                    sq = math.sqrt(disc)
                    out.append(complex((tr + sq) / 2.0, 0.0))
                    out.append(complex((tr - sq) / 2.0, 0.0))
                else:
                    sq = math.sqrt(-disc)
                    out.append(complex(tr / 2.0, sq / 2.0))
                    out.append(complex(tr / 2.0, -sq / 2.0))
                mm -= 2
                continue
            mu = h[mm - 1][mm - 1]
            for i in range(mm):
                h[i][i] -= mu
            cs = []
            for i in range(mm - 1):
                xx, yy = h[i][i], h[i + 1][i]
                r = math.hypot(xx, yy)
                if r == 0.0:
                    c, s = 1.0, 0.0
                else:
                    c, s = xx / r, -yy / r
                cs.append((c, s))
                for j in range(i, mm):
                    a0, a1 = h[i][j], h[i + 1][j]
                    h[i][j] = c * a0 - s * a1
                    h[i + 1][j] = s * a0 + c * a1
            for i in range(mm - 1):
                c, s = cs[i]
                for k2 in range(mm):
                    a0, a1 = h[k2][i], h[k2][i + 1]
                    h[k2][i] = c * a0 - s * a1
                    h[k2][i + 1] = s * a0 + c * a1
            for i in range(mm):
                h[i][i] += mu
            sweeps += 1
            if sweeps > 200 * max(n, 1):
                raise LinAlgError('eigvals: no convergence')
        if mm == 2:
            aa, bb = h[0][0], h[0][1]
            cc, dd = h[1][0], h[1][1]
            tr = aa + dd
            disc = tr * tr - 4.0 * (aa * dd - bb * cc)
            if disc >= 0.0:
                sq = math.sqrt(disc)
                out.append(complex((tr + sq) / 2.0, 0.0))
                out.append(complex((tr - sq) / 2.0, 0.0))
            else:
                sq = math.sqrt(-disc)
                out.append(complex(tr / 2.0, sq / 2.0))
                out.append(complex(tr / 2.0, -sq / 2.0))
        elif mm == 1:
            out.append(complex(h[0][0], 0.0))
        return ndarray(out, (n,), complex128)

    def svd(self, a, full_matrices=True):
        a = _as_2d(a)
        m, n = a._shape
        ata = _matmul_impl(a.T, a)
        w = self.eigvalsh(ata)
        s = sqrt(maximum(w, 0.0))
        order = argsort(s)[::-1]
        s = s[order.tolist() if hasattr(order, 'tolist') else list(order._data)]
        return eye(m), s, eye(n)

    def qr(self, a):
        a = _as_2d(a)
        m, n = a._shape
        cols = []
        for j in range(n):
            v = [float(a._data[i * n + j]) for i in range(m)]
            for q in cols:
                r = _bsum(q[k] * v[k] for k in range(m))
                v = [v[k] - r * q[k] for k in range(m)]
            nrm = math.sqrt(_bsum(x * x for x in v))
            cols.append([x / nrm if nrm else 0.0 for x in v])
        q = zeros((m, n))
        for j, c in enumerate(cols):
            for i in range(m):
                q._data[i * n + j] = c[i]
        r = _matmul_impl(q.T, a)
        return q, r

    def cholesky(self, a):
        a = _as_2d(a)
        n = a._shape[0]
        out = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = _bsum(out[i][k] * out[j][k] for k in range(j))
                if i == j:
                    v = float(a._data[i * n + i]) - s
                    if v <= 0:
                        raise LinAlgError('not positive definite')
                    out[i][j] = math.sqrt(v)
                else:
                    out[i][j] = (float(a._data[i * n + j]) - s) / out[j][j]
        flat = []
        for row in out:
            flat.extend(row)
        return ndarray(flat, (n, n), float64)

    def matrix_rank(self, a, tol=None):
        a = _as_2d(a)
        m = [[float(v) for v in row] for row in a.tolist()]
        rows, cols = len(m), len(m[0])
        r = 0
        for c in range(cols):
            piv = None
            for i in range(r, rows):
                if _babs(m[i][c]) > (tol or 1e-10):
                    piv = i
                    break
            if piv is None:
                continue
            m[r], m[piv] = m[piv], m[r]
            for i in range(r + 1, rows):
                f = m[i][c] / m[r][c]
                for k in range(c, cols):
                    m[i][k] -= f * m[r][k]
            r += 1
        return r


linalg = _LinalgModule()


class _Generator:
    def __init__(self, seed=None):
        self._rng = _py_random.Random(seed)

    def standard_normal(self, size=None, dtype=None):
        dt = _resolve_dtype(dtype, float64)
        if isinstance(size, list):
            size = tuple(size)
        if size is None:
            return _cast_value(self._gauss(), dt)
        n = _numel(size) if isinstance(size, tuple) else int(size)
        shape = size if isinstance(size, tuple) else (n,)
        return ndarray([_cast_value(self._gauss(), dt) for _ in range(n)], shape, dt)

    def _gauss(self):
        u1 = self._rng.random() or 1e-300
        u2 = self._rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if isinstance(size, list):
            size = tuple(size)
        if size is None:
            return loc + scale * self._gauss()
        n = _numel(size) if isinstance(size, tuple) else int(size)
        shape = size if isinstance(size, tuple) else (n,)
        return ndarray([loc + scale * self._gauss() for _ in range(n)], shape, float64)

    def uniform(self, low=0.0, high=1.0, size=None):
        if isinstance(size, list):
            size = tuple(size)
        if size is None:
            return low + (high - low) * self._rng.random()
        n = _numel(size) if isinstance(size, tuple) else int(size)
        shape = size if isinstance(size, tuple) else (n,)
        return ndarray([low + (high - low) * self._rng.random() for _ in range(n)],
                       shape, float64)

    def random(self, size=None):
        return self.uniform(0.0, 1.0, size=size)

    def rand(self, *shape):
        n = _numel(shape)
        return ndarray([self._rng.random() for _ in range(n)], tuple(shape), float64)

    def randn(self, *shape):
        n = _numel(shape)
        return ndarray([self._gauss() for _ in range(n)], tuple(shape), float64)

    def randint(self, low, high=None, size=None):
        if high is None:
            low, high = 0, low
        if isinstance(size, list):
            size = tuple(size)
        if size is None:
            return self._rng.randrange(int(low), int(high))
        n = _numel(size) if isinstance(size, tuple) else int(size)
        shape = size if isinstance(size, tuple) else (n,)
        return ndarray([self._rng.randrange(int(low), int(high)) for _ in range(n)],
                       shape, int64)

    integers = randint

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, int):
            pool = list(range(int(a)))
        else:
            pool = list(a._data) if isinstance(a, ndarray) else list(a)
        weights = None
        if p is not None:
            pw = list(p._data) if isinstance(p, ndarray) else list(p)
            total = _bsum(pw)
            acc = 0.0
            weights = []
            for w in pw:
                acc += w / total
                weights.append(acc)

        def pick():
            if weights is None:
                return self._rng.choice(pool)
            r = self._rng.random()
            for v, c in zip(pool, weights):
                if r < c:
                    return v
            return pool[-1]

        if isinstance(size, list):
            size = tuple(size)
        if size is None:
            return pick()
        n = _numel(size) if isinstance(size, tuple) else int(size)
        shape = size if isinstance(size, tuple) else (n,)
        if replace:
            out = [pick() for _ in range(n)]
        else:
            out = self._rng.sample(pool, n)
        dt = _scalar_dtype(out[0]) if out else int64
        return ndarray(out, shape, dt or int64)

    def shuffle(self, a):
        if isinstance(a, ndarray):
            idx = list(range(a._shape[0]))
            self._rng.shuffle(idx)
            a[:] = a[idx]
        else:
            self._rng.shuffle(a)

    def permutation(self, x):
        if isinstance(x, int):
            x = arange(x)
        x = asarray(x).copy()
        self.shuffle(x)
        return x

    def seed(self, s=None):
        self._rng.seed(s)


class _RandomModule:
    Generator = _Generator

    def __init__(self):
        self._g = _Generator()

    def seed(self, s=None):
        self._g.seed(s)

    def default_rng(self, seed=None):
        return _Generator(seed)

    def RandomState(self, seed=None):
        return _Generator(seed)

    def standard_normal(self, size=None, dtype=None):
        return self._g.standard_normal(size, dtype=dtype)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._g.normal(loc, scale, size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._g.uniform(low, high, size)

    def random(self, size=None):
        return self._g.random(size)

    def rand(self, *shape):
        return self._g.rand(*shape)

    def randn(self, *shape):
        return self._g.randn(*shape)

    def randint(self, low, high=None, size=None):
        return self._g.randint(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        return self._g.choice(a, size, replace, p)

    def shuffle(self, a):
        return self._g.shuffle(a)


random = _RandomModule()


def _struct_fmt(dt):
    if dt is bool_:
        return 'B'
    if dt is int8:
        return 'b'
    if dt is uint8:
        return 'B'
    if dt is int16:
        return 'h'
    if dt is uint16:
        return 'H'
    if dt is int32:
        return 'i'
    if dt is uint32:
        return 'I'
    if dt is int64:
        return 'q'
    if dt is uint64:
        return 'Q'
    if dt is float16:
        return 'e'
    if dt is float32:
        return 'f'
    if dt is float64:
        return 'd'
    if dt is complex64:
        return 'ff'
    if dt is complex128:
        return 'dd'
    raise TypeError(f'unsupported dtype for buffer: {dt}')


def frombuffer(buf, dt=None, **kw):
    dt = _resolve_dtype(_pop_dtype(dt, kw), uint8)
    if isinstance(buf, (bytes, bytearray, memoryview)):
        raw = bytes(buf)
    elif hasattr(buf, 'tobytes'):
        raw = buf.tobytes()
    else:
        try:
            raw = bytes(buf)
        except TypeError:
            raw = bytes(bytearray(buf))
    fmt = _struct_fmt(dt)
    if dt in (complex64, complex128):
        comp = 'f' if dt is complex64 else 'd'
        n = len(raw) // (_struct.calcsize(comp) * 2)
        vals = _struct.unpack(f'<{n * 2}{comp}', raw[:n * 2 * _struct.calcsize(comp)])
        out = [complex(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
        return ndarray(out, (n,), dt)
    n = len(raw) // _struct.calcsize(fmt)
    vals = _struct.unpack(f'<{n}{fmt}', raw[:n * _struct.calcsize(fmt)])
    out = []
    for v in vals:
        if dt is bool_:
            out.append(bool(v))
        else:
            out.append(_cast_value(v, dt))
    return ndarray(out, (n,), dt)


class _CtypesLib:
    def as_array(self, ptr, shape=None):
        if shape is None:
            raise ValueError('shape is required')
        if isinstance(shape, int):
            shape = (shape,)
        addr = ptr if isinstance(ptr, int) else ctypes.cast(ptr, ctypes.c_void_p).value
        buf = (ctypes.c_double * _numel(shape)).from_address(addr)
        return ndarray([float(v) for v in buf], tuple(shape), float64)


ctypeslib = _CtypesLib()


def shape(a):
    return asarray(a)._shape


def ndim(a):
    return asarray(a).ndim


def size(a):
    return asarray(a).size


def copy(a):
    return asarray(a).copy()


def result_type(*args):
    dt = None
    for x in args:
        if isinstance(x, ndarray):
            dt = _promote(dt, x.dtype)
        elif isinstance(x, _DType):
            dt = _promote(dt, x)
        else:
            dt = _promote(dt, _scalar_dtype(x))
    return dt or float64


pi = math.pi
e = math.e
inf = float('inf')
nan = float('nan')
Inf = inf
NaN = nan
NINF = float('-inf')
PINF = inf
PZERO = 0.0
NZERO = -0.0
newaxis = None
euler_gamma = 0.5772156649015329


class _ErrState:
    def __init__(self, **kw):
        self._kw = kw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def errstate(**kw):
    return _ErrState(**kw)


def seterr(**kw):
    return {k: 'ignore' for k in ('divide', 'over', 'under', 'invalid')}


def geterr():
    return {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}


class _Finfo:
    def __init__(self, dt):
        dt = dtype(dt)
        if dt is float16:
            self.bits, self.eps, self.max, self.min = 16, 9.77e-4, 65504.0, -65504.0
            self.tiny = 6.1e-5
        elif dt is float32:
            self.bits, self.eps, self.max, self.min = 32, 1.19e-7, 3.4028235e38, -3.4028235e38
            self.tiny = 1.17549435e-38
        else:
            self.bits, self.eps, self.max, self.min = 64, 2.22e-16, 1.7976931348623157e308, -1.7976931348623157e308
            self.tiny = 2.2250738585072014e-308
        self.dtype = dt


def finfo(dt):
    return _Finfo(dt)


class _Iinfo:
    def __init__(self, dt):
        dt = dtype(dt)
        bits = dt.itemsize * 8
        if dt.kind == 'u':
            self.min, self.max = 0, 2 ** bits - 1
        else:
            self.min, self.max = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        self.bits = bits
        self.dtype = dt


def iinfo(dt):
    return _Iinfo(dt)


__version__ = 'ntri-pure/1.0'
version = type('version', (), {'version': __version__, 'release': True})()


def broadcast_to(a, shape):
    a = asarray(a)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)
    return ndarray(_broadcast_to_data(a._data, a._shape, shape), shape, a.dtype)


def broadcast_arrays(*args):
    shapes = [asarray(x)._shape for x in args]
    out = shapes[0]
    for s in shapes[1:]:
        out = _broadcast_shapes(out, s)
    return [broadcast_to(x, out) for x in args]


import abc as _abc


class integer(metaclass=_abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, c):
        if isinstance(c, type) and issubclass(c, int):
            return True
        return NotImplemented


class floating(metaclass=_abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, c):
        if c is float:
            return True
        return NotImplemented


class complexfloating(metaclass=_abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, c):
        if c is complex:
            return True
        return NotImplemented


class number(metaclass=_abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, c):
        if isinstance(c, type) and issubclass(c, (int, float, complex)):
            return True
        return NotImplemented


class inexact(metaclass=_abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, c):
        if c is float or c is complex:
            return True
        return NotImplemented


class generic:
    pass


class flexible(generic):
    pass


signedinteger = integer
unsignedinteger = integer


def diff(a, n=1, axis=-1):
    a = asarray(a)
    for _ in range(int(n)):
        ax = axis + a.ndim if axis < 0 else axis
        m = a._shape[ax]
        inner = _numel(a._shape[ax + 1:]) if ax + 1 < a.ndim else 1
        outer = _numel(a._shape[:ax])
        step = m * inner
        out = []
        for o in range(outer):
            base = o * step
            for j in range(inner):
                for i in range(m - 1):
                    out.append(a._data[base + j + (i + 1) * inner] - a._data[base + j + i * inner])
        shape = list(a._shape)
        shape[ax] = _bmax(m - 1, 0)
        a = ndarray(out, tuple(shape), _promote(a.dtype, a.dtype))
    return a


def nan_to_num(a, nan=0.0, posinf=None, neginf=None):
    a = asarray(a)

    def fix(v):
        if isinstance(v, complex):
            r = nan if v.real != v.real else v.real
            i = nan if v.imag != v.imag else v.imag
            if v.real in (inf, -inf):
                r = (posinf if posinf is not None else 1.7976931348623157e308) if v.real > 0 else (neginf if neginf is not None else -1.7976931348623157e308)
            if v.imag in (inf, -inf):
                i = (posinf if posinf is not None else 1.7976931348623157e308) if v.imag > 0 else (neginf if neginf is not None else -1.7976931348623157e308)
            return complex(r, i)
        if v != v:
            return nan
        if v in (inf, -inf):
            big = 1.7976931348623157e308
            if v > 0:
                return posinf if posinf is not None else big
            return neginf if neginf is not None else -big
        return v
    return ndarray([_cast_value(fix(v), a.dtype) for v in a._data],
                   a._shape, a.dtype)


def copy(a):
    return asarray(a).copy()


def logical_and(a, b):
    return asarray(a)._logic(b, lambda x, y: x and y)


def logical_or(a, b):
    return asarray(a)._logic(b, lambda x, y: x or y)


def logical_xor(a, b):
    return asarray(a)._logic(b, lambda x, y: (x and not y) or (y and not x))


def logical_not(a):
    return ndarray([not bool(v) for v in asarray(a)._data],
                   asarray(a)._shape, bool_)


def polyfit(x, y, deg):
    x = asarray(x).flatten()
    y = asarray(y).flatten()
    deg = int(deg)
    n = x.size
    cols = []
    for p in range(deg, -1, -1):
        cols.append([float(v) ** p for v in x._data])
    m = len(cols)
    ata = [[0.0] * m for _ in range(m)]
    aty = [0.0] * m
    for i in range(m):
        for j in range(m):
            ata[i][j] = _bsum(cols[i][k] * cols[j][k] for k in range(n))
        aty[i] = _bsum(cols[i][k] * float(y._data[k]) for k in range(n))
    coef = linalg.solve(ndarray([v for row in ata for v in row], (m, m), float64),
                        ndarray(aty, (m,), float64))
    return coef


def _npy_descr(dt):
    table = {bool_: '|b1', int8: '|i1', uint8: '|u1', int16: '<i2', uint16: '<u2',
             int32: '<i4', uint32: '<u4', int64: '<i8', uint64: '<u8',
             float16: '<f2', float32: '<f4', float64: '<f8',
             complex64: '<c8', complex128: '<c16'}
    return table[dt]


def _split_top_level(s, sep):
    parts = []
    depth = 0
    cur = []
    for ch in s:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append(''.join(cur))
    return parts


def _parse_shape(s):
    s = s.strip()
    if s in ('()', ''):
        return ()
    s = s.strip('()')
    return tuple(int(x) for x in s.split(',') if x.strip())


def _npy_from_descr(descr):
    d = descr.strip('<>=|')
    table = {'b1': bool_, 'i1': int8, 'u1': uint8, 'i2': int16, 'u2': uint16,
             'i4': int32, 'u4': uint32, 'i8': int64, 'u8': uint64,
             'f2': float16, 'f4': float32, 'f8': float64,
             'c8': complex64, 'c16': complex128, 'f8': float64}
    if d not in table:
        raise ValueError(f'descr {descr!r} not supported')
    return table[d]


def save(file, arr):
    import io as _io
    arr = asarray(arr)
    header = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (
        _npy_descr(arr.dtype), repr(arr._shape))
    header += ' ' * (64 - (len(header) + 10) % 64 - 1) + '\n'
    blob = b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header.encode('latin1')
    blob += arr.tobytes()
    if hasattr(file, 'write'):
        file.write(blob)
    else:
        with open(file, 'wb') as f:
            f.write(blob)


def load(file, allow_pickle=False):
    import io as _io
    if hasattr(file, 'read'):
        blob = file.read()
    else:
        with open(file, 'rb') as f:
            blob = f.read()
    if blob[:2] == b'PK':
        return _load_npz(blob)
    if blob[:6] != b'\x93NUMPY':
        raise ValueError('not an npy file')
    hlen = int.from_bytes(blob[8:10], 'little')
    header = blob[10:10 + hlen].decode('latin1')
    info = {}
    for part in _split_top_level(header.strip().strip('{}'), ','):
        part = part.strip()
        if not part or ':' not in part:
            continue
        k, v = part.split(':', 1)
        info[k.strip().strip("'\"")] = v.strip()
    dt = _npy_from_descr(info['descr'].strip("'\""))
    shape = _parse_shape(info['shape'])
    if isinstance(shape, int):
        shape = (shape,)
    raw = blob[10 + hlen:]
    n = _numel(shape)
    if dt in (complex64, complex128):
        comp = 'f' if dt is complex64 else 'd'
        sz = _struct.calcsize(comp)
        vals = _struct.unpack(f'<{n * 2}{comp}', raw[:n * 2 * sz])
        out = [complex(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    else:
        fmt = _struct_fmt(dt)
        sz = _struct.calcsize(fmt)
        vals = _struct.unpack(f'<{n}{fmt}', raw[:n * sz])
        out = [_cast_value(v, dt) for v in vals]
    return ndarray(out, tuple(shape), dt)
