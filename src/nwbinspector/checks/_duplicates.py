"""Checks for data that are stored twice within one NWB file: units, series, columns, rows and times."""

from __future__ import annotations

import hashlib
import weakref
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

import h5py
import numpy as np
from hdmf.common import DynamicTable, DynamicTableRegion, VectorData, VectorIndex
from pynwb import NWBFile, TimeSeries
from pynwb.misc import Units

from .._registration import Importance, InspectorMessage, register_check

# Datasets at or below this size are read and hashed whole; larger ones are sampled.
FULL_READ_BYTES = 32 * 1024**2
# Size of each sampled block along the first axis, in bytes.
BLOCK_BYTES = 1024**2
# A ragged time column is reported when repeats are at least MIN_REPEAT_FRACTION of all its
# values, or at least MIN_ROW_REPEAT_FRACTION of one row with 100 values or more. Isolated
# repeats (a few spikes per unit, as spike sorters sometimes leave) stay below both.
MIN_REPEAT_FRACTION = 0.01
MIN_ROW_REPEAT_FRACTION = 0.1
# Number of sampled blocks for a dataset too large to read whole.
N_BLOCKS = 5
# Largest table (rows) the row-level duplicate check reads.
MAX_TABLE_ROWS = 2_000_000
# Most uncompressed chunk data that the sampled column comparison of one series may decompress.
MAX_FETCH_BYTES = 64 * 1024**2
# Most columns a 2-D TimeSeries may have for the column comparison.
MAX_SERIES_COLUMNS = 1024
# Columns need at least this many distinct values to be compared with each other.
MIN_DISTINCT_FOR_COLUMNS = 3
# How many examples to list in a message.
N_EXAMPLES = 5

TIME_COLUMN_NAMES = ("start_time", "timestamp", "timestamps", "time", "event_time", "onset")


# --------------------------------------------------------------------------- helpers


def _h5(obj: Any) -> Optional[h5py.Dataset]:
    return obj if isinstance(obj, h5py.Dataset) else None


def _same_h5_object(a: Any, b: Any) -> bool:
    """True when two datasets are the same HDF5 object (a link), not two copies."""
    a5, b5 = _h5(a), _h5(b)
    if a5 is None or b5 is None:
        return a is b
    try:
        return a5.id == b5.id
    except Exception:
        return a5.name == b5.name and a5.file == b5.file


def _shape(data: Any) -> tuple:
    return tuple(data.shape) if hasattr(data, "shape") else np.asarray(data).shape


def _dtype(data: Any) -> np.dtype:
    dt = getattr(data, "dtype", None)
    if dt is None:
        return np.asarray(data).dtype
    try:
        return np.dtype(dt)
    except TypeError:
        return np.dtype(object)  # hdmf string wrappers report dtype as a str


def _is_reference_dtype(dtype: Any) -> bool:
    try:
        return h5py.check_dtype(ref=dtype) is not None
    except Exception:
        return False


def _as_bytes(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr)
    if arr.dtype.kind == "O":
        return repr(arr.tolist()).encode()
    return np.ascontiguousarray(arr).tobytes()


def _is_constant(arr: np.ndarray) -> bool:
    arr = np.asarray(arr)
    if arr.size <= 1:
        return True
    if arr.dtype.kind == "f":
        nan = np.isnan(arr)
        if nan.all():
            return True
        if nan.any():
            return False
    try:
        flat = arr.reshape(-1)
        return bool((flat == flat[0]).all())
    except Exception:
        return len(set(map(repr, arr.reshape(-1).tolist()))) <= 1


def _blocks(data: Any, block_bytes: int = BLOCK_BYTES, n_blocks: int = N_BLOCKS) -> list[tuple]:
    """
    Index tuples that sample a large dataset at N_BLOCKS evenly spaced places along axis 0,
    each holding at most about BLOCK_BYTES (trailing axes are cut down for very wide rows,
    such as imaging frames).
    """
    shape = _shape(data)
    itemsize = _dtype(data).itemsize
    dims = list(shape)
    row_bytes = int(np.prod(dims[1:], dtype=np.int64)) * itemsize
    k = max(1, min(shape[0], block_bytes // max(1, row_bytes)))
    dims[0] = k
    while int(np.prod(dims, dtype=np.int64)) * itemsize > block_bytes and max(dims[1:], default=1) > 1:
        j = 1 + int(np.argmax(dims[1:]))
        dims[j] = max(1, dims[j] // 2)
    starts = sorted(set(int(s) for s in np.linspace(0, max(0, shape[0] - k), n_blocks)))
    tail = tuple(slice(0, d) for d in dims[1:])
    return [(slice(s, s + k),) + tail for s in starts]


def _fingerprint(data: Any) -> tuple[str, bool, bool]:
    """Return (digest, exact, constant). exact is False when only sampled blocks were hashed."""
    shape = _shape(data)
    nbytes = int(np.prod(shape, dtype=np.int64)) * _dtype(data).itemsize
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((shape, str(_dtype(data)))).encode())
    if nbytes <= FULL_READ_BYTES or len(shape) == 0:
        arr = np.asarray(data[()] if hasattr(data, "shape") and len(shape) == 0 else data[:])
        h.update(_as_bytes(arr))
        return h.hexdigest(), True, _is_constant(arr)
    constant = True
    first = None
    for sl in _blocks(data):
        arr = np.asarray(data[sl])
        h.update(_as_bytes(arr))
        if constant:
            if not _is_constant(arr):
                constant = False
            else:
                v = arr.reshape(-1)[:1]
                if first is None:
                    first = v
                elif not np.array_equal(first, v, equal_nan=arr.dtype.kind == "f"):
                    constant = False
    return h.hexdigest(), False, constant


def _head_hash(data: Any, nbytes: int = 64 * 1024) -> str:
    """Hash of a small block at the start of a dataset, for a cheap first comparison."""
    first = _blocks(data, block_bytes=nbytes, n_blocks=1)[0]
    return hashlib.blake2b(_as_bytes(np.asarray(data[first])), digest_size=16).hexdigest()


def _is_axis(row: np.ndarray) -> bool:
    """A strictly monotonic 1-D row reads as an axis (bin centres or edges), not as data."""
    d = np.diff(np.asarray(row, dtype=float))
    return d.size > 0 and (bool((d > 0).all()) or bool((d < 0).all()))


def _object_path(obj: Any) -> str:
    parts = []
    while obj is not None and not isinstance(obj, NWBFile):
        parts.append(obj.name)
        obj = obj.parent
    return "/".join(reversed(parts))


def _fmt_list(items: Iterable, n: int = N_EXAMPLES) -> str:
    items = list(items)
    s = ", ".join(str(i) for i in items[:n])
    return s + (f", and {len(items) - n} more" if len(items) > n else "")


def _plain_columns(table: DynamicTable) -> list[VectorData]:
    """Non-ragged columns of a table (index columns and indexed targets are left out)."""
    indexed = set()
    for col in table.columns:
        if isinstance(col, VectorIndex):
            indexed.add(id(col.target))
    out = []
    for col in table.columns:
        if isinstance(col, VectorIndex) or id(col) in indexed:
            continue
        out.append(col)
    return out


def _column_keys(col: VectorData) -> Optional[list]:
    """One hashable key per row of a column, or None if the column cannot be compared."""
    data = col.data
    if _is_reference_dtype(_dtype(data)):
        f = data.file
        return [f[r].name if r else None for r in data[:]]
    arr = np.asarray(data[:])
    if arr.ndim == 1:
        if arr.dtype.kind == "f":
            # make NaN compare equal to NaN
            return [b"nan" if v != v else v for v in arr.tolist()]
        if arr.dtype.kind == "O":
            return [repr(v) for v in arr.tolist()]
        return arr.tolist()
    return [_as_bytes(row) for row in arr]


# --------------------------------------------------------------------------- checks


@register_check(importance=Importance.CRITICAL, neurodata_type=Units)
def check_units_identical_spike_trains(units: Units, min_spikes: int = 1) -> Optional[Iterable[InspectorMessage]]:
    """
    Rows of the units table should not hold the same spike train.

    Units are grouped by spike count, which comes from the index alone, and only units that
    share a count with another unit have their spike times read and hashed.

    Best Practice: :ref:`best_practice_no_duplicate_units`
    """
    if "spike_times" not in units.colnames or len(units.id) < 2:
        return None
    index = np.asarray(units.spike_times_index.data[:], dtype=np.int64)
    starts = np.concatenate([[0], index[:-1]])
    counts = index - starts
    by_count = defaultdict(list)
    for row, c in enumerate(counts.tolist()):
        if c >= min_spikes:
            by_count[c].append(row)
    candidates = [rows for rows in by_count.values() if len(rows) > 1]
    if not candidates:
        return None

    spike_data = units.spike_times.data
    total_bytes = int(_shape(spike_data)[0]) * _dtype(spike_data).itemsize
    all_spikes = np.asarray(spike_data[:]) if total_bytes <= 4 * FULL_READ_BYTES else None

    def train(row: int) -> np.ndarray:
        s, e = int(starts[row]), int(index[row])
        return all_spikes[s:e] if all_spikes is not None else np.asarray(spike_data[s:e])

    groups = []
    for rows in candidates:
        by_hash = defaultdict(list)
        for row in rows:
            by_hash[hashlib.blake2b(_as_bytes(train(row)), digest_size=16).hexdigest()].append(row)
        groups += [g for g in by_hash.values() if len(g) > 1]
    if not groups:
        return None

    ids = np.asarray(units.id.data[:])
    same_id = [g for g in groups if len(set(ids[g].tolist())) == 1]
    diff_id = [g for g in groups if len(set(ids[g].tolist())) > 1]
    messages = []
    if same_id:
        extra = sum(len(g) - 1 for g in same_id)
        ex = [f"id {ids[g[0]]} (rows {', '.join(map(str, g))})" for g in same_id]
        messages.append(
            InspectorMessage(
                message=(
                    f"{len(same_id)} unit id(s) are stored in more than one row with the identical spike train "
                    f"({extra} extra rows of {len(ids)}): {_fmt_list(ex)}. The unit is stored twice, so an "
                    "analysis that treats each row as a unit counts it twice."
                )
            )
        )
    if diff_id:
        ex = [f"ids {', '.join(str(ids[r]) for r in g)} ({counts[g[0]]} spikes)" for g in diff_id]
        messages.append(
            InspectorMessage(
                message=(
                    f"{len(diff_id)} group(s) of units with different ids hold the identical spike train: "
                    f"{_fmt_list(ex)}."
                )
            )
        )
    return messages


@register_check(importance=Importance.CRITICAL, neurodata_type=DynamicTable)
def check_table_duplicate_rows(table: DynamicTable) -> Optional[Iterable[InspectorMessage]]:
    """
    Rows of an event-like table should not repeat each other in every column, and the per-row
    arrays of an array-valued column should not repeat across rows.

    Best Practice: :ref:`best_practice_no_duplicate_rows`
    """
    n = len(table.id)
    if n < 2 or n > MAX_TABLE_ROWS:
        return None
    columns = _plain_columns(table)
    if not columns:
        return None
    messages = []

    # (i) array-valued columns: each row holds a vector; repeated non-constant vectors.
    # Units are left to check_units_identical_spike_trains, so a stored-twice unit is reported once.
    for col in ([] if isinstance(table, Units) else columns):
        shape = _shape(col.data)
        if len(shape) < 2 or isinstance(col, DynamicTableRegion) or shape[0] != n:
            continue
        arr = np.asarray(col.data[:])
        if arr.ndim == 2 and arr.dtype.kind in "fiu" and _is_axis(arr[0]):
            # every row holds the same sorted axis (bin centres, frequencies, lags): expected
            continue
        keys: list[Optional[bytes]] = []
        constant_rows = 0
        for row in arr:
            if _is_constant(row):
                constant_rows += 1
                keys.append(None)
            else:
                keys.append(_as_bytes(row))
        counts = Counter(k for k in keys if k is not None)
        dup_rows = sum(v for v in counts.values() if v > 1)
        if dup_rows:
            biggest = max(counts.values())
            messages.append(
                InspectorMessage(
                    message=(
                        f"Column '{col.name}' holds the same array in more than one row: {dup_rows} of {n} rows "
                        f"repeat another row's values (largest group {biggest} rows)."
                    )
                )
            )

    # (ii) event-like tables: rows identical in every plain column.
    time_cols = [c for c in columns if c.name in TIME_COLUMN_NAMES or c.name.endswith("_time")]
    if time_cols:
        keyed = []
        for col in columns:
            col_keys: Optional[list]
            try:
                col_keys = _column_keys(col)
            except Exception:
                col_keys = None
            if col_keys is not None and len(col_keys) == n:
                keyed.append(col_keys)
        if keyed:
            rows = Counter(zip(*keyed))
            extra = sum(v - 1 for v in rows.values() if v > 1)
            if extra:
                biggest = max(rows.values())
                messages.append(
                    InspectorMessage(
                        message=(
                            f"{extra} of {n} rows repeat another row in every column "
                            f"({', '.join(c.name for c in columns)}); the most repeated row appears {biggest} "
                            "times. Each repeated row is an event or interval stored more than once."
                        )
                    )
                )
    return messages or None


_identical_series_cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _in_acquisition(ts: TimeSeries, nwbfile: NWBFile) -> bool:
    return nwbfile.acquisition.get(ts.name) is ts


def _identical_time_series_groups(nwbfile: NWBFile) -> list[dict]:
    """
    Groups of TimeSeries in one file whose data are identical, computed once per file and shared by
    check_identical_time_series and check_redundant_time_series_copies.

    Series are grouped by data shape and dtype, so only series that could match are read. Pairs
    whose data are the same HDF5 object (a link) are not reported, nor are constant data on a
    rate-based time base (for example two empty or all-zero channels).
    """
    try:
        return _identical_series_cache[nwbfile]
    except (KeyError, TypeError):
        pass
    groups = defaultdict(list)
    for obj in nwbfile.objects.values():
        if not isinstance(obj, TimeSeries) or obj.data is None:
            continue
        try:
            shape = _shape(obj.data)
        except Exception:
            continue
        if len(shape) == 0 or shape[0] < 2:
            continue
        groups[(shape, str(_dtype(obj.data)))].append(obj)

    found = []
    for series in groups.values():
        if len(series) < 2:
            continue
        # Stage 1: hash a small block at the start of each series; only series that agree there
        # go on to the full (or sampled) comparison.
        by_head = defaultdict(list)
        for ts in series:
            by_head[_head_hash(ts.data)].append(ts)
        by_fp = defaultdict(list)
        info = {}
        for head_group in by_head.values():
            if len(head_group) < 2:
                continue
            for ts in head_group:
                fp, exact, const = _fingerprint(ts.data)
                info[id(ts)] = (exact, const)
                by_fp[fp].append(ts)
        for same in by_fp.values():
            if len(same) < 2:
                continue
            # drop members that are links to an earlier member's data
            distinct: list[TimeSeries] = []
            for ts in same:
                if not any(_same_h5_object(ts.data, d.data) for d in distinct):
                    distinct.append(ts)
            if len(distinct) < 2:
                continue
            constant = info[id(distinct[0])][1]

            def time_base(ts: TimeSeries) -> tuple:
                if ts.timestamps is not None:
                    return ("timestamps", _fingerprint(ts.timestamps)[0])
                return ("rate", ts.starting_time, ts.rate)

            if constant:
                # Constant data (all ones, all zeros) only count as a duplicate when the series also carry
                # the same explicit timestamps, stored separately rather than linked.
                by_tb = defaultdict(list)
                for ts in distinct:
                    tb = time_base(ts)
                    if tb[0] == "timestamps":
                        by_tb[tb].append(ts)
                subgroups = []
                for g in by_tb.values():
                    if len(g) > 1 and not all(_same_h5_object(g[0].timestamps, t.timestamps) for t in g[1:]):
                        subgroups.append((g, True))
            else:
                subgroups = [(distinct, len({time_base(t) for t in distinct}) == 1)]
            for group, same_time in subgroups:
                # A copy is most likely wrong data when the series are of different kinds: an event series
                # (constant values at explicit times) under two names, different neurodata types or units,
                # or a processed series identical to raw acquisition, which the processing should have changed.
                in_acq = {_in_acquisition(t, nwbfile) for t in group}
                wrong = (
                    constant
                    or len({type(t) for t in group}) > 1
                    or len({getattr(t, "unit", None) for t in group}) > 1
                    or len(in_acq) > 1
                )
                found.append(
                    dict(
                        group=group,
                        exact=all(info[id(t)][0] for t in group),
                        constant=constant,
                        same_time=same_time,
                        wrong=wrong,
                    )
                )
    try:
        _identical_series_cache[nwbfile] = found
    except TypeError:
        pass
    return found


def _identical_series_message(g: dict, advice: str) -> InspectorMessage:
    group = g["group"]
    names = [_object_path(t) for t in group]
    how = "identical" if g["exact"] else f"identical on {N_BLOCKS} sampled blocks of"
    tb = "and the same time base" if g["same_time"] else "but different time bases (timestamps or rate)"
    extra = " The data are constant and the timestamps are identical." if g["constant"] else ""
    return InspectorMessage(
        message=(
            f"{len(group)} TimeSeries hold {how} data of shape {_shape(group[0].data)} {tb}: "
            f"{', '.join(names)}.{extra} {advice}"
        )
    )


@register_check(importance=Importance.CRITICAL, neurodata_type=NWBFile)
def check_identical_time_series(nwbfile: NWBFile) -> Optional[Iterable[InspectorMessage]]:
    """
    Two TimeSeries of different kinds should not hold identical data.

    Reported here are identical series that are of different neurodata types or units, a processed
    series identical to a series in acquisition, and event-like series (constant values) that share
    explicit timestamps under different names. In each case one series most likely holds the wrong data.
    Identical series of the same kind are reported by check_redundant_time_series_copies instead.

    Best Practice: :ref:`best_practice_no_identical_series`
    """
    messages = [
        _identical_series_message(
            g, "One of them most likely holds the wrong data; if it is meant to be a copy, link it instead."
        )
        for g in _identical_time_series_groups(nwbfile)
        if g["wrong"]
    ]
    return messages or None


@register_check(importance=Importance.BEST_PRACTICE_VIOLATION, neurodata_type=NWBFile)
def check_redundant_time_series_copies(nwbfile: NWBFile) -> Optional[Iterable[InspectorMessage]]:
    """
    Two TimeSeries of the same kind should not store the same data twice; store it once and link
    the second series to it. Check that both are meant to hold the same data.

    Best Practice: :ref:`best_practice_no_identical_series`
    """
    messages = [
        _identical_series_message(
            g,
            "If both are meant to hold the same data, store it once and link the other series to it; "
            "otherwise one of them holds the wrong data.",
        )
        for g in _identical_time_series_groups(nwbfile)
        if not g["wrong"]
    ]
    return messages or None


@register_check(importance=Importance.CRITICAL, neurodata_type=DynamicTable)
def check_table_identical_columns(table: DynamicTable) -> Optional[Iterable[InspectorMessage]]:
    """
    Two scalar columns of one table should not hold identical, non-trivial values.

    Best Practice: :ref:`best_practice_no_identical_columns`
    """
    n = len(table.id)
    if n < MIN_DISTINCT_FOR_COLUMNS or n > MAX_TABLE_ROWS:
        return None
    by_type = defaultdict(list)
    for col in _plain_columns(table):
        if isinstance(col, DynamicTableRegion):
            continue
        shape = _shape(col.data)
        dt = _dtype(col.data)
        if len(shape) != 1 or dt.kind not in "fiub" or _is_reference_dtype(dt):
            continue
        by_type[(shape, dt.kind)].append(col)
    messages = []
    for cols in by_type.values():
        if len(cols) < 2:
            continue
        by_hash = defaultdict(list)
        for col in cols:
            arr = np.asarray(col.data[:])
            finite = arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr
            if len(np.unique(finite)) < MIN_DISTINCT_FOR_COLUMNS:
                continue
            by_hash[hashlib.blake2b(_as_bytes(arr.astype(np.float64)), digest_size=16).hexdigest()].append(col)
        for same in by_hash.values():
            # time columns of one table often coincide by design (stop_time and return_stop_time)
            same = [c for c in same if not c.name.endswith("time")] if len(same) > 1 else same
            if len(same) < 2:
                continue
            messages.append(
                InspectorMessage(
                    message=(
                        f"Columns {', '.join(repr(c.name) for c in same)} hold identical values in all {n} rows. "
                        "One of them most likely holds the other's data."
                    )
                )
            )
    return messages or None


def _chunk_rows(data: Any) -> Optional[int]:
    h = _h5(data)
    return h.chunks[0] if h is not None and h.chunks else None


def _fetch_bytes(data: Any, rows: slice, cols: Optional[list] = None) -> int:
    """
    Uncompressed bytes that reading data[rows, cols] of a 2-D dataset decompresses. HDF5 reads
    whole chunks, so a few rows of a dataset chunked as (1048576, 16) cost far more than their size.
    """
    shape = _shape(data)
    itemsize = _dtype(data).itemsize
    cols = list(range(shape[1])) if cols is None else cols
    h = _h5(data)
    if h is None or not h.chunks:
        return (rows.stop - rows.start) * len(cols) * itemsize
    cr, cc = h.chunks[0], h.chunks[1]
    n_row_chunks = (min(rows.stop, shape[0]) - 1) // cr - rows.start // cr + 1
    n_col_chunks = len({c // cc for c in cols})
    return n_row_chunks * n_col_chunks * cr * cc * itemsize


def _row_blocks(data: Any, block_bytes: int = BLOCK_BYTES, n_blocks: int = N_BLOCKS) -> list[slice]:
    """
    Row slices that sample a 2-D dataset at n_blocks places, each of at most about block_bytes and
    each inside a single chunk row, so that a block decompresses one row of chunks and not two.
    """
    n, m = _shape(data)[:2]
    k = max(1, min(n, block_bytes // max(1, m * _dtype(data).itemsize)))
    cr = _chunk_rows(data)
    if cr:
        k = min(k, cr)
    out: list[slice] = []
    for s in np.linspace(0, max(0, n - k), n_blocks).astype(int).tolist():
        if cr:
            s = min((s // cr) * cr, n - k)
        if not out or s != out[-1].start:
            out.append(slice(s, s + k))
    return out


def _column_digests(arr: np.ndarray) -> list:
    return [hashlib.blake2b(_as_bytes(arr[:, j]), digest_size=16).hexdigest() for j in range(arr.shape[1])]


def _read_columns(data: Any, rows: slice, cols: list) -> np.ndarray:
    if _h5(data) is not None:
        return np.asarray(data[rows, cols])
    lo, hi = min(cols), max(cols) + 1
    return np.asarray(data[rows, lo:hi])[:, [c - lo for c in cols]]


@register_check(importance=Importance.CRITICAL, neurodata_type=TimeSeries)
def check_time_series_identical_columns(
    time_series: TimeSeries, max_fetch_bytes: int = MAX_FETCH_BYTES
) -> Optional[InspectorMessage]:
    """
    Two columns (channels or dimensions) of a 2-D TimeSeries should not be identical, for
    example the z coordinate of a position that copies y.

    Small series are compared whole. Large series are compared in two stages: every column on a
    first block of rows, then only the columns that match there on up to N_BLOCKS - 1 further
    blocks. Blocks are aligned to the HDF5 chunk rows, and a series whose first block alone would
    decompress more than max_fetch_bytes of chunks is skipped, since reading it remotely is slow.

    Best Practice: :ref:`best_practice_no_identical_columns`
    """
    data = time_series.data
    if data is None:
        return None
    try:
        shape = _shape(data)
    except Exception:
        return None
    if len(shape) != 2 or shape[1] < 2 or shape[1] > MAX_SERIES_COLUMNS or shape[0] < MIN_DISTINCT_FOR_COLUMNS:
        return None
    if _dtype(data).kind not in "fiub":
        return None

    def nontrivial(col: np.ndarray) -> bool:
        finite = col[~np.isnan(col)] if col.dtype.kind == "f" else col
        return len(np.unique(finite)) >= MIN_DISTINCT_FOR_COLUMNS

    exact = shape[0] * shape[1] * _dtype(data).itemsize <= FULL_READ_BYTES
    if exact:
        arr = np.asarray(data[:])
        keep = [j for j in range(shape[1]) if nontrivial(arr[:, j])]
        digests = _column_digests(arr)
        keys: dict[int, tuple[str, ...]] = {j: (digests[j],) for j in keep}
        n_blocks = 0
    else:
        blocks = _row_blocks(data)
        budget = max_fetch_bytes - _fetch_bytes(data, blocks[0])
        if budget < 0:
            return None
        head = np.asarray(data[blocks[0]])
        keep = [j for j in range(shape[1]) if nontrivial(head[:, j])]
        digests = _column_digests(head)
        keys = {j: (digests[j],) for j in keep}
        n_blocks = 1
        for rows in blocks[1:]:
            counts = Counter(keys.values())
            cand = sorted(j for j, k in keys.items() if counts[k] > 1)
            if not cand:
                break
            cost = _fetch_bytes(data, rows, cand)
            if cost > budget:
                break
            budget -= cost
            block_digests = _column_digests(_read_columns(data, rows, cand))
            keys = {j: keys[j] + (d,) for j, d in zip(cand, block_digests)}
            n_blocks += 1

    by_key = defaultdict(list)
    for j, k in keys.items():
        by_key[k].append(j)
    groups = [g for g in by_key.values() if len(g) > 1]
    if not groups:
        return None
    where = "in every row" if exact else f"on {n_blocks} sampled block(s) of rows"
    return InspectorMessage(
        message=(
            f"{len(groups)} group(s) of columns of the data are identical {where}: "
            f"{_fmt_list(['columns ' + ' = '.join(map(str, g)) for g in groups])}."
        )
    )


def _coarse_resolution(values: np.ndarray, step: float = 1e-3) -> bool:
    """True when every time is a multiple of 1 ms or coarser, so two events can share a value."""
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)][:100_000]
    if v.size == 0:
        return False
    return bool(np.allclose(v / step, np.round(v / step), rtol=0, atol=1e-3))


def _count_repeats(values: np.ndarray) -> int:
    """Number of values that equal another value in the same array (NaN ignored)."""
    v = np.asarray(values).reshape(-1)
    if v.dtype.kind == "f":
        v = v[~np.isnan(v)]
    if v.size < 2:
        return 0
    v = np.sort(v, kind="stable")
    return int(np.count_nonzero(v[1:] == v[:-1]))


@register_check(importance=Importance.CRITICAL, neurodata_type=DynamicTable)
def check_table_ragged_times_repeated(table: DynamicTable) -> Optional[Iterable[InspectorMessage]]:
    """
    A ragged time column (such as the event_times of an event type) should not hold the same time
    more than once in one row; a repeated time is an event stored twice. Spike times of a Units
    table are left to check_ascending_spike_times.

    Best Practice: :ref:`best_practice_no_duplicate_rows`
    """
    messages = []
    for col in table.columns:
        if not isinstance(col, VectorIndex):
            continue
        target = col.target
        name = target.name
        if not (name.endswith("times") or name.endswith("_time") or name == "timestamps"):
            continue
        if isinstance(table, Units) and name == "spike_times":
            continue  # repeated spike times are reported by check_ascending_spike_times
        dt = _dtype(target.data)
        if dt.kind != "f" or len(_shape(target.data)) != 1:
            continue
        if _shape(target.data)[0] * dt.itemsize > 4 * FULL_READ_BYTES:
            continue
        values = np.asarray(target.data[:])
        if _coarse_resolution(values):
            continue  # times rounded to 1 ms or coarser can coincide without being stored twice
        index = np.asarray(col.data[:], dtype=np.int64)
        starts = np.concatenate([[0], index[:-1]])
        per_row = [_count_repeats(values[s:e]) for s, e in zip(starts.tolist(), index.tolist())]
        total = sum(per_row)
        if not total:
            continue
        lengths = index - starts
        row_frac = max((r / n for r, n in zip(per_row, lengths.tolist()) if n >= 100), default=0.0)
        if total / max(1, len(values)) < MIN_REPEAT_FRACTION and row_frac < MIN_ROW_REPEAT_FRACTION:
            continue
        rows = [i for i, r in enumerate(per_row) if r]
        ids = np.asarray(table.id.data[:])
        worst = max(rows, key=lambda i: per_row[i] / max(1, index[i] - starts[i]))
        n_worst = int(index[worst] - starts[worst])
        messages.append(
            InspectorMessage(
                message=(
                    f"Column '{name}' repeats a time within a row: {total} repeated values of {len(values)} in "
                    f"{len(rows)} of {len(index)} rows (ids {_fmt_list(ids[rows].tolist())}). In row id "
                    f"{ids[worst]}, {per_row[worst]} of {n_worst} times are repeats "
                    f"({n_worst - per_row[worst]} distinct)."
                )
            )
        )
    return messages or None


@register_check(importance=Importance.CRITICAL, neurodata_type=TimeSeries)
def check_timestamps_repeated(time_series: TimeSeries) -> Optional[InspectorMessage]:
    """
    Timestamps of a TimeSeries should not repeat; each repeat is a sample or event stored twice.

    Best Practice: :ref:`best_practice_no_duplicate_rows`
    """
    ts = time_series.timestamps
    if ts is None or isinstance(ts, TimeSeries):
        return None
    try:
        shape = _shape(ts)
    except Exception:
        return None
    if len(shape) != 1 or shape[0] < 2:
        return None
    exact = shape[0] * _dtype(ts).itemsize <= FULL_READ_BYTES // 4
    if exact:
        values = np.asarray(ts[:])
        if _coarse_resolution(values):
            return None
        repeats = _count_repeats(values)
        n = shape[0]
    else:
        blocks = [np.asarray(ts[sl]) for sl in _blocks(ts)]
        if _coarse_resolution(np.concatenate(blocks)):
            return None
        repeats = sum(_count_repeats(b) for b in blocks)
        n = sum(len(b) for b in blocks)
    if not repeats or repeats / n < MIN_REPEAT_FRACTION:
        return None
    where = "" if exact else f" in {N_BLOCKS} sampled blocks"
    return InspectorMessage(
        message=f"{repeats} of {n} timestamps{where} repeat another timestamp ({100 * repeats / n:.1f}%)."
    )
